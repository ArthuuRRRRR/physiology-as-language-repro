from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_architecture.downstream.CNN_TCN.model import (
    SleepStageCNNTCN,
    count_parameters,
)


STAGE_NAMES = ["Wake", "Light", "Deep", "REM"]
IGNORE_INDEX = -100


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def canonical_dataset(name: str) -> str:
    name = str(name).lower()

    if name.startswith("shhs1"):
        return "shhs1"
    if name.startswith("mesa"):
        return "mesa"
    if name.startswith("cfs"):
        return "cfs"

    return name


class MMapWindowDataset(Dataset):
    def __init__(
        self,
        split_dir: Path,
        input_type: str = "synthesized",
    ):
        self.split_dir = Path(split_dir)
        self.input_type = input_type

        metadata_path = self.split_dir / "metadata.json"
        index_path = self.split_dir / "index.jsonl"

        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)

        with metadata_path.open("r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.num_samples = int(self.metadata["num_samples"])
        self.eeg_shape = tuple(self.metadata["eeg_shape"])
        self.label_shape = tuple(self.metadata["label_shape"])
        self.eeg_dtype = np.dtype(self.metadata["eeg_dtype"])
        self.label_dtype = np.dtype(self.metadata["label_dtype"])

        self.dataset_name = canonical_dataset(
            self.metadata.get("dataset", self.split_dir.parent.name)
        )

        if input_type == "synthesized":
            self.eeg_path = self.split_dir / "synthesized_eeg.mmap"
        elif input_type == "ground_truth":
            self.eeg_path = self.split_dir / "ground_truth_eeg.mmap"
        else:
            raise ValueError(f"Unknown input type: {input_type}")

        self.labels_path = self.split_dir / "labels.mmap"

        if not self.eeg_path.exists():
            raise FileNotFoundError(self.eeg_path)
        if not self.labels_path.exists():
            raise FileNotFoundError(self.labels_path)

        self.index_rows = []
        if index_path.exists():
            with index_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.index_rows.append(json.loads(line))

        if self.index_rows and len(self.index_rows) != self.num_samples:
            raise RuntimeError(
                f"{self.split_dir}: "
                f"index={len(self.index_rows)}, "
                f"metadata={self.num_samples}"
            )

        self._eeg = None
        self._labels = None

    def _open(self) -> None:
        if self._eeg is None:
            self._eeg = np.memmap(
                self.eeg_path,
                dtype=self.eeg_dtype,
                mode="r",
                shape=self.eeg_shape,
            )

        if self._labels is None:
            self._labels = np.memmap(
                self.labels_path,
                dtype=self.label_dtype,
                mode="r",
                shape=self.label_shape,
            )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_eeg"] = None
        state["_labels"] = None
        return state

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int):
        self._open()

        eeg = np.asarray(
            self._eeg[index],
            dtype=np.float32,
        ).copy()

        labels = np.asarray(
            self._labels[index],
            dtype=np.int64,
        ).copy()

        if self.index_rows:
            row = self.index_rows[index]
            dataset = canonical_dataset(
                row.get("dataset", self.dataset_name)
            )
            subject_id = str(
                row.get("subject_id", f"{dataset}_{index}")
            )
        else:
            dataset = self.dataset_name
            subject_id = f"{dataset}_{index}"

        return {
            "eeg": torch.from_numpy(eeg).unsqueeze(0),
            "labels": torch.from_numpy(labels),
            "dataset": dataset,
            "subject_id": subject_id,
        }

    def class_counts(self) -> np.ndarray:
        self._open()

        counts = np.zeros(4, dtype=np.int64)

        for i in range(self.num_samples):
            labels = np.asarray(
                self._labels[i],
                dtype=np.int64,
            )

            valid = (labels >= 0) & (labels < 4)

            if valid.any():
                counts += np.bincount(
                    labels[valid],
                    minlength=4,
                )[:4]

        return counts


def build_dataset(
    data_root: Path,
    dataset_name: str,
    split: str,
    input_type: str,
) -> MMapWindowDataset:
    split_dir = Path(data_root) / dataset_name / split

    dataset = MMapWindowDataset(
        split_dir,
        input_type=input_type,
    )

    print(
        f"{split:5s} {dataset_name:6s}: "
        f"{len(dataset)} windows"
    )

    return dataset


def make_class_weights(
    dataset: MMapWindowDataset,
    power: float,
    device: torch.device,
) -> torch.Tensor:
    counts = dataset.class_counts()

    if power <= 0:
        weights = np.ones(4, dtype=np.float32)
    else:
        safe_counts = np.maximum(counts, 1).astype(np.float64)
        weights = (safe_counts.sum() / safe_counts) ** power
        weights /= weights.mean()
        weights = weights.astype(np.float32)

    print("Train class counts:", counts.tolist())
    print(
        "Class weights:",
        [round(float(x), 4) for x in weights],
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )


def build_scheduler(
    optimizer,
    warmup_steps: int,
    total_steps: int,
):
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(
                1e-4,
                float(step + 1) / float(warmup_steps),
            )

        if total_steps <= warmup_steps:
            return 1.0

        progress = (
            (step - warmup_steps)
            / (total_steps - warmup_steps)
        )
        progress = min(1.0, max(0.0, progress))

        cosine = 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

        # Decay to 10% of the peak learning rate.
        return 0.1 + 0.9 * cosine

    return LambdaLR(optimizer, lr_lambda)


def make_scaler(use_amp: bool):
    if not use_amp:
        return None

    try:
        return torch.amp.GradScaler("cuda")
    except Exception:
        return torch.cuda.amp.GradScaler()


def compute_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict:
    targets = np.asarray(targets, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)

    accuracy = accuracy_score(targets, predictions)

    macro_f1 = f1_score(
        targets,
        predictions,
        labels=[0, 1, 2, 3],
        average="macro",
        zero_division=0,
    )

    kappa = cohen_kappa_score(
        targets,
        predictions,
        labels=[0, 1, 2, 3],
    )

    per_class_f1 = f1_score(
        targets,
        predictions,
        labels=[0, 1, 2, 3],
        average=None,
        zero_division=0,
    )

    cm = confusion_matrix(
        targets,
        predictions,
        labels=[0, 1, 2, 3],
    )

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "kappa": float(kappa),
        "f1_per_class": {
            name: float(value)
            for name, value in zip(
                STAGE_NAMES,
                per_class_f1,
            )
        },
        "confusion_matrix": cm.tolist(),
        "num_segments": int(len(targets)),
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    class_weights,
    device,
    use_amp,
) -> dict:
    model.train()

    total_loss = 0.0
    total_valid = 0
    total_correct = 0

    for batch in loader:
        eeg = batch["eeg"].to(
            device,
            non_blocking=True,
        )
        labels = batch["labels"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(eeg)

            loss = F.cross_entropy(
                logits.transpose(1, 2),
                labels,
                weight=class_weights,
                ignore_index=IGNORE_INDEX,
            )

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            optimizer.step()

        scheduler.step()

        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            valid = (labels >= 0) & (labels < 4)

            n_valid = int(valid.sum().item())
            total_valid += n_valid

            total_correct += int(
                (
                    predictions[valid]
                    == labels[valid]
                )
                .sum()
                .item()
            )

            total_loss += float(loss.item()) * n_valid

    return {
        "loss": total_loss / max(total_valid, 1),
        "accuracy": total_correct / max(total_valid, 1),
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    use_amp,
) -> dict:
    model.eval()

    all_targets = []
    all_predictions = []
    total_loss = 0.0
    total_valid = 0
    subjects = set()

    for batch in loader:
        eeg = batch["eeg"].to(
            device,
            non_blocking=True,
        )
        labels = batch["labels"].to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(eeg)

        predictions = logits.argmax(dim=-1)
        valid = (labels >= 0) & (labels < 4)

        if valid.any():
            loss_sum = F.cross_entropy(
                logits[valid].float(),
                labels[valid],
                reduction="sum",
            )

            total_loss += float(loss_sum.item())
            total_valid += int(valid.sum().item())

        for b in range(labels.shape[0]):
            mask = valid[b]

            if not mask.any():
                continue

            all_targets.append(
                labels[b][mask]
                .detach()
                .cpu()
                .numpy()
            )

            all_predictions.append(
                predictions[b][mask]
                .detach()
                .cpu()
                .numpy()
            )

            subjects.add(
                str(batch["subject_id"][b])
            )

    if not all_targets:
        raise RuntimeError(
            "No valid evaluation segments."
        )

    targets = np.concatenate(all_targets)
    predictions = np.concatenate(all_predictions)

    metrics = compute_metrics(
        targets,
        predictions,
    )

    metrics["loss"] = (
        total_loss / max(total_valid, 1)
    )
    metrics["num_subjects"] = len(subjects)

    return metrics


def save_checkpoint(
    path: Path,
    model,
    epoch: int,
    args,
    metrics: dict,
) -> None:
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "model_config": model.model_config(),
            "val_metrics": metrics,
            "input_type": args.input_type,
            "dataset": args.dataset,
        },
        path,
    )


def print_metrics(
    title: str,
    metrics: dict,
) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)

    print(
        f"loss={metrics['loss']:.4f} "
        f"acc={metrics['accuracy']:.4f} "
        f"MF1={metrics['macro_f1']:.4f} "
        f"kappa={metrics['kappa']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="CNN-TCN sleep staging classifier"
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "outputs/downstream_ViT_synth"
        ),
    )

    parser.add_argument(
        "--dataset",
        choices=["shhs1", "mesa", "cfs"],
        required=True,
    )

    parser.add_argument(
        "--input-type",
        choices=["synthesized", "ground_truth"],
        default="synthesized",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--eval-only",
        action="store_true",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-2,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--class-weight-power",
        type=float,
        default=0.5,
        help="0 disables weighting; 0.5 uses sqrt inverse-frequency weights",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    use_amp = device.type == "cuda"

    print("=" * 72)
    print("CNN-TCN SLEEP STAGING")
    print("=" * 72)
    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print("Dataset:", args.dataset)
    print("Input:", args.input_type)

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    if args.eval_only:
        if args.checkpoint is None:
            raise ValueError(
                "--checkpoint is required with --eval-only"
            )

        checkpoint = torch.load(
            args.checkpoint,
            map_location="cpu",
            weights_only=False,
        )

        model = SleepStageCNNTCN(
            **checkpoint["model_config"]
        )
        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )
        model = model.to(device)

        dataset = build_dataset(
            args.data_root,
            args.dataset,
            args.split,
            args.input_type,
        )

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=use_amp,
            persistent_workers=args.num_workers > 0,
        )

        metrics = evaluate(
            model,
            loader,
            device,
            use_amp,
        )

        print_metrics(
            f"{args.split.upper()} RESULTS",
            metrics,
        )

        print(
            f"Subjects: {metrics['num_subjects']}"
        )

        print("\nF1 per class:")
        for stage, value in metrics[
            "f1_per_class"
        ].items():
            print(f"  {stage:5s}: {value:.4f}")

        print(
            "\nConfusion matrix "
            "(rows=true, cols=pred)"
        )
        print(
            np.asarray(
                metrics["confusion_matrix"]
            )
        )

        return

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    train_dataset = build_dataset(
        args.data_root,
        args.dataset,
        "train",
        args.input_type,
    )

    val_dataset = build_dataset(
        args.data_root,
        args.dataset,
        "val",
        args.input_type,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=use_amp,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_amp,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )

    # Always initialize from scratch.
    model = SleepStageCNNTCN().to(device)

    print()
    print("Train windows:", len(train_dataset))
    print("Val windows:", len(val_dataset))
    print(
        "Trainable parameters:",
        f"{count_parameters(model):,}",
    )

    class_weights = make_class_weights(
        train_dataset,
        args.class_weight_power,
        device,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = (
        args.epochs * len(train_loader)
    )
    warmup_steps = (
        args.warmup_epochs
        * len(train_loader)
    )

    scheduler = build_scheduler(
        optimizer,
        warmup_steps,
        total_steps,
    )

    scaler = make_scaler(use_amp)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_macro_f1 = -1.0

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            class_weights,
            device,
            use_amp,
        )

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            use_amp,
        )

        print(
            f"\nEpoch {epoch:02d}/{args.epochs}"
        )

        print(
            f"Train loss={train_metrics['loss']:.4f} "
            f"acc={train_metrics['accuracy']:.4f}"
        )

        print(
            f"Val   loss={val_metrics['loss']:.4f} "
            f"acc={val_metrics['accuracy']:.4f} "
            f"MF1={val_metrics['macro_f1']:.4f} "
            f"kappa={val_metrics['kappa']:.4f}"
        )

        save_checkpoint(
            args.output_dir / "checkpoint_latest.pt",
            model,
            epoch,
            args,
            val_metrics,
        )

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]

            save_checkpoint(
                args.output_dir
                / "checkpoint_best_macro_f1.pt",
                model,
                epoch,
                args,
                val_metrics,
            )

            print(
                "  -> new best Macro-F1 checkpoint"
            )

    print()
    print(
        "Best validation Macro-F1:",
        f"{best_macro_f1:.4f}",
    )
    print(
        "Checkpoint:",
        args.output_dir
        / "checkpoint_best_macro_f1.pt",
    )


if __name__ == "__main__":
    main()
