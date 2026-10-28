from __future__ import annotations

import argparse
import json
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
    f1_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from paper_architecture.downstream.ViT.dataset import (
    build_multi_dataset,
)

from paper_architecture.downstream.ViT.model import (
    SleepStageViT,
)


IGNORE_INDEX = -100

CLASS_NAMES = (
    "Wake",
    "Light",
    "Deep",
    "REM",
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Dataset names
# ============================================================

def canonical_dataset(
    value: str,
) -> str:

    value = str(value).lower()

    if value.startswith("shhs1"):
        return "shhs1"

    if value.startswith("mesa"):
        return "mesa"

    if value.startswith("cfs"):
        return "cfs"

    return value


# ============================================================
# Scheduler
# ============================================================

def build_scheduler(
    optimizer,
    warmup_steps: int,
):

    def lr_lambda(step: int):

        if warmup_steps <= 0:
            return 1.0

        if step < warmup_steps:

            return float(
                step + 1
            ) / float(
                warmup_steps
            )

        return 1.0

    return LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
    )


# ============================================================
# AMP scaler
# ============================================================

def make_scaler(enabled: bool):

    try:

        return torch.amp.GradScaler(
            "cuda",
            enabled=enabled,
        )

    except (AttributeError, TypeError):

        return torch.cuda.amp.GradScaler(
            enabled=enabled,
        )


# ============================================================
# Metrics
# ============================================================

def compute_metrics(
    targets,
    predictions,
):

    targets = np.asarray(
        targets,
        dtype=np.int64,
    )

    predictions = np.asarray(
        predictions,
        dtype=np.int64,
    )

    result = {
        "accuracy": float(
            accuracy_score(
                targets,
                predictions,
            )
        ),

        "macro_f1": float(
            f1_score(
                targets,
                predictions,
                labels=[0, 1, 2, 3],
                average="macro",
                zero_division=0,
            )
        ),

        "kappa": float(
            cohen_kappa_score(
                targets,
                predictions,
                labels=[0, 1, 2, 3],
            )
        ),

        "num_epochs": int(
            len(targets)
        ),
    }

    class_f1 = f1_score(
        targets,
        predictions,
        labels=[0, 1, 2, 3],
        average=None,
        zero_division=0,
    )

    result["f1_per_class"] = {
        name: float(value)
        for name, value
        in zip(
            CLASS_NAMES,
            class_f1,
        )
    }

    return result


# ============================================================
# Evaluation
# ============================================================

@torch.inference_mode()
def evaluate(
    model,
    loader,
    device,
    use_amp,
):

    model.eval()

    total_loss = 0.0
    total_valid = 0

    global_targets = []
    global_predictions = []

    dataset_targets = defaultdict(
        list
    )

    dataset_predictions = defaultdict(
        list
    )

    for batch in loader:

        eeg = (
            batch["eeg"]
            .to(
                device,
                non_blocking=True,
            )
        )

        labels = (
            batch["labels"]
            .to(
                device,
                non_blocking=True,
            )
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):

            logits = model(
                eeg
            )

            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    4,
                ),
                labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )

        predictions = (
            logits.argmax(
                dim=-1
            )
        )

        valid = (
            labels
            != IGNORE_INDEX
        )

        n_valid = int(
            valid.sum().item()
        )

        total_loss += float(
            loss.item()
        )

        total_valid += (
            n_valid
        )

        labels_np = (
            labels
            .detach()
            .cpu()
            .numpy()
        )

        pred_np = (
            predictions
            .detach()
            .cpu()
            .numpy()
        )

        valid_np = (
            labels_np
            != IGNORE_INDEX
        )

        global_targets.extend(
            labels_np[
                valid_np
            ].tolist()
        )

        global_predictions.extend(
            pred_np[
                valid_np
            ].tolist()
        )

        for i, raw_name in enumerate(
            batch["dataset"]
        ):

            dataset_name = (
                canonical_dataset(
                    raw_name
                )
            )

            mask = valid_np[i]

            dataset_targets[
                dataset_name
            ].extend(
                labels_np[
                    i
                ][
                    mask
                ].tolist()
            )

            dataset_predictions[
                dataset_name
            ].extend(
                pred_np[
                    i
                ][
                    mask
                ].tolist()
            )

    metrics = compute_metrics(
        global_targets,
        global_predictions,
    )

    metrics["loss"] = (
        total_loss
        / max(
            total_valid,
            1,
        )
    )

    metrics[
        "per_dataset"
    ] = {}

    for dataset_name in sorted(
        dataset_targets
    ):

        metrics[
            "per_dataset"
        ][
            dataset_name
        ] = compute_metrics(
            dataset_targets[
                dataset_name
            ],
            dataset_predictions[
                dataset_name
            ],
        )

    return metrics


# ============================================================
# Train one epoch
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    device,
    use_amp,
):

    model.train()

    total_loss = 0.0
    total_valid = 0
    total_correct = 0

    for batch in loader:

        eeg = (
            batch["eeg"]
            .to(
                device,
                non_blocking=True,
            )
        )

        labels = (
            batch["labels"]
            .to(
                device,
                non_blocking=True,
            )
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):

            logits = model(
                eeg
            )

            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    4,
                ),
                labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="mean",
            )

        scaler.scale(
            loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        scheduler.step()

        with torch.no_grad():

            valid = (
                labels
                != IGNORE_INDEX
            )

            prediction = (
                logits.argmax(
                    dim=-1
                )
            )

            n_valid = int(
                valid.sum().item()
            )

            n_correct = int(
                (
                    prediction[
                        valid
                    ]
                    == labels[
                        valid
                    ]
                )
                .sum()
                .item()
            )

        total_loss += (
            float(
                loss.item()
            )
            * n_valid
        )

        total_valid += (
            n_valid
        )

        total_correct += (
            n_correct
        )

    return {
        "loss": (
            total_loss
            / max(
                total_valid,
                1,
            )
        ),

        "accuracy": (
            total_correct
            / max(
                total_valid,
                1,
            )
        ),

        "num_epochs": (
            total_valid
        ),
    }


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    metrics,
    config,
):

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": (
                model.state_dict()
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "metrics": metrics,
            "config": config,
        },
        path,
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "outputs/"
            "downstream_ViT_synth"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "paper_architecture/"
            "downstream/ViT/"
            "outputs/synth"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-2,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
    )

    args = parser.parse_args()

    set_seed(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    use_amp = (
        device.type == "cuda"
        and not args.no_amp
    )

    print()
    print("=" * 72)
    print("ViT SLEEP STAGING — SYNTH EEG")
    print("=" * 72)

    print(
        "Device:",
        device,
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(
                device
            ),
        )

    print(
        "AMP:",
        use_amp,
    )

    # ========================================================
    # Three datasets
    # ========================================================

    train_dirs = [
        args.data_root
        / "shhs1"
        / "train",

        args.data_root
        / "mesa"
        / "train",

        args.data_root
        / "cfs"
        / "train",
    ]

    val_dirs = [
        args.data_root
        / "shhs1"
        / "val",

        args.data_root
        / "mesa"
        / "val",

        args.data_root
        / "cfs"
        / "val",
    ]

    train_dataset = (
        build_multi_dataset(
            train_dirs
        )
    )

    val_dataset = (
        build_multi_dataset(
            val_dirs
        )
    )

    print()
    print(
        "Train windows:",
        len(
            train_dataset
        ),
    )

    print(
        "Val windows:",
        len(
            val_dataset
        ),
    )

    # ========================================================
    # Loaders
    # ========================================================

    generator = (
        torch.Generator()
    )

    generator.manual_seed(
        args.seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=(
            device.type
            == "cuda"
        ),
        persistent_workers=(
            args.num_workers
            > 0
        ),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.type
            == "cuda"
        ),
        persistent_workers=(
            args.num_workers
            > 0
        ),
        drop_last=False,
    )

    # ========================================================
    # Model
    # ========================================================

    model = (
        SleepStageViT()
        .to(
            device
        )
    )

    num_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "Trainable parameters:",
        f"{num_parameters:,}",
    )

    # ========================================================
    # Optimization
    # ========================================================

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=(
            args.weight_decay
        ),
    )

    warmup_steps = (
        args.warmup_epochs
        * len(
            train_loader
        )
    )

    scheduler = build_scheduler(
        optimizer,
        warmup_steps,
    )

    scaler = make_scaler(
        use_amp
    )

    # ========================================================
    # Output
    # ========================================================

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = {
        "input": (
            "synthesized EEG"
        ),
        "datasets": [
            "shhs1",
            "mesa",
            "cfs",
        ],
        "epochs": (
            args.epochs
        ),
        "warmup_epochs": (
            args.warmup_epochs
        ),
        "lr": (
            args.lr
        ),
        "weight_decay": (
            args.weight_decay
        ),
        "batch_size": (
            args.batch_size
        ),
        "optimizer": (
            "AdamW"
        ),
        "scheduler": (
            "linear warmup then constant"
        ),
        "loss": (
            "cross entropy"
        ),
        "seed": (
            args.seed
        ),
        "trainable_parameters": (
            num_parameters
        ),
    }

    (
        args.output_dir
        / "config.json"
    ).write_text(
        json.dumps(
            config,
            indent=2,
        )
    )

    history = []

    best_macro_f1 = -1.0
    best_accuracy = -1.0

    # ========================================================
    # Training
    # ========================================================

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        train_metrics = (
            train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                use_amp=use_amp,
            )
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            use_amp=use_amp,
        )

        row = {
            "epoch": epoch,
            "train": (
                train_metrics
            ),
            "val": (
                val_metrics
            ),
        }

        history.append(
            row
        )

        print()
        print(
            f"Epoch {epoch:02d}/"
            f"{args.epochs}"
        )

        print(
            "Train "
            f"loss={train_metrics['loss']:.4f} "
            f"acc={train_metrics['accuracy']:.4f}"
        )

        print(
            "Val   "
            f"loss={val_metrics['loss']:.4f} "
            f"acc={val_metrics['accuracy']:.4f} "
            f"MF1={val_metrics['macro_f1']:.4f} "
            f"kappa={val_metrics['kappa']:.4f}"
        )

        for dataset_name, metrics in (
            val_metrics[
                "per_dataset"
            ].items()
        ):

            print(
                f"  {dataset_name:5s} | "
                f"acc={metrics['accuracy']:.4f} | "
                f"MF1={metrics['macro_f1']:.4f} | "
                f"kappa={metrics['kappa']:.4f}"
            )

        # ----------------------------------------------------
        # Latest
        # ----------------------------------------------------

        save_checkpoint(
            args.output_dir
            / "checkpoint_latest.pt",
            model,
            optimizer,
            epoch,
            val_metrics,
            config,
        )

        # ----------------------------------------------------
        # Best macro-F1
        # ----------------------------------------------------

        if (
            val_metrics[
                "macro_f1"
            ]
            > best_macro_f1
        ):

            best_macro_f1 = (
                val_metrics[
                    "macro_f1"
                ]
            )

            save_checkpoint(
                args.output_dir
                / "checkpoint_best_macro_f1.pt",
                model,
                optimizer,
                epoch,
                val_metrics,
                config,
            )

        # ----------------------------------------------------
        # Best accuracy
        # ----------------------------------------------------

        if (
            val_metrics[
                "accuracy"
            ]
            > best_accuracy
        ):

            best_accuracy = (
                val_metrics[
                    "accuracy"
                ]
            )

            save_checkpoint(
                args.output_dir
                / "checkpoint_best_accuracy.pt",
                model,
                optimizer,
                epoch,
                val_metrics,
                config,
            )

        (
            args.output_dir
            / "history.json"
        ).write_text(
            json.dumps(
                history,
                indent=2,
            )
        )

    print()
    print("=" * 72)
    print("TRAINING COMPLETE")
    print("=" * 72)

    print(
        "Best val macro-F1:",
        f"{best_macro_f1:.4f}",
    )

    print(
        "Best val accuracy:",
        f"{best_accuracy:.4f}",
    )


if __name__ == "__main__":
    main()
