from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset


IGNORE_INDEX = -100
NUM_CLASSES = 4
INPUT_DIM = 256


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Match Keondo's reproducibility settings.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


# ============================================================
# Dataset
# ============================================================

class SleepStagingDataset(Dataset):

    def __init__(
        self,
        root: Path,
        split: str,
        source: str,
    ):

        self.split_dir = (
            root
            / split
        )

        metadata_path = (
            self.split_dir
            / "metadata.json"
        )

        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            self.metadata = json.load(file)

        self.num_samples = int(
            self.metadata[
                "num_samples"
            ]
        )

        if source == "gt":

            eeg_name = (
                "ground_truth_eeg.mmap"
            )

        elif source == "synth":

            eeg_name = (
                "synthesized_eeg.mmap"
            )

        else:

            raise ValueError(
                f"Unknown source: {source}"
            )

        self.eeg_path = (
            self.split_dir
            / eeg_name
        )

        self.label_path = (
            self.split_dir
            / "labels.mmap"
        )

        self.eeg = np.memmap(
            self.eeg_path,
            mode="r",
            dtype=np.float16,
            shape=(
                self.num_samples,
                256,
                512,
            ),
        )

        self.labels = np.memmap(
            self.label_path,
            mode="r",
            dtype=np.int8,
            shape=(
                self.num_samples,
                512,
            ),
        )

    def __len__(self):

        return self.num_samples

    def __getitem__(
        self,
        index,
    ):

        eeg = np.array(
            self.eeg[index],
            dtype=np.float32,
            copy=True,
        )

        labels = np.array(
            self.labels[index],
            dtype=np.int64,
            copy=True,
        )

        return {
            "eeg":
                torch.from_numpy(
                    eeg
                ),

            "labels":
                torch.from_numpy(
                    labels
                ),
        }


# ============================================================
# Linear probe
# ============================================================

class SleepStageLinear(nn.Module):

    def __init__(self):

        super().__init__()

        self.classifier = nn.Linear(
            INPUT_DIM,
            NUM_CLASSES,
        )

    def forward(
        self,
        eeg,
    ):

        # eeg:
        # (B, 256, 512)
        #
        # Each 30-s column becomes one independent
        # 256-dimensional sample.

        eeg = eeg.transpose(
            1,
            2,
        )

        # (B,512,256)
        # ->
        # (B,512,4)

        return self.classifier(
            eeg
        )


# ============================================================
# Metrics
# ============================================================

def compute_metrics(
    targets,
    predictions,
):

    targets = np.concatenate(
        targets
    )

    predictions = np.concatenate(
        predictions
    )

    return {
        "accuracy":
            accuracy_score(
                targets,
                predictions,
            ),

        "macro_f1":
            f1_score(
                targets,
                predictions,
                average="macro",
                labels=[
                    0,
                    1,
                    2,
                    3,
                ],
                zero_division=0,
            ),

        "kappa":
            cohen_kappa_score(
                targets,
                predictions,
            ),
    }


# ============================================================
# Epoch
# ============================================================

def run_epoch(
    *,
    model,
    loader,
    device,
    optimizer=None,
    scheduler=None,
    use_bf16=False,
):

    training = (
        optimizer is not None
    )

    model.train(
        training
    )

    total_loss = 0.0
    total_valid = 0

    all_targets = []
    all_predictions = []

    criterion = nn.CrossEntropyLoss(
        ignore_index=IGNORE_INDEX,
        reduction="sum",
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

        if training:

            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(
            training
        ):

            # Keondo trains/evaluates with bf16-mixed precision.
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=(
                    use_bf16
                    and device.type == "cuda"
                ),
            ):

                logits = model(
                    eeg
                )

                loss_sum = criterion(
                    logits.reshape(
                        -1,
                        NUM_CLASSES,
                    ),
                    labels.reshape(
                        -1
                    ),
                )

                valid = (
                    labels
                    != IGNORE_INDEX
                )

                n_valid = int(
                    valid.sum().item()
                )

                if n_valid == 0:
                    continue

                loss = (
                    loss_sum
                    / n_valid
                )

            if training:

                loss.backward()

                optimizer.step()

                # Keondo's CosineAnnealingLR is stepped every
                # optimizer step, not once per epoch.
                if scheduler is not None:
                    scheduler.step()

        predictions = (
            logits.argmax(
                dim=-1
            )
        )

        valid_targets = (
            labels[
                valid
            ]
            .detach()
            .cpu()
            .numpy()
        )

        valid_predictions = (
            predictions[
                valid
            ]
            .detach()
            .cpu()
            .numpy()
        )

        all_targets.append(
            valid_targets
        )

        all_predictions.append(
            valid_predictions
        )

        total_loss += float(
            loss_sum.item()
        )

        total_valid += (
            n_valid
        )

    if total_valid == 0:

        raise RuntimeError(
            "No valid sleep-stage labels."
        )

    metrics = compute_metrics(
        all_targets,
        all_predictions,
    )

    metrics[
        "cross_entropy"
    ] = (
        total_loss
        / total_valid
    )

    metrics[
        "epochs"
    ] = total_valid

    return metrics


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
            "downstream_sleep_staging_schedule75"
        ),
    )

    parser.add_argument(
        "--source",
        choices=[
            "gt",
            "synth",
        ],
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
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

    args = parser.parse_args()

    set_seed(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    print(
        "Source:",
        args.source,
    )

    # ========================================================
    # Data
    # ========================================================

    train_dataset = (
        SleepStagingDataset(
            root=args.data_root,
            split="train",
            source=args.source,
        )
    )

    val_dataset = (
        SleepStagingDataset(
            root=args.data_root,
            split="val",
            source=args.source,
        )
    )

    print(
        "Train windows:",
        len(train_dataset),
    )

    print(
        "Validation windows:",
        len(val_dataset),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(
            args.num_workers > 0
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(
            args.num_workers > 0
        ),
    )

    # ========================================================
    # Model
    # ========================================================

    model = (
        SleepStageLinear()
        .to(device)
    )

    print(
        "Trainable parameters:",
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        ),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(
            0.9,
            0.99,
        ),
        weight_decay=(
            args.weight_decay
        ),
    )

    # Keondo computes cosine decay over optimizer steps.
    # On this single-GPU linear-probe run there is one optimizer
    # update per train-loader batch.
    steps_per_epoch = len(train_loader)
    max_steps = int(
        steps_per_epoch
        * args.epochs
    ) + 1

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_steps,
            eta_min=1e-8,
        )
    )

    use_bf16 = (
        device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )

    print(
        "Precision:",
        "bf16-mixed"
        if use_bf16
        else "fp32",
    )
    print(
        "Scheduler:",
        f"CosineAnnealingLR per step "
        f"(T_max={max_steps}, eta_min=1e-8)",
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_path = (
        args.output_dir
        / "checkpoint_best.pt"
    )

    latest_path = (
        args.output_dir
        / "checkpoint_latest.pt"
    )

    history_path = (
        args.output_dir
        / "history.jsonl"
    )

    best_macro_f1 = (
        -float("inf")
    )

    # ========================================================
    # Training
    # ========================================================

    for epoch_index in range(
        args.epochs
    ):

        epoch = (
            epoch_index
            + 1
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            use_bf16=use_bf16,
        )

        with torch.no_grad():

            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                use_bf16=use_bf16,
            )

        print()
        print(
            f"Epoch {epoch:02d}/"
            f"{args.epochs}"
            f" | LR={current_lr:.8g}"
        )

        print(
            "Train"
            f" | CE={train_metrics['cross_entropy']:.4f}"
            f" | ACC={train_metrics['accuracy']:.4f}"
            f" | MF1={train_metrics['macro_f1']:.4f}"
            f" | Kappa={train_metrics['kappa']:.4f}"
        )

        print(
            "Val  "
            f" | CE={val_metrics['cross_entropy']:.4f}"
            f" | ACC={val_metrics['accuracy']:.4f}"
            f" | MF1={val_metrics['macro_f1']:.4f}"
            f" | Kappa={val_metrics['kappa']:.4f}"
        )

        checkpoint = {
            "epoch":
                epoch,

            "source":
                args.source,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "train_metrics":
                train_metrics,

            "validation_metrics":
                val_metrics,

            "config": {
                "input_dim":
                    INPUT_DIM,

                "num_classes":
                    NUM_CLASSES,

                "epochs":
                    args.epochs,

                "batch_size":
                    args.batch_size,

                "lr":
                    args.lr,

                "weight_decay":
                    args.weight_decay,

                "betas": [
                    0.9,
                    0.99,
                ],

                "scheduler":
                    "cosine_per_optimizer_step",

                "scheduler_t_max":
                    max_steps,

                "final_lr":
                    1e-8,

                "precision":
                    (
                        "bf16-mixed"
                        if use_bf16
                        else "fp32"
                    ),
            },
        }

        torch.save(
            checkpoint,
            latest_path,
        )

        # Best model selected ONLY using validation.
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

            torch.save(
                checkpoint,
                best_path,
            )

            best_marker = (
                " [BEST]"
            )

        else:

            best_marker = ""

        with history_path.open(
            "a",
            encoding="utf-8",
        ) as file:

            file.write(
                json.dumps(
                    {
                        "epoch":
                            epoch,

                        "lr":
                            current_lr,

                        "train":
                            train_metrics,

                        "validation":
                            val_metrics,
                    }
                )
                + "\n"
            )

        print(
            "Best VAL MF1:",
            f"{best_macro_f1:.4f}"
            + best_marker,
        )

    print()
    print("=" * 72)
    print("TRAINING COMPLETE")
    print("=" * 72)

    print(
        "Source:",
        args.source,
    )

    print(
        "Best validation MF1:",
        best_macro_f1,
    )

    print(
        "Checkpoint:",
        best_path,
    )


if __name__ == "__main__":
    main()