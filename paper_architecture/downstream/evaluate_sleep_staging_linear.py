from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset


IGNORE_INDEX = -100
NUM_CLASSES = 4
INPUT_DIM = 256

CLASS_NAMES = [
    "Wake",
    "Light",
    "Deep",
    "REM",
]


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

        with (
            self.split_dir
            / "metadata.json"
        ).open(
            "r",
            encoding="utf-8",
        ) as file:

            metadata = json.load(
                file
            )

        self.num_samples = int(
            metadata[
                "num_samples"
            ]
        )

        if source == "gt":

            filename = (
                "ground_truth_eeg.mmap"
            )

        elif source == "synth":

            filename = (
                "synthesized_eeg.mmap"
            )

        else:

            raise ValueError(
                source
            )

        self.eeg = np.memmap(
            self.split_dir
            / filename,
            mode="r",
            dtype=np.float16,
            shape=(
                self.num_samples,
                256,
                512,
            ),
        )

        self.labels = np.memmap(
            self.split_dir
            / "labels.mmap",
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
# Model
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

        # (B,256,512)
        # ->
        # (B,512,256)
        # ->
        # (B,512,4)

        eeg = eeg.transpose(
            1,
            2,
        )

        return self.classifier(
            eeg
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
            "downstream_sleep_staging_schedule75"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
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
        "--split",
        choices=[
            "train",
            "val",
            "test",
        ],
        default="test",
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
        "--output",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # ========================================================
    # Checkpoint
    # ========================================================

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    checkpoint_source = (
        checkpoint.get(
            "source"
        )
    )

    if (
        checkpoint_source
        is not None
        and checkpoint_source
        != args.source
    ):

        raise ValueError(
            "Checkpoint was trained on "
            f"{checkpoint_source}, "
            f"but evaluation requested "
            f"{args.source}."
        )

    model = (
        SleepStageLinear()
        .to(device)
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    model.eval()

    # ========================================================
    # Dataset
    # ========================================================

    dataset = (
        SleepStagingDataset(
            root=args.data_root,
            split=args.split,
            source=args.source,
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(
            args.num_workers > 0
        ),
    )

    print()
    print("=" * 72)
    print("SLEEP STAGING LINEAR PROBE")
    print("=" * 72)

    print(
        "Source:",
        args.source,
    )

    print(
        "Split:",
        args.split,
    )

    print(
        "Windows:",
        len(dataset),
    )

    print(
        "Checkpoint epoch:",
        checkpoint.get(
            "epoch",
            "unknown",
        ),
    )

    # ========================================================
    # Inference
    # ========================================================

    all_targets = []
    all_predictions = []

    with torch.inference_mode():

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

            logits = model(
                eeg
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

            all_targets.append(
                labels[
                    valid
                ]
                .cpu()
                .numpy()
            )

            all_predictions.append(
                predictions[
                    valid
                ]
                .cpu()
                .numpy()
            )

    targets = np.concatenate(
        all_targets
    )

    predictions = np.concatenate(
        all_predictions
    )

    # ========================================================
    # Metrics
    # ========================================================

    accuracy = accuracy_score(
        targets,
        predictions,
    )

    macro_f1 = f1_score(
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
    )

    kappa = cohen_kappa_score(
        targets,
        predictions,
    )

    per_class_f1 = f1_score(
        targets,
        predictions,
        average=None,
        labels=[
            0,
            1,
            2,
            3,
        ],
        zero_division=0,
    )

    cm = confusion_matrix(
        targets,
        predictions,
        labels=[
            0,
            1,
            2,
            3,
        ],
    )

    print()
    print("=" * 72)
    print("FINAL METRICS")
    print("=" * 72)

    print(
        "Epochs:",
        len(targets),
    )

    print(
        f"Accuracy : {accuracy:.4f}"
        f" ({accuracy * 100:.2f}%)"
    )

    print(
        f"Macro-F1 : {macro_f1:.4f}"
        f" ({macro_f1 * 100:.2f}%)"
    )

    print(
        f"Cohen κ  : {kappa:.4f}"
    )

    print()
    print("Per-class F1:")

    for name, value in zip(
        CLASS_NAMES,
        per_class_f1,
    ):

        print(
            f"  {name:<6}: "
            f"{value:.4f}"
        )

    print()
    print(
        "Confusion matrix "
        "(rows=true, columns=pred):"
    )

    print(
        cm
    )

    print("=" * 72)

    # ========================================================
    # Save metrics
    # ========================================================

    result = {
        "source":
            args.source,

        "split":
            args.split,

        "checkpoint":
            str(
                args.checkpoint
            ),

        "checkpoint_epoch":
            checkpoint.get(
                "epoch"
            ),

        "num_epochs":
            int(
                len(targets)
            ),

        "accuracy":
            float(
                accuracy
            ),

        "macro_f1":
            float(
                macro_f1
            ),

        "cohen_kappa":
            float(
                kappa
            ),

        "per_class_f1": {
            name:
                float(value)

            for name, value
            in zip(
                CLASS_NAMES,
                per_class_f1,
            )
        },

        "confusion_matrix":
            cm.tolist(),
    }

    if args.output is not None:

        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with args.output.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                result,
                file,
                indent=2,
            )

        print(
            "Saved:",
            args.output,
        )


if __name__ == "__main__":
    main()