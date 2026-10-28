from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader

from paper_architecture.downstream.ViT.dataset import build_multi_dataset
from paper_architecture.downstream.ViT.model import SleepStageViT


STAGE_NAMES = ["Wake", "Light", "Deep", "REM"]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
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

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 72)
    print("ViT SLEEP STAGING — HELD-OUT EVALUATION")
    print("=" * 72)
    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print("Checkpoint:", args.checkpoint)
    print("Data root :", args.data_root)
    print("Split     :", args.split)

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    split_dir = args.data_root / args.split

    dataset = build_multi_dataset(
        [split_dir]
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    print()
    print("Windows:", len(dataset))

    # ---------------------------------------------------------
    # Checkpoint
    # ---------------------------------------------------------

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    model_config = checkpoint.get(
        "model_config",
        {},
    )

    model = SleepStageViT(
        **model_config
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model = model.to(device)
    model.eval()

    print(
        "Checkpoint epoch:",
        checkpoint.get("epoch", "unknown"),
    )

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    all_targets = []
    all_predictions = []
    subject_ids = set()

    total_loss = 0.0
    total_segments = 0

    use_amp = device.type == "cuda"

    with torch.no_grad():

        for batch_index, batch in enumerate(
            loader,
            start=1,
        ):

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

            valid = (
                (labels >= 0)
                & (labels < 4)
            )

            valid_logits = logits[valid]
            valid_labels = labels[valid]
            valid_predictions = predictions[valid]

            if valid_labels.numel() > 0:

                loss = F.cross_entropy(
                    valid_logits.float(),
                    valid_labels,
                    reduction="sum",
                )

                total_loss += float(
                    loss.item()
                )

                total_segments += int(
                    valid_labels.numel()
                )

                all_targets.append(
                    valid_labels
                    .detach()
                    .cpu()
                    .numpy()
                )

                all_predictions.append(
                    valid_predictions
                    .detach()
                    .cpu()
                    .numpy()
                )

            ids = batch.get(
                "subject_id",
                None,
            )

            if ids is not None:
                if isinstance(
                    ids,
                    (list, tuple),
                ):
                    subject_ids.update(
                        str(x) for x in ids
                    )
                else:
                    subject_ids.add(
                        str(ids)
                    )

            if (
                batch_index % 25 == 0
                or batch_index == len(loader)
            ):
                print(
                    f"{batch_index}/{len(loader)}",
                    flush=True,
                )

    if not all_targets:
        raise RuntimeError(
            "No valid labels found."
        )

    targets = np.concatenate(
        all_targets
    )

    predictions = np.concatenate(
        all_predictions
    )

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    accuracy = accuracy_score(
        targets,
        predictions,
    )

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

    f1_per_class = f1_score(
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

    print()
    print("=" * 72)
    print("FINAL HELD-OUT RESULTS")
    print("=" * 72)

    print(
        "Subjects      :",
        len(subject_ids)
        if subject_ids
        else "unknown",
    )
    print(
        "Windows       :",
        len(dataset),
    )
    print(
        "Valid segments:",
        len(targets),
    )
    print(
        "CE loss       :",
        f"{total_loss / total_segments:.4f}",
    )
    print(
        "Accuracy      :",
        f"{accuracy:.4f}",
        f"({100 * accuracy:.2f}%)",
    )
    print(
        "Macro-F1      :",
        f"{macro_f1:.4f}",
    )
    print(
        "Kappa         :",
        f"{kappa:.4f}",
    )

    print()
    print("F1 per class:")

    for name, value in zip(
        STAGE_NAMES,
        f1_per_class,
    ):
        print(
            f"  {name:5s}: {value:.4f}"
        )

    print()
    print("Confusion matrix")
    print("(rows=true, columns=predicted)")
    print(cm)


if __name__ == "__main__":
    main()
