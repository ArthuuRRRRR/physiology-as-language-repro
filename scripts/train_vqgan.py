import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import PhysiologyPairDataset
from src.models.vqgan import VQGAN


def correlation_loss(x, y, eps=1e-8):
    """
    Encourage reconstruction to preserve the global
    correlation structure of the target spectrogram.
    """

    x = x.flatten(start_dim=1)
    y = y.flatten(start_dim=1)

    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)

    numerator = (x * y).sum(dim=1)

    denominator = (
        torch.sqrt((x ** 2).sum(dim=1) + eps)
        * torch.sqrt((y ** 2).sum(dim=1) + eps)
    )

    correlation = numerator / denominator

    return 1.0 - correlation.mean()


def compute_losses(model, eeg):
    """
    Run the VQGAN and calculate all training losses.
    """

    reconstruction, indices, vq_loss = model(eeg)

    reconstruction_loss = F.l1_loss(
        reconstruction,
        eeg,
    )

    corr_loss = correlation_loss(
        reconstruction,
        eeg,
    )

    total_loss = (
        reconstruction_loss
        + vq_loss
        + 0.1 * corr_loss
    )

    return {
        "loss": total_loss,
        "reconstruction": reconstruction_loss,
        "vq": vq_loss,
        "correlation": corr_loss,
        "indices": indices,
    }


def run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    max_batches=None,
):
    """
    Run one training or validation epoch.
    """

    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    totals = {
        "loss": 0.0,
        "reconstruction": 0.0,
        "vq": 0.0,
        "correlation": 0.0,
    }

    processed_batches = 0

    for batch_index, batch in enumerate(loader):

        if (
            max_batches is not None
            and batch_index >= max_batches
        ):
            break

        eeg = (
            batch["eeg_spectrogram"]
            .unsqueeze(1)
            .to(device)
        )

        if training:
            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(training):
            losses = compute_losses(
                model,
                eeg,
            )

            if training:
                losses["loss"].backward()
                optimizer.step()

        for name in totals:
            totals[name] += losses[name].item()

        processed_batches += 1

    if processed_batches == 0:
        raise RuntimeError(
            "No batch was processed."
        )

    return {
        name: value / processed_batches
        for name, value in totals.items()
    }


def save_reconstruction(
    model,
    dataset,
    device,
    output_path,
):
    """
    Save one validation target and its reconstruction.
    """

    model.eval()

    sample = dataset[0]

    eeg = (
        sample["eeg_spectrogram"]
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        reconstruction, indices, _ = model(eeg)

    original = (
        eeg[0, 0]
        .detach()
        .cpu()
        .numpy()
    )

    reconstructed = (
        reconstruction[0, 0]
        .detach()
        .cpu()
        .numpy()
    )

    difference = np.abs(
        original - reconstructed
    )

    print(
        "Token grid shape:",
        tuple(indices.shape),
    )

    print(
        "Unique codebook entries:",
        torch.unique(indices).numel(),
    )

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(12, 8),
    )

    axes[0].imshow(
        original,
        aspect="auto",
        origin="lower",
        vmin=0,
        vmax=1,
    )
    axes[0].set_title(
        "Ground-truth EEG spectrogram"
    )

    axes[1].imshow(
        reconstructed,
        aspect="auto",
        origin="lower",
        vmin=0,
        vmax=1,
    )
    axes[1].set_title(
        "VQGAN reconstruction"
    )

    axes[2].imshow(
        difference,
        aspect="auto",
        origin="lower",
    )
    axes[2].set_title(
        "Absolute difference"
    )

    for axis in axes:
        axis.set_xlabel(
            "Time (30-second epochs)"
        )
        axis.set_ylabel(
            "Frequency bins"
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(
        f"Reconstruction saved: {output_path}"
    )


def load_datasets(
    data_root,
    normalization_file,
    held_out_fold,
):
    """
    Build training and validation datasets for one CV run.
    """

    with normalization_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        normalization = json.load(file)

    run = normalization["runs"][
        str(held_out_fold)
    ]

    train_folds = run["train_folds"]
    min_db = float(run["min_db"])
    max_db = float(run["max_db"])

    train_directories = [
        data_root / f"fold_{fold}"
        for fold in train_folds
    ]

    validation_directory = (
        data_root / f"fold_{held_out_fold}"
    )

    train_dataset = PhysiologyPairDataset(
        train_directories,
        min_db=min_db,
        max_db=max_db,
    )

    validation_dataset = PhysiologyPairDataset(
        validation_directory,
        min_db=min_db,
        max_db=max_db,
    )

    return (
        train_dataset,
        validation_dataset,
        train_folds,
        min_db,
        max_db,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train the EEG VQGAN using patient-wise "
            "SHHS cross-validation."
        )
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "outputs/shhs_preprocessed"
        ),
    )

    parser.add_argument(
        "--normalization-file",
        type=Path,
        default=Path(
            "outputs/normalization/"
            "shhs_cv_normalization.json"
        ),
    )

    parser.add_argument(
        "--held-out-fold",
        type=int,
        choices=[0, 1, 2, 3],
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/vqgan"),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=4.8e-5,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Limit training batches for a smoke test.",
    )

    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Limit validation batches for a smoke test.",
    )

    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    run_output_dir = (
        args.output_dir
        / f"held_out_fold_{args.held_out_fold}"
    )

    run_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        train_dataset,
        validation_dataset,
        train_folds,
        min_db,
        max_db,
    ) = load_datasets(
        data_root=args.data_root,
        normalization_file=(
            args.normalization_file
        ),
        held_out_fold=args.held_out_fold,
    )

    print("Device:", device)
    print("Training folds:", train_folds)
    print(
        "Training samples:",
        len(train_dataset),
    )
    print(
        "Held-out fold:",
        args.held_out_fold,
    )
    print(
        "Validation samples:",
        len(validation_dataset),
    )
    print(
        f"Normalization: "
        f"[{min_db:.2f}, {max_db:.2f}] dB"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = VQGAN().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    best_validation_loss = float("inf")

    best_checkpoint_path = (
        run_output_dir
        / "checkpoint_best.pt"
    )

    for epoch in range(1, args.epochs + 1):

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            max_batches=(
                args.max_train_batches
            ),
        )

        with torch.no_grad():
            validation_metrics = run_epoch(
                model=model,
                loader=validation_loader,
                device=device,
                optimizer=None,
                max_batches=(
                    args.max_val_batches
                ),
            )

        print(
            f"Epoch {epoch:03d} | "
            f"train={train_metrics['loss']:.4f} | "
            f"val={validation_metrics['loss']:.4f} | "
            f"recon={validation_metrics['reconstruction']:.4f} | "
            f"vq={validation_metrics['vq']:.4f} | "
            f"corr={validation_metrics['correlation']:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "held_out_fold": args.held_out_fold,
            "train_folds": train_folds,
            "min_db": min_db,
            "max_db": max_db,
            "model_state_dict": (
                model.state_dict()
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "train_metrics": train_metrics,
            "validation_metrics": (
                validation_metrics
            ),
        }

        torch.save(
            checkpoint,
            run_output_dir
            / "checkpoint_latest.pt",
        )

        if (
            validation_metrics["loss"]
            < best_validation_loss
        ):
            best_validation_loss = (
                validation_metrics["loss"]
            )

            torch.save(
                checkpoint,
                best_checkpoint_path,
            )

    best_checkpoint = torch.load(
        best_checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        best_checkpoint["model_state_dict"]
    )

    save_reconstruction(
        model=model,
        dataset=validation_dataset,
        device=device,
        output_path=(
            run_output_dir
            / "reconstruction.png"
        ),
    )

    print(
        f"Best validation loss: "
        f"{best_validation_loss:.4f}"
    )


if __name__ == "__main__":
    main()