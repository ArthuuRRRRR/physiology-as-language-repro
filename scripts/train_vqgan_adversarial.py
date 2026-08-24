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
from src.models.discriminator import (
    PatchDiscriminator,
    discriminator_hinge_loss,
    generator_adversarial_loss,
)


def correlation_loss(x, y, eps=1e-8):
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


def set_requires_grad(model, value):
    for parameter in model.parameters():
        parameter.requires_grad_(value)


def load_datasets(
    data_root,
    normalization_file,
    held_out_fold,
):
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


def reconstruction_losses(
    reconstruction,
    target,
    vq_loss,
    corr_weight,
):
    reconstruction_loss = F.l1_loss(
        reconstruction,
        target,
    )

    corr_loss = correlation_loss(
        reconstruction,
        target,
    )

    base_loss = (
        reconstruction_loss
        + vq_loss
        + corr_weight * corr_loss
    )

    return (
        base_loss,
        reconstruction_loss,
        corr_loss,
    )


def train_epoch(
    model,
    discriminator,
    loader,
    optimizer_vqgan,
    optimizer_disc,
    device,
    corr_weight,
    adv_weight,
    max_batches=None,
):
    model.train()
    discriminator.train()

    totals = {
        "generator": 0.0,
        "discriminator": 0.0,
        "reconstruction": 0.0,
        "vq": 0.0,
        "correlation": 0.0,
        "adversarial": 0.0,
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

        # ----------------------------------
        # 1. Train discriminator
        # ----------------------------------

        discriminator.train()

        set_requires_grad(
            discriminator,
            True,
        )

        optimizer_disc.zero_grad(
            set_to_none=True
        )

        with torch.no_grad():
            fake_eeg, _, _ = model(eeg)

        real_logits = discriminator(eeg)

        fake_logits = discriminator(
            fake_eeg.detach()
        )

        discriminator_loss = (
            discriminator_hinge_loss(
                real_logits,
                fake_logits,
            )
        )

        discriminator_loss.backward()
        optimizer_disc.step()

        # ----------------------------------
        # 2. Train VQGAN generator
        # ----------------------------------

        discriminator.eval()

        set_requires_grad(
            discriminator,
            False,
        )

        optimizer_vqgan.zero_grad(
            set_to_none=True
        )

        reconstruction, _, vq_loss = model(eeg)

        (
            base_loss,
            reconstruction_loss,
            corr_loss,
        ) = reconstruction_losses(
            reconstruction=reconstruction,
            target=eeg,
            vq_loss=vq_loss,
            corr_weight=corr_weight,
        )

        fake_logits = discriminator(
            reconstruction
        )

        adversarial_loss = (
            generator_adversarial_loss(
                fake_logits
            )
        )

        generator_loss = (
            base_loss
            + adv_weight * adversarial_loss
        )

        generator_loss.backward()
        optimizer_vqgan.step()

        set_requires_grad(
            discriminator,
            True,
        )

        totals["generator"] += (
            generator_loss.item()
        )

        totals["discriminator"] += (
            discriminator_loss.item()
        )

        totals["reconstruction"] += (
            reconstruction_loss.item()
        )

        totals["vq"] += vq_loss.item()

        totals["correlation"] += (
            corr_loss.item()
        )

        totals["adversarial"] += (
            adversarial_loss.item()
        )

        processed_batches += 1

    if processed_batches == 0:
        raise RuntimeError(
            "No training batch was processed."
        )

    return {
        name: value / processed_batches
        for name, value in totals.items()
    }


@torch.no_grad()
def validate_epoch(
    model,
    loader,
    device,
    corr_weight,
    max_batches=None,
):
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

        reconstruction, _, vq_loss = model(eeg)

        (
            base_loss,
            reconstruction_loss,
            corr_loss,
        ) = reconstruction_losses(
            reconstruction=reconstruction,
            target=eeg,
            vq_loss=vq_loss,
            corr_weight=corr_weight,
        )

        totals["loss"] += base_loss.item()

        totals["reconstruction"] += (
            reconstruction_loss.item()
        )

        totals["vq"] += vq_loss.item()

        totals["correlation"] += (
            corr_loss.item()
        )

        processed_batches += 1

    if processed_batches == 0:
        raise RuntimeError(
            "No validation batch was processed."
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
        "Token grid:",
        tuple(indices.shape),
    )

    print(
        "Unique codes in example:",
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
        "Adversarial VQGAN reconstruction"
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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Adversarial fine-tuning of a pretrained "
            "SHHS VQGAN."
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
        "--vqgan-checkpoint",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/vqgan_adversarial"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
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
        "--disc-lr",
        type=float,
        default=4.8e-5,
    )

    parser.add_argument(
        "--corr-weight",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--adv-weight",
        type=float,
        default=0.01,
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
    )

    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
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

    if args.vqgan_checkpoint is None:
        vqgan_checkpoint = (
            Path("outputs/vqgan")
            / f"held_out_fold_{args.held_out_fold}"
            / "checkpoint_best.pt"
        )
    else:
        vqgan_checkpoint = (
            args.vqgan_checkpoint
        )

    if not vqgan_checkpoint.exists():
        raise FileNotFoundError(
            f"Baseline checkpoint not found: "
            f"{vqgan_checkpoint}"
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
    print("Baseline:", vqgan_checkpoint)
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
    print(
        "Adversarial weight:",
        args.adv_weight,
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

    baseline_checkpoint = torch.load(
        vqgan_checkpoint,
        map_location=device,
    )

    model.load_state_dict(
        baseline_checkpoint[
            "model_state_dict"
        ]
    )

    discriminator = (
        PatchDiscriminator()
        .to(device)
    )

    optimizer_vqgan = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    optimizer_disc = torch.optim.Adam(
        discriminator.parameters(),
        lr=args.disc_lr,
    )

    best_validation_loss = float("inf")

    best_checkpoint_path = (
        run_output_dir
        / "checkpoint_best.pt"
    )

    for epoch in range(1, args.epochs + 1):

        train_metrics = train_epoch(
            model=model,
            discriminator=discriminator,
            loader=train_loader,
            optimizer_vqgan=optimizer_vqgan,
            optimizer_disc=optimizer_disc,
            device=device,
            corr_weight=args.corr_weight,
            adv_weight=args.adv_weight,
            max_batches=(
                args.max_train_batches
            ),
        )

        validation_metrics = validate_epoch(
            model=model,
            loader=validation_loader,
            device=device,
            corr_weight=args.corr_weight,
            max_batches=(
                args.max_val_batches
            ),
        )

        print(
            f"Epoch {epoch:03d} | "
            f"G={train_metrics['generator']:.4f} | "
            f"D={train_metrics['discriminator']:.4f} | "
            f"adv={train_metrics['adversarial']:.4f} | "
            f"train_recon="
            f"{train_metrics['reconstruction']:.4f} | "
            f"val={validation_metrics['loss']:.4f} | "
            f"val_recon="
            f"{validation_metrics['reconstruction']:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "held_out_fold": (
                args.held_out_fold
            ),
            "train_folds": train_folds,
            "min_db": min_db,
            "max_db": max_db,
            "corr_weight": args.corr_weight,
            "adv_weight": args.adv_weight,
            "baseline_checkpoint": str(
                vqgan_checkpoint
            ),
            "model_state_dict": (
                model.state_dict()
            ),
            "discriminator_state_dict": (
                discriminator.state_dict()
            ),
            "optimizer_vqgan_state_dict": (
                optimizer_vqgan.state_dict()
            ),
            "optimizer_disc_state_dict": (
                optimizer_disc.state_dict()
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