import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
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
        parameter.requires_grad = value


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
        .cpu()
        .numpy()
    )

    reconstructed = (
        reconstruction[0, 0]
        .cpu()
        .numpy()
    )

    difference = abs(
        original - reconstructed
    )

    print()
    print(
        "Token grid:",
        tuple(indices.shape)
    )

    print(
        "Unique codes in example:",
        torch.unique(indices).numel()
    )

    fig, axes = plt.subplots(
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
        "Ground-truth EEG"
    )

    axes[1].imshow(
        reconstructed,
        aspect="auto",
        origin="lower",
        vmin=0,
        vmax=1,
    )

    axes[1].set_title(
        "VQGAN + adversarial reconstruction"
    )

    axes[2].imshow(
        difference,
        aspect="auto",
        origin="lower",
    )

    axes[2].set_title(
        "Absolute difference"
    )

    for ax in axes:
        ax.set_xlabel(
            "Time (30-second epochs)"
        )
        ax.set_ylabel(
            "Frequency bins"
        )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        default="outputs/shhs_preprocessed",
    )

    parser.add_argument(
        "--vqgan-checkpoint",
        default="outputs/vqgan/checkpoint_best.pt",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/vqgan_adversarial",
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

    args = parser.parse_args()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    dataset = PhysiologyPairDataset(
        args.data_dir
    )

    print(
        "Samples:",
        len(dataset)
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    # ----------------------
    # VQGAN
    # ----------------------

    model = VQGAN().to(device)

    checkpoint = torch.load(
        args.vqgan_checkpoint,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        "Loaded VQGAN checkpoint:",
        args.vqgan_checkpoint
    )

    # ----------------------
    # Discriminator
    # ----------------------

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

    best_recon = float("inf")

    for epoch in range(args.epochs):

        model.train()
        discriminator.train()

        total_generator = 0.0
        total_discriminator = 0.0

        total_recon = 0.0
        total_vq = 0.0
        total_corr = 0.0
        total_adv = 0.0

        for batch in loader:

            eeg = (
                batch["eeg_spectrogram"]
                .unsqueeze(1)
                .to(device)
            )

            # ==================================
            # 1. Train discriminator
            # ==================================

            set_requires_grad(
                discriminator,
                True,
            )

            optimizer_disc.zero_grad()

            with torch.no_grad():

                fake_eeg, _, _ = model(
                    eeg
                )

            real_logits = discriminator(
                eeg
            )

            fake_logits = discriminator(
                fake_eeg.detach()
            )

            d_loss = (
                discriminator_hinge_loss(
                    real_logits,
                    fake_logits,
                )
            )

            d_loss.backward()

            optimizer_disc.step()

            # ==================================
            # 2. Train VQGAN / generator
            # ==================================

            set_requires_grad(
                discriminator,
                False,
            )

            optimizer_vqgan.zero_grad()

            reconstruction, indices, vq_loss = (
                model(eeg)
            )

            recon_loss = F.l1_loss(
                reconstruction,
                eeg,
            )

            corr_loss = correlation_loss(
                reconstruction,
                eeg,
            )

            fake_logits = discriminator(
                reconstruction
            )

            adv_loss = (
                generator_adversarial_loss(
                    fake_logits
                )
            )

            generator_loss = (
                recon_loss
                + vq_loss
                + args.corr_weight * corr_loss
                + args.adv_weight * adv_loss
            )

            generator_loss.backward()

            optimizer_vqgan.step()

            set_requires_grad(
                discriminator,
                True,
            )

            total_generator += (
                generator_loss.item()
            )

            total_discriminator += (
                d_loss.item()
            )

            total_recon += (
                recon_loss.item()
            )

            total_vq += (
                vq_loss.item()
            )

            total_corr += (
                corr_loss.item()
            )

            total_adv += (
                adv_loss.item()
            )

        n = len(loader)

        avg_generator = (
            total_generator / n
        )

        avg_discriminator = (
            total_discriminator / n
        )

        avg_recon = (
            total_recon / n
        )

        avg_vq = (
            total_vq / n
        )

        avg_corr = (
            total_corr / n
        )

        avg_adv = (
            total_adv / n
        )

        print(
            f"Epoch {epoch + 1:03d} | "
            f"G={avg_generator:.4f} | "
            f"D={avg_discriminator:.4f} | "
            f"recon={avg_recon:.4f} | "
            f"vq={avg_vq:.4f} | "
            f"corr={avg_corr:.4f} | "
            f"adv={avg_adv:.4f}"
        )

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict":
                model.state_dict(),
            "discriminator_state_dict":
                discriminator.state_dict(),
            "optimizer_vqgan_state_dict":
                optimizer_vqgan.state_dict(),
            "optimizer_disc_state_dict":
                optimizer_disc.state_dict(),
            "reconstruction_loss":
                avg_recon,
        }

        torch.save(
            checkpoint,
            output_dir
            / "checkpoint_latest.pt",
        )

        # Use reconstruction loss to select
        # the best checkpoint because GAN
        # generator loss can be negative.
        if avg_recon < best_recon:

            best_recon = avg_recon

            torch.save(
                checkpoint,
                output_dir
                / "checkpoint_best.pt",
            )

    save_reconstruction(
        model=model,
        dataset=dataset,
        device=device,
        output_path=(
            output_dir
            / "reconstruction.png"
        ),
    )

    print()
    print(
        "Best reconstruction loss:",
        best_recon
    )


if __name__ == "__main__":
    main()
