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


def correlation_loss(x, y, eps=1e-8):
    """
    Encourage reconstruction to preserve the correlation
    structure of the target spectrogram.
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


def save_reconstruction(
    model,
    dataset,
    device,
    output_path,
):
    """
    Save one ground-truth / reconstruction comparison.
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

    difference = abs(
        original - reconstructed
    )

    print()
    print(
        "Token grid shape:",
        tuple(indices.shape),
    )

    print(
        "Unique codebook entries used:",
        torch.unique(indices).numel(),
    )

    print(
        "Total tokens:",
        indices.numel(),
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

    print(
        f"Reconstruction saved to: "
        f"{output_path}"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        default="outputs/mesa_preprocessed",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/vqgan",
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
        "Number of samples:",
        len(dataset),
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    model = VQGAN().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    best_loss = float("inf")

    for epoch in range(args.epochs):

        model.train()

        total_loss = 0.0
        total_recon = 0.0
        total_vq = 0.0
        total_corr = 0.0

        for batch in loader:

            eeg = (
                batch["eeg_spectrogram"]
                .unsqueeze(1)
                .to(device)
            )

            optimizer.zero_grad()

            reconstruction, indices, vq_loss = model(
                eeg
            )

            reconstruction_loss = F.l1_loss(
                reconstruction,
                eeg,
            )

            corr_loss = correlation_loss(
                reconstruction,
                eeg,
            )

            loss = (
                reconstruction_loss
                + vq_loss
                + 0.1 * corr_loss
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            total_recon += (
                reconstruction_loss.item()
            )

            total_vq += vq_loss.item()

            total_corr += (
                corr_loss.item()
            )

        n = len(loader)

        avg_loss = total_loss / n
        avg_recon = total_recon / n
        avg_vq = total_vq / n
        avg_corr = total_corr / n

        print(
            f"Epoch {epoch + 1:03d} | "
            f"loss={avg_loss:.4f} | "
            f"recon={avg_recon:.4f} | "
            f"vq={avg_vq:.4f} | "
            f"corr={avg_corr:.4f}"
        )

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": (
                model.state_dict()
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "loss": avg_loss,
        }

        # Always keep latest checkpoint
        torch.save(
            checkpoint,
            output_dir
            / "checkpoint_latest.pt",
        )

        # Keep best checkpoint
        if avg_loss < best_loss:

            best_loss = avg_loss

            torch.save(
                checkpoint,
                output_dir
                / "checkpoint_best.pt",
            )

    # Visual reconstruction after training
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
        f"Best training loss: "
        f"{best_loss:.4f}"
    )


if __name__ == "__main__":
    main()