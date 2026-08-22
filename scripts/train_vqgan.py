import argparse

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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        default="outputs/mesa_preprocessed",
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

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    dataset = PhysiologyPairDataset(
        args.data_dir
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
            total_recon += reconstruction_loss.item()
            total_vq += vq_loss.item()
            total_corr += corr_loss.item()

        n = len(loader)

        print(
            f"Epoch {epoch + 1:03d} | "
            f"loss={total_loss / n:.4f} | "
            f"recon={total_recon / n:.4f} | "
            f"vq={total_vq / n:.4f} | "
            f"corr={total_corr / n:.4f}"
        )


if __name__ == "__main__":
    main()
