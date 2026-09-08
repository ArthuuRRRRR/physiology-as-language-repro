"""Train a shared EEG VQGAN from multi-dataset JSONL manifests.

The disclosed Physiology-as-Language VQGAN settings are used by default:
Adam, learning rate 4.8e-5, 200 epochs, and effective batch size 120.

The paper does not disclose the correlation/adversarial weights.  Their
defaults (0.1 and 0.01) are inherited from this repository's earlier SHHS
training scripts and are therefore recorded in every checkpoint.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from src.models.discriminator import (
    PatchDiscriminator,
    discriminator_hinge_loss,
    generator_adversarial_loss,
)
from src.models.vqgan import VQGAN


class ManifestEEGDataset(Dataset):
    """Load and normalize EEG spectrograms listed in a JSONL manifest."""

    def __init__(
        self,
        manifest_path: Path,
        min_db: float,
        max_db: float,
    ) -> None:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        if max_db <= min_db:
            raise ValueError(f"Invalid dB bounds: {min_db}, {max_db}")

        self.manifest_path = manifest_path
        self.min_db = float(min_db)
        self.max_db = float(max_db)
        self.entries: list[dict] = []

        with manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if "path" not in entry or "dataset" not in entry:
                    raise KeyError(
                        f"Invalid entry at {manifest_path}:{line_number}"
                    )
                self.entries.append(entry)

        if not self.entries:
            raise RuntimeError(f"Empty manifest: {manifest_path}")

    def __len__(self) -> int:
        return len(self.entries)

    @staticmethod
    def _resolve_path(path_value: str) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        entry = self.entries[index]
        path = self._resolve_path(entry["path"])

        if not path.exists():
            raise FileNotFoundError(f"Sample not found: {path}")

        with np.load(path, allow_pickle=False) as sample:
            if "eeg_spectrogram_db" not in sample:
                raise KeyError(
                    f"eeg_spectrogram_db missing from {path}; "
                    f"available keys: {sample.files}"
                )
            eeg_db = sample["eeg_spectrogram_db"].astype(
                np.float32,
                copy=False,
            )

        if eeg_db.shape != (256, 512):
            raise ValueError(
                f"Expected (256, 512) in {path}, got {eeg_db.shape}"
            )
        if not np.isfinite(eeg_db).all():
            raise ValueError(f"Non-finite EEG spectrogram in {path}")

        eeg = (eeg_db - self.min_db) / (self.max_db - self.min_db)
        eeg = np.clip(eeg, 0.0, 1.0).astype(np.float32, copy=False)

        return {
            "eeg_spectrogram": torch.from_numpy(eeg.copy()),
            "dataset": str(entry["dataset"]),
            "file_path": str(path),
        }

    def dataset_counts(self) -> dict[str, int]:
        counts = Counter(str(entry["dataset"]) for entry in self.entries)
        return dict(sorted(counts.items()))


def correlation_loss(x: Tensor, y: Tensor, eps: float = 1e-8) -> Tensor:
    """One minus mean per-example Pearson correlation."""
    x = x.float().flatten(start_dim=1)
    y = y.float().flatten(start_dim=1)
    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    numerator = (x * y).sum(dim=1)
    denominator = torch.sqrt((x.square()).sum(dim=1) + eps) * torch.sqrt(
        (y.square()).sum(dim=1) + eps
    )
    return 1.0 - (numerator / denominator.clamp_min(eps)).mean()


def set_requires_grad(module: nn.Module, value: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(value)


def amp_context(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "none":
        return nullcontext()
    dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def build_scaler(enabled: bool):
    # torch.amp.GradScaler is preferred, but retain compatibility with older
    # PyTorch releases used by some cluster environments.
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def compute_generator_losses(
    model: VQGAN,
    discriminator: PatchDiscriminator,
    eeg: Tensor,
    corr_weight: float,
    adv_weight: float,
    adversarial_active: bool,
) -> tuple[Tensor, dict[str, Tensor], Tensor]:
    reconstruction, indices, vq_loss = model(eeg)
    reconstruction_loss = F.l1_loss(reconstruction, eeg)
    corr_loss = correlation_loss(reconstruction, eeg)

    if adversarial_active:
        adversarial_loss = generator_adversarial_loss(
            discriminator(reconstruction)
        )
    else:
        adversarial_loss = reconstruction_loss.new_zeros(())

    total = (
        reconstruction_loss
        + vq_loss
        + corr_weight * corr_loss
        + adv_weight * adversarial_loss
    )
    losses = {
        "generator": total,
        "reconstruction": reconstruction_loss,
        "vq": vq_loss,
        "correlation": corr_loss,
        "adversarial": adversarial_loss,
    }
    return reconstruction, losses, indices


def train_epoch(
    model: VQGAN,
    discriminator: PatchDiscriminator,
    loader: DataLoader,
    optimizer_vqgan: torch.optim.Optimizer,
    optimizer_disc: torch.optim.Optimizer,
    scaler_vqgan,
    scaler_disc,
    device: torch.device,
    epoch: int,
    corr_weight: float,
    adv_weight: float,
    discriminator_start_epoch: int,
    gradient_accumulation_steps: int,
    amp_dtype: str,
    log_every: int,
    max_batches: int | None,
) -> dict[str, float]:
    model.train()
    discriminator.train()

    adversarial_active = epoch >= discriminator_start_epoch
    total_batches = len(loader)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)
    if total_batches <= 0:
        raise RuntimeError("No training batch is available")

    totals = {
        "generator": 0.0,
        "discriminator": 0.0,
        "reconstruction": 0.0,
        "vq": 0.0,
        "correlation": 0.0,
        "adversarial": 0.0,
    }
    total_examples = 0
    optimizer_updates = 0
    pending = 0
    group_size = 0

    for batch_index, batch in enumerate(loader):
        if batch_index >= total_batches:
            break

        if pending == 0:
            group_size = min(
                gradient_accumulation_steps,
                total_batches - batch_index,
            )
            optimizer_vqgan.zero_grad(set_to_none=True)
            optimizer_disc.zero_grad(set_to_none=True)

        eeg = batch["eeg_spectrogram"].unsqueeze(1).to(
            device,
            non_blocking=True,
        )
        batch_size = eeg.shape[0]

        if adversarial_active:
            set_requires_grad(discriminator, True)
            with torch.no_grad(), amp_context(device, amp_dtype):
                fake_eeg, _, _ = model(eeg)
            with amp_context(device, amp_dtype):
                discriminator_loss = discriminator_hinge_loss(
                    discriminator(eeg),
                    discriminator(fake_eeg.detach()),
                )
            scaler_disc.scale(discriminator_loss / group_size).backward()
        else:
            discriminator_loss = eeg.new_zeros(())

        set_requires_grad(discriminator, False)
        with amp_context(device, amp_dtype):
            _, generator_losses, _ = compute_generator_losses(
                model=model,
                discriminator=discriminator,
                eeg=eeg,
                corr_weight=corr_weight,
                adv_weight=adv_weight,
                adversarial_active=adversarial_active,
            )
        scaler_vqgan.scale(
            generator_losses["generator"] / group_size
        ).backward()
        set_requires_grad(discriminator, True)

        pending += 1
        if pending == group_size:
            if adversarial_active:
                scaler_disc.step(optimizer_disc)
                scaler_disc.update()
            scaler_vqgan.step(optimizer_vqgan)
            scaler_vqgan.update()
            optimizer_updates += 1
            pending = 0

        totals["discriminator"] += float(discriminator_loss.detach()) * batch_size
        for name, value in generator_losses.items():
            totals[name] += float(value.detach()) * batch_size
        total_examples += batch_size

        if log_every > 0 and (batch_index + 1) % log_every == 0:
            print(
                f"  train batch {batch_index + 1}/{total_batches} | "
                f"recon={totals['reconstruction'] / total_examples:.4f} | "
                f"corr={totals['correlation'] / total_examples:.4f} | "
                f"vq={totals['vq'] / total_examples:.4f}",
                flush=True,
            )

    metrics = {name: value / total_examples for name, value in totals.items()}
    metrics["examples"] = float(total_examples)
    metrics["batches"] = float(total_batches)
    metrics["optimizer_updates"] = float(optimizer_updates)
    metrics["adversarial_active"] = float(adversarial_active)
    return metrics


@torch.no_grad()
def validate_epoch(
    model: VQGAN,
    loader: DataLoader,
    device: torch.device,
    corr_weight: float,
    amp_dtype: str,
    max_batches: int | None,
) -> dict[str, float]:
    model.eval()
    total_batches = len(loader)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)
    if total_batches <= 0:
        raise RuntimeError("No validation batch is available")

    totals = {
        "loss": 0.0,
        "reconstruction": 0.0,
        "vq": 0.0,
        "correlation": 0.0,
    }
    used_codes = torch.zeros(8192, dtype=torch.bool)
    total_examples = 0

    for batch_index, batch in enumerate(loader):
        if batch_index >= total_batches:
            break
        eeg = batch["eeg_spectrogram"].unsqueeze(1).to(
            device,
            non_blocking=True,
        )
        with amp_context(device, amp_dtype):
            reconstruction, indices, vq_loss = model(eeg)
            reconstruction_loss = F.l1_loss(reconstruction, eeg)
            corr_loss = correlation_loss(reconstruction, eeg)
            loss = reconstruction_loss + vq_loss + corr_weight * corr_loss

        batch_size = eeg.shape[0]
        totals["loss"] += float(loss) * batch_size
        totals["reconstruction"] += float(reconstruction_loss) * batch_size
        totals["vq"] += float(vq_loss) * batch_size
        totals["correlation"] += float(corr_loss) * batch_size
        total_examples += batch_size
        used_codes[torch.unique(indices.detach().cpu()).long()] = True

    metrics = {name: value / total_examples for name, value in totals.items()}
    metrics["unique_codes"] = float(used_codes.sum())
    metrics["examples"] = float(total_examples)
    metrics["batches"] = float(total_batches)
    return metrics


@torch.no_grad()
def save_reconstruction(
    model: VQGAN,
    dataset: ManifestEEGDataset,
    device: torch.device,
    output_path: Path,
) -> None:
    model.eval()
    sample = dataset[0]
    eeg = sample["eeg_spectrogram"].unsqueeze(0).unsqueeze(0).to(device)
    reconstruction, indices, _ = model(eeg)

    target = eeg[0, 0].float().cpu().numpy()
    prediction = reconstruction[0, 0].float().cpu().numpy()
    difference = np.abs(target - prediction)

    figure, axes = plt.subplots(3, 1, figsize=(12, 8))
    axes[0].imshow(target, aspect="auto", origin="lower", vmin=0, vmax=1)
    axes[0].set_title("Ground-truth EEG spectrogram")
    axes[1].imshow(
        prediction,
        aspect="auto",
        origin="lower",
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Multi-dataset VQGAN reconstruction")
    axes[2].imshow(difference, aspect="auto", origin="lower")
    axes[2].set_title("Absolute difference")
    for axis in axes:
        axis.set_xlabel("Time (30-second epochs)")
        axis.set_ylabel("Frequency bins")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(figure)

    print(f"Token grid: {tuple(indices.shape)}")
    print(f"Unique codes in example: {torch.unique(indices).numel()}")
    print(f"Reconstruction saved: {output_path}")


def load_normalization(path: Path) -> tuple[dict, float, float]:
    if not path.exists():
        raise FileNotFoundError(f"Normalization file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        normalization = json.load(handle)
    min_db = float(normalization["min_db"])
    max_db = float(normalization["max_db"])
    if max_db <= min_db:
        raise ValueError(f"Invalid dB bounds in {path}: {min_db}, {max_db}")
    return normalization, min_db, max_db


def checkpoint_payload(
    epoch: int,
    args: argparse.Namespace,
    min_db: float,
    max_db: float,
    model: VQGAN,
    discriminator: PatchDiscriminator,
    optimizer_vqgan: torch.optim.Optimizer,
    optimizer_disc: torch.optim.Optimizer,
    scaler_vqgan,
    scaler_disc,
    train_metrics: dict[str, float],
    validation_metrics: dict[str, float],
    best_validation_loss: float,
) -> dict:
    return {
        "epoch": epoch,
        "architecture": "vqgan_multidataset_paper_reproduction",
        "min_db": min_db,
        "max_db": max_db,
        "train_manifest": str(args.train_manifest),
        "val_manifest": str(args.val_manifest),
        "corr_weight": args.corr_weight,
        "adv_weight": args.adv_weight,
        "discriminator_start_epoch": args.discriminator_start_epoch,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": (
            args.batch_size * args.gradient_accumulation_steps
        ),
        "learning_rate": args.lr,
        "discriminator_learning_rate": args.disc_lr,
        "amp_dtype": args.amp_dtype,
        "model_state_dict": model.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizer_vqgan_state_dict": optimizer_vqgan.state_dict(),
        "optimizer_disc_state_dict": optimizer_disc.state_dict(),
        "scaler_vqgan_state_dict": scaler_vqgan.state_dict(),
        "scaler_disc_state_dict": scaler_disc.state_dict(),
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "best_validation_loss": best_validation_loss,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a shared VQGAN from multi-dataset manifests."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("outputs/vqgan_multidataset_preprocessed"),
    )
    parser.add_argument("--train-manifest", type=Path, default=None)
    parser.add_argument("--val-manifest", type=Path, default=None)
    parser.add_argument("--normalization-file", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/vqgan_multidataset"),
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
    )
    parser.add_argument("--lr", type=float, default=4.8e-5)
    parser.add_argument("--disc-lr", type=float, default=4.8e-5)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--corr-weight", type=float, default=0.1)
    parser.add_argument("--adv-weight", type=float, default=0.01)
    parser.add_argument(
        "--discriminator-start-epoch",
        type=int,
        default=1,
        help="First epoch using adversarial training (paper does not specify).",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--amp-dtype",
        choices=["bfloat16", "float16", "none"],
        default="bfloat16",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    args = parser.parse_args()

    if args.batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("Batch size and accumulation steps must be positive")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")

    if args.train_manifest is None:
        args.train_manifest = args.data_root / "train_manifest.jsonl"
    if args.val_manifest is None:
        args.val_manifest = args.data_root / "val_manifest.jsonl"
    if args.normalization_file is None:
        args.normalization_file = args.data_root / "normalization.json"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    normalization, min_db, max_db = load_normalization(
        args.normalization_file
    )
    train_dataset = ManifestEEGDataset(args.train_manifest, min_db, max_db)
    validation_dataset = ManifestEEGDataset(args.val_manifest, min_db, max_db)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )

    model = VQGAN().to(device)
    discriminator = PatchDiscriminator().to(device)
    optimizer_vqgan = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
    )
    optimizer_disc = torch.optim.Adam(
        discriminator.parameters(),
        lr=args.disc_lr,
        betas=(args.beta1, args.beta2),
    )
    fp16_scaling = device.type == "cuda" and args.amp_dtype == "float16"
    scaler_vqgan = build_scaler(fp16_scaling)
    scaler_disc = build_scaler(fp16_scaling)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = args.output_dir / "checkpoint_latest.pt"
    best_path = args.output_dir / "checkpoint_best.pt"

    start_epoch = 1
    best_validation_loss = float("inf")
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        discriminator.load_state_dict(
            checkpoint["discriminator_state_dict"],
            strict=True,
        )
        optimizer_vqgan.load_state_dict(
            checkpoint["optimizer_vqgan_state_dict"]
        )
        optimizer_disc.load_state_dict(
            checkpoint["optimizer_disc_state_dict"]
        )
        if "scaler_vqgan_state_dict" in checkpoint:
            scaler_vqgan.load_state_dict(checkpoint["scaler_vqgan_state_dict"])
        if "scaler_disc_state_dict" in checkpoint:
            scaler_disc.load_state_dict(checkpoint["scaler_disc_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation_loss = float(
            checkpoint.get("best_validation_loss", float("inf"))
        )

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")
    print(f"Training datasets: {train_dataset.dataset_counts()}")
    print(f"Validation datasets: {validation_dataset.dataset_counts()}")
    print(f"Normalization: [{min_db:.4f}, {max_db:.4f}] dB")
    print(f"Physical batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(
        "Effective batch size: "
        f"{args.batch_size * args.gradient_accumulation_steps}"
    )
    print(f"AMP dtype: {args.amp_dtype}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Correlation weight (reproduction choice): {args.corr_weight}")
    print(f"Adversarial weight (reproduction choice): {args.adv_weight}")
    print(f"Discriminator starts at epoch: {args.discriminator_start_epoch}")
    print(
        "VQGAN parameters: "
        f"{sum(parameter.numel() for parameter in model.parameters()):,}"
    )

    # Persist the exact data/training specification separately from checkpoints.
    run_config = vars(args).copy()
    run_config.update(
        {
            "data_root": str(args.data_root),
            "train_manifest": str(args.train_manifest),
            "val_manifest": str(args.val_manifest),
            "normalization_file": str(args.normalization_file),
            "output_dir": str(args.output_dir),
            "resume": None if args.resume is None else str(args.resume),
            "min_db": min_db,
            "max_db": max_db,
            "normalization": normalization,
        }
    )
    with (args.output_dir / "run_config.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(run_config, handle, indent=2)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.monotonic()
        train_metrics = train_epoch(
            model=model,
            discriminator=discriminator,
            loader=train_loader,
            optimizer_vqgan=optimizer_vqgan,
            optimizer_disc=optimizer_disc,
            scaler_vqgan=scaler_vqgan,
            scaler_disc=scaler_disc,
            device=device,
            epoch=epoch,
            corr_weight=args.corr_weight,
            adv_weight=args.adv_weight,
            discriminator_start_epoch=args.discriminator_start_epoch,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            amp_dtype=args.amp_dtype,
            log_every=args.log_every,
            max_batches=args.max_train_batches,
        )
        validation_metrics = validate_epoch(
            model=model,
            loader=validation_loader,
            device=device,
            corr_weight=args.corr_weight,
            amp_dtype=args.amp_dtype,
            max_batches=args.max_val_batches,
        )
        elapsed_minutes = (time.monotonic() - epoch_start) / 60.0

        improved = validation_metrics["loss"] < best_validation_loss
        if improved:
            best_validation_loss = validation_metrics["loss"]

        payload = checkpoint_payload(
            epoch=epoch,
            args=args,
            min_db=min_db,
            max_db=max_db,
            model=model,
            discriminator=discriminator,
            optimizer_vqgan=optimizer_vqgan,
            optimizer_disc=optimizer_disc,
            scaler_vqgan=scaler_vqgan,
            scaler_disc=scaler_disc,
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            best_validation_loss=best_validation_loss,
        )
        torch.save(payload, latest_path)
        if improved:
            torch.save(payload, best_path)

        print(
            f"Epoch {epoch:03d} | "
            f"G={train_metrics['generator']:.4f} | "
            f"D={train_metrics['discriminator']:.4f} | "
            f"train_MAE={train_metrics['reconstruction']:.4f} | "
            f"val={validation_metrics['loss']:.4f} | "
            f"val_MAE={validation_metrics['reconstruction']:.4f} | "
            f"val_corr_loss={validation_metrics['correlation']:.4f} | "
            f"val_codes={int(validation_metrics['unique_codes'])} | "
            f"minutes={elapsed_minutes:.1f}",
            flush=True,
        )

    if not best_path.exists():
        raise RuntimeError("No best checkpoint was produced")
    best_checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
    save_reconstruction(
        model=model,
        dataset=validation_dataset,
        device=device,
        output_path=args.output_dir / "reconstruction_best.png",
    )
    print(f"Best epoch: {best_checkpoint['epoch']}")
    print(f"Best validation loss: {best_validation_loss:.4f}")


if __name__ == "__main__":
    main()
