"""Train the paper-oriented respiration-to-EEG masked Transformer.

The model architecture and principal optimization values follow the
Physiology-as-Language supplement.  Dataset selection, SHHS normalization and
window balancing are kept identical to the current reproduction pipeline so
architectural comparisons remain controlled.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_architecture import PaperMaskedRespirationToEEGTransformer
from src.data.dataset import PhysiologyPairDataset
from src.models.vqgan import VQGAN


DEFAULT_DATA_ROOT = Path("outputs/shhs_preprocessed")
DEFAULT_VQGAN_CHECKPOINT = Path(
    "outputs/vqgan_adversarial_w0001/held_out_fold_0/checkpoint_best.pt"
)
DEFAULT_OUTPUT_DIR = Path("outputs/paper_masked_transformer")

MODEL_WINDOW_SEC = 256 * 60
SUPPORTED_WINDOW_INDICES = (0, 1)
ARCHITECTURE_NAME = "paper_masked_transformer_mage_reproduction"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model_state_dict(checkpoint: dict) -> dict:
    for key in (
        "model_state_dict",
        "transformer_state_dict",
        "vqgan_state_dict",
        "generator_state_dict",
    ):
        if key in checkpoint:
            return checkpoint[key]
    raise KeyError(
        "No model state dictionary found. "
        f"Available keys: {list(checkpoint.keys())}"
    )


def load_frozen_vqgan(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[VQGAN, dict]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"VQGAN checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model = VQGAN()
    model.load_state_dict(get_model_state_dict(checkpoint), strict=True)
    model = model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, checkpoint


def create_datasets(
    data_root: Path,
    held_out_fold: int,
    min_db: float,
    max_db: float,
    number_of_folds: int = 4,
):
    train_folds = [
        fold for fold in range(number_of_folds) if fold != held_out_fold
    ]
    train_directories = [data_root / f"fold_{fold}" for fold in train_folds]
    validation_directory = data_root / f"fold_{held_out_fold}"

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
    return train_dataset, validation_dataset, train_folds


def filter_supported_windows(
    dataset: PhysiologyPairDataset,
    supported_windows=SUPPORTED_WINDOW_INDICES,
):
    """Keep the same window000/window001 protocol used for V5/V6."""
    selected_indices = []
    selected_window_indices = []
    counts = {int(window_index): 0 for window_index in supported_windows}

    for dataset_index, file_path in enumerate(dataset.files):
        with np.load(file_path) as sample:
            if "start_sec" not in sample:
                raise KeyError(f"start_sec is required: {file_path}")
            start_sec = int(
                round(float(np.asarray(sample["start_sec"]).reshape(-1)[0]))
            )

        window_index = start_sec // MODEL_WINDOW_SEC
        if window_index not in supported_windows:
            continue
        selected_indices.append(dataset_index)
        selected_window_indices.append(int(window_index))
        counts[int(window_index)] += 1

    for window_index in supported_windows:
        if counts[int(window_index)] < 1:
            raise RuntimeError(f"No samples found for window {window_index}")

    return Subset(dataset, selected_indices), selected_window_indices, counts


def create_balanced_window_sampler(window_indices: list[int], seed: int):
    counts = {
        window_index: window_indices.count(window_index)
        for window_index in SUPPORTED_WINDOW_INDICES
    }
    weights = torch.tensor(
        [1.0 / counts[window_index] for window_index in window_indices],
        dtype=torch.double,
    )
    samples_per_epoch = 2 * max(counts.values())
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=samples_per_epoch,
        replacement=True,
        generator=generator,
    )
    return sampler, samples_per_epoch


def standardize_respiration(respiration: Tensor, eps: float = 1e-6) -> Tensor:
    """Normalize once over the complete 256-minute respiration window."""
    mean = respiration.mean(dim=(1, 2), keepdim=True)
    std = respiration.std(dim=(1, 2), keepdim=True, unbiased=False)
    return (respiration - mean) / (std + eps)


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int,
):
    """Paper schedule: linear warm-up followed by cosine decay."""
    if not 0 <= warmup_epochs < epochs:
        raise ValueError("warmup_epochs must be in [0, epochs)")

    def multiplier(epoch_index: int) -> float:
        if warmup_epochs > 0 and epoch_index < warmup_epochs:
            return (epoch_index + 1) / warmup_epochs
        decay_epochs = max(epochs - warmup_epochs, 1)
        progress = (epoch_index - warmup_epochs) / decay_epochs
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)


def run_epoch(
    model: PaperMaskedRespirationToEEGTransformer,
    vqgan: VQGAN,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler=None,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    max_batches: int | None = None,
    log_every: int = 500,
) -> dict[str, float | int]:
    training = optimizer is not None
    model.train(training)
    vqgan.eval()

    if training:
        optimizer.zero_grad(set_to_none=True)

    batches_to_run = len(loader)
    if max_batches is not None:
        batches_to_run = min(batches_to_run, max_batches)

    total_loss = 0.0
    total_correct = 0
    total_top5 = 0
    total_positions = 0
    total_mask_ratio = 0.0
    processed_batches = 0
    successful_updates = 0
    skipped_updates = 0
    predicted_codes_seen = torch.zeros(
        model.codebook_size,
        dtype=torch.bool,
        device=device,
    )
    target_codes_seen = torch.zeros_like(predicted_codes_seen)

    for batch_index, batch in enumerate(loader, start=1):
        if batch_index > batches_to_run:
            break

        respiration = batch["respiration"].to(device, non_blocking=True)
        respiration = standardize_respiration(respiration)
        eeg = (
            batch["eeg_spectrogram"]
            .unsqueeze(1)
            .to(device, non_blocking=True)
        )

        # The pretrained VQGAN is frozen and supplies only target token IDs.
        with torch.no_grad():
            _, eeg_tokens, _ = vqgan.encode(eeg)

        # Paper training: variable truncated-Gaussian masking.
        # Paper inference/validation: every EEG token is masked.
        mask_ratio = None if training else 1.0

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                outputs = model(
                    respiration=respiration,
                    eeg_tokens=eeg_tokens,
                    mask_ratio=mask_ratio,
                )
                loss = outputs["loss"]

        if not torch.isfinite(loss):
            raise ValueError("Non-finite loss detected")

        if training:
            if scaler is None:
                raise ValueError("A GradScaler is required during training")
            scaler.scale(loss / gradient_accumulation_steps).backward()

            should_update = (
                batch_index % gradient_accumulation_steps == 0
                or batch_index == batches_to_run
            )
            if should_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=max_grad_norm,
                )
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()
                if scale_after < scale_before:
                    skipped_updates += 1
                else:
                    successful_updates += 1
                optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            logits = outputs["logits"]
            targets = outputs["targets"]
            mask = outputs["mask"]
            masked_logits = logits[mask]
            masked_targets = targets[mask]

            predictions = masked_logits.argmax(dim=-1)
            top5_predictions = masked_logits.topk(k=5, dim=-1).indices
            positions = masked_targets.numel()

            total_loss += float(loss.item()) * positions
            total_correct += int((predictions == masked_targets).sum().item())
            total_top5 += int(
                (top5_predictions == masked_targets.unsqueeze(-1))
                .any(dim=-1)
                .sum()
                .item()
            )
            total_positions += positions
            total_mask_ratio += float(outputs["mask_ratio"])
            processed_batches += 1
            predicted_codes_seen[predictions.unique()] = True
            target_codes_seen[masked_targets.unique()] = True

        if log_every > 0 and batch_index % log_every == 0:
            mode = "train" if training else "validation"
            print(
                f"  {mode} batch {batch_index}"
                f" | CE={total_loss / total_positions:.4f}"
                f" | acc={total_correct / total_positions:.4f}"
                f" | mask={total_mask_ratio / processed_batches:.4f}",
                flush=True,
            )

    if processed_batches == 0:
        raise RuntimeError("No batch was processed")
    if training and successful_updates == 0:
        raise RuntimeError(
            "Every optimizer update was skipped by AMP; gradients are non-finite"
        )

    return {
        "loss": total_loss / total_positions,
        "cross_entropy_loss": total_loss / total_positions,
        "accuracy": total_correct / total_positions,
        "top5_accuracy": total_top5 / total_positions,
        "mask_ratio": total_mask_ratio / processed_batches,
        "unique_predicted_codes": int(predicted_codes_seen.sum().item()),
        "unique_target_codes": int(target_codes_seen.sum().item()),
        "positions": total_positions,
        "batches": processed_batches,
        "optimizer_updates": successful_updates,
        "skipped_optimizer_updates": skipped_updates,
    }


def load_resume_checkpoint(
    checkpoint_path: Path,
    model: PaperMaskedRespirationToEEGTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    held_out_fold: int,
) -> tuple[int, float]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("architecture") != ARCHITECTURE_NAME:
        raise ValueError("Resume checkpoint has an incompatible architecture")
    if int(checkpoint.get("held_out_fold")) != held_out_fold:
        raise ValueError("Resume checkpoint uses a different held-out fold")
    if checkpoint.get("model_config") != model.get_config():
        raise ValueError("Resume checkpoint model configuration does not match")

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return int(checkpoint["epoch"]), float(checkpoint["best_validation_loss"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the paper masked respiration-to-EEG Transformer"
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--vqgan-checkpoint", type=Path, default=DEFAULT_VQGAN_CHECKPOINT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--held-out-fold", type=int, choices=[0, 1, 2, 3], required=True
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=192)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.125e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--overfit-samples", type=int, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient-accumulation-steps must be positive")
    if args.warmup_epochs >= args.epochs:
        raise ValueError("warmup-epochs must be smaller than epochs")
    if args.overfit_samples is not None and args.overfit_samples < 1:
        raise ValueError("overfit-samples must be positive")

    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_directory = args.output_dir / f"held_out_fold_{args.held_out_fold}"
    output_directory.mkdir(parents=True, exist_ok=True)

    vqgan, vqgan_checkpoint = load_frozen_vqgan(
        args.vqgan_checkpoint,
        device,
    )
    min_db = float(vqgan_checkpoint["min_db"])
    max_db = float(vqgan_checkpoint["max_db"])

    train_dataset, validation_dataset, train_folds = create_datasets(
        data_root=args.data_root,
        held_out_fold=args.held_out_fold,
        min_db=min_db,
        max_db=max_db,
    )
    train_dataset, train_window_indices, train_window_counts = (
        filter_supported_windows(train_dataset)
    )
    validation_dataset, _, validation_window_counts = filter_supported_windows(
        validation_dataset
    )

    train_sampler = None
    samples_per_training_epoch = len(train_dataset)
    if args.overfit_samples is not None:
        number_of_samples = min(args.overfit_samples, len(train_dataset))
        same_subset = Subset(train_dataset, list(range(number_of_samples)))
        train_dataset = same_subset
        validation_dataset = same_subset
        samples_per_training_epoch = number_of_samples
        print(
            "Overfit diagnostic:",
            number_of_samples,
            "identical training/validation samples",
        )
    else:
        train_sampler, samples_per_training_epoch = create_balanced_window_sampler(
            train_window_indices,
            args.seed,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    codebook_size = int(vqgan.quantizer.codebook.weight.shape[0])
    model = PaperMaskedRespirationToEEGTransformer(
        codebook_size=codebook_size
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = create_scheduler(
        optimizer,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda"),
    )

    start_epoch = 0
    best_validation_loss = float("inf")
    if args.resume_checkpoint is not None:
        start_epoch, best_validation_loss = load_resume_checkpoint(
            checkpoint_path=args.resume_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            held_out_fold=args.held_out_fold,
        )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps

    print("Device:", device)
    print("Architecture:", ARCHITECTURE_NAME)
    print("Training folds:", train_folds)
    print("Held-out fold:", args.held_out_fold)
    print("Training samples:", len(train_dataset))
    print("Samples per training epoch:", samples_per_training_epoch)
    print("Training windows:", train_window_counts)
    print("Validation samples:", len(validation_dataset))
    print("Validation windows:", validation_window_counts)
    print("Normalization:", f"[{min_db:.2f}, {max_db:.2f}] dB")
    print("Transformer parameters:", f"{trainable_parameters:,}")
    print("Encoder/decoder blocks: 8 / 8")
    print("Embedding/heads: 768 / 8")
    print("Physical batch size:", args.batch_size)
    print("Gradient accumulation:", args.gradient_accumulation_steps)
    print("Effective batch size:", effective_batch_size)
    print("Loss: masked-token cross-entropy only")
    print("Epochs / warmup:", args.epochs, "/", args.warmup_epochs)
    print("Peak learning rate:", args.lr)
    print("Weight decay:", args.weight_decay)
    print(
        "VQGAN trainable parameters:",
        sum(parameter.requires_grad for parameter in vqgan.parameters()),
    )

    for epoch_index in range(start_epoch, args.epochs):
        epoch_number = epoch_index + 1
        learning_rate = optimizer.param_groups[0]["lr"]

        train_metrics = run_epoch(
            model=model,
            vqgan=vqgan,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            max_batches=args.max_train_batches,
            log_every=args.log_every,
        )
        validation_metrics = run_epoch(
            model=model,
            vqgan=vqgan,
            loader=validation_loader,
            device=device,
            max_batches=args.max_val_batches,
            log_every=args.log_every,
        )

        is_best = validation_metrics["loss"] < best_validation_loss
        if is_best:
            best_validation_loss = float(validation_metrics["loss"])

        print(
            f"Epoch {epoch_number:03d}"
            f" | lr={learning_rate:.2e}"
            f" | train_CE={train_metrics['loss']:.4f}"
            f" | train_acc={train_metrics['accuracy']:.4f}"
            f" | train_top5={train_metrics['top5_accuracy']:.4f}"
            f" | train_mask={train_metrics['mask_ratio']:.4f}"
            f" | train_codes={train_metrics['unique_predicted_codes']}"
            f" | val_CE={validation_metrics['loss']:.4f}"
            f" | val_acc={validation_metrics['accuracy']:.4f}"
            f" | val_top5={validation_metrics['top5_accuracy']:.4f}"
            f" | val_codes={validation_metrics['unique_predicted_codes']}",
            flush=True,
        )

        # Save the scheduler state prepared for the following epoch so a
        # resumed run does not repeat the previous epoch's learning rate.
        scheduler.step()

        checkpoint = {
            "version": 1,
            "architecture": ARCHITECTURE_NAME,
            "model_config": model.get_config(),
            "epoch": epoch_number,
            "held_out_fold": args.held_out_fold,
            "train_folds": train_folds,
            "min_db": min_db,
            "max_db": max_db,
            "learning_rate": learning_rate,
            "peak_learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_epochs": args.warmup_epochs,
            "physical_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": effective_batch_size,
            "max_grad_norm": args.max_grad_norm,
            "vqgan_checkpoint": str(args.vqgan_checkpoint),
            "supported_window_indices": list(SUPPORTED_WINDOW_INDICES),
            "train_window_counts": train_window_counts,
            "validation_window_counts": validation_window_counts,
            "balanced_samples_per_epoch": samples_per_training_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "best_validation_loss": best_validation_loss,
        }
        torch.save(checkpoint, output_directory / "checkpoint_latest.pt")
        if is_best:
            torch.save(checkpoint, output_directory / "checkpoint_best.pt")

    print("Best validation loss:", f"{best_validation_loss:.4f}")


if __name__ == "__main__":
    main()
