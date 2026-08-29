import argparse
import math
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.data.dataset import PhysiologyPairDataset
from src.models.mage_transformer import (
    MAGERespirationToEEGTransformer,
)

from src.models.masked_transformer import (
    mask_eeg_tokens,
)
from src.models.vqgan import VQGAN


DEFAULT_DATA_ROOT = Path(
    "outputs/shhs_preprocessed"
)

DEFAULT_VQGAN_CHECKPOINT = Path(
    "outputs/vqgan_adversarial_w0001/"
    "held_out_fold_0/checkpoint_best.pt"
)

DEFAULT_OUTPUT_DIR = Path(
    "outputs/masked_transformer_joint"
)


def set_seed(seed):
    """
    Set random seeds for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model_state_dict(checkpoint):
    """
    Retrieve model weights from a checkpoint.
    """

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
    checkpoint_path,
    device,
):
    """
    Load and freeze the trained EEG VQGAN.
    """

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "VQGAN checkpoint not found: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model = VQGAN()

    model.load_state_dict(
        get_model_state_dict(checkpoint),
        strict=True,
    )

    model = model.to(device)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    return model, checkpoint


def load_initial_transformer(
    model,
    checkpoint_path,
    held_out_fold,
):
    """
    Initialize the Transformer from an existing checkpoint.

    Only model weights are restored. The optimizer and scheduler
    are deliberately reinitialized for fine-tuning.
    """

    if checkpoint_path is None:
        return None

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Initial Transformer checkpoint not found: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    architecture = checkpoint.get(
        "architecture"
    )

    if architecture not in (
        None,
        "mage_visible_encoder_decoder",
    ):
        raise ValueError(
            "Incompatible checkpoint architecture: "
            f"{architecture}"
        )

    checkpoint_fold = checkpoint.get(
        "held_out_fold"
    )

    if (
        checkpoint_fold is not None
        and int(checkpoint_fold)
        != held_out_fold
    ):
        raise ValueError(
            "Checkpoint held-out fold does not "
            "match the requested held-out fold"
        )

    model.load_state_dict(
        get_model_state_dict(checkpoint),
        strict=True,
    )

    return checkpoint


def create_datasets(
    data_root,
    held_out_fold,
    min_db,
    max_db,
    number_of_folds=4,
):
    """
    Create patient-wise training and validation datasets.
    """

    train_folds = [
        fold
        for fold in range(number_of_folds)
        if fold != held_out_fold
    ]

    train_directories = [
        data_root / f"fold_{fold}"
        for fold in train_folds
    ]

    validation_directory = (
        data_root
        / f"fold_{held_out_fold}"
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
    )


def standardize_respiration(
    respiration,
    eps=1e-6,
):
    """
    Normalize each complete 256-minute respiration window.

    A single mean and standard deviation are calculated over
    all 64 four-minute segments, preserving relative amplitude
    differences between segments.
    """

    mean = respiration.mean(
        dim=(1, 2),
        keepdim=True,
    )

    std = respiration.std(
        dim=(1, 2),
        keepdim=True,
        unbiased=False,
    )

    return (
        respiration - mean
    ) / (std + eps)


def create_scheduler(
    optimizer,
    epochs,
    warmup_epochs,
):
    """
    Linear warm-up followed by cosine learning-rate decay.
    """

    if warmup_epochs < 0:
        raise ValueError(
            "warmup_epochs cannot be negative"
        )

    if warmup_epochs >= epochs:
        raise ValueError(
            "warmup_epochs must be smaller than epochs"
        )

    def learning_rate_multiplier(epoch_index):
        if (
            warmup_epochs > 0
            and epoch_index < warmup_epochs
        ):
            return (
                epoch_index + 1
            ) / warmup_epochs

        decay_epochs = max(
            epochs - warmup_epochs,
            1,
        )

        progress = (
            epoch_index - warmup_epochs
        ) / decay_epochs

        progress = min(
            max(progress, 0.0),
            1.0,
        )

        return 0.5 * (
            1.0
            + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=learning_rate_multiplier,
    )


def run_epoch(
    model,
    vqgan,
    loader,
    device,
    optimizer=None,
    scaler=None,
    gradient_accumulation_steps=1,
    full_mask_probability=0.5,
    max_batches=None,
    log_every=500,
):
    """
    Run one training or validation epoch.

    Training:
        A configurable fraction of samples is fully masked.
        The remaining samples use the paper-inspired masking
        distribution.

    Validation:
        Every EEG token is masked because inference starts
        entirely from respiration.
    """

    training = optimizer is not None

    model.train(training)
    vqgan.eval()

    if training:
        optimizer.zero_grad(
            set_to_none=True
        )

    batches_to_run = len(loader)

    if max_batches is not None:
        batches_to_run = min(
            batches_to_run,
            max_batches,
        )

    total_loss = 0.0
    total_correct = 0
    total_top5 = 0
    total_positions = 0
    total_mask_ratio = 0.0
    processed_batches = 0

    for batch_index, batch in enumerate(
        loader,
        start=1,
    ):
        if batch_index > batches_to_run:
            break

        respiration = (
            batch["respiration"]
            .to(
                device,
                non_blocking=True,
            )
        )

        eeg = (
            batch["eeg_spectrogram"]
            .unsqueeze(1)
            .to(
                device,
                non_blocking=True,
            )
        )

        respiration = standardize_respiration(
            respiration
        )

        # The VQGAN is frozen and only creates target tokens.
        with torch.no_grad():
            _, eeg_tokens, _ = vqgan.encode(
                eeg
            )

        if training:
            (
                masked_tokens,
                mask,
                mask_ratios,
            ) = mask_eeg_tokens(
                eeg_tokens,
                mask_token_id=(
                    model.mask_token_id
                ),
                full_mask_probability=(
                    full_mask_probability
                ),
            )

        else:
            # Inference condition: every EEG token is hidden.
            masked_tokens = torch.full_like(
                eeg_tokens,
                fill_value=model.mask_token_id,
            )

            mask = torch.ones_like(
                eeg_tokens,
                dtype=torch.bool,
            )

            mask_ratios = torch.ones(
                eeg_tokens.shape[0],
                device=device,
            )

        targets_flat = eeg_tokens.flatten(
            start_dim=1
        )

        mask_flat = mask.flatten(
            start_dim=1
        )

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                logits = model(
                    respiration,
                    masked_tokens,
                )

                masked_logits = logits[
                    mask_flat
                ]

                masked_targets = targets_flat[
                    mask_flat
                ]

                loss = F.cross_entropy(
                    masked_logits,
                    masked_targets,
                )

        if not torch.isfinite(loss):
            raise ValueError(
                "Non-finite training loss detected"
            )

        if training:
            if scaler is None:
                raise ValueError(
                    "A GradScaler is required "
                    "during training"
                )

            scaled_loss = (
                loss
                / gradient_accumulation_steps
            )

            scaler.scale(
                scaled_loss
            ).backward()

            should_update = (
                batch_index
                % gradient_accumulation_steps
                == 0
                or batch_index
                == batches_to_run
            )

            if should_update:
                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad(
                    set_to_none=True
                )

        with torch.no_grad():
            predictions = (
                masked_logits.argmax(
                    dim=-1
                )
            )

            top5_predictions = (
                masked_logits.topk(
                    k=5,
                    dim=-1,
                ).indices
            )

            correct = (
                predictions
                == masked_targets
            ).sum()

            top5_correct = (
                top5_predictions
                == masked_targets.unsqueeze(-1)
            ).any(dim=-1).sum()

            number_of_positions = (
                masked_targets.numel()
            )

            total_loss += (
                loss.item()
                * number_of_positions
            )

            total_correct += correct.item()
            total_top5 += top5_correct.item()
            total_positions += number_of_positions

            total_mask_ratio += (
                mask_ratios.mean().item()
            )

            processed_batches += 1

        if (
            log_every > 0
            and batch_index % log_every == 0
        ):
            mode = (
                "train"
                if training
                else "validation"
            )

            running_loss = (
                total_loss
                / total_positions
            )

            print(
                f"  {mode} batch {batch_index}"
                f" | loss={running_loss:.4f}",
                flush=True,
            )

    if processed_batches == 0:
        raise RuntimeError(
            "No batch was processed"
        )

    return {
        "loss": (
            total_loss
            / total_positions
        ),
        "accuracy": (
            total_correct
            / total_positions
        ),
        "top5_accuracy": (
            total_top5
            / total_positions
        ),
        "mask_ratio": (
            total_mask_ratio
            / processed_batches
        ),
        "positions": total_positions,
        "batches": processed_batches,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train the joint respiration-to-EEG "
            "Masked Transformer."
        )
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
    )

    parser.add_argument(
        "--vqgan-checkpoint",
        type=Path,
        default=DEFAULT_VQGAN_CHECKPOINT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--held-out-fold",
        type=int,
        choices=[0, 1, 2, 3],
        required=True,
    )
    parser.add_argument(
    "--overfit-samples",
    type=int,
    default=None,
    help=(
        "Diagnostic: train and validate on the same "
        "small fixed subset."
        ),
    )

    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional Transformer checkpoint used "
            "to initialize model weights. Optimizer "
            "state is not restored."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1.125e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--full-mask-probability",
        type=float,
        default=0.00,
        help=(
            "Probability that all EEG tokens of a "
            "training example are masked."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
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

    parser.add_argument(
        "--log-every",
        type=int,
        default=500,
    )

    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError(
            "epochs must be at least 1"
        )

    if args.batch_size < 1:
        raise ValueError(
            "batch_size must be at least 1"
        )

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "gradient-accumulation-steps "
            "must be at least 1"
        )

    if not (
        0.0
        <= args.full_mask_probability
        <= 1.0
    ):
        raise ValueError(
            "full-mask-probability must be "
            "between 0 and 1"
        )

    set_seed(args.seed)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision(
            "high"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    output_directory = (
        args.output_dir
        / f"held_out_fold_{args.held_out_fold}"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        vqgan,
        vqgan_checkpoint,
    ) = load_frozen_vqgan(
        args.vqgan_checkpoint,
        device,
    )

    min_db = float(
        vqgan_checkpoint["min_db"]
    )

    max_db = float(
        vqgan_checkpoint["max_db"]
    )

    (
        train_dataset,
        validation_dataset,
        train_folds,
    ) = create_datasets(
        data_root=args.data_root,
        held_out_fold=args.held_out_fold,
        min_db=min_db,
        max_db=max_db,
    )
    if args.overfit_samples is not None:
        if args.overfit_samples < 1:
            raise ValueError(
                "overfit-samples must be at least 1"
            )

        number_of_samples = min(
            args.overfit_samples,
            len(train_dataset),
        )

        overfit_dataset = Subset(
            train_dataset,
            list(range(number_of_samples)),
        )

        train_dataset = overfit_dataset
        validation_dataset = overfit_dataset

        print(
            "Overfitting diagnostic:",
            number_of_samples,
            "identical training/validation samples",
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(
            args.num_workers > 0
        ),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(
            args.num_workers > 0
        ),
    )

    model = (
        MAGERespirationToEEGTransformer()
        .to(device)
    )

    initial_checkpoint = (
        load_initial_transformer(
            model=model,
            checkpoint_path=(
                args.initial_checkpoint
            ),
            held_out_fold=(
                args.held_out_fold
            ),
        )
    )

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

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    effective_batch_size = (
        args.batch_size
        * args.gradient_accumulation_steps
    )

    print("Device:", device)
    print(
        "Model: MAGE-style visible-token encoder-decoder"
    )
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
        "Normalization:",
        f"[{min_db:.2f}, {max_db:.2f}] dB",
    )
    print(
        "Transformer parameters:",
        f"{trainable_parameters:,}",
    )
    print(
        "Physical batch size:",
        args.batch_size,
    )
    print(
        "Gradient accumulation:",
        args.gradient_accumulation_steps,
    )
    print(
        "Effective batch size:",
        effective_batch_size,
    )
    print(
        "Full-mask probability:",
        args.full_mask_probability,
    )
    print(
        "Learning rate:",
        args.lr,
    )
    print(
        "VQGAN trainable parameters:",
        sum(
            parameter.requires_grad
            for parameter in vqgan.parameters()
        ),
    )

    if initial_checkpoint is not None:
        print(
            "Initial Transformer checkpoint:",
            args.initial_checkpoint,
        )
        print(
            "Initial checkpoint epoch:",
            initial_checkpoint.get("epoch"),
        )

    best_validation_loss = float("inf")

    for epoch_index in range(args.epochs):
        learning_rate = optimizer.param_groups[
            0
        ]["lr"]

        train_metrics = run_epoch(
            model=model,
            vqgan=vqgan,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            gradient_accumulation_steps=(
                args.gradient_accumulation_steps
            ),
            full_mask_probability=(
                args.full_mask_probability
            ),
            max_batches=(
                args.max_train_batches
            ),
            log_every=args.log_every,
        )

        validation_metrics = run_epoch(
            model=model,
            vqgan=vqgan,
            loader=validation_loader,
            device=device,
            max_batches=(
                args.max_val_batches
            ),
            log_every=args.log_every,
        )

        epoch_number = epoch_index + 1

        print(
            f"Epoch {epoch_number:03d}"
            f" | lr={learning_rate:.2e}"
            f" | train_loss="
            f"{train_metrics['loss']:.4f}"
            f" | train_acc="
            f"{train_metrics['accuracy']:.4f}"
            f" | train_mask="
            f"{train_metrics['mask_ratio']:.4f}"
            f" | val_loss="
            f"{validation_metrics['loss']:.4f}"
            f" | val_acc="
            f"{validation_metrics['accuracy']:.4f}"
            f" | val_top5="
            f"{validation_metrics['top5_accuracy']:.4f}",
            flush=True,
        )

        checkpoint = {
            "version": 3,
            "architecture": (
                "mage_visible_encoder_decoder"
            ),
            "epoch": epoch_number,
            "held_out_fold": (
                args.held_out_fold
            ),
            "train_folds": train_folds,
            "min_db": min_db,
            "max_db": max_db,
            "learning_rate": learning_rate,
            "weight_decay": (
                args.weight_decay
            ),
            "full_mask_probability": (
                args.full_mask_probability
            ),
            "initial_checkpoint": (
                str(args.initial_checkpoint)
                if args.initial_checkpoint
                is not None
                else None
            ),
            "vqgan_checkpoint": str(
                args.vqgan_checkpoint
            ),
            "model_state_dict": (
                model.state_dict()
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "scheduler_state_dict": (
                scheduler.state_dict()
            ),
            "train_metrics": (
                train_metrics
            ),
            "validation_metrics": (
                validation_metrics
            ),
        }

        torch.save(
            checkpoint,
            output_directory
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
                output_directory
                / "checkpoint_best.pt",
            )

        scheduler.step()

    print()
    print(
        "Best validation loss:",
        f"{best_validation_loss:.4f}",
    )


if __name__ == "__main__":
    main()