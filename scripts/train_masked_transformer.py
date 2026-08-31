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
from torch.utils.data import (
    DataLoader,
    Subset,
    WeightedRandomSampler,
)

from src.data.dataset import PhysiologyPairDataset
from src.models.masked_transformer import (
    RespirationToEEGTransformer,
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
    "outputs/masked_transformer_time_aligned_v6"
)

MODEL_WINDOW_SEC = 256 * 60
SUPPORTED_WINDOW_INDICES = (0, 1)


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

    supported_architectures = {
        "joint_codebook_tied_transformer_v3",
        "time_aligned_codebook_transformer_v6",
    }

    if architecture not in supported_architectures:
        raise ValueError(
            "Incompatible checkpoint architecture: "
            f"{architecture}"
        )

    checkpoint_model_config = checkpoint.get(
        "model_config"
    )

    requested_config = model.get_config()

    if architecture == "joint_codebook_tied_transformer_v3":
        # V3/V5 checkpoints predate the explicit aligned path.
        # Every shared parameter must still have the same config.
        compatible_v3_config = dict(requested_config)
        compatible_v3_config.pop(
            "use_aligned_respiration_conditioning",
            None,
        )

        if checkpoint_model_config != compatible_v3_config:
            raise ValueError(
                "Initial V3 checkpoint model_config does not "
                "match the shared V6 configuration"
            )

    elif checkpoint_model_config != requested_config:
        raise ValueError(
            "Initial V6 checkpoint model_config does not "
            "match the requested V6 model"
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

    if architecture == "joint_codebook_tied_transformer_v3":
        incompatible = model.load_state_dict(
            get_model_state_dict(checkpoint),
            strict=False,
        )

        allowed_missing_prefixes = (
            "aligned_respiration_projection.",
            "aligned_input_gate",
            "aligned_output_gate",
            "aligned_latent_projection.",
        )

        unexpected_missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith(
                allowed_missing_prefixes
            )
        ]

        if (
            unexpected_missing
            or incompatible.unexpected_keys
        ):
            raise ValueError(
                "Unexpected V3-to-V6 state mismatch. "
                f"Missing: {unexpected_missing}; "
                "unexpected: "
                f"{incompatible.unexpected_keys}"
            )

    else:
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


def filter_supported_windows(
    dataset,
    supported_windows=SUPPORTED_WINDOW_INDICES,
):
    """Keep window000/window001 and return aligned window IDs."""

    selected_indices = []
    selected_window_indices = []
    counts = {
        int(window_index): 0
        for window_index in supported_windows
    }

    if not hasattr(dataset, "files"):
        raise TypeError(
            "Window filtering requires a dataset with a files list"
        )

    for dataset_index, file_path in enumerate(
        dataset.files
    ):
        with np.load(file_path) as sample:
            if "start_sec" not in sample:
                raise KeyError(
                    "start_sec is required for window-aware training: "
                    f"{file_path}"
                )

            start_sec = int(
                round(
                    float(
                        np.asarray(
                            sample["start_sec"]
                        ).reshape(-1)[0]
                    )
                )
            )

        window_index = (
            start_sec // MODEL_WINDOW_SEC
        )

        if window_index not in supported_windows:
            continue

        selected_indices.append(dataset_index)
        selected_window_indices.append(
            int(window_index)
        )
        counts[int(window_index)] += 1

    for window_index in supported_windows:
        if counts[int(window_index)] < 1:
            raise RuntimeError(
                "No samples found for supported window "
                f"{window_index}"
            )

    return (
        Subset(dataset, selected_indices),
        selected_window_indices,
        counts,
    )


def create_balanced_window_sampler(
    window_indices,
    seed,
):
    """Create an approximately 50/50 sampler for windows 0 and 1."""

    counts = {
        window_index: window_indices.count(
            window_index
        )
        for window_index in SUPPORTED_WINDOW_INDICES
    }

    weights = torch.tensor(
        [
            1.0 / counts[window_index]
            for window_index in window_indices
        ],
        dtype=torch.double,
    )

    samples_per_epoch = 2 * max(
        counts.values()
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=samples_per_epoch,
        replacement=True,
        generator=generator,
    )

    return sampler, samples_per_epoch


def window_indices_from_batch(
    batch,
    device,
):
    """Convert start_sec metadata into window indices 0 or 1."""

    if "start_sec" not in batch:
        raise KeyError(
            "start_sec is missing from a training batch"
        )

    start_sec = torch.as_tensor(
        batch["start_sec"],
        device=device,
    ).reshape(-1)

    window_indices = torch.div(
        start_sec.round().long(),
        MODEL_WINDOW_SEC,
        rounding_mode="floor",
    )

    supported = torch.zeros_like(
        window_indices,
        dtype=torch.bool,
    )

    for window_index in SUPPORTED_WINDOW_INDICES:
        supported |= (
            window_indices == window_index
        )

    if not supported.all():
        raise ValueError(
            "Unsupported window index in batch: "
            f"{window_indices.detach().cpu().tolist()}"
        )

    return window_indices


def calculate_training_losses(
    logits,
    predicted_latents,
    aligned_predicted_latents,
    target_tokens,
    mask,
    normalized_codebook,
    semantic_loss_weight,
    temporal_loss_weight,
    temporal_cosine_weight,
    alignment_loss_weight,
):
    """Calculate token, semantic, temporal, and aligned losses."""

    batch_size = target_tokens.shape[0]
    grid_height = target_tokens.shape[1]
    grid_width = target_tokens.shape[2]

    targets_flat = target_tokens.flatten(
        start_dim=1
    )
    mask_flat = mask.flatten(start_dim=1)

    masked_logits = logits[mask_flat]
    masked_targets = targets_flat[mask_flat]

    cross_entropy_loss = F.cross_entropy(
        masked_logits,
        masked_targets,
    )

    # Compute latent losses in float32 even when the
    # Transformer forward pass uses mixed precision.
    predicted_latents_float = (
        predicted_latents.float()
    )
    codebook_float = normalized_codebook.float()

    target_vectors = F.embedding(
        targets_flat,
        codebook_float,
    )

    predicted_unit = F.normalize(
        predicted_latents_float,
        p=2,
        dim=-1,
        eps=1e-6,
    )

    target_unit = F.normalize(
        target_vectors,
        p=2,
        dim=-1,
        eps=1e-6,
    )

    semantic_cosine = (
        predicted_unit * target_unit
    ).sum(dim=-1)

    semantic_loss = (
        1.0 - semantic_cosine[mask_flat]
    ).mean()

    if aligned_predicted_latents is not None:
        if (
            aligned_predicted_latents.shape
            != predicted_latents.shape
        ):
            raise ValueError(
                "Aligned latent shape does not match final "
                "predicted latent shape: "
                f"{tuple(aligned_predicted_latents.shape)} "
                f"vs {tuple(predicted_latents.shape)}"
            )

        aligned_unit = F.normalize(
            aligned_predicted_latents.float(),
            p=2,
            dim=-1,
            eps=1e-6,
        )

        aligned_cosine = (
            aligned_unit * target_unit
        ).sum(dim=-1)

        # This auxiliary head never receives EEG tokens, so every
        # aligned position can be supervised even during partial
        # masking. In the recommended V6 run all positions are masked.
        alignment_loss = (
            1.0 - aligned_cosine
        ).mean()

    else:
        alignment_loss = (
            predicted_latents_float.sum() * 0.0
        )

    latent_dim = predicted_unit.shape[-1]

    predicted_grid = predicted_unit.reshape(
        batch_size,
        grid_height,
        grid_width,
        latent_dim,
    )

    target_grid = target_unit.reshape(
        batch_size,
        grid_height,
        grid_width,
        latent_dim,
    )

    predicted_delta = (
        predicted_grid[:, :, 1:, :]
        - predicted_grid[:, :, :-1, :]
    )

    target_delta = (
        target_grid[:, :, 1:, :]
        - target_grid[:, :, :-1, :]
    )

    temporal_mask = (
        mask[:, :, 1:]
        & mask[:, :, :-1]
    )

    target_delta_norm = torch.linalg.vector_norm(
        target_delta,
        ord=2,
        dim=-1,
    )

    temporal_mask &= target_delta_norm > 1e-6

    if temporal_mask.any():
        temporal_smooth_l1 = F.smooth_l1_loss(
            predicted_delta[temporal_mask],
            target_delta[temporal_mask],
        )

        temporal_cosine = F.cosine_similarity(
            predicted_delta,
            target_delta,
            dim=-1,
            eps=1e-6,
        )

        temporal_cosine_loss = (
            1.0
            - temporal_cosine[temporal_mask]
        ).mean()

        temporal_loss = (
            temporal_smooth_l1
            + temporal_cosine_weight
            * temporal_cosine_loss
        )

    else:
        zero = predicted_latents_float.sum() * 0.0
        temporal_smooth_l1 = zero
        temporal_cosine_loss = zero
        temporal_loss = zero

    total_loss = (
        cross_entropy_loss
        + semantic_loss_weight * semantic_loss
        + temporal_loss_weight * temporal_loss
        + alignment_loss_weight * alignment_loss
    )

    return {
        "total": total_loss,
        "cross_entropy": cross_entropy_loss,
        "semantic": semantic_loss,
        "alignment": alignment_loss,
        "temporal": temporal_loss,
        "temporal_smooth_l1": (
            temporal_smooth_l1
        ),
        "temporal_cosine": (
            temporal_cosine_loss
        ),
        "masked_logits": masked_logits,
        "masked_targets": masked_targets,
    }


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
    normalized_codebook,
    semantic_loss_weight=0.5,
    temporal_loss_weight=1.0,
    temporal_cosine_weight=0.25,
    alignment_loss_weight=0.5,
    optimizer=None,
    scaler=None,
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,
    full_mask_probability=1.0,
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
    total_cross_entropy_loss = 0.0
    total_semantic_loss = 0.0
    total_alignment_loss = 0.0
    total_temporal_loss = 0.0
    total_temporal_smooth_l1 = 0.0
    total_temporal_cosine_loss = 0.0
    total_correct = 0
    total_top5 = 0
    total_positions = 0
    total_mask_ratio = 0.0
    processed_batches = 0
    successful_optimizer_updates = 0
    skipped_optimizer_updates = 0
    predicted_code_seen = torch.zeros(
        model.codebook_size,
        dtype=torch.bool,
        device=device,
    )
    target_code_seen = torch.zeros(
        model.codebook_size,
        dtype=torch.bool,
        device=device,
    )

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

        window_indices = window_indices_from_batch(
            batch,
            device,
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

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                model_outputs = model(
                    respiration,
                    masked_tokens,
                    window_index=window_indices,
                    codebook=normalized_codebook,
                    return_latents=True,
                    return_alignment_latents=(
                        model.use_aligned_respiration_conditioning
                    ),
                )

                if (
                    model.use_aligned_respiration_conditioning
                ):
                    (
                        logits,
                        predicted_latents,
                        aligned_predicted_latents,
                    ) = model_outputs

                else:
                    (
                        logits,
                        predicted_latents,
                    ) = model_outputs
                    aligned_predicted_latents = None

            losses = calculate_training_losses(
                logits=logits,
                predicted_latents=(
                    predicted_latents
                ),
                aligned_predicted_latents=(
                    aligned_predicted_latents
                ),
                target_tokens=eeg_tokens,
                mask=mask,
                normalized_codebook=(
                    normalized_codebook
                ),
                semantic_loss_weight=(
                    semantic_loss_weight
                ),
                temporal_loss_weight=(
                    temporal_loss_weight
                ),
                temporal_cosine_weight=(
                    temporal_cosine_weight
                ),
                alignment_loss_weight=(
                    alignment_loss_weight
                ),
            )

            loss = losses["total"]
            masked_logits = losses[
                "masked_logits"
            ]
            masked_targets = losses[
                "masked_targets"
            ]

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
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=max_grad_norm,
                )

                scale_before_update = (
                    scaler.get_scale()
                )

                scaler.step(optimizer)
                scaler.update()

                scale_after_update = (
                    scaler.get_scale()
                )

                if (
                    scale_after_update
                    < scale_before_update
                ):
                    skipped_optimizer_updates += 1

                else:
                    successful_optimizer_updates += 1

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

            total_cross_entropy_loss += (
                losses["cross_entropy"].item()
                * number_of_positions
            )

            total_semantic_loss += (
                losses["semantic"].item()
                * number_of_positions
            )

            total_alignment_loss += (
                losses["alignment"].item()
                * number_of_positions
            )

            total_temporal_loss += (
                losses["temporal"].item()
                * number_of_positions
            )

            total_temporal_smooth_l1 += (
                losses[
                    "temporal_smooth_l1"
                ].item()
                * number_of_positions
            )

            total_temporal_cosine_loss += (
                losses[
                    "temporal_cosine"
                ].item()
                * number_of_positions
            )

            total_correct += correct.item()
            total_top5 += top5_correct.item()
            total_positions += number_of_positions

            predicted_code_seen[
                predictions.unique()
            ] = True

            target_code_seen[
                masked_targets.unique()
            ] = True

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
                f" | loss={running_loss:.4f}"
                " | ce="
                f"{total_cross_entropy_loss / total_positions:.4f}"
                " | semantic="
                f"{total_semantic_loss / total_positions:.4f}"
                " | alignment="
                f"{total_alignment_loss / total_positions:.4f}"
                " | temporal="
                f"{total_temporal_loss / total_positions:.4f}"
                " | temp_l1="
                f"{total_temporal_smooth_l1 / total_positions:.4f}"
                " | temp_cos="
                f"{total_temporal_cosine_loss / total_positions:.4f}",
                flush=True,
            )

    if processed_batches == 0:
        raise RuntimeError(
            "No batch was processed"
        )

    if (
        training
        and successful_optimizer_updates == 0
    ):
        raise RuntimeError(
            "Every optimizer update was skipped by AMP. "
            "The gradients are non-finite; do not start "
            "the full training run."
        )

    return {
        "loss": (
            total_loss
            / total_positions
        ),
        "cross_entropy_loss": (
            total_cross_entropy_loss
            / total_positions
        ),
        "semantic_loss": (
            total_semantic_loss
            / total_positions
        ),
        "alignment_loss": (
            total_alignment_loss
            / total_positions
        ),
        "temporal_loss": (
            total_temporal_loss
            / total_positions
        ),
        "temporal_smooth_l1_loss": (
            total_temporal_smooth_l1
            / total_positions
        ),
        "temporal_cosine_loss": (
            total_temporal_cosine_loss
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
        "unique_predicted_codes": int(
            predicted_code_seen.sum().item()
        ),
        "unique_target_codes": int(
            target_code_seen.sum().item()
        ),
        "positions": total_positions,
        "batches": processed_batches,
        "optimizer_updates": (
            successful_optimizer_updates
        ),
        "skipped_optimizer_updates": (
            skipped_optimizer_updates
        ),
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
        "--max-grad-norm",
        type=float,
        default=1.0,
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
        "--aligned-lr-multiplier",
        type=float,
        default=4.0,
        help=(
            "Learning-rate multiplier for the new V6 aligned "
            "conditioning modules and gates."
        ),
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
        default=1.00,
        help=(
            "Probability that all EEG tokens of a "
            "training example are masked."
        ),
    )

    parser.add_argument(
        "--semantic-loss-weight",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--alignment-loss-weight",
        type=float,
        default=0.50,
        help=(
            "Weight of the respiration-only local latent "
            "prediction loss. This directly supervises the "
            "64 aligned four-minute positions."
        ),
    )

    parser.add_argument(
        "--temporal-loss-weight",
        type=float,
        default=1.00,
    )

    parser.add_argument(
        "--temporal-cosine-weight",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--codebook-temperature",
        type=float,
        default=0.07,
    )

    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=384,
    )

    parser.add_argument(
        "--num-heads",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--num-encoder-layers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--num-decoder-layers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--mlp-ratio",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
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

    if args.max_grad_norm <= 0:
        raise ValueError(
            "max-grad-norm must be positive"
        )

    if args.lr <= 0:
        raise ValueError(
            "lr must be positive"
        )

    if args.aligned_lr_multiplier <= 0:
        raise ValueError(
            "aligned-lr-multiplier must be positive"
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

    if args.semantic_loss_weight < 0:
        raise ValueError(
            "semantic-loss-weight cannot be negative"
        )

    if args.alignment_loss_weight < 0:
        raise ValueError(
            "alignment-loss-weight cannot be negative"
        )

    if args.temporal_loss_weight < 0:
        raise ValueError(
            "temporal-loss-weight cannot be negative"
        )

    if args.temporal_cosine_weight < 0:
        raise ValueError(
            "temporal-cosine-weight cannot be negative"
        )

    if args.codebook_temperature <= 0:
        raise ValueError(
            "codebook-temperature must be positive"
        )

    if args.embedding_dim < 1:
        raise ValueError(
            "embedding-dim must be positive"
        )

    if args.num_heads < 1:
        raise ValueError(
            "num-heads must be positive"
        )

    if args.num_encoder_layers < 1:
        raise ValueError(
            "num-encoder-layers must be positive"
        )

    if args.num_decoder_layers < 1:
        raise ValueError(
            "num-decoder-layers must be positive"
        )

    if args.mlp_ratio < 1:
        raise ValueError(
            "mlp-ratio must be positive"
        )

    if args.embedding_dim % args.num_heads != 0:
        raise ValueError(
            "embedding-dim must be divisible by num-heads"
        )

    if not 0.0 <= args.dropout < 1.0:
        raise ValueError(
            "dropout must be in [0, 1)"
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

    (
        train_dataset,
        train_window_indices,
        train_window_counts,
    ) = filter_supported_windows(
        train_dataset
    )

    (
        validation_dataset,
        validation_window_indices,
        validation_window_counts,
    ) = filter_supported_windows(
        validation_dataset
    )

    train_sampler = None
    samples_per_training_epoch = len(
        train_dataset
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
        train_window_indices = (
            train_window_indices[
                :number_of_samples
            ]
        )

        print(
            "Overfitting diagnostic:",
            number_of_samples,
            "identical training/validation samples",
        )

    else:
        (
            train_sampler,
            samples_per_training_epoch,
        ) = create_balanced_window_sampler(
            window_indices=(
                train_window_indices
            ),
            seed=args.seed,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
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

    model = RespirationToEEGTransformer(
        codebook_size=(
            vqgan.quantizer.codebook.weight.shape[0]
        ),
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_encoder_layers=(
            args.num_encoder_layers
        ),
        num_decoder_layers=(
            args.num_decoder_layers
        ),
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        num_window_types=2,
        use_window_embedding=True,
        codebook_dim=(
            vqgan.quantizer.codebook.weight.shape[1]
        ),
        use_codebook_tied_output=True,
        codebook_temperature=(
            args.codebook_temperature
        ),
        use_aligned_respiration_conditioning=True,
    ).to(device)

    normalized_codebook = F.normalize(
        vqgan.quantizer.codebook.weight.detach(),
        p=2,
        dim=1,
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

    base_parameters = []
    aligned_parameters = []

    for parameter_name, parameter in (
        model.named_parameters()
    ):
        if parameter_name.startswith("aligned_"):
            aligned_parameters.append(parameter)

        else:
            base_parameters.append(parameter)

    if not aligned_parameters:
        raise RuntimeError(
            "No V6 aligned parameters were found"
        )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": base_parameters,
                "lr": args.lr,
            },
            {
                "params": aligned_parameters,
                "lr": (
                    args.lr
                    * args.aligned_lr_multiplier
                ),
            },
        ],
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
        "Model: time-aligned codebook Transformer V6"
    )
    print("Training folds:", train_folds)
    print(
        "Training samples:",
        len(train_dataset),
    )
    print(
        "Balanced samples per training epoch:",
        samples_per_training_epoch,
    )
    print(
        "Training windows before balancing:",
        train_window_counts,
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
        "Validation windows:",
        validation_window_counts,
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
        "Maximum gradient norm:",
        args.max_grad_norm,
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
        "Loss weights: CE=1.0 | latent semantic=",
        args.semantic_loss_weight,
        "| aligned respiration=",
        args.alignment_loss_weight,
        "| temporal=",
        args.temporal_loss_weight,
    )
    print(
        "Temporal cosine weight:",
        args.temporal_cosine_weight,
    )
    print(
        "Initial codebook temperature:",
        args.codebook_temperature,
    )
    print(
        "Base learning rate:",
        args.lr,
    )
    print(
        "Aligned learning rate:",
        args.lr * args.aligned_lr_multiplier,
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
        aligned_learning_rate = optimizer.param_groups[
            1
        ]["lr"]

        train_metrics = run_epoch(
            model=model,
            vqgan=vqgan,
            loader=train_loader,
            device=device,
            normalized_codebook=(
                normalized_codebook
            ),
            semantic_loss_weight=(
                args.semantic_loss_weight
            ),
            temporal_loss_weight=(
                args.temporal_loss_weight
            ),
            temporal_cosine_weight=(
                args.temporal_cosine_weight
            ),
            alignment_loss_weight=(
                args.alignment_loss_weight
            ),
            optimizer=optimizer,
            scaler=scaler,
            gradient_accumulation_steps=(
                args.gradient_accumulation_steps
            ),
            max_grad_norm=args.max_grad_norm,
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
            normalized_codebook=(
                normalized_codebook
            ),
            semantic_loss_weight=(
                args.semantic_loss_weight
            ),
            temporal_loss_weight=(
                args.temporal_loss_weight
            ),
            temporal_cosine_weight=(
                args.temporal_cosine_weight
            ),
            alignment_loss_weight=(
                args.alignment_loss_weight
            ),
            max_batches=(
                args.max_val_batches
            ),
            log_every=args.log_every,
        )

        epoch_number = epoch_index + 1

        aligned_input_gate = float(
            torch.tanh(
                model.aligned_input_gate.detach()
            ).item()
        )
        aligned_output_gate = float(
            torch.tanh(
                model.aligned_output_gate.detach()
            ).item()
        )

        print(
            f"Epoch {epoch_number:03d}"
            f" | lr={learning_rate:.2e}"
            f" | aligned_lr={aligned_learning_rate:.2e}"
            f" | train_loss="
            f"{train_metrics['loss']:.4f}"
            f" | train_ce="
            f"{train_metrics['cross_entropy_loss']:.4f}"
            f" | train_sem="
            f"{train_metrics['semantic_loss']:.4f}"
            f" | train_align="
            f"{train_metrics['alignment_loss']:.4f}"
            f" | train_temp="
            f"{train_metrics['temporal_loss']:.4f}"
            f" | train_temp_l1="
            f"{train_metrics['temporal_smooth_l1_loss']:.4f}"
            f" | train_temp_cos="
            f"{train_metrics['temporal_cosine_loss']:.4f}"
            f" | train_acc="
            f"{train_metrics['accuracy']:.4f}"
            f" | train_codes="
            f"{train_metrics['unique_predicted_codes']}"
            f" | updates="
            f"{train_metrics['optimizer_updates']}"
            f" | skipped="
            f"{train_metrics['skipped_optimizer_updates']}"
            f" | train_mask="
            f"{train_metrics['mask_ratio']:.4f}"
            f" | val_loss="
            f"{validation_metrics['loss']:.4f}"
            f" | val_ce="
            f"{validation_metrics['cross_entropy_loss']:.4f}"
            f" | val_sem="
            f"{validation_metrics['semantic_loss']:.4f}"
            f" | val_align="
            f"{validation_metrics['alignment_loss']:.4f}"
            f" | val_temp="
            f"{validation_metrics['temporal_loss']:.4f}"
            f" | val_temp_l1="
            f"{validation_metrics['temporal_smooth_l1_loss']:.4f}"
            f" | val_temp_cos="
            f"{validation_metrics['temporal_cosine_loss']:.4f}"
            f" | val_acc="
            f"{validation_metrics['accuracy']:.4f}"
            f" | val_codes="
            f"{validation_metrics['unique_predicted_codes']}"
            f" | val_top5="
            f"{validation_metrics['top5_accuracy']:.4f}"
            f" | gate_in={aligned_input_gate:.4f}"
            f" | gate_out={aligned_output_gate:.4f}",
            flush=True,
        )

        checkpoint = {
            "version": 6,
            "architecture": (
                "time_aligned_codebook_transformer_v6"
            ),
            "model_config": model.get_config(),
            "epoch": epoch_number,
            "held_out_fold": (
                args.held_out_fold
            ),
            "train_folds": train_folds,
            "min_db": min_db,
            "max_db": max_db,
            "learning_rate": learning_rate,
            "aligned_learning_rate": (
                aligned_learning_rate
            ),
            "aligned_lr_multiplier": (
                args.aligned_lr_multiplier
            ),
            "weight_decay": (
                args.weight_decay
            ),
            "max_grad_norm": (
                args.max_grad_norm
            ),
            "full_mask_probability": (
                args.full_mask_probability
            ),
            "semantic_loss_weight": (
                args.semantic_loss_weight
            ),
            "alignment_loss_weight": (
                args.alignment_loss_weight
            ),
            "temporal_loss_weight": (
                args.temporal_loss_weight
            ),
            "temporal_cosine_weight": (
                args.temporal_cosine_weight
            ),
            "codebook_temperature": (
                args.codebook_temperature
            ),
            "supported_window_indices": list(
                SUPPORTED_WINDOW_INDICES
            ),
            "train_window_counts": (
                train_window_counts
            ),
            "validation_window_counts": (
                validation_window_counts
            ),
            "balanced_samples_per_epoch": (
                samples_per_training_epoch
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
            "aligned_input_gate": (
                aligned_input_gate
            ),
            "aligned_output_gate": (
                aligned_output_gate
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
