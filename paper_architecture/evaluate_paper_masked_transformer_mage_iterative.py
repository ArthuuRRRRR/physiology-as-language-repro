"""Evaluate the paper respiration-to-EEG Transformer with MAGE-style
iterative masked generation.

IMPORTANT
---------
This file does NOT replace the literal paper one-shot evaluator.

It is a decoding ablation using the official MAGE generation procedure:
    - start with every EEG position masked;
    - no physical token dropping during generation;
    - categorical token sampling;
    - cosine remasking schedule;
    - confidence + Gumbel noise;
    - iterative refinement.

The original paper architecture and trained checkpoint are unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from paper_architecture import (
    PaperMaskedRespirationToEEGTransformer,
)

from src.models.vqgan import VQGAN

from paper_architecture.evaluate_paper_masked_transformer import (
    PaperTransformerEvaluationDataset,
    ShuffledRespirationDataset,
    standardize_respiration,
    decode_token_ids,
    load_state_dict,
    create_accumulator,
    update_accumulator,
    finalize_accumulator,
    save_comparison,
    save_time_block_plot,
    MODEL_WINDOW_SEC,
    DIAGNOSTIC_BLOCK_SEC,
)


DEFAULT_TRANSFORMER_CHECKPOINT = Path(
    "outputs/paper_masked_transformer_multidataset_full/"
    "checkpoint_best.pt"
)

DEFAULT_VQGAN_CHECKPOINT = Path(
    "outputs/vqgan_multidataset/checkpoint_best.pt"
)

DEFAULT_TRANSFORMER_DATA_ROOT = Path(
    "outputs/paper_transformer_data"
)

DEFAULT_PREPROCESSED_ROOT = Path(
    "outputs/vqgan_multidataset_preprocessed"
)


# ============================================================
# Exact MAGE-style confidence masking
# ============================================================

def mask_by_random_topk(
    mask_len: int,
    probs: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    Adaptation of MAGE's mask_by_random_topk.

    Lower-confidence tokens are remasked.

    probs:
        (B, N)

    Returns:
        bool mask (B, N)
    """

    if probs.ndim != 2:
        raise ValueError(
            f"Expected probs (B, N), got {tuple(probs.shape)}"
        )

    if probs.shape[0] != 1:
        raise ValueError(
            "This diagnostic evaluator currently expects batch_size=1."
        )

    mask_len = int(mask_len)

    if mask_len < 1:
        raise ValueError(
            "mask_len must be at least 1"
        )

    if mask_len > probs.shape[1]:
        raise ValueError(
            "mask_len exceeds token count"
        )

    # Official MAGE:
    # confidence = log(prob) + temperature * Gumbel noise.
    #
    # NumPy is deliberately used here to stay close to the
    # public MAGE implementation.
    gumbel = np.random.gumbel(
        size=tuple(probs.shape)
    )

    gumbel = torch.as_tensor(
        gumbel,
        device=probs.device,
        dtype=probs.dtype,
    )

    confidence = (
        torch.log(
            probs.clamp_min(1e-12)
        )
        + temperature * gumbel
    )

    sorted_confidence, _ = torch.sort(
        confidence,
        dim=-1,
    )

    cutoff = sorted_confidence[
        :,
        mask_len - 1 : mask_len,
    ]

    return confidence <= cutoff


# ============================================================
# Run the existing model with externally specified masks
# ============================================================

def forward_with_generation_mask(
    model: PaperMaskedRespirationToEEGTransformer,
    respiration: torch.Tensor,
    eeg_tokens: torch.Tensor,
    all_mask: torch.Tensor,
):
    """
    Run the existing paper model without changing its weights or
    implementation.

    Training normally creates:
        drop_mask
        all_mask

    For MAGE generation, official MAGE uses:
        drop_mask = 0

    while all_mask represents the EEG positions that remain unknown.

    We temporarily override only encoder._make_masks so the original
    encoder/decoder path is reused exactly.
    """

    if all_mask.dtype != torch.bool:
        all_mask = all_mask.bool()

    batch_size = eeg_tokens.shape[0]

    expected_shape = (
        batch_size,
        model.num_eeg_tokens,
    )

    if tuple(all_mask.shape) != expected_shape:
        raise ValueError(
            f"Expected all_mask {expected_shape}, "
            f"got {tuple(all_mask.shape)}"
        )

    original_make_masks = (
        model.encoder._make_masks
    )

    def fixed_make_masks(
        batch_size,
        device,
        mask_ratio,
    ):
        if batch_size != all_mask.shape[0]:
            raise RuntimeError(
                "Generation batch size changed unexpectedly."
            )

        drop_mask = torch.zeros(
            (
                batch_size,
                model.num_eeg_tokens,
            ),
            dtype=torch.bool,
            device=device,
        )

        fixed_all_mask = all_mask.to(
            device=device
        )

        used_ratio = float(
            fixed_all_mask
            .float()
            .mean()
            .item()
        )

        return (
            drop_mask,
            fixed_all_mask,
            used_ratio,
        )

    model.encoder._make_masks = (
        fixed_make_masks
    )

    try:
        outputs = model(
            respiration=respiration,
            eeg_tokens=eeg_tokens,
            # Value itself is ignored by our fixed _make_masks.
            mask_ratio=1.0,
        )

    finally:
        model.encoder._make_masks = (
            original_make_masks
        )

    return outputs


# ============================================================
# MAGE iterative generation
# ============================================================

@torch.no_grad()
def generate_mage_iterative(
    model: PaperMaskedRespirationToEEGTransformer,
    respiration: torch.Tensor,
    num_iterations: int = 12,
    choice_temperature: float = 4.5,
):
    """
    Cross-modal adaptation of official MAGE generation.

    Official MAGE generation:
        1. start fully masked
        2. no physical token dropping
        3. categorical sampling
        4. keep previous known tokens
        5. cosine masking schedule
        6. confidence + Gumbel noise
        7. repeat

    Returns
    -------
    generated_tokens
        (B, 512)

    committed_logits
        logits associated with the step at which each position
        became committed. Used for CE/top-5 diagnostics without
        scoring a token after it has already been made visible.

    remaining_counts
        number of still-masked positions after each iteration.
    """

    if num_iterations < 1:
        raise ValueError(
            "num_iterations must be >= 1"
        )

    if choice_temperature < 0:
        raise ValueError(
            "choice_temperature must be >= 0"
        )

    batch_size = respiration.shape[0]

    if batch_size != 1:
        raise ValueError(
            "Use batch_size=1 for this diagnostic."
        )

    num_tokens = model.num_eeg_tokens
    codebook_size = model.codebook_size
    device = respiration.device

    # The encoder requires real code IDs as eeg_tokens.
    #
    # Unknown positions contain a harmless placeholder 0.
    # all_mask causes the encoder to replace those positions
    # with its true EEG mask embedding before attention.
    current_ids = torch.zeros(
        (
            batch_size,
            num_tokens,
        ),
        dtype=torch.long,
        device=device,
    )

    current_mask = torch.ones(
        (
            batch_size,
            num_tokens,
        ),
        dtype=torch.bool,
        device=device,
    )

    committed_logits = torch.empty(
        (
            batch_size,
            num_tokens,
            codebook_size,
        ),
        dtype=torch.float32,
        device=device,
    )

    committed_positions = torch.zeros(
        (
            batch_size,
            num_tokens,
        ),
        dtype=torch.bool,
        device=device,
    )

    remaining_counts = []

    for step in range(
        num_iterations
    ):
        # ----------------------------------------------------
        # Transformer forward with:
        #     drop_mask = 0
        #     all_mask = current unknown positions
        # ----------------------------------------------------

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(
                device.type == "cuda"
            ),
        ):
            outputs = (
                forward_with_generation_mask(
                    model=model,
                    respiration=respiration,
                    eeg_tokens=current_ids,
                    all_mask=current_mask,
                )
            )

        logits = outputs[
            "logits"
        ].float()

        # ----------------------------------------------------
        # Official MAGE categorical sampling
        # ----------------------------------------------------

        sample_dist = (
            torch.distributions
            .Categorical(
                logits=logits
            )
        )

        sampled_ids = (
            sample_dist.sample()
        )

        # Already known tokens cannot change.
        sampled_ids = torch.where(
            current_mask,
            sampled_ids,
            current_ids,
        )

        # ----------------------------------------------------
        # Cosine schedule
        # ----------------------------------------------------

        ratio = (
            float(step + 1)
            / float(num_iterations)
        )

        mask_ratio = math.cos(
            math.pi
            / 2.0
            * ratio
        )

        probs = F.softmax(
            logits,
            dim=-1,
        )

        selected_probs = torch.gather(
            probs,
            dim=-1,
            index=sampled_ids.unsqueeze(-1),
        ).squeeze(-1)

        # Known tokens receive infinite confidence so they
        # cannot be remasked.
        selected_probs = torch.where(
            current_mask,
            selected_probs,
            torch.full_like(
                selected_probs,
                float("inf"),
            ),
        )

        unknown_count = int(
            current_mask.sum().item()
        )

        requested_mask_len = int(
            math.floor(
                num_tokens
                * mask_ratio
            )
        )

        # MAGE keeps at least one new prediction each round
        # and also leaves at least one masked token for a
        # following round.
        mask_len = max(
            1,
            min(
                unknown_count - 1,
                requested_mask_len,
            ),
        )

        # If only one unknown token remains, official MAGE's
        # clamp leaves it masked in token_indices, but the
        # final returned sampled_ids contains its prediction.
        if unknown_count <= 1:
            mask_len = 1

        step_temperature = (
            choice_temperature
            * (
                1.0 - ratio
            )
        )

        next_mask = mask_by_random_topk(
            mask_len=mask_len,
            probs=selected_probs,
            temperature=step_temperature,
        )

        # ----------------------------------------------------
        # Logits stored at commit time
        # ----------------------------------------------------

        if (
            step
            == num_iterations - 1
        ):
            newly_committed = (
                current_mask
            )

        else:
            newly_committed = (
                current_mask
                & ~next_mask
            )

        committed_logits[
            newly_committed
        ] = logits[
            newly_committed
        ]

        committed_positions |= (
            newly_committed
        )

        # Sampled values become the candidate context.
        current_ids = sampled_ids

        if (
            step
            == num_iterations - 1
        ):
            remaining_counts.append(
                0
            )
            break

        current_mask = next_mask

        remaining_counts.append(
            int(
                current_mask
                .sum()
                .item()
            )
        )

    if not committed_positions.all():
        raise RuntimeError(
            "Iterative decoding ended with "
            "uncommitted positions."
        )

    if torch.any(
        current_ids < 0
    ) or torch.any(
        current_ids
        >= codebook_size
    ):
        raise RuntimeError(
            "Generated invalid VQ code."
        )

    return (
        current_ids,
        committed_logits,
        remaining_counts,
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the paper Transformer with "
            "MAGE-consistent iterative decoding."
        )
    )

    parser.add_argument(
        "--transformer-checkpoint",
        type=Path,
        default=DEFAULT_TRANSFORMER_CHECKPOINT,
    )

    parser.add_argument(
        "--vqgan-checkpoint",
        type=Path,
        default=DEFAULT_VQGAN_CHECKPOINT,
    )

    parser.add_argument(
        "--transformer-data-root",
        type=Path,
        default=DEFAULT_TRANSFORMER_DATA_ROOT,
    )

    parser.add_argument(
        "--preprocessed-root",
        type=Path,
        default=DEFAULT_PREPROCESSED_ROOT,
    )

    parser.add_argument(
        "--split",
        choices=[
            "train",
            "val",
            "test",
        ],
        default="test",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/"
            "paper_multidataset_mesa_test_"
            "mage_iterative"
        ),
    )

    parser.add_argument(
        "--num-iterations",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--choice-temperature",
        type=float,
        default=4.5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--shuffle-respiration",
        action="store_true",
    )

    args = parser.parse_args()

    if args.num_iterations < 1:
        raise ValueError(
            "--num-iterations must be >= 1"
        )

    if (
        args.max_samples is not None
        and args.max_samples < 1
    ):
        raise ValueError(
            "--max-samples must be >= 1"
        )

    torch.manual_seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(
                device
            )
        )

    # ========================================================
    # Transformer
    # ========================================================

    transformer_checkpoint = torch.load(
        args.transformer_checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    model_config = (
        transformer_checkpoint[
            "model_config"
        ]
    )

    transformer = (
        PaperMaskedRespirationToEEGTransformer(
            **model_config
        )
        .to(device)
    )

    transformer_state = load_state_dict(
        transformer_checkpoint,
        [
            "model_state_dict",
            "transformer_state_dict",
        ],
    )

    transformer.load_state_dict(
        transformer_state,
        strict=True,
    )

    transformer.eval()

    checkpoint_epoch = (
        int(
            transformer_checkpoint.get(
                "epoch",
                -1,
            )
        )
        + 1
    )

    print(
        "Transformer checkpoint:",
        args.transformer_checkpoint
    )

    print(
        "Checkpoint epoch:",
        checkpoint_epoch
    )

    print(
        "Shared temporal position:",
        transformer.model_config.get(
            "shared_temporal_position",
            False,
        )
    )

    print(
        "MAGE iterations:",
        args.num_iterations
    )

    print(
        "MAGE choice temperature:",
        args.choice_temperature
    )

    # ========================================================
    # VQGAN
    # ========================================================

    vqgan_checkpoint = torch.load(
        args.vqgan_checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    vqgan = VQGAN().to(
        device
    )

    vqgan_state = load_state_dict(
        vqgan_checkpoint,
        [
            "model_state_dict",
            "vqgan_state_dict",
            "generator_state_dict",
            "state_dict",
        ],
    )

    vqgan.load_state_dict(
        vqgan_state,
        strict=True,
    )

    vqgan.eval()

    for parameter in (
        vqgan.parameters()
    ):
        parameter.requires_grad = False

    # ========================================================
    # Dataset
    # ========================================================

    base_dataset = (
        PaperTransformerEvaluationDataset(
            transformer_data_root=(
                args.transformer_data_root
            ),
            preprocessed_root=(
                args.preprocessed_root
            ),
            split=args.split,
        )
    )

    dataset = base_dataset

    if args.shuffle_respiration:
        dataset = (
            ShuffledRespirationDataset(
                base_dataset
            )
        )

    number_to_evaluate = len(
        dataset
    )

    if args.max_samples is not None:
        number_to_evaluate = min(
            number_to_evaluate,
            args.max_samples,
        )

    print(
        "Evaluated samples:",
        number_to_evaluate
    )

    print(
        "Shuffled respiration:",
        args.shuffle_respiration
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    # ========================================================
    # Metrics
    # ========================================================

    global_accumulator = (
        create_accumulator()
    )

    window_accumulators = {}
    time_block_accumulators = {}

    blocks_per_window = (
        MODEL_WINDOW_SEC
        // DIAGNOSTIC_BLOCK_SEC
    )

    comparison_saved = False
    evaluated_samples = 0
    first_schedule = None

    with torch.inference_mode():

        for batch in loader:

            if (
                evaluated_samples
                >= number_to_evaluate
            ):
                break

            respiration = (
                batch[
                    "respiration"
                ]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            respiration = (
                standardize_respiration(
                    respiration
                )
            )

            true_tokens = (
                batch[
                    "eeg_tokens"
                ]
                .to(
                    device,
                    non_blocking=True,
                )
                .long()
            )

            eeg = (
                batch[
                    "eeg_spectrogram"
                ]
                .unsqueeze(1)
                .to(
                    device,
                    non_blocking=True,
                )
            )

            (
                predicted_flat,
                committed_logits,
                remaining_counts,
            ) = generate_mage_iterative(
                model=transformer,
                respiration=respiration,
                num_iterations=(
                    args.num_iterations
                ),
                choice_temperature=(
                    args.choice_temperature
                ),
            )

            if first_schedule is None:
                first_schedule = (
                    remaining_counts
                )

                print(
                    "Remaining masks after each iteration:",
                    first_schedule,
                )

            predicted_tokens = (
                predicted_flat
                .view_as(
                    true_tokens
                )
            )

            targets_flat = (
                true_tokens.flatten(
                    start_dim=1
                )
            )

            position_losses = (
                F.cross_entropy(
                    committed_logits.reshape(
                        -1,
                        committed_logits.shape[-1],
                    ),
                    targets_flat.reshape(
                        -1
                    ),
                    reduction="none",
                )
                .view_as(
                    targets_flat
                )
            )

            correct = (
                predicted_flat
                == targets_flat
            )

            top5_predictions = (
                committed_logits.topk(
                    k=5,
                    dim=-1,
                ).indices
            )

            top5_correct = (
                top5_predictions
                == targets_flat.unsqueeze(
                    -1
                )
            ).any(
                dim=-1
            )

            oracle_reconstruction = (
                decode_token_ids(
                    vqgan,
                    true_tokens,
                )
            )

            predicted_reconstruction = (
                decode_token_ids(
                    vqgan,
                    predicted_tokens,
                )
            )

            # ------------------------------------------------
            # Global
            # ------------------------------------------------

            update_accumulator(
                global_accumulator,
                position_losses,
                correct,
                top5_correct,
                eeg,
                predicted_reconstruction,
                oracle_reconstruction,
                true_tokens,
                predicted_tokens,
            )

            start_sec = int(
                batch[
                    "start_sec"
                ]
                .reshape(-1)[0]
                .item()
            )

            window_index = (
                start_sec
                // MODEL_WINDOW_SEC
            )

            window_name = (
                f"window{window_index:03d}"
            )

            if (
                window_name
                not in window_accumulators
            ):
                window_accumulators[
                    window_name
                ] = create_accumulator(
                    start_sec
                )

            update_accumulator(
                window_accumulators[
                    window_name
                ],
                position_losses,
                correct,
                top5_correct,
                eeg,
                predicted_reconstruction,
                oracle_reconstruction,
                true_tokens,
                predicted_tokens,
            )

            # ------------------------------------------------
            # 32-minute blocks
            # ------------------------------------------------

            eeg_columns_per_block = (
                eeg.shape[-1]
                // blocks_per_window
            )

            token_columns_per_block = (
                true_tokens.shape[-1]
                // blocks_per_window
            )

            loss_grid = (
                position_losses
                .reshape_as(
                    true_tokens
                )
            )

            correct_grid = (
                correct
                .reshape_as(
                    true_tokens
                )
            )

            top5_grid = (
                top5_correct
                .reshape_as(
                    true_tokens
                )
            )

            for local_block in range(
                blocks_per_window
            ):
                block_start_sec = (
                    start_sec
                    + local_block
                    * DIAGNOSTIC_BLOCK_SEC
                )

                block_end_sec = (
                    block_start_sec
                    + DIAGNOSTIC_BLOCK_SEC
                )

                absolute_block_index = (
                    block_start_sec
                    // DIAGNOSTIC_BLOCK_SEC
                )

                block_name = (
                    f"block"
                    f"{absolute_block_index:03d}"
                )

                if (
                    block_name
                    not in time_block_accumulators
                ):
                    time_block_accumulators[
                        block_name
                    ] = create_accumulator(
                        block_start_sec,
                        block_end_sec,
                    )

                eeg_start = (
                    local_block
                    * eeg_columns_per_block
                )

                eeg_end = (
                    eeg_start
                    + eeg_columns_per_block
                )

                token_start = (
                    local_block
                    * token_columns_per_block
                )

                token_end = (
                    token_start
                    + token_columns_per_block
                )

                update_accumulator(
                    time_block_accumulators[
                        block_name
                    ],

                    loss_grid[
                        ...,
                        token_start:token_end,
                    ],

                    correct_grid[
                        ...,
                        token_start:token_end,
                    ],

                    top5_grid[
                        ...,
                        token_start:token_end,
                    ],

                    eeg[
                        ...,
                        eeg_start:eeg_end,
                    ],

                    predicted_reconstruction[
                        ...,
                        eeg_start:eeg_end,
                    ],

                    oracle_reconstruction[
                        ...,
                        eeg_start:eeg_end,
                    ],

                    true_tokens[
                        ...,
                        token_start:token_end,
                    ],

                    predicted_tokens[
                        ...,
                        token_start:token_end,
                    ],
                )

            # ------------------------------------------------
            # Figure
            # ------------------------------------------------

            if not comparison_saved:

                save_comparison(
                    ground_truth=eeg,
                    oracle_reconstruction=(
                        oracle_reconstruction
                    ),
                    predicted_reconstruction=(
                        predicted_reconstruction
                    ),
                    output_path=(
                        args.output_dir
                        / "reconstruction_comparison.png"
                    ),
                    zero_respiration=False,
                    shuffle_respiration=(
                        args.shuffle_respiration
                    ),
                )

                comparison_saved = True

            evaluated_samples += 1

            if (
                evaluated_samples % 10 == 0
                or evaluated_samples
                == number_to_evaluate
            ):
                running = (
                    finalize_accumulator(
                        global_accumulator
                    )
                )

                print(
                    f"{evaluated_samples}/"
                    f"{number_to_evaluate}"
                    f" | CE="
                    f"{running['cross_entropy']:.4f}"
                    f" | acc="
                    f"{running['token_accuracy']:.4f}"
                    f" | MAE="
                    f"{running['predicted_eeg_mae']:.4f}"
                    f" | temporal_corr="
                    f"{running['predicted_eeg_temporal_correlation']:.4f}"
                    f" | SNR="
                    f"{running['predicted_eeg_snr_db']:.2f} dB",
                    flush=True,
                )

    # ========================================================
    # Finalize
    # ========================================================

    global_metrics = (
        finalize_accumulator(
            global_accumulator
        )
    )

    metrics_by_window = {
        name: finalize_accumulator(
            accumulator
        )
        for name, accumulator
        in sorted(
            window_accumulators.items()
        )
    }

    metrics_by_time_block = {
        name: finalize_accumulator(
            accumulator
        )
        for name, accumulator
        in sorted(
            time_block_accumulators.items(),
            key=lambda item: (
                item[1]["start_sec"]
            ),
        )
    }

    metrics = {
        "decoding": (
            "mage_consistent_iterative"
        ),

        "transformer_checkpoint": str(
            args.transformer_checkpoint
        ),

        "vqgan_checkpoint": str(
            args.vqgan_checkpoint
        ),

        "checkpoint_epoch": (
            checkpoint_epoch
        ),

        "split": args.split,

        "evaluated_samples": (
            evaluated_samples
        ),

        "num_iterations": (
            args.num_iterations
        ),

        "choice_temperature": (
            args.choice_temperature
        ),

        "seed": args.seed,

        "shuffle_respiration": (
            args.shuffle_respiration
        ),

        "remaining_mask_schedule": (
            first_schedule
        ),

        **{
            key: value
            for key, value
            in global_metrics.items()
            if key not in (
                "start_sec",
                "start_min",
            )
        },

        "metrics_by_window": (
            metrics_by_window
        ),

        "metrics_by_time_block": (
            metrics_by_time_block
        ),
    }

    metrics_path = (
        args.output_dir
        / "metrics.json"
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2,
        )

    save_time_block_plot(
        metrics_by_time_block,
        args.output_dir
        / "temporal_correlation_by_time_block.png",
    )

    # ========================================================
    # Print
    # ========================================================

    print()
    print("=" * 72)
    print(
        "MAGE-STYLE ITERATIVE DECODING"
    )

    print(
        "Samples:",
        evaluated_samples
    )

    print(
        "Iterations:",
        args.num_iterations
    )

    print(
        "Choice temperature:",
        args.choice_temperature
    )

    print(
        "Mask schedule:",
        first_schedule
    )

    print(
        "CE:",
        f"{global_metrics['cross_entropy']:.4f}"
    )

    print(
        "Token accuracy:",
        f"{global_metrics['token_accuracy']:.4f}"
    )

    print(
        "Top-5 accuracy:",
        f"{global_metrics['token_top5_accuracy']:.4f}"
    )

    print(
        "EEG MAE:",
        f"{global_metrics['predicted_eeg_mae']:.4f}"
    )

    print(
        "EEG correlation:",
        f"{global_metrics['predicted_eeg_correlation']:.4f}"
    )

    print(
        "Temporal correlation:",
        f"{global_metrics['predicted_eeg_temporal_correlation']:.4f}"
    )

    print(
        "SNR:",
        f"{global_metrics['predicted_eeg_snr_db']:.2f} dB"
    )

    print(
        "Predicted codes:",
        global_metrics[
            "unique_predicted_codes"
        ]
    )

    print(
        "Target codes:",
        global_metrics[
            "unique_target_codes"
        ]
    )

    print()
    print(
        "ORACLE VQGAN"
    )

    print(
        "Oracle MAE:",
        f"{global_metrics['oracle_vqgan_mae']:.4f}"
    )

    print(
        "Oracle correlation:",
        f"{global_metrics['oracle_vqgan_correlation']:.4f}"
    )

    print(
        "Oracle temporal correlation:",
        f"{global_metrics['oracle_vqgan_temporal_correlation']:.4f}"
    )

    print()
    print(
        "Metrics by window:"
    )

    for (
        window_name,
        values,
    ) in metrics_by_window.items():

        print(
            f"  {window_name}"
            f" | start="
            f"{values['start_min']:.0f} min"
            f" | n="
            f"{values['evaluated_samples']}"
            f" | CE="
            f"{values['cross_entropy']:.4f}"
            f" | acc="
            f"{values['token_accuracy']:.4f}"
            f" | MAE="
            f"{values['predicted_eeg_mae']:.4f}"
            f" | temporal_corr="
            f"{values['predicted_eeg_temporal_correlation']:.4f}"
            f" | SNR="
            f"{values['predicted_eeg_snr_db']:.2f} dB"
            f" | predicted_codes="
            f"{values['unique_predicted_codes']}"
        )

    print()
    print(
        "Metrics saved:",
        metrics_path
    )


if __name__ == "__main__":
    main()
