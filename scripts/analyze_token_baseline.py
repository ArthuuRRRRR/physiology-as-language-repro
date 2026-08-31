import argparse
import json
import math
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
from src.models.masked_transformer import (
    RespirationToEEGTransformer,
)
from src.models.mage_transformer import (
    MAGERespirationToEEGTransformer,
)
from src.models.vqgan import VQGAN

from scripts.evaluate_masked_transformer import (
    ShuffledRespirationDataset,
    decode_token_ids,
    load_state_dict,
    sample_correlation,
    sample_snr,
    standardize_respiration,
    temporal_correlation,
)


DEFAULT_DATA_ROOT = Path(
    "outputs/shhs_preprocessed"
)

DEFAULT_VQGAN_CHECKPOINT = Path(
    "outputs/vqgan_adversarial_w0001/"
    "held_out_fold_0/checkpoint_best.pt"
)

DEFAULT_TRANSFORMER_CHECKPOINT = Path(
    "outputs/masked_transformer_joint_long/"
    "held_out_fold_0/checkpoint_best.pt"
)

DEFAULT_OUTPUT_DIR = Path(
    "outputs/token_majority_baseline"
)


def load_vqgan(
    checkpoint_path,
    device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model = VQGAN().to(device)

    model.load_state_dict(
        load_state_dict(
            checkpoint,
            possible_keys=[
                "model_state_dict",
                "vqgan_state_dict",
                "generator_state_dict",
            ],
        ),
        strict=True,
    )

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    return model, checkpoint


def load_transformer(
    checkpoint_path,
    device,
):
    """Load either supported respiration-to-EEG Transformer."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    architecture = checkpoint.get(
        "architecture",
        "joint_concatenated_transformer",
    )

    if architecture == "mage_visible_encoder_decoder":
        model = MAGERespirationToEEGTransformer()

    elif architecture in (
        None,
        "joint_concatenated_transformer",
    ):
        model = RespirationToEEGTransformer()

    else:
        raise ValueError(
            "Unsupported Transformer architecture: "
            f"{architecture}"
        )

    model.load_state_dict(
        load_state_dict(
            checkpoint,
            possible_keys=[
                "model_state_dict",
                "transformer_state_dict",
            ],
        ),
        strict=True,
    )

    model = model.to(device)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    return model, checkpoint


def create_semantic_accumulator():
    """Create storage for codebook-vector distance values."""

    return {
        "positions": 0,
        "exact_matches": 0,
        "cosine_chunks": [],
        "l2_chunks": [],
        "incorrect_cosine_chunks": [],
        "incorrect_l2_chunks": [],
    }


def update_semantic_accumulator(
    accumulator,
    target_tokens,
    predicted_tokens,
    normalized_codebook,
):
    """Accumulate cosine similarity and L2 codebook distance."""

    target_ids = target_tokens.reshape(-1).long()
    predicted_ids = predicted_tokens.reshape(-1).long()

    if target_ids.shape != predicted_ids.shape:
        raise ValueError(
            "Target and predicted token shapes differ: "
            f"{target_ids.shape} vs {predicted_ids.shape}"
        )

    target_ids = target_ids.to(
        normalized_codebook.device
    )
    predicted_ids = predicted_ids.to(
        normalized_codebook.device
    )

    target_vectors = F.embedding(
        target_ids,
        normalized_codebook,
    )
    predicted_vectors = F.embedding(
        predicted_ids,
        normalized_codebook,
    )

    cosine = (
        target_vectors
        * predicted_vectors
    ).sum(dim=-1).clamp(-1.0, 1.0)

    l2_distance = torch.linalg.vector_norm(
        target_vectors - predicted_vectors,
        ord=2,
        dim=-1,
    )

    exact = target_ids == predicted_ids
    incorrect = ~exact

    accumulator["positions"] += target_ids.numel()
    accumulator["exact_matches"] += int(
        exact.sum().item()
    )

    accumulator["cosine_chunks"].append(
        cosine.detach().cpu()
    )
    accumulator["l2_chunks"].append(
        l2_distance.detach().cpu()
    )

    if incorrect.any():
        accumulator[
            "incorrect_cosine_chunks"
        ].append(
            cosine[incorrect].detach().cpu()
        )
        accumulator[
            "incorrect_l2_chunks"
        ].append(
            l2_distance[incorrect].detach().cpu()
        )


def distribution_summary(values):
    """Return JSON-safe summary statistics for one tensor."""

    if values.numel() < 1:
        return None

    values = values.float()
    quantiles = torch.quantile(
        values,
        torch.tensor(
            [0.05, 0.25, 0.50, 0.75, 0.95],
            dtype=values.dtype,
        ),
    )

    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "std": float(
            values.std(unbiased=False)
        ),
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
    }


def finalize_semantic_accumulator(
    accumulator,
):
    """Finalize semantic codebook-distance metrics."""

    if accumulator["positions"] < 1:
        raise ValueError(
            "Cannot finalize empty semantic metrics"
        )

    cosine = torch.cat(
        accumulator["cosine_chunks"]
    )
    l2_distance = torch.cat(
        accumulator["l2_chunks"]
    )

    if accumulator[
        "incorrect_cosine_chunks"
    ]:
        incorrect_cosine = torch.cat(
            accumulator[
                "incorrect_cosine_chunks"
            ]
        )
        incorrect_l2 = torch.cat(
            accumulator[
                "incorrect_l2_chunks"
            ]
        )

    else:
        incorrect_cosine = torch.empty(0)
        incorrect_l2 = torch.empty(0)

    return {
        "positions": accumulator["positions"],
        "exact_token_accuracy": (
            accumulator["exact_matches"]
            / accumulator["positions"]
        ),
        "cosine_similarity": (
            distribution_summary(cosine)
        ),
        "l2_distance": (
            distribution_summary(l2_distance)
        ),
        "incorrect_tokens_only": {
            "cosine_similarity": (
                distribution_summary(
                    incorrect_cosine
                )
            ),
            "l2_distance": (
                distribution_summary(
                    incorrect_l2
                )
            ),
        },
    }


def print_semantic_metrics(
    condition_name,
    metrics,
):
    """Print the decision metrics for one prediction condition."""

    cosine = metrics["cosine_similarity"]
    l2_distance = metrics["l2_distance"]
    incorrect_cosine = metrics[
        "incorrect_tokens_only"
    ]["cosine_similarity"]

    print(f"  {condition_name}")
    print(
        "    exact_accuracy=",
        f"{metrics['exact_token_accuracy']:.4f}",
        sep="",
    )
    print(
        "    cosine_mean=",
        f"{cosine['mean']:.4f}",
        " | cosine_median=",
        f"{cosine['median']:.4f}",
        " | cosine_p05=",
        f"{cosine['p05']:.4f}",
        " | cosine_p95=",
        f"{cosine['p95']:.4f}",
        sep="",
    )
    print(
        "    incorrect_only_cosine_mean=",
        f"{incorrect_cosine['mean']:.4f}",
        " | incorrect_only_cosine_median=",
        f"{incorrect_cosine['median']:.4f}",
        sep="",
    )
    print(
        "    l2_mean=",
        f"{l2_distance['mean']:.4f}",
        " | l2_median=",
        f"{l2_distance['median']:.4f}",
        sep="",
    )


def evaluate_transformer_semantics(
    transformer,
    vqgan,
    loader,
    device,
    normalized_codebook,
    max_samples,
    condition_name,
):
    """Evaluate full-mask Transformer predictions in VQ space."""

    accumulator = create_semantic_accumulator()
    evaluated_samples = 0

    with torch.inference_mode():
        for batch in loader:
            if evaluated_samples >= max_samples:
                break

            remaining = max_samples - evaluated_samples

            respiration = batch["respiration"][
                :remaining
            ].to(
                device,
                non_blocking=True,
            )

            eeg = (
                batch["eeg_spectrogram"][:remaining]
                .unsqueeze(1)
                .to(
                    device,
                    non_blocking=True,
                )
            )

            respiration = standardize_respiration(
                respiration
            )

            _, true_tokens, _ = vqgan.encode(eeg)

            masked_tokens = torch.full_like(
                true_tokens,
                fill_value=(
                    transformer.mask_token_id
                ),
            )

            logits = transformer(
                respiration,
                masked_tokens,
            )

            predicted_tokens = (
                logits.argmax(dim=-1)
                .reshape_as(true_tokens)
            )

            update_semantic_accumulator(
                accumulator=accumulator,
                target_tokens=true_tokens,
                predicted_tokens=(
                    predicted_tokens
                ),
                normalized_codebook=(
                    normalized_codebook
                ),
            )

            evaluated_samples += eeg.shape[0]

            if evaluated_samples % 25 == 0:
                print(
                    f"{condition_name} samples evaluated:",
                    evaluated_samples,
                    flush=True,
                )

    metrics = finalize_semantic_accumulator(
        accumulator
    )
    metrics["evaluated_samples"] = evaluated_samples

    return metrics


def evaluate_fixed_token_semantics(
    vqgan,
    loader,
    predicted_token_grid,
    device,
    normalized_codebook,
    max_samples,
    condition_name,
):
    """Evaluate one fixed 8 x 64 token template in VQ space."""

    predicted_token_grid = (
        predicted_token_grid
        .reshape(1, 8, 64)
        .long()
        .to(device)
    )

    accumulator = create_semantic_accumulator()
    evaluated_samples = 0

    with torch.inference_mode():
        for batch in loader:
            if evaluated_samples >= max_samples:
                break

            remaining = max_samples - evaluated_samples

            eeg = (
                batch["eeg_spectrogram"][:remaining]
                .unsqueeze(1)
                .to(
                    device,
                    non_blocking=True,
                )
            )

            _, true_tokens, _ = vqgan.encode(eeg)

            predicted_tokens = (
                predicted_token_grid.expand(
                    true_tokens.shape[0],
                    -1,
                    -1,
                )
            )

            update_semantic_accumulator(
                accumulator=accumulator,
                target_tokens=true_tokens,
                predicted_tokens=(
                    predicted_tokens
                ),
                normalized_codebook=(
                    normalized_codebook
                ),
            )

            evaluated_samples += eeg.shape[0]

            if evaluated_samples % 25 == 0:
                print(
                    f"{condition_name} samples evaluated:",
                    evaluated_samples,
                    flush=True,
                )

    metrics = finalize_semantic_accumulator(
        accumulator
    )
    metrics["evaluated_samples"] = evaluated_samples

    return metrics


def build_position_counts(
    vqgan,
    loader,
    device,
    vocabulary_size,
    max_samples=None,
):
    """
    Count training token occurrences separately
    for each of the 512 EEG positions.
    """

    position_counts = torch.zeros(
        512,
        vocabulary_size,
        dtype=torch.int64,
    )

    processed_samples = 0

    with torch.inference_mode():
        for batch_index, batch in enumerate(
            loader,
            start=1,
        ):
            eeg = (
                batch["eeg_spectrogram"]
                .unsqueeze(1)
            )

            if max_samples is not None:
                remaining = (
                    max_samples
                    - processed_samples
                )

                if remaining <= 0:
                    break

                eeg = eeg[:remaining]

            eeg = eeg.to(
                device,
                non_blocking=True,
            )

            _, tokens, _ = vqgan.encode(eeg)

            tokens = (
                tokens
                .flatten(start_dim=1)
                .detach()
                .cpu()
                .long()
            )

            batch_size = tokens.shape[0]
            number_of_positions = tokens.shape[1]

            positions = (
                torch.arange(
                    number_of_positions,
                    dtype=torch.long,
                )
                .unsqueeze(0)
                .expand(batch_size, -1)
                .reshape(-1)
            )

            token_ids = tokens.reshape(-1)

            increments = torch.ones(
                token_ids.numel(),
                dtype=torch.int64,
            )

            position_counts.index_put_(
                (
                    positions,
                    token_ids,
                ),
                increments,
                accumulate=True,
            )

            processed_samples += batch_size

            if batch_index % 250 == 0:
                print(
                    "Training samples processed:",
                    processed_samples,
                    flush=True,
                )

    return position_counts, processed_samples


def entropy_from_counts(counts):
    """
    Shannon entropy in nats.
    """

    counts = counts.double()

    total = counts.sum()

    if total <= 0:
        return 0.0

    probabilities = counts / total

    probabilities = probabilities[
        probabilities > 0
    ]

    return float(
        -(
            probabilities
            * torch.log(probabilities)
        ).sum()
    )


def calculate_codebook_statistics(
    position_counts,
):
    global_counts = position_counts.sum(
        dim=0
    )

    active_codes = int(
        (global_counts > 0).sum()
    )

    global_entropy = entropy_from_counts(
        global_counts
    )

    global_perplexity = math.exp(
        global_entropy
    )

    position_entropies = []

    for position_index in range(
        position_counts.shape[0]
    ):
        position_entropies.append(
            entropy_from_counts(
                position_counts[
                    position_index
                ]
            )
        )

    mean_position_entropy = float(
        np.mean(position_entropies)
    )

    mean_position_perplexity = float(
        np.mean(
            np.exp(position_entropies)
        )
    )

    top_values, top_indices = (
        global_counts.topk(
            k=20
        )
    )

    total_tokens = int(
        global_counts.sum()
    )

    top_codes = []

    for code, count in zip(
        top_indices.tolist(),
        top_values.tolist(),
    ):
        top_codes.append(
            {
                "code": int(code),
                "count": int(count),
                "fraction": (
                    float(count)
                    / total_tokens
                ),
            }
        )

    return {
        "global_counts": global_counts,
        "active_codes": active_codes,
        "inactive_codes": (
            position_counts.shape[1]
            - active_codes
        ),
        "global_entropy": global_entropy,
        "global_perplexity": (
            global_perplexity
        ),
        "mean_position_entropy": (
            mean_position_entropy
        ),
        "mean_position_perplexity": (
            mean_position_perplexity
        ),
        "top_codes": top_codes,
    }


def save_comparison(
    ground_truth,
    oracle_reconstruction,
    majority_reconstruction,
    output_path,
):
    ground_truth = (
        ground_truth[0, 0]
        .detach()
        .cpu()
        .numpy()
    )

    oracle_reconstruction = (
        oracle_reconstruction[0, 0]
        .detach()
        .cpu()
        .numpy()
    )

    majority_reconstruction = (
        majority_reconstruction[0, 0]
        .detach()
        .cpu()
        .numpy()
    )

    difference = np.abs(
        ground_truth
        - majority_reconstruction
    )

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(14, 12),
    )

    images = [
        ground_truth,
        oracle_reconstruction,
        majority_reconstruction,
        difference,
    ]

    titles = [
        "Ground-truth EEG spectrogram",
        (
            "Frozen VQGAN reconstruction "
            "from true EEG tokens"
        ),
        (
            "Position-wise majority-token "
            "baseline"
        ),
        (
            "Absolute difference: "
            "ground truth vs majority baseline"
        ),
    ]

    for index, axis in enumerate(axes):
        image_kwargs = {
            "aspect": "auto",
            "origin": "lower",
        }

        if index < 3:
            image_kwargs["vmin"] = 0
            image_kwargs["vmax"] = 1

        axis.imshow(
            images[index],
            **image_kwargs,
        )

        axis.set_title(titles[index])
        axis.set_xlabel(
            "Time (30-second epochs)"
        )
        axis.set_ylabel(
            "Frequency bins"
        )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


def evaluate_majority_baseline(
    vqgan,
    validation_loader,
    position_counts,
    device,
    max_samples,
    normalized_codebook,
):
    majority_tokens_flat = (
        position_counts.argmax(
            dim=1
        )
    )

    top5_tokens = (
        position_counts.topk(
            k=5,
            dim=1,
        ).indices
    )

    global_counts = position_counts.sum(
        dim=0
    )

    position_totals = position_counts.sum(
        dim=1
    )

    vocabulary_size = (
        position_counts.shape[1]
    )

    total_correct = 0
    total_top5 = 0
    total_positions = 0
    total_laplace_nll = 0.0

    global_unseen = 0
    position_unseen = 0

    total_mae = 0.0
    total_correlation = 0.0
    total_temporal_correlation = 0.0
    total_snr = 0.0

    total_oracle_mae = 0.0
    total_oracle_correlation = 0.0
    total_oracle_temporal_correlation = 0.0

    validation_codes = set()

    evaluated_samples = 0
    comparison_saved = False
    semantic_accumulator = (
        create_semantic_accumulator()
    )

    output_example = None

    with torch.inference_mode():
        for batch in validation_loader:
            if evaluated_samples >= max_samples:
                break

            remaining = (
                max_samples
                - evaluated_samples
            )

            eeg = (
                batch["eeg_spectrogram"]
                .unsqueeze(1)
            )

            eeg = eeg[:remaining].to(
                device,
                non_blocking=True,
            )

            _, true_tokens, _ = vqgan.encode(
                eeg
            )

            true_flat_cpu = (
                true_tokens
                .flatten(start_dim=1)
                .detach()
                .cpu()
                .long()
            )

            batch_size = (
                true_flat_cpu.shape[0]
            )

            predicted_flat_cpu = (
                majority_tokens_flat
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

            predicted_tokens = (
                predicted_flat_cpu
                .reshape(
                    batch_size,
                    8,
                    64,
                )
                .to(device)
            )

            update_semantic_accumulator(
                accumulator=(
                    semantic_accumulator
                ),
                target_tokens=true_tokens,
                predicted_tokens=(
                    predicted_tokens
                ),
                normalized_codebook=(
                    normalized_codebook
                ),
            )

            oracle_reconstruction = (
                decode_token_ids(
                    vqgan,
                    true_tokens,
                )
            )

            majority_reconstruction = (
                decode_token_ids(
                    vqgan,
                    predicted_tokens,
                )
            )

            positions = (
                torch.arange(512)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

            target_counts = position_counts[
                positions,
                true_flat_cpu,
            ]

            smoothed_probabilities = (
                target_counts.double() + 1.0
            ) / (
                position_totals
                .unsqueeze(0)
                .double()
                + vocabulary_size
            )

            total_laplace_nll += float(
                -torch.log(
                    smoothed_probabilities
                ).sum()
            )

            total_correct += int(
                (
                    predicted_flat_cpu
                    == true_flat_cpu
                ).sum()
            )

            total_top5 += int(
                (
                    top5_tokens
                    .unsqueeze(0)
                    == true_flat_cpu.unsqueeze(-1)
                )
                .any(dim=-1)
                .sum()
            )

            total_positions += (
                true_flat_cpu.numel()
            )

            global_unseen += int(
                (
                    global_counts[
                        true_flat_cpu
                    ]
                    == 0
                ).sum()
            )

            position_unseen += int(
                (target_counts == 0).sum()
            )

            total_mae += (
                torch.abs(
                    eeg
                    - majority_reconstruction
                )
                .flatten(start_dim=1)
                .mean(dim=1)
                .sum()
                .item()
            )

            total_correlation += (
                sample_correlation(
                    eeg,
                    majority_reconstruction,
                )
                .sum()
                .item()
            )

            total_temporal_correlation += (
                temporal_correlation(
                    eeg,
                    majority_reconstruction,
                )
                .sum()
                .item()
            )

            total_snr += (
                sample_snr(
                    eeg,
                    majority_reconstruction,
                )
                .sum()
                .item()
            )

            total_oracle_mae += (
                torch.abs(
                    eeg
                    - oracle_reconstruction
                )
                .flatten(start_dim=1)
                .mean(dim=1)
                .sum()
                .item()
            )

            total_oracle_correlation += (
                sample_correlation(
                    eeg,
                    oracle_reconstruction,
                )
                .sum()
                .item()
            )

            total_oracle_temporal_correlation += (
                temporal_correlation(
                    eeg,
                    oracle_reconstruction,
                )
                .sum()
                .item()
            )

            validation_codes.update(
                true_flat_cpu
                .reshape(-1)
                .tolist()
            )

            if not comparison_saved:
                output_example = {
                    "ground_truth": eeg,
                    "oracle": (
                        oracle_reconstruction
                    ),
                    "majority": (
                        majority_reconstruction
                    ),
                }

                comparison_saved = True

            evaluated_samples += batch_size

            if evaluated_samples % 25 == 0:
                print(
                    "Validation samples evaluated:",
                    evaluated_samples,
                )

    metrics = {
        "evaluated_samples": (
            evaluated_samples
        ),
        "laplace_smoothed_nll": (
            total_laplace_nll
            / total_positions
        ),
        "token_accuracy": (
            total_correct
            / total_positions
        ),
        "token_top5_accuracy": (
            total_top5
            / total_positions
        ),
        "predicted_eeg_mae": (
            total_mae
            / evaluated_samples
        ),
        "predicted_eeg_correlation": (
            total_correlation
            / evaluated_samples
        ),
        "predicted_eeg_temporal_correlation": (
            total_temporal_correlation
            / evaluated_samples
        ),
        "predicted_eeg_snr_db": (
            total_snr
            / evaluated_samples
        ),
        "oracle_vqgan_mae": (
            total_oracle_mae
            / evaluated_samples
        ),
        "oracle_vqgan_correlation": (
            total_oracle_correlation
            / evaluated_samples
        ),
        "oracle_vqgan_temporal_correlation": (
            total_oracle_temporal_correlation
            / evaluated_samples
        ),
        "global_unseen_token_rate": (
            global_unseen
            / total_positions
        ),
        "position_unseen_token_rate": (
            position_unseen
            / total_positions
        ),
        "unique_validation_codes": len(
            validation_codes
        ),
        "unique_majority_codes": int(
            torch.unique(
                majority_tokens_flat
            ).numel()
        ),
        "semantic_codebook_distance": (
            finalize_semantic_accumulator(
                semantic_accumulator
            )
        ),
    }

    return (
        metrics,
        majority_tokens_flat,
        output_example,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze VQGAN token usage and evaluate "
            "a position-wise majority-token baseline."
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
        "--transformer-checkpoint",
        type=Path,
        default=DEFAULT_TRANSFORMER_CHECKPOINT,
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
        default=0,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--recompute-position-counts",
        action="store_true",
        help=(
            "Ignore a cached training position-count "
            "tensor and recompute it."
        ),
    )

    parser.add_argument(
        "--semantic-only",
        action="store_true",
        help=(
            "Run only codebook-similarity evaluation "
            "and reuse an existing majority baseline."
        ),
    )

    parser.add_argument(
        "--majority-metrics",
        type=Path,
        default=None,
        help=(
            "Existing majority-baseline metrics.json "
            "containing majority_token_grid."
        ),
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    vqgan, checkpoint = load_vqgan(
        args.vqgan_checkpoint,
        device,
    )

    transformer, transformer_checkpoint = (
        load_transformer(
            args.transformer_checkpoint,
            device,
        )
    )

    transformer_fold = int(
        transformer_checkpoint["held_out_fold"]
    )

    if transformer_fold != args.held_out_fold:
        raise ValueError(
            "Transformer held-out fold does not match "
            f"--held-out-fold: {transformer_fold} vs "
            f"{args.held_out_fold}"
        )

    normalized_codebook = F.normalize(
        vqgan.quantizer.codebook.weight.detach(),
        p=2,
        dim=1,
    )

    min_db = float(
        checkpoint["min_db"]
    )

    max_db = float(
        checkpoint["max_db"]
    )

    train_folds = [
        fold
        for fold in range(4)
        if fold != args.held_out_fold
    ]

    train_directories = [
        args.data_root
        / f"fold_{fold}"
        for fold in train_folds
    ]

    validation_directory = (
        args.data_root
        / f"fold_{args.held_out_fold}"
    )

    train_dataset = PhysiologyPairDataset(
        train_directories,
        min_db=min_db,
        max_db=max_db,
    )

    validation_dataset = (
        PhysiologyPairDataset(
            validation_directory,
            min_db=min_db,
            max_db=max_db,
        )
    )

    output_directory = (
        args.output_dir
        / f"held_out_fold_{args.held_out_fold}"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    shuffled_validation_dataset = (
        ShuffledRespirationDataset(
            base_dataset=validation_dataset,
            number_of_samples=min(
                args.max_val_samples,
                len(validation_dataset),
            ),
        )
    )

    shuffled_validation_loader = DataLoader(
        shuffled_validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    print("Training folds:", train_folds)
    print(
        "Training samples:",
        len(train_dataset),
    )
    print(
        "Validation samples used:",
        args.max_val_samples,
    )

    if args.semantic_only:
        majority_metrics_path = (
            args.majority_metrics
            if args.majority_metrics is not None
            else (
                DEFAULT_OUTPUT_DIR
                / (
                    "held_out_fold_"
                    f"{args.held_out_fold}"
                )
                / "metrics.json"
            )
        )

        if not majority_metrics_path.exists():
            raise FileNotFoundError(
                "Existing majority-baseline metrics "
                "not found: "
                f"{majority_metrics_path}. Run once "
                "without --semantic-only or provide "
                "--majority-metrics."
            )

        with majority_metrics_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            majority_summary = json.load(file)

        if "majority_token_grid" not in (
            majority_summary
        ):
            raise KeyError(
                "majority_token_grid is missing from "
                f"{majority_metrics_path}"
            )

        majority_token_grid = torch.tensor(
            majority_summary[
                "majority_token_grid"
            ],
            dtype=torch.long,
        )

        correct_semantic_metrics = (
            evaluate_transformer_semantics(
                transformer=transformer,
                vqgan=vqgan,
                loader=validation_loader,
                device=device,
                normalized_codebook=(
                    normalized_codebook
                ),
                max_samples=args.max_val_samples,
                condition_name=(
                    "Correct respiration"
                ),
            )
        )

        shuffled_semantic_metrics = (
            evaluate_transformer_semantics(
                transformer=transformer,
                vqgan=vqgan,
                loader=shuffled_validation_loader,
                device=device,
                normalized_codebook=(
                    normalized_codebook
                ),
                max_samples=args.max_val_samples,
                condition_name=(
                    "Shuffled respiration"
                ),
            )
        )

        majority_semantic_metrics = (
            evaluate_fixed_token_semantics(
                vqgan=vqgan,
                loader=validation_loader,
                predicted_token_grid=(
                    majority_token_grid
                ),
                device=device,
                normalized_codebook=(
                    normalized_codebook
                ),
                max_samples=args.max_val_samples,
                condition_name=(
                    "Position-majority baseline"
                ),
            )
        )

        semantic_summary = {
            "transformer_checkpoint": str(
                args.transformer_checkpoint
            ),
            "vqgan_checkpoint": str(
                args.vqgan_checkpoint
            ),
            "majority_metrics_source": str(
                majority_metrics_path
            ),
            "held_out_fold": args.held_out_fold,
            "max_validation_samples": (
                args.max_val_samples
            ),
            "semantic_codebook_similarity": {
                "correct_respiration": (
                    correct_semantic_metrics
                ),
                "shuffled_respiration": (
                    shuffled_semantic_metrics
                ),
                "position_majority_baseline": (
                    majority_semantic_metrics
                ),
            },
        }

        semantic_metrics_path = (
            output_directory
            / "semantic_similarity.json"
        )

        with semantic_metrics_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                semantic_summary,
                file,
                indent=2,
            )

        print()
        print("Semantic VQ codebook similarity:")
        print_semantic_metrics(
            "Correct respiration",
            correct_semantic_metrics,
        )
        print_semantic_metrics(
            "Shuffled respiration",
            shuffled_semantic_metrics,
        )
        print_semantic_metrics(
            "Position-majority baseline",
            majority_semantic_metrics,
        )
        print("Saved:", semantic_metrics_path)

        return

    position_counts_path = (
        output_directory
        / "training_position_counts.pt"
    )

    if (
        position_counts_path.exists()
        and not args.recompute_position_counts
        and args.max_train_samples is None
    ):
        counts_payload = torch.load(
            position_counts_path,
            map_location="cpu",
            weights_only=False,
        )

        position_counts = counts_payload[
            "position_counts"
        ]
        processed_training_samples = int(
            counts_payload[
                "processed_training_samples"
            ]
        )

        if int(
            counts_payload["held_out_fold"]
        ) != args.held_out_fold:
            raise ValueError(
                "Cached position counts belong to a "
                "different held-out fold"
            )

        if (
            position_counts.shape[1]
            != normalized_codebook.shape[0]
        ):
            raise ValueError(
                "Cached position-count vocabulary does "
                "not match the current VQGAN codebook"
            )

        print(
            "Loaded cached training position counts:",
            position_counts_path,
        )

    else:
        (
            position_counts,
            processed_training_samples,
        ) = build_position_counts(
            vqgan=vqgan,
            loader=train_loader,
            device=device,
            vocabulary_size=(
                normalized_codebook.shape[0]
            ),
            max_samples=(
                args.max_train_samples
            ),
        )

        if args.max_train_samples is None:
            torch.save(
                {
                    "held_out_fold": (
                        args.held_out_fold
                    ),
                    "processed_training_samples": (
                        processed_training_samples
                    ),
                    "position_counts": (
                        position_counts
                    ),
                },
                position_counts_path,
            )

    statistics = (
        calculate_codebook_statistics(
            position_counts
        )
    )

    correct_semantic_metrics = (
        evaluate_transformer_semantics(
            transformer=transformer,
            vqgan=vqgan,
            loader=validation_loader,
            device=device,
            normalized_codebook=(
                normalized_codebook
            ),
            max_samples=args.max_val_samples,
            condition_name=(
                "Correct respiration"
            ),
        )
    )

    shuffled_semantic_metrics = (
        evaluate_transformer_semantics(
            transformer=transformer,
            vqgan=vqgan,
            loader=shuffled_validation_loader,
            device=device,
            normalized_codebook=(
                normalized_codebook
            ),
            max_samples=args.max_val_samples,
            condition_name=(
                "Shuffled respiration"
            ),
        )
    )

    (
        baseline_metrics,
        majority_tokens,
        example,
    ) = evaluate_majority_baseline(
        vqgan=vqgan,
        validation_loader=(
            validation_loader
        ),
        position_counts=position_counts,
        device=device,
        max_samples=(
            args.max_val_samples
        ),
        normalized_codebook=(
            normalized_codebook
        ),
    )

    save_comparison(
        ground_truth=(
            example["ground_truth"]
        ),
        oracle_reconstruction=(
            example["oracle"]
        ),
        majority_reconstruction=(
            example["majority"]
        ),
        output_path=(
            output_directory
            / "majority_baseline.png"
        ),
    )

    summary = {
        "transformer_checkpoint": str(
            args.transformer_checkpoint
        ),
        "vqgan_checkpoint": str(
            args.vqgan_checkpoint
        ),
        "transformer_architecture": (
            transformer_checkpoint.get(
                "architecture",
                "joint_concatenated_transformer",
            )
        ),
        "held_out_fold": (
            args.held_out_fold
        ),
        "train_folds": train_folds,
        "processed_training_samples": (
            processed_training_samples
        ),
        "training_token_statistics": {
            "active_codes": (
                statistics["active_codes"]
            ),
            "inactive_codes": (
                statistics["inactive_codes"]
            ),
            "global_entropy": (
                statistics["global_entropy"]
            ),
            "global_perplexity": (
                statistics["global_perplexity"]
            ),
            "mean_position_entropy": (
                statistics[
                    "mean_position_entropy"
                ]
            ),
            "mean_position_perplexity": (
                statistics[
                    "mean_position_perplexity"
                ]
            ),
            "top_codes": (
                statistics["top_codes"]
            ),
        },
        "majority_baseline": (
            baseline_metrics
        ),
        "majority_token_grid": (
            majority_tokens
            .reshape(8, 64)
            .tolist()
        ),
        "semantic_codebook_similarity": {
            "correct_respiration": (
                correct_semantic_metrics
            ),
            "shuffled_respiration": (
                shuffled_semantic_metrics
            ),
            "position_majority_baseline": (
                baseline_metrics[
                    "semantic_codebook_distance"
                ]
            ),
        },
    }

    metrics_path = (
        output_directory
        / "metrics.json"
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print()
    print(
        "Processed training samples:",
        processed_training_samples,
    )
    print(
        "Active training codes:",
        statistics["active_codes"],
    )
    print(
        "Inactive training codes:",
        statistics["inactive_codes"],
    )
    print(
        "Global token perplexity:",
        f"{statistics['global_perplexity']:.2f}",
    )
    print(
        "Mean position perplexity:",
        (
            f"{statistics['mean_position_perplexity']:.2f}"
        ),
    )
    print(
        "Majority token accuracy:",
        (
            f"{baseline_metrics['token_accuracy']:.4f}"
        ),
    )
    print(
        "Majority top-5 accuracy:",
        (
            f"{baseline_metrics['token_top5_accuracy']:.4f}"
        ),
    )
    print(
        "Majority EEG MAE:",
        (
            f"{baseline_metrics['predicted_eeg_mae']:.4f}"
        ),
    )
    print(
        "Majority temporal correlation:",
        (
            f"{baseline_metrics['predicted_eeg_temporal_correlation']:.4f}"
        ),
    )
    print(
        "Majority EEG SNR:",
        (
            f"{baseline_metrics['predicted_eeg_snr_db']:.2f} dB"
        ),
    )
    print(
        "Global unseen validation rate:",
        (
            f"{baseline_metrics['global_unseen_token_rate']:.6f}"
        ),
    )
    print(
        "Position unseen validation rate:",
        (
            f"{baseline_metrics['position_unseen_token_rate']:.6f}"
        ),
    )
    print(
        "Unique validation codes:",
        (
            baseline_metrics["unique_validation_codes"]
        ),
    )
    print(
        "Unique majority codes:",
        (
            baseline_metrics["unique_majority_codes"]
        ),
    )

    print()
    print("Semantic VQ codebook similarity:")
    print_semantic_metrics(
        "Correct respiration",
        correct_semantic_metrics,
    )
    print_semantic_metrics(
        "Shuffled respiration",
        shuffled_semantic_metrics,
    )
    print_semantic_metrics(
        "Position-majority baseline",
        baseline_metrics[
            "semantic_codebook_distance"
        ],
    )
    print("Saved:", metrics_path)


if __name__ == "__main__":
    main()
