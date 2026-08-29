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
from torch.utils.data import DataLoader

from src.data.dataset import PhysiologyPairDataset
from src.models.vqgan import VQGAN

from scripts.evaluate_masked_transformer import (
    decode_token_ids,
    load_state_dict,
    sample_correlation,
    sample_snr,
    temporal_correlation,
)


DEFAULT_DATA_ROOT = Path(
    "outputs/shhs_preprocessed"
)

DEFAULT_VQGAN_CHECKPOINT = Path(
    "outputs/vqgan_adversarial_w0001/"
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

    print("Training folds:", train_folds)
    print(
        "Training samples:",
        len(train_dataset),
    )
    print(
        "Validation samples used:",
        args.max_val_samples,
    )

    (
        position_counts,
        processed_training_samples,
    ) = build_position_counts(
        vqgan=vqgan,
        loader=train_loader,
        device=device,
        vocabulary_size=8192,
        max_samples=(
            args.max_train_samples
        ),
    )

    statistics = (
        calculate_codebook_statistics(
            position_counts
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
    )

    output_directory = (
        args.output_dir
        / f"held_out_fold_{args.held_out_fold}"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
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
    print("Saved:", metrics_path)


if __name__ == "__main__":
    main()