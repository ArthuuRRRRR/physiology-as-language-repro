"""Evaluate respiration-to-EEG prediction and temporal-window effects."""

import argparse
import itertools
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import (
    DataLoader,
    Dataset,
)

from src.data.dataset import PhysiologyPairDataset
from src.models.masked_transformer import (
    RespirationToEEGTransformer,
)
from src.models.mage_transformer import (
    MAGERespirationToEEGTransformer,
)
from src.models.vqgan import VQGAN


DEFAULT_TRANSFORMER_CHECKPOINT = Path(
    "outputs/masked_transformer_joint_long/"
    "held_out_fold_0/checkpoint_best.pt"
)

DEFAULT_VQGAN_CHECKPOINT = Path(
    "outputs/vqgan_adversarial_w0001/"
    "held_out_fold_0/checkpoint_best.pt"
)

DEFAULT_DATA_ROOT = Path(
    "outputs/shhs_preprocessed"
)

MODEL_WINDOW_SEC = 256 * 60
DIAGNOSTIC_BLOCK_SEC = 32 * 60


class ShuffledRespirationDataset(Dataset):
    """
    Replace each target's respiration with respiration
    from another evaluated participant at the same
    temporal window offset.

    A deterministic circular permutation is selected
    independently inside each start_sec group.
    """

    def __init__(
        self,
        base_dataset,
        number_of_samples,
    ):
        self.base_dataset = base_dataset

        self.number_of_samples = min(
            number_of_samples,
            len(base_dataset),
        )

        if self.number_of_samples < 2:
            raise ValueError(
                "At least two samples are required "
                "for shuffled respiration"
            )

        self.target_indices = list(
            range(self.number_of_samples)
        )

        patient_ids = []
        start_seconds = []

        for index in self.target_indices:
            sample = base_dataset[index]

            if "patient_id" not in sample:
                raise KeyError(
                    "patient_id is required for "
                    "participant-wise shuffling"
                )

            patient_ids.append(
                str(sample["patient_id"])
            )

            if "start_sec" not in sample:
                raise KeyError(
                    "start_sec is required for "
                    "window-matched shuffling"
                )

            start_seconds.append(
                int(
                    round(
                        float(
                            _metadata_scalar(
                                sample["start_sec"]
                            )
                        )
                    )
                )
            )

        indices_by_start = {}

        for index, start_sec in enumerate(
            start_seconds
        ):
            indices_by_start.setdefault(
                start_sec,
                [],
            ).append(index)

        self.source_indices = [
            None for _ in self.target_indices
        ]
        self.group_shifts = {}

        for start_sec, group_indices in sorted(
            indices_by_start.items()
        ):
            if len(group_indices) < 2:
                raise RuntimeError(
                    "At least two samples are required "
                    "for window-matched shuffling at "
                    f"start_sec={start_sec}"
                )

            selected_shift = None

            for shift in range(
                1,
                len(group_indices),
            ):
                valid = all(
                    patient_ids[target_index]
                    != patient_ids[
                        group_indices[
                            (
                                group_position
                                + shift
                            )
                            % len(group_indices)
                        ]
                    ]
                    for group_position, target_index
                    in enumerate(group_indices)
                )

                if valid:
                    selected_shift = shift
                    break

            if selected_shift is None:
                raise RuntimeError(
                    "Could not create a participant-wise "
                    "respiration derangement for "
                    f"start_sec={start_sec}"
                )

            self.group_shifts[start_sec] = (
                selected_shift
            )

            for group_position, target_index in (
                enumerate(group_indices)
            ):
                source_index = group_indices[
                    (
                        group_position
                        + selected_shift
                    )
                    % len(group_indices)
                ]
                self.source_indices[
                    target_index
                ] = source_index

        self.patient_ids = patient_ids
        self.start_seconds = start_seconds

        print("Respiration shifts by window:")

        for start_sec, shift in (
            self.group_shifts.items()
        ):
            print(
                f"  start={start_sec / 60:.0f} min "
                f"| shift={shift} "
                "| samples="
                f"{len(indices_by_start[start_sec])}"
            )

    def __len__(self):
        return self.number_of_samples

    def __getitem__(self, index):
        target_index = self.target_indices[
            index
        ]

        source_index = self.source_indices[
            index
        ]

        target_sample = dict(
            self.base_dataset[target_index]
        )

        source_sample = self.base_dataset[
            source_index
        ]

        target_patient = str(
            target_sample["patient_id"]
        )

        source_patient = str(
            source_sample["patient_id"]
        )

        if target_patient == source_patient:
            raise RuntimeError(
                "Respiration source and EEG target "
                "belong to the same participant"
            )

        target_start = int(
            round(
                float(
                    _metadata_scalar(
                        target_sample["start_sec"]
                    )
                )
            )
        )

        source_start = int(
            round(
                float(
                    _metadata_scalar(
                        source_sample["start_sec"]
                    )
                )
            )
        )

        if target_start != source_start:
            raise RuntimeError(
                "Respiration source and EEG target "
                "do not have the same start_sec: "
                f"{source_start} vs {target_start}"
            )

        target_sample["respiration"] = (
            source_sample["respiration"]
        )

        target_sample[
            "respiration_source_patient_id"
        ] = source_patient

        return target_sample



def _metadata_scalar(value):
    """Convert scalar metadata values to a stable Python value."""

    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(
                "Expected scalar metadata, got tensor with "
                f"shape {tuple(value.shape)}"
            )
        return value.item()

    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(
                "Expected scalar metadata, got array with "
                f"shape {value.shape}"
            )
        return value.reshape(-1)[0].item()

    return value


def _paired_group_key(sample):
    """
    Return a stable recording-level key for paired-window matching.

    We always include patient_id.  If the dataset exposes a visit or
    recording identifier, it is also included so that two windows from
    different SHHS visits cannot accidentally be paired together.
    """

    if "patient_id" not in sample:
        raise KeyError(
            "patient_id is required for paired-window evaluation"
        )

    patient_id = str(
        _metadata_scalar(sample["patient_id"])
    )

    identity_keys = (
        "record_id",
        "recording_id",
        "record_name",
        "visit",
    )

    for key in identity_keys:
        if key in sample:
            value = str(
                _metadata_scalar(sample[key])
            )
            return f"{patient_id}::{key}={value}", key

    return patient_id, "patient_id"


class PairedWindowsDataset(Dataset):
    """
    Select the same recording/participant cohort at two time windows.

    The two window indices with the largest common cohort are chosen.
    For example, if window000 and window001 are both available for the
    largest number of recordings, those are selected.  --max-samples
    remains a cap on total evaluated samples, so 500 samples correspond
    to at most 250 paired recordings (2 windows each).
    """

    def __init__(
        self,
        base_dataset,
        number_of_samples,
    ):
        self.base_dataset = base_dataset

        if number_of_samples < 2:
            raise ValueError(
                "paired-windows requires max-samples >= 2"
            )

        group_to_windows = {}
        grouping_fields = set()

        for index in range(len(base_dataset)):
            sample = base_dataset[index]

            if "start_sec" not in sample:
                raise KeyError(
                    "start_sec is required for paired-window "
                    "evaluation"
                )

            group_key, grouping_field = (
                _paired_group_key(sample)
            )
            grouping_fields.add(grouping_field)

            start_sec = int(
                round(
                    float(
                        _metadata_scalar(
                            sample["start_sec"]
                        )
                    )
                )
            )

            window_index = (
                start_sec // MODEL_WINDOW_SEC
            )

            windows = group_to_windows.setdefault(
                group_key,
                {},
            )

            # Keep the first sample if duplicate metadata maps more
            # than one example to the same recording/window.
            windows.setdefault(window_index, index)

        pair_counts = {}

        for windows in group_to_windows.values():
            available = sorted(windows)

            for pair in itertools.combinations(
                available,
                2,
            ):
                pair_counts[pair] = (
                    pair_counts.get(pair, 0) + 1
                )

        if not pair_counts:
            raise RuntimeError(
                "No participant/recording has at least two "
                "temporal windows to pair"
            )

        # Largest common cohort first; on a tie prefer the earlier
        # temporal windows for a deterministic diagnostic.
        selected_pair = min(
            pair_counts,
            key=lambda pair: (
                -pair_counts[pair],
                pair,
            ),
        )

        candidate_groups = [
            group_key
            for group_key, windows in (
                group_to_windows.items()
            )
            if all(
                window_index in windows
                for window_index in selected_pair
            )
        ]

        candidate_groups.sort(
            key=lambda group_key: min(
                group_to_windows[group_key][
                    window_index
                ]
                for window_index in selected_pair
            )
        )

        samples_per_group = len(selected_pair)
        max_groups = (
            number_of_samples // samples_per_group
        )

        if max_groups < 1:
            raise ValueError(
                "max-samples is too small for one paired group"
            )

        selected_groups = candidate_groups[
            :max_groups
        ]

        if not selected_groups:
            raise RuntimeError(
                "No paired groups remain after applying "
                "max-samples"
            )

        selected_indices = []

        for group_key in selected_groups:
            for window_index in selected_pair:
                selected_indices.append(
                    group_to_windows[group_key][
                        window_index
                    ]
                )

        self.selected_indices = selected_indices
        self.window_indices = tuple(
            int(value) for value in selected_pair
        )
        self.group_count = len(selected_groups)
        self.total_available_groups = len(
            candidate_groups
        )
        self.grouping_fields = tuple(
            sorted(grouping_fields)
        )

        print(
            "Paired windows:",
            ", ".join(
                f"window{value:03d}"
                for value in self.window_indices
            ),
        )
        print(
            "Paired groups selected:",
            self.group_count,
            "/",
            self.total_available_groups,
        )
        print(
            "Paired samples selected:",
            len(self.selected_indices),
        )
        print(
            "Paired grouping metadata:",
            ", ".join(self.grouping_fields),
        )

    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, index):
        return self.base_dataset[
            self.selected_indices[index]
        ]


def load_state_dict(
    checkpoint,
    possible_keys,
):
    """
    Retrieve a model state dictionary from a checkpoint.
    """

    for key in possible_keys:
        if key in checkpoint:
            return checkpoint[key]

    raise KeyError(
        "No model state dictionary found. "
        f"Available keys: {list(checkpoint.keys())}"
    )


def standardize_respiration(
    respiration,
    eps=1e-6,
):
    """
    Apply the same global per-window normalization
    used during Transformer training.
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


def decode_token_ids(
    vqgan,
    token_ids,
):
    """
    Convert an EEG token grid into a reconstructed
    EEG spectrogram using the frozen VQGAN decoder.

    token_ids:
        (B, 8, 64)

    output:
        (B, 1, 256, 512)
    """

    codebook = F.normalize(
        vqgan.quantizer.codebook.weight,
        p=2,
        dim=1,
    )

    quantized = F.embedding(
        token_ids,
        codebook,
    )

    # (B, 8, 64, 32)
    # -> (B, 32, 8, 64)
    quantized = (
        quantized
        .permute(0, 3, 1, 2)
        .contiguous()
    )

    return vqgan.decode(quantized)


def sample_correlation(
    target,
    prediction,
    eps=1e-8,
):
    """
    Pearson correlation calculated separately
    for every spectrogram.
    """

    target = target.flatten(
        start_dim=1
    )

    prediction = prediction.flatten(
        start_dim=1
    )

    target = target - target.mean(
        dim=1,
        keepdim=True,
    )

    prediction = prediction - prediction.mean(
        dim=1,
        keepdim=True,
    )

    numerator = (
        target * prediction
    ).sum(dim=1)

    denominator = (
        torch.sqrt(
            (target ** 2).sum(dim=1)
            + eps
        )
        * torch.sqrt(
            (prediction ** 2).sum(dim=1)
            + eps
        )
    )

    return numerator / denominator


def temporal_correlation(
    target,
    prediction,
    eps=1e-8,
):
    """
    Measure temporal agreement after removing each
    frequency bin's temporal mean.
    """

    target = target - target.mean(
        dim=-1,
        keepdim=True,
    )

    prediction = prediction - prediction.mean(
        dim=-1,
        keepdim=True,
    )

    return sample_correlation(
        target,
        prediction,
        eps=eps,
    )


def sample_snr(
    target,
    prediction,
    eps=1e-8,
):
    """
    Signal-to-noise ratio in dB for each sample.
    """

    target = target.flatten(
        start_dim=1
    )

    prediction = prediction.flatten(
        start_dim=1
    )

    signal_power = (
        target ** 2
    ).mean(dim=1)

    noise_power = (
        (target - prediction) ** 2
    ).mean(dim=1)

    return 10.0 * torch.log10(
        (signal_power + eps)
        / (noise_power + eps)
    )


def create_window_accumulator(start_sec):
    """
    Create metric sums for one temporal window offset.
    """

    return {
        "start_sec": int(start_sec),
        "evaluated_samples": 0,
        "positions": 0,
        "cross_entropy_sum": 0.0,
        "correct": 0,
        "top5_correct": 0,
        "mae_sum": 0.0,
        "correlation_sum": 0.0,
        "temporal_correlation_sum": 0.0,
        "snr_sum": 0.0,
        "oracle_mae_sum": 0.0,
        "oracle_correlation_sum": 0.0,
        "oracle_temporal_correlation_sum": 0.0,
        "target_codes": set(),
        "predicted_codes": set(),
    }


def finalize_window_accumulator(accumulator):
    """
    Convert metric sums into JSON-serializable averages.
    """

    samples = accumulator["evaluated_samples"]
    positions = accumulator["positions"]

    if samples < 1 or positions < 1:
        raise ValueError(
            "Cannot finalize an empty window accumulator"
        )

    return {
        "start_sec": accumulator["start_sec"],
        "start_min": accumulator["start_sec"] / 60.0,
        "evaluated_samples": samples,
        "cross_entropy": (
            accumulator["cross_entropy_sum"]
            / positions
        ),
        "token_accuracy": (
            accumulator["correct"]
            / positions
        ),
        "token_top5_accuracy": (
            accumulator["top5_correct"]
            / positions
        ),
        "predicted_eeg_mae": (
            accumulator["mae_sum"]
            / samples
        ),
        "predicted_eeg_correlation": (
            accumulator["correlation_sum"]
            / samples
        ),
        "predicted_eeg_temporal_correlation": (
            accumulator[
                "temporal_correlation_sum"
            ]
            / samples
        ),
        "predicted_eeg_snr_db": (
            accumulator["snr_sum"]
            / samples
        ),
        "oracle_vqgan_mae": (
            accumulator["oracle_mae_sum"]
            / samples
        ),
        "oracle_vqgan_correlation": (
            accumulator["oracle_correlation_sum"]
            / samples
        ),
        "oracle_vqgan_temporal_correlation": (
            accumulator[
                "oracle_temporal_correlation_sum"
            ]
            / samples
        ),
        "unique_target_codes": len(
            accumulator["target_codes"]
        ),
        "unique_predicted_codes": len(
            accumulator["predicted_codes"]
        ),
    }


def create_time_block_accumulator(
    start_sec,
    end_sec,
):
    """Create metric sums for one absolute-time block."""

    accumulator = create_window_accumulator(
        start_sec
    )
    accumulator["end_sec"] = int(end_sec)

    return accumulator


def finalize_time_block_accumulator(
    accumulator,
):
    """Finalize one absolute-time block."""

    values = finalize_window_accumulator(
        accumulator
    )
    values["end_sec"] = accumulator["end_sec"]
    values["end_min"] = (
        accumulator["end_sec"] / 60.0
    )

    return values


def save_time_block_plot(
    metrics_by_time_block,
    output_path,
):
    """Plot predicted and oracle temporal correlation over the night."""

    block_values = list(
        metrics_by_time_block.values()
    )

    midpoints = [
        (
            values["start_min"]
            + values["end_min"]
        )
        / 2.0
        for values in block_values
    ]

    predicted = [
        values[
            "predicted_eeg_temporal_correlation"
        ]
        for values in block_values
    ]

    oracle = [
        values[
            "oracle_vqgan_temporal_correlation"
        ]
        for values in block_values
    ]

    plt.figure(figsize=(10, 5))

    plt.plot(
        midpoints,
        predicted,
        marker="o",
        linewidth=2,
        label="Transformer prediction",
    )

    plt.plot(
        midpoints,
        oracle,
        marker="o",
        linewidth=2,
        label="Oracle VQGAN",
    )

    plt.axvline(
        MODEL_WINDOW_SEC / 60.0,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="256-minute window boundary",
    )

    plt.xlabel("Absolute time from recording start (minutes)")
    plt.ylabel("Temporal correlation")
    plt.title(
        "Temporal correlation by absolute 32-minute block"
    )
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_comparison(
    ground_truth,
    oracle_reconstruction,
    predicted_reconstruction,
    output_path,
    zero_respiration,
    shuffle_respiration,
):
    """
    Save a visual comparison for the first sample.
    """

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

    predicted_reconstruction = (
        predicted_reconstruction[0, 0]
        .detach()
        .cpu()
        .numpy()
    )

    difference = np.abs(
        ground_truth
        - predicted_reconstruction
    )

    if shuffle_respiration:
        prediction_title = (
            "EEG reconstructed from "
            "another participant's respiration"
        )

    elif zero_respiration:
        prediction_title = (
            "EEG reconstructed with "
            "zero respiration"
        )

    else:
        prediction_title = (
            "EEG reconstructed from "
            "correct synchronized respiration"
        )

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(14, 12),
    )

    images = [
        ground_truth,
        oracle_reconstruction,
        predicted_reconstruction,
        difference,
    ]

    titles = [
        "Ground-truth EEG spectrogram",
        (
            "Frozen VQGAN reconstruction "
            "from true EEG tokens"
        ),
        prediction_title,
        (
            "Absolute difference: "
            "ground truth vs prediction"
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

        axis.set_title(
            titles[index]
        )

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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the joint respiration-to-EEG "
            "Masked Transformer."
        )
    )

    parser.add_argument(
        "--transformer-checkpoint",
        type=Path,
        default=(
            DEFAULT_TRANSFORMER_CHECKPOINT
        ),
    )

    parser.add_argument(
        "--vqgan-checkpoint",
        type=Path,
        default=DEFAULT_VQGAN_CHECKPOINT,
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--paired-windows",
        action="store_true",
        help=(
            "Evaluate the same participant/recording cohort "
            "at the two temporal windows with the largest "
            "common coverage."
        ),
    )

    respiration_group = (
        parser.add_mutually_exclusive_group()
    )

    respiration_group.add_argument(
        "--zero-respiration",
        action="store_true",
        help=(
            "Replace normalized respiration "
            "with zeros."
        ),
    )

    respiration_group.add_argument(
        "--shuffle-respiration",
        action="store_true",
        help=(
            "Replace respiration with respiration "
            "from another validation participant."
        ),
    )

    args = parser.parse_args()

    if args.max_samples < 1:
        raise ValueError(
            "max-samples must be at least 1"
        )

    if not (
        args.transformer_checkpoint.exists()
    ):
        raise FileNotFoundError(
            "Transformer checkpoint not found: "
            f"{args.transformer_checkpoint}"
        )

    if not args.vqgan_checkpoint.exists():
        raise FileNotFoundError(
            "VQGAN checkpoint not found: "
            f"{args.vqgan_checkpoint}"
        )

    transformer_checkpoint = torch.load(
        args.transformer_checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    held_out_fold = int(
        transformer_checkpoint[
            "held_out_fold"
        ]
    )

    min_db = float(
        transformer_checkpoint["min_db"]
    )

    max_db = float(
        transformer_checkpoint["max_db"]
    )

    if args.output_dir is None:
        if args.shuffle_respiration:
            suffix = (
                "evaluation_shuffled_respiration"
            )

        elif args.zero_respiration:
            suffix = (
                "evaluation_zero_respiration"
            )

        elif args.paired_windows:
            suffix = "evaluation_paired_windows"

        else:
            suffix = "evaluation"

        output_dir = (
            args.transformer_checkpoint.parent
            / suffix
        )

    else:
        output_dir = args.output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)
    print(
        "Held-out fold:",
        held_out_fold,
    )
    print(
        "Checkpoint epoch:",
        transformer_checkpoint.get(
            "epoch"
        ),
    )
    print(
        "Architecture:",
        transformer_checkpoint.get(
            "architecture",
            "unknown",
        ),
    )
    print(
        "Zero respiration:",
        args.zero_respiration,
    )
    print(
        "Shuffled respiration:",
        args.shuffle_respiration,
    )

    print(
        "Paired windows:",
        args.paired_windows,
    )

    validation_directory = (
        args.data_root
        / f"fold_{held_out_fold}"
    )

    dataset = PhysiologyPairDataset(
        validation_directory,
        min_db=min_db,
        max_db=max_db,
    )

    if "start_sec" not in dataset[0]:
        raise KeyError(
            "The dataset must return start_sec for the "
            "window diagnostic. Update src/data/dataset.py "
            "and confirm that the SHHS NPZ files contain "
            "start_sec metadata."
        )

    paired_window_indices = None
    paired_group_count = None
    paired_available_group_count = None

    if args.paired_windows:
        paired_dataset = PairedWindowsDataset(
            base_dataset=dataset,
            number_of_samples=args.max_samples,
        )
        paired_window_indices = list(
            paired_dataset.window_indices
        )
        paired_group_count = (
            paired_dataset.group_count
        )
        paired_available_group_count = (
            paired_dataset.total_available_groups
        )
        dataset = paired_dataset

    if args.shuffle_respiration:
        dataset = ShuffledRespirationDataset(
            base_dataset=dataset,
            number_of_samples=min(
                args.max_samples,
                len(dataset),
            ),
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

    architecture = transformer_checkpoint.get(
        "architecture",
        "joint_concatenated_transformer",
    )

    if architecture == "joint_codebook_tied_transformer_v3":
        model_config = transformer_checkpoint.get(
            "model_config"
        )

        if model_config is None:
            raise KeyError(
                "The V3 checkpoint is missing model_config."
            )

        transformer = RespirationToEEGTransformer(
            **model_config
        ).to(device)

    elif architecture == "joint_concatenated_transformer_v2":
        model_config = transformer_checkpoint.get(
            "model_config"
        )

        if model_config is None:
            raise KeyError(
                "The V2 checkpoint is missing model_config. "
                "Use a checkpoint produced by the updated "
                "training script."
            )

        transformer = RespirationToEEGTransformer(
            **model_config
        ).to(device)

    elif architecture == "mage_visible_encoder_decoder":
        transformer = (
            MAGERespirationToEEGTransformer()
            .to(device)
        )

    elif architecture in (
        None,
        "joint_concatenated_transformer",
    ):
        # Explicit legacy configuration.
        transformer = (
            RespirationToEEGTransformer(
                embedding_dim=768,
                num_heads=8,
                num_encoder_layers=8,
                num_decoder_layers=8,
                mlp_ratio=4,
                dropout=0.1,
                num_window_types=2,
                use_window_embedding=False,
            )
            .to(device)
        )

    else:
        raise ValueError(
            "Unsupported Transformer architecture: "
            f"{architecture}"
        )

    transformer_state = load_state_dict(
        transformer_checkpoint,
        possible_keys=[
            "model_state_dict",
            "transformer_state_dict",
        ],
    )

    transformer.load_state_dict(
        transformer_state,
        strict=True,
    )

    transformer.eval()

    vqgan_checkpoint = torch.load(
        args.vqgan_checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    vqgan = VQGAN().to(device)

    vqgan_state = load_state_dict(
        vqgan_checkpoint,
        possible_keys=[
            "model_state_dict",
            "vqgan_state_dict",
            "generator_state_dict",
        ],
    )

    vqgan.load_state_dict(
        vqgan_state,
        strict=True,
    )

    vqgan.eval()

    for parameter in vqgan.parameters():
        parameter.requires_grad = False

    normalized_codebook = F.normalize(
        vqgan.quantizer.codebook.weight.detach(),
        p=2,
        dim=1,
    )

    mask_token_id = getattr(
        transformer,
        "mask_token_id",
        8192,
    )

    total_cross_entropy = 0.0
    total_positions = 0
    total_correct = 0
    total_top5 = 0

    total_mae = 0.0
    total_correlation = 0.0
    total_temporal_correlation = 0.0
    total_snr = 0.0

    total_oracle_mae = 0.0
    total_oracle_correlation = 0.0
    total_oracle_temporal_correlation = 0.0

    target_codes = set()
    predicted_codes = set()

    evaluated_samples = 0
    comparison_saved = False
    window_accumulators = {}
    time_block_accumulators = {}

    if (
        MODEL_WINDOW_SEC
        % DIAGNOSTIC_BLOCK_SEC
        != 0
    ):
        raise ValueError(
            "DIAGNOSTIC_BLOCK_SEC must divide "
            "MODEL_WINDOW_SEC exactly"
        )

    blocks_per_window = (
        MODEL_WINDOW_SEC
        // DIAGNOSTIC_BLOCK_SEC
    )

    with torch.inference_mode():
        for batch in loader:
            if (
                evaluated_samples
                >= args.max_samples
            ):
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

            # Shuffled samples are normalized only after
            # the complete respiration window is replaced.
            respiration = (
                standardize_respiration(
                    respiration
                )
            )

            if args.zero_respiration:
                respiration = torch.zeros_like(
                    respiration
                )

            _, true_tokens, _ = vqgan.encode(
                eeg
            )

            masked_tokens = torch.full_like(
                true_tokens,
                fill_value=mask_token_id,
            )

            forward_kwargs = {}

            if getattr(
                transformer,
                "use_window_embedding",
                False,
            ):
                if "start_sec" not in batch:
                    raise KeyError(
                        "start_sec is required by the "
                        "window embedding"
                    )

                start_seconds = torch.as_tensor(
                    batch["start_sec"],
                    device=device,
                ).reshape(-1)

                window_indices = torch.div(
                    start_seconds.round().long(),
                    MODEL_WINDOW_SEC,
                    rounding_mode="floor",
                )

                forward_kwargs["window_index"] = (
                    window_indices
                )

            if getattr(
                transformer,
                "use_codebook_tied_output",
                False,
            ):
                forward_kwargs["codebook"] = (
                    normalized_codebook
                )

            logits = transformer(
                respiration,
                masked_tokens,
                **forward_kwargs,
            )

            targets_flat = true_tokens.flatten(
                start_dim=1
            )

            position_losses = F.cross_entropy(
                logits.reshape(
                    -1,
                    logits.shape[-1],
                ),
                targets_flat.reshape(-1),
                reduction="none",
            ).view_as(
                targets_flat
            )

            loss = position_losses.sum()

            predicted_flat = logits.argmax(
                dim=-1
            )

            predicted_tokens = (
                predicted_flat.view_as(
                    true_tokens
                )
            )

            top5_predictions = logits.topk(
                k=5,
                dim=-1,
            ).indices

            correct = (
                predicted_flat
                == targets_flat
            )

            top5_correct = (
                top5_predictions
                == targets_flat.unsqueeze(-1)
            ).any(dim=-1)

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

            batch_size = eeg.shape[0]
            positions = targets_flat.numel()

            sample_mae = (
                torch.abs(
                    eeg
                    - predicted_reconstruction
                )
                .flatten(start_dim=1)
                .mean(dim=1)
            )

            sample_global_correlation = (
                sample_correlation(
                    eeg,
                    predicted_reconstruction,
                )
            )

            sample_temporal_correlation = (
                temporal_correlation(
                    eeg,
                    predicted_reconstruction,
                )
            )

            sample_snr_values = sample_snr(
                eeg,
                predicted_reconstruction,
            )

            sample_oracle_mae = (
                torch.abs(
                    eeg
                    - oracle_reconstruction
                )
                .flatten(start_dim=1)
                .mean(dim=1)
            )

            sample_oracle_correlation = (
                sample_correlation(
                    eeg,
                    oracle_reconstruction,
                )
            )

            sample_oracle_temporal_correlation = (
                temporal_correlation(
                    eeg,
                    oracle_reconstruction,
                )
            )

            total_cross_entropy += (
                loss.item()
            )

            total_positions += positions

            total_correct += (
                correct.sum().item()
            )

            total_top5 += (
                top5_correct.sum().item()
            )

            total_mae += (
                sample_mae
                .sum()
                .item()
            )

            total_correlation += (
                sample_global_correlation
                .sum()
                .item()
            )

            total_temporal_correlation += (
                sample_temporal_correlation
                .sum()
                .item()
            )

            total_snr += (
                sample_snr_values
                .sum()
                .item()
            )

            total_oracle_mae += (
                sample_oracle_mae
                .sum()
                .item()
            )

            total_oracle_correlation += (
                sample_oracle_correlation
                .sum()
                .item()
            )

            total_oracle_temporal_correlation += (
                sample_oracle_temporal_correlation
                .sum()
                .item()
            )

            if "start_sec" not in batch:
                raise KeyError(
                    "start_sec is missing from an "
                    "evaluation batch"
                )

            positions_per_sample = (
                targets_flat.shape[1]
            )

            for sample_index in range(batch_size):
                start_value = batch[
                    "start_sec"
                ][sample_index]

                if torch.is_tensor(start_value):
                    start_value = start_value.item()

                start_sec = int(
                    round(float(start_value))
                )

                window_index = (
                    start_sec
                    // MODEL_WINDOW_SEC
                )

                window_name = (
                    f"window{window_index:03d}"
                )

                if window_name not in (
                    window_accumulators
                ):
                    window_accumulators[
                        window_name
                    ] = create_window_accumulator(
                        start_sec
                    )

                accumulator = (
                    window_accumulators[
                        window_name
                    ]
                )

                if (
                    accumulator["start_sec"]
                    != start_sec
                ):
                    raise ValueError(
                        "Different start times mapped to "
                        f"the same {window_name}: "
                        f"{accumulator['start_sec']} and "
                        f"{start_sec}"
                    )

                accumulator[
                    "evaluated_samples"
                ] += 1

                accumulator[
                    "positions"
                ] += positions_per_sample

                accumulator[
                    "cross_entropy_sum"
                ] += (
                    position_losses[
                        sample_index
                    ]
                    .sum()
                    .item()
                )

                accumulator["correct"] += (
                    correct[sample_index]
                    .sum()
                    .item()
                )

                accumulator[
                    "top5_correct"
                ] += (
                    top5_correct[sample_index]
                    .sum()
                    .item()
                )

                accumulator["mae_sum"] += (
                    sample_mae[sample_index]
                    .item()
                )

                accumulator[
                    "correlation_sum"
                ] += (
                    sample_global_correlation[
                        sample_index
                    ]
                    .item()
                )

                accumulator[
                    "temporal_correlation_sum"
                ] += (
                    sample_temporal_correlation[
                        sample_index
                    ]
                    .item()
                )

                accumulator["snr_sum"] += (
                    sample_snr_values[
                        sample_index
                    ]
                    .item()
                )

                accumulator[
                    "oracle_mae_sum"
                ] += (
                    sample_oracle_mae[
                        sample_index
                    ]
                    .item()
                )

                accumulator[
                    "oracle_correlation_sum"
                ] += (
                    sample_oracle_correlation[
                        sample_index
                    ]
                    .item()
                )

                accumulator[
                    "oracle_temporal_correlation_sum"
                ] += (
                    sample_oracle_temporal_correlation[
                        sample_index
                    ]
                    .item()
                )

                accumulator[
                    "target_codes"
                ].update(
                    true_tokens[sample_index]
                    .detach()
                    .cpu()
                    .reshape(-1)
                    .tolist()
                )

                accumulator[
                    "predicted_codes"
                ].update(
                    predicted_tokens[
                        sample_index
                    ]
                    .detach()
                    .cpu()
                    .reshape(-1)
                    .tolist()
                )

                if (
                    eeg.shape[-1]
                    % blocks_per_window
                    != 0
                ):
                    raise ValueError(
                        "EEG time columns cannot be split "
                        "into equal diagnostic blocks: "
                        f"{eeg.shape[-1]} columns"
                    )

                if (
                    true_tokens.shape[-1]
                    % blocks_per_window
                    != 0
                ):
                    raise ValueError(
                        "Token time columns cannot be split "
                        "into equal diagnostic blocks: "
                        f"{true_tokens.shape[-1]} columns"
                    )

                eeg_columns_per_block = (
                    eeg.shape[-1]
                    // blocks_per_window
                )

                token_columns_per_block = (
                    true_tokens.shape[-1]
                    // blocks_per_window
                )

                sample_loss_grid = (
                    position_losses[sample_index]
                    .reshape_as(
                        true_tokens[sample_index]
                    )
                )

                sample_correct_grid = (
                    correct[sample_index]
                    .reshape_as(
                        true_tokens[sample_index]
                    )
                )

                sample_top5_grid = (
                    top5_correct[sample_index]
                    .reshape_as(
                        true_tokens[sample_index]
                    )
                )

                for local_block_index in range(
                    blocks_per_window
                ):
                    block_start_sec = (
                        start_sec
                        + local_block_index
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
                        f"block{absolute_block_index:03d}"
                    )

                    if block_name not in (
                        time_block_accumulators
                    ):
                        time_block_accumulators[
                            block_name
                        ] = create_time_block_accumulator(
                            block_start_sec,
                            block_end_sec,
                        )

                    block_accumulator = (
                        time_block_accumulators[
                            block_name
                        ]
                    )

                    if (
                        block_accumulator["start_sec"]
                        != block_start_sec
                        or block_accumulator["end_sec"]
                        != block_end_sec
                    ):
                        raise ValueError(
                            "Different time ranges mapped to "
                            f"the same {block_name}"
                        )

                    eeg_start = (
                        local_block_index
                        * eeg_columns_per_block
                    )
                    eeg_end = (
                        eeg_start
                        + eeg_columns_per_block
                    )

                    token_start = (
                        local_block_index
                        * token_columns_per_block
                    )
                    token_end = (
                        token_start
                        + token_columns_per_block
                    )

                    target_eeg_block = eeg[
                        sample_index : sample_index + 1,
                        ...,
                        eeg_start:eeg_end,
                    ]

                    predicted_eeg_block = (
                        predicted_reconstruction[
                            sample_index : sample_index + 1,
                            ...,
                            eeg_start:eeg_end,
                        ]
                    )

                    oracle_eeg_block = (
                        oracle_reconstruction[
                            sample_index : sample_index + 1,
                            ...,
                            eeg_start:eeg_end,
                        ]
                    )

                    block_losses = sample_loss_grid[
                        ...,
                        token_start:token_end,
                    ]

                    block_correct = (
                        sample_correct_grid[
                            ...,
                            token_start:token_end,
                        ]
                    )

                    block_top5_correct = (
                        sample_top5_grid[
                            ...,
                            token_start:token_end,
                        ]
                    )

                    target_token_block = true_tokens[
                        sample_index,
                        ...,
                        token_start:token_end,
                    ]

                    predicted_token_block = (
                        predicted_tokens[
                            sample_index,
                            ...,
                            token_start:token_end,
                        ]
                    )

                    block_accumulator[
                        "evaluated_samples"
                    ] += 1

                    block_accumulator[
                        "positions"
                    ] += block_losses.numel()

                    block_accumulator[
                        "cross_entropy_sum"
                    ] += block_losses.sum().item()

                    block_accumulator["correct"] += (
                        block_correct.sum().item()
                    )

                    block_accumulator[
                        "top5_correct"
                    ] += block_top5_correct.sum().item()

                    block_accumulator["mae_sum"] += (
                        torch.abs(
                            target_eeg_block
                            - predicted_eeg_block
                        )
                        .mean()
                        .item()
                    )

                    block_accumulator[
                        "correlation_sum"
                    ] += sample_correlation(
                        target_eeg_block,
                        predicted_eeg_block,
                    )[0].item()

                    block_accumulator[
                        "temporal_correlation_sum"
                    ] += temporal_correlation(
                        target_eeg_block,
                        predicted_eeg_block,
                    )[0].item()

                    block_accumulator["snr_sum"] += (
                        sample_snr(
                            target_eeg_block,
                            predicted_eeg_block,
                        )[0].item()
                    )

                    block_accumulator[
                        "oracle_mae_sum"
                    ] += (
                        torch.abs(
                            target_eeg_block
                            - oracle_eeg_block
                        )
                        .mean()
                        .item()
                    )

                    block_accumulator[
                        "oracle_correlation_sum"
                    ] += sample_correlation(
                        target_eeg_block,
                        oracle_eeg_block,
                    )[0].item()

                    block_accumulator[
                        "oracle_temporal_correlation_sum"
                    ] += temporal_correlation(
                        target_eeg_block,
                        oracle_eeg_block,
                    )[0].item()

                    block_accumulator[
                        "target_codes"
                    ].update(
                        target_token_block
                        .detach()
                        .cpu()
                        .reshape(-1)
                        .tolist()
                    )

                    block_accumulator[
                        "predicted_codes"
                    ].update(
                        predicted_token_block
                        .detach()
                        .cpu()
                        .reshape(-1)
                        .tolist()
                    )

            target_codes.update(
                true_tokens
                .detach()
                .cpu()
                .reshape(-1)
                .tolist()
            )

            predicted_codes.update(
                predicted_tokens
                .detach()
                .cpu()
                .reshape(-1)
                .tolist()
            )

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
                        output_dir
                        / "reconstruction_comparison.png"
                    ),
                    zero_respiration=(
                        args.zero_respiration
                    ),
                    shuffle_respiration=(
                        args.shuffle_respiration
                    ),
                )

                comparison_saved = True

            evaluated_samples += batch_size

            if (
                evaluated_samples % 25
                == 0
            ):
                print(
                    "Evaluated:",
                    evaluated_samples,
                )

    metrics_by_window = {
        window_name: finalize_window_accumulator(
            accumulator
        )
        for window_name, accumulator in sorted(
            window_accumulators.items()
        )
    }

    metrics_by_time_block = {
        block_name: finalize_time_block_accumulator(
            accumulator
        )
        for block_name, accumulator in sorted(
            time_block_accumulators.items()
        )
    }

    metrics = {
        "transformer_checkpoint": str(
            args.transformer_checkpoint
        ),
        "vqgan_checkpoint": str(
            args.vqgan_checkpoint
        ),
        "held_out_fold": held_out_fold,
        "checkpoint_epoch": (
            transformer_checkpoint.get(
                "epoch"
            )
        ),
        "architecture": architecture,
        "model_config": (
            transformer_checkpoint.get(
                "model_config"
            )
        ),
        "semantic_loss_weight": (
            transformer_checkpoint.get(
                "semantic_loss_weight"
            )
        ),
        "temporal_loss_weight": (
            transformer_checkpoint.get(
                "temporal_loss_weight"
            )
        ),
        "temporal_cosine_weight": (
            transformer_checkpoint.get(
                "temporal_cosine_weight"
            )
        ),
        "codebook_temperature": (
            transformer_checkpoint.get(
                "codebook_temperature"
            )
        ),
        "zero_respiration": (
            args.zero_respiration
        ),
        "shuffle_respiration": (
            args.shuffle_respiration
        ),
        "paired_windows": args.paired_windows,
        "paired_window_indices": (
            paired_window_indices
        ),
        "paired_group_count": (
            paired_group_count
        ),
        "paired_available_group_count": (
            paired_available_group_count
        ),
        "evaluated_samples": (
            evaluated_samples
        ),
        "cross_entropy": (
            total_cross_entropy
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
        "unique_target_codes": len(
            target_codes
        ),
        "unique_predicted_codes": len(
            predicted_codes
        ),
        "metrics_by_window": metrics_by_window,
        "diagnostic_block_sec": (
            DIAGNOSTIC_BLOCK_SEC
        ),
        "metrics_by_time_block": (
            metrics_by_time_block
        ),
    }

    time_block_plot_path = (
        output_dir
        / "temporal_correlation_by_time_block.png"
    )

    save_time_block_plot(
        metrics_by_time_block,
        time_block_plot_path,
    )

    metrics_path = (
        output_dir / "metrics.json"
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

    print()
    print(
        "Evaluated samples:",
        evaluated_samples,
    )
    print(
        "Cross-entropy:",
        f"{metrics['cross_entropy']:.4f}",
    )
    print(
        "Token accuracy:",
        f"{metrics['token_accuracy']:.4f}",
    )
    print(
        "Token top-5 accuracy:",
        (
            f"{metrics['token_top5_accuracy']:.4f}"
        ),
    )
    print(
        "Predicted EEG MAE:",
        f"{metrics['predicted_eeg_mae']:.4f}",
    )
    print(
        "Predicted EEG correlation:",
        (
            f"{metrics['predicted_eeg_correlation']:.4f}"
        ),
    )
    print(
        "Predicted EEG temporal correlation:",
        (
            f"{metrics['predicted_eeg_temporal_correlation']:.4f}"
        ),
    )
    print(
        "Predicted EEG SNR:",
        (
            f"{metrics['predicted_eeg_snr_db']:.2f} dB"
        ),
    )
    print(
        "Oracle VQGAN MAE:",
        f"{metrics['oracle_vqgan_mae']:.4f}",
    )
    print(
        "Oracle VQGAN correlation:",
        (
            f"{metrics['oracle_vqgan_correlation']:.4f}"
        ),
    )
    print(
        "Oracle temporal correlation:",
        (
            f"{metrics['oracle_vqgan_temporal_correlation']:.4f}"
        ),
    )
    print(
        "Unique target codes:",
        metrics["unique_target_codes"],
    )
    print(
        "Unique predicted codes:",
        metrics["unique_predicted_codes"],
    )

    print()
    print("Metrics by window:")

    for window_name, window_values in (
        metrics_by_window.items()
    ):
        print(
            f"  {window_name} "
            f"| start={window_values['start_min']:.0f} min "
            f"| n={window_values['evaluated_samples']} "
            f"| CE={window_values['cross_entropy']:.4f} "
            f"| acc={window_values['token_accuracy']:.4f} "
            f"| MAE={window_values['predicted_eeg_mae']:.4f} "
            "| temporal_corr="
            f"{window_values['predicted_eeg_temporal_correlation']:.4f} "
            f"| SNR={window_values['predicted_eeg_snr_db']:.2f} dB "
            "| predicted_codes="
            f"{window_values['unique_predicted_codes']}"
        )

    print()
    print("Metrics by absolute 32-minute block:")

    for block_values in (
        metrics_by_time_block.values()
    ):
        print(
            "  "
            f"{block_values['start_min']:.0f}-"
            f"{block_values['end_min']:.0f} min "
            f"| n={block_values['evaluated_samples']} "
            f"| CE={block_values['cross_entropy']:.4f} "
            f"| acc={block_values['token_accuracy']:.4f} "
            f"| MAE={block_values['predicted_eeg_mae']:.4f} "
            "| temporal_corr="
            f"{block_values['predicted_eeg_temporal_correlation']:.4f} "
            "| oracle_temporal_corr="
            f"{block_values['oracle_vqgan_temporal_correlation']:.4f} "
            f"| SNR={block_values['predicted_eeg_snr_db']:.2f} dB"
        )

    print(
        "Comparison saved:",
        output_dir
        / "reconstruction_comparison.png",
    )
    print(
        "Metrics saved:",
        metrics_path,
    )
    print(
        "Time-block plot saved:",
        time_block_plot_path,
    )


if __name__ == "__main__":
    main()
