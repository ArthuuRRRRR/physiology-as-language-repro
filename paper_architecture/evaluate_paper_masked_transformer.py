"""Evaluate the current multi-dataset paper respiration-to-EEG Transformer.

Pipeline:

    respiration mmap
        ↓
    Paper Transformer
        ↓
    predicted EEG tokens (8 x 64)
        ↓
    frozen multi-dataset VQGAN decoder
        ↓
    reconstructed EEG spectrogram (256 x 512)

Metrics are kept compatible with the previous V5/V6 evaluator:
    - cross entropy
    - exact token accuracy
    - top-5 token accuracy
    - EEG MAE
    - global EEG correlation
    - temporal EEG correlation
    - EEG SNR
    - oracle VQGAN reconstruction metrics

Paper inference is one-shot:
all 512 EEG tokens are predicted from respiration alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from paper_architecture import PaperMaskedRespirationToEEGTransformer
from src.models.vqgan import VQGAN


DEFAULT_TRANSFORMER_CHECKPOINT = Path(
    "outputs/paper_masked_transformer_multidataset_full/checkpoint_best.pt"
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

MODEL_WINDOW_SEC = 256 * 60
DIAGNOSTIC_BLOCK_SEC = 32 * 60


# ============================================================
# Utilities
# ============================================================

def load_state_dict(checkpoint, possible_keys):
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
    """Same normalization used during Transformer training."""

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
    ) / (
        std + eps
    )


# ============================================================
# Dataset
# ============================================================

class PaperTransformerEvaluationDataset(Dataset):
    """
    Join Transformer mmap data with the original preprocessed
    EEG spectrogram corresponding to each window.
    """

    def __init__(
        self,
        transformer_data_root,
        preprocessed_root,
        split,
    ):
        super().__init__()

        self.transformer_data_root = Path(
            transformer_data_root
        )

        self.preprocessed_root = Path(
            preprocessed_root
        )

        self.split = split

        split_dir = (
            self.transformer_data_root
            / split
        )

        metadata_path = (
            split_dir
            / "metadata.json"
        )

        index_path = (
            split_dir
            / "index.jsonl"
        )

        source_manifest_path = (
            self.preprocessed_root
            / f"{split}_manifest.jsonl"
        )

        normalization_path = (
            self.preprocessed_root
            / "normalization.json"
        )

        for path in [
            metadata_path,
            index_path,
            source_manifest_path,
            normalization_path,
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing required file: {path}"
                )

        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            self.metadata = json.load(file)

        with normalization_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            normalization = json.load(file)

        self.min_db = float(
            normalization["min_db"]
        )

        self.max_db = float(
            normalization["max_db"]
        )

        self.num_samples = int(
            self.metadata["num_samples"]
        )

        self.resp_shape = tuple(
            self.metadata[
                "respiration"
            ][
                "shape"
            ]
        )

        self.token_shape = tuple(
            self.metadata[
                "eeg_tokens"
            ][
                "shape"
            ]
        )

        self.resp_path = (
            split_dir
            / self.metadata[
                "respiration"
            ][
                "path"
            ]
        )

        self.token_path = (
            split_dir
            / self.metadata[
                "eeg_tokens"
            ][
                "path"
            ]
        )

        with index_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            self.index_rows = [
                json.loads(line)
                for line in file
            ]

        with source_manifest_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            self.source_rows = [
                json.loads(line)
                for line in file
            ]

        if len(self.index_rows) != self.num_samples:
            raise RuntimeError(
                "index.jsonl length does not match metadata."
            )

        if len(self.source_rows) != self.num_samples:
            raise RuntimeError(
                "Source manifest length does not match mmap dataset."
            )

        # ----------------------------------------------------
        # Strict alignment check
        # ----------------------------------------------------

        for index, (
            mmap_row,
            source_row,
        ) in enumerate(
            zip(
                self.index_rows,
                self.source_rows,
            )
        ):
            for key in [
                "dataset",
                "subject_id",
                "window_index",
            ]:
                mmap_value = str(
                    mmap_row[key]
                )

                source_value = str(
                    source_row[key]
                )

                if mmap_value != source_value:
                    raise RuntimeError(
                        "Dataset alignment failure "
                        f"at index {index}: "
                        f"{key}={mmap_value} "
                        f"vs {source_value}"
                    )

        self._respiration = None
        self._tokens = None

        print(
            "Dataset alignment check: OK"
        )

        print(
            "Samples:",
            self.num_samples
        )

        print(
            "Respiration shape:",
            self.resp_shape
        )

        print(
            "Token shape:",
            self.token_shape
        )

        print(
            "EEG normalization:",
            self.min_db,
            self.max_db,
        )

    def __len__(self):
        return self.num_samples

    def _ensure_open(self):
        if self._respiration is None:
            self._respiration = np.memmap(
                self.resp_path,
                dtype="float32",
                mode="r",
                shape=self.resp_shape,
            )

        if self._tokens is None:
            self._tokens = np.memmap(
                self.token_path,
                dtype="uint16",
                mode="r",
                shape=self.token_shape,
            )

    def __getstate__(self):
        state = self.__dict__.copy()

        state["_respiration"] = None
        state["_tokens"] = None

        return state

    def _load_ground_truth_eeg(
        self,
        source_row,
    ):
        file_path = Path(
            source_row["path"]
        )

        if not file_path.exists():
            alternate = (
                PROJECT_ROOT
                / file_path
            )

            if alternate.exists():
                file_path = alternate

        if not file_path.exists():
            raise FileNotFoundError(
                file_path
            )

        with np.load(
            file_path,
            allow_pickle=False,
        ) as sample:
            eeg_db = np.asarray(
                sample[
                    "eeg_spectrogram_db"
                ],
                dtype=np.float32,
            )

        if eeg_db.shape != (
            256,
            512,
        ):
            raise RuntimeError(
                "Unexpected EEG spectrogram shape "
                f"{eeg_db.shape}: {file_path}"
            )

        eeg = np.clip(
            eeg_db,
            self.min_db,
            self.max_db,
        )

        eeg = (
            eeg
            - self.min_db
        ) / (
            self.max_db
            - self.min_db
        )

        return eeg.astype(
            np.float32,
            copy=False,
        )

    def __getitem__(
        self,
        index,
    ):
        self._ensure_open()

        respiration = np.array(
            self._respiration[index],
            dtype=np.float32,
            copy=True,
        )

        tokens = np.array(
            self._tokens[index],
            dtype=np.int64,
            copy=True,
        )

        index_row = self.index_rows[index]
        source_row = self.source_rows[index]

        eeg = self._load_ground_truth_eeg(
            source_row
        )

        window_index = int(
            index_row["window_index"]
        )

        return {
            "respiration": torch.from_numpy(
                respiration
            ),

            "eeg_tokens": torch.from_numpy(
                tokens
            ),

            "eeg_spectrogram": torch.from_numpy(
                eeg
            ),

            "dataset": str(
                index_row["dataset"]
            ),

            "subject_id": str(
                index_row["subject_id"]
            ),

            "window_index": window_index,

            "start_sec": (
                window_index
                * MODEL_WINDOW_SEC
            ),
        }


# ============================================================
# Shuffled respiration
# ============================================================

class ShuffledRespirationDataset(Dataset):
    """
    Replace respiration with respiration from another subject
    at the same window index.

    Windows containing only one sample are excluded completely
    from shuffled evaluation, because a valid participant-wise
    shuffle is impossible.
    """

    def __init__(
        self,
        base_dataset,
    ):
        self.base_dataset = base_dataset

        groups = defaultdict(list)

        for index, row in enumerate(
            base_dataset.index_rows
        ):
            window_index = int(
                row["window_index"]
            )

            groups[
                window_index
            ].append(index)

        self.target_indices = []
        self.source_indices = []

        for (
            window_index,
            indices,
        ) in sorted(
            groups.items()
        ):
            if len(indices) < 2:
                print(
                    "Skipping shuffle for window",
                    window_index,
                    "- fewer than 2 samples",
                )

                continue

            subject_ids = [
                str(
                    base_dataset.index_rows[
                        index
                    ][
                        "subject_id"
                    ]
                )
                for index in indices
            ]

            selected_shift = None

            for shift in range(
                1,
                len(indices),
            ):
                valid = all(
                    subject_ids[position]
                    != subject_ids[
                        (
                            position
                            + shift
                        )
                        % len(indices)
                    ]
                    for position
                    in range(
                        len(indices)
                    )
                )

                if valid:
                    selected_shift = shift
                    break

            if selected_shift is None:
                raise RuntimeError(
                    "Could not create "
                    "participant-wise shuffle "
                    f"for window {window_index}"
                )

            print(
                "Shuffle window",
                window_index,
                "shift=",
                selected_shift,
                "samples=",
                len(indices),
            )

            for position, target_index in enumerate(
                indices
            ):
                source_index = indices[
                    (
                        position
                        + selected_shift
                    )
                    % len(indices)
                ]

                self.target_indices.append(
                    target_index
                )

                self.source_indices.append(
                    source_index
                )

        if not self.target_indices:
            raise RuntimeError(
                "No samples available for shuffled evaluation."
            )

        print(
            "Shuffled samples selected:",
            len(self.target_indices),
        )

    def __len__(self):
        return len(
            self.target_indices
        )

    def __getitem__(
        self,
        index,
    ):
        target_index = (
            self.target_indices[
                index
            ]
        )

        source_index = (
            self.source_indices[
                index
            ]
        )

        target = dict(
            self.base_dataset[
                target_index
            ]
        )

        source = self.base_dataset[
            source_index
        ]

        target_subject = str(
            target["subject_id"]
        )

        source_subject = str(
            source["subject_id"]
        )

        if (
            target_subject
            == source_subject
        ):
            raise RuntimeError(
                "Respiration source and EEG target "
                "belong to the same participant."
            )

        if (
            int(
                target["window_index"]
            )
            != int(
                source["window_index"]
            )
        ):
            raise RuntimeError(
                "Shuffled respiration does not "
                "come from the same window index."
            )

        target[
            "respiration"
        ] = source[
            "respiration"
        ]

        target[
            "respiration_source_subject"
        ] = source_subject

        return target


# ============================================================
# VQGAN decoding
# ============================================================

def decode_token_ids(
    vqgan,
    token_ids,
):
    """
    token_ids:
        (B, 8, 64)

    output:
        (B, 1, 256, 512)
    """

    codebook = F.normalize(
        vqgan
        .quantizer
        .codebook
        .weight,
        p=2,
        dim=1,
    )

    quantized = F.embedding(
        token_ids,
        codebook,
    )

    quantized = (
        quantized
        .permute(
            0,
            3,
            1,
            2,
        )
        .contiguous()
    )

    return vqgan.decode(
        quantized
    )


# ============================================================
# Metrics
# ============================================================

def sample_correlation(
    target,
    prediction,
    eps=1e-8,
):
    target = target.flatten(
        start_dim=1
    )

    prediction = prediction.flatten(
        start_dim=1
    )

    target = (
        target
        - target.mean(
            dim=1,
            keepdim=True,
        )
    )

    prediction = (
        prediction
        - prediction.mean(
            dim=1,
            keepdim=True,
        )
    )

    numerator = (
        target
        * prediction
    ).sum(
        dim=1
    )

    denominator = (
        torch.sqrt(
            (
                target ** 2
            ).sum(
                dim=1
            )
            + eps
        )
        *
        torch.sqrt(
            (
                prediction ** 2
            ).sum(
                dim=1
            )
            + eps
        )
    )

    return (
        numerator
        / denominator
    )


def temporal_correlation(
    target,
    prediction,
    eps=1e-8,
):
    target = (
        target
        - target.mean(
            dim=-1,
            keepdim=True,
        )
    )

    prediction = (
        prediction
        - prediction.mean(
            dim=-1,
            keepdim=True,
        )
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
    target = target.flatten(
        start_dim=1
    )

    prediction = prediction.flatten(
        start_dim=1
    )

    signal_power = (
        target ** 2
    ).mean(
        dim=1
    )

    noise_power = (
        (
            target
            - prediction
        ) ** 2
    ).mean(
        dim=1
    )

    return (
        10.0
        * torch.log10(
            (
                signal_power
                + eps
            )
            /
            (
                noise_power
                + eps
            )
        )
    )


# ============================================================
# Accumulators
# ============================================================

def create_accumulator(
    start_sec=0,
    end_sec=None,
):
    return {
        "start_sec": int(
            start_sec
        ),

        "end_sec": (
            None
            if end_sec is None
            else int(
                end_sec
            )
        ),

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


def update_accumulator(
    accumulator,
    position_losses,
    correct,
    top5_correct,
    target_eeg,
    predicted_eeg,
    oracle_eeg,
    target_tokens,
    predicted_tokens,
):
    accumulator[
        "evaluated_samples"
    ] += 1

    accumulator[
        "positions"
    ] += position_losses.numel()

    accumulator[
        "cross_entropy_sum"
    ] += float(
        position_losses.sum().item()
    )

    accumulator[
        "correct"
    ] += int(
        correct.sum().item()
    )

    accumulator[
        "top5_correct"
    ] += int(
        top5_correct.sum().item()
    )

    mae = torch.abs(
        target_eeg
        - predicted_eeg
    ).flatten(
        start_dim=1
    ).mean(
        dim=1
    )

    corr = sample_correlation(
        target_eeg,
        predicted_eeg,
    )

    temp_corr = temporal_correlation(
        target_eeg,
        predicted_eeg,
    )

    snr = sample_snr(
        target_eeg,
        predicted_eeg,
    )

    oracle_mae = torch.abs(
        target_eeg
        - oracle_eeg
    ).flatten(
        start_dim=1
    ).mean(
        dim=1
    )

    oracle_corr = sample_correlation(
        target_eeg,
        oracle_eeg,
    )

    oracle_temp_corr = temporal_correlation(
        target_eeg,
        oracle_eeg,
    )

    accumulator[
        "mae_sum"
    ] += float(
        mae.item()
    )

    accumulator[
        "correlation_sum"
    ] += float(
        corr.item()
    )

    accumulator[
        "temporal_correlation_sum"
    ] += float(
        temp_corr.item()
    )

    accumulator[
        "snr_sum"
    ] += float(
        snr.item()
    )

    accumulator[
        "oracle_mae_sum"
    ] += float(
        oracle_mae.item()
    )

    accumulator[
        "oracle_correlation_sum"
    ] += float(
        oracle_corr.item()
    )

    accumulator[
        "oracle_temporal_correlation_sum"
    ] += float(
        oracle_temp_corr.item()
    )

    accumulator[
        "target_codes"
    ].update(
        target_tokens
        .detach()
        .cpu()
        .reshape(-1)
        .tolist()
    )

    accumulator[
        "predicted_codes"
    ].update(
        predicted_tokens
        .detach()
        .cpu()
        .reshape(-1)
        .tolist()
    )


def finalize_accumulator(
    accumulator,
):
    samples = accumulator[
        "evaluated_samples"
    ]

    positions = accumulator[
        "positions"
    ]

    if (
        samples < 1
        or positions < 1
    ):
        raise ValueError(
            "Cannot finalize empty accumulator."
        )

    result = {
        "start_sec": (
            accumulator[
                "start_sec"
            ]
        ),

        "start_min": (
            accumulator[
                "start_sec"
            ]
            / 60.0
        ),

        "evaluated_samples": samples,

        "cross_entropy": (
            accumulator[
                "cross_entropy_sum"
            ]
            / positions
        ),

        "token_accuracy": (
            accumulator[
                "correct"
            ]
            / positions
        ),

        "token_top5_accuracy": (
            accumulator[
                "top5_correct"
            ]
            / positions
        ),

        "predicted_eeg_mae": (
            accumulator[
                "mae_sum"
            ]
            / samples
        ),

        "predicted_eeg_correlation": (
            accumulator[
                "correlation_sum"
            ]
            / samples
        ),

        "predicted_eeg_temporal_correlation": (
            accumulator[
                "temporal_correlation_sum"
            ]
            / samples
        ),

        "predicted_eeg_snr_db": (
            accumulator[
                "snr_sum"
            ]
            / samples
        ),

        "oracle_vqgan_mae": (
            accumulator[
                "oracle_mae_sum"
            ]
            / samples
        ),

        "oracle_vqgan_correlation": (
            accumulator[
                "oracle_correlation_sum"
            ]
            / samples
        ),

        "oracle_vqgan_temporal_correlation": (
            accumulator[
                "oracle_temporal_correlation_sum"
            ]
            / samples
        ),

        "unique_target_codes": len(
            accumulator[
                "target_codes"
            ]
        ),

        "unique_predicted_codes": len(
            accumulator[
                "predicted_codes"
            ]
        ),
    }

    if (
        accumulator[
            "end_sec"
        ]
        is not None
    ):
        result[
            "end_sec"
        ] = accumulator[
            "end_sec"
        ]

        result[
            "end_min"
        ] = (
            accumulator[
                "end_sec"
            ]
            / 60.0
        )

    return result


# ============================================================
# Figures
# ============================================================

def save_comparison(
    ground_truth,
    oracle_reconstruction,
    predicted_reconstruction,
    output_path,
    zero_respiration,
    shuffle_respiration,
):
    ground_truth = (
        ground_truth[
            0,
            0,
        ]
        .detach()
        .cpu()
        .numpy()
    )

    oracle_reconstruction = (
        oracle_reconstruction[
            0,
            0,
        ]
        .detach()
        .cpu()
        .numpy()
    )

    predicted_reconstruction = (
        predicted_reconstruction[
            0,
            0,
        ]
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

    figure, axes = plt.subplots(
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
            "Frozen multi-dataset VQGAN "
            "reconstruction from true tokens"
        ),

        prediction_title,

        (
            "Absolute difference: "
            "ground truth vs prediction"
        ),
    ]

    for index, axis in enumerate(
        axes
    ):
        kwargs = {
            "aspect": "auto",
            "origin": "lower",
        }

        if index < 3:
            kwargs["vmin"] = 0
            kwargs["vmax"] = 1

        axis.imshow(
            images[index],
            **kwargs,
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

    plt.close(
        figure
    )


def save_time_block_plot(
    metrics_by_time_block,
    output_path,
):
    values = sorted(
        metrics_by_time_block.values(),
        key=lambda item: item[
            "start_sec"
        ],
    )

    midpoints = [
        (
            item["start_min"]
            + item["end_min"]
        )
        / 2.0
        for item in values
    ]

    predicted = [
        item[
            "predicted_eeg_temporal_correlation"
        ]
        for item in values
    ]

    oracle = [
        item[
            "oracle_vqgan_temporal_correlation"
        ]
        for item in values
    ]

    plt.figure(
        figsize=(10, 5)
    )

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

    plt.xlabel(
        "Absolute time from recording start (minutes)"
    )

    plt.ylabel(
        "Temporal correlation"
    )

    plt.title(
        "Temporal correlation by absolute 32-minute block"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the current multi-dataset "
            "paper respiration-to-EEG Transformer."
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
        default=None,
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

    comparison_group = (
        parser.add_mutually_exclusive_group()
    )

    comparison_group.add_argument(
        "--comparison-index",
        type=int,
        default=None,
        help=(
            "Zero-based evaluation index used for the saved "
            "qualitative comparison. If neither comparison "
            "selector is provided, index 100 is used to "
            "preserve the current evaluator behaviour."
        ),
    )

    comparison_group.add_argument(
        "--comparison-subject-id",
        type=str,
        default=None,
        help=(
            "Subject ID whose reconstruction should be saved "
            "for the qualitative comparison."
        ),
    )

    parser.add_argument(
        "--comparison-window-index",
        type=int,
        default=None,
        help=(
            "Optional window index used together with "
            "--comparison-subject-id. If omitted, the first "
            "available window for that subject is selected."
        ),
    )

    respiration_group = (
        parser
        .add_mutually_exclusive_group()
    )

    respiration_group.add_argument(
        "--zero-respiration",
        action="store_true",
    )

    respiration_group.add_argument(
        "--shuffle-respiration",
        action="store_true",
    )

    args = parser.parse_args()

    if (
        args.comparison_index is None
        and args.comparison_subject_id is None
    ):
        args.comparison_index = 100

    if (
        args.comparison_index is not None
        and args.comparison_index < 0
    ):
        raise ValueError(
            "--comparison-index must be >= 0."
        )

    if (
        args.comparison_window_index is not None
        and args.comparison_window_index < 0
    ):
        raise ValueError(
            "--comparison-window-index must be >= 0."
        )

    if (
        args.comparison_window_index is not None
        and args.comparison_subject_id is None
    ):
        raise ValueError(
            "--comparison-window-index requires "
            "--comparison-subject-id."
        )

    if (
        args.max_samples
        is not None
        and args.max_samples < 1
    ):
        raise ValueError(
            "--max-samples must be positive."
        )

    if not args.transformer_checkpoint.exists():
        raise FileNotFoundError(
            args.transformer_checkpoint
        )

    if not args.vqgan_checkpoint.exists():
        raise FileNotFoundError(
            args.vqgan_checkpoint
        )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    if args.output_dir is None:
        suffix = (
            f"evaluation_{args.split}"
        )

        if args.zero_respiration:
            suffix += "_zero_respiration"

        elif args.shuffle_respiration:
            suffix += "_shuffled_respiration"

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

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Transformer
    # --------------------------------------------------------

    transformer_checkpoint = torch.load(
        args.transformer_checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    model_config = (
        transformer_checkpoint.get(
            "model_config"
        )
    )

    if model_config is None:
        raise KeyError(
            "Transformer checkpoint is missing model_config."
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

    checkpoint_epoch = int(
        transformer_checkpoint[
            "epoch"
        ]
    ) + 1

    print(
        "Transformer checkpoint:",
        args.transformer_checkpoint
    )

    print(
        "Checkpoint epoch:",
        checkpoint_epoch
    )

    print(
        "Architecture:",
        transformer_checkpoint.get(
            "architecture",
            "unknown",
        )
    )

    # --------------------------------------------------------
    # VQGAN
    # --------------------------------------------------------

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

    for parameter in vqgan.parameters():
        parameter.requires_grad = False

    print(
        "VQGAN checkpoint:",
        args.vqgan_checkpoint
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

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
        "Zero respiration:",
        args.zero_respiration
    )

    print(
        "Shuffled respiration:",
        args.shuffle_respiration
    )

    if args.comparison_subject_id is not None:
        print(
            "Qualitative comparison selector:",
            "subject_id=",
            args.comparison_subject_id,
            "window_index=",
            (
                "first available"
                if args.comparison_window_index is None
                else args.comparison_window_index
            ),
        )
    else:
        print(
            "Qualitative comparison selector:",
            "evaluation_index=",
            args.comparison_index,
        )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.type
            == "cuda"
        ),
    )

    # --------------------------------------------------------
    # Accumulators
    # --------------------------------------------------------

    global_accumulator = (
        create_accumulator()
    )

    window_accumulators = {}
    time_block_accumulators = {}

    if (
        MODEL_WINDOW_SEC
        % DIAGNOSTIC_BLOCK_SEC
        != 0
    ):
        raise RuntimeError(
            "Diagnostic block must divide model window."
        )

    blocks_per_window = (
        MODEL_WINDOW_SEC
        // DIAGNOSTIC_BLOCK_SEC
    )

    comparison_saved = False
    comparison_metadata = None
    evaluated_samples = 0

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

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

            if args.zero_respiration:
                respiration = (
                    torch.zeros_like(
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

            # ------------------------------------------------
            # Paper one-shot inference
            # ------------------------------------------------

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(
                    device.type
                    == "cuda"
                ),
            ):
                logits = (
                    transformer
                    .predict_from_respiration(
                        respiration
                    )
                )

            targets_flat = (
                true_tokens
                .flatten(
                    start_dim=1
                )
            )

            position_losses = (
                F.cross_entropy(
                    logits.reshape(
                        -1,
                        logits.shape[-1],
                    ),
                    targets_flat.reshape(
                        -1
                    ),
                    reduction="none",
                )
                .reshape_as(
                    targets_flat
                )
            )

            predicted_flat = (
                logits.argmax(
                    dim=-1
                )
            )

            predicted_tokens = (
                predicted_flat
                .view_as(
                    true_tokens
                )
            )

            top5_predictions = (
                logits.topk(
                    k=5,
                    dim=-1,
                ).indices
            )

            correct = (
                predicted_flat
                == targets_flat
            )

            top5_correct = (
                top5_predictions
                == targets_flat.unsqueeze(
                    -1
                )
            ).any(
                dim=-1
            )

            # ------------------------------------------------
            # VQGAN reconstruction
            # ------------------------------------------------

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
            # Global metrics
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

            # ------------------------------------------------
            # Window metrics
            # ------------------------------------------------

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
                f"window"
                f"{window_index:03d}"
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

            if (
                eeg.shape[-1]
                % blocks_per_window
                != 0
            ):
                raise RuntimeError(
                    "EEG width cannot be split "
                    "into 32-minute blocks."
                )

            if (
                true_tokens.shape[-1]
                % blocks_per_window
                != 0
            ):
                raise RuntimeError(
                    "Token width cannot be split "
                    "into 32-minute blocks."
                )

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
            # Visual comparison
            # ------------------------------------------------

            batch_dataset = str(
                batch["dataset"][0]
            )

            batch_subject_id = str(
                batch["subject_id"][0]
            )

            if args.comparison_subject_id is not None:
                comparison_match = (
                    batch_subject_id
                    == args.comparison_subject_id
                )

                if (
                    comparison_match
                    and args.comparison_window_index
                    is not None
                ):
                    comparison_match = (
                        window_index
                        == args.comparison_window_index
                    )
            else:
                comparison_match = (
                    evaluated_samples
                    == args.comparison_index
                )

            if (
                not comparison_saved
                and comparison_match
            ):

                def to_numpy(x):
                    if torch.is_tensor(x):
                        return x.detach().float().cpu().numpy()
                    return np.asarray(x)

                np.savez_compressed(
                    output_dir / "reconstruction_arrays.npz",
                    ground_truth=to_numpy(eeg),
                    oracle=to_numpy(oracle_reconstruction),
                    prediction=to_numpy(predicted_reconstruction),
                )

                comparison_metadata = {
                    "evaluation_index": int(
                        evaluated_samples
                    ),
                    "dataset": batch_dataset,
                    "subject_id": batch_subject_id,
                    "window_index": int(
                        window_index
                    ),
                    "start_sec": int(
                        start_sec
                    ),
                    "start_min": float(
                        start_sec / 60.0
                    ),
                }

                if (
                    args.shuffle_respiration
                    and "respiration_source_subject"
                    in batch
                ):
                    comparison_metadata[
                        "respiration_source_subject"
                    ] = str(
                        batch[
                            "respiration_source_subject"
                        ][0]
                    )

                comparison_metadata_path = (
                    output_dir
                    / "reconstruction_metadata.json"
                )

                with comparison_metadata_path.open(
                    "w",
                    encoding="utf-8",
                ) as file:
                    json.dump(
                        comparison_metadata,
                        file,
                        indent=2,
                    )

                print(
                    "Reconstruction arrays saved:",
                    output_dir / "reconstruction_arrays.npz",
                )

                print(
                    "Qualitative example:",
                    f"eval_index={evaluated_samples}",
                    f"dataset={batch_dataset}",
                    f"subject_id={batch_subject_id}",
                    f"window_index={window_index}",
                )

                print(
                    "Reconstruction metadata saved:",
                    comparison_metadata_path,
                )

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

            evaluated_samples += 1
            if (
                evaluated_samples
                % 25
                == 0
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

    # --------------------------------------------------------
    # Finalize
    # --------------------------------------------------------

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
                item[1][
                    "start_sec"
                ]
            ),
        )
    }

    metrics = {
        "transformer_checkpoint": str(
            args.transformer_checkpoint
        ),

        "vqgan_checkpoint": str(
            args.vqgan_checkpoint
        ),

        "checkpoint_epoch": checkpoint_epoch,

        "split": args.split,

        "evaluated_samples": evaluated_samples,

        "zero_respiration": (
            args.zero_respiration
        ),

        "shuffle_respiration": (
            args.shuffle_respiration
        ),

        "comparison_selector": {
            "comparison_index": (
                args.comparison_index
            ),
            "comparison_subject_id": (
                args.comparison_subject_id
            ),
            "comparison_window_index": (
                args.comparison_window_index
            ),
        },

        "comparison_saved": (
            comparison_saved
        ),

        "comparison_example": (
            comparison_metadata
        ),

        **{
            key: value
            for key, value
            in global_metrics.items()
            if key not in [
                "start_sec",
                "start_min",
            ]
        },

        "metrics_by_window": (
            metrics_by_window
        ),

        "metrics_by_time_block": (
            metrics_by_time_block
        ),
    }

    metrics_path = (
        output_dir
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

    time_plot_path = (
        output_dir
        / "temporal_correlation_by_time_block.png"
    )

    save_time_block_plot(
        metrics_by_time_block,
        time_plot_path,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 72)

    print(
        "FINAL METRICS"
    )

    print(
        "Samples:",
        evaluated_samples
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
        "Metrics by absolute 32-minute block:"
    )

    for values in (
        metrics_by_time_block.values()
    ):

        print(
            f"  "
            f"{values['start_min']:.0f}-"
            f"{values['end_min']:.0f} min"
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
            f" | oracle_temporal_corr="
            f"{values['oracle_vqgan_temporal_correlation']:.4f}"
            f" | SNR="
            f"{values['predicted_eeg_snr_db']:.2f} dB"
        )

    print()

    if comparison_saved:
        print(
            "Comparison saved:",
            output_dir
            / "reconstruction_comparison.png"
        )

        print(
            "Reconstruction arrays saved:",
            output_dir
            / "reconstruction_arrays.npz"
        )

        print(
            "Reconstruction metadata saved:",
            output_dir
            / "reconstruction_metadata.json"
        )

    else:
        print(
            "Comparison saved: NO - requested "
            "qualitative example was not found "
            "within the evaluated samples."
        )

    print(
        "Metrics saved:",
        metrics_path
    )

    print(
        "Time-block plot saved:",
        time_plot_path
    )


if __name__ == "__main__":
    main()
