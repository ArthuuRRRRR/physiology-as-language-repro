from __future__ import annotations

import argparse
import csv
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ============================================================
# Project
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Reuse the exact evaluation implementation used for schedule75.
import paper_architecture.evaluate_paper_masked_transformer as ev


# ============================================================
# Defaults
# ============================================================

DEFAULT_TRANSFORMER_CHECKPOINT = Path(
    "outputs/paper_masked_transformer_schedule75/checkpoint_best.pt"
)

DEFAULT_VQGAN_CHECKPOINT = Path(
    "outputs/vqgan_multidataset_final_ddp/checkpoint_best.pt"
)

DEFAULT_MAIN_TRANSFORMER_ROOT = Path(
    "outputs/paper_transformer_data_final"
)

DEFAULT_MAIN_PREPROCESSED_ROOT = Path(
    "outputs/vqgan_multidataset_preprocessed_final_gpu4"
)

DEFAULT_TEST_TRANSFORMER_ROOT = Path(
    "outputs/paper_transformer_data_shhs1_test_final"
)

DEFAULT_TEST_PREPROCESSED_ROOT = Path(
    "outputs/vqgan_shhs1_test_preprocessed_final"
)

DEFAULT_TRAIN_IDS = Path(
    "data_splits/shhs1/train.txt"
)

DEFAULT_VAL_IDS = Path(
    "data_splits/intersections/shhs1_validation_filtered.txt"
)

DEFAULT_TEST_IDS = Path(
    "data_splits/intersections/shhs1_wav2sleep_heldout.txt"
)

DEFAULT_LABEL_ROOT = Path(
    "/hdd2/kdpark/sleep_datasets/"
    "shhs1/sleep_staging_labels_5sec"
)

DEFAULT_OUTPUT_ROOT = Path(
    "outputs/downstream_sleep_staging_schedule75"
)


# ============================================================
# Constants
# ============================================================

FREQ_BINS = 256
WINDOW_SEGMENTS = 512
SEGMENT_SECONDS = 30

IGNORE_INDEX = -100

STAGE_NAMES = (
    "Wake",
    "Light",
    "Deep",
    "REM",
)


# SHHS 4-stage mapping used for downstream staging.
#
# Wake -> Wake
# Stage 1 + Stage 2 -> Light
# Stage 3 + Stage 4 -> Deep
# REM -> REM
#
# Extra aliases are included only to make the loader robust to
# equivalent naming conventions.
SHHS1_STAGE_MAP = {
    "Wake": 0,
    "W": 0,
    "wake": 0,

    "Stage 1 sleep": 1,
    "Stage 1": 1,
    "N1": 1,

    "Stage 2 sleep": 1,
    "Stage 2": 1,
    "N2": 1,

    "Stage 3 sleep": 2,
    "Stage 3": 2,
    "N3": 2,

    "Stage 4 sleep": 2,
    "Stage 4": 2,
    "N4": 2,

    "REM sleep": 3,
    "REM": 3,
    "R": 3,
}


# ============================================================
# ID utilities
# ============================================================

def normalize_id(value) -> str:

    value = str(value).strip().lower()

    for prefix in (
        "shhs1-",
        "shhs1_train-",
        "shhs1_val-",
        "shhs1_test-",
    ):
        if value.startswith(prefix):
            value = value[len(prefix):]

    return value


def canonical_shhs1_id(value) -> str:

    return (
        "shhs1-"
        + normalize_id(value)
    )


def is_shhs1(row: dict) -> bool:

    name = str(
        row.get("dataset", "")
    ).lower()

    return name.startswith(
        "shhs1"
    )


def read_subject_ids(
    path: Path,
) -> set[str]:

    values = set()

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            values.add(
                normalize_id(line)
            )

    if not values:
        raise RuntimeError(
            f"No subject IDs found in {path}"
        )

    return values


# ============================================================
# Filtered evaluator dataset
# ============================================================

class SelectedEvaluationDataset(Dataset):

    def __init__(
        self,
        *,
        wanted_ids: set[str],
        transformer_data_root: Path,
        preprocessed_root: Path,
        split: str,
        fallback_transformer_root: Path | None = None,
        fallback_preprocessed_root: Path | None = None,
        fallback_split: str | None = None,
    ):

        super().__init__()

        self.wanted_ids = set(
            wanted_ids
        )

        self.selected = []

        # ----------------------------------------------------
        # Main source
        # ----------------------------------------------------

        self.main_dataset = (
            ev.PaperTransformerEvaluationDataset(
                transformer_data_root=(
                    transformer_data_root
                ),
                preprocessed_root=(
                    preprocessed_root
                ),
                split=split,
            )
        )

        found_subjects = set()

        for index, row in enumerate(
            self.main_dataset.index_rows
        ):

            if not is_shhs1(row):
                continue

            subject_id = normalize_id(
                row["subject_id"]
            )

            if subject_id not in self.wanted_ids:
                continue

            self.selected.append(
                (
                    "main",
                    index,
                    row,
                )
            )

            found_subjects.add(
                subject_id
            )

        main_subjects = set(
            found_subjects
        )

        # ----------------------------------------------------
        # Optional fallback
        #
        # Same logic as your intersection evaluator:
        # if a requested test subject is not available in the
        # dedicated held-out source, recover it from FINAL val.
        # ----------------------------------------------------

        self.fallback_dataset = None

        if (
            fallback_transformer_root is not None
            and fallback_preprocessed_root is not None
            and fallback_split is not None
        ):

            missing = (
                self.wanted_ids
                - found_subjects
            )

            if missing:

                self.fallback_dataset = (
                    ev.PaperTransformerEvaluationDataset(
                        transformer_data_root=(
                            fallback_transformer_root
                        ),
                        preprocessed_root=(
                            fallback_preprocessed_root
                        ),
                        split=fallback_split,
                    )
                )

                for index, row in enumerate(
                    self.fallback_dataset.index_rows
                ):

                    if not is_shhs1(row):
                        continue

                    subject_id = normalize_id(
                        row["subject_id"]
                    )

                    if subject_id not in missing:
                        continue

                    self.selected.append(
                        (
                            "fallback",
                            index,
                            row,
                        )
                    )

                    found_subjects.add(
                        subject_id
                    )

        missing_final = (
            self.wanted_ids
            - found_subjects
        )

        print()
        print("=" * 72)
        print("SUBJECT SELECTION")
        print("=" * 72)

        print(
            "Requested subjects:",
            len(self.wanted_ids),
        )

        print(
            "Found subjects:",
            len(found_subjects),
        )

        print(
            "Main subjects:",
            len(main_subjects),
        )

        print(
            "Total windows:",
            len(self.selected),
        )

        print(
            "Missing subjects:",
            len(missing_final),
        )

        if missing_final:

            print(
                "Missing IDs:",
                sorted(
                    missing_final
                )[:50],
            )

            raise RuntimeError(
                "Some requested SHHS1 subjects "
                "could not be found."
            )

        if not self.selected:

            raise RuntimeError(
                "No SHHS1 windows selected."
            )

        print("=" * 72)

    def __len__(self):

        return len(
            self.selected
        )

    def __getitem__(
        self,
        index,
    ):

        (
            source,
            original_index,
            row,
        ) = self.selected[index]

        if source == "main":

            item = self.main_dataset[
                original_index
            ]

        else:

            if self.fallback_dataset is None:
                raise RuntimeError(
                    "Fallback dataset is not initialized."
                )

            item = self.fallback_dataset[
                original_index
            ]

        item = dict(
            item
        )

        subject_id = canonical_shhs1_id(
            row["subject_id"]
        )

        item[
            "_builder_subject_id"
        ] = subject_id

        item[
            "_builder_subject_norm"
        ] = normalize_id(
            subject_id
        )

        item[
            "_builder_dataset"
        ] = str(
            row.get(
                "dataset",
                "shhs1",
            )
        )

        start_sec = row.get(
            "start_sec",
            item.get(
                "start_sec",
                0,
            ),
        )

        if torch.is_tensor(
            start_sec
        ):
            start_sec = int(
                start_sec.item()
            )

        item[
            "_builder_start_sec"
        ] = int(
            start_sec
        )

        item[
            "_builder_source"
        ] = source

        item[
            "_builder_original_index"
        ] = int(
            original_index
        )

        return item


# ============================================================
# SHHS1 sleep-stage labels
# ============================================================

@lru_cache(maxsize=8192)
def load_shhs1_labels_30sec_cached(
    label_root_string: str,
    subject_id: str,
) -> np.ndarray:

    label_root = Path(
        label_root_string
    )

    subject_id = canonical_shhs1_id(
        subject_id
    )

    path = (
        label_root
        / f"{subject_id}_5sec.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            path
        )

    with path.open(
        newline="",
        encoding="utf-8",
    ) as file:

        rows = list(
            csv.DictReader(
                file
            )
        )

    # --------------------------------------------------------
    # Original SHHS preprocessing:
    # 6 x 5 seconds = one 30-second epoch.
    # --------------------------------------------------------

    if len(rows) % 6 != 0:

        raise RuntimeError(
            f"{subject_id}: 5-s label count "
            f"{len(rows)} is not divisible by 6"
        )

    labels = []

    for start in range(
        0,
        len(rows),
        6,
    ):

        block = rows[
            start:
            start + 6
        ]

        stage_names = [
            row["StageName"]
            for row in block
        ]

        # Exact logic from the old downstream builder:
        # the six 5-s annotations must all represent the same
        # 30-s epoch.
        if len(
            set(stage_names)
        ) != 1:

            raise RuntimeError(
                f"{subject_id}: mixed 30-s block "
                f"at rows {start}:{start + 6}: "
                f"{stage_names}"
            )

        stage = stage_names[0]

        if stage not in SHHS1_STAGE_MAP:

            raise RuntimeError(
                f"{subject_id}: "
                f"unknown stage {stage!r}"
            )

        labels.append(
            SHHS1_STAGE_MAP[
                stage
            ]
        )

    return np.asarray(
        labels,
        dtype=np.int8,
    )


def labels_for_window(
    *,
    label_root: Path,
    subject_id: str,
    start_sec: int,
) -> np.ndarray:

    labels_full = (
        load_shhs1_labels_30sec_cached(
            str(
                label_root.resolve()
            ),
            subject_id,
        )
    )

    if (
        start_sec
        % SEGMENT_SECONDS
        != 0
    ):

        raise ValueError(
            f"{subject_id}: "
            f"start_sec={start_sec} "
            f"is not aligned to "
            f"{SEGMENT_SECONDS} seconds."
        )

    start_epoch = (
        start_sec
        // SEGMENT_SECONDS
    )

    end_epoch = (
        start_epoch
        + WINDOW_SEGMENTS
    )

    result = np.full(
        (
            WINDOW_SEGMENTS,
        ),
        IGNORE_INDEX,
        dtype=np.int8,
    )

    available = labels_full[
        start_epoch:
        end_epoch
    ]

    result[
        :len(available)
    ] = available.astype(
        np.int8
    )

    return result


# ============================================================
# Model loading
# ============================================================

def load_models(
    *,
    transformer_checkpoint_path: Path,
    vqgan_checkpoint_path: Path,
    device: torch.device,
):

    print()
    print("=" * 72)
    print("LOADING MODELS")
    print("=" * 72)

    # --------------------------------------------------------
    # Transformer
    # --------------------------------------------------------

    transformer_checkpoint = torch.load(
        transformer_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if (
        "model_config"
        not in transformer_checkpoint
    ):
        raise KeyError(
            "Transformer checkpoint "
            "is missing model_config."
        )

    if (
        "model_state_dict"
        not in transformer_checkpoint
    ):
        raise KeyError(
            "Transformer checkpoint "
            "is missing model_state_dict."
        )

    transformer = (
        ev.PaperMaskedRespirationToEEGTransformer(
            **transformer_checkpoint[
                "model_config"
            ]
        )
        .to(device)
    )

    transformer.load_state_dict(
        transformer_checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    transformer.eval()

    # --------------------------------------------------------
    # VQGAN
    # --------------------------------------------------------

    vqgan_checkpoint = torch.load(
        vqgan_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    vqgan = (
        ev.VQGAN()
        .to(device)
    )

    # Same possible checkpoint formats supported here.
    if (
        "model_state_dict"
        in vqgan_checkpoint
    ):

        vqgan_state = (
            vqgan_checkpoint[
                "model_state_dict"
            ]
        )

    elif (
        "state_dict"
        in vqgan_checkpoint
    ):

        vqgan_state = (
            vqgan_checkpoint[
                "state_dict"
            ]
        )

    elif "model" in vqgan_checkpoint:

        vqgan_state = (
            vqgan_checkpoint[
                "model"
            ]
        )

    else:

        raise KeyError(
            "Could not locate VQGAN state dict. "
            f"Available keys: "
            f"{list(vqgan_checkpoint.keys())}"
        )

    # Support a checkpoint saved from DDP with module. prefix.
    if any(
        key.startswith(
            "module."
        )
        for key in vqgan_state
    ):

        vqgan_state = {
            key.removeprefix(
                "module."
            ): value
            for key, value
            in vqgan_state.items()
        }

    vqgan.load_state_dict(
        vqgan_state,
        strict=True,
    )

    vqgan.eval()

    for parameter in (
        transformer.parameters()
    ):
        parameter.requires_grad_(
            False
        )

    for parameter in (
        vqgan.parameters()
    ):
        parameter.requires_grad_(
            False
        )

    print(
        "Transformer:",
        transformer_checkpoint_path,
    )

    print(
        "Transformer epoch:",
        transformer_checkpoint.get(
            "epoch",
            "unknown",
        ),
    )

    print(
        "VQGAN:",
        vqgan_checkpoint_path,
    )

    print("=" * 72)

    return (
        transformer,
        vqgan,
    )


# ============================================================
# Shape utilities
# ============================================================

def normalize_eeg_shape(
    tensor: torch.Tensor,
) -> torch.Tensor:

    # Expected output:
    #
    # (B, 256, 512)

    if tensor.ndim == 4:

        if tensor.shape[1] != 1:

            raise ValueError(
                "Expected one EEG channel, "
                f"got {tuple(tensor.shape)}"
            )

        tensor = tensor[
            :,
            0,
        ]

    if tensor.ndim != 3:

        raise ValueError(
            "Expected EEG batch with "
            f"3 dimensions, got "
            f"{tuple(tensor.shape)}"
        )

    if (
        tensor.shape[1]
        == FREQ_BINS
        and tensor.shape[2]
        == WINDOW_SEGMENTS
    ):

        return tensor

    if (
        tensor.shape[1]
        == WINDOW_SEGMENTS
        and tensor.shape[2]
        == FREQ_BINS
    ):

        return tensor.transpose(
            1,
            2,
        )

    raise ValueError(
        "Unexpected EEG shape: "
        f"{tuple(tensor.shape)}"
    )


# ============================================================
# Build one split
# ============================================================

def build_split(
    *,
    split_name: str,
    dataset: Dataset,
    output_root: Path,
    transformer,
    vqgan,
    label_root: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    transformer_checkpoint: Path,
    vqgan_checkpoint: Path,
    subject_list_path: Path,
):

    output_dir = (
        output_root
        / split_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    n_samples = len(
        dataset
    )

    print()
    print("=" * 72)
    print(
        f"BUILDING {split_name.upper()}"
    )
    print("=" * 72)

    print(
        "Windows:",
        n_samples,
    )

    print(
        "Output:",
        output_dir,
    )

    # ========================================================
    # Allocate mmap files
    # ========================================================

    gt_path = (
        output_dir
        / "ground_truth_eeg.mmap"
    )

    synth_path = (
        output_dir
        / "synthesized_eeg.mmap"
    )

    labels_path = (
        output_dir
        / "labels.mmap"
    )

    gt_mmap = np.memmap(
        gt_path,
        mode="w+",
        dtype=np.float16,
        shape=(
            n_samples,
            FREQ_BINS,
            WINDOW_SEGMENTS,
        ),
    )

    synth_mmap = np.memmap(
        synth_path,
        mode="w+",
        dtype=np.float16,
        shape=(
            n_samples,
            FREQ_BINS,
            WINDOW_SEGMENTS,
        ),
    )

    labels_mmap = np.memmap(
        labels_path,
        mode="w+",
        dtype=np.int8,
        shape=(
            n_samples,
            WINDOW_SEGMENTS,
        ),
    )

    # ========================================================
    # Loader
    # ========================================================

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(
            device.type
            == "cuda"
        ),
        persistent_workers=(
            num_workers > 0
        ),
    )

    # ========================================================
    # Index
    # ========================================================

    index_path = (
        output_dir
        / "index.jsonl"
    )

    index_file = (
        index_path.open(
            "w",
            encoding="utf-8",
        )
    )

    write_position = 0

    # ========================================================
    # Inference
    # ========================================================

    with torch.inference_mode():

        for batch_index, batch in enumerate(
            loader,
            start=1,
        ):

            # ------------------------------------------------
            # Respiration
            # ------------------------------------------------

            respiration = (
                batch[
                    "respiration"
                ]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            # EXACT same standardization as Transformer train
            # and evaluator.
            respiration = (
                ev.standardize_respiration(
                    respiration
                )
            )

            # ------------------------------------------------
            # True VQGAN EEG tokens
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Ground-truth spectrogram
            # ------------------------------------------------

            gt = (
                batch[
                    "eeg_spectrogram"
                ]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            gt = normalize_eeg_shape(
                gt
            )

            # ------------------------------------------------
            # Paper one-shot inference:
            #
            # respiration only
            # -> all 512 VQGAN tokens
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

            # ------------------------------------------------
            # VQGAN decoder
            #
            # EXACT same helper used by evaluator.
            # ------------------------------------------------

            synthesized = (
                ev.decode_token_ids(
                    vqgan,
                    predicted_tokens,
                )
            )

            synthesized = (
                normalize_eeg_shape(
                    synthesized
                )
            )

            # ------------------------------------------------
            # CPU arrays
            # ------------------------------------------------

            gt_np = (
                gt
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype(
                    np.float16
                )
            )

            synth_np = (
                synthesized
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype(
                    np.float16
                )
            )

            batch_size_actual = (
                gt_np.shape[0]
            )

            # ------------------------------------------------
            # Builder metadata
            # ------------------------------------------------

            subject_ids = (
                batch[
                    "_builder_subject_id"
                ]
            )

            subject_norms = (
                batch[
                    "_builder_subject_norm"
                ]
            )

            dataset_names = (
                batch[
                    "_builder_dataset"
                ]
            )

            start_secs = (
                batch[
                    "_builder_start_sec"
                ]
            )

            sources = (
                batch[
                    "_builder_source"
                ]
            )

            original_indices = (
                batch[
                    "_builder_original_index"
                ]
            )

            # ------------------------------------------------
            # Write samples
            # ------------------------------------------------

            for local_index in range(
                batch_size_actual
            ):

                global_index = (
                    write_position
                    + local_index
                )

                subject_id = str(
                    subject_ids[
                        local_index
                    ]
                )

                subject_norm = str(
                    subject_norms[
                        local_index
                    ]
                )

                start_sec_value = (
                    start_secs[
                        local_index
                    ]
                )

                if torch.is_tensor(
                    start_sec_value
                ):

                    start_sec = int(
                        start_sec_value.item()
                    )

                else:

                    start_sec = int(
                        start_sec_value
                    )

                # --------------------------------------------
                # Sleep-stage labels corresponding exactly to
                # this 256-minute spectrogram window.
                # --------------------------------------------

                labels = labels_for_window(
                    label_root=(
                        label_root
                    ),
                    subject_id=(
                        subject_id
                    ),
                    start_sec=(
                        start_sec
                    ),
                )

                # --------------------------------------------
                # Write mmap
                # --------------------------------------------

                gt_mmap[
                    global_index
                ] = gt_np[
                    local_index
                ]

                synth_mmap[
                    global_index
                ] = synth_np[
                    local_index
                ]

                labels_mmap[
                    global_index
                ] = labels

                # --------------------------------------------
                # Index metadata
                # --------------------------------------------

                original_index_value = (
                    original_indices[
                        local_index
                    ]
                )

                if torch.is_tensor(
                    original_index_value
                ):

                    original_index_value = int(
                        original_index_value.item()
                    )

                else:

                    original_index_value = int(
                        original_index_value
                    )

                row = {
                    "sample_index":
                        global_index,

                    "dataset":
                        str(
                            dataset_names[
                                local_index
                            ]
                        ),

                    "subject_id":
                        subject_id,

                    "subject_id_normalized":
                        subject_norm,

                    "start_sec":
                        start_sec,

                    "window_index":
                        int(
                            start_sec
                            // (
                                WINDOW_SEGMENTS
                                * SEGMENT_SECONDS
                            )
                        ),

                    "source":
                        str(
                            sources[
                                local_index
                            ]
                        ),

                    "source_index":
                        original_index_value,

                    "valid_segments":
                        int(
                            (
                                labels
                                != IGNORE_INDEX
                            )
                            .sum()
                        ),
                }

                index_file.write(
                    json.dumps(
                        row
                    )
                    + "\n"
                )

            write_position += (
                batch_size_actual
            )

            if (
                batch_index % 50
                == 0
                or write_position
                == n_samples
            ):

                print(
                    f"  {write_position}/"
                    f"{n_samples} windows",
                    flush=True,
                )

    index_file.close()

    # ========================================================
    # Final check
    # ========================================================

    if write_position != n_samples:

        raise RuntimeError(
            f"Expected {n_samples} windows "
            f"but wrote {write_position}."
        )

    # ========================================================
    # Flush mmap
    # ========================================================

    gt_mmap.flush()
    synth_mmap.flush()
    labels_mmap.flush()

    del gt_mmap
    del synth_mmap
    del labels_mmap

    # ========================================================
    # Metadata
    # ========================================================

    metadata = {
        "num_samples":
            n_samples,

        "eeg_shape": [
            n_samples,
            FREQ_BINS,
            WINDOW_SEGMENTS,
        ],

        "label_shape": [
            n_samples,
            WINDOW_SEGMENTS,
        ],

        "eeg_dtype":
            "float16",

        "label_dtype":
            "int8",

        "ignore_index":
            IGNORE_INDEX,

        "class_names":
            list(
                STAGE_NAMES
            ),

        "class_mapping": {
            "Wake": 0,
            "Stage 1 sleep": 1,
            "Stage 2 sleep": 1,
            "Stage 3 sleep": 2,
            "Stage 4 sleep": 2,
            "REM sleep": 3,
        },

        "segment_seconds":
            SEGMENT_SECONDS,

        "window_segments":
            WINDOW_SEGMENTS,

        "window_minutes":
            (
                WINDOW_SEGMENTS
                * SEGMENT_SECONDS
                // 60
            ),

        "dataset":
            "shhs1",

        "split":
            split_name,

        "subject_list":
            str(
                subject_list_path
            ),

        "transformer_checkpoint":
            str(
                transformer_checkpoint
            ),

        "vqgan_checkpoint":
            str(
                vqgan_checkpoint
            ),

        "inference":
            (
                "one-shot respiration-only "
                "predict_from_respiration"
            ),

        "transformer_output":
            "argmax VQGAN token IDs",

        "vqgan_decoding":
            (
                "evaluate_paper_masked_transformer."
                "decode_token_ids"
            ),

        "label_source":
            str(
                label_root
            ),

        "label_original_resolution_sec":
            5,

        "label_downstream_resolution_sec":
            30,
    }

    with (
        output_dir
        / "metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    print()
    print(
        f"{split_name.upper()} DONE"
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build fresh SHHS1 downstream "
            "sleep-staging GT + schedule75 "
            "reconstructed EEG spectrograms."
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
        default=(
            DEFAULT_VQGAN_CHECKPOINT
        ),
    )

    parser.add_argument(
        "--main-transformer-root",
        type=Path,
        default=(
            DEFAULT_MAIN_TRANSFORMER_ROOT
        ),
    )

    parser.add_argument(
        "--main-preprocessed-root",
        type=Path,
        default=(
            DEFAULT_MAIN_PREPROCESSED_ROOT
        ),
    )

    parser.add_argument(
        "--test-transformer-root",
        type=Path,
        default=(
            DEFAULT_TEST_TRANSFORMER_ROOT
        ),
    )

    parser.add_argument(
        "--test-preprocessed-root",
        type=Path,
        default=(
            DEFAULT_TEST_PREPROCESSED_ROOT
        ),
    )

    parser.add_argument(
        "--train-ids",
        type=Path,
        default=(
            DEFAULT_TRAIN_IDS
        ),
    )

    parser.add_argument(
        "--val-ids",
        type=Path,
        default=(
            DEFAULT_VAL_IDS
        ),
    )

    parser.add_argument(
        "--test-ids",
        type=Path,
        default=(
            DEFAULT_TEST_IDS
        ),
    )

    parser.add_argument(
        "--label-root",
        type=Path,
        default=(
            DEFAULT_LABEL_ROOT
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            DEFAULT_OUTPUT_ROOT
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        choices=[
            "train",
            "val",
            "test",
        ],
        default=[
            "train",
            "val",
            "test",
        ],
    )

    args = parser.parse_args()

    # ========================================================
    # Basic path checks
    # ========================================================

    required_paths = [
        args.transformer_checkpoint,
        args.vqgan_checkpoint,
        args.main_transformer_root,
        args.main_preprocessed_root,
        args.train_ids,
        args.val_ids,
        args.test_ids,
        args.label_root,
    ]

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(
                path
            )

    if "test" in args.splits:

        if not (
            args.test_transformer_root.exists()
        ):

            raise FileNotFoundError(
                args.test_transformer_root
            )

        if not (
            args.test_preprocessed_root.exists()
        ):

            raise FileNotFoundError(
                args.test_preprocessed_root
            )

    # --------------------------------------------------------
    # Do not silently overwrite an existing downstream cache.
    # --------------------------------------------------------

    if args.output_root.exists():

        existing = list(
            args.output_root.glob(
                "**/*.mmap"
            )
        )

        if existing:

            raise FileExistsError(
                f"{args.output_root} already contains "
                "mmap files. Delete the requested split "
                "or the directory before rebuilding."
            )

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Splits
    # ========================================================

    train_ids = read_subject_ids(
        args.train_ids
    )

    val_ids = read_subject_ids(
        args.val_ids
    )

    test_ids = read_subject_ids(
        args.test_ids
    )

    # Strict subject-wise split separation.
    train_val_overlap = (
        train_ids
        & val_ids
    )

    train_test_overlap = (
        train_ids
        & test_ids
    )

    val_test_overlap = (
        val_ids
        & test_ids
    )

    if train_val_overlap:

        raise RuntimeError(
            "Train / validation subject overlap: "
            f"{len(train_val_overlap)}"
        )

    if train_test_overlap:

        raise RuntimeError(
            "Train / test subject overlap: "
            f"{len(train_test_overlap)}"
        )

    if val_test_overlap:

        raise RuntimeError(
            "Validation / test subject overlap: "
            f"{len(val_test_overlap)}"
        )

    print()
    print("=" * 72)
    print("FINAL SHHS1 SPLITS")
    print("=" * 72)

    print(
        "Train subjects:",
        len(train_ids),
    )

    print(
        "Validation subjects:",
        len(val_ids),
    )

    print(
        "Held-out test subjects:",
        len(test_ids),
    )

    print("=" * 72)

    # ========================================================
    # Device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    if (
        device.type
        == "cuda"
    ):

        print(
            "GPU:",
            torch.cuda.get_device_name(
                device
            ),
        )

    # ========================================================
    # Models
    # ========================================================

    (
        transformer,
        vqgan,
    ) = load_models(
        transformer_checkpoint_path=(
            args.transformer_checkpoint
        ),
        vqgan_checkpoint_path=(
            args.vqgan_checkpoint
        ),
        device=device,
    )

    # ========================================================
    # TRAIN
    # ========================================================

    if "train" in args.splits:

        train_dataset = (
            SelectedEvaluationDataset(
                wanted_ids=(
                    train_ids
                ),
                transformer_data_root=(
                    args.main_transformer_root
                ),
                preprocessed_root=(
                    args.main_preprocessed_root
                ),
                split="train",
            )
        )

        build_split(
            split_name="train",
            dataset=train_dataset,
            output_root=(
                args.output_root
            ),
            transformer=transformer,
            vqgan=vqgan,
            label_root=(
                args.label_root
            ),
            device=device,
            batch_size=(
                args.batch_size
            ),
            num_workers=(
                args.num_workers
            ),
            transformer_checkpoint=(
                args.transformer_checkpoint
            ),
            vqgan_checkpoint=(
                args.vqgan_checkpoint
            ),
            subject_list_path=(
                args.train_ids
            ),
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    if "val" in args.splits:

        val_dataset = (
            SelectedEvaluationDataset(
                wanted_ids=(
                    val_ids
                ),
                transformer_data_root=(
                    args.main_transformer_root
                ),
                preprocessed_root=(
                    args.main_preprocessed_root
                ),
                split="val",
            )
        )

        build_split(
            split_name="val",
            dataset=val_dataset,
            output_root=(
                args.output_root
            ),
            transformer=transformer,
            vqgan=vqgan,
            label_root=(
                args.label_root
            ),
            device=device,
            batch_size=(
                args.batch_size
            ),
            num_workers=(
                args.num_workers
            ),
            transformer_checkpoint=(
                args.transformer_checkpoint
            ),
            vqgan_checkpoint=(
                args.vqgan_checkpoint
            ),
            subject_list_path=(
                args.val_ids
            ),
        )

    # ========================================================
    # HELD-OUT TEST
    #
    # Same principle as your intersection utility:
    #
    # 1. dedicated held-out test source
    # 2. recover requested IDs from final validation if needed
    # ========================================================

    if "test" in args.splits:

        test_dataset = (
            SelectedEvaluationDataset(
                wanted_ids=(
                    test_ids
                ),
                transformer_data_root=(
                    args.test_transformer_root
                ),
                preprocessed_root=(
                    args.test_preprocessed_root
                ),
                split="test",

                fallback_transformer_root=(
                    args.main_transformer_root
                ),

                fallback_preprocessed_root=(
                    args.main_preprocessed_root
                ),

                fallback_split="val",
            )
        )

        build_split(
            split_name="test",
            dataset=test_dataset,
            output_root=(
                args.output_root
            ),
            transformer=transformer,
            vqgan=vqgan,
            label_root=(
                args.label_root
            ),
            device=device,
            batch_size=(
                args.batch_size
            ),
            num_workers=(
                args.num_workers
            ),
            transformer_checkpoint=(
                args.transformer_checkpoint
            ),
            vqgan_checkpoint=(
                args.vqgan_checkpoint
            ),
            subject_list_path=(
                args.test_ids
            ),
        )

    # ========================================================
    # Done
    # ========================================================

    print()
    print("=" * 72)
    print(
        "DOWNSTREAM DATASET COMPLETE"
    )
    print("=" * 72)

    print(
        "Output:",
        args.output_root,
    )

    print(
        "Transformer:",
        args.transformer_checkpoint,
    )

    print(
        "VQGAN:",
        args.vqgan_checkpoint,
    )

    print("=" * 72)


if __name__ == "__main__":
    main()