from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import paper_architecture.evaluate_paper_masked_transformer as ev

from paper_architecture.downstream.build_sleep_staging_shhs1 import (
    load_models,
)


# ============================================================
# Fixed configuration
# ============================================================

TRANSFORMER_CHECKPOINT = Path(
    "outputs/paper_masked_transformer_resume140_mae_every1/"
    "snapshots/checkpoint_epoch146.pt"
)

VQGAN_CHECKPOINT = Path(
    "outputs/vqgan_multidataset_final_ddp/"
    "checkpoint_best.pt"
)

MAIN_TRANSFORMER_ROOT = Path(
    "outputs/paper_transformer_data_final"
)

MAIN_PREPROCESSED_ROOT = Path(
    "outputs/vqgan_multidataset_preprocessed_final_gpu4"
)

OUTPUT_ROOT = Path(
    "outputs/downstream_ViT_synth"
)


# Existing completed datasets
SHHS1_EXISTING = Path(
    "outputs/downstream_sleep_staging_epoch146_resume140"
)

CFS_EXISTING = Path(
    "outputs/downstream_ViT_synth/cfs"
)


# ============================================================
# MESA configuration
# ============================================================

MESA_LABEL_ROOT = Path(
    "/hdd2/kdpark/sleep_datasets/mesa/annotations"
)

MESA_SPLITS = {
    "train": Path(
        "data_splits/mesa/train.txt"
    ),

    "val": Path(
        "data_splits/intersections/"
        "mesa_validation_filtered.txt"
    ),

    "test": Path(
        "data_splits/intersections/"
        "mesa_wav2sleep_heldout.txt"
    ),
}


FREQ_BINS = 256
WINDOW_SEGMENTS = 512
IGNORE_INDEX = -100

STAGE_NAMES = (
    "Wake",
    "Light",
    "Deep",
    "REM",
)

# Existing project convention:
#
# 0 = Wake
# 1 = N1
# 2 = N2
# 3 = N3
# 4 = REM
#
# Downstream:
# Wake / Light / Deep / REM
MESA_STAGE_MAP = np.asarray(
    [
        0,  # Wake
        1,  # N1 -> Light
        1,  # N2 -> Light
        2,  # N3 -> Deep
        3,  # REM
    ],
    dtype=np.int8,
)


# ============================================================
# Utilities
# ============================================================

def normalize_mesa_id(
    value,
) -> str:

    value = str(value).strip().lower()

    value = Path(value).stem

    for prefix in (
        "mesa-sleep-",
        "mesa_sleep_",
        "mesa-",
    ):
        if value.startswith(prefix):
            value = value[len(prefix):]

    return value


def load_subject_ids(
    path: Path,
) -> set[str]:

    values = set()

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            values.add(
                normalize_mesa_id(
                    line
                )
            )

    if not values:
        raise RuntimeError(
            f"No subject IDs found: {path}"
        )

    return values


# ============================================================
# Selected MESA dataset
# ============================================================

class SelectedMesaDataset(Dataset):

    def __init__(
        self,
        entries,
    ):
        super().__init__()

        self.entries = list(entries)

    def __len__(self):
        return len(self.entries)

    def __getitem__(
        self,
        index,
    ):
        dataset, source_index = (
            self.entries[index]
        )

        return dataset[
            source_index
        ]


def select_from_source(
    dataset,
    wanted_ids,
    *,
    exclude_subjects=None,
):

    if exclude_subjects is None:
        exclude_subjects = set()

    entries = []
    subjects = set()

    for index, row in enumerate(
        dataset.index_rows
    ):

        dataset_name = str(
            row["dataset"]
        ).lower()

        if not dataset_name.startswith(
            "mesa"
        ):
            continue

        sid = normalize_mesa_id(
            row["subject_id"]
        )

        if sid not in wanted_ids:
            continue

        if sid in exclude_subjects:
            continue

        entries.append(
            (
                dataset,
                index,
            )
        )

        subjects.add(
            sid
        )

    return (
        entries,
        subjects,
    )


def build_mesa_split_dataset(
    split_name,
):

    wanted_ids = load_subject_ids(
        MESA_SPLITS[
            split_name
        ]
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    if split_name == "train":

        source = (
            ev.PaperTransformerEvaluationDataset(
                transformer_data_root=(
                    MAIN_TRANSFORMER_ROOT
                ),
                preprocessed_root=(
                    MAIN_PREPROCESSED_ROOT
                ),
                split="train",
            )
        )

        entries, subjects = (
            select_from_source(
                source,
                wanted_ids,
            )
        )

        print()
        print(
            "MESA TRAIN:"
        )
        print(
            "requested subjects:",
            len(wanted_ids),
        )
        print(
            "found subjects    :",
            len(subjects),
        )
        print(
            "windows           :",
            len(entries),
        )

        return (
            SelectedMesaDataset(
                entries
            ),
            subjects,
        )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if split_name == "val":

        source = (
            ev.PaperTransformerEvaluationDataset(
                transformer_data_root=(
                    MAIN_TRANSFORMER_ROOT
                ),
                preprocessed_root=(
                    MAIN_PREPROCESSED_ROOT
                ),
                split="val",
            )
        )

        entries, subjects = (
            select_from_source(
                source,
                wanted_ids,
            )
        )

        print()
        print(
            "MESA VAL:"
        )
        print(
            "requested subjects:",
            len(wanted_ids),
        )
        print(
            "found subjects    :",
            len(subjects),
        )
        print(
            "windows           :",
            len(entries),
        )

        return (
            SelectedMesaDataset(
                entries
            ),
            subjects,
        )

    # --------------------------------------------------------
    # TEST
    #
    # Important:
    # Same construction already used by the intersection
    # evaluator:
    #
    #   main = MESA test
    #   fallback = MESA val
    #
    # Expected:
    #   29 subjects from test
    #   33 subjects added from val
    #   62 subjects total
    #   97 windows total
    # --------------------------------------------------------

    if split_name == "test":

        main_source = (
            ev.PaperTransformerEvaluationDataset(
                transformer_data_root=(
                    MAIN_TRANSFORMER_ROOT
                ),
                preprocessed_root=(
                    MAIN_PREPROCESSED_ROOT
                ),
                split="test",
            )
        )

        val_source = (
            ev.PaperTransformerEvaluationDataset(
                transformer_data_root=(
                    MAIN_TRANSFORMER_ROOT
                ),
                preprocessed_root=(
                    MAIN_PREPROCESSED_ROOT
                ),
                split="val",
            )
        )

        (
            main_entries,
            main_subjects,
        ) = select_from_source(
            main_source,
            wanted_ids,
        )

        (
            val_entries,
            val_subjects,
        ) = select_from_source(
            val_source,
            wanted_ids,
            exclude_subjects=(
                main_subjects
            ),
        )

        entries = (
            main_entries
            + val_entries
        )

        subjects = (
            main_subjects
            | val_subjects
        )

        missing = (
            wanted_ids
            - subjects
        )

        print()
        print("=" * 72)
        print(
            "MESA TEST INTERSECTION"
        )
        print("=" * 72)

        print(
            "Subjects requested :",
            len(wanted_ids),
        )

        print(
            "Main subjects      :",
            len(main_subjects),
        )

        print(
            "Val subjects added :",
            len(val_subjects),
        )

        print(
            "Subjects found     :",
            len(subjects),
        )

        print(
            "Main windows       :",
            len(main_entries),
        )

        print(
            "Val windows added  :",
            len(val_entries),
        )

        print(
            "Total windows      :",
            len(entries),
        )

        print(
            "Missing subjects   :",
            len(missing),
        )

        if missing:
            raise RuntimeError(
                "Missing MESA held-out subjects: "
                + ", ".join(
                    sorted(missing)
                )
            )

        return (
            SelectedMesaDataset(
                entries
            ),
            subjects,
        )

    raise ValueError(
        split_name
    )


# ============================================================
# MESA labels
# ============================================================

def load_mesa_labels(
    subject_id,
    cache,
):

    sid = str(
        subject_id
    )

    if sid in cache:
        return cache[sid]

    path = (
        MESA_LABEL_ROOT
        / f"{sid}_ann.npy"
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    labels_5class = (
        np.load(
            path,
            allow_pickle=False,
        )
        .astype(
            np.int64,
            copy=False,
        )
        .reshape(-1)
    )

    if labels_5class.size == 0:
        raise RuntimeError(
            f"Empty labels: {path}"
        )

    if (
        labels_5class.min() < 0
        or labels_5class.max() > 4
    ):
        raise RuntimeError(
            f"Unexpected MESA labels "
            f"{np.unique(labels_5class)} "
            f"in {path}"
        )

    labels_4class = (
        MESA_STAGE_MAP[
            labels_5class
        ]
    )

    cache[sid] = (
        labels_4class
    )

    return (
        labels_4class
    )


# ============================================================
# Synthesis
# ============================================================

@torch.inference_mode()
def synthesize_batch(
    transformer,
    vqgan,
    respiration,
    device,
):

    respiration = (
        respiration
        .to(
            device,
            non_blocking=True,
        )
        .float()
    )

    respiration = (
        ev.standardize_respiration(
            respiration
        )
    )

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

        tokens = (
            logits
            .argmax(
                dim=-1
            )
            .view(
                respiration.shape[0],
                8,
                64,
            )
        )

        reconstruction = (
            ev.decode_token_ids(
                vqgan,
                tokens,
            )
        )

    reconstruction = (
        reconstruction[
            :,
            0,
        ]
        .float()
        .clamp(
            0.0,
            1.0,
        )
        .cpu()
        .numpy()
        .astype(
            np.float32,
            copy=False,
        )
    )

    return reconstruction


# ============================================================
# Build MESA downstream split
# ============================================================

def build_mesa_split(
    *,
    split_name,
    dataset,
    transformer,
    vqgan,
    device,
    batch_size,
):

    output_dir = (
        OUTPUT_ROOT
        / "mesa"
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
        f"BUILD MESA {split_name.upper()}"
    )
    print("=" * 72)

    print(
        "Windows:",
        n_samples,
    )

    synth_mmap = np.memmap(
        output_dir
        / "synthesized_eeg.mmap",
        dtype=np.float16,
        mode="w+",
        shape=(
            n_samples,
            FREQ_BINS,
            WINDOW_SEGMENTS,
        ),
    )

    labels_mmap = np.memmap(
        output_dir
        / "labels.mmap",
        dtype=np.int8,
        mode="w+",
        shape=(
            n_samples,
            WINDOW_SEGMENTS,
        ),
    )

    labels_mmap[:] = (
        IGNORE_INDEX
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type
            == "cuda"
        ),
    )

    labels_cache = {}

    output_index = (
        output_dir
        / "index.jsonl"
    )

    sample_index = 0

    with output_index.open(
        "w",
        encoding="utf-8",
    ) as index_file:

        for batch in loader:

            synth = (
                synthesize_batch(
                    transformer,
                    vqgan,
                    batch[
                        "respiration"
                    ],
                    device,
                )
            )

            batch_size_actual = (
                synth.shape[0]
            )

            for local_index in range(
                batch_size_actual
            ):

                sid = (
                    batch[
                        "subject_id"
                    ][
                        local_index
                    ]
                )

                window_index = int(
                    batch[
                        "window_index"
                    ][
                        local_index
                    ]
                    .item()
                )

                all_labels = (
                    load_mesa_labels(
                        sid,
                        labels_cache,
                    )
                )

                label_start = (
                    window_index
                    * WINDOW_SEGMENTS
                )

                label_end = (
                    label_start
                    + WINDOW_SEGMENTS
                )

                window_labels = (
                    all_labels[
                        label_start:
                        label_end
                    ]
                )

                # Same fixed-window convention as the
                # existing preprocessing.
                if (
                    len(window_labels)
                    != WINDOW_SEGMENTS
                ):
                    raise RuntimeError(
                        f"{sid} "
                        f"window={window_index}: "
                        f"expected 512 labels, "
                        f"got {len(window_labels)}"
                    )

                synth_mmap[
                    sample_index
                ] = (
                    synth[
                        local_index
                    ]
                    .astype(
                        np.float16
                    )
                )

                labels_mmap[
                    sample_index
                ] = (
                    window_labels
                )

                row = {
                    "sample_index": (
                        sample_index
                    ),
                    "dataset": (
                        "mesa"
                    ),
                    "source_dataset": (
                        batch[
                            "dataset"
                        ][
                            local_index
                        ]
                    ),
                    "subject_id": (
                        sid
                    ),
                    "subject_id_normalized": (
                        normalize_mesa_id(
                            sid
                        )
                    ),
                    "window_index": (
                        window_index
                    ),
                    "start_sec": (
                        window_index
                        * 512
                        * 30
                    ),
                    "valid_segments": (
                        WINDOW_SEGMENTS
                    ),
                }

                index_file.write(
                    json.dumps(
                        row
                    )
                    + "\n"
                )

                sample_index += 1

            if (
                sample_index % 100
                < batch_size_actual
                or sample_index
                == n_samples
            ):
                print(
                    f"{split_name}: "
                    f"{sample_index}/"
                    f"{n_samples}"
                )

    synth_mmap.flush()
    labels_mmap.flush()

    del synth_mmap
    del labels_mmap

    metadata = {
        "num_samples": (
            n_samples
        ),
        "eeg_shape": [
            n_samples,
            256,
            512,
        ],
        "label_shape": [
            n_samples,
            512,
        ],
        "eeg_dtype": (
            "float16"
        ),
        "label_dtype": (
            "int8"
        ),
        "ignore_index": (
            IGNORE_INDEX
        ),
        "class_names": (
            list(
                STAGE_NAMES
            )
        ),
        "class_mapping": {
            "Wake": 0,
            "N1": 1,
            "N2": 1,
            "N3": 2,
            "REM": 3,
        },
        "segment_seconds": 30,
        "window_segments": 512,
        "window_minutes": 256,
        "dataset": "mesa",
        "split": split_name,
        "subject_list": str(
            MESA_SPLITS[
                split_name
            ]
        ),
        "transformer_checkpoint": str(
            TRANSFORMER_CHECKPOINT
        ),
        "vqgan_checkpoint": str(
            VQGAN_CHECKPOINT
        ),
        "inference": (
            "one-shot respiration-only "
            "predict_from_respiration"
        ),
        "transformer_output": (
            "argmax VQGAN token IDs"
        ),
        "vqgan_decoding": (
            "evaluate_paper_masked_transformer."
            "decode_token_ids"
        ),
        "label_source": str(
            MESA_LABEL_ROOT
        ),
        "label_original_resolution_sec": 30,
        "label_downstream_resolution_sec": 30,
    }

    (
        output_dir
        / "metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    print(
        split_name,
        "DONE"
    )


# ============================================================
# Existing SHHS1 / CFS registration
# ============================================================

def prepare_existing_layout():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    shhs_target = (
        OUTPUT_ROOT
        / "shhs1"
    )

    # SHHS1 epoch146 has already been generated.
    # Do not regenerate several thousand windows.
    if not shhs_target.exists():

        if not SHHS1_EXISTING.exists():
            raise FileNotFoundError(
                SHHS1_EXISTING
            )

        shhs_target.symlink_to(
            SHHS1_EXISTING.resolve(),
            target_is_directory=True,
        )

        print(
            "SHHS1 linked:",
            shhs_target,
            "->",
            SHHS1_EXISTING,
        )

    else:
        print(
            "SHHS1 already ready:",
            shhs_target,
        )

    # CFS was built directly at the final location.
    if not CFS_EXISTING.exists():
        raise FileNotFoundError(
            CFS_EXISTING
        )

    print(
        "CFS already ready:",
        CFS_EXISTING,
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--mesa-only",
        action="store_true",
    )

    args = parser.parse_args()

    print()
    print("=" * 72)
    print("ViT DOWNSTREAM DATA BUILD")
    print("=" * 72)

    print(
        "Transformer:",
        TRANSFORMER_CHECKPOINT,
    )

    print(
        "VQGAN:",
        VQGAN_CHECKPOINT,
    )

    print()

    if not args.mesa_only:
        prepare_existing_layout()

    mesa_root = (
        OUTPUT_ROOT
        / "mesa"
    )

    if mesa_root.exists():
        raise FileExistsError(
            f"{mesa_root} already exists. "
            "Remove it only if you want to rebuild MESA."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(
                device
            ),
        )

    transformer, vqgan = (
        load_models(
            transformer_checkpoint_path=(
                TRANSFORMER_CHECKPOINT
            ),
            vqgan_checkpoint_path=(
                VQGAN_CHECKPOINT
            ),
            device=device,
        )
    )

    for split_name in (
        "train",
        "val",
        "test",
    ):

        (
            dataset,
            subjects,
        ) = (
            build_mesa_split_dataset(
                split_name
            )
        )

        build_mesa_split(
            split_name=split_name,
            dataset=dataset,
            transformer=transformer,
            vqgan=vqgan,
            device=device,
            batch_size=args.batch_size,
        )

    print()
    print("=" * 72)
    print("ALL THREE DATASETS READY")
    print("=" * 72)

    print(
        OUTPUT_ROOT
        / "shhs1"
    )

    print(
        OUTPUT_ROOT
        / "mesa"
    )

    print(
        OUTPUT_ROOT
        / "cfs"
    )


if __name__ == "__main__":
    main()
