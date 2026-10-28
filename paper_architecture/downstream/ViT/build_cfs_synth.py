from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import paper_architecture.evaluate_paper_masked_transformer as ev
from paper_architecture.downstream.build_sleep_staging_shhs1 import load_models


# ============================================================
# Paths
# ============================================================

TRANSFORMER_ROOT = Path(
    "outputs/cfs_all_vit/transformer_data/all"
)

TRANSFORMER_CHECKPOINT = Path(
    "outputs/paper_masked_transformer_resume140_mae_every1/"
    "snapshots/checkpoint_epoch146.pt"
)

VQGAN_CHECKPOINT = Path(
    "outputs/vqgan_multidataset_final_ddp/checkpoint_best.pt"
)

CFS_SPLITS = (
    Path.home()
    / "wav2sleep/src/wav2sleep/data/splits/cfs"
)

CFS_XML_ROOT = Path(
    "/hdd2/kdpark/sleep_datasets/CFS_raw/"
    "polysomnography/annotations-events-nsrr"
)

OUTPUT_ROOT = Path(
    "outputs/downstream_ViT_synth/cfs"
)

FREQ_BINS = 256
WINDOW_SEGMENTS = 512
IGNORE_INDEX = -100


# ============================================================
# IDs
# ============================================================

def normalize_id(value: str) -> str:

    x = str(value).strip().lower()
    x = Path(x).stem

    for prefix in (
        "cfs-visit5-",
        "cfs_visit5_",
        "cfs-",
    ):
        if x.startswith(prefix):
            x = x[len(prefix):]

    return x


def load_split_ids(
    split: str,
) -> set[str]:

    path = (
        CFS_SPLITS
        / f"{split}.txt"
    )

    return {
        normalize_id(x)
        for x in path.read_text().splitlines()
        if x.strip()
    }


# ============================================================
# NSRR XML sleep stages
# ============================================================

def clean_tag(tag: str) -> str:
    return tag.split("}")[-1]


def stage_to_class(
    concept: str,
) -> int | None:

    s = concept.strip().lower()

    # Wake
    if (
        "sleep stage w" in s
        or "wake" in s
    ):
        return 0

    # Light = N1 + N2
    if (
        "sleep stage 1" in s
        or "stage 1 sleep" in s
        or "n1" in s
    ):
        return 1

    if (
        "sleep stage 2" in s
        or "stage 2 sleep" in s
        or "n2" in s
    ):
        return 1

    # Deep = N3 + N4
    if (
        "sleep stage 3" in s
        or "stage 3 sleep" in s
        or "n3" in s
    ):
        return 2

    if (
        "sleep stage 4" in s
        or "stage 4 sleep" in s
        or "n4" in s
    ):
        return 2

    # REM
    if (
        "sleep stage r" in s
        or "rem" in s
    ):
        return 3

    return None


def load_cfs_labels(
    subject_id: str,
) -> np.ndarray:

    sid = normalize_id(
        subject_id
    )

    xml_path = (
        CFS_XML_ROOT
        / f"cfs-visit5-{sid}-nsrr.xml"
    )

    if not xml_path.exists():
        raise FileNotFoundError(
            xml_path
        )

    root = ET.parse(
        xml_path
    ).getroot()

    events = []

    for element in root.iter():

        if clean_tag(
            element.tag
        ) != "ScoredEvent":
            continue

        values = {}

        for child in element:

            values[
                clean_tag(child.tag)
            ] = child.text

        concept = (
            values.get("EventConcept")
            or values.get("Name")
            or ""
        )

        stage = stage_to_class(
            concept
        )

        if stage is None:
            continue

        try:
            start = float(
                values["Start"]
            )

            duration = float(
                values["Duration"]
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

        events.append(
            (
                start,
                duration,
                stage,
            )
        )

    if not events:
        raise RuntimeError(
            f"No sleep stages found: {xml_path}"
        )

    max_end = max(
        start + duration
        for start, duration, _
        in events
    )

    n_epochs = int(
        np.ceil(
            max_end / 30.0
        )
    )

    labels = np.full(
        n_epochs,
        IGNORE_INDEX,
        dtype=np.int8,
    )

    for (
        start,
        duration,
        stage,
    ) in events:

        first = int(
            round(
                start / 30.0
            )
        )

        count = max(
            1,
            int(
                round(
                    duration / 30.0
                )
            ),
        )

        end = min(
            first + count,
            n_epochs,
        )

        labels[
            first:end
        ] = stage

    return labels


# ============================================================
# Synthesis
# ============================================================

@torch.inference_mode()
def synthesize(
    transformer,
    vqgan,
    respiration_np: np.ndarray,
    device: torch.device,
) -> np.ndarray:

    respiration = (
        torch.from_numpy(
            respiration_np
        )
        .float()
        .to(
            device,
            non_blocking=True,
        )
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
            device.type == "cuda"
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
            .argmax(dim=-1)
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
        reconstruction[:, 0]
        .float()
        .clamp(0.0, 1.0)
        .cpu()
        .numpy()
    )

    return reconstruction.astype(
        np.float32,
        copy=False,
    )


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    index_path = (
        TRANSFORMER_ROOT
        / "index.jsonl"
    )

    rows = []

    with index_path.open() as f:
        for line in f:
            rows.append(
                json.loads(line)
            )

    n_all = len(rows)

    respiration = np.memmap(
        TRANSFORMER_ROOT
        / "respiration.mmap",
        dtype=np.float32,
        mode="r",
        shape=(
            n_all,
            64,
            2400,
        ),
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
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

    split_ids = {
        split: load_split_ids(
            split
        )
        for split in (
            "train",
            "val",
            "test",
        )
    }

    # --------------------------------------------------------
    # Assign each CFS window to wav2sleep split
    # --------------------------------------------------------

    split_rows = {
        "train": [],
        "val": [],
        "test": [],
    }

    for i, row in enumerate(
        rows
    ):

        sid = normalize_id(
            row["subject_id"]
        )

        for split in (
            "train",
            "val",
            "test",
        ):

            if sid in split_ids[split]:

                split_rows[
                    split
                ].append(
                    (
                        i,
                        row,
                    )
                )

                break

    print()
    print("=" * 72)
    print("CFS WINDOWS")
    print("=" * 72)

    for split in (
        "train",
        "val",
        "test",
    ):

        subjects = {
            normalize_id(
                row["subject_id"]
            )
            for _, row
            in split_rows[split]
        }

        print(
            f"{split:5s}: "
            f"{len(subjects):4d} subjects | "
            f"{len(split_rows[split]):4d} windows"
        )

    # --------------------------------------------------------
    # Build each split
    # --------------------------------------------------------

    label_cache = {}

    generation_batch_size = 8

    for split in (
        "train",
        "val",
        "test",
    ):

        selected = (
            split_rows[split]
        )

        out_dir = (
            OUTPUT_ROOT
            / split
        )

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        n = len(selected)

        synth_mmap = np.memmap(
            out_dir
            / "synthesized_eeg.mmap",
            dtype=np.float16,
            mode="w+",
            shape=(
                n,
                FREQ_BINS,
                WINDOW_SEGMENTS,
            ),
        )

        labels_mmap = np.memmap(
            out_dir
            / "labels.mmap",
            dtype=np.int8,
            mode="w+",
            shape=(
                n,
                WINDOW_SEGMENTS,
            ),
        )

        labels_mmap[:] = (
            IGNORE_INDEX
        )

        output_index = (
            out_dir
            / "index.jsonl"
        )

        with output_index.open(
            "w"
        ) as fout:

            for batch_start in range(
                0,
                n,
                generation_batch_size,
            ):

                batch_items = (
                    selected[
                        batch_start:
                        batch_start
                        + generation_batch_size
                    ]
                )

                source_indices = [
                    source_i
                    for source_i, _
                    in batch_items
                ]

                resp_batch = np.asarray(
                    respiration[
                        source_indices
                    ],
                    dtype=np.float32,
                )

                synth = synthesize(
                    transformer,
                    vqgan,
                    resp_batch,
                    device,
                )

                for j, (
                    source_i,
                    row,
                ) in enumerate(
                    batch_items
                ):

                    out_i = (
                        batch_start
                        + j
                    )

                    synth_mmap[
                        out_i
                    ] = synth[j]

                    sid = normalize_id(
                        row["subject_id"]
                    )

                    if sid not in label_cache:
                        label_cache[
                            sid
                        ] = load_cfs_labels(
                            sid
                        )

                    full_labels = (
                        label_cache[sid]
                    )

                    start_sec = int(
                        row.get(
                            "start_sec",
                            int(
                                row.get(
                                    "window_index",
                                    0,
                                )
                            )
                            * 512
                            * 30,
                        )
                    )

                    start_epoch = (
                        start_sec // 30
                    )

                    end_epoch = (
                        start_epoch
                        + WINDOW_SEGMENTS
                    )

                    available = (
                        full_labels[
                            start_epoch:
                            end_epoch
                        ]
                    )

                    valid = min(
                        len(available),
                        WINDOW_SEGMENTS,
                    )

                    if valid > 0:
                        labels_mmap[
                            out_i,
                            :valid,
                        ] = (
                            available[
                                :valid
                            ]
                        )

                    index_row = dict(
                        row
                    )

                    index_row[
                        "sample_index"
                    ] = out_i

                    index_row[
                        "subject_id_normalized"
                    ] = sid

                    index_row[
                        "valid_segments"
                    ] = int(
                        np.sum(
                            labels_mmap[
                                out_i
                            ]
                            != IGNORE_INDEX
                        )
                    )

                    fout.write(
                        json.dumps(
                            index_row
                        )
                        + "\n"
                    )

                done = min(
                    batch_start
                    + generation_batch_size,
                    n,
                )

                if (
                    done % 100 == 0
                    or done == n
                ):
                    print(
                        f"{split}: "
                        f"{done}/{n}"
                    )

        synth_mmap.flush()
        labels_mmap.flush()

        metadata = {
            "num_samples": n,
            "eeg_shape": [
                n,
                256,
                512,
            ],
            "label_shape": [
                n,
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
            "class_names": [
                "Wake",
                "Light",
                "Deep",
                "REM",
            ],
            "dataset": "cfs",
            "split": split,
            "segment_seconds": 30,
            "window_segments": 512,
            "transformer_checkpoint": str(
                TRANSFORMER_CHECKPOINT
            ),
            "vqgan_checkpoint": str(
                VQGAN_CHECKPOINT
            ),
            "input_source": str(
                TRANSFORMER_ROOT
            ),
        }

        (
            out_dir
            / "metadata.json"
        ).write_text(
            json.dumps(
                metadata,
                indent=2,
            )
        )

        print(
            split,
            "DONE"
        )

    print()
    print("=" * 72)
    print("CFS SYNTH DOWNSTREAM DONE")
    print("=" * 72)
    print(OUTPUT_ROOT)


if __name__ == "__main__":
    main()
