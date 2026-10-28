from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


DATA_ROOT = Path(
    "outputs/downstream_sleep_staging_epoch146_resume140"
)

OUTPUT_ROOT = Path(
    "outputs/downstream_survival_features_synth_epoch146"
)

SPLITS = [
    "train",
    "val",
    "test",
]


def build_split(split: str) -> None:
    split_dir = DATA_ROOT / split

    with (
        split_dir / "metadata.json"
    ).open(
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    with (
        split_dir / "index.jsonl"
    ).open(
        "r",
        encoding="utf-8",
    ) as f:
        index_rows = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    shape = tuple(
        metadata["eeg_shape"]
    )

    dtype = np.dtype(
        metadata["eeg_dtype"]
    )

    if shape[1:] != (
        256,
        512,
    ):
        raise RuntimeError(
            f"{split}: unexpected EEG shape {shape}"
        )

    if len(index_rows) != shape[0]:
        raise RuntimeError(
            f"{split}: index length "
            f"{len(index_rows)} != {shape[0]}"
        )

    eeg = np.memmap(
        split_dir
        / "synthesized_eeg.mmap",
        mode="r",
        dtype=dtype,
        shape=shape,
    )

    by_subject = defaultdict(
        list
    )

    for position, row in enumerate(
        index_rows
    ):
        subject = str(
            row["subject_id"]
        )

        by_subject[
            subject
        ].append(
            (
                int(
                    row["window_index"]
                ),
                position,
            )
        )

    subject_ids = []
    features = []

    for subject in sorted(
        by_subject
    ):
        windows = sorted(
            by_subject[subject],
            key=lambda x: x[0],
        )

        freq_sum = np.zeros(
            256,
            dtype=np.float64,
        )

        num_columns = 0

        for _, position in windows:
            window = np.asarray(
                eeg[position],
                dtype=np.float32,
            )

            freq_sum += window.sum(
                axis=1,
                dtype=np.float64,
            )

            num_columns += (
                window.shape[1]
            )

        feature = (
            freq_sum
            / num_columns
        ).astype(
            np.float32
        )

        subject_ids.append(
            subject
        )

        features.append(
            feature
        )

    features = np.stack(
        features,
        axis=0,
    )

    np.save(
        OUTPUT_ROOT
        / f"{split}_features.npy",
        features,
    )

    with (
        OUTPUT_ROOT
        / f"{split}_subjects.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        for subject in subject_ids:
            f.write(
                subject + "\n"
            )

    print()
    print("=" * 72)
    print(split.upper())
    print("=" * 72)
    print(
        "Windows :",
        shape[0],
    )
    print(
        "Subjects:",
        len(subject_ids),
    )
    print(
        "Features:",
        features.shape,
    )
    print(
        "Mean:",
        float(
            features.mean()
        ),
    )
    print(
        "Std:",
        float(
            features.std()
        ),
    )
    print(
        "Min:",
        float(
            features.min()
        ),
    )
    print(
        "Max:",
        float(
            features.max()
        ),
    )


def main():
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    for split in SPLITS:
        build_split(
            split
        )


if __name__ == "__main__":
    main()
