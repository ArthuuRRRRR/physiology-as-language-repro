from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_index(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_roster(root: Path, split: str) -> list[str]:
    path = root / f"{split}_list.txt"

    return [
        line.strip().split("/")[-1]
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def extract_split(
    data_root: Path,
    roster_root: Path,
    split: str,
    output_dir: Path,
) -> None:

    split_dir = data_root / split

    metadata = load_json(split_dir / "metadata.json")
    rows = load_index(split_dir / "index.jsonl")

    eeg_shape = tuple(
        int(x) for x in metadata["eeg_shape"]
    )

    if len(eeg_shape) != 3:
        raise RuntimeError(
            f"{split}: expected (N, F, T), got {eeg_shape}"
        )

    if eeg_shape[1] != 256:
        raise RuntimeError(
            f"{split}: expected 256 frequency bins, "
            f"got {eeg_shape[1]}"
        )

    if len(rows) != eeg_shape[0]:
        raise RuntimeError(
            f"{split}: {len(rows)} index rows "
            f"but {eeg_shape[0]} EEG windows"
        )

    dtype = np.dtype(
        metadata.get("eeg_dtype", "float16")
    )

    eeg = np.memmap(
        split_dir / "synthesized_eeg.mmap",
        dtype=dtype,
        mode="r",
        shape=eeg_shape,
    )

    roster = read_roster(roster_root, split)

    roster_set = set(roster)

    # Accumulate every generated temporal column belonging
    # to the same SHHS1 recording.
    sums = defaultdict(
        lambda: np.zeros(256, dtype=np.float64)
    )
    counts = defaultdict(int)

    windows = defaultdict(int)

    for i, row in enumerate(rows):

        subject = str(row["subject_id"])

        if subject not in roster_set:
            continue

        spec = np.asarray(
            eeg[i],
            dtype=np.float32,
        )

        # spec = (256 frequency bins, T temporal columns)
        if spec.shape[0] != 256:
            raise RuntimeError(
                f"{subject}: unexpected shape {spec.shape}"
            )

        if not np.isfinite(spec).all():
            raise RuntimeError(
                f"{subject}: non-finite spectrogram values"
            )

        sums[subject] += spec.sum(
            axis=1,
            dtype=np.float64,
        )

        counts[subject] += spec.shape[1]
        windows[subject] += 1

    subjects = []
    features = []

    missing = []

    # Preserve exactly the roster ordering.
    for subject in roster:

        if counts[subject] == 0:
            missing.append(subject)
            continue

        feature = (
            sums[subject] / counts[subject]
        ).astype(np.float32)

        subjects.append(subject)
        features.append(feature)

    features = np.stack(features)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out = (
        output_dir
        / f"{split}_shard00.npz"
    )

    np.savez_compressed(
        out,
        subjects=np.asarray(subjects),
        features=features,
    )

    print()
    print("=" * 70)
    print(split.upper())
    print("=" * 70)
    print("Roster subjects :", len(roster))
    print("Subjects saved   :", len(subjects))
    print("Missing subjects :", len(missing))
    print("Feature shape    :", features.shape)
    print("Windows used     :", sum(windows.values()))
    print("Temporal columns :", sum(counts.values()))

    if missing:
        print("Missing IDs:")
        for s in missing:
            print(" ", s)

    print("Saved:", out)


def main():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--data_root",
        type=Path,
        default=Path(
            "outputs/downstream_sleep_staging_epoch146_resume140"
        ),
    )

    p.add_argument(
        "--roster_root",
        type=Path,
        default=Path(
            "data/shhs_meta_overlap_wav2sleep"
        ),
    )

    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path(
            "outputs/survival_features_pasl_epoch146"
        ),
    )

    args = p.parse_args()

    for split in ("train", "val", "test"):
        extract_split(
            data_root=args.data_root,
            roster_root=args.roster_root,
            split=split,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
