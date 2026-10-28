#!/usr/bin/env python3
"""Extract DREAMT recordings into aligned 30-second epochs.

This is the raw extraction step. 
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap


DATA_DIR = Path(
    "/hdd2/kdpark/sleep_datasets/dreamt/physionet.org/"
    "files/dreamt/2.2.0/data_100Hz"
)

FS = 100
EPOCH_SEC = 30
EPOCH_SIZE = FS * EPOCH_SEC

CHANNELS = {
    "EEG": ["C4-M1", "F4-M1", "O2-M1", "Fp1-O2", "T3 - CZ", "CZ - T4"],
    "EOG": ["E1", "E2"],
    "EMG": ["CHIN", "LAT", "RAT"],
    "ECG": ["ECG"],
    "PPG": ["BVP"],
    "ABD": ["ABDOMEN"],
    "CHEST": ["THORAX"],
    "AIRFLOW": ["PTAF", "FLOW"],
    "SNORE": ["SNORE"],
    "SaO2": ["SAO2"],
    "ACC": ["ACC_X", "ACC_Y", "ACC_Z"],
    "TEMP": ["TEMP"],
    "EDA": ["EDA"],
    "HR": ["HR"],
    "IBI": ["IBI"],
}

STAGES = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4, "REM": 4}


def last_timestamp(csv_path):
    """Read the last CSV timestamp without loading the whole file."""
    with csv_path.open("rb") as file:
        file.seek(max(0, csv_path.stat().st_size - 65536))
        line = file.read().splitlines()[-1]
    return float(line.split(b",", 1)[0])


def sample_positions(timestamps, start_time):
    """Convert DREAMT timestamps to 100 Hz sample indices (handles midnight)."""
    seconds = (np.asarray(timestamps) - start_time) % 86400
    return np.rint(seconds * FS).astype(np.int64)


def detect_epoch_offset(csv_path, start_time):
    """Find the 30-s label grid from the first observed sleep-stage changes."""
    offsets = Counter()
    previous_stage = None

    for chunk in pd.read_csv(
        csv_path,
        usecols=["TIMESTAMP", "Sleep_Stage"],
        chunksize=100_000,
    ):
        stages = chunk["Sleep_Stage"].fillna("Missing").astype(str).str.strip().str.upper().to_numpy()
        changes = np.flatnonzero(stages[1:] != stages[:-1]) + 1
        if previous_stage is not None and stages[0] != previous_stage:
            changes = np.r_[0, changes]

        if len(changes):
            times = chunk["TIMESTAMP"].to_numpy()[changes]
            positions = sample_positions(times, start_time)
            offsets.update((positions % EPOCH_SIZE).tolist())

        previous_stage = stages[-1]
        if sum(offsets.values()) >= 5:
            break

    if not offsets:
        raise ValueError(
            "No sleep-stage transitions found. Use --offset-seconds "
            "to specify the annotation grid manually."
        )

    offset, count = offsets.most_common(1)[0]
    if len(offsets) > 1:
        raise ValueError(
            f"Inconsistent sleep-stage boundaries: {dict(offsets)}. "
            "Check the original annotations before processing."
        )
    print(f"  Annotation offset: {offset / FS:.2f} s ({count} transitions checked)")
    return int(offset)


def extract_subject(csv_path, output_dir, dry_run=False, offset_seconds=None):
    subject = csv_path.name.replace("_PSG_df.csv", "")
    header = pd.read_csv(csv_path, nrows=0).columns

    if "TIMESTAMP" not in header or "Sleep_Stage" not in header:
        raise ValueError(f"{subject}: TIMESTAMP or Sleep_Stage column missing")

    channels = [
        (group, name)
        for group, names in CHANNELS.items()
        for name in names
        if name in header
    ]
    names = [name for _, name in channels]

    start_time = float(
        pd.read_csv(csv_path, usecols=["TIMESTAMP"], nrows=1).iloc[0, 0]
    )
    end_sample = int(sample_positions([last_timestamp(csv_path)], start_time)[0]) + 1

    if offset_seconds is None:
        offset = detect_epoch_offset(csv_path, start_time)
    else:
        offset = int(round(offset_seconds * FS))
        if not 0 <= offset < EPOCH_SIZE:
            raise ValueError("--offset-seconds must be between 0 and 30")

    # The first partial annotation epoch and any last partial epoch are dropped.
    n_epochs = (end_sample - offset) // EPOCH_SIZE
    if n_epochs <= 0:
        raise ValueError(f"{subject}: no complete 30-second epochs")

    first_used = offset
    last_used = offset + n_epochs * EPOCH_SIZE
    print(f"\n{subject}: {n_epochs} complete annotation-aligned epochs, {len(names)} channels")

    seen = np.zeros((n_epochs, EPOCH_SIZE), dtype=bool)
    finite = np.ones((n_epochs, len(names)), dtype=bool)
    duplicated = np.zeros(n_epochs, dtype=bool)
    labels = np.full(n_epochs, -2, dtype=np.int8)

    outputs = {}
    if not dry_run:
        for group, name in channels:
            path = output_dir / "signals_MAE" / group / name / f"{subject}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            outputs[name] = open_memmap(
                path, mode="w+", dtype="float32", shape=(n_epochs, EPOCH_SIZE)
            )
            outputs[name][:] = 0

    columns = ["TIMESTAMP", "Sleep_Stage", *names]
    dtypes = {name: "float32" for name in names}
    dtypes["TIMESTAMP"] = "float64"

    previous_sample = None
    gaps = 0
    duplicates = 0

    for chunk in pd.read_csv(
        csv_path, usecols=columns, dtype=dtypes, chunksize=100_000
    ):
        samples = sample_positions(chunk["TIMESTAMP"].to_numpy(), start_time)
        steps = np.diff(samples)
        if previous_sample is not None:
            steps = np.r_[samples[0] - previous_sample, steps]
        if np.any(steps < 0):
            raise ValueError(f"{subject}: timestamps are not chronological")
        gaps += int(np.count_nonzero(steps > 1))
        duplicates += int(np.count_nonzero(steps == 0))
        previous_sample = samples[-1]

        # Ignore samples outside complete, annotation-aligned epochs.
        keep = (samples >= first_used) & (samples < last_used)
        if not np.any(keep):
            continue

        indices = np.flatnonzero(keep)
        aligned = samples[indices] - offset
        epoch_ids = aligned // EPOCH_SIZE
        positions = aligned % EPOCH_SIZE
        signals = chunk[names].to_numpy(dtype=np.float32)[indices]
        raw_stages = chunk["Sleep_Stage"].iloc[indices]
        stages = (
            raw_stages.fillna("Missing").astype(str).str.strip().str.upper()
            .map(STAGES).fillna(-1).to_numpy(dtype=np.int8)
        )

        boundaries = np.r_[0, np.flatnonzero(np.diff(epoch_ids)) + 1, len(indices)]
        for begin, end in zip(boundaries[:-1], boundaries[1:]):
            epoch = int(epoch_ids[begin])
            slots = positions[begin:end]
            values = signals[begin:end]

            if seen[epoch, slots].any() or np.any(np.diff(slots) == 0):
                duplicated[epoch] = True
            seen[epoch, slots] = True
            finite[epoch] &= np.isfinite(values).all(axis=0)

            unique_stages = np.unique(stages[begin:end])
            stage = int(unique_stages[0]) if len(unique_stages) == 1 else -1
            if labels[epoch] == -2:
                labels[epoch] = stage
            elif labels[epoch] != stage:
                labels[epoch] = -1

            if not dry_run:
                for col, name in enumerate(names):
                    outputs[name][epoch, slots] = np.nan_to_num(
                        values[:, col], nan=0.0, posinf=0.0, neginf=0.0
                    )

    complete = seen.all(axis=1) & ~duplicated
    valid = finite & complete[:, None]
    labels[labels == -2] = -1

    report = {
        "subject": subject,
        "annotation_offset_seconds": offset / FS,
        "discarded_start_seconds": offset / FS,
        "discarded_end_seconds": (end_sample - last_used) / FS,
        "epochs": int(n_epochs),
        "complete_epochs": int(complete.sum()),
        "timestamp_gaps": gaps,
        "duplicate_timestamps": duplicates,
        "sleep_stages": {
            str(stage): int((labels == stage).sum())
            for stage in [-1, 0, 1, 2, 3, 4]
        },
        "valid_epochs_per_channel": {
            name: int(valid[:, i].sum()) for i, name in enumerate(names)
        },
    }

    if not dry_run:
        for signal in outputs.values():
            signal.flush()

        annotations = output_dir / "annotations"
        annotations.mkdir(parents=True, exist_ok=True)
        np.save(annotations / f"{subject}_ann.npy", labels)

        for i, (group, name) in enumerate(channels):
            folder = output_dir / "quality_masks" / group / name
            folder.mkdir(parents=True, exist_ok=True)
            np.save(folder / f"{subject}_valid.npy", valid[:, i])

        reports = output_dir / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        (reports / f"{subject}.json").write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description="Extract DREAMT 30-second epochs")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dreamt_extracted"))
    parser.add_argument("--subject", help="Single participant, e.g. S002")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--offset-seconds", type=float,
        help="Optional annotation-grid offset. By default it is detected automatically."
    )
    args = parser.parse_args()

    files = sorted(args.data_dir.glob("S*_PSG_df.csv"))
    if args.subject:
        files = [f for f in files if f.name == f"{args.subject}_PSG_df.csv"]
    if not files:
        parser.error("No DREAMT CSV files found")

    for csv_path in files:
        subject = csv_path.name.replace("_PSG_df.csv", "")
        report_path = args.output_dir / "reports" / f"{subject}.json"
        if report_path.exists() and not args.overwrite and not args.dry_run:
            print(f"Skipping {subject} (already extracted). Use --overwrite to replace.")
            continue
        extract_subject(
            csv_path, args.output_dir,
            dry_run=args.dry_run, offset_seconds=args.offset_seconds,
        )


if __name__ == "__main__":
    main()
