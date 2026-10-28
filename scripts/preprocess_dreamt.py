#!/usr/bin/env python3
"""Filter aligned DREAMT recordings and create per-channel memory maps.

Input : outputs/dreamt_extracted (30-second epochs at 100 Hz)
Output: outputs/dreamt_processed (filtered arrays, labels and mmap files)

Run `process` before `pack`. Train/validation/test splits are provided separately.
"""

import argparse
import importlib.util
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
from scipy.signal import butter, cheby2, sosfiltfilt


SAMPLE_RATE = 100
SAMPLES_PER_EPOCH = 30 * SAMPLE_RATE
EXTRACTED_DIR = Path("outputs/dreamt_extracted")
PROCESSED_DIR = Path("outputs/dreamt_processed")

FILTERS = {
    "EEG": ("bandpass", 0.3, 40),
    "EOG": ("bandpass", 0.3, 40),
    "ECG": ("bandpass", 0.3, 40),
    "EMG": ("bandpass", 10, 40),
    "ABD": ("lowpass", 15),
    "CHEST": ("lowpass", 15),
    "AIRFLOW": ("lowpass", 15),
    "PPG": ("ppg", 8),
}

MAIN_CHANNELS = [
    ("EEG", "C4-M1"),
    ("ECG", "ECG"),
    ("PPG", "BVP"),
    ("ABD", "ABDOMEN"),
]


def load_polarity_functions(path):
    """Load the existing ECG and PPG polarity correction functions."""
    if path is None:
        try:
            from data_utils import correct_ecg_inversion, correct_ppg_inversion
        except ImportError as exc:
            raise RuntimeError(
                "Cannot import data_utils.py. Supply --utils-file or use "
                "--skip-polarity for a preliminary test."
            ) from exc
        return correct_ecg_inversion, correct_ppg_inversion

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location("signal_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.correct_ecg_inversion, module.correct_ppg_inversion


def make_filter(group):
    """Return a filter for this modality, or None when not required."""
    settings = FILTERS.get(group)
    if settings is None:
        return None

    kind, *cutoffs = settings
    if kind == "ppg":
        return cheby2(8, 40, cutoffs[0], btype="lowpass",
                      fs=SAMPLE_RATE, output="sos")
    if kind == "lowpass":
        return butter(4, cutoffs[0], btype="lowpass",
                      fs=SAMPLE_RATE, output="sos")
    return butter(4, cutoffs, btype="bandpass",
                  fs=SAMPLE_RATE, output="sos")


def valid_segments(mask):
    """Iterate over uninterrupted runs of valid 30-second epochs."""
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(int)))
    return list(zip(edges[::2], edges[1::2]))


def filter_signal(epochs, mask, sos):
    """Filter continuous valid segments without shifting epoch boundaries."""
    output = np.zeros(epochs.shape, dtype=np.float32)
    for start, end in valid_segments(mask):
        signal = np.asarray(epochs[start:end], dtype=np.float64).reshape(-1)
        if sos is not None:
            signal = sosfiltfilt(sos, signal)
        output[start:end] = signal.reshape(-1, SAMPLES_PER_EPOCH)
    return output


def correct_ecg(epochs, mask, correction):
    """Correct ECG polarity separately within each valid epoch."""
    for index in np.flatnonzero(mask):
        signal, _ = correction(epochs[index].copy(), SAMPLE_RATE)
        signal = np.asarray(signal, dtype=np.float32).reshape(-1)
        if signal.size != SAMPLES_PER_EPOCH:
            raise ValueError("ECG polarity correction changed epoch length")
        epochs[index] = signal
    return epochs


def correct_ppg(epochs, mask, correction):
    """Estimate PPG orientation from the longest uninterrupted valid segment."""
    segments = valid_segments(mask)
    if not segments:
        raise ValueError("No valid PPG epochs")

    start, end = max(segments, key=lambda pair: pair[1] - pair[0])
    original = epochs[start:end].reshape(-1).astype(np.float64)
    corrected, _ = correction(original, SAMPLE_RATE)
    corrected = np.asarray(corrected).reshape(-1)
    if corrected.shape != original.shape:
        raise ValueError("PPG polarity correction changed signal length")

    # The correction routine should only reverse the waveform orientation.
    keep_error = np.mean((corrected - original) ** 2)
    flip_error = np.mean((corrected + original) ** 2)
    if flip_error < keep_error:
        epochs[mask] *= -1
    return epochs


def normalize_ppg(epochs, mask):
    """Clip PPG at three standard deviations, then normalize per recording."""
    if not mask.any():
        return epochs

    values = epochs[mask].astype(np.float64)
    mean, std = values.mean(), values.std()
    if std < 1e-8:
        epochs[mask] = 0
        return epochs

    values = np.clip(values, mean - 3 * std, mean + 3 * std)
    mean, std = values.mean(), values.std()
    epochs[mask] = (values - mean) / std if std >= 1e-8 else 0
    return epochs


def process_subject(subject, extracted_dir, processed_dir, polarity_functions, overwrite):
    """Filter one recording and keep its original epoch alignment."""
    input_dir = extracted_dir / "signals_MAE"
    files = sorted(input_dir.glob(f"*/*/{subject}.npy"))
    annotation = extracted_dir / "annotations" / f"{subject}_ann.npy"
    report_path = processed_dir / "reports" / f"{subject}.json"

    if not files or not annotation.exists():
        raise FileNotFoundError(f"Missing extracted data or annotations for {subject}")
    if report_path.exists() and not overwrite:
        print(f"{subject}: already processed (use --overwrite to rerun)")
        return

    available = {(f.parent.parent.name, f.parent.name) for f in files}
    missing = set(MAIN_CHANNELS) - available
    if missing:
        raise ValueError(f"{subject}: missing main channels: {sorted(missing)}")

    labels = np.load(annotation, allow_pickle=False)
    report = {"subject": subject, "epochs": int(len(labels)),
              "sampling_rate": SAMPLE_RATE,
              "polarity_corrected": polarity_functions is not None,
              "channels": {}}

    for path in files:
        group, channel = path.parent.parent.name, path.parent.name
        mask_path = (extracted_dir / "quality_masks" / group / channel /
                     f"{subject}_valid.npy")
        epochs = np.load(path, mmap_mode="r", allow_pickle=False)
        mask = np.load(mask_path, allow_pickle=False).astype(bool)
        if epochs.shape != (len(labels), SAMPLES_PER_EPOCH):
            raise ValueError(f"Unexpected dimensions in {path}: {epochs.shape}")
        if mask.shape != (len(labels),):
            raise ValueError(f"Unexpected quality mask in {mask_path}")

        if group == "PPG" and polarity_functions is not None:
            oriented = np.asarray(epochs, dtype=np.float32).copy()
            oriented = correct_ppg(oriented, mask, polarity_functions[1])
            output = filter_signal(oriented, mask, make_filter(group))
        else:
            output = filter_signal(epochs, mask, make_filter(group))

        if group == "ECG" and polarity_functions is not None:
            output = correct_ecg(output, mask, polarity_functions[0])
        if group == "PPG":
            output = normalize_ppg(output, mask)
        if not np.isfinite(output).all():
            raise ValueError(f"Non-finite output for {subject} {group}/{channel}")

        destination = (processed_dir / "signals_MAE_processed" /
                       group / channel / f"{subject}.npy")
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, output)

        quality_destination = (processed_dir / "quality_masks" /
                               group / channel / f"{subject}_valid.npy")
        quality_destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(quality_destination, mask)
        report["channels"][f"{group}/{channel}"] = int(mask.sum())

    target = processed_dir / "annotations" / annotation.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(annotation, target)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"{subject}: processed {len(files)} channels, {len(labels)} epochs")


def pack_channel(subjects, split, group, channel, processed_dir, packed_dir, overwrite):
    """Create a memory map and matching subject-position metadata."""
    paths = [(subject, processed_dir / "signals_MAE_processed" / group / channel /
              f"{subject}.npy") for subject in subjects]
    paths = [(subject, path) for subject, path in paths if path.exists()]
    if not paths:
        return
    if (group, channel) in MAIN_CHANNELS and len(paths) != len(subjects):
        raise ValueError(f"Missing {group}/{channel} for some subjects in {split}")

    counts = []
    for _, path in paths:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.ndim != 2 or array.shape[1] != SAMPLES_PER_EPOCH:
            raise ValueError(f"Unexpected dimensions: {path} {array.shape}")
        counts.append(array.shape[0])

    folder = packed_dir / split / group
    folder.mkdir(parents=True, exist_ok=True)
    data_path = folder / f"{channel}.mmap"
    info_path = folder / f"{channel}.pickle"
    if not overwrite and (data_path.exists() or info_path.exists()):
        raise FileExistsError(f"Already exists: {data_path}")

    shape = (sum(counts), SAMPLES_PER_EPOCH)
    data = np.memmap(data_path, dtype="float32", mode="w+", shape=shape)
    sig_info = {}
    start = 0
    for (subject, path), count in zip(paths, counts):
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        end = start + count
        data[start:end] = values
        numbers = np.asarray(values, dtype=np.float64)
        sig_info[subject] = {
            "pos": (start, end),
            "stats": {"sum": float(numbers.sum()),
                      "sumsq": float(np.square(numbers).sum())},
        }
        start = end
    data.flush()
    del data

    with info_path.open("wb") as handle:
        pickle.dump({"sig_info": sig_info, "data_shape": shape}, handle)
    print(f"{split}/{group}/{channel}: {shape[0]} epochs")


def pack_dataset(splits_path, processed_dir, packed_dir, overwrite):
    """Pack only the subjects explicitly listed in the split JSON."""
    splits = json.loads(Path(splits_path).read_text())
    if not isinstance(splits, dict) or not set(splits).issubset({"train", "val", "test"}):
        raise ValueError("Split file must contain train, val and/or test lists")
    if any(not isinstance(ids, list) for ids in splits.values()):
        raise ValueError("Every split must be a list of subject IDs")

    subjects = [subject for ids in splits.values() for subject in ids]
    if len(subjects) != len(set(subjects)):
        raise ValueError("Some subjects appear in more than one split")
    for subject in subjects:
        report_path = processed_dir / "reports" / f"{subject}.json"
        if not report_path.exists():
            raise FileNotFoundError(report_path)
        report = json.loads(report_path.read_text())
        if not report["polarity_corrected"]:
            raise ValueError(f"{subject}: polarity correction was skipped")

    signal_root = processed_dir / "signals_MAE_processed"
    for split, split_subjects in splits.items():
        if not split_subjects:
            continue
        channels = sorted({(f.parent.parent.name, f.parent.name)
                           for subject in split_subjects
                           for f in signal_root.glob(f"*/*/{subject}.npy")})
        for group, channel in channels:
            pack_channel(split_subjects, split, group, channel,
                         processed_dir, packed_dir, overwrite)
        for subject in split_subjects:
            annotation = processed_dir / "annotations" / f"{subject}_ann.npy"
            target = processed_dir / "annotations_split" / split / annotation.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(annotation, target)


def main():
    parser = argparse.ArgumentParser(description="DREAMT signal preprocessing")
    commands = parser.add_subparsers(dest="command", required=True)

    process = commands.add_parser("process", help="Filter extracted signal arrays")
    process.add_argument("--subject", help="Subject ID, for example S002")
    process.add_argument("--extracted-dir", type=Path, default=EXTRACTED_DIR)
    process.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    process.add_argument("--utils-file", type=Path, help="Path to data_utils.py")
    process.add_argument("--skip-polarity", action="store_true",
                         help="Preliminary test only; not suitable for final packing")
    process.add_argument("--overwrite", action="store_true")

    pack = commands.add_parser("pack", help="Create train/val/test mmap files")
    pack.add_argument("--split-file", type=Path, required=True)
    pack.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    pack.add_argument("--packed-dir", type=Path)
    pack.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    if args.command == "process":
        corrections = None if args.skip_polarity else load_polarity_functions(args.utils_file)
        if args.subject:
            subjects = [args.subject]
        else:
            subjects = sorted(p.name.removesuffix("_ann.npy")
                              for p in (args.extracted_dir / "annotations").glob("*_ann.npy"))
        if not subjects:
            parser.error("No extracted subjects found")
        for subject in subjects:
            process_subject(subject, args.extracted_dir, args.processed_dir,
                            corrections, args.overwrite)
    else:
        packed_dir = args.packed_dir or args.processed_dir / "mmap_signals_MAE_processed"
        pack_dataset(args.split_file, args.processed_dir, packed_dir, args.overwrite)


if __name__ == "__main__":
    main()
