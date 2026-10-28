#!/usr/bin/env python3

import json
import pickle
from pathlib import Path

import numpy as np
from scipy.signal import welch

SOURCE = Path("/hdd2/kdpark/sleep_datasets/cfs")

MANIFESTS = Path(
    "retraining_wav2sleep/outputs/cfs_wav2sleep_splits"
)

OUT = Path(
    "retraining_wav2sleep/outputs/downstream/"
    "cfs_keondo_gt/cfs"
)

CHANNEL = "C3-M2"
STAGE_MAP = np.array([0, 1, 1, 2, 3], dtype=np.int8)

EPOCHS_PER_WINDOW = 512
FREQ_BINS = 256
FS = 100


def normalize_subject(s):
    s = str(s)

    if s.endswith(".npy"):
        s = s[:-4]

    if not s.startswith("cfs-visit5-"):
        s = "cfs-visit5-" + s

    return s


def make_spectrum(raw):
    """
    Welch PSD per 30-second epoch.
    Log-power + per-recording percentile normalization.

    Note: This spectral conversion has not yet been
    verified as identical to Keondo's CNN-TCN pipeline.
    """
    n_epochs = len(raw)

    spec = np.empty(
        (n_epochs, FREQ_BINS),
        dtype=np.float32
    )

    for start in range(0, n_epochs, 128):
        stop = min(n_epochs, start + 128)

        x = np.asarray(
            raw[start:stop],
            dtype=np.float32
        )

        if not np.all(np.isfinite(x)):
            raise ValueError("Non-finite EEG samples")

        _, psd = welch(
            x,
            fs=FS,
            window="hann",
            nperseg=512,
            noverlap=256,
            nfft=512,
            axis=1,
            scaling="density",
            average="mean",
        )

        spec[start:stop] = (
            10.0 * np.log10(
                np.maximum(
                    psd[:, :FREQ_BINS],
                    1e-20
                )
            )
        )

    p_lo, p_hi = np.percentile(
        spec, [1, 99]
    )

    if p_hi <= p_lo:
        raise ValueError("Degenerate spectrogram")

    spec -= np.float32(p_lo)
    spec /= np.float32(p_hi - p_lo)

    np.clip(spec, 0, 1, out=spec)

    return spec


def build(split):

    print(
        f"\n========== {split.upper()} ==========",
        flush=True
    )

    meta_path = (
        SOURCE
        / "mmap_signals_MAE_processed_wav2sleep"
        / split
        / "EEG"
        / f"{CHANNEL}.pickle"
    )

    mmap_path = meta_path.with_suffix(".mmap")

    with meta_path.open("rb") as f:
        source_meta = pickle.load(f)

    source_info = source_meta["sig_info"]
    source_shape = tuple(source_meta["data_shape"])

    if len(source_shape) != 2 or source_shape[1] != FS * 30:
        raise ValueError(
            f"Unexpected raw shape: {source_shape}"
        )

    raw_eeg = np.memmap(
        mmap_path,
        dtype=np.float32,
        mode="r",
        shape=source_shape
    )

    manifest_path = (
        MANIFESTS / f"{split}_manifest.jsonl"
    )

    with manifest_path.open() as f:
        manifest = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    subjects = sorted({
        normalize_subject(row["subject_id"])
        for row in manifest
    })

    if not subjects:
        raise ValueError(f"Empty split: {split}")

    entries = []

    for sid in subjects:

        if sid not in source_info:
            raise FileNotFoundError(
                f"{sid}: EEG missing in {split}"
            )

        start, end = map(
            int,
            source_info[sid]["pos"]
        )

        if not (0 <= start < end <= len(raw_eeg)):
            raise ValueError(
                f"{sid}: invalid EEG boundaries"
            )

        ann_candidates = [
            SOURCE / "annotations_split"
            / split / f"{sid}_ann.npy",

            SOURCE / "annotations"
            / f"{sid}_ann.npy",
        ]

        ann = next(
            (p for p in ann_candidates if p.exists()),
            None
        )

        if ann is None:
            raise FileNotFoundError(
                f"{sid}: no annotations"
            )

        labels = np.load(
            ann,
            mmap_mode="r",
            allow_pickle=False
        )

        if labels.ndim != 1 or len(labels) != end - start:
            raise ValueError(
                f"{sid}: EEG={end-start}, "
                f"labels={labels.shape}"
            )

        if not np.isin(labels, np.arange(5)).all():
            raise ValueError(
                f"{sid}: unexpected labels "
                f"{np.unique(labels)}"
            )

        entries.append(
            (sid, start, end, ann)
        )

    num_windows = sum(
        (end - start + EPOCHS_PER_WINDOW - 1)
        // EPOCHS_PER_WINDOW
        for _, start, end, _ in entries
    )

    dest = OUT / split
    dest.mkdir(parents=True, exist_ok=True)

    if (dest / "metadata.json").exists():
        raise FileExistsError(
            f"{dest} already completed; refusing overwrite"
        )

    eeg_shape = (
        num_windows,
        FREQ_BINS,
        EPOCHS_PER_WINDOW
    )

    label_shape = (
        num_windows,
        EPOCHS_PER_WINDOW
    )

    eeg_out = np.memmap(
        dest / "ground_truth_eeg.mmap",
        dtype=np.float32,
        mode="w+",
        shape=eeg_shape
    )

    labels_out = np.memmap(
        dest / "labels.mmap",
        dtype=np.int64,
        mode="w+",
        shape=label_shape
    )

    eeg_out[:] = 0.0
    labels_out[:] = -100

    index_path = dest / "index.jsonl"

    k = 0
    stage_hist = np.zeros(4, dtype=np.int64)

    with index_path.open("w") as index_file:

        for j, (sid, start, end, ann) in enumerate(entries):

            spectrum = make_spectrum(
                raw_eeg[start:end]
            )

            original_labels = np.load(
                ann,
                allow_pickle=False
            ).astype(np.int64)

            labels = STAGE_MAP[
                original_labels
            ]

            stage_hist += np.bincount(
                labels,
                minlength=4
            )[:4]

            for wi, offset in enumerate(
                range(
                    0,
                    len(labels),
                    EPOCHS_PER_WINDOW
                )
            ):

                n = min(
                    EPOCHS_PER_WINDOW,
                    len(labels) - offset
                )

                eeg_out[k, :, :n] = (
                    spectrum[
                        offset:offset+n
                    ].T
                )

                labels_out[k, :n] = (
                    labels[
                        offset:offset+n
                    ]
                )

                index_file.write(
                    json.dumps({
                        "dataset": "cfs",
                        "subject_id": sid,
                        "window_index": wi,
                        "start_epoch_in_trimmed_recording": offset,
                        "valid_epochs": n,
                    }) + "\n"
                )

                k += 1

            if (
                (j + 1) % 25 == 0
                or j + 1 == len(entries)
            ):
                print(
                    f"  {j+1}/{len(entries)} subjects, "
                    f"{k}/{num_windows} windows",
                    flush=True
                )

    assert k == num_windows

    eeg_out.flush()
    labels_out.flush()

    metadata = {
        "dataset": "cfs",
        "split": split,
        "num_samples": num_windows,
        "eeg_shape": list(eeg_shape),
        "label_shape": list(label_shape),
        "eeg_dtype": "float32",
        "label_dtype": "int64",
        "source_channel": CHANNEL,
        "source_subjects": len(entries),
        "stage_counts": stage_hist.tolist(),
        "spectral_method": (
            "Welch PSD fs=100 nfft=512, "
            "log10 power, per-recording "
            "1st/99th percentile scaling"
        ),
    }

    (dest / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    print(
        f"DONE {split}: "
        f"{len(entries)} subjects / "
        f"{num_windows} windows",
        flush=True
    )

    print(
        "Stage counts:",
        stage_hist.tolist(),
        flush=True
    )


if __name__ == "__main__":

    for split in ("train", "val", "test"):
        build(split)

    print(
        "\nAll splits ready.",
        flush=True
    )
