import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data.preprocessing import (
    eeg_to_multitaper_spectrogram,
    resample_frequency_axis,
    spectrogram_to_db,
)


# ============================================================
# Constants
# ============================================================

DATA_ROOT = Path("/hdd2/kdpark/sleep_datasets")

EEG_FS = 100.0
EPOCH_SAMPLES = 3000
WINDOW_EPOCHS = 512


# ============================================================
# Dataset-specific corrections
# ============================================================

# Two KVSS recordings have amplitudes several orders of
# magnitude larger than the rest of the cohort.
KVSS_EXCLUDED_SUBJECTS = {
    "B2019-EM-01-0125",
    "B2019-EM-01-0163",
}


# MrOS2 contains two clear amplitude populations.
# Subjects below this raw standard deviation are rescaled
# by x1000 before normalization and PSD computation.
MROS2_LOW_SCALE_STD_THRESHOLD = 1e-5
MROS2_LOW_SCALE_FACTOR = 1000.0


# ============================================================
# Sources
# ============================================================

SOURCES = [
    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    {
        "name": "phy_train",
        "split": "train",
        "pickle": (
            DATA_ROOT
            / "physionet_2018"
            / "mmap_signals_MAE_processed"
            / "EEG"
            / "C4-M1.pickle"
        ),
    },

    # PhysioNet test is used for pretraining,
    # following the SleepMaMi setting.
    {
        "name": "phy_test",
        "split": "train",
        "pickle": (
            DATA_ROOT
            / "physionet_test"
            / "mmap_signals_MAE_processed"
            / "EEG"
            / "C4-M1.pickle"
        ),
    },

    {
        "name": "shhs1_train",
        "split": "train",
        "pickle": (
            DATA_ROOT
            / "shhs1"
            / "mmap_signals_MAE_processed"
            / "train"
            / "EEG"
            / "C4-A1.pickle"
        ),
    },

    {
        "name": "shhs2",
        "split": "train",
        "pickle": (
            DATA_ROOT
            / "shhs2"
            / "mmap_signals_MAE_processed"
            / "train"
            / "EEG"
            / "C4-A1.pickle"
        ),
    },

    {
        "name": "kiss_train",
        "split": "train",
        "pickle": (
            DATA_ROOT
            / "kiss"
            / "mmap_signals_MAE_processed"
            / "train"
            / "EEG"
            / "C4-A1.pickle"
        ),
    },

    {
        "name": "kvss",
        "split": "train",
        "pickle": (
            DATA_ROOT
            / "kvss"
            / "mmap_signals_MAE_processed"
            / "train"
            / "EEG"
            / "C4-A1.pickle"
        ),
    },

    # MrOS1 train + val + test are all part of VQGAN training.
    {
        "name": "mros1_train",
        "split": "train",
        "pickle": (
            DATA_ROOT
            / "mros1"
            / "mmap_signals_MAE_processed"
            / "train"
            / "EEG"
            / "C3-A2.pickle"
        ),
    },

    {
        "name": "mros1_val",
        "split": "train",
        "pickle": (
            DATA_ROOT
            / "mros1"
            / "mmap_signals_MAE_processed"
            / "val"
            / "EEG"
            / "C3-A2.pickle"
        ),
    },

    {
        "name": "mros1_test",
        "split": "train",
        "pickle": (
            DATA_ROOT
            / "mros1"
            / "mmap_signals_MAE_processed"
            / "test"
            / "EEG"
            / "C3-A2.pickle"
        ),
    },

    # MrOS2 train + val + test are all part of VQGAN training.
    {
        "name": "mros2_train",
        "split": "train",
        "pickle": (
            DATA_ROOT
            / "mros2"
            / "mmap_signals_MAE_processed"
            / "train"
            / "EEG"
            / "C3-M2.pickle"
        ),
    },

    {
        "name": "mros2_val",
        "split": "train",
        "pickle": (
            DATA_ROOT
            / "mros2"
            / "mmap_signals_MAE_processed"
            / "val"
            / "EEG"
            / "C3-M2.pickle"
        ),
    },

    {
        "name": "mros2_test",
        "split": "train",
        "pickle": (
            DATA_ROOT
            / "mros2"
            / "mmap_signals_MAE_processed"
            / "test"
            / "EEG"
            / "C3-M2.pickle"
        ),
    },

    {
        "name": "mesa_train",
        "split": "train",
        "pickle": (
            DATA_ROOT
            / "mesa"
            / "mmap_signals_MAE_processed"
            / "train"
            / "EEG"
            / "C4-M1.pickle"
        ),
    },

    # --------------------------------------------------------
    # VALIDATION
    #
    # IMPORTANT:
    # These splits use normalization statistics estimated
    # from the corresponding TRAIN split.
    # --------------------------------------------------------

    {
        "name": "shhs1_val",
        "split": "val",
        "stats_from": "shhs1_train",
        "pickle": (
            DATA_ROOT
            / "shhs1"
            / "mmap_signals_MAE_processed"
            / "val"
            / "EEG"
            / "C4-A1.pickle"
        ),
    },

    {
        "name": "kiss_val",
        "split": "val",
        "stats_from": "kiss_train",
        "pickle": (
            DATA_ROOT
            / "kiss"
            / "mmap_signals_MAE_processed"
            / "val"
            / "EEG"
            / "C4-A1.pickle"
        ),
    },

    {
        "name": "mesa_val",
        "split": "val",
        "stats_from": "mesa_train",
        "pickle": (
            DATA_ROOT
            / "mesa"
            / "mmap_signals_MAE_processed"
            / "val"
            / "EEG"
            / "C4-M1.pickle"
        ),
    },

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    {
        "name": "mesa_test",
        "split": "test",
        "stats_from": "mesa_train",
        "pickle": (
            DATA_ROOT
            / "mesa"
            / "mmap_signals_MAE_processed"
            / "test"
            / "EEG"
            / "C4-M1.pickle"
        ),
    },
]


SOURCE_BY_NAME = {
    source["name"]: source
    for source in SOURCES
}


# ============================================================
# Utilities
# ============================================================

def safe_name(value):
    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(value),
    )


def load_metadata(pickle_path):
    with pickle_path.open("rb") as file:
        metadata = pickle.load(file)

    if "sig_info" not in metadata:
        raise KeyError(
            f"sig_info missing from {pickle_path}"
        )

    if "data_shape" not in metadata:
        raise KeyError(
            f"data_shape missing from {pickle_path}"
        )

    return metadata


# ============================================================
# Subject statistics
# ============================================================

def subject_raw_std(
    info,
    sig_len,
):
    start, end = info["pos"]

    n = (
        end - start
    ) * sig_len

    if n <= 0:
        return 0.0

    total_sum = float(
        info["stats"]["sum"]
    )

    total_sumsq = float(
        info["stats"]["sumsq"]
    )

    mean = total_sum / n

    variance = (
        total_sumsq / n
        - mean ** 2
    )

    variance = max(
        float(variance),
        0.0,
    )

    return float(
        np.sqrt(variance)
    )


# ============================================================
# Dataset corrections
# ============================================================

def should_exclude_subject(
    source_name,
    subject_id,
):
    if (
        source_name == "kvss"
        and subject_id in KVSS_EXCLUDED_SUBJECTS
    ):
        return True

    return False


def subject_scale_factor(
    source_name,
    info,
    sig_len,
):
    # Only MrOS2 needs the x1000 correction.
    if not source_name.startswith(
        "mros2_"
    ):
        return 1.0

    raw_std = subject_raw_std(
        info,
        sig_len,
    )

    if (
        raw_std
        < MROS2_LOW_SCALE_STD_THRESHOLD
    ):
        return MROS2_LOW_SCALE_FACTOR

    return 1.0


# ============================================================
# SleepMaMi-style global statistics
# ============================================================

def compute_sleepmami_stats(
    metadata,
    source_name,
):
    """
    Compute global mean/std using the same basic formula
    as SleepMaMi, after applying our scale corrections.

    mean =
        sum(x) / total_number_of_values

    std =
        sqrt(
            E[x^2] - E[x]^2
        )
    """

    sum_mu = 0.0
    sum_musq = 0.0
    cnt = 0

    sig_len = int(
        metadata["data_shape"][1]
    )

    scaled_subjects = 0
    excluded_subjects = 0

    for subject_id, info in (
        metadata["sig_info"].items()
    ):

        if should_exclude_subject(
            source_name,
            subject_id,
        ):
            excluded_subjects += 1
            continue

        start, end = info["pos"]

        scale = subject_scale_factor(
            source_name,
            info,
            sig_len,
        )

        if scale != 1.0:
            scaled_subjects += 1

        # If x' = scale * x:
        #
        # sum(x') =
        #     scale * sum(x)
        #
        # sum(x'^2) =
        #     scale^2 * sum(x^2)

        sum_mu += (
            scale
            * float(
                info["stats"]["sum"]
            )
        )

        sum_musq += (
            scale ** 2
            * float(
                info["stats"]["sumsq"]
            )
        )

        cnt += (
            end - start
        )

    if cnt <= 0:
        raise RuntimeError(
            f"No valid epochs found for {source_name}"
        )

    mean = (
        sum_mu
        / cnt
        / sig_len
    )

    variance = (
        sum_musq
        / cnt
        / sig_len
        - mean ** 2
    )

    variance = max(
        float(variance),
        1e-12,
    )

    std = float(
        np.sqrt(variance)
    )

    return {
        "mean": float(mean),
        "std": std,
        "num_epochs": int(cnt),
        "epoch_samples": sig_len,
        "scaled_subjects": (
            scaled_subjects
        ),
        "excluded_subjects": (
            excluded_subjects
        ),
    }


# ============================================================
# dB sampler
# ============================================================

class DBSampler:
    """
    Collect a bounded random sample of TRAIN dB values.

    It is used to estimate robust train-only normalization
    percentiles without keeping every spectrogram in RAM.
    """

    def __init__(
        self,
        max_values=2_000_000,
        values_per_window=512,
        seed=42,
    ):
        self.max_values = max_values

        self.values_per_window = (
            values_per_window
        )

        self.rng = np.random.default_rng(
            seed
        )

        self.chunks = []
        self.total_values = 0

    def add(
        self,
        array,
    ):
        flat = np.asarray(
            array,
            dtype=np.float32,
        ).reshape(-1)

        count = min(
            self.values_per_window,
            flat.size,
        )

        indices = self.rng.choice(
            flat.size,
            size=count,
            replace=False,
        )

        sampled = flat[
            indices
        ]

        self.chunks.append(
            sampled
        )

        self.total_values += (
            sampled.size
        )

        if (
            self.total_values
            > 2 * self.max_values
        ):
            self._compact()

    def _compact(self):
        if not self.chunks:
            return

        values = np.concatenate(
            self.chunks
        )

        if values.size > self.max_values:
            indices = self.rng.choice(
                values.size,
                size=self.max_values,
                replace=False,
            )

            values = values[
                indices
            ]

        self.chunks = [
            values.astype(
                np.float32,
                copy=False,
            )
        ]

        self.total_values = (
            values.size
        )

    def get_values(self):
        self._compact()

        if not self.chunks:
            raise RuntimeError(
                "No training dB values were sampled."
            )

        return self.chunks[0]


# ============================================================
# Spectrogram
# ============================================================

def make_spectrogram(
    eeg_epochs,
    mean,
    std,
):
    """
    Input:
        EEG epochs: (512, 3000)

    Output:
        dB spectrogram: (256, 512)
    """

    expected_input_shape = (
        WINDOW_EPOCHS,
        EPOCH_SAMPLES,
    )

    if eeg_epochs.shape != expected_input_shape:
        raise ValueError(
            "Unexpected EEG window shape: "
            f"{eeg_epochs.shape}. "
            f"Expected {expected_input_shape}."
        )

    eeg_epochs = np.asarray(
        eeg_epochs,
        dtype=np.float32,
    )

    # Global dataset/channel z-score.
    eeg_epochs = (
        eeg_epochs
        - mean
    ) / std

    # Existing preprocessing function expects one
    # continuous waveform and internally recreates
    # the 30-second epochs.
    eeg_waveform = (
        eeg_epochs.reshape(-1)
    )

    spectrogram, freqs = (
        eeg_to_multitaper_spectrogram(
            eeg=eeg_waveform,
            fs_eeg=EEG_FS,
        )
    )

    spectrogram, freqs = (
        resample_frequency_axis(
            spectrogram=spectrogram,
            freqs=freqs,
            n_freq_bins=256,
        )
    )

    spectrogram_db = (
        spectrogram_to_db(
            spectrogram
        )
    )

    expected_output_shape = (
        256,
        WINDOW_EPOCHS,
    )

    if (
        spectrogram_db.shape
        != expected_output_shape
    ):
        raise RuntimeError(
            "Unexpected spectrogram shape: "
            f"{spectrogram_db.shape}. "
            f"Expected {expected_output_shape}."
        )

    if not np.isfinite(
        spectrogram_db
    ).all():
        raise RuntimeError(
            "Non-finite values found "
            "in EEG spectrogram."
        )

    return (
        spectrogram_db,
        freqs,
    )


# ============================================================
# Process one source
# ============================================================

def process_source(
    source,
    output_root,
    db_sampler,
    manifests,
    max_windows_per_source=None,
    overwrite=False,
):
    pickle_path = (
        source["pickle"]
    )

    print()
    print("=" * 72)

    print(
        source["name"]
    )

    print(
        "role:",
        source["split"],
    )

    print(
        "pickle:",
        pickle_path,
    )

    if not pickle_path.exists():
        raise FileNotFoundError(
            pickle_path
        )

    metadata = load_metadata(
        pickle_path
    )

    # --------------------------------------------------------
    # Decide where normalization statistics come from.
    # --------------------------------------------------------

    stats_source_name = source.get(
        "stats_from",
        source["name"],
    )

    if (
        stats_source_name
        == source["name"]
    ):
        stats_metadata = metadata

    else:
        stats_source = SOURCE_BY_NAME[
            stats_source_name
        ]

        stats_metadata = load_metadata(
            stats_source["pickle"]
        )

    stats = compute_sleepmami_stats(
        stats_metadata,
        stats_source_name,
    )

    print(
        "normalization stats from:",
        stats_source_name,
    )

    print(
        "subjects:",
        len(
            metadata["sig_info"]
        ),
    )

    print(
        "epochs used for stats:",
        stats["num_epochs"],
    )

    print(
        "mean:",
        stats["mean"],
    )

    print(
        "std:",
        stats["std"],
    )

    print(
        "scaled subjects in stats:",
        stats["scaled_subjects"],
    )

    print(
        "excluded subjects in stats:",
        stats["excluded_subjects"],
    )

    # --------------------------------------------------------
    # Open mmap
    # --------------------------------------------------------

    mmap_path = (
        pickle_path.with_suffix(
            ".mmap"
        )
    )

    if not mmap_path.exists():
        raise FileNotFoundError(
            mmap_path
        )

    data = np.memmap(
        mmap_path,
        dtype="float32",
        mode="r",
        shape=tuple(
            metadata["data_shape"]
        ),
    )

    source_output = (
        output_root
        / source["split"]
        / source["name"]
    )

    source_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_windows = 0
    used_subjects = 0

    sig_len = int(
        metadata["data_shape"][1]
    )

    # --------------------------------------------------------
    # Subjects
    # --------------------------------------------------------

    for subject_id, info in (
        metadata["sig_info"].items()
    ):

        if should_exclude_subject(
            source["name"],
            subject_id,
        ):
            continue

        start, end = info["pos"]

        num_epochs = (
            end - start
        )

        num_windows = (
            num_epochs
            // WINDOW_EPOCHS
        )

        if num_windows == 0:
            continue

        scale = subject_scale_factor(
            source["name"],
            info,
            sig_len,
        )

        subject_used = False

        # ----------------------------------------------------
        # Non-overlapping 256-minute windows
        # ----------------------------------------------------

        for window_index in range(
            num_windows
        ):
            if (
                max_windows_per_source
                is not None
                and source_windows
                >= max_windows_per_source
            ):
                break

            local_start = (
                window_index
                * WINDOW_EPOCHS
            )

            global_start = (
                start
                + local_start
            )

            global_end = (
                global_start
                + WINDOW_EPOCHS
            )

            eeg_epochs = np.asarray(
                data[
                    global_start:
                    global_end
                ],
                dtype=np.float32,
            )

            # MrOS2 scale harmonization.
            if scale != 1.0:
                eeg_epochs = (
                    eeg_epochs
                    * scale
                )

            filename = (
                f"{source['name']}__"
                f"{safe_name(subject_id)}__"
                f"window{window_index:03d}.npz"
            )

            output_path = (
                source_output
                / filename
            )

            # ------------------------------------------------
            # Process or reuse existing sample
            # ------------------------------------------------

            if (
                output_path.exists()
                and not overwrite
            ):
                with np.load(
                    output_path,
                    allow_pickle=False,
                ) as saved:
                    spectrogram_db = (
                        saved[
                            "eeg_spectrogram_db"
                        ]
                    )

            else:
                (
                    spectrogram_db,
                    freqs,
                ) = make_spectrogram(
                    eeg_epochs=eeg_epochs,
                    mean=stats["mean"],
                    std=stats["std"],
                )

                np.savez(
                    output_path,

                    eeg_spectrogram_db=(
                        spectrogram_db
                    ),

                    freqs=freqs,

                    dataset=np.asarray(
                        source["name"]
                    ),

                    split=np.asarray(
                        source["split"]
                    ),

                    subject_id=np.asarray(
                        str(subject_id)
                    ),

                    window_index=np.asarray(
                        window_index,
                        dtype=np.int32,
                    ),

                    eeg_mean=np.asarray(
                        stats["mean"],
                        dtype=np.float32,
                    ),

                    eeg_std=np.asarray(
                        stats["std"],
                        dtype=np.float32,
                    ),

                    scale_factor=np.asarray(
                        scale,
                        dtype=np.float32,
                    ),

                    normalization_source=np.asarray(
                        stats_source_name
                    ),
                )

            # dB bounds are estimated from TRAIN only.
            if (
                source["split"]
                == "train"
            ):
                db_sampler.add(
                    spectrogram_db
                )

            manifests[
                source["split"]
            ].append(
                {
                    "path": str(
                        output_path
                    ),

                    "dataset": (
                        source["name"]
                    ),

                    "subject_id": (
                        str(subject_id)
                    ),

                    "window_index": (
                        window_index
                    ),

                    "scale_factor": (
                        scale
                    ),

                    "normalization_source": (
                        stats_source_name
                    ),
                }
            )

            source_windows += 1
            subject_used = True

        if subject_used:
            used_subjects += 1

        if (
            max_windows_per_source
            is not None
            and source_windows
            >= max_windows_per_source
        ):
            break

    print(
        "generated windows:",
        source_windows,
    )

    print(
        "subjects contributing:",
        used_subjects,
    )

    return (
        stats,
        source_windows,
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess the multi-dataset EEG pool "
            "for shared VQGAN training."
        )
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "outputs/"
            "vqgan_multidataset_preprocessed"
        ),
    )

    parser.add_argument(
        "--max-windows-per-source",
        type=int,
        default=None,
        help=(
            "Optional sanity-check limit. "
            "Example: 5 processes at most "
            "five windows per source."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--percentile-low",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--percentile-high",
        type=float,
        default=99.9,
    )

    args = parser.parse_args()

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    db_sampler = DBSampler()

    manifests = {
        "train": [],
        "val": [],
        "test": [],
    }

    waveform_stats = {}
    window_counts = {}

    # --------------------------------------------------------
    # Preprocess all logical sources
    # --------------------------------------------------------

    for source in SOURCES:
        stats, count = process_source(
            source=source,
            output_root=args.output_root,
            db_sampler=db_sampler,
            manifests=manifests,
            max_windows_per_source=(
                args.max_windows_per_source
            ),
            overwrite=args.overwrite,
        )

        waveform_stats[
            source["name"]
        ] = stats

        window_counts[
            source["name"]
        ] = count

    # --------------------------------------------------------
    # TRAIN-only dB normalization
    # --------------------------------------------------------

    train_db_values = (
        db_sampler.get_values()
    )

    min_db = float(
        np.percentile(
            train_db_values,
            args.percentile_low,
        )
    )

    max_db = float(
        np.percentile(
            train_db_values,
            args.percentile_high,
        )
    )

    if max_db <= min_db:
        raise RuntimeError(
            "Invalid dB normalization bounds: "
            f"{min_db}, {max_db}"
        )

    # --------------------------------------------------------
    # Save normalization metadata
    # --------------------------------------------------------

    normalization = {
        "min_db": min_db,
        "max_db": max_db,

        "percentile_low": (
            args.percentile_low
        ),

        "percentile_high": (
            args.percentile_high
        ),

        "waveform_stats": (
            waveform_stats
        ),

        "window_counts": (
            window_counts
        ),

        "kvss_excluded_subjects": (
            sorted(
                KVSS_EXCLUDED_SUBJECTS
            )
        ),

        "mros2_low_scale_std_threshold": (
            MROS2_LOW_SCALE_STD_THRESHOLD
        ),

        "mros2_low_scale_factor": (
            MROS2_LOW_SCALE_FACTOR
        ),

        "validation_normalization": {
            "shhs1_val": "shhs1_train",
            "kiss_val": "kiss_train",
            "mesa_val": "mesa_train",
            "mesa_test": "mesa_train",
        },
    }

    normalization_path = (
        args.output_root
        / "normalization.json"
    )

    with normalization_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            normalization,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Save manifests
    # --------------------------------------------------------

    for split, entries in (
        manifests.items()
    ):
        manifest_path = (
            args.output_root
            / f"{split}_manifest.jsonl"
        )

        with manifest_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            for entry in entries:
                file.write(
                    json.dumps(
                        entry
                    )
                    + "\n"
                )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("DONE")

    print(
        "Train windows:",
        len(
            manifests["train"]
        ),
    )

    print(
        "Validation windows:",
        len(
            manifests["val"]
        ),
    )

    print(
        "Test windows:",
        len(
            manifests["test"]
        ),
    )

    print(
        "TRAIN dB bounds:",
        min_db,
        max_db,
    )

    print(
        "Normalization:",
        normalization_path,
    )

    print(
        "Train manifest:",
        args.output_root
        / "train_manifest.jsonl",
    )

    print(
        "Validation manifest:",
        args.output_root
        / "val_manifest.jsonl",
    )

    print(
        "Test manifest:",
        args.output_root
        / "test_manifest.jsonl",
    )


if __name__ == "__main__":
    main()