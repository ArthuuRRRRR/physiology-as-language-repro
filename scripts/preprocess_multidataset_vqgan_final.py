import argparse
import json
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.data.preprocessing import (
    eeg_to_multitaper_spectrogram,
    resample_frequency_axis,
    spectrogram_to_db,
)


# ============================================================
# Constants
# ============================================================

DATA_ROOT = Path(
    "/hdd2/kdpark/sleep_datasets"
)

EEG_FS = 100.0
EPOCH_SAMPLES = 3000
WINDOW_EPOCHS = 512

# Same flat-signal threshold as Keondo.
EEG_FLAT_STD_THRESHOLD = 1e-4


# ============================================================
# Dataset-specific corrections
# ============================================================

# Recordings identified as clear channel/amplitude failures.
# Exclusions are applied BOTH when computing dataset statistics
# and when generating VQGAN spectrogram windows.
EXCLUDED_SUBJECTS = {
    "kvss": {
        "B2019-EM-01-0125",
        "B2019-EM-01-0163",
    },
    "shhs2": {
        "shhs2-204890",
    },
    "shhs1": {
        "shhs1-202345",
        "shhs1-204822",
    },
    "kiss": {
        "A2016-EM-01-0071",
    },
}


# MrOS2 contains two clear amplitude populations.
# Subjects below this raw standard deviation are rescaled
# by x250 before normalization and PSD computation.
MROS2_LOW_SCALE_STD_THRESHOLD = 1e-5
MROS2_LOW_SCALE_FACTOR = 250.0


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

def safe_name(
    value,
):
    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(value),
    )


def load_metadata(
    pickle_path,
):

    with pickle_path.open(
        "rb"
    ) as file:
        metadata = pickle.load(
            file
        )

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

    start, end = info[
        "pos"
    ]

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

    mean = (
        total_sum / n
    )

    variance = (
        total_sumsq / n
        - mean ** 2
    )

    variance = max(
        float(variance),
        0.0,
    )

    return float(
        np.sqrt(
            variance
        )
    )


# ============================================================
# Dataset corrections
# ============================================================

def source_family(
    source_name,
):
    """Map logical split names to the dataset name used by exclusions."""

    if source_name.startswith("shhs1"):
        return "shhs1"

    if source_name.startswith("kiss"):
        return "kiss"

    return source_name


def should_exclude_subject(
    source_name,
    subject_id,
):

    family = source_family(
        source_name
    )

    return str(subject_id) in EXCLUDED_SUBJECTS.get(
        family,
        set(),
    )


def print_subject_count_check():
    """
    Cheap preflight check: inspect only pickle metadata.

    No mmap is opened, no spectrogram is generated, and no output
    dataset is written.  This lets us compare subject/recording counts
    with Keondo before launching the expensive preprocessing.
    """

    print()
    print("=" * 88)
    print("SUBJECT COUNT CHECK AFTER EXCLUSIONS")
    print("=" * 88)
    print(
        f"{'source':<18} {'role':<6} "
        f"{'metadata':>9} {'excluded':>9} "
        f"{'remaining':>10} {'>=512 epochs':>12}"
    )

    train_metadata = 0
    train_excluded = 0
    train_remaining = 0
    train_usable = 0

    for source in SOURCES:
        metadata = load_metadata(
            source["pickle"]
        )
        sig_info = metadata[
            "sig_info"
        ]

        total = len(sig_info)

        excluded_ids = [
            str(subject_id)
            for subject_id in sig_info
            if should_exclude_subject(
                source["name"],
                subject_id,
            )
        ]

        excluded = len(
            excluded_ids
        )
        remaining = total - excluded

        usable = 0
        for subject_id, info in sig_info.items():

            if should_exclude_subject(
                source["name"],
                subject_id,
            ):
                continue

            start, end = info[
                "pos"
            ]

            if (
                end - start
            ) // WINDOW_EPOCHS > 0:
                usable += 1

        print(
            f"{source['name']:<18} "
            f"{source['split']:<6} "
            f"{total:>9} "
            f"{excluded:>9} "
            f"{remaining:>10} "
            f"{usable:>12}"
        )

        if excluded_ids:
            print(
                "  excluded:",
                ", ".join(
                    excluded_ids
                ),
            )

        if source["split"] == "train":
            train_metadata += total
            train_excluded += excluded
            train_remaining += remaining
            train_usable += usable

    print("-" * 88)
    print(
        f"{'TRAIN TOTAL':<18} {'train':<6} "
        f"{train_metadata:>9} "
        f"{train_excluded:>9} "
        f"{train_remaining:>10} "
        f"{train_usable:>12}"
    )
    print()
    print(
        "NOTE: these are metadata entries/recordings per source. "
        "The '>=512 epochs' column is the number that can actually "
        "produce at least one 256-minute VQGAN window."
    )


def subject_scale_factor(
    source_name,
    info,
    sig_len,
):

    # Only MrOS2 needs the x250 correction.
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
        return (
            MROS2_LOW_SCALE_FACTOR
        )

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
        metadata[
            "data_shape"
        ][1]
    )

    scaled_subjects = 0
    excluded_subjects = 0

    for (
        subject_id,
        info,
    ) in metadata[
        "sig_info"
    ].items():

        if should_exclude_subject(
            source_name,
            subject_id,
        ):
            excluded_subjects += 1
            continue

        start, end = info[
            "pos"
        ]

        scale = (
            subject_scale_factor(
                source_name,
                info,
                sig_len,
            )
        )

        if scale != 1.0:
            scaled_subjects += 1

        sum_mu += (
            scale
            * float(
                info[
                    "stats"
                ][
                    "sum"
                ]
            )
        )

        sum_musq += (
            scale ** 2
            * float(
                info[
                    "stats"
                ][
                    "sumsq"
                ]
            )
        )

        cnt += (
            end - start
        )

    if cnt <= 0:
        raise RuntimeError(
            "No valid epochs found "
            f"for {source_name}"
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
        np.sqrt(
            variance
        )
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
# Exact TRAIN dB collector
# ============================================================

class DBCollector:
    """
    Collect ALL valid TRAIN spectrogram dB values on local disk.

    This replaces the previous bounded random DBSampler. No dB
    values are sampled: every finite value from every valid 30-s
    EEG epoch in the training pool contributes to the percentile
    calculation.

    Values are streamed as float32 to a temporary binary file so
    RAM usage stays bounded. Exact percentiles are then computed
    from a writable numpy.memmap with overwrite_input=True, which
    avoids creating a second full-size in-memory copy.
    """

    def __init__(
        self,
        cache_path=None,
    ):

        if cache_path is None:
            cache_path = (
                Path("/tmp")
                / f"vqgan_train_valid_db_{os.getpid()}.float32"
            )

        self.cache_path = Path(
            cache_path
        )

        self.cache_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # This preprocessing run is intentionally from scratch.
        # Never append to values left by an older run.
        if self.cache_path.exists():
            self.cache_path.unlink()

        self.file = self.cache_path.open(
            "wb"
        )

        self.total_values = 0
        self.total_bytes = 0
        self.closed = False

        print(
            "Exact TRAIN dB cache:",
            self.cache_path,
        )

    def add(
        self,
        array,
        valid_mask=None,
    ):

        if self.closed:
            raise RuntimeError(
                "Cannot add values after DBCollector was closed."
            )

        array = np.asarray(
            array,
            dtype=np.float32,
        )

        # Spectrogram layout: frequency x time.
        if valid_mask is not None:

            valid_mask = np.asarray(
                valid_mask,
                dtype=bool,
            )

            expected_shape = (
                array.shape[1],
            )

            if valid_mask.shape != expected_shape:
                raise ValueError(
                    "Unexpected valid-mask shape: "
                    f"{valid_mask.shape}. "
                    f"Expected {expected_shape}."
                )

            # Keep every frequency value from every valid epoch.
            array = array[
                :,
                valid_mask,
            ]

        flat = array.reshape(-1)

        if flat.size == 0:
            return

        # Valid columns are expected to be finite, but keep this
        # guard so only finite dB values can enter the percentiles.
        flat = flat[
            np.isfinite(flat)
        ]

        if flat.size == 0:
            return

        flat = flat.astype(
            np.float32,
            copy=False,
        )

        flat.tofile(
            self.file
        )

        self.total_values += int(
            flat.size
        )

        self.total_bytes += int(
            flat.nbytes
        )

    def close(
        self,
    ):

        if self.closed:
            return

        self.file.flush()
        os.fsync(
            self.file.fileno()
        )
        self.file.close()
        self.closed = True

    def compute_quantiles(
        self,
        quantile_low,
        quantile_high,
    ):

        self.close()

        if self.total_values <= 0:
            raise RuntimeError(
                "No valid training dB values were collected."
            )

        expected_bytes = (
            self.total_values
            * np.dtype(np.float32).itemsize
        )

        actual_bytes = (
            self.cache_path.stat().st_size
        )

        if actual_bytes != expected_bytes:
            raise RuntimeError(
                "dB cache size mismatch: "
                f"expected {expected_bytes} bytes, "
                f"found {actual_bytes}."
            )

        print()
        print("=" * 72)
        print("EXACT TRAIN dB QUANTILES")
        print("all valid dB values:", self.total_values)
        print(
            "cache size GiB:",
            f"{actual_bytes / (1024 ** 3):.3f}",
        )
        print(
            "quantiles:",
            quantile_low,
            quantile_high,
            "(equivalent percentiles:",
            100.0 * quantile_low,
            100.0 * quantile_high,
            ")",
        )

        values = np.memmap(
            self.cache_path,
            dtype=np.float32,
            mode="r+",
            shape=(self.total_values,),
        )

        bounds = np.quantile(
            values,
            [
                quantile_low,
                quantile_high,
            ],
            method="linear",
            overwrite_input=True,
        )

        # Flush the in-place partition performed by numpy before
        # releasing the mmap. The cache is only temporary.
        values.flush()
        del values

        min_db = float(
            bounds[0]
        )
        max_db = float(
            bounds[1]
        )

        return min_db, max_db

    def cleanup(
        self,
    ):

        self.close()

        if self.cache_path.exists():
            self.cache_path.unlink()


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
        frequency axis: (256,)
        valid_mask: (512,)

    Invalid epochs follow Keondo's rejection rule:
        non-finite OR std <= 1e-4
        OR non-finite PSD in dB.

    They keep their temporal positions.
    """

    expected_input_shape = (
        WINDOW_EPOCHS,
        EPOCH_SAMPLES,
    )

    if (
        eeg_epochs.shape
        != expected_input_shape
    ):
        raise ValueError(
            "Unexpected EEG window shape: "
            f"{eeg_epochs.shape}. "
            f"Expected {expected_input_shape}."
        )

    eeg_epochs = np.asarray(
        eeg_epochs,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Global dataset/channel z-score.
    # --------------------------------------------------------

    eeg_epochs = (
        eeg_epochs
        - mean
    ) / std

    # Existing preprocessing function takes a continuous
    # waveform and recreates the 30-s epochs internally.
    eeg_waveform = (
        eeg_epochs.reshape(-1)
    )

    # --------------------------------------------------------
    # Keondo-style rejection rule
    # --------------------------------------------------------

    (
        spectrogram,
        freqs,
        valid_mask,
    ) = (
        eeg_to_multitaper_spectrogram(
            eeg=eeg_waveform,
            fs_eeg=EEG_FS,
            reject_invalid=True,
            flat_std_threshold=(
                EEG_FLAT_STD_THRESHOLD
            ),
            return_valid_mask=True,
        )
    )

    if (
        valid_mask.shape
        != (WINDOW_EPOCHS,)
    ):
        raise RuntimeError(
            "Unexpected EEG validity-mask shape: "
            f"{valid_mask.shape}. "
            f"Expected {(WINDOW_EPOCHS,)}."
        )

    # --------------------------------------------------------
    # Arthur's frequency resizing
    # --------------------------------------------------------

    (
        spectrogram,
        freqs,
    ) = resample_frequency_axis(
        spectrogram=spectrogram,
        freqs=freqs,
        n_freq_bins=256,
    )

    # --------------------------------------------------------
    # Arthur's PSD -> dB conversion
    # --------------------------------------------------------

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

    # Only valid columns are required to be finite.
    if valid_mask.any():

        if not np.isfinite(
            spectrogram_db[
                :,
                valid_mask,
            ]
        ).all():

            raise RuntimeError(
                "Non-finite values found "
                "inside valid EEG spectrogram epochs."
            )

    # --------------------------------------------------------
    # Temporary placeholder
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # The final TRAIN min_db is not known yet.
    #
    # These columns are completely excluded from the TRAIN
    # quantile collector. After min_db has been computed,
    # they are rewritten to min_db so that the existing
    # VQGAN loader automatically maps them to exactly 0.
    #
    # This means train_vqgan_multidataset.py does NOT need
    # to be modified.

    spectrogram_db[
        :,
        ~valid_mask,
    ] = 0.0

    return (
        spectrogram_db,
        freqs,
        valid_mask,
    )


# ============================================================
# Process one source
# ============================================================

def process_source(
    source,
    output_root,
    db_collector,
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

    metadata = (
        load_metadata(
            pickle_path
        )
    )

    # --------------------------------------------------------
    # Decide where normalization statistics come from.
    # --------------------------------------------------------

    stats_source_name = (
        source.get(
            "stats_from",
            source["name"],
        )
    )

    if (
        stats_source_name
        == source["name"]
    ):

        stats_metadata = (
            metadata
        )

    else:

        stats_source = (
            SOURCE_BY_NAME[
                stats_source_name
            ]
        )

        stats_metadata = (
            load_metadata(
                stats_source[
                    "pickle"
                ]
            )
        )

    stats = (
        compute_sleepmami_stats(
            stats_metadata,
            stats_source_name,
        )
    )

    print(
        "normalization stats from:",
        stats_source_name,
    )

    print(
        "subjects:",
        len(
            metadata[
                "sig_info"
            ]
        ),
    )

    print(
        "epochs used for stats:",
        stats[
            "num_epochs"
        ],
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
        stats[
            "scaled_subjects"
        ],
    )

    print(
        "excluded subjects in stats:",
        stats[
            "excluded_subjects"
        ],
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
            metadata[
                "data_shape"
            ]
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

    total_valid_epochs = 0
    total_rejected_epochs = 0

    sig_len = int(
        metadata[
            "data_shape"
        ][1]
    )

    # --------------------------------------------------------
    # Subjects
    # --------------------------------------------------------

    for (
        subject_id,
        info,
    ) in metadata[
        "sig_info"
    ].items():

        if should_exclude_subject(
            source["name"],
            subject_id,
        ):
            continue

        start, end = (
            info["pos"]
        )

        num_epochs = (
            end - start
        )

        num_windows = (
            num_epochs
            // WINDOW_EPOCHS
        )

        if num_windows == 0:
            continue

        scale = (
            subject_scale_factor(
                source["name"],
                info,
                sig_len,
            )
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

                    if (
                        "eeg_valid_mask"
                        not in saved.files
                    ):
                        raise RuntimeError(
                            f"{output_path} was "
                            "generated before the "
                            "EEG rejection rule. "
                            "Rerun with --overwrite."
                        )

                    spectrogram_db = (
                        saved[
                            "eeg_spectrogram_db"
                        ].astype(
                            np.float32,
                            copy=False,
                        )
                    )

                    valid_mask = (
                        saved[
                            "eeg_valid_mask"
                        ].astype(
                            bool
                        )
                    )

            else:

                (
                    spectrogram_db,
                    freqs,
                    valid_mask,
                ) = make_spectrogram(
                    eeg_epochs=eeg_epochs,
                    mean=stats[
                        "mean"
                    ],
                    std=stats[
                        "std"
                    ],
                )

                np.savez(
                    output_path,

                    eeg_spectrogram_db=(
                        spectrogram_db
                    ),

                    eeg_valid_mask=(
                        valid_mask.astype(
                            np.uint8
                        )
                    ),

                    freqs=freqs,

                    dataset=np.asarray(
                        source[
                            "name"
                        ]
                    ),

                    split=np.asarray(
                        source[
                            "split"
                        ]
                    ),

                    subject_id=np.asarray(
                        str(
                            subject_id
                        )
                    ),

                    window_index=np.asarray(
                        window_index,
                        dtype=np.int32,
                    ),

                    eeg_mean=np.asarray(
                        stats[
                            "mean"
                        ],
                        dtype=np.float32,
                    ),

                    eeg_std=np.asarray(
                        stats[
                            "std"
                        ],
                        dtype=np.float32,
                    ),

                    scale_factor=np.asarray(
                        scale,
                        dtype=np.float32,
                    ),

                    normalization_source=(
                        np.asarray(
                            stats_source_name
                        )
                    ),
                )

            n_valid = int(
                valid_mask.sum()
            )

            n_rejected = int(
                valid_mask.size
                - n_valid
            )

            total_valid_epochs += (
                n_valid
            )

            total_rejected_epochs += (
                n_rejected
            )

            # ------------------------------------------------
            # TRAIN dB bounds:
            # ONLY valid epochs contribute.
            # ------------------------------------------------

            if (
                source["split"]
                == "train"
            ):

                db_collector.add(
                    spectrogram_db,
                    valid_mask=(
                        valid_mask
                    ),
                )

            manifests[
                source["split"]
            ].append(
                {
                    "path": str(
                        output_path
                    ),

                    "dataset": (
                        source[
                            "name"
                        ]
                    ),

                    "subject_id": (
                        str(
                            subject_id
                        )
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

                    "valid_eeg_epochs": (
                        n_valid
                    ),

                    "rejected_eeg_epochs": (
                        n_rejected
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

    print(
        "valid EEG epochs:",
        total_valid_epochs,
    )

    print(
        "rejected EEG epochs:",
        total_rejected_epochs,
    )

    if (
        total_valid_epochs
        + total_rejected_epochs
        > 0
    ):

        rejection_rate = (
            100.0
            * total_rejected_epochs
            / (
                total_valid_epochs
                + total_rejected_epochs
            )
        )

        print(
            "EEG rejection rate:",
            f"{rejection_rate:.4f}%",
        )

    return (
        stats,
        source_windows,
    )


# ============================================================
# Finalize invalid EEG epochs
# ============================================================

def finalize_invalid_epochs(
    manifests,
    min_db,
):
    """
    Replace invalid EEG epochs by min_db in every stored
    dB spectrogram.

    The existing VQGAN loader performs:

        (eeg_db - min_db) / (max_db - min_db)

    Therefore min_db maps exactly to 0.0.

    This reproduces Keondo's final behaviour:
        normalized_spec[~valid] = 0.0

    without requiring any change to the VQGAN training loader.
    """

    print()
    print("=" * 72)

    print(
        "FINALIZING INVALID EEG EPOCHS"
    )

    num_files = 0
    num_invalid_epochs = 0

    for entries in manifests.values():

        for entry in entries:

            path = Path(
                entry["path"]
            )

            with np.load(
                path,
                allow_pickle=False,
            ) as sample:

                if (
                    "eeg_valid_mask"
                    not in sample.files
                ):
                    raise RuntimeError(
                        "eeg_valid_mask missing "
                        f"from {path}"
                    )

                payload = {
                    key: sample[
                        key
                    ]
                    for key
                    in sample.files
                }

            valid_mask = (
                payload[
                    "eeg_valid_mask"
                ].astype(
                    bool
                )
            )

            eeg_db = np.asarray(
                payload[
                    "eeg_spectrogram_db"
                ],
                dtype=np.float32,
            ).copy()

            if (
                valid_mask.shape
                != (
                    eeg_db.shape[1],
                )
            ):
                raise RuntimeError(
                    "Mask/spectrogram shape "
                    f"mismatch in {path}: "
                    f"{valid_mask.shape} vs "
                    f"{eeg_db.shape}"
                )

            invalid_mask = (
                ~valid_mask
            )

            n_invalid = int(
                invalid_mask.sum()
            )

            num_invalid_epochs += (
                n_invalid
            )

            # Most files have no rejected epochs. Avoid rewriting
            # an entire NPZ on NFS when there is nothing to change.
            if n_invalid == 0:
                num_files += 1
                continue

            # Final dB representation of invalid epochs.
            #
            # The existing [0,1] loader maps this
            # value exactly to zero.
            eeg_db[
                :,
                invalid_mask,
            ] = np.float32(
                min_db
            )

            payload[
                "eeg_spectrogram_db"
            ] = eeg_db

            tmp_path = path.with_suffix(".tmp.npz")

            np.savez(
                tmp_path,
                **payload,
            )

            tmp_path.replace(path)

            num_files += 1

    print(
        "files finalized:",
        num_files,
    )

    print(
        "invalid epochs mapped to min_db:",
        num_invalid_epochs,
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = (
        argparse.ArgumentParser(
            description=(
                "Preprocess the multi-dataset "
                "EEG pool for shared VQGAN "
                "training."
            )
        )
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "outputs/"
            "vqgan_multidataset_preprocessed_artifact_all_db"
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
        "--check-subject-counts-only",
        action="store_true",
        help=(
            "Print metadata/exclusion/usable-subject counts and exit "
            "without generating spectrograms."
        ),
    )

    parser.add_argument(
        "--quantile-low",
        type=float,
        default=0.001,
        help=(
            "Lower dB quantile in [0,1]. "
            "0.001 = 0.1th percentile."
        ),
    )

    parser.add_argument(
        "--quantile-high",
        type=float,
        default=0.999,
        help=(
            "Upper dB quantile in [0,1]. "
            "0.999 = 99.9th percentile."
        ),
    )

    parser.add_argument(
        "--db-cache-path",
        type=Path,
        default=None,
        help=(
            "Optional local binary cache for ALL valid TRAIN dB values. "
            "Default: /tmp/vqgan_train_valid_db_<pid>.float32"
        ),
    )

    parser.add_argument(
        "--keep-db-cache",
        action="store_true",
        help=(
            "Keep the temporary all-dB binary cache after percentile "
            "calculation. By default it is deleted."
        ),
    )

    args = (
        parser.parse_args()
    )

    if args.check_subject_counts_only:
        print_subject_count_check()
        return

    if not (
        0.0 <= args.quantile_low
        < args.quantile_high
        <= 1.0
    ):
        raise ValueError(
            "Quantiles must satisfy "
            "0 <= quantile_low < quantile_high <= 1. "
            f"Got {args.quantile_low}, {args.quantile_high}."
        )

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    db_collector = DBCollector(
        cache_path=(
            args.db_cache_path
        ),
    )

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

        stats, count = (
            process_source(
                source=source,
                output_root=(
                    args.output_root
                ),
                db_collector=(
                    db_collector
                ),
                manifests=(
                    manifests
                ),
                max_windows_per_source=(
                    args.max_windows_per_source
                ),
                overwrite=(
                    args.overwrite
                ),
            )
        )

        waveform_stats[
            source["name"]
        ] = stats

        window_counts[
            source["name"]
        ] = count

    # --------------------------------------------------------
    # TRAIN-only exact dB normalization
    #
    # ALL finite dB values from ALL valid TRAIN EEG epochs
    # contribute. There is no random sampling or value cap.
    # --------------------------------------------------------

    min_db, max_db = (
        db_collector.compute_quantiles(
            quantile_low=(
                args.quantile_low
            ),
            quantile_high=(
                args.quantile_high
            ),
        )
    )

    total_train_db_values = (
        db_collector.total_values
    )

    db_cache_path = str(
        db_collector.cache_path
    )

    # Save the exact bounds immediately, before the final NPZ rewrite.
    # If NFS fails during finalization, the expensive quantile result
    # is still preserved on disk.
    bounds_checkpoint_path = (
        args.output_root
        / "exact_db_bounds.json"
    )

    with bounds_checkpoint_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "min_db": min_db,
                "max_db": max_db,
                "quantile_low": args.quantile_low,
                "quantile_high": args.quantile_high,
                "percentile_low": 100.0 * args.quantile_low,
                "percentile_high": 100.0 * args.quantile_high,
                "num_values": total_train_db_values,
                "mode": "all_valid_train_db_values_exact",
                "dtype": "float32",
                "cache_path": db_cache_path,
            },
            file,
            indent=2,
        )

    if max_db <= min_db:
        raise RuntimeError(
            "Invalid dB normalization bounds: "
            f"{min_db}, {max_db}"
        )

    # --------------------------------------------------------
    # Now that min_db is known, map every rejected epoch
    # to min_db.
    #
    # Existing VQGAN loader:
    #
    #   (min_db - min_db) /
    #   (max_db - min_db)
    #
    # = 0 exactly.
    # --------------------------------------------------------

    finalize_invalid_epochs(
        manifests=manifests,
        min_db=min_db,
    )

    # --------------------------------------------------------
    # Save normalization metadata
    # --------------------------------------------------------

    total_valid_epochs = sum(
        entry.get(
            "valid_eeg_epochs",
            0,
        )
        for entries
        in manifests.values()
        for entry
        in entries
    )

    total_rejected_epochs = sum(
        entry.get(
            "rejected_eeg_epochs",
            0,
        )
        for entries
        in manifests.values()
        for entry
        in entries
    )

    total_epochs = (
        total_valid_epochs
        + total_rejected_epochs
    )

    if total_epochs > 0:

        overall_rejection_rate = (
            total_rejected_epochs
            / total_epochs
        )

    else:

        overall_rejection_rate = 0.0

    normalization = {

        "min_db": min_db,
        "max_db": max_db,

        "quantile_low": (
            args.quantile_low
        ),

        "quantile_high": (
            args.quantile_high
        ),

        "percentile_low": (
            100.0 * args.quantile_low
        ),

        "percentile_high": (
            100.0 * args.quantile_high
        ),

        "db_percentile_estimation": {
            "mode": (
                "all_valid_train_db_values_exact"
            ),
            "num_values": (
                total_train_db_values
            ),
            "dtype": "float32",
            "temporary_cache": (
                db_cache_path
            ),
            "cache_kept": bool(
                args.keep_db_cache
            ),
        },

        "waveform_stats": (
            waveform_stats
        ),

        "window_counts": (
            window_counts
        ),

        "eeg_rejection": {
            "enabled": True,

            "rule": (
                "finite AND std > 1e-4; "
                "multitaper only on usable "
                "epochs; reject non-finite "
                "dB PSD outputs"
            ),

            "flat_std_threshold": (
                EEG_FLAT_STD_THRESHOLD
            ),

            "total_valid_epochs": (
                total_valid_epochs
            ),

            "total_rejected_epochs": (
                total_rejected_epochs
            ),

            "overall_rejection_rate": (
                overall_rejection_rate
            ),

            "invalid_final_value": (
                "min_db -> 0 after [0,1] "
                "normalization"
            ),
        },

        "excluded_subjects": {
            dataset: sorted(subjects)
            for dataset, subjects
            in EXCLUDED_SUBJECTS.items()
        },

        "mros2_low_scale_std_threshold": (
            MROS2_LOW_SCALE_STD_THRESHOLD
        ),

        "mros2_low_scale_factor": (
            MROS2_LOW_SCALE_FACTOR
        ),

        "validation_normalization": {
            "shhs1_val": (
                "shhs1_train"
            ),
            "kiss_val": (
                "kiss_train"
            ),
            "mesa_val": (
                "mesa_train"
            ),
            "mesa_test": (
                "mesa_train"
            ),
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

    for (
        split,
        entries,
    ) in manifests.items():

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
            manifests[
                "train"
            ]
        ),
    )

    print(
        "Validation windows:",
        len(
            manifests[
                "val"
            ]
        ),
    )

    print(
        "Test windows:",
        len(
            manifests[
                "test"
            ]
        ),
    )

    print(
        "Valid EEG epochs:",
        total_valid_epochs,
    )

    print(
        "Rejected EEG epochs:",
        total_rejected_epochs,
    )

    print(
        "Overall rejection rate:",
        f"{100.0 * overall_rejection_rate:.4f}%",
    )

    print(
        "Exact TRAIN dB bounds:",
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

    print(
        "Exact dB bounds checkpoint:",
        bounds_checkpoint_path,
    )

    # Delete the large local binary cache only after the complete
    # preprocessing pipeline succeeded. If the run fails earlier,
    # the cache is intentionally left in place for debugging/recovery.
    if not args.keep_db_cache:
        db_collector.cleanup()


if __name__ == "__main__":
    main()