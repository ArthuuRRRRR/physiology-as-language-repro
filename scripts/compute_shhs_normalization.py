import json
from pathlib import Path

import numpy as np


DATA_DIR = Path(
    "outputs/shhs_preprocessed"
)

OUTPUT_PATH = Path(
    "outputs/normalization/shhs_cv_normalization.json"
)

NUM_FOLDS = 4

# Reproduction choices:
# the paper does not specify the exact normalization bounds.
SAMPLES_PER_FILE = 2048
LOWER_PERCENTILE = 0.1
UPPER_PERCENTILE = 99.9
SEED = 42


def sample_fold(fold_index):
    """
    Sample dB values from every spectrogram of one fold.
    """

    fold_dir = DATA_DIR / f"fold_{fold_index}"

    files = sorted(
        fold_dir.glob("*.npz")
    )

    if not files:
        raise RuntimeError(
            f"No NPZ files found in {fold_dir}"
        )

    rng = np.random.default_rng(
        SEED + fold_index
    )

    sampled_values = []

    for index, file_path in enumerate(
        files,
        start=1,
    ):
        with np.load(
            file_path,
            allow_pickle=False,
        ) as sample:
            if "eeg_spectrogram_db" not in sample:
                raise KeyError(
                    "Missing eeg_spectrogram_db in "
                    f"{file_path}"
                )

            values = sample[
                "eeg_spectrogram_db"
            ].astype(
                np.float32,
                copy=False,
            ).ravel()

        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite values in {file_path}"
            )

        sample_size = min(
            SAMPLES_PER_FILE,
            values.size,
        )

        selected_indices = rng.choice(
            values.size,
            size=sample_size,
            replace=False,
        )

        sampled_values.append(
            values[selected_indices]
        )

        if index % 500 == 0:
            print(
                f"Fold {fold_index}: "
                f"{index}/{len(files)} files"
            )

    return np.concatenate(
        sampled_values
    ).astype(np.float32)


def main():
    fold_samples = {}

    for fold_index in range(NUM_FOLDS):
        print(
            f"\nSampling fold {fold_index}"
        )

        fold_samples[fold_index] = (
            sample_fold(fold_index)
        )

        print(
            "Sampled values:",
            len(fold_samples[fold_index]),
        )

    runs = {}

    for held_out_fold in range(NUM_FOLDS):
        train_folds = [
            fold_index
            for fold_index in range(NUM_FOLDS)
            if fold_index != held_out_fold
        ]

        train_values = np.concatenate(
            [
                fold_samples[fold_index]
                for fold_index in train_folds
            ]
        )

        min_db, max_db = np.percentile(
            train_values,
            [
                LOWER_PERCENTILE,
                UPPER_PERCENTILE,
            ],
        )

        runs[str(held_out_fold)] = {
            "held_out_fold": held_out_fold,
            "train_folds": train_folds,
            "min_db": float(min_db),
            "max_db": float(max_db),
            "sampled_values": int(
                train_values.size
            ),
        }

        print(
            f"Held-out fold {held_out_fold} | "
            f"train={train_folds} | "
            f"min={min_db:.2f} dB | "
            f"max={max_db:.2f} dB"
        )

    output = {
        "version": 1,
        "dataset": "SHHS",
        "lower_percentile": LOWER_PERCENTILE,
        "upper_percentile": UPPER_PERCENTILE,
        "samples_per_file": SAMPLES_PER_FILE,
        "seed": SEED,
        "reproduction_choice": (
            "The paper specifies normalization to "
            "[0,1] but does not provide the exact bounds."
        ),
        "runs": runs,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()

