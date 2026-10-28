from __future__ import annotations

import json
import numpy as np
from collections import defaultdict
from pathlib import Path


SPLITS = {
    "train": Path(
        "outputs/vqgan_multidataset_preprocessed_final_gpu4/"
        "train_manifest.jsonl"
    ),
    "val": Path(
        "outputs/vqgan_multidataset_preprocessed_final_gpu4/"
        "val_manifest.jsonl"
    ),
    "test": Path(
        "outputs/vqgan_shhs1_test_preprocessed_final/"
        "test_manifest.jsonl"
    ),
}

OUTPUT = Path(
    "outputs/downstream_survival_features_gt"
)

MIN_DB = -52.59674347305298
MAX_DB = 20.398672103881836


def main():
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    for split, manifest_path in SPLITS.items():

        by_subject = defaultdict(list)

        with manifest_path.open(
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:
                if not line.strip():
                    continue

                row = json.loads(line)

                subject = str(
                    row["subject_id"]
                )

                if not subject.startswith(
                    "shhs1-"
                ):
                    continue

                by_subject[subject].append(
                    row
                )

        subject_ids = []
        features = []

        for subject in sorted(
            by_subject
        ):

            # Sum frequency bins across every
            # available 30-s column/window.
            freq_sum = np.zeros(
                256,
                dtype=np.float64,
            )

            count = 0

            rows = sorted(
                by_subject[subject],
                key=lambda x: int(
                    x["window_index"]
                ),
            )

            for row in rows:

                path = Path(
                    row["path"]
                )

                with np.load(
                    path,
                    allow_pickle=False,
                ) as z:

                    eeg_db = np.asarray(
                        z["eeg_spectrogram_db"],
                        dtype=np.float32,
                    )

                eeg = np.clip(
                    eeg_db,
                    MIN_DB,
                    MAX_DB,
                )

                eeg = (
                    eeg - MIN_DB
                ) / (
                    MAX_DB - MIN_DB
                )

                freq_sum += eeg.sum(
                    axis=1,
                    dtype=np.float64,
                )

                count += eeg.shape[1]

            feature = (
                freq_sum / count
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
            features
        )

        np.save(
            OUTPUT / f"{split}_features.npy",
            features,
        )

        with (
            OUTPUT
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
            "Subjects:",
            len(subject_ids),
        )
        print(
            "Features:",
            features.shape,
        )
        print(
            "Mean:",
            float(features.mean()),
        )
        print(
            "Std:",
            float(features.std()),
        )


if __name__ == "__main__":
    main()
