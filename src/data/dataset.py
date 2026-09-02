from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import (
    normalize_db_spectrogram,
)


class PhysiologyPairDataset(Dataset):
    """
    Dataset for preprocessed respiration-EEG pairs.

    It supports:
    - legacy spectrograms already normalized to [0, 1];
    - new SHHS spectrograms saved in dB.

    SHHS window metadata is returned when available so that
    evaluation can be grouped by recording and temporal offset.
    """

    def __init__(
        self,
        data_dir,
        min_db=None,
        max_db=None,
    ):
        if isinstance(data_dir, (str, Path)):
            self.data_dirs = [Path(data_dir)]
        else:
            self.data_dirs = [
                Path(path)
                for path in data_dir
            ]

        for directory in self.data_dirs:
            if not directory.exists():
                raise FileNotFoundError(
                    "Dataset directory not found: "
                    f"{directory}"
                )

        self.files = sorted(
            file_path
            for directory in self.data_dirs
            for file_path in directory.glob("*.npz")
        )

        if not self.files:
            raise RuntimeError(
                "No .npz samples found in: "
                f"{self.data_dirs}"
            )

        if (min_db is None) != (max_db is None):
            raise ValueError(
                "min_db and max_db must be provided together"
            )

        self.min_db = min_db
        self.max_db = max_db

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_path = self.files[index]

        with np.load(file_path) as sample:
            respiration = sample[
                "respiration"
            ].astype(np.float32)

            if "eeg_spectrogram_db" in sample:
                if (
                    self.min_db is None
                    or self.max_db is None
                ):
                    raise ValueError(
                        "This sample contains EEG in dB. "
                        "Training normalization bounds "
                        "min_db and max_db are required."
                    )

                eeg_db = sample[
                    "eeg_spectrogram_db"
                ].astype(np.float32)

                eeg_spectrogram = (
                    normalize_db_spectrogram(
                        eeg_db,
                        min_db=self.min_db,
                        max_db=self.max_db,
                    )
                )

            elif "eeg_spectrogram" in sample:
                eeg_spectrogram = sample[
                    "eeg_spectrogram"
                ].astype(np.float32)

            elif "eeg" in sample:
                eeg_spectrogram = sample[
                    "eeg"
                ].astype(np.float32)

            else:
                raise KeyError(
                    "No EEG spectrogram found in "
                    f"{file_path}. "
                    f"Available keys: {sample.files}"
                )

            freqs = None

            if "freqs" in sample:
                freqs = sample[
                    "freqs"
                ].astype(np.float32)

            patient_id = None

            if "patient_id" in sample:
                patient_id = str(
                    sample["patient_id"]
                )

            fold = None

            if "fold" in sample:
                fold = int(
                    sample["fold"]
                )

            start_sec = None

            if "start_sec" in sample:
                start_sec = float(
                    sample["start_sec"]
                )

            source_file = None

            if "source_file" in sample:
                source_file = str(
                    sample["source_file"]
                )

            visit = None

            if "visit" in sample:
                visit = str(
                    sample["visit"]
                )

        if respiration.ndim != 2:
            raise ValueError(
                "Invalid respiration shape in "
                f"{file_path}: {respiration.shape}"
            )

        if eeg_spectrogram.shape != (256, 512):
            raise ValueError(
                "Invalid EEG spectrogram shape in "
                f"{file_path}: "
                f"{eeg_spectrogram.shape}"
            )

        if not np.isfinite(respiration).all():
            raise ValueError(
                f"Non-finite respiration in {file_path}"
            )

        if not np.isfinite(eeg_spectrogram).all():
            raise ValueError(
                f"Non-finite EEG in {file_path}"
            )

        output = {
            "respiration": torch.from_numpy(
                respiration
            ),
            "eeg_spectrogram": torch.from_numpy(
                eeg_spectrogram
            ),
            "file_path": str(file_path),
        }

        if freqs is not None:
            output["freqs"] = torch.from_numpy(
                freqs
            )

        if patient_id is not None:
            output["patient_id"] = patient_id

        if fold is not None:
            output["fold"] = fold

        if start_sec is not None:
            output["start_sec"] = start_sec

        if source_file is not None:
            output["source_file"] = source_file

        if visit is not None:
            output["visit"] = visit

        return output