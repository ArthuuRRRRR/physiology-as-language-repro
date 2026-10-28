from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


IGNORE_INDEX = -100

STAGE_NAMES = (
    "Wake",
    "Light",
    "Deep",
    "REM",
)


def safe_torch_load(path: Path):

    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


class SleepStagingNightDataset(Dataset):

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        input_type: str,
    ) -> None:

        self.data_root = Path(data_root)

        if split not in {
            "train",
            "val",
            "test",
        }:
            raise ValueError(
                f"Unknown split: {split}"
            )

        if input_type not in {
            "ground_truth",
            "synthesized",
        }:
            raise ValueError(
                f"Unknown input type: {input_type}"
            )

        self.split = split
        self.input_type = input_type

        self.split_dir = (
            self.data_root
            / split
        )

        if not self.split_dir.exists():
            raise FileNotFoundError(
                f"Missing split directory: {self.split_dir}"
            )

        self.files = sorted(
            self.split_dir.glob("*.pt")
        )

        if not self.files:
            raise RuntimeError(
                f"No .pt PSG files found in {self.split_dir}"
            )

    def __len__(
        self,
    ) -> int:

        return len(self.files)

    def __getitem__(
        self,
        index: int,
    ) -> dict:

        path = self.files[index]

        item = safe_torch_load(
            path
        )

        if not isinstance(
            item,
            dict,
        ):
            raise TypeError(
                f"{path} must contain a dictionary."
            )

        if self.input_type not in item:
            raise KeyError(
                f"{path} does not contain "
                f"'{self.input_type}'. "
                f"Available keys: {list(item.keys())}"
            )

        if "labels" not in item:
            raise KeyError(
                f"{path} does not contain labels."
            )

        eeg = torch.as_tensor(
            item[self.input_type],
            dtype=torch.float32,
        )

        labels = torch.as_tensor(
            item["labels"],
            dtype=torch.long,
        ).reshape(-1)

        # ----------------------------------------------------
        # Normalize spectrogram shape
        # ----------------------------------------------------

        if (
            eeg.ndim == 3
            and eeg.shape[0] == 1
        ):
            eeg = eeg[0]

        if eeg.ndim != 2:
            raise ValueError(
                f"{path}: expected a 2D spectrogram, "
                f"got {tuple(eeg.shape)}"
            )

        # Expected = (256,T)
        if eeg.shape[0] == 256:
            pass

        elif eeg.shape[1] == 256:
            eeg = eeg.transpose(
                0,
                1,
            )

        else:
            raise ValueError(
                f"{path}: spectrogram must have 256 "
                f"frequency bins, got {tuple(eeg.shape)}"
            )

        # ----------------------------------------------------
        # Check temporal alignment
        # ----------------------------------------------------

        if eeg.shape[1] != labels.numel():
            raise ValueError(
                f"{path}: spectrogram contains "
                f"{eeg.shape[1]} epochs but labels "
                f"contains {labels.numel()}."
            )

        # ----------------------------------------------------
        # Check labels
        # ----------------------------------------------------

        unique_labels = set(
            labels.unique().tolist()
        )

        allowed_labels = {
            0,
            1,
            2,
            3,
            IGNORE_INDEX,
        }

        if not unique_labels.issubset(
            allowed_labels
        ):
            raise ValueError(
                f"{path}: unexpected labels "
                f"{sorted(unique_labels)}. "
                f"Expected 0,1,2,3 or {IGNORE_INDEX}."
            )

        valid_mask = (
            labels
            != IGNORE_INDEX
        )

        night_id = str(
            item.get(
                "night_id",
                path.stem,
            )
        )

        dataset = str(
            item.get(
                "dataset",
                "shhs1",
            )
        )

        return {
            "eeg": eeg,
            "labels": labels,
            "valid_mask": valid_mask,
            "night_id": night_id,
            "dataset": dataset,
        }


def collate_sleep_nights(
    batch: list[dict],
) -> dict:

    if not batch:
        raise ValueError(
            "Empty batch."
        )

    batch_size = len(
        batch
    )

    freq_bins = batch[
        0
    ]["eeg"].shape[0]

    max_t = max(
        item["eeg"].shape[1]
        for item in batch
    )

    eeg = torch.zeros(
        (
            batch_size,
            freq_bins,
            max_t,
        ),
        dtype=torch.float32,
    )

    labels = torch.full(
        (
            batch_size,
            max_t,
        ),
        fill_value=IGNORE_INDEX,
        dtype=torch.long,
    )

    valid_mask = torch.zeros(
        (
            batch_size,
            max_t,
        ),
        dtype=torch.bool,
    )

    night_ids = []
    datasets = []

    for i, item in enumerate(
        batch
    ):

        t = item[
            "eeg"
        ].shape[1]

        eeg[
            i,
            :,
            :t,
        ] = item[
            "eeg"
        ]

        labels[
            i,
            :t,
        ] = item[
            "labels"
        ]

        valid_mask[
            i,
            :t,
        ] = item[
            "valid_mask"
        ]

        night_ids.append(
            item["night_id"]
        )

        datasets.append(
            item["dataset"]
        )

    return {
        "eeg": eeg,
        "labels": labels,
        "valid_mask": valid_mask,
        "night_id": night_ids,
        "dataset": datasets,
    }

