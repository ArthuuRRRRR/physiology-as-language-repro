from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset


FREQ_BINS = 256
WINDOW_SEGMENTS = 512
IGNORE_INDEX = -100


class SleepStagingMMapDataset(Dataset):
    """
    Sleep-staging dataset backed by mmap files.

    Expected directory structure:

        split_dir/
            synthesized_eeg.mmap
            ground_truth_eeg.mmap
            labels.mmap
            index.jsonl
            metadata.json

    EEG:
        (N, 256, 512)

    Labels:
        (N, 512)

    Returned EEG tensor:
        (1, 256, 512)

    Returned labels:
        (512,)
    """

    def __init__(
        self,
        split_dir: str | Path,
        *,
        input_type: str = "synth",
    ) -> None:

        super().__init__()

        self.split_dir = Path(split_dir)

        if input_type not in {"synth", "gt"}:
            raise ValueError(
                f"input_type must be 'synth' or 'gt', got {input_type!r}"
            )

        self.input_type = input_type

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        metadata_path = self.split_dir / "metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)

        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            self.metadata = json.load(f)

        self.num_samples = int(
            self.metadata["num_samples"]
        )

        eeg_shape = tuple(
            self.metadata["eeg_shape"]
        )

        label_shape = tuple(
            self.metadata["label_shape"]
        )

        expected_eeg_shape = (
            self.num_samples,
            FREQ_BINS,
            WINDOW_SEGMENTS,
        )

        expected_label_shape = (
            self.num_samples,
            WINDOW_SEGMENTS,
        )

        if eeg_shape != expected_eeg_shape:
            raise ValueError(
                f"Unexpected EEG shape in {metadata_path}: "
                f"{eeg_shape}, expected {expected_eeg_shape}"
            )

        if label_shape != expected_label_shape:
            raise ValueError(
                f"Unexpected label shape in {metadata_path}: "
                f"{label_shape}, expected {expected_label_shape}"
            )

        # ----------------------------------------------------
        # Paths
        # ----------------------------------------------------

        if self.input_type == "synth":
            self.eeg_path = (
                self.split_dir
                / "synthesized_eeg.mmap"
            )
        else:
            self.eeg_path = (
                self.split_dir
                / "ground_truth_eeg.mmap"
            )

        self.labels_path = (
            self.split_dir
            / "labels.mmap"
        )

        self.index_path = (
            self.split_dir
            / "index.jsonl"
        )

        for path in (
            self.eeg_path,
            self.labels_path,
            self.index_path,
        ):
            if not path.exists():
                raise FileNotFoundError(path)

        # ----------------------------------------------------
        # Index information
        # ----------------------------------------------------

        self.index: list[dict[str, Any]] = []

        with self.index_path.open(
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                self.index.append(
                    json.loads(line)
                )

        if len(self.index) != self.num_samples:
            raise RuntimeError(
                f"{self.index_path}: "
                f"{len(self.index)} index entries "
                f"but metadata says {self.num_samples} samples."
            )

        # ----------------------------------------------------
        # mmap objects are opened lazily.
        #
        # Important when using DataLoader(num_workers > 0).
        # ----------------------------------------------------

        self._eeg_mmap = None
        self._labels_mmap = None

    def _open_mmaps(self) -> None:

        if self._eeg_mmap is None:

            self._eeg_mmap = np.memmap(
                self.eeg_path,
                mode="r",
                dtype=np.float16,
                shape=(
                    self.num_samples,
                    FREQ_BINS,
                    WINDOW_SEGMENTS,
                ),
            )

        if self._labels_mmap is None:

            self._labels_mmap = np.memmap(
                self.labels_path,
                mode="r",
                dtype=np.int8,
                shape=(
                    self.num_samples,
                    WINDOW_SEGMENTS,
                ),
            )

    def __len__(self) -> int:

        return self.num_samples

    def __getitem__(
        self,
        idx: int,
    ) -> dict[str, Any]:

        self._open_mmaps()

        # np.array(..., copy=True) is deliberate:
        # PyTorch should not directly reference the read-only mmap.
        eeg_np = np.array(
            self._eeg_mmap[idx],
            dtype=np.float32,
            copy=True,
        )

        labels_np = np.array(
            self._labels_mmap[idx],
            dtype=np.int64,
            copy=True,
        )

        eeg = torch.from_numpy(
            eeg_np
        ).unsqueeze(0)

        labels = torch.from_numpy(
            labels_np
        )

        info = self.index[idx]

        return {
            "eeg": eeg,
            "labels": labels,
            "dataset": str(
                info.get(
                    "dataset",
                    self.metadata.get(
                        "dataset",
                        "unknown",
                    ),
                )
            ),
            "subject_id": str(
                info.get(
                    "subject_id",
                    "unknown",
                )
            ),
            "subject_id_normalized": str(
                info.get(
                    "subject_id_normalized",
                    "",
                )
            ),
            "window_index": int(
                info.get(
                    "window_index",
                    0,
                )
            ),
            "valid_segments": int(
                info.get(
                    "valid_segments",
                    WINDOW_SEGMENTS,
                )
            ),
        }

    def __getstate__(self):

        state = self.__dict__.copy()

        state["_eeg_mmap"] = None
        state["_labels_mmap"] = None

        return state


def build_multi_dataset(
    split_dirs: list[str | Path],
    *,
    input_type: str = "synth",
) -> Dataset:
    """
    Concatenate SHHS1 / MESA / CFS datasets while preserving
    their dataset and subject metadata.
    """

    datasets = [
        SleepStagingMMapDataset(
            path,
            input_type=input_type,
        )
        for path in split_dirs
    ]

    if not datasets:
        raise ValueError(
            "No dataset directories supplied."
        )

    if len(datasets) == 1:
        return datasets[0]

    return ConcatDataset(
        datasets
    )
