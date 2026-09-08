import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.models.vqgan import VQGAN


DATA_ROOT = Path("/hdd2/kdpark/sleep_datasets")

EEG_FREQ_BINS = 256
EEG_TIME_BINS = 512

RESP_ORIGINAL_FS = 100
RESP_TARGET_FS = 10

WINDOW_EPOCHS = 512
EPOCH_SEC = 30

RESP_SEGMENT_SEC = 240
RESP_SEGMENTS = 64
RESP_SAMPLES_PER_SEGMENT = 2400

EXPECTED_RESP_SAMPLES = (
    RESP_SEGMENTS
    * RESP_SAMPLES_PER_SEGMENT
)


ABD_PATHS = {
    "phy_train": (
        DATA_ROOT
        / "physionet_2018"
        / "mmap_signals_MAE_processed"
        / "ABD"
        / "ABD.pickle"
    ),

    "phy_test": (
        DATA_ROOT
        / "physionet_test"
        / "mmap_signals_MAE_processed"
        / "ABD"
        / "ABD.pickle"
    ),

    "shhs1_train": (
        DATA_ROOT
        / "shhs1"
        / "mmap_signals_MAE_processed"
        / "train"
        / "ABD"
        / "ABD.pickle"
    ),

    "shhs1_val": (
        DATA_ROOT
        / "shhs1"
        / "mmap_signals_MAE_processed"
        / "val"
        / "ABD"
        / "ABD.pickle"
    ),

    "shhs2": (
        DATA_ROOT
        / "shhs2"
        / "mmap_signals_MAE_processed"
        / "train"
        / "ABD"
        / "ABD.pickle"
    ),

    "kiss_train": (
        DATA_ROOT
        / "kiss"
        / "mmap_signals_MAE_processed"
        / "train"
        / "ABD"
        / "ABD.pickle"
    ),

    "kiss_val": (
        DATA_ROOT
        / "kiss"
        / "mmap_signals_MAE_processed"
        / "val"
        / "ABD"
        / "ABD.pickle"
    ),

    "kvss": (
        DATA_ROOT
        / "kvss"
        / "mmap_signals_MAE_processed"
        / "train"
        / "ABD"
        / "ABD.pickle"
    ),

    "mros1_train": (
        DATA_ROOT
        / "mros1"
        / "mmap_signals_MAE_processed"
        / "train"
        / "ABD"
        / "ABD.pickle"
    ),

    "mros1_val": (
        DATA_ROOT
        / "mros1"
        / "mmap_signals_MAE_processed"
        / "val"
        / "ABD"
        / "ABD.pickle"
    ),

    "mros1_test": (
        DATA_ROOT
        / "mros1"
        / "mmap_signals_MAE_processed"
        / "test"
        / "ABD"
        / "ABD.pickle"
    ),

    "mros2_train": (
        DATA_ROOT
        / "mros2"
        / "mmap_signals_MAE_processed"
        / "train"
        / "ABD"
        / "ABD.pickle"
    ),

    "mros2_val": (
        DATA_ROOT
        / "mros2"
        / "mmap_signals_MAE_processed"
        / "val"
        / "ABD"
        / "ABD.pickle"
    ),

    "mros2_test": (
        DATA_ROOT
        / "mros2"
        / "mmap_signals_MAE_processed"
        / "test"
        / "ABD"
        / "ABD.pickle"
    ),

    "mesa_train": (
        DATA_ROOT
        / "mesa"
        / "mmap_signals_MAE_processed"
        / "train"
        / "ABD"
        / "ABD.pickle"
    ),

    "mesa_val": (
        DATA_ROOT
        / "mesa"
        / "mmap_signals_MAE_processed"
        / "val"
        / "ABD"
        / "ABD.pickle"
    ),

    "mesa_test": (
        DATA_ROOT
        / "mesa"
        / "mmap_signals_MAE_processed"
        / "test"
        / "ABD"
        / "ABD.pickle"
    ),
}


RESP_STATS_FROM = {
    "shhs1_val": "shhs1_train",
    "kiss_val": "kiss_train",
    "mesa_val": "mesa_train",
    "mesa_test": "mesa_train",
}


def load_pickle(path):
    with path.open("rb") as f:
        return pickle.load(f)


def compute_global_stats(metadata):
    total_sum = 0.0
    total_sumsq = 0.0
    total_epochs = 0

    sig_len = int(
        metadata["data_shape"][1]
    )

    for info in metadata["sig_info"].values():
        start, end = info["pos"]

        total_sum += float(
            info["stats"]["sum"]
        )

        total_sumsq += float(
            info["stats"]["sumsq"]
        )

        total_epochs += (
            end - start
        )

    if total_epochs <= 0:
        raise RuntimeError(
            "No epochs available."
        )

    n = (
        total_epochs
        * sig_len
    )

    mean = (
        total_sum
        / n
    )

    variance = (
        total_sumsq
        / n
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
    }


def load_checkpoint(
    checkpoint_path,
    device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    model = VQGAN()

    state_dict = None

    if isinstance(checkpoint, dict):
        for key in [
            "model_state_dict",
            "state_dict",
            "model",
            "vqgan_state_dict",
        ]:
            if (
                key in checkpoint
                and isinstance(
                    checkpoint[key],
                    dict,
                )
            ):
                state_dict = checkpoint[key]
                break

    if state_dict is None:
        if isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            raise RuntimeError(
                "Unknown checkpoint format."
            )

    cleaned = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]

        cleaned[key] = value

    model.load_state_dict(
        cleaned,
        strict=True,
    )

    model.to(device)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    return model


def load_manifests(
    input_root,
    split,
    max_samples=None,
):
    manifest_path = (
        input_root
        / f"{split}_manifest.jsonl"
    )

    rows = []

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            rows.append(
                json.loads(line)
            )

            if (
                max_samples is not None
                and len(rows)
                >= max_samples
            ):
                break

    return rows


class ABDSource:
    def __init__(
        self,
        dataset_name,
    ):
        pickle_path = ABD_PATHS[
            dataset_name
        ]

        self.metadata = load_pickle(
            pickle_path
        )

        self.subjects = self.metadata[
            "sig_info"
        ]

        self.shape = tuple(
            self.metadata[
                "data_shape"
            ]
        )

        self.epoch_samples = int(
            self.shape[1]
        )

        if self.epoch_samples != 3000:
            raise RuntimeError(
                f"{dataset_name}: expected "
                f"3000 samples/epoch, got "
                f"{self.epoch_samples}"
            )

        mmap_path = (
            pickle_path.with_suffix(
                ".mmap"
            )
        )

        self.data = np.memmap(
            mmap_path,
            dtype="float32",
            mode="r",
            shape=self.shape,
        )


def build_resp_stats():
    stats = {}

    for dataset in ABD_PATHS:
        stats_dataset = RESP_STATS_FROM.get(
            dataset,
            dataset,
        )

        if stats_dataset in stats:
            stats[dataset] = stats[
                stats_dataset
            ]
            continue

        metadata = load_pickle(
            ABD_PATHS[
                stats_dataset
            ]
        )

        current_stats = (
            compute_global_stats(
                metadata
            )
        )

        stats[
            stats_dataset
        ] = current_stats

        stats[
            dataset
        ] = current_stats

    return stats


def extract_respiration(
    row,
    source,
    stats,
):
    subject_id = row[
        "subject_id"
    ]

    window_index = int(
        row["window_index"]
    )

    info = source.subjects.get(
        subject_id
    )

    if info is None:
        raise KeyError(
            f"Missing ABD subject "
            f"{subject_id}"
        )

    subject_start, subject_end = (
        info["pos"]
    )

    start = (
        subject_start
        + window_index
        * WINDOW_EPOCHS
    )

    end = (
        start
        + WINDOW_EPOCHS
    )

    if end > subject_end:
        raise RuntimeError(
            f"ABD too short: "
            f"{subject_id} "
            f"window={window_index}"
        )

    epochs = np.asarray(
        source.data[
            start:end
        ],
        dtype=np.float32,
    )

    expected_shape = (
        WINDOW_EPOCHS,
        3000,
    )

    if epochs.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected ABD shape: "
            f"{epochs.shape}"
        )

    # Multi-dataset harmonization using
    # dataset/channel global statistics.
    epochs = (
        epochs
        - stats["mean"]
    ) / stats["std"]

    waveform = (
        epochs.reshape(-1)
    )

    # 100 Hz -> 10 Hz
    waveform_10hz = resample_poly(
        waveform,
        up=1,
        down=10,
    ).astype(
        np.float32,
        copy=False,
    )

    if (
        waveform_10hz.size
        != EXPECTED_RESP_SAMPLES
    ):
        raise RuntimeError(
            "Unexpected respiration "
            f"length: "
            f"{waveform_10hz.size}, "
            f"expected "
            f"{EXPECTED_RESP_SAMPLES}"
        )

    respiration = (
        waveform_10hz.reshape(
            RESP_SEGMENTS,
            RESP_SAMPLES_PER_SEGMENT,
        )
    )

    return respiration


def load_eeg_spectrogram(
    row,
    min_db,
    max_db,
):
    path = Path(
        row["path"]
    )

    with np.load(
        path,
        allow_pickle=False,
    ) as npz:
        eeg_db = np.asarray(
            npz[
                "eeg_spectrogram_db"
            ],
            dtype=np.float32,
        )

    if eeg_db.shape != (
        EEG_FREQ_BINS,
        EEG_TIME_BINS,
    ):
        raise RuntimeError(
            f"Unexpected EEG shape "
            f"{eeg_db.shape} in {path}"
        )

    # Same normalization used by VQGAN.
    eeg = np.clip(
        eeg_db,
        min_db,
        max_db,
    )

    eeg = (
        eeg - min_db
    ) / (
        max_db - min_db
    )

    return eeg.astype(
        np.float32,
        copy=False,
    )


def extract_token_tensor(indices):
    if isinstance(indices, (list, tuple)):
        indices = indices[0]

    if not torch.is_tensor(indices):
        raise RuntimeError(
            "VQGAN indices are not "
            "a tensor."
        )

    # Possible:
    # (B, 8, 64)
    if (
        indices.ndim == 3
        and tuple(
            indices.shape[-2:]
        ) == (8, 64)
    ):
        return indices

    # Possible:
    # (B, 1, 8, 64)
    if (
        indices.ndim == 4
        and indices.shape[1] == 1
        and tuple(
            indices.shape[-2:]
        ) == (8, 64)
    ):
        return indices[:, 0]

    # Possible:
    # (B, 512)
    if (
        indices.ndim == 2
        and indices.shape[1] == 512
    ):
        return indices.reshape(
            indices.shape[0],
            8,
            64,
        )

    raise RuntimeError(
        "Unexpected VQGAN token shape: "
        f"{tuple(indices.shape)}"
    )


@torch.no_grad()
def tokenize_batch(
    model,
    eeg_batch,
    device,
):
    tensor = torch.from_numpy(
        np.stack(eeg_batch)
    )

    tensor = (
        tensor
        .unsqueeze(1)
        .to(
            device,
            non_blocking=True,
        )
    )

    output = model(
        tensor
    )

    if not isinstance(
        output,
        (tuple, list),
    ):
        raise RuntimeError(
            "Unexpected VQGAN output."
        )

    if len(output) < 2:
        raise RuntimeError(
            "VQGAN output does not "
            "contain token indices."
        )

    indices = output[1]

    indices = extract_token_tensor(
        indices
    )

    tokens = (
        indices
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.uint16,
            copy=False,
        )
    )

    return tokens


def process_split(
    split,
    rows,
    input_root,
    output_root,
    model,
    device,
    resp_stats,
    min_db,
    max_db,
    batch_size,
):
    print()
    print("=" * 72)
    print(
        f"SPLIT: {split}"
    )
    print(
        f"samples: {len(rows)}"
    )

    split_dir = (
        output_root
        / split
    )

    split_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    n = len(rows)

    respiration_mmap = np.memmap(
        split_dir
        / "respiration.mmap",
        dtype="float32",
        mode="w+",
        shape=(
            n,
            64,
            2400,
        ),
    )

    tokens_mmap = np.memmap(
        split_dir
        / "eeg_tokens.mmap",
        dtype="uint16",
        mode="w+",
        shape=(
            n,
            8,
            64,
        ),
    )

    source_cache = {}

    index_path = (
        split_dir
        / "index.jsonl"
    )

    batch_eeg = []
    batch_rows = []
    batch_indices = []

    completed = 0

    def flush_batch():
        nonlocal batch_eeg
        nonlocal batch_rows
        nonlocal batch_indices

        if not batch_eeg:
            return

        token_batch = tokenize_batch(
            model,
            batch_eeg,
            device,
        )

        for local_index, output_index in enumerate(
            batch_indices
        ):
            tokens_mmap[
                output_index
            ] = token_batch[
                local_index
            ]

        batch_eeg = []
        batch_rows = []
        batch_indices = []

    with index_path.open(
        "w",
        encoding="utf-8",
    ) as index_file:

        for i, row in enumerate(rows):
            dataset = row[
                "dataset"
            ]

            if dataset not in source_cache:
                source_cache[
                    dataset
                ] = ABDSource(
                    dataset
                )

            source = source_cache[
                dataset
            ]

            stats_dataset = (
                RESP_STATS_FROM.get(
                    dataset,
                    dataset,
                )
            )

            stats = resp_stats[
                stats_dataset
            ]

            respiration = (
                extract_respiration(
                    row,
                    source,
                    stats,
                )
            )

            respiration_mmap[
                i
            ] = respiration

            eeg = (
                load_eeg_spectrogram(
                    row,
                    min_db,
                    max_db,
                )
            )

            batch_eeg.append(
                eeg
            )

            batch_rows.append(
                row
            )

            batch_indices.append(
                i
            )

            index_file.write(
                json.dumps(
                    {
                        "index": i,
                        "dataset": (
                            dataset
                        ),
                        "subject_id": (
                            row[
                                "subject_id"
                            ]
                        ),
                        "window_index": int(
                            row[
                                "window_index"
                            ]
                        ),
                        "resp_stats_from": (
                            stats_dataset
                        ),
                    }
                )
                + "\n"
            )

            if (
                len(batch_eeg)
                >= batch_size
            ):
                flush_batch()

            completed += 1

            if (
                completed % 100
                == 0
                or completed == n
            ):
                print(
                    f"{split}: "
                    f"{completed}/{n}"
                )

        flush_batch()

    respiration_mmap.flush()
    tokens_mmap.flush()

    metadata = {
        "num_samples": n,

        "respiration": {
            "path": "respiration.mmap",
            "shape": [
                n,
                64,
                2400,
            ],
            "dtype": "float32",
            "sampling_rate_hz": 10,
        },

        "eeg_tokens": {
            "path": "eeg_tokens.mmap",
            "shape": [
                n,
                8,
                64,
            ],
            "dtype": "uint16",
        },

        "index": "index.jsonl",
    }

    with (
        split_dir
        / "metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    del respiration_mmap
    del tokens_mmap

    print(
        f"{split}: DONE"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(
            "outputs/"
            "vqgan_multidataset_preprocessed"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "outputs/"
            "paper_transformer_data"
        ),
    )

    parser.add_argument(
        "--vqgan-checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    normalization_path = (
        args.input_root
        / "normalization.json"
    )

    with normalization_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        normalization = json.load(
            f
        )

    min_db = float(
        normalization[
            "min_db"
        ]
    )

    max_db = float(
        normalization[
            "max_db"
        ]
    )

    print(
        "EEG dB bounds:",
        min_db,
        max_db,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "device:",
        device,
    )

    model = load_checkpoint(
        args.vqgan_checkpoint,
        device,
    )

    print(
        "VQGAN loaded:",
        args.vqgan_checkpoint,
    )

    resp_stats = (
        build_resp_stats()
    )

    with (
        args.output_root
        / "respiration_stats.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            resp_stats,
            f,
            indent=2,
        )

    for split in [
        "train",
        "val",
        "test",
    ]:
        rows = load_manifests(
            args.input_root,
            split,
            max_samples=(
                args.max_samples_per_split
            ),
        )

        process_split(
            split=split,
            rows=rows,
            input_root=args.input_root,
            output_root=args.output_root,
            model=model,
            device=device,
            resp_stats=resp_stats,
            min_db=min_db,
            max_db=max_db,
            batch_size=args.batch_size,
        )

    print()
    print("=" * 72)
    print("ALL DONE")


if __name__ == "__main__":
    main()
