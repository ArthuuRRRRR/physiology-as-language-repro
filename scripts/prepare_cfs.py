from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import preprocess_multidataset_vqgan_final as prep
from scripts import build_transformer_dataset as build


# ============================================================
# CFS sources
# ============================================================

CFS_ROOT = Path(
    "/hdd2/kdpark/sleep_datasets/cfs/"
    "mmap_signals_MAE_processed"
)

EEG_TRAIN = CFS_ROOT / "train/EEG/C4-M1.pickle"
EEG_TEST = CFS_ROOT / "test/EEG/C4-M1.pickle"

ABD_TRAIN = CFS_ROOT / "train/ABD/ABD.pickle"
ABD_TEST = CFS_ROOT / "test/ABD/ABD.pickle"
EEG_VAL = CFS_ROOT / "val/EEG/C4-M1.pickle"
ABD_VAL = CFS_ROOT / "val/ABD/ABD.pickle"


FINAL_NORMALIZATION = Path(
    "retraining_wav2sleep/outputs/shared/preprocessing/"
    "normalization.json"
)



def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )


    parser.add_argument("--seed", type=int, required=True)

    args = parser.parse_args()

    if args.output_root is None:
        args.output_root = Path(
            f"retraining_wav2sleep/outputs/seed_{args.seed}/cfs_full"
        )

    vqgan_ckpt = Path(
        f"retraining_wav2sleep/outputs/seed_{args.seed}/"
        "vqgan/checkpoint_best_final.pt"
    )

    if not vqgan_ckpt.is_file():
        raise FileNotFoundError(vqgan_ckpt)


    print("CFS FULL: all eligible subjects from train/val/test")

    if args.output_root.exists():
        raise FileExistsError(
            f"{args.output_root} already exists"
        )

    # ========================================================
    # Final VQGAN normalization bounds
    # ========================================================

    norm = json.loads(
        FINAL_NORMALIZATION.read_text()
    )

    min_db = float(norm["min_db"])
    max_db = float(norm["max_db"])

    print("=" * 72)
    print("CFS ALL — PER-RECORDING PREPROCESSING")
    print("=" * 72)
    print("min_db:", min_db)
    print("max_db:", max_db)

    # ========================================================
    # Sources
    # ========================================================

    sources = {
        "cfs_train": {
            "eeg": EEG_TRAIN,
            "abd": ABD_TRAIN,
        },
        "cfs_test": {
            "eeg": EEG_TEST,
            "abd": ABD_TEST,
        },
        "cfs_val": {
            "eeg": EEG_VAL,
            "abd": ABD_VAL,
        },
    }

    pre_root = (
        args.output_root
        / "preprocessed"
    )

    rows = []

    total_valid = 0
    total_rejected = 0
    all_subjects = set()

    # ========================================================
    # Process BOTH original CFS pools
    # ========================================================

    for dataset_name, paths in sources.items():

        print()
        print("=" * 72)
        print("SOURCE:", dataset_name)
        print("=" * 72)

        eeg_meta = load_pickle(
            paths["eeg"]
        )

        abd_meta = load_pickle(
            paths["abd"]
        )

        eeg_shape = tuple(
            eeg_meta["data_shape"]
        )

        eeg = np.memmap(
            paths["eeg"].with_suffix(".mmap"),
            dtype="float32",
            mode="r",
            shape=eeg_shape,
        )

        eeg_subjects = (
            eeg_meta["sig_info"]
        )

        abd_subjects = (
            abd_meta["sig_info"]
        )

        out_dir = (
            pre_root
            / "all"
            / dataset_name
        )

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        source_subjects = set()
        source_windows = 0

        for subject_id, eeg_info in eeg_subjects.items():

            if subject_id not in abd_subjects:
                continue

            eeg_start, eeg_end = (
                eeg_info["pos"]
            )

            abd_start, abd_end = (
                abd_subjects[
                    subject_id
                ]["pos"]
            )

            # ================================================
            # Per-recording z-score
            # ================================================

            n_values = (
                (eeg_end - eeg_start)
                * eeg_shape[1]
            )

            rec_mean = (
                float(
                    eeg_info["stats"]["sum"]
                )
                / n_values
            )

            rec_var = (
                float(
                    eeg_info["stats"]["sumsq"]
                )
                / n_values
                - rec_mean ** 2
            )

            rec_std = float(
                np.sqrt(
                    max(
                        rec_var,
                        1e-12,
                    )
                )
            )

            # ================================================
            # Complete 256-minute windows only
            # ================================================

            eeg_windows = (
                eeg_end - eeg_start
            ) // prep.WINDOW_EPOCHS

            abd_windows = (
                abd_end - abd_start
            ) // prep.WINDOW_EPOCHS

            n_windows = min(
                eeg_windows,
                abd_windows,
            )

            for window_index in range(
                n_windows
            ):

                start = (
                    eeg_start
                    + window_index
                    * prep.WINDOW_EPOCHS
                )

                end = (
                    start
                    + prep.WINDOW_EPOCHS
                )

                eeg_window = np.asarray(
                    eeg[start:end],
                    dtype=np.float32,
                )

                (
                    spec_db,
                    freqs,
                    valid_mask,
                ) = prep.make_spectrogram(
                    eeg_window,
                    rec_mean,
                    rec_std,
                )

                # Same final preprocessing behaviour
                spec_db[
                    :,
                    ~valid_mask,
                ] = min_db

                n_valid = int(
                    valid_mask.sum()
                )

                n_rejected = int(
                    valid_mask.size
                    - n_valid
                )

                total_valid += n_valid
                total_rejected += n_rejected

                path = (
                    out_dir
                    / (
                        f"{dataset_name}__"
                        f"{subject_id}__"
                        f"window{window_index:03d}.npz"
                    )
                )

                np.savez(
                    path,

                    eeg_spectrogram_db=(
                        spec_db.astype(
                            np.float32,
                            copy=False,
                        )
                    ),

                    eeg_valid_mask=(
                        valid_mask.astype(
                            np.uint8
                        )
                    ),

                    freqs=freqs,

                    dataset=np.asarray(
                        dataset_name
                    ),

                    split=np.asarray(
                        "all"
                    ),

                    subject_id=np.asarray(
                        str(subject_id)
                    ),

                    window_index=np.asarray(
                        window_index,
                        dtype=np.int32,
                    ),

                    eeg_mean=np.asarray(
                        rec_mean,
                        dtype=np.float32,
                    ),

                    eeg_std=np.asarray(
                        rec_std,
                        dtype=np.float32,
                    ),

                    scale_factor=np.asarray(
                        1.0,
                        dtype=np.float32,
                    ),

                    normalization_source=np.asarray(
                        "cfs_per_recording"
                    ),
                )

                rows.append(
                    {
                        "path": str(path),
                        "dataset": dataset_name,
                        "subject_id": str(
                            subject_id
                        ),
                        "window_index": (
                            window_index
                        ),
                        "scale_factor": 1.0,
                        "normalization_source": (
                            "cfs_per_recording"
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

            if n_windows > 0:
                source_subjects.add(
                    str(subject_id)
                )
                all_subjects.add(
                    str(subject_id)
                )

        print(
            "Subjects:",
            len(source_subjects),
        )

        print(
            "Windows :",
            source_windows,
        )

    if not rows:
        raise RuntimeError(
            "No CFS windows generated."
        )

    # ========================================================
    # Manifest
    # ========================================================

    pre_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        pre_root
        / "all_manifest.jsonl"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in rows:
            f.write(
                json.dumps(row)
                + "\n"
            )

    final_norm = dict(norm)

    final_norm[
        "external_dataset"
    ] = "cfs_full"

    final_norm[
        "external_eeg_stats_source"
    ] = "per_recording"

    (
        pre_root
        / "normalization.json"
    ).write_text(
        json.dumps(
            final_norm,
            indent=2,
        )
    )

    print()
    print("=" * 72)
    print("CFS PREPROCESSING COMPLETE")
    print("=" * 72)

    print(
        "Subjects:",
        len(all_subjects),
    )

    print(
        "Windows :",
        len(rows),
    )

    print(
        "Valid EEG epochs:",
        total_valid,
    )

    print(
        "Rejected EEG epochs:",
        total_rejected,
    )

    # ========================================================
    # Transformer data
    # ========================================================
    #
    # Keep source names separate because respiration lives
    # in two original CFS mmap pools.
    # Both use CFS train respiration statistics.
    # ========================================================

    build.ABD_PATHS[
        "cfs_train"
    ] = ABD_TRAIN

    build.ABD_PATHS[
        "cfs_test"
    ] = ABD_TEST

    build.ABD_PATHS["cfs_val"] = ABD_VAL

    build.RESP_STATS_FROM[
        "cfs_train"
    ] = "cfs_train"

    build.RESP_STATS_FROM[
        "cfs_test"
    ] = "cfs_train"

    build.RESP_STATS_FROM["cfs_val"] = "cfs_train"

    train_manifest = Path(
        "retraining_wav2sleep/outputs/cfs_wav2sleep_splits/"
        "train_manifest.jsonl"
    )
    train_ids = {
        str(json.loads(line)["subject_id"]).strip()
        for line in train_manifest.read_text().splitlines()
        if line.strip()
    }

    total_sum = total_sumsq = 0.0
    total_values = 0
    found = set()

    for physical_path in (ABD_TRAIN, ABD_VAL, ABD_TEST):
        metadata = load_pickle(physical_path)
        samples_per_epoch = int(metadata["data_shape"][1])

        for sid, info in metadata["sig_info"].items():
            sid = str(sid).strip()
            if sid not in train_ids:
                continue
            if sid in found:
                raise RuntimeError(f"Duplicate train subject: {sid}")

            found.add(sid)
            start, end = info["pos"]
            total_values += int(end - start) * samples_per_epoch
            total_sum += float(info["stats"]["sum"])
            total_sumsq += float(info["stats"]["sumsq"])

    if found != train_ids:
        raise RuntimeError(
            f"Missing train subjects: {sorted(train_ids - found)}"
        )

    mean = total_sum / total_values
    variance = max(total_sumsq / total_values - mean**2, 1e-12)
    abd_stats = {
        "mean": float(mean),
        "std": float(np.sqrt(variance)),
    }

    print("Official TRAIN ABD statistics:", len(found), "subjects")
    print(abd_stats)

    respiration_stats = {
        "cfs_train": abd_stats,
        "cfs_test": abd_stats,
        "cfs_val": abd_stats,
    }

    transformer_root = (
        args.output_root
        / "transformer_data"
    )

    transformer_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        transformer_root
        / "respiration_stats.json"
    ).write_text(
        json.dumps(
            respiration_stats,
            indent=2,
        )
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Device:", device)

    vqgan = build.load_checkpoint(
        vqgan_ckpt,
        device,
    )

    print(
        "VQGAN:",
        vqgan_ckpt,
    )

    build.process_split(
        split="all",
        rows=rows,
        input_root=pre_root,
        output_root=transformer_root,
        model=vqgan,
        device=device,
        resp_stats=respiration_stats,
        min_db=min_db,
        max_db=max_db,
        batch_size=args.batch_size,
    )

    print()
    print("=" * 72)
    print("ALL CFS DONE")
    print("=" * 72)

    print(
        "Preprocessed:",
        pre_root,
    )

    print(
        "Transformer data:",
        transformer_root,
    )


if __name__ == "__main__":
    main()
