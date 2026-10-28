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


CFS_ROOT = Path(
    "/hdd2/kdpark/sleep_datasets/cfs/"
    "mmap_signals_MAE_processed"
)

EEG_TRAIN = CFS_ROOT / "train/EEG/C4-M1.pickle"
EEG_TEST = CFS_ROOT / "test/EEG/C4-M1.pickle"

ABD_TRAIN = CFS_ROOT / "train/ABD/ABD.pickle"
ABD_TEST = CFS_ROOT / "test/ABD/ABD.pickle"

FINAL_NORMALIZATION = Path(
    "outputs/vqgan_multidataset_preprocessed_final_gpu4/"
    "normalization.json"
)

FINAL_VQGAN = Path(
    "outputs/vqgan_multidataset_final_ddp/checkpoint_best.pt"
)


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/cfs_external_final"),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    if args.output_root.exists():
        raise FileExistsError(
            f"{args.output_root} already exists"
        )

    # ========================================================
    # Load metadata
    # ========================================================

    eeg_train_meta = load_pickle(EEG_TRAIN)
    eeg_test_meta = load_pickle(EEG_TEST)

    abd_train_meta = load_pickle(ABD_TRAIN)
    abd_test_meta = load_pickle(ABD_TEST)

    # ========================================================
    # EEG normalization stats = CFS TRAIN
    # Same global dataset/channel normalization as final pipeline
    # ========================================================

    print("=" * 72)
    print("CFS EEG NORMALIZATION")
    print("mode: PER-RECORDING z-score")

    # ========================================================
    # Final global dB bounds = SAME AS TRAINED VQGAN
    # ========================================================

    norm = json.loads(
        FINAL_NORMALIZATION.read_text()
    )

    min_db = float(norm["min_db"])
    max_db = float(norm["max_db"])

    print()
    print("FINAL VQGAN dB bounds:")
    print("min_db:", min_db)
    print("max_db:", max_db)

    # ========================================================
    # Open TEST EEG mmap
    # ========================================================

    eeg_shape = tuple(
        eeg_test_meta["data_shape"]
    )

    eeg = np.memmap(
        EEG_TEST.with_suffix(".mmap"),
        dtype="float32",
        mode="r",
        shape=eeg_shape,
    )

    eeg_subjects = eeg_test_meta["sig_info"]
    abd_subjects = abd_test_meta["sig_info"]

    # ========================================================
    # Preprocess CFS TEST EEG
    # ========================================================

    pre_root = (
        args.output_root / "preprocessed"
    )

    out_test = (
        pre_root / "test" / "cfs_test"
    )

    out_test.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    total_valid = 0
    total_rejected = 0
    used_subjects = set()

    for subject_id, eeg_info in eeg_subjects.items():

        if subject_id not in abd_subjects:
            continue

        eeg_start, eeg_end = eeg_info["pos"]
        abd_start, abd_end = abd_subjects[
            subject_id
        ]["pos"]

        # Per-recording EEG normalization.
        # Important for CFS because recordings contain
        # heterogeneous amplitude scales.
        n_values = (
            (eeg_end - eeg_start)
            * eeg_shape[1]
        )

        rec_mean = (
            float(eeg_info["stats"]["sum"])
            / n_values
        )

        rec_var = (
            float(eeg_info["stats"]["sumsq"])
            / n_values
            - rec_mean ** 2
        )

        rec_std = float(
            np.sqrt(
                max(rec_var, 1e-12)
            )
        )

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

        for window_index in range(n_windows):

            if (
                args.max_windows is not None
                and len(rows) >= args.max_windows
            ):
                break

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

            # Final preprocessing behaviour:
            # rejected epochs -> min_db -> exactly 0 after [0,1]
            spec_db[:, ~valid_mask] = min_db

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
                out_test
                / (
                    f"cfs_test__{subject_id}__"
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
                    "cfs_test"
                ),
                split=np.asarray(
                    "test"
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
                    "dataset": "cfs_test",
                    "subject_id": str(subject_id),
                    "window_index": window_index,
                    "scale_factor": 1.0,
                    "normalization_source": (
                        "cfs_train"
                    ),
                    "valid_eeg_epochs": (
                        n_valid
                    ),
                    "rejected_eeg_epochs": (
                        n_rejected
                    ),
                }
            )

            used_subjects.add(
                str(subject_id)
            )

        if (
            args.max_windows is not None
            and len(rows) >= args.max_windows
        ):
            break

    if not rows:
        raise RuntimeError(
            "No aligned CFS windows found."
        )

    # ========================================================
    # Save manifest + SAME final normalization
    # ========================================================

    with (
        pre_root / "test_manifest.jsonl"
    ).open("w") as f:
        for row in rows:
            f.write(
                json.dumps(row) + "\n"
            )

    final_norm = dict(norm)

    final_norm[
        "external_dataset"
    ] = "cfs"

    final_norm[
        "external_eeg_stats_source"
    ] = "per_recording"

    (
        pre_root / "normalization.json"
    ).write_text(
        json.dumps(
            final_norm,
            indent=2,
        )
    )

    print()
    print("=" * 72)
    print("CFS PREPROCESSING COMPLETE")
    print("Subjects:", len(used_subjects))
    print("Windows :", len(rows))
    print("Valid EEG epochs:", total_valid)
    print(
        "Rejected EEG epochs:",
        total_rejected,
    )

    total = (
        total_valid
        + total_rejected
    )

    print(
        "Rejection rate:",
        f"{100 * total_rejected / total:.4f}%"
    )

    # ========================================================
    # Build Transformer input
    # ABD TEST normalized using CFS TRAIN statistics
    # ========================================================

    build.ABD_PATHS[
        "cfs_train"
    ] = ABD_TRAIN

    build.ABD_PATHS[
        "cfs_test"
    ] = ABD_TEST

    build.RESP_STATS_FROM[
        "cfs_test"
    ] = "cfs_train"

    abd_stats = (
        build.compute_global_stats(
            abd_train_meta
        )
    )

    respiration_stats = {
        "cfs_train": abd_stats,
        "cfs_test": abd_stats,
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
        FINAL_VQGAN,
        device,
    )

    print(
        "VQGAN:",
        FINAL_VQGAN,
    )

    build.process_split(
        split="test",
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
    print("ALL DONE")
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
