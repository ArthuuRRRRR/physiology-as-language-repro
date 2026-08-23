import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import numpy as np

from src.data.shhs_loader import load_shhs_pair
from src.data.preprocessing import (
    MODEL_WINDOW_SEC,
    TARGET_RESP_FS,
    preprocess_pair,
)


DEFAULT_SHHS_ROOT = Path(
    "/datasets/sleep/#_SHHS/polysomnography/edfs"
)

DEFAULT_SPLIT_FILE = Path(
    "outputs/splits/shhs_patient_folds.json"
)

DEFAULT_OUTPUT_DIR = Path(
    "outputs/shhs_preprocessed"
)


def load_fold_records(
    split_file,
    root,
    fold_index,
):
    """
    Load EDF records belonging to one patient-wise fold.
    """

    if not split_file.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_file}"
        )

    with split_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    num_folds = manifest["num_folds"]

    if fold_index < 0 or fold_index >= num_folds:
        raise ValueError(
            f"fold must be between 0 and {num_folds - 1}"
        )

    fold_data = next(
        fold
        for fold in manifest["folds"]
        if fold["fold"] == fold_index
    )

    records = []

    for record in fold_data["records"]:
        records.append(
            {
                "patient_id": record["patient_id"],
                "visit": record["visit"],
                "fold": fold_index,
                "edf_path": (
                    root / record["relative_path"]
                ),
            }
        )

    return records, manifest


def process_record(
    record,
    output_dir,
    resp_channel,
    overwrite=False,
):
    """
    Preprocess one SHHS EDF into non-overlapping
    256-minute respiration/EEG pairs.
    """

    edf_path = record["edf_path"]
    patient_id = record["patient_id"]
    visit = record["visit"]
    fold_index = record["fold"]

    print(f"\nLoading: {edf_path.name}")

    data = load_shhs_pair(
        edf_path,
        resp_channel=resp_channel,
    )

    duration_sec = data["duration_sec"]

    n_windows = int(
        duration_sec // MODEL_WINDOW_SEC
    )

    print(
        f"  Patient      : {patient_id}"
    )
    print(
        f"  Visit        : {visit}"
    )
    print(
        f"  Fold         : {fold_index}"
    )
    print(
        f"  Duration     : {duration_sec / 3600:.2f} h"
    )
    print(
        f"  Resp channel : {data['resp_label']} "
        f"({data['fs_resp']} Hz -> "
        f"{TARGET_RESP_FS} Hz)"
    )
    print(
        f"  EEG channel  : {data['eeg_label']} "
        f"({data['fs_eeg']} Hz)"
    )
    print(
        f"  Windows      : {n_windows}"
    )

    if n_windows == 0:
        print(
            "  SKIP: recording shorter than one model window"
        )
        return 0

    saved = 0

    for window_idx in range(n_windows):
        start_sec = (
            window_idx * MODEL_WINDOW_SEC
        )

        output_name = (
            f"{visit}-{patient_id}-"
            f"window{window_idx:03d}.npz"
        )

        output_path = output_dir / output_name

        if output_path.exists() and not overwrite:
            print(
                f"  EXISTS window {window_idx:03d} "
                "-> skipping"
            )
            continue

        try:
            respiration, eeg_db, freqs = (
                preprocess_pair(
                    respiration=data["respiration"],
                    eeg=data["eeg"],
                    fs_resp=data["fs_resp"],
                    fs_eeg=data["fs_eeg"],
                    start_sec=start_sec,
                    output_scale="db",
                )
            )

            if respiration.shape != (64, 2400):
                raise ValueError(
                    "Unexpected respiration shape: "
                    f"{respiration.shape}"
                )

            if eeg_db.shape != (256, 512):
                raise ValueError(
                    "Unexpected EEG shape: "
                    f"{eeg_db.shape}"
                )

            if not np.isfinite(respiration).all():
                raise ValueError(
                    "Respiration contains non-finite values"
                )

            if not np.isfinite(eeg_db).all():
                raise ValueError(
                    "EEG spectrogram contains "
                    "non-finite values"
                )

            np.savez_compressed(
                output_path,
                respiration=respiration.astype(
                    np.float32
                ),
                eeg_spectrogram_db=eeg_db.astype(
                    np.float32
                ),
                freqs=freqs.astype(
                    np.float32
                ),
                fs_resp_original=np.float32(
                    data["fs_resp"]
                ),
                fs_resp_processed=np.float32(
                    TARGET_RESP_FS
                ),
                fs_eeg=np.float32(
                    data["fs_eeg"]
                ),
                resp_channel=data["resp_label"],
                eeg_channel=data["eeg_label"],
                eeg_original_unit=(
                    data["eeg_original_unit"]
                ),
                patient_id=patient_id,
                visit=visit,
                fold=np.int16(fold_index),
                start_sec=np.float32(start_sec),
                source_file=edf_path.name,
                spectrogram_scale="db",
            )

            print(
                f"  Saved window {window_idx:03d}"
                f" | resp={respiration.shape}"
                f" | eeg={eeg_db.shape}"
                f" | min={eeg_db.min():.2f} dB"
                f" | max={eeg_db.max():.2f} dB"
            )

            saved += 1

        except Exception as error:
            print(
                f"  ERROR window {window_idx:03d}: "
                f"{error}"
            )

    return saved


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess one patient-wise SHHS fold."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_SHHS_ROOT,
    )

    parser.add_argument(
        "--split-file",
        type=Path,
        default=DEFAULT_SPLIT_FILE,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        help="Fold to preprocess: 0, 1, 2, or 3",
    )

    parser.add_argument(
        "--patient-id",
        type=str,
        default=None,
        help="Process only one patient ID",
    )

    parser.add_argument(
        "--visit",
        choices=["shhs1", "shhs2"],
        default=None,
        help="Optionally restrict processing to one visit",
    )

    parser.add_argument(
        "--resp-channel",
        default="ABDO RES",
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    records, manifest = load_fold_records(
        split_file=args.split_file,
        root=args.root,
        fold_index=args.fold,
    )

    if args.patient_id is not None:
        records = [
            record
            for record in records
            if record["patient_id"] == args.patient_id
        ]

    if args.visit is not None:
        records = [
            record
            for record in records
            if record["visit"] == args.visit
        ]

    if not records:
        raise ValueError(
            "No EDF record matches the requested filters"
        )

    if args.max_files is not None:
        records = records[:args.max_files]

    fold_output_dir = (
        args.output_dir / f"fold_{args.fold}"
    )

    fold_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Split seed   : {manifest['seed']}")
    print(f"Fold         : {args.fold}")
    print(f"EDF records  : {len(records)}")
    print(f"Output       : {fold_output_dir}")
    print(f"Resp channel : {args.resp_channel}")

    total_saved = 0

    for index, record in enumerate(
        records,
        start=1,
    ):
        print(
            f"\n[{index}/{len(records)}]"
        )

        try:
            total_saved += process_record(
                record=record,
                output_dir=fold_output_dir,
                resp_channel=args.resp_channel,
                overwrite=args.overwrite,
            )

        except Exception as error:
            print(
                f"FAILED "
                f"{record['edf_path'].name}: "
                f"{error}"
            )

    print("\nDone.")
    print(
        f"Saved {total_saved} window(s)"
    )


if __name__ == "__main__":
    main()