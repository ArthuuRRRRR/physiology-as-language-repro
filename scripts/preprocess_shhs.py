import argparse
import sys
from pathlib import Path

# Add project root to Python path when running this script directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.data.shhs_loader import load_shhs_pair
from src.data.preprocessing import (
    preprocess_pair,
    MODEL_WINDOW_SEC,
)

DEFAULT_SHHS_ROOT = Path(
    "/datasets/sleep/#_SHHS/polysomnography/edfs"
)

DEFAULT_OUTPUT_DIR = Path(
    "outputs/shhs_preprocessed"
)


def get_edf_files(root, visit):
    """
    Return SHHS EDF files for shhs1, shhs2, or both.
    """

    if visit == "both":
        folders = ["shhs1", "shhs2"]
    else:
        folders = [visit]

    files = []

    for folder in folders:
        visit_dir = root / folder

        if not visit_dir.exists():
            raise FileNotFoundError(
                f"SHHS directory not found: {visit_dir}"
            )

        files.extend(
            sorted(visit_dir.glob("*.edf"))
        )

    return files


def process_file(
    edf_path,
    output_dir,
    resp_channel,
    overwrite=False,
):
    """
    Preprocess one SHHS EDF file into non-overlapping
    256-minute respiration/EEG pairs.
    """

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
        f"  Duration     : {duration_sec / 3600:.2f} h"
    )
    print(
        f"  Resp channel : {data['resp_label']} "
        f"({data['fs_resp']} Hz)"
    )
    print(
        f"  EEG channel  : {data['eeg_label']} "
        f"({data['fs_eeg']} Hz)"
    )
    print(
        f"  EEG unit     : "
        f"{data['eeg_original_unit']} -> "
        f"{data['eeg_unit']}"
    )
    print(
        f"  Windows      : {n_windows}"
    )

    if n_windows == 0:
        print("  SKIP: recording shorter than one model window")
        return 0

    # Preserve shhs1/shhs2 in output filename
    visit = edf_path.parent.name
    subject_name = edf_path.stem

    saved = 0

    for window_idx in range(n_windows):

        start_sec = (
            window_idx * MODEL_WINDOW_SEC
        )

        output_name = (
            f"{visit}-{subject_name}-"
            f"window{window_idx:03d}.npz"
        )

        output_path = (
            output_dir / output_name
        )

        if output_path.exists() and not overwrite:
            print(
                f"  EXISTS window {window_idx:03d} "
                f"-> skipping"
            )
            continue

        try:
            respiration, eeg_spec, freqs = (
                preprocess_pair(
                    respiration=data["respiration"],
                    eeg=data["eeg"],
                    fs_resp=data["fs_resp"],
                    fs_eeg=data["fs_eeg"],
                    start_sec=start_sec,
                )
            )

            # Basic sanity checks
            if respiration.ndim != 2:
                raise ValueError(
                    f"Unexpected respiration shape: "
                    f"{respiration.shape}"
                )

            if eeg_spec.shape != (256, 512):
                raise ValueError(
                    f"Unexpected EEG shape: "
                    f"{eeg_spec.shape}"
                )

            if not np.isfinite(respiration).all():
                raise ValueError(
                    "Respiration contains non-finite values"
                )

            if not np.isfinite(eeg_spec).all():
                raise ValueError(
                    "EEG spectrogram contains non-finite values"
                )

            np.savez_compressed(
                output_path,
                respiration=respiration.astype(
                    np.float32
                ),
                eeg=eeg_spec.astype(
                    np.float32
                ),
                freqs=freqs.astype(
                    np.float32
                ),
                fs_resp=np.float32(
                    data["fs_resp"]
                ),
                fs_eeg=np.float32(
                    data["fs_eeg"]
                ),
                resp_channel=data[
                    "resp_label"
                ],
                eeg_channel=data[
                    "eeg_label"
                ],
                start_sec=np.float32(
                    start_sec
                ),
                source_file=edf_path.name,
                visit=visit,
            )

            print(
                f"  Saved window {window_idx:03d}"
                f" | resp={respiration.shape}"
                f" | eeg={eeg_spec.shape}"
                f" | mean={eeg_spec.mean():.3f}"
                f" | std={eeg_spec.std():.3f}"
            )

            saved += 1

        except Exception as e:
            print(
                f"  ERROR window {window_idx:03d}: {e}"
            )

    return saved


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess raw SHHS respiration + EEG "
            "for Physiology as Language reproduction."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_SHHS_ROOT,
        help="SHHS EDF root directory",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where NPZ files are saved",
    )

    parser.add_argument(
        "--visit",
        choices=[
            "shhs1",
            "shhs2",
            "both",
        ],
        default="both",
        help="Which SHHS visit to preprocess",
    )

    parser.add_argument(
        "--resp-channel",
        type=str,
        default="ABDO RES",
        help=(
            "Respiration belt channel. "
            "Default: ABDO RES. "
            "THOR RES can also be tested."
        ),
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help=(
            "Maximum number of EDF files to process. "
            "Useful for smoke tests."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing NPZ files",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = get_edf_files(
        args.root,
        args.visit,
    )

    if args.max_files is not None:
        files = files[:args.max_files]

    print(
        f"Found {len(files)} EDF file(s)"
    )
    print(
        f"Visit        : {args.visit}"
    )
    print(
        f"Resp channel : {args.resp_channel}"
    )
    print(
        f"Output       : {args.output_dir}"
    )

    total_saved = 0

    for i, edf_path in enumerate(
        files,
        start=1,
    ):
        print(
            f"\n[{i}/{len(files)}]"
        )

        try:
            total_saved += process_file(
                edf_path=edf_path,
                output_dir=args.output_dir,
                resp_channel=args.resp_channel,
                overwrite=args.overwrite,
            )

        except Exception as e:
            print(
                f"FAILED {edf_path.name}: {e}"
            )

    print("\nDone.")
    print(
        f"Saved {total_saved} window(s)"
    )


if __name__ == "__main__":
    main()

