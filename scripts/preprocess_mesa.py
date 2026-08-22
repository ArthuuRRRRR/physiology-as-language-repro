import argparse
from pathlib import Path

import numpy as np

from src.data.mesa_loader import load_mesa_pair
from src.data.preprocessing import (
    MODEL_WINDOW_SEC,
    preprocess_pair,
)


DEFAULT_MESA_ROOT = Path(
    "/hdd2/kdpark/sleep_datasets/MESA_raw/polysomnography/edfs"
)


def preprocess_edf(edf_path, output_dir):
    data = load_mesa_pair(edf_path)

    duration_sec = data["duration_sec"]

    # Number of complete 256-minute windows
    n_windows = int(duration_sec // MODEL_WINDOW_SEC)

    subject_id = edf_path.stem

    print(
        f"{subject_id}: "
        f"{duration_sec / 3600:.2f} h "
        f"-> {n_windows} windows"
    )

    saved = 0

    for window_idx in range(n_windows):

        start_sec = window_idx * MODEL_WINDOW_SEC

        respiration, eeg_spectrogram, freqs = preprocess_pair(
            respiration=data["respiration"],
            eeg=data["eeg"],
            fs_resp=data["fs_resp"],
            fs_eeg=data["fs_eeg"],
            start_sec=start_sec,
        )

        output_path = (
            output_dir
            / f"{subject_id}-window{window_idx:03d}.npz"
        )

        np.savez_compressed(
            output_path,
            respiration=respiration,
            eeg_spectrogram=eeg_spectrogram,
            freqs=freqs,
        )

        print(
            f"  saved {output_path.name} "
            f"| resp={respiration.shape} "
            f"| eeg={eeg_spectrogram.shape}"
        )

        saved += 1

    return saved


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mesa-root",
        type=Path,
        default=DEFAULT_MESA_ROOT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/mesa_preprocessed"),
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit number of EDF files for development/testing.",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    edf_files = sorted(
        args.mesa_root.glob("*.edf")
    )

    if args.max_files is not None:
        edf_files = edf_files[: args.max_files]

    print("EDF files:", len(edf_files))

    total_samples = 0

    for edf_path in edf_files:
        try:
            total_samples += preprocess_edf(
                edf_path,
                args.output_dir,
            )

        except Exception as exc:
            print(
                f"ERROR {edf_path.name}: {exc}"
            )

    print()
    print("Finished.")
    print("Total samples:", total_samples)


if __name__ == "__main__":
    main()
