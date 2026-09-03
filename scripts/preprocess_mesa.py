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
    skipped = 0

    for window_idx in range(n_windows):

        start_sec = window_idx * MODEL_WINDOW_SEC

        output_path = (
            output_dir
            / f"{subject_id}-window{window_idx:03d}.npz"
        )

        # Resume mode:
        # skip already preprocessed windows
        if output_path.exists():
            print(
                f"  skip {output_path.name} "
                f"(already exists)"
            )
            skipped += 1
            continue

        respiration, eeg_spectrogram, freqs = preprocess_pair(
            respiration=data["respiration"],
            eeg=data["eeg"],
            fs_resp=data["fs_resp"],
            fs_eeg=data["fs_eeg"],
            start_sec=start_sec,
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

    return saved, skipped


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

    parser.add_argument(
        "--start-subject",
        type=str,
        default=None,
        help=(
            "Resume preprocessing from this MESA subject ID. "
            "Example: --start-subject 6672"
        ),
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    edf_files = sorted(
        args.mesa_root.glob("*.edf")
    )

    # Resume directly from a chosen subject
    if args.start_subject is not None:
        start_name = (
            f"mesa-sleep-{args.start_subject}.edf"
        )

        edf_files = [
            p for p in edf_files
            if p.name >= start_name
        ]

    if args.max_files is not None:
        edf_files = edf_files[: args.max_files]

    print("EDF files to process:", len(edf_files))

    total_saved = 0
    total_skipped = 0

    for edf_path in edf_files:
        try:
            saved, skipped = preprocess_edf(
                edf_path,
                args.output_dir,
            )

            total_saved += saved
            total_skipped += skipped

        except Exception as exc:
            print(
                f"ERROR {edf_path.name}: {exc}"
            )

    print()
    print("Finished.")
    print("New samples saved:", total_saved)
    print("Existing samples skipped:", total_skipped)
    print(
        "Total considered:",
        total_saved + total_skipped,
    )


if __name__ == "__main__":
    main()