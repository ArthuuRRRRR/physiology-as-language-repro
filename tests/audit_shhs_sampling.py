from collections import Counter
from pathlib import Path

import pyedflib


ROOT = Path(
    "/datasets/sleep/#_SHHS/polysomnography/edfs"
)

RESP_CHANNEL = "ABDO RES"
EEG_CHANNEL = "EEG"


for visit in ["shhs1", "shhs2"]:

    files = sorted(
        (ROOT / visit).glob("*.edf")
    )

    resp_fs_counts = Counter()
    eeg_fs_counts = Counter()
    pairs = Counter()

    missing_resp = 0
    missing_eeg = 0
    errors = 0

    print(f"\n=== {visit.upper()} ===")
    print("Files:", len(files))

    for i, path in enumerate(files, 1):

        try:
            edf = pyedflib.EdfReader(str(path))

            try:
                labels = edf.getSignalLabels()

                if RESP_CHANNEL not in labels:
                    missing_resp += 1
                    continue

                if EEG_CHANNEL not in labels:
                    missing_eeg += 1
                    continue

                resp_idx = labels.index(
                    RESP_CHANNEL
                )

                eeg_idx = labels.index(
                    EEG_CHANNEL
                )

                fs_resp = float(
                    edf.getSampleFrequency(
                        resp_idx
                    )
                )

                fs_eeg = float(
                    edf.getSampleFrequency(
                        eeg_idx
                    )
                )

                resp_fs_counts[fs_resp] += 1
                eeg_fs_counts[fs_eeg] += 1
                pairs[(fs_resp, fs_eeg)] += 1

            finally:
                edf.close()

        except Exception as e:
            errors += 1
            print(
                "ERROR:",
                path.name,
                e,
            )

        if i % 500 == 0:
            print(
                f"Processed {i}/{len(files)}"
            )

    print("\nRespiration sampling rates:")
    for fs, count in sorted(
        resp_fs_counts.items()
    ):
        print(
            f"  {fs:6.1f} Hz : {count}"
        )

    print("\nEEG sampling rates:")
    for fs, count in sorted(
        eeg_fs_counts.items()
    ):
        print(
            f"  {fs:6.1f} Hz : {count}"
        )

    print("\nResp/EEG combinations:")
    for pair, count in sorted(
        pairs.items()
    ):
        print(
            f"  Resp {pair[0]:6.1f} Hz"
            f" | EEG {pair[1]:6.1f} Hz"
            f" : {count}"
        )

    print("\nMissing respiration:", missing_resp)
    print("Missing EEG        :", missing_eeg)
    print("Errors             :", errors)
