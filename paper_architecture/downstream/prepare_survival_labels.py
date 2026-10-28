from __future__ import annotations

import csv
import json
from pathlib import Path


CLINICAL_CSV = Path(
    "/datasets/sleep/#_SHHS/datasets/"
    "shhs-cvd-summary-dataset-0.20.0.csv"
)

SPLITS = {
    "train": Path(
        "outputs/vqgan_multidataset_preprocessed_final_gpu4/"
        "train_manifest.jsonl"
    ),
    "val": Path(
        "outputs/vqgan_multidataset_preprocessed_final_gpu4/"
        "val_manifest.jsonl"
    ),
    "test": Path(
        "outputs/vqgan_shhs1_test_preprocessed_final/"
        "test_manifest.jsonl"
    ),
}

OUTPUT_ROOT = Path(
    "outputs/downstream_survival_labels"
)

OUTCOMES = {
    "angina": "ang_date",
    "cvd_death": "cvd_dthdt",
    "chf": "chf_date",
    "chd_death": "chd_dthdt",
    "mi": "mi_date",
    "stroke": "stk_date",
}

DEFAULT_CENSOR_DAYS = 365 * 6


def parse_float(value):
    if value is None:
        return None

    value = str(value).strip()

    if value == "":
        return None

    try:
        return float(value)
    except ValueError:
        return None


def load_shhs_subjects(manifest_path):
    subjects = set()

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            subject_id = str(
                row["subject_id"]
            )

            if subject_id.startswith(
                "shhs1-"
            ):
                subjects.add(subject_id)

    return sorted(subjects)


def load_clinical():
    clinical = {}

    with CLINICAL_CSV.open(
        newline="",
        encoding="utf-8-sig",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            nsrrid = str(
                row["nsrrid"]
            ).strip()

            # Same principle as the SleepMaMi loader:
            # if duplicated, keep first occurrence.
            if nsrrid not in clinical:
                clinical[nsrrid] = row

    return clinical


def build_label(
    row,
    event_col,
    time_col,
):
    event_raw = parse_float(
        row.get(event_col)
    )

    time_raw = parse_float(
        row.get(time_col)
    )

    censor = parse_float(
        row.get("censdate")
    )

    # Missing event -> 0
    event = (
        0
        if event_raw is None
        else int(event_raw >= 1)
    )

    # SleepMaMi logic:
    # event marked positive but missing/zero event date
    # becomes censored.
    if (
        event == 1
        and (
            time_raw is None
            or time_raw == 0
        )
    ):
        event = 0

    # Missing/zero event time -> censor date
    if (
        time_raw is None
        or time_raw == 0
    ):
        time = censor
    else:
        time = time_raw

    # Defensive fallback used in their code
    if (
        time is None
        or time == 0
    ):
        time = float(
            DEFAULT_CENSOR_DAYS
        )

    return event, float(time)


def main():
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    clinical = load_clinical()

    print(
        "Clinical subjects:",
        len(clinical),
    )

    print()

    for split, manifest in SPLITS.items():

        subjects = load_shhs_subjects(
            manifest
        )

        rows_out = []
        missing = []

        for subject_id in subjects:

            # shhs1-200001 -> 200001
            nsrrid = subject_id.split(
                "shhs1-",
                1,
            )[1]

            clinical_row = clinical.get(
                nsrrid
            )

            if clinical_row is None:
                missing.append(
                    subject_id
                )
                continue

            output = {
                "subject_id": subject_id,
                "nsrrid": nsrrid,
            }

            for (
                event_col,
                time_col,
            ) in OUTCOMES.items():

                event, time = build_label(
                    clinical_row,
                    event_col,
                    time_col,
                )

                output[
                    f"{event_col}_event"
                ] = event

                output[
                    f"{event_col}_time"
                ] = time

            rows_out.append(output)

        output_path = (
            OUTPUT_ROOT
            / f"{split}.csv"
        )

        fieldnames = [
            "subject_id",
            "nsrrid",
        ]

        for outcome in OUTCOMES:
            fieldnames.extend([
                f"{outcome}_event",
                f"{outcome}_time",
            ])

        with output_path.open(
            "w",
            newline="",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(
                rows_out
            )

        print(
            "=" * 72
        )
        print(
            split.upper()
        )
        print(
            "=" * 72
        )

        print(
            "SHHS1 subjects in manifest:",
            len(subjects),
        )

        print(
            "Subjects with clinical labels:",
            len(rows_out),
        )

        print(
            "Missing clinical subjects:",
            len(missing),
        )

        print()

        for outcome in OUTCOMES:

            events = sum(
                int(
                    r[
                        f"{outcome}_event"
                    ]
                )
                for r in rows_out
            )

            print(
                f"{outcome:12s}",
                f"events={events:5d}",
                f"censored={len(rows_out)-events:5d}",
            )

        print()
        print(
            "Saved:",
            output_path,
        )
        print()


if __name__ == "__main__":
    main()
