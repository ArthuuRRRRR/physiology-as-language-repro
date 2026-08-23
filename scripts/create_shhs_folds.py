import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path


DEFAULT_SHHS_ROOT = Path(
    "/datasets/sleep/#_SHHS/polysomnography/edfs"
)

DEFAULT_OUTPUT_PATH = Path(
    "outputs/splits/shhs_patient_folds.json"
)

VALID_VISITS = ("shhs1", "shhs2")

FILENAME_PATTERN = re.compile(
    r"^(shhs[12])-(\d+)\.edf$",
    re.IGNORECASE,
)


def parse_shhs_filename(filename):
    """
    Extract visit and patient ID from an EDF filename.

    Example:
        shhs2-200077.edf
        -> visit="shhs2"
        -> patient_id="200077"
    """

    match = FILENAME_PATTERN.fullmatch(filename)

    if match is None:
        raise ValueError(
            f"Invalid SHHS filename: {filename}"
        )

    visit = match.group(1).lower()
    patient_id = match.group(2)

    return visit, patient_id


def collect_records(root):
    """
    Collect SHHS1 and SHHS2 EDF files.

    The relative path is saved so the manifest does not
    depend completely on one absolute server path.
    """

    records = []
    seen_patient_visits = set()

    for expected_visit in VALID_VISITS:
        visit_dir = root / expected_visit

        if not visit_dir.exists():
            raise FileNotFoundError(
                f"SHHS directory not found: {visit_dir}"
            )

        edf_files = sorted(
            path
            for path in visit_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() == ".edf"
        )

        for edf_path in edf_files:
            visit, patient_id = parse_shhs_filename(
                edf_path.name
            )

            if visit != expected_visit:
                raise ValueError(
                    f"Visit mismatch: {edf_path}"
                )

            patient_visit = (
                patient_id,
                visit,
            )

            if patient_visit in seen_patient_visits:
                raise ValueError(
                    "Duplicate visit found for "
                    f"patient {patient_id}: {visit}"
                )

            seen_patient_visits.add(patient_visit)

            records.append(
                {
                    "patient_id": patient_id,
                    "visit": visit,
                    "relative_path": (
                        edf_path.relative_to(root).as_posix()
                    ),
                }
            )

    return sorted(
        records,
        key=lambda record: (
            record["patient_id"],
            record["visit"],
        ),
    )


def assign_patients_to_folds(
    records,
    num_folds,
    seed,
):
    """
    Assign each patient to exactly one fold.

    The seed and shuffle method are reproduction choices:
    they are not specified in the paper.
    """

    patient_ids = sorted(
        {
            record["patient_id"]
            for record in records
        }
    )

    rng = random.Random(seed)
    rng.shuffle(patient_ids)

    return {
        patient_id: index % num_folds
        for index, patient_id in enumerate(patient_ids)
    }


def build_manifest(
    records,
    fold_by_patient,
    root,
    num_folds,
    seed,
):
    record_counts = Counter(
        record["patient_id"]
        for record in records
    )

    unexpected_counts = {
        patient_id: count
        for patient_id, count
        in record_counts.items()
        if count not in (1, 2)
    }

    if unexpected_counts:
        raise ValueError(
            "Unexpected number of visits: "
            f"{unexpected_counts}"
        )

    folds = []

    for fold_index in range(num_folds):
        patient_ids = sorted(
            patient_id
            for patient_id, assigned_fold
            in fold_by_patient.items()
            if assigned_fold == fold_index
        )

        patient_set = set(patient_ids)

        fold_records = [
            record
            for record in records
            if record["patient_id"] in patient_set
        ]

        visit_counts = Counter(
            record["visit"]
            for record in fold_records
        )

        folds.append(
            {
                "fold": fold_index,
                "num_patients": len(patient_ids),
                "num_records": len(fold_records),
                "visit_counts": dict(visit_counts),
                "patient_ids": patient_ids,
                "records": fold_records,
            }
        )

    return {
        "version": 1,
        "dataset": "SHHS",
        "edf_root": str(root),
        "num_folds": num_folds,
        "seed": seed,
        "paper_requirement": (
            "4-fold patient-wise cross-validation"
        ),
        "reproduction_choice": (
            "Patient IDs are sorted, shuffled with the "
            "saved seed, then distributed round-robin."
        ),
        "patient_id_rule": (
            "The numeric suffix shared by SHHS1 and SHHS2"
        ),
        "summary": {
            "num_patients": len(record_counts),
            "num_records": len(records),
            "single_visit_patients": sum(
                count == 1
                for count in record_counts.values()
            ),
            "two_visit_patients": sum(
                count == 2
                for count in record_counts.values()
            ),
        },
        "folds": folds,
    }


def validate_manifest(manifest):
    """
    Confirm that one patient never appears in two folds.
    """

    patient_to_fold = {}
    total_records = 0

    for fold in manifest["folds"]:
        fold_index = fold["fold"]
        total_records += len(fold["records"])

        for record in fold["records"]:
            patient_id = record["patient_id"]

            previous_fold = patient_to_fold.get(
                patient_id
            )

            if (
                previous_fold is not None
                and previous_fold != fold_index
            ):
                raise ValueError(
                    f"Patient leakage: {patient_id} "
                    f"is in folds {previous_fold} "
                    f"and {fold_index}"
                )

            patient_to_fold[patient_id] = fold_index

    expected_patients = manifest["summary"][
        "num_patients"
    ]
    expected_records = manifest["summary"][
        "num_records"
    ]

    if len(patient_to_fold) != expected_patients:
        raise ValueError(
            "Patient count mismatch after splitting"
        )

    if total_records != expected_records:
        raise ValueError(
            "Record count mismatch after splitting"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create reproducible patient-wise SHHS folds."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_SHHS_ROOT,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )

    parser.add_argument(
        "--num-folds",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if args.num_folds < 2:
        raise ValueError(
            "num_folds must be at least 2"
        )

    records = collect_records(args.root)

    fold_by_patient = assign_patients_to_folds(
        records=records,
        num_folds=args.num_folds,
        seed=args.seed,
    )

    manifest = build_manifest(
        records=records,
        fold_by_patient=fold_by_patient,
        root=args.root,
        num_folds=args.num_folds,
        seed=args.seed,
    )

    validate_manifest(manifest)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )

    print(f"Saved: {args.output}")
    print(
        f"Patients: "
        f"{manifest['summary']['num_patients']}"
    )
    print(
        f"Records: "
        f"{manifest['summary']['num_records']}"
    )
    print(
        f"Single visit: "
        f"{manifest['summary']['single_visit_patients']}"
    )
    print(
        f"Two visits: "
        f"{manifest['summary']['two_visit_patients']}"
    )

    for fold in manifest["folds"]:
        print(
            f"Fold {fold['fold']}: "
            f"{fold['num_patients']} patients | "
            f"{fold['num_records']} records | "
            f"{fold['visit_counts']}"
        )

    print("Validation passed: no patient leakage.")


if __name__ == "__main__":
    main()
