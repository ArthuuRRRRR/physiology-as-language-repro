import sys
import argparse
from pathlib import Path

from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import paper_architecture.evaluate_paper_masked_transformer as ev


# ============================================================
# ID utilities
# ============================================================

def normalize_id(x):
    x = str(x).strip().lower()

    for prefix in (
        "shhs1-",
        "shhs1_val-",
        "shhs1_test-",
        "shhs2-",
        "mesa-",
    ):
        if x.startswith(prefix):
            x = x[len(prefix):]

    return x


def correct_dataset(row, target_dataset):
    name = str(row["dataset"]).lower()

    if target_dataset == "shhs1":
        return name.startswith("shhs1")

    if target_dataset == "mesa":
        return name.startswith("mesa")

    return False


# ============================================================
# Additional arguments
# ============================================================

parser = argparse.ArgumentParser(add_help=False)

parser.add_argument(
    "--subject-list",
    type=Path,
    required=True,
)

parser.add_argument(
    "--intersection-dataset",
    choices=["shhs1", "mesa"],
    required=True,
)

custom_args, remaining = parser.parse_known_args()

# Give remaining arguments back to original evaluator
sys.argv = [sys.argv[0]] + remaining


# ============================================================
# Load requested held-out IDs
# ============================================================

with custom_args.subject_list.open(
    "r",
    encoding="utf-8",
) as f:

    wanted_ids = {
        normalize_id(line)
        for line in f
        if line.strip()
        and not line.strip().startswith("#")
    }


print()
print("=" * 72)
print("INTERSECTION CONFIG")
print(
    "Dataset            :",
    custom_args.intersection_dataset,
)
print(
    "Subjects requested :",
    len(wanted_ids),
)
print("=" * 72)
print()


# Keep original dataset class before monkey patching
OriginalDataset = ev.PaperTransformerEvaluationDataset


# ============================================================
# Combined intersection dataset
# ============================================================

class IntersectionDataset(Dataset):

    def __init__(
        self,
        transformer_data_root,
        preprocessed_root,
        split,
    ):
        super().__init__()

        # ----------------------------------------------------
        # Main test dataset
        # ----------------------------------------------------

        self.main_dataset = OriginalDataset(
            transformer_data_root=transformer_data_root,
            preprocessed_root=preprocessed_root,
            split=split,
        )

        self.selected = []
        found_subjects = set()

        # ----------------------------------------------------
        # Select intersection subjects already in main test
        # ----------------------------------------------------

        for i, row in enumerate(
            self.main_dataset.index_rows
        ):

            if not correct_dataset(
                row,
                custom_args.intersection_dataset,
            ):
                continue

            sid = normalize_id(
                row["subject_id"]
            )

            if sid in wanted_ids:
                self.selected.append(
                    ("main", i)
                )
                found_subjects.add(
                    sid
                )

        main_windows = len(
            self.selected
        )

        main_subjects = len(
            found_subjects
        )

        # ----------------------------------------------------
        # Missing subjects are recovered from FINAL validation
        # ----------------------------------------------------

        missing_after_main = (
            wanted_ids
            - found_subjects
        )

        self.val_dataset = OriginalDataset(
            transformer_data_root=(
                "outputs/"
                "paper_transformer_data_final"
            ),
            preprocessed_root=(
                "outputs/"
                "vqgan_multidataset_preprocessed_final_gpu4"
            ),
            split="val",
        )

        val_windows = 0
        val_subjects = set()

        for i, row in enumerate(
            self.val_dataset.index_rows
        ):

            if not correct_dataset(
                row,
                custom_args.intersection_dataset,
            ):
                continue

            sid = normalize_id(
                row["subject_id"]
            )

            if sid in missing_after_main:
                self.selected.append(
                    ("val", i)
                )

                found_subjects.add(
                    sid
                )

                val_subjects.add(
                    sid
                )

                val_windows += 1

        # ----------------------------------------------------
        # Final checks
        # ----------------------------------------------------

        missing_final = (
            wanted_ids
            - found_subjects
        )

        print()
        print("=" * 72)
        print("INTERSECTION FILTER")
        print(
            "Dataset            :",
            custom_args.intersection_dataset,
        )
        print(
            "Subjects requested :",
            len(wanted_ids),
        )
        print(
            "Subjects found     :",
            len(found_subjects),
        )
        print(
            "Main subjects      :",
            main_subjects,
        )
        print(
            "Val subjects added :",
            len(val_subjects),
        )
        print(
            "Main windows       :",
            main_windows,
        )
        print(
            "Val windows added  :",
            val_windows,
        )
        print(
            "Total windows      :",
            len(self.selected),
        )
        print(
            "Missing subjects   :",
            len(missing_final),
        )

        if missing_final:
            print(
                "Missing IDs        :",
                sorted(
                    missing_final
                ),
            )

        print("=" * 72)
        print()

        if not self.selected:
            raise RuntimeError(
                "No intersection samples found."
            )

        if missing_final:
            raise RuntimeError(
                "Some requested held-out subjects "
                "could not be found."
            )

    def __len__(self):
        return len(
            self.selected
        )

    def __getitem__(
        self,
        index,
    ):
        source, original_index = (
            self.selected[index]
        )

        if source == "main":
            return self.main_dataset[
                original_index
            ]

        return self.val_dataset[
            original_index
        ]


# ============================================================
# Replace evaluator dataset
# ============================================================

ev.PaperTransformerEvaluationDataset = (
    IntersectionDataset
)


if "--shuffle-respiration" in sys.argv:
    raise ValueError(
        "Intersection evaluator does not "
        "support --shuffle-respiration."
    )


# ============================================================
# Run original evaluator
# ============================================================

ev.main()
