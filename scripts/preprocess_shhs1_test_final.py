import json
import shutil
from pathlib import Path

import preprocess_multidataset_vqgan_final as pp


FINAL_ROOT = Path(
    "outputs/vqgan_multidataset_preprocessed_final_gpu4"
)

OUTPUT_ROOT = Path(
    "outputs/vqgan_shhs1_test_preprocessed_final"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

with (
    FINAL_ROOT / "normalization.json"
).open("r") as f:
    normalization = json.load(f)

min_db = float(normalization["min_db"])
max_db = float(normalization["max_db"])

print("Using FINAL TRAIN bounds:")
print("min_db:", min_db)
print("max_db:", max_db)

source = {
    "name": "shhs1_test",
    "split": "test",
    "stats_from": "shhs1_train",
    "pickle": Path(
        "/hdd2/kdpark/sleep_datasets/"
        "shhs1/mmap_signals_MAE_processed/"
        "test/EEG/C4-A1.pickle"
    ),
}

manifests = {
    "train": [],
    "val": [],
    "test": [],
}

stats, count = pp.process_source(
    source=source,
    output_root=OUTPUT_ROOT,
    db_collector=None,
    manifests=manifests,
    max_windows_per_source=None,
    overwrite=False,
)

# Invalid EEG epochs must map to the SAME final TRAIN min_db.
pp.finalize_invalid_epochs(
    manifests=manifests,
    min_db=min_db,
)

# Use exactly the normalization learned from the final TRAIN pool.
shutil.copy2(
    FINAL_ROOT / "normalization.json",
    OUTPUT_ROOT / "normalization.json",
)

with (
    OUTPUT_ROOT / "test_manifest.jsonl"
).open("w") as f:
    for row in manifests["test"]:
        f.write(
            json.dumps(row) + "\n"
        )

print()
print("=" * 72)
print("SHHS1 TEST DONE")
print("Raw recordings: 1617")
print("Generated 256-min windows:", count)
print(
    "Manifest:",
    OUTPUT_ROOT / "test_manifest.jsonl",
)
