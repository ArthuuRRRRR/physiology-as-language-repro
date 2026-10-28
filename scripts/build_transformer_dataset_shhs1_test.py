import json
from pathlib import Path

import torch

import build_transformer_dataset as b


INPUT_ROOT = Path(
    "outputs/vqgan_shhs1_test_preprocessed_final"
)

OUTPUT_ROOT = Path(
    "outputs/paper_transformer_data_shhs1_test_final"
)

VQGAN_CHECKPOINT = Path(
    "outputs/vqgan_multidataset_final_ddp/checkpoint_best.pt"
)

# Add SHHS1 held-out respiration source.
b.ABD_PATHS["shhs1_test"] = (
    b.DATA_ROOT
    / "shhs1"
    / "mmap_signals_MAE_processed"
    / "test"
    / "ABD"
    / "ABD.pickle"
)

# IMPORTANT:
# SHHS1 test uses SHHS1 TRAIN respiration statistics.
b.RESP_STATS_FROM["shhs1_test"] = "shhs1_train"


OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

with (
    INPUT_ROOT / "normalization.json"
).open("r") as f:
    normalization = json.load(f)

min_db = float(normalization["min_db"])
max_db = float(normalization["max_db"])

print("EEG dB bounds:", min_db, max_db)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("device:", device)

model = b.load_checkpoint(
    VQGAN_CHECKPOINT,
    device,
)

print("VQGAN loaded:", VQGAN_CHECKPOINT)

resp_stats = b.build_resp_stats()

rows = b.load_manifests(
    INPUT_ROOT,
    "test",
)

print("SHHS1 test windows:", len(rows))

b.process_split(
    split="test",
    rows=rows,
    input_root=INPUT_ROOT,
    output_root=OUTPUT_ROOT,
    model=model,
    device=device,
    resp_stats=resp_stats,
    min_db=min_db,
    max_db=max_db,
    batch_size=30,
)

with (
    OUTPUT_ROOT / "respiration_stats.json"
).open("w") as f:
    json.dump(
        resp_stats,
        f,
        indent=2,
    )

print()
print("=" * 72)
print("SHHS1 TEST TRANSFORMER DATA DONE")
