#!/usr/bin/env bash
set -e

RUN="outputs/paper_masked_transformer_resume140_mae_every1"
LATEST="$RUN/checkpoint_latest.pt"
SNAPDIR="$RUN/snapshots"

mkdir -p "$SNAPDIR"

EPOCH=$(python - <<PY
import torch
c = torch.load(
    "$LATEST",
    map_location="cpu",
    weights_only=False
)
print(int(c["epoch"]) + 1)
PY
)

SNAP="$SNAPDIR/checkpoint_epoch${EPOCH}.pt"

echo "========================================"
echo "Saving epoch $EPOCH"
echo "From: $LATEST"
echo "To  : $SNAP"
echo "========================================"

cp --reflink=auto "$LATEST" "$SNAP"

python - <<PY
import torch
p = "$SNAP"
c = torch.load(p, map_location="cpu", weights_only=False)
print("Snapshot saved:", p)
print("Epoch:", int(c["epoch"]) + 1)
print("Validation CE:", c["validation_metrics"]["loss"])
print("Validation reconstruction MAE:",
      c.get("validation_reconstruction_mae"))
PY

echo
echo "Starting SHHS1 held-out evaluation..."

PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python -u \
paper_architecture/evaluate_paper_masked_transformer_intersection.py \
  --intersection-dataset shhs1 \
  --subject-list data_splits/intersections/shhs1_wav2sleep_heldout.txt \
  --transformer-checkpoint "$SNAP" \
  --vqgan-checkpoint outputs/vqgan_multidataset_final_ddp/checkpoint_best.pt \
  --transformer-data-root outputs/paper_transformer_data_shhs1_test_final \
  --preprocessed-root outputs/vqgan_shhs1_test_preprocessed_final \
  --split test \
  --output-dir "$RUN/evaluation_shhs1_heldout290_epoch${EPOCH}"

echo
echo "========================================"
echo "EPOCH $EPOCH RESULT"
echo "========================================"

python - <<PY
import json
p = "$RUN/evaluation_shhs1_heldout290_epoch${EPOCH}/metrics.json"
with open(p) as f:
    m = json.load(f)

for key in [
    "predicted_eeg_mae",
    "eeg_correlation",
    "temporal_correlation",
    "snr_db",
]:
    if key in m:
        print(f"{key}: {m[key]}")
PY
