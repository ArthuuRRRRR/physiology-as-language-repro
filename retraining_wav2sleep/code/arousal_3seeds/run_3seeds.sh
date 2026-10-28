#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

DATA_BASE="retraining_wav2sleep/outputs/downstream"

# Check that synthesized EEG data exist for every seed.
for seed in 0 1 42; do
    for split in train val test; do
        folder="$DATA_BASE/seed_${seed}/shhs1/$split"
        for file in metadata.json index.jsonl synthesized_eeg.mmap labels.mmap; do
            if [[ ! -f "$folder/$file" ]]; then
                echo "MISSING: $folder/$file"
                exit 1
            fi
        done
    done
done

for seed in 0 1 42; do
    echo "========================================"
    echo "AROUSAL DETECTION - PASL SEED $seed"
    echo "========================================"

    python retraining_wav2sleep/code/arousal_3seeds/train_arousal_linear.py \
        --pasl-seed "$seed" \
        --seed 42 \
        --probe-train-subjects 300 \
        --probe-val-subjects 150 \
        --test-subjects 0 \
        --epochs 20 \
        "$@"
done
