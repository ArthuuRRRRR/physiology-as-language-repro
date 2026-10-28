#!/usr/bin/env bash
set -euo pipefail

cd ~/physiology_as_language_repro

SCRIPT="retraining_wav2sleep/code/arousal_3seeds/train_arousal_linear.py"
BASE="retraining_wav2sleep/outputs"

# Check data and annotations before training
python - <<'PY'
import json
from pathlib import Path

base = Path("retraining_wav2sleep/outputs")

for dataset in ("mesa", "cfs"):
    label_root = Path(
        f"/hdd2/kdpark/sleep_datasets/{dataset}/arousal_epoch_labels"
    )
    if not label_root.is_dir():
        raise FileNotFoundError(label_root)

    print(f"\n===== {dataset.upper()} =====")

    for split in ("train", "val", "test"):
        reference = None

        for seed in (0, 1, 42):
            folder = base / "downstream" / f"seed_{seed}" / dataset / split

            for filename in (
                "metadata.json",
                "index.jsonl",
                "synthesized_eeg.mmap",
                "labels.mmap",
            ):
                if not (folder / filename).is_file():
                    raise FileNotFoundError(folder / filename)

            rows = [
                json.loads(line)
                for line in (folder / "index.jsonl").read_text().splitlines()
                if line.strip()
            ]

            keys = sorted(
                (str(r["subject_id"]), int(r["window_index"]))
                for r in rows
            )

            if reference is None:
                reference = keys
            elif keys != reference:
                raise RuntimeError(
                    f"{dataset}/{split}: different windows across seeds"
                )

            subjects = {sid for sid, _ in keys}
            matched = sum(
                (label_root / f"{sid}_arousal_epoch.npy").is_file()
                for sid in subjects
            )

            print(
                f"seed={seed:2d} {split:5s}: "
                f"{len(rows)} windows, "
                f"{matched}/{len(subjects)} subjects with arousal labels"
            )

            if matched == 0:
                raise RuntimeError(
                    f"No matching arousal labels: {dataset}/{split}/seed_{seed}"
                )

print("\nPREFLIGHT OK")
PY

for dataset in mesa cfs; do
    for seed in 0 1 42; do

        DATA="$BASE/downstream/seed_${seed}/${dataset}"
        LABELS="/hdd2/kdpark/sleep_datasets/${dataset}/arousal_epoch_labels"
        OUTPUT="$BASE/arousal_3seeds/seed_${seed}/${dataset}"

        mkdir -p "$OUTPUT"

        echo "================================================="
        echo "AROUSAL - $(echo "$dataset" | tr 'a-z' 'A-Z') - PASL SEED $seed"
        echo "================================================="

        python "$SCRIPT" \
            --pasl-seed "$seed" \
            --data-root "$DATA" \
            --arousal-root "$LABELS" \
            --output-dir "$OUTPUT" \
            --seed 42 \
            --probe-train-subjects 300 \
            --probe-val-subjects 150 \
            --test-subjects 0 \
            --epochs 20 \
            --gpu 0 \
            2>&1 | tee "$OUTPUT/train.log"

    done
done

echo "ALL AROUSAL EXPERIMENTS COMPLETED"
