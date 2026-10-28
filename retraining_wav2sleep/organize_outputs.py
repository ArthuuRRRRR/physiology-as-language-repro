#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "outputs"
DEST = ROOT / "by_task"

def category(p):
    name = p.name.lower()

    if p.is_file() and name.endswith(".log"):
        return "logs"

    if not p.is_dir():
        return None

    if name in ("seed_0", "seed_1", "seed_42", "shared"):
        return "pretraining"

    if name == "downstream":
        return "synthesized_eeg"

    if name == "arousal_3seeds":
        return "arousal"

    if "split" in name:
        return "splits"

    if name.startswith("cnn_tcn_"):
        for dataset in ("shhs1", "mesa", "cfs", "dreamt"):
            if dataset in name:
                return f"sleep_staging/{dataset}"
        return "sleep_staging/other"

    return None


parser = argparse.ArgumentParser()
parser.add_argument("--apply", action="store_true")
args = parser.parse_args()

if not ROOT.is_dir():
    raise FileNotFoundError(ROOT)

moves = []

for src in sorted(ROOT.iterdir()):
    if src.name == "by_task" or src.is_symlink():
        continue

    group = category(src)
    if group is None:
        print("KEEP:", src.name)
        continue

    dst = DEST / group / src.name

    if dst.exists() or dst.is_symlink():
        raise FileExistsError(f"Destination already exists: {dst}")

    moves.append((src, dst))

print(f"\nPlanned moves: {len(moves)}")

for src, dst in moves:
    print(f"{src.name}  ->  {dst.relative_to(ROOT)}")

if not args.apply:
    print("\nDRY RUN: nothing changed.")
    print("Run again with --apply to organize.")
else:
    DEST.mkdir(parents=True, exist_ok=True)
    manifest = DEST / "move_manifest.jsonl"

    for src, dst in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)

        src.rename(dst)

        # Preserve the original path for existing scripts.
        try:
            src.symlink_to(
                os.path.relpath(dst, start=src.parent),
                target_is_directory=dst.is_dir(),
            )
        except Exception:
            dst.rename(src)
            raise

        with manifest.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "original": str(src),
                "destination": str(dst),
            }) + "\n")

        print("MOVED:", src.name)

    print("\nOrganization completed.")
    print("Original paths preserved through symlinks.")
