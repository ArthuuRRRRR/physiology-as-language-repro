"""Calculer les statistiques d'un nouveau canal sur les données TRAIN seulement."""

import argparse
import json
from pathlib import Path

import numpy as np

from .__main__ import read_manifest
from .preprocessing import validate_recording


def estimate(rows):
    total_sum = 0.0
    total_sumsq = 0.0
    count = 0

    for row in rows:
        signal = np.load(row["source"], mmap_mode="r", allow_pickle=False)
        signal = validate_recording(signal, row["sampling_rate"])

        for start in range(0, signal.size, 1000000):
            chunk = signal[start:start + 1000000].astype(np.float64)
            total_sum += float(chunk.sum())
            total_sumsq += float(np.square(chunk).sum())
            count += chunk.size

    mean = total_sum / count
    variance = max(total_sumsq / count - mean ** 2, 1e-12)

    return {"mean": mean, "std": float(np.sqrt(variance))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--key", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)

    rows = read_manifest(args.train_manifest)
    statistics = estimate(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({args.key: statistics}, indent=2, allow_nan=False) + "\n"
    )
    print(f"Statistiques sauvegardées : {args.output}")


if __name__ == "__main__":
    main()
