#!/usr/bin/env python3
"""Build SHHS1 Wav2Sleep CNN-TCN data for any PasL seed."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

SPLITS = ("train", "val", "test")
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "retraining_wav2sleep/outputs"
BUILDER = ROOT / "paper_architecture/downstream/build_sleep_staging_shhs1.py"
LABELS = Path("/hdd2/kdpark/sleep_datasets/shhs1/sleep_staging_labels_5sec")


def cache_ok(path, kind):
    if kind == "transformer":
        return all(
            (path / s / "index.jsonl").is_file()
            and (path / s / "metadata.json").is_file()
            and (path / s / "respiration.mmap").is_file()
            and (path / s / "eeg_tokens.mmap").is_file()
            for s in SPLITS
        )
    return (
        (path / "normalization.json").is_file()
        and all(
            (path / f"{s}_manifest.jsonl").is_file()
            for s in SPLITS
        )
    )


def discover(kind, seed):
    for folder in (BASE / f"seed_{seed}", BASE / "shared"):
        if not folder.exists():
            continue

        targets = (
            folder.rglob("metadata.json")
            if kind == "transformer"
            else folder.rglob("normalization.json")
        )

        roots = set()
        for file in targets:
            root = (
                file.parent.parent
                if kind == "transformer" and file.parent.name == "train"
                else file.parent
            )
            if cache_ok(root, kind):
                roots.add(root)

        if len(roots) == 1:
            return next(iter(roots))

        if len(roots) > 1:
            raise RuntimeError(
                f"Multiple {kind} caches under {folder}:\n"
                + "\n".join(map(str, sorted(roots)))
                + f"\nPass --{kind}-root explicitly."
            )

    raise FileNotFoundError(
        f"No {kind} cache found for seed {seed}. "
        f"Pass --{kind}-root."
    )


def pairs(index_path):
    result = []

    with index_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            subject = str(row.get("subject_id", "")).strip()
            dataset = str(row.get("dataset", "")).lower()

            if dataset.startswith("shhs1") or (
                not dataset and subject.lower().startswith("shhs1")
            ):
                result.append(
                    (subject, int(row["window_index"]))
                )

    if not result or len(result) != len(set(result)):
        raise RuntimeError(
            f"Empty or duplicate SHHS1 windows: {index_path}"
        )

    return result


def check_output(output, expected):
    import numpy as np

    for split in SPLITS:
        folder = output / "shhs1" / split

        for name in (
            "synthesized_eeg.mmap",
            "ground_truth_eeg.mmap",
            "labels.mmap",
            "index.jsonl",
            "metadata.json",
        ):
            if not (folder / name).is_file():
                raise RuntimeError(f"Missing output: {folder / name}")

        meta = json.loads(
            (folder / "metadata.json").read_text()
        )

        n = int(meta["num_samples"])
        eeg_shape = tuple(meta["eeg_shape"])
        label_shape = tuple(meta["label_shape"])

        if eeg_shape != (n, 256, 512):
            raise RuntimeError(f"Invalid EEG shape: {eeg_shape}")

        if label_shape != (n, 512):
            raise RuntimeError(f"Invalid labels shape: {label_shape}")

        rows = [
            json.loads(line)
            for line in (folder / "index.jsonl").read_text().splitlines()
            if line.strip()
        ]

        actual = [
            (str(r["subject_id"]), int(r["window_index"]))
            for r in rows
        ]

        if (
            len(actual) != n
            or set(actual) != set(expected[split])
            or n != len(expected[split])
        ):
            raise RuntimeError(
                f"{split}: output does not match Wav2Sleep index"
            )

        eeg_dtype = np.dtype(meta["eeg_dtype"])
        label_dtype = np.dtype(meta["label_dtype"])

        for name in ("synthesized_eeg.mmap", "ground_truth_eeg.mmap"):
            expected_bytes = int(np.prod(eeg_shape)) * eeg_dtype.itemsize
            if (folder / name).stat().st_size != expected_bytes:
                raise RuntimeError(f"Invalid file size: {folder / name}")

        label_file = folder / "labels.mmap"
        expected_bytes = int(np.prod(label_shape)) * label_dtype.itemsize

        if label_file.stat().st_size != expected_bytes:
            raise RuntimeError(f"Invalid label file: {label_file}")

        labels = np.memmap(
            label_file,
            dtype=label_dtype,
            mode="r",
            shape=label_shape,
        )

        valid = int(((labels >= 0) & (labels < 4)).sum())

        if valid == 0:
            raise RuntimeError(f"No valid labels in {split}")

        print(
            f"OK {split}: {n} windows, "
            f"{valid} valid staging labels"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--transformer-root", type=Path)
    p.add_argument("--preprocessed-root", type=Path)
    p.add_argument("--label-root", type=Path, default=LABELS)
    p.add_argument("--output-root", type=Path)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--check-only", action="store_true")

    a = p.parse_args()

    transformer_ckpt = (
        BASE / f"seed_{a.seed}/transformer/checkpoint_best_mae.pt"
    )
    vqgan_ckpt = (
        BASE / f"seed_{a.seed}/vqgan/checkpoint_best_final.pt"
    )

    transformer_root = (
        a.transformer_root or discover("transformer", a.seed)
    )
    preprocessed_root = (
        a.preprocessed_root or discover("preprocessed", a.seed)
    )

    out = (
        a.output_root
        or BASE / f"downstream/seed_{a.seed}"
    )

    for path in (
        BUILDER,
        transformer_ckpt,
        vqgan_ckpt,
        a.label_root,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    if not cache_ok(transformer_root, "transformer"):
        raise RuntimeError(f"Invalid Transformer cache: {transformer_root}")

    if not cache_ok(preprocessed_root, "preprocessed"):
        raise RuntimeError(f"Invalid preprocessing cache: {preprocessed_root}")

    # Wav2Sleep indices are the source of truth.
    expected = {
        s: pairs(transformer_root / s / "index.jsonl")
        for s in SPLITS
    }

    subjects = {
        s: {x.lower() for x, _ in expected[s]}
        for s in SPLITS
    }

    # Check subject-level split separation.
    for i, first in enumerate(SPLITS):
        for second in SPLITS[i + 1:]:
            if subjects[first] & subjects[second]:
                raise RuntimeError(
                    f"Subject leakage: {first} / {second}"
                )

    # Verify that the same windows exist in VQGAN preprocessing.
    for split in SPLITS:
        preprocessed = pairs(
            preprocessed_root / f"{split}_manifest.jsonl"
        )

        if set(preprocessed) != set(expected[split]):
            raise RuntimeError(
                f"{split}: VQGAN manifests and Transformer indices differ"
            )

    print("Seed:", a.seed)
    print("Transformer:", transformer_ckpt)
    print("VQGAN:", vqgan_ckpt)
    print("Transformer cache:", transformer_root)
    print("Preprocessed cache:", preprocessed_root)
    print("Output:", out)

    for split in SPLITS:
        print(
            f"{split}: {len(subjects[split])} subjects, "
            f"{len(expected[split])} windows"
        )

    if a.check_only:
        print("CHECK OK — no files generated")
        return

    # Never overwrite previous experiments.
    if (out / "shhs1").exists():
        raise FileExistsError(
            f"Output exists: {out / 'shhs1'}. "
            "Use a different --output-root."
        )

    ids_dir = out / "split_ids"
    ids_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        (ids_dir / f"{split}.txt").write_text(
            "".join(
                f"{subject}\n"
                for subject in sorted(subjects[split])
            )
        )

    # Reuse the existing synthesis and annotation implementation.
    cmd = [
        sys.executable,
        str(BUILDER),
        "--transformer-checkpoint", str(transformer_ckpt),
        "--vqgan-checkpoint", str(vqgan_ckpt),
        "--main-transformer-root", str(transformer_root),
        "--main-preprocessed-root", str(preprocessed_root),
        "--test-transformer-root", str(transformer_root),
        "--test-preprocessed-root", str(preprocessed_root),
        "--train-ids", str(ids_dir / "train.txt"),
        "--val-ids", str(ids_dir / "val.txt"),
        "--test-ids", str(ids_dir / "test.txt"),
        "--label-root", str(a.label_root),
        "--output-root", str(out / "shhs1"),
        "--batch-size", str(a.batch_size),
        "--num-workers", str(a.num_workers),
        "--splits", *SPLITS,
    ]

    subprocess.run(cmd, check=True)

    check_output(out, expected)

    provenance = {
        "seed": a.seed,
        "transformer_checkpoint": str(transformer_ckpt),
        "vqgan_checkpoint": str(vqgan_ckpt),
        "transformer_root": str(transformer_root),
        "preprocessed_root": str(preprocessed_root),
        "label_root": str(a.label_root),
        "split_window_counts": {
            s: len(expected[s]) for s in SPLITS
        },
    }

    (out / "provenance.json").write_text(
        json.dumps(provenance, indent=2)
    )

    print("READY FOR CNN-TCN:", out)


if __name__ == "__main__":
    main()
