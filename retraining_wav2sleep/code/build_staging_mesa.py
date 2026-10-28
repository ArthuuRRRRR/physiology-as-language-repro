#!/usr/bin/env python3
"""Generate MESA Wav2Sleep staging data for any PasL seed."""

import argparse
import importlib.util
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "retraining_wav2sleep/outputs"
BUILDER = ROOT / "paper_architecture/downstream/build_sleep_staging_shhs1.py"
LABEL_ROOT = Path("/hdd2/kdpark/sleep_datasets/mesa/annotations")
SPLITS = ("train", "val", "test")
STAGE_MAP = np.array([0, 1, 1, 2, 3], dtype=np.int8)


def norm(value):
    value = str(value).strip().lower()
    for prefix in (
        "mesa_train__", "mesa_val__", "mesa_test__",
        "mesa_train-", "mesa_val-", "mesa_test-",
        "mesa-sleep-", "mesa-",
    ):
        if value.startswith(prefix):
            value = value[len(prefix):]
    return value


def is_mesa(row):
    dataset = str(row.get("dataset", "")).lower()
    subject = str(row.get("subject_id", "")).lower()
    return dataset.startswith("mesa") or (
        not dataset and subject.startswith("mesa")
    )


def pairs(path):
    result = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if is_mesa(row):
                result.append((
                    norm(row["subject_id"]),
                    int(row["window_index"]),
                ))

    if not result or len(result) != len(set(result)):
        raise RuntimeError(f"Invalid MESA windows: {path}")
    return result


def cache_ok(path, kind):
    if kind == "transformer":
        return all(
            all((path / split / name).is_file() for name in (
                "index.jsonl", "metadata.json",
                "respiration.mmap", "eeg_tokens.mmap",
            ))
            for split in SPLITS
        )

    return (
        (path / "normalization.json").is_file()
        and all(
            (path / f"{split}_manifest.jsonl").is_file()
            for split in SPLITS
        )
    )


def discover(kind, seed):
    for folder in (BASE / f"seed_{seed}", BASE / "shared"):
        if not folder.exists():
            continue

        filename = (
            "metadata.json" if kind == "transformer"
            else "normalization.json"
        )

        roots = set()

        for file in folder.rglob(filename):
            path = (
                file.parent.parent
                if kind == "transformer"
                and file.parent.name == "train"
                else file.parent
            )
            if cache_ok(path, kind):
                roots.add(path)

        if len(roots) == 1:
            return next(iter(roots))

        if len(roots) > 1:
            raise RuntimeError(
                f"Multiple {kind} caches: {sorted(map(str, roots))}. "
                f"Specify --{kind}-root."
            )

    raise FileNotFoundError(
        f"No {kind} cache for seed {seed}"
    )


def annotation_path(root, subject):
    sid = norm(subject)

    candidates = [
        root / f"mesa-sleep-{sid}_ann.npy",
        root / f"{sid}_ann.npy",
        root / f"mesa-{sid}_ann.npy",
    ]

    matches = [p for p in candidates if p.is_file()]

    if len(matches) != 1:
        raise FileNotFoundError(
            f"Cannot identify MESA annotation for {subject}: "
            f"{candidates}"
        )

    return matches[0]


@lru_cache(maxsize=8192)
def mesa_labels(label_root_string, subject):
    path = annotation_path(
        Path(label_root_string), subject
    )

    labels = np.load(
        path, allow_pickle=False
    ).astype(np.int64, copy=False).reshape(-1)

    if (
        labels.size == 0
        or labels.min() < 0
        or labels.max() > 4
    ):
        raise RuntimeError(
            f"Invalid MESA annotations: {path}"
        )

    return STAGE_MAP[labels]


def check_output(destination, expected):
    for split in SPLITS:
        folder = destination / split

        required = (
            "synthesized_eeg.mmap",
            "ground_truth_eeg.mmap",
            "labels.mmap",
            "index.jsonl",
            "metadata.json",
        )

        for name in required:
            if not (folder / name).is_file():
                raise FileNotFoundError(folder / name)

        meta_file = folder / "metadata.json"
        meta = json.loads(meta_file.read_text())

        n = int(meta["num_samples"])
        eeg_shape = tuple(meta["eeg_shape"])
        label_shape = tuple(meta["label_shape"])

        if (
            n != len(expected[split])
            or eeg_shape != (n, 256, 512)
            or label_shape != (n, 512)
        ):
            raise RuntimeError(
                f"Invalid output dimensions: {split}"
            )

        rows = [
            json.loads(line)
            for line in (folder / "index.jsonl")
                .read_text().splitlines()
            if line.strip()
        ]

        actual = [
            (norm(r["subject_id"]), int(r["window_index"]))
            for r in rows
        ]

        if (
            len(actual) != n
            or set(actual) != set(expected[split])
        ):
            raise RuntimeError(
                f"MESA index mismatch: {split}"
            )

        for name in (
            "synthesized_eeg.mmap",
            "ground_truth_eeg.mmap",
        ):
            expected_bytes = (
                int(np.prod(eeg_shape))
                * np.dtype(meta["eeg_dtype"]).itemsize
            )
            if (folder / name).stat().st_size != expected_bytes:
                raise RuntimeError(
                    f"Invalid EEG file size: {folder / name}"
                )

        label_file = folder / "labels.mmap"
        label_dtype = np.dtype(meta["label_dtype"])
        expected_bytes = (
            int(np.prod(label_shape))
            * label_dtype.itemsize
        )
        if label_file.stat().st_size != expected_bytes:
            raise RuntimeError(
                f"Invalid label file size: {label_file}"
            )

        labels = np.memmap(
            label_file,
            dtype=label_dtype,
            mode="r",
            shape=label_shape,
        )

        valid = int(
            ((labels >= 0) & (labels < 4)).sum()
        )

        if valid == 0:
            raise RuntimeError(
                f"No valid labels: {split}"
            )

        meta.update(
            dataset="mesa",
            label_original_resolution_sec=30,
            label_downstream_resolution_sec=30,
        )
        meta_file.write_text(
            json.dumps(meta, indent=2)
        )

        print(
            f"OK MESA {split}: {n} windows, "
            f"{valid} valid labels"
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--transformer-root", type=Path)
    parser.add_argument("--preprocessed-root", type=Path)
    parser.add_argument("--label-root", type=Path, default=LABEL_ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    transformer_ckpt = (
        BASE / f"seed_{args.seed}/transformer/checkpoint_best_mae.pt"
    )
    vqgan_ckpt = (
        BASE / f"seed_{args.seed}/vqgan/checkpoint_best_final.pt"
    )

    troot = args.transformer_root or discover(
        "transformer", args.seed
    )
    proot = args.preprocessed_root or discover(
        "preprocessed", args.seed
    )
    out = (
        args.output_root
        or BASE / f"downstream/seed_{args.seed}"
    )
    destination = out / "mesa"

    for path in (
        BUILDER, transformer_ckpt,
        vqgan_ckpt, args.label_root,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    if not cache_ok(troot, "transformer"):
        raise RuntimeError("Invalid Transformer cache")
    if not cache_ok(proot, "preprocessed"):
        raise RuntimeError("Invalid preprocessing cache")

    expected = {
        s: pairs(troot / s / "index.jsonl")
        for s in SPLITS
    }
    subjects = {
        s: {subject for subject, _ in expected[s]}
        for s in SPLITS
    }

    for i, split in enumerate(SPLITS):
        for other in SPLITS[i + 1:]:
            if subjects[split] & subjects[other]:
                raise RuntimeError(
                    f"Subject leakage: {split}/{other}"
                )

        available = pairs(
            proot / f"{split}_manifest.jsonl"
        )
        if set(available) != set(expected[split]):
            raise RuntimeError(
                f"Transformer/VQGAN mismatch: {split}"
            )

        for subject in subjects[split]:
            annotation_path(args.label_root, subject)

    print(f"Seed: {args.seed}")
    print(f"Transformer: {transformer_ckpt}")
    print(f"VQGAN: {vqgan_ckpt}")
    print(f"Transformer cache: {troot}")
    print(f"Preprocessed cache: {proot}")
    print(f"Output: {destination}")

    for split in SPLITS:
        print(
            f"{split}: {len(subjects[split])} subjects, "
            f"{len(expected[split])} windows"
        )

    if args.check_only:
        print("CHECK OK - no files generated")
        return

    if destination.exists():
        raise FileExistsError(
            f"Output already exists: {destination}"
        )

    ids_dir = out / "split_ids_mesa"
    ids_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        (ids_dir / f"{split}.txt").write_text(
            "".join(
                f"mesa-sleep-{sid}\n"
                for sid in sorted(subjects[split])
            )
        )

    # Import original synthesis pipeline without modifying it.
    for path in (BUILDER.parent, ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    spec = importlib.util.spec_from_file_location(
        "mesa_synthesis_builder", BUILDER
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    # Change only the dataset selection and labels.
    module.normalize_id = norm
    module.canonical_shhs1_id = (
        lambda sid: "mesa-sleep-" + norm(sid)
    )
    module.is_shhs1 = is_mesa
    module.load_shhs1_labels_30sec_cached = mesa_labels

    cmd = [
        str(BUILDER),
        "--transformer-checkpoint", str(transformer_ckpt),
        "--vqgan-checkpoint", str(vqgan_ckpt),
        "--main-transformer-root", str(troot),
        "--main-preprocessed-root", str(proot),
        "--test-transformer-root", str(troot),
        "--test-preprocessed-root", str(proot),
        "--train-ids", str(ids_dir / "train.txt"),
        "--val-ids", str(ids_dir / "val.txt"),
        "--test-ids", str(ids_dir / "test.txt"),
        "--label-root", str(args.label_root),
        "--output-root", str(destination),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--splits", *SPLITS,
    ]

    old_argv = sys.argv
    try:
        sys.argv = cmd
        module.main()
    finally:
        sys.argv = old_argv

    check_output(destination, expected)

    provenance = {
        "seed": args.seed,
        "dataset": "mesa",
        "transformer_checkpoint": str(transformer_ckpt),
        "vqgan_checkpoint": str(vqgan_ckpt),
        "transformer_root": str(troot),
        "preprocessed_root": str(proot),
        "label_root": str(args.label_root),
        "split_window_counts": {
            s: len(expected[s]) for s in SPLITS
        },
    }

    (destination / "provenance.json").write_text(
        json.dumps(provenance, indent=2)
    )

    print("READY FOR CNN-TCN:", destination)


if __name__ == "__main__":
    main()
