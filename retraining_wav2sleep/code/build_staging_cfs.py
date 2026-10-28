#!/usr/bin/env python3
"""Build CFS CNN-TCN data using existing PasL caches."""

import argparse
import importlib.util
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "retraining_wav2sleep/outputs"
ENGINE = ROOT / "paper_architecture/downstream/build_sleep_staging_shhs1.py"
SPLITS_DIR = BASE / "cfs_wav2sleep_splits"
LABEL_ROOT = Path(
    "/hdd2/kdpark/sleep_datasets/CFS_raw/"
    "polysomnography/annotations-events-nsrr"
)
SPLITS = ("train", "val", "test")


def norm(value):
    s = str(value).strip().lower()
    for prefix in (
        "cfs-visit5-",
        "cfs_train__",
        "cfs_val__",
        "cfs_test__",
    ):
        if s.startswith(prefix):
            return s[len(prefix):]
    return s


def key(row):
    return norm(row["subject_id"]), int(row["window_index"])


def read_jsonl(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def unique_keys(rows, source):
    keys = [key(r) for r in rows]
    if not keys or len(keys) != len(set(keys)):
        raise RuntimeError(f"Empty or duplicate windows: {source}")
    return set(keys)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--check-only", action="store_true")
    a = p.parse_args()

    ckpt_t = BASE / f"seed_{a.seed}/transformer/checkpoint_best_mae.pt"
    ckpt_v = BASE / f"seed_{a.seed}/vqgan/checkpoint_best_final.pt"

    cache = BASE / f"seed_{a.seed}/cfs_full"
    tr = cache / "transformer_data"
    pp = cache / "preprocessed"
    out = BASE / f"downstream/seed_{a.seed}/cfs"

    for path in (
        ENGINE, ckpt_t, ckpt_v, LABEL_ROOT,
        tr / "all/index.jsonl",
        tr / "all/metadata.json",
        tr / "all/respiration.mmap",
        tr / "all/eeg_tokens.mmap",
        pp / "all_manifest.jsonl",
        pp / "normalization.json",
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    expected = {}
    subjects = {}

    for split in SPLITS:
        rows = read_jsonl(
            SPLITS_DIR / f"{split}_manifest.jsonl"
        )
        expected[split] = unique_keys(rows, split)
        subjects[split] = {
            subject for subject, _ in expected[split]
        }

        for subject in subjects[split]:
            xml = LABEL_ROOT / f"cfs-visit5-{subject}-nsrr.xml"
            if not xml.is_file():
                raise FileNotFoundError(xml)

    for i, split in enumerate(SPLITS):
        for other in SPLITS[i + 1:]:
            if subjects[split] & subjects[other]:
                raise RuntimeError(
                    f"Subject leakage: {split}/{other}"
                )

    index = read_jsonl(tr / "all/index.jsonl")
    manifest = read_jsonl(pp / "all_manifest.jsonl")

    index_keys = unique_keys(index, "Transformer")
    prep_keys = unique_keys(manifest, "Preprocessed")

    if index_keys != prep_keys:
        raise RuntimeError(
            "Transformer / VQGAN cache mismatch"
        )

    for split in SPLITS:
        missing = expected[split] - index_keys
        if missing:
            raise RuntimeError(
                f"{split}: {len(missing)} windows missing "
                f"from CFS cache: {sorted(missing)[:5]}"
            )

    print("Seed:", a.seed)
    print("Transformer:", ckpt_t)
    print("VQGAN:", ckpt_v)
    print("CFS cache:", cache)
    print("Output:", out)
    print("Cached windows:", len(index_keys))

    for split in SPLITS:
        print(
            f"{split}: {len(subjects[split])} subjects, "
            f"{len(expected[split])} windows"
        )

    if a.check_only:
        print("CHECK OK - no files generated")
        return

    if out.exists():
        raise FileExistsError(
            f"Output already exists: {out}"
        )

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ENGINE.parent))

    from paper_architecture.downstream.ViT import (
        build_cfs_synth as cfs
    )

    cfs.CFS_XML_ROOT = LABEL_ROOT

    spec = importlib.util.spec_from_file_location(
        "cfs_staging_engine", ENGINE
    )
    engine = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = engine
    spec.loader.exec_module(engine)

    # Adapt SHHS1 selection to CFS.
    engine.normalize_id = norm
    engine.canonical_shhs1_id = (
        lambda sid: "cfs-visit5-" + norm(sid)
    )
    engine.is_shhs1 = lambda row: str(
        row.get("dataset", "")
    ).lower().startswith("cfs")

    # Use the existing NSRR XML annotation loader.
    @lru_cache(maxsize=8192)
    def load_labels(label_root_string, subject_id):
        labels = np.asarray(
            cfs.load_cfs_labels(subject_id),
            dtype=np.int8,
        ).reshape(-1)

        if labels.size == 0:
            raise RuntimeError(
                f"Empty annotations: {subject_id}"
            )

        return labels

    engine.load_shhs1_labels_30sec_cached = load_labels

    # Read the original 'all' cache; select exact split windows.
    def selected_init(
        self, *, wanted_ids, transformer_data_root,
        preprocessed_root, split, **kwargs
    ):
        self.wanted_ids = set(wanted_ids)

        self.main_dataset = (
            engine.ev.PaperTransformerEvaluationDataset(
                transformer_data_root=transformer_data_root,
                preprocessed_root=preprocessed_root,
                split="all",
            )
        )

        self.fallback_dataset = None

        target = expected[split]

        self.selected = [
            ("main", i, row)
            for i, row in enumerate(self.main_dataset.index_rows)
            if key(row) in target
        ]

        actual = {
            key(row) for _, _, row in self.selected
        }

        if actual != target or len(self.selected) != len(target):
            raise RuntimeError(
                f"{split}: selected windows mismatch"
            )

        print(
            f"CFS {split}: {len(self.selected)} windows selected"
        )

    original_getitem = (
        engine.SelectedEvaluationDataset.__getitem__
    )

    def selected_getitem(self, index):
        item = original_getitem(self, index)
        row = self.selected[index][2]

        # 512 epochs of 30 seconds per window.
        start_sec = int(row["window_index"]) * 512 * 30

        item["_builder_start_sec"] = start_sec
        item["start_sec"] = start_sec

        return item

    engine.SelectedEvaluationDataset.__init__ = selected_init
    engine.SelectedEvaluationDataset.__getitem__ = selected_getitem

    ids = BASE / f"downstream/seed_{a.seed}/split_ids_cfs"
    ids.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        (ids / f"{split}.txt").write_text(
            "".join(
                f"cfs-visit5-{sid}\n"
                for sid in sorted(subjects[split])
            )
        )

    cmd = [
        str(ENGINE),
        "--transformer-checkpoint", str(ckpt_t),
        "--vqgan-checkpoint", str(ckpt_v),
        "--main-transformer-root", str(tr),
        "--main-preprocessed-root", str(pp),
        "--test-transformer-root", str(tr),
        "--test-preprocessed-root", str(pp),
        "--train-ids", str(ids / "train.txt"),
        "--val-ids", str(ids / "val.txt"),
        "--test-ids", str(ids / "test.txt"),
        "--label-root", str(LABEL_ROOT),
        "--output-root", str(out),
        "--batch-size", str(a.batch_size),
        "--num-workers", str(a.num_workers),
        "--splits", *SPLITS,
    ]

    old_argv = sys.argv

    try:
        sys.argv = cmd
        engine.main()
    finally:
        sys.argv = old_argv

    # Verify completed outputs against official split manifests.
    for split in SPLITS:
        folder = out / split

        meta_file = folder / "metadata.json"
        meta = json.loads(meta_file.read_text())

        n = int(meta["num_samples"])

        if n != len(expected[split]):
            raise RuntimeError(
                f"{split}: invalid sample count"
            )

        rows = read_jsonl(folder / "index.jsonl")

        if unique_keys(rows, split) != expected[split]:
            raise RuntimeError(
                f"{split}: output index mismatch"
            )

        for name in (
            "synthesized_eeg.mmap",
            "ground_truth_eeg.mmap",
            "labels.mmap",
        ):
            if not (folder / name).is_file():
                raise FileNotFoundError(folder / name)

        label_shape = tuple(meta["label_shape"])
        if label_shape != (n, 512):
            raise RuntimeError(
                f"{split}: wrong label shape"
            )

        labels = np.memmap(
            folder / "labels.mmap",
            dtype=np.dtype(meta["label_dtype"]),
            mode="r",
            shape=label_shape,
        )

        valid = int(
            ((labels >= 0) & (labels <= 3)).sum()
        )

        if valid == 0:
            raise RuntimeError(
                f"{split}: no valid annotations"
            )

        meta["dataset"] = "cfs"
        meta["label_original_resolution_sec"] = 30
        meta["label_downstream_resolution_sec"] = 30

        meta_file.write_text(
            json.dumps(meta, indent=2)
        )

        print(
            f"OK CFS {split}: {n} windows, "
            f"{valid} valid labels"
        )

    provenance = {
        "dataset": "cfs",
        "seed": a.seed,
        "transformer_checkpoint": str(ckpt_t),
        "vqgan_checkpoint": str(ckpt_v),
        "transformer_root": str(tr),
        "preprocessed_root": str(pp),
        "split_source": str(SPLITS_DIR),
        "label_source": str(LABEL_ROOT),
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
