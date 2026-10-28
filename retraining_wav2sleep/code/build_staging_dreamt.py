#!/usr/bin/env python3
"""Generate DREAMT 4-class CNN-TCN data from existing PaSL caches."""
import argparse
import importlib.util
import json
import pickle
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "retraining_wav2sleep/outputs"
ENGINE = ROOT / "paper_architecture/downstream/build_sleep_staging_shhs1.py"
DREAMT = Path("/hdd2/kdpark/sleep_datasets/dreamt/preprocessing_output")
PROCESSED = DREAMT / "processed"
SPLITS = ("train", "val", "test")
STAGE_MAP = np.array([0, 1, 1, 2, 3], dtype=np.int16)


def read_jsonl(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def subject_id(x):
    return str(x).strip().upper()


def key(row):
    return subject_id(row["subject_id"]), int(row["window_index"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--allow-partial-coverage", action="store_true",
                   help="Pilot only: use truncated first-window DREAMT caches")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    a = p.parse_args()

    seed_root = BASE / f"seed_{a.seed}"
    ckpt_t = seed_root / "transformer/checkpoint_best_mae.pt"
    ckpt_v = seed_root / "vqgan/checkpoint_best_final.pt"
    cache = seed_root / "dreamt_full"
    tr = cache / "transformer_data"
    pp = cache / "preprocessed"
    out = BASE / f"downstream/seed_{a.seed}/dreamt"

    for f in [ENGINE, ckpt_t, ckpt_v, tr / "all/index.jsonl",
              tr / "all/metadata.json", tr / "all/respiration.mmap",
              tr / "all/eeg_tokens.mmap", pp / "all_manifest.jsonl",
              pp / "normalization.json", DREAMT / "dreamt_splits.json"]:
        if not f.is_file():
            raise FileNotFoundError(f)

    split_ids = json.loads((DREAMT / "dreamt_splits.json").read_text())
    subjects = {s: {subject_id(x) for x in split_ids[s]} for s in SPLITS}
    for s in SPLITS:
        if not subjects[s]:
            raise RuntimeError(f"Empty {s} split")
    for i, s in enumerate(SPLITS):
        for t in SPLITS[i + 1:]:
            if subjects[s] & subjects[t]:
                raise RuntimeError(f"Subject leakage: {s}/{t}")
    subject_split = {sid: s for s in SPLITS for sid in subjects[s]}

    indices = read_jsonl(tr / "all/index.jsonl")
    manifests = read_jsonl(pp / "all_manifest.jsonl")
    index_keys = [key(r) for r in indices]
    manifest_keys = [key(r) for r in manifests]
    if len(index_keys) != len(set(index_keys)) or set(index_keys) != set(manifest_keys):
        raise RuntimeError("Transformer/preprocessed duplicate or mismatched windows")
    expected = {}
    for s in SPLITS:
        rows = [r for r in indices if r.get("dataset") == f"dreamt_{s}"]
        expected[s] = {key(r) for r in rows}
        if not expected[s] or {sid for sid, _ in expected[s]} != subjects[s]:
            raise RuntimeError(f"{s}: cache/subject split mismatch")
        for sid, wi in expected[s]:
            if sid not in subject_split or wi < 0:
                raise RuntimeError(f"Unexpected cache row: {sid}/{wi}")

    missing_windows = []
    for s in SPLITS:
        eeg_meta = PROCESSED / "mmap_signals_MAE_processed" / s / "EEG/C4-M1.pickle"
        if not eeg_meta.is_file():
            raise FileNotFoundError(eeg_meta)
        with eeg_meta.open("rb") as f:
            info = pickle.load(f)["sig_info"]
        for sid in subjects[s]:
            matching = [v for name, v in info.items() if subject_id(name) == sid]
            if len(matching) != 1:
                raise RuntimeError(f"{s}: EEG metadata missing {sid}")
            start, end = matching[0]["pos"]
            length = int(end) - int(start)
            annotation = PROCESSED / "annotations_split" / s / f"{sid}_ann.npy"
            if not annotation.is_file():
                raise FileNotFoundError(annotation)
            labels = np.load(annotation, mmap_mode="r", allow_pickle=False)
            if len(labels) != length or not np.isin(labels, [-1, 0, 1, 2, 3, 4]).all():
                raise RuntimeError(f"{sid}: invalid EEG/annotation alignment")
            # Cover all original 30-s epochs, including a padded last window.
            wanted = {(sid, wi) for wi in range((length + 511) // 512)}
            missing_windows.extend((s, sid, wi) for _, wi in sorted(wanted - expected[s]))

    print("Seed:", a.seed)
    print("Transformer:", ckpt_t)
    print("VQGAN:", ckpt_v)
    print("Cache:", cache)
    for s in SPLITS:
        print(f"{s}: {len(subjects[s])} subjects, {len(expected[s])} cached windows")
    print("Windows absent for full-night coverage:", len(missing_windows))
    if missing_windows:
        print("First missing:", missing_windows[:6])
    if a.check_only:
        print("CHECK COMPLETE (no generation)")
        return
    if missing_windows and not a.allow_partial_coverage:
        raise RuntimeError(
            "DREAMT cache has truncated nights. Rebuild full windows before "
            "final evaluation. For a first-window-only pilot, pass "
            "--allow-partial-coverage."
        )
    if out.exists():
        raise FileExistsError(f"Output already exists: {out}")

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ENGINE.parent))
    spec = importlib.util.spec_from_file_location("dreamt_staging_engine", ENGINE)
    engine = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = engine
    spec.loader.exec_module(engine)

    engine.normalize_id = lambda value: subject_id(value).lower()
    engine.canonical_shhs1_id = subject_id
    engine.is_shhs1 = lambda row: str(row.get("dataset", "")).startswith("dreamt_")

    @lru_cache(maxsize=256)
    def load_labels(_root_string, sid):
        sid = subject_id(sid)
        s = subject_split[sid]
        src = PROCESSED / "annotations_split" / s / f"{sid}_ann.npy"
        values = np.load(src, allow_pickle=False)
        result = np.full(values.shape, -100, dtype=np.int16)
        valid = (values >= 0) & (values < 5)
        result[valid] = STAGE_MAP[values[valid].astype(np.int64)]
        return result

    engine.load_shhs1_labels_30sec_cached = load_labels

    def selected_init(self, *, wanted_ids, transformer_data_root,
                      preprocessed_root, split, **kwargs):
        self.wanted_ids = set(wanted_ids)
        self.main_dataset = engine.ev.PaperTransformerEvaluationDataset(
            transformer_data_root=transformer_data_root,
            preprocessed_root=preprocessed_root,
            split="all",
        )
        self.fallback_dataset = None
        self.selected = [
            ("main", i, row)
            for i, row in enumerate(self.main_dataset.index_rows)
            if row.get("dataset") == f"dreamt_{split}" and key(row) in expected[split]
        ]
        actual = {key(row) for _, _, row in self.selected}
        if actual != expected[split] or len(actual) != len(self.selected):
            raise RuntimeError(f"{split}: cached windows selection mismatch")
        print(f"DREAMT {split}: {len(self.selected)} windows selected")

    original_getitem = engine.SelectedEvaluationDataset.__getitem__

    def selected_getitem(self, i):
        item = original_getitem(self, i)
        row = self.selected[i][2]
        start_sec = int(row["window_index"]) * 512 * 30
        item["_builder_start_sec"] = start_sec
        item["start_sec"] = start_sec
        return item

    engine.SelectedEvaluationDataset.__init__ = selected_init
    engine.SelectedEvaluationDataset.__getitem__ = selected_getitem

    ids_dir = out.parent / f"split_ids_dreamt_seed_{a.seed}"
    ids_dir.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        (ids_dir / f"{s}.txt").write_text(
            "".join(f"{sid}\n" for sid in sorted(subjects[s]))
        )

    argv = [str(ENGINE),
            "--transformer-checkpoint", str(ckpt_t),
            "--vqgan-checkpoint", str(ckpt_v),
            "--main-transformer-root", str(tr),
            "--main-preprocessed-root", str(pp),
            "--test-transformer-root", str(tr),
            "--test-preprocessed-root", str(pp),
            "--train-ids", str(ids_dir / "train.txt"),
            "--val-ids", str(ids_dir / "val.txt"),
            "--test-ids", str(ids_dir / "test.txt"),
            "--label-root", str(PROCESSED / "annotations"),
            "--output-root", str(out),
            "--batch-size", str(a.batch_size),
            "--num-workers", str(a.num_workers),
            "--splits", *SPLITS]
    old = sys.argv
    try:
        sys.argv = argv
        engine.main()
    finally:
        sys.argv = old

    for s in SPLITS:
        folder = out / s
        meta = json.loads((folder / "metadata.json").read_text())
        n = int(meta["num_samples"])
        if n != len(expected[s]) or tuple(meta["label_shape"]) != (n, 512):
            raise RuntimeError(f"{s}: incorrect output shape/count")
        actual = {key(r) for r in read_jsonl(folder / "index.jsonl")}
        if actual != expected[s]:
            raise RuntimeError(f"{s}: output windows mismatch")
        labels = np.memmap(folder / "labels.mmap", mode="r",
                           dtype=np.dtype(meta["label_dtype"]),
                           shape=(n, 512))
        if not np.isin(labels, [-100, 0, 1, 2, 3]).all():
            raise RuntimeError(f"{s}: unexpected output labels")
        print(f"{s}: {n} windows, {np.count_nonzero(labels >= 0)} valid labels")
        meta["dataset"] = "dreamt"
        meta["label_original_resolution_sec"] = 30
        meta["label_downstream_resolution_sec"] = 30
        meta["partial_coverage_pilot"] = bool(missing_windows)
        (folder / "metadata.json").write_text(json.dumps(meta, indent=2))
    print("DONE:", out)


if __name__ == "__main__":
    main()
