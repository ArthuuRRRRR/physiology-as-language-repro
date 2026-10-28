#!/usr/bin/env python3
"""PaSL inference on Keondo-preprocessed CFS respiration."""

import argparse
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

from paper_architecture import PaperMaskedRespirationToEEGTransformer
from src.models.vqgan import VQGAN
from paper_architecture.evaluate_paper_masked_transformer import (
    decode_token_ids,
    load_state_dict,
    standardize_respiration,
)

ROOT = Path("retraining_wav2sleep/outputs")
SRC = Path(
    "/hdd2/kdpark/sleep_datasets/cfs/"
    "mmap_signals_MAE_processed_wav2sleep"
)
GT = ROOT / "downstream/cfs_keondo_gt/cfs"


def load_models(seed, device):
    base = ROOT / f"seed_{seed}"

    tp = base / "transformer/checkpoint_best_mae.pt"
    vp = base / "vqgan/checkpoint_best_final.pt"

    for p in (tp, vp):
        if not p.is_file():
            raise FileNotFoundError(p)

    tck = torch.load(
        tp, map_location="cpu", weights_only=False
    )

    transformer = PaperMaskedRespirationToEEGTransformer(
        **tck["model_config"]
    ).to(device)

    transformer.load_state_dict(
        load_state_dict(
            tck,
            ["model_state_dict", "transformer_state_dict"]
        ),
        strict=True,
    )
    transformer.eval()
    del tck

    vck = torch.load(
        vp, map_location="cpu", weights_only=False
    )

    vqgan = VQGAN().to(device)

    vqgan.load_state_dict(
        load_state_dict(
            vck,
            [
                "model_state_dict",
                "vqgan_state_dict",
                "generator_state_dict",
                "state_dict",
            ],
        ),
        strict=True,
    )
    vqgan.eval()
    del vck

    return transformer, vqgan, tp, vp


def get_inputs(split, channel):
    gt = GT / split

    gm = json.loads(
        (gt / "metadata.json").read_text()
    )

    rows = [
        json.loads(x)
        for x in (gt / "index.jsonl").read_text().splitlines()
        if x.strip()
    ]

    n = gm["num_samples"]

    if (
        len(rows) != n
        or tuple(gm["eeg_shape"]) != (n, 256, 512)
        or tuple(gm["label_shape"]) != (n, 512)
    ):
        raise RuntimeError(
            f"Unexpected GT layout: {split}"
        )

    resp_dir = SRC / split / "RESP"
    pickle_path = resp_dir / f"{channel}.pickle"
    mmap_path = resp_dir / f"{channel}.mmap"

    with pickle_path.open("rb") as f:
        info = pickle.load(f)

    info_by_subject = info["sig_info"]
    shape = tuple(info["data_shape"])

    if len(shape) != 2 or shape[1] != 3000:
        raise RuntimeError(
            f"Unexpected respiration shape: {shape}"
        )

    groups = {}

    for i, row in enumerate(rows):
        sid = row["subject_id"]
        offset = int(
            row["start_epoch_in_trimmed_recording"]
        )
        valid = int(row["valid_epochs"])

        if sid not in info_by_subject:
            raise RuntimeError(
                f"{split}: missing respiration {sid}"
            )

        start, stop = (
            int(x)
            for x in info_by_subject[sid]["pos"]
        )

        if not (
            0 <= offset < stop - start
            and 0 < valid <= 512
            and offset + valid <= stop - start
        ):
            raise RuntimeError(
                f"{split}: invalid offset "
                f"{sid}: {offset}+{valid}/{stop-start}"
            )

        groups.setdefault(sid, []).append(
            (i, offset, valid)
        )

    return (
        gm, rows, groups,
        mmap_path, shape, info_by_subject
    )


def run(seed, split, channel, batch_size, check_only):
    (
        gm, rows, groups,
        mmap_path, shape, src_info
    ) = get_inputs(split, channel)

    dest = (
        ROOT
        / f"downstream/cfs_keondo_pasl_seed{seed}"
        / "cfs"
        / split
    )

    print(
        f"{split}: {len(rows)} windows / "
        f"{len(groups)} subjects / channel={channel}",
        flush=True,
    )

    if check_only:
        return

    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite: {dest}"
        )

    dest.mkdir(parents=True, exist_ok=True)

    for name in (
        "labels.mmap",
        "ground_truth_eeg.mmap",
    ):
        (dest / name).symlink_to(
            (GT / split / name).resolve()
        )

    shutil.copyfile(
        GT / split / "index.jsonl",
        dest / "index.jsonl",
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    transformer, vqgan, tp, vp = load_models(
        seed, device
    )

    target = np.memmap(
        dest / "synthesized_eeg.mmap",
        dtype=np.float32,
        mode="w+",
        shape=(len(rows), 256, 512),
    )

    raw = np.memmap(
        mmap_path,
        dtype=np.float32,
        mode="r",
        shape=shape,
    )

    done = 0

    with torch.inference_mode():

        for si, (sid, windows) in enumerate(
            groups.items(), 1
        ):

            start, stop = (
                int(x)
                for x in src_info[sid]["pos"]
            )

            # Keondo respiration: 100 Hz -> 10 Hz
            x100 = np.asarray(
                raw[start:stop]
            ).reshape(-1)

            x10 = resample_poly(
                x100, up=1, down=10
            ).astype(np.float32)

            if len(x10) != (stop - start) * 300:
                raise RuntimeError(
                    f"Downsampling failed: {sid}"
                )

            for k in range(
                0, len(windows), batch_size
            ):
                subset = windows[k:k + batch_size]

                # PaSL: 64 blocks x 2400 samples
                xb = np.zeros(
                    (len(subset), 64, 2400),
                    dtype=np.float32,
                )

                for j, (_, offset, valid) in enumerate(
                    subset
                ):
                    xb[j].reshape(-1)[:valid * 300] = (
                        x10[
                            offset * 300:
                            (offset + valid) * 300
                        ]
                    )

                resp = torch.from_numpy(
                    xb
                ).to(device)

                resp = standardize_respiration(
                    resp
                )

                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    logits = (
                        transformer.predict_from_respiration(
                            resp
                        )
                    )

                if (
                    logits.ndim != 3
                    or logits.shape[1] != 512
                ):
                    raise RuntimeError(
                        f"Bad logits: {tuple(logits.shape)}"
                    )

                tokens = (
                    logits.argmax(dim=-1)
                    .reshape(-1, 8, 64)
                )

                decoded = decode_token_ids(
                    vqgan, tokens
                )

                y = (
                    decoded[:, 0]
                    .float()
                    .cpu()
                    .numpy()
                )

                if (
                    y.shape != (len(subset), 256, 512)
                    or not np.isfinite(y).all()
                ):
                    raise RuntimeError(
                        f"Invalid predictions: {y.shape}"
                    )

                for j, (idx, _, _) in enumerate(subset):
                    target[idx] = np.clip(
                        y[j], 0.0, 1.0
                    )

                done += len(subset)

            if si % 25 == 0 or si == len(groups):
                target.flush()
                print(
                    f"{split}: {si}/{len(groups)} "
                    f"subjects, {done}/{len(rows)} windows",
                    flush=True,
                )

    if done != len(rows):
        raise RuntimeError(
            "Some windows were not generated"
        )

    target.flush()

    meta = dict(gm)
    meta.update(
        synthesis="PaSL",
        seed=seed,
        source_channel=channel,
        source_respiration_mmap=str(mmap_path),
        transformer_checkpoint=str(tp),
        vqgan_checkpoint=str(vp),
        spectral_representation="VQGAN-native",
    )

    (dest / "metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )

    print(f"COMPLETE: {dest}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--seed", type=int,
        required=True, choices=(42, 0, 1)
    )
    p.add_argument(
        "--split",
        choices=("train", "val", "test", "all"),
        default="all",
    )
    p.add_argument(
        "--channel", default="AIRFLOW"
    )
    p.add_argument(
        "--batch-size", type=int, default=2
    )
    p.add_argument(
        "--check-only", action="store_true"
    )

    a = p.parse_args()

    if a.batch_size < 1:
        p.error("batch-size must be positive")

    splits = (
        ("train", "val", "test")
        if a.split == "all"
        else (a.split,)
    )

    # Check all inputs before writing anything.
    for split in splits:
        get_inputs(split, a.channel)

    for split in splits:
        run(
            a.seed, split,
            a.channel, a.batch_size,
            a.check_only,
        )


if __name__ == "__main__":
    main()
