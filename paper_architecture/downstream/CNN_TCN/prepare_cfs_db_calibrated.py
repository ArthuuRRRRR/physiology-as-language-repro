from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np


SRC_ROOT = Path("outputs/downstream_ViT_synth/cfs")
DST_ROOT = Path("outputs/downstream_CFS_db_calibrated/cfs")

NORMALIZATION = Path(
    "outputs/vqgan_multidataset_preprocessed_final_gpu4/"
    "normalization.json"
)

SCALE = 1.292520
OFFSET = -6.5463


with NORMALIZATION.open() as f:
    norm = json.load(f)

MIN_DB = float(norm["min_db"])
MAX_DB = float(norm["max_db"])


def process_split(split: str):
    src = SRC_ROOT / split
    dst = DST_ROOT / split

    dst.mkdir(parents=True, exist_ok=True)

    with (src / "metadata.json").open() as f:
        meta = json.load(f)

    shape = tuple(meta["eeg_shape"])
    dtype = np.dtype(meta["eeg_dtype"])

    x_in = np.memmap(
        src / "synthesized_eeg.mmap",
        dtype=dtype,
        mode="r",
        shape=shape,
    )

    x_out = np.memmap(
        dst / "synthesized_eeg.mmap",
        dtype="float16",
        mode="w+",
        shape=shape,
    )

    print(f"{split}: {shape}")

    for i in range(shape[0]):
        # Current synthesized EEG is in VQGAN normalized [0,1] space.
        x = np.asarray(
            x_in[i],
            dtype=np.float32,
        )

        # [0,1] -> VQGAN dB
        db_vqgan = (
            x * (MAX_DB - MIN_DB)
            + MIN_DB
        )

        # VQGAN dB -> estimated original CFS dB
        db_cfs = (
            db_vqgan - OFFSET
        ) / SCALE

        # Keep the same numeric representation expected
        # by the downstream classifier.
        x_cfs = (
            db_cfs - MIN_DB
        ) / (
            MAX_DB - MIN_DB
        )

        x_cfs = np.clip(
            x_cfs,
            0.0,
            1.0,
        )

        x_out[i] = x_cfs.astype(
            np.float16
        )

        if (
            (i + 1) % 100 == 0
            or i + 1 == shape[0]
        ):
            print(
                f"  {i + 1}/{shape[0]}"
            )

    x_out.flush()

    # Labels/index stay exactly identical.
    for filename in [
        "labels.mmap",
        "index.jsonl",
        "metadata.json",
    ]:
        shutil.copy2(
            src / filename,
            dst / filename,
        )

    meta["eeg_dtype"] = "float16"

    with (dst / "metadata.json").open(
        "w"
    ) as f:
        json.dump(
            meta,
            f,
            indent=2,
        )

    del x_in
    del x_out


for split in ["train", "val", "test"]:
    process_split(split)

print()
print("Done:", DST_ROOT)
