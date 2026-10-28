from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_architecture import PaperMaskedRespirationToEEGTransformer
from src.models.vqgan import VQGAN


TRANSFORMER_CHECKPOINT = Path(
    "outputs/paper_masked_transformer_resume140_mae_every1/"
    "snapshots/checkpoint_epoch146.pt"
)

VQGAN_CHECKPOINT = Path(
    "outputs/vqgan_multidataset_final_ddp/checkpoint_best.pt"
)

CANONICAL_SUBJECT_ROOT = Path(
    "outputs/downstream_survival_features_gt"
)

OUTPUT_ROOT = Path(
    "outputs/downstream_survival_features_synth_epoch146_full"
)

SPLITS = {
    "train": {
        "root": Path(
            "outputs/paper_transformer_data_final"
        ),
        "source_split": "train",
    },
    "val": {
        "root": Path(
            "outputs/paper_transformer_data_final"
        ),
        "source_split": "val",
    },
    "test": {
        "root": Path(
            "outputs/paper_transformer_data_shhs1_test_final"
        ),
        "source_split": "test",
    },
}

BATCH_SIZE = 8
SEED = 42


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def load_state_dict(
    checkpoint,
    possible_keys,
):
    for key in possible_keys:
        value = checkpoint.get(key)

        if isinstance(value, dict):
            return value

    raise KeyError(
        f"No state dict found. "
        f"Available keys: {list(checkpoint.keys())}"
    )


def standardize_respiration(
    respiration,
    eps=1e-6,
):
    mean = respiration.mean(
        dim=(1, 2),
        keepdim=True,
    )

    std = respiration.std(
        dim=(1, 2),
        keepdim=True,
        unbiased=False,
    )

    return (
        respiration - mean
    ) / (
        std + eps
    )


def decode_token_ids(
    vqgan,
    token_ids,
):
    codebook = F.normalize(
        vqgan.quantizer.codebook.weight,
        p=2,
        dim=1,
    )

    quantized = F.embedding(
        token_ids,
        codebook,
    )

    quantized = quantized.permute(
        0,
        3,
        1,
        2,
    ).contiguous()

    return vqgan.decode(
        quantized
    )


def load_models(device):
    checkpoint = torch.load(
        TRANSFORMER_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    model_config = checkpoint[
        "model_config"
    ]

    transformer = (
        PaperMaskedRespirationToEEGTransformer(
            **model_config
        ).to(device)
    )

    transformer.load_state_dict(
        load_state_dict(
            checkpoint,
            [
                "model_state_dict",
                "transformer_state_dict",
            ],
        ),
        strict=True,
    )

    transformer.eval()

    vq_checkpoint = torch.load(
        VQGAN_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    vqgan = VQGAN().to(
        device
    )

    vqgan.load_state_dict(
        load_state_dict(
            vq_checkpoint,
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

    for p in transformer.parameters():
        p.requires_grad = False

    for p in vqgan.parameters():
        p.requires_grad = False

    return transformer, vqgan


def load_canonical_subjects(
    split,
):
    path = (
        CANONICAL_SUBJECT_ROOT
        / f"{split}_subjects.txt"
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        subjects = [
            line.strip()
            for line in f
            if line.strip()
        ]

    return subjects


def load_transformer_split(
    root,
    split,
):
    split_dir = (
        root
        / split
    )

    with (
        split_dir / "metadata.json"
    ).open(
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    with (
        split_dir / "index.jsonl"
    ).open(
        "r",
        encoding="utf-8",
    ) as f:
        rows = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    if len(rows) != int(
        metadata["num_samples"]
    ):
        raise RuntimeError(
            f"{split}: index/metadata mismatch"
        )

    resp_info = metadata[
        "respiration"
    ]

    shape = tuple(
        resp_info["shape"]
    )

    respiration = np.memmap(
        split_dir
        / resp_info["path"],
        mode="r",
        dtype=resp_info["dtype"],
        shape=shape,
    )

    return rows, respiration


@torch.inference_mode()
def synthesize(
    transformer,
    vqgan,
    respiration_np,
    device,
):
    respiration = torch.from_numpy(
        np.asarray(
            respiration_np,
            dtype=np.float32,
        )
    ).to(
        device,
        non_blocking=True,
    )

    respiration = (
        standardize_respiration(
            respiration
        )
    )

    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=(
            device.type == "cuda"
        ),
    ):
        logits = (
            transformer
            .predict_from_respiration(
                respiration
            )
        )

        tokens = (
            logits
            .argmax(dim=-1)
            .view(
                respiration.shape[0],
                8,
                64,
            )
        )

        eeg = decode_token_ids(
            vqgan,
            tokens,
        )

    eeg = (
        eeg[:, 0]
        .float()
        .clamp(0.0, 1.0)
        .cpu()
        .numpy()
    )

    return eeg


def build_split(
    split,
    config,
    transformer,
    vqgan,
    device,
):
    canonical_subjects = (
        load_canonical_subjects(
            split
        )
    )

    canonical_set = set(
        canonical_subjects
    )

    rows, respiration = (
        load_transformer_split(
            config["root"],
            config["source_split"],
        )
    )

    selected = []

    for mmap_index, row in enumerate(
        rows
    ):
        subject = str(
            row["subject_id"]
        )

        if subject in canonical_set:
            selected.append(
                (
                    mmap_index,
                    subject,
                )
            )

    found_subjects = {
        subject
        for _, subject in selected
    }

    missing = (
        canonical_set
        - found_subjects
    )

    if missing:
        raise RuntimeError(
            f"{split}: "
            f"{len(missing)} canonical subjects "
            f"have no Transformer windows. "
            f"First missing: "
            f"{sorted(missing)[:10]}"
        )

    print()
    print("=" * 72)
    print(split.upper())
    print("=" * 72)
    print(
        "Canonical subjects:",
        len(canonical_subjects),
    )
    print(
        "Selected windows   :",
        len(selected),
    )

    freq_sum = defaultdict(
        lambda: np.zeros(
            256,
            dtype=np.float64,
        )
    )

    column_count = defaultdict(
        int
    )

    for start in range(
        0,
        len(selected),
        BATCH_SIZE,
    ):
        batch = selected[
            start:
            start + BATCH_SIZE
        ]

        indices = [
            x[0]
            for x in batch
        ]

        subjects = [
            x[1]
            for x in batch
        ]

        respiration_batch = (
            np.stack(
                [
                    np.asarray(
                        respiration[i],
                        dtype=np.float32,
                    )
                    for i in indices
                ],
                axis=0,
            )
        )

        eeg_batch = synthesize(
            transformer,
            vqgan,
            respiration_batch,
            device,
        )

        for subject, eeg in zip(
            subjects,
            eeg_batch,
        ):
            freq_sum[
                subject
            ] += eeg.sum(
                axis=1,
                dtype=np.float64,
            )

            column_count[
                subject
            ] += eeg.shape[1]

        done = min(
            start + BATCH_SIZE,
            len(selected),
        )

        if (
            done % 100 == 0
            or done == len(selected)
        ):
            print(
                f"{split}: "
                f"{done}/{len(selected)}",
                flush=True,
            )

    features = []

    for subject in canonical_subjects:
        features.append(
            (
                freq_sum[subject]
                / column_count[subject]
            ).astype(
                np.float32
            )
        )

    features = np.stack(
        features,
        axis=0,
    )

    np.save(
        OUTPUT_ROOT
        / f"{split}_features.npy",
        features,
    )

    with (
        OUTPUT_ROOT
        / f"{split}_subjects.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        for subject in canonical_subjects:
            f.write(
                subject + "\n"
            )

    print(
        "Features:",
        features.shape,
    )
    print(
        "Mean:",
        float(
            features.mean()
        ),
    )
    print(
        "Std:",
        float(
            features.std()
        ),
    )


def main():
    seed_everything()

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    print(
        "Transformer:",
        TRANSFORMER_CHECKPOINT,
    )

    transformer, vqgan = (
        load_models(
            device
        )
    )

    for split, config in (
        SPLITS.items()
    ):
        build_split(
            split,
            config,
            transformer,
            vqgan,
            device,
        )


if __name__ == "__main__":
    main()
