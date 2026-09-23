from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from arousal_linear_model import (
    ArousalWakeLinear,
    count_parameters,
)


DEFAULT_DATA_ROOT = Path(
    "outputs/downstream_sleep_staging_epoch146_resume140"
)

DEFAULT_AROUSAL_ROOT = Path(
    "/hdd2/kdpark/sleep_datasets/shhs1/arousal_epoch_labels"
)

DEFAULT_OUTPUT_DIR = Path(
    "outputs/downstream_arousal_epoch146_resume140/shhs1"
)

WAKE_STAGE = 0
AROUSAL_LABEL_SEC = 30


@dataclass
class SplitTable:
    split: str
    eeg: np.memmap
    stage: np.memmap
    rows: list[dict]
    selected_subjects: list[str]
    row_indices: np.ndarray
    epoch_indices: np.ndarray
    arousal: np.ndarray
    wake: np.ndarray
    sleep: np.ndarray
    subject_ids: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_index(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def subsample(items: list[str], n: int) -> list[str]:
    """
    Same evenly-spaced subject subsampling used in Keondo's eval_spec.py.
    n <= 0 or n >= len(items) -> all subjects.
    """
    items = list(items)

    if not n or n <= 0 or n >= len(items):
        return items

    idx = np.linspace(
        0,
        len(items) - 1,
        int(n),
    ).round().astype(int)

    return [
        items[i]
        for i in sorted(set(idx.tolist()))
    ]


def arousal_path(
    arousal_root: Path,
    subject_id: str,
) -> Path:
    return (
        arousal_root
        / f"{subject_id}_arousal_epoch.npy"
    )


def load_arousal_meta(
    arousal_root: Path,
) -> dict:
    path = (
        arousal_root
        / "arousal_epoch_meta.json"
    )
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_split_table(
    data_root: Path,
    arousal_root: Path,
    split: str,
    subject_cap: int,
) -> SplitTable:
    """
    Build Keondo-style epoch-level probe data from PaSL's already generated
    30-s spectrogram bins.

    PaSL cache:
        synthesized_eeg.mmap: (N_windows, 256, 512)
        labels.mmap:          (N_windows, 512)

    Since PaSL already has one 256-bin spectrogram column per 30-s epoch,
    fpe = 1 and no temporal concatenation/resampling is needed.
    """
    split_dir = data_root / split

    metadata = load_json(
        split_dir / "metadata.json"
    )
    rows = load_index(
        split_dir / "index.jsonl"
    )

    eeg_shape = tuple(
        int(x)
        for x in metadata["eeg_shape"]
    )
    label_shape = tuple(
        int(x)
        for x in metadata["label_shape"]
    )

    if len(eeg_shape) != 3:
        raise RuntimeError(
            f"{split}: expected EEG shape (N,F,T), got {eeg_shape}"
        )
    if eeg_shape[1] != 256:
        raise RuntimeError(
            f"{split}: expected 256 frequency bins, got {eeg_shape[1]}"
        )
    if eeg_shape[2] != 512:
        raise RuntimeError(
            f"{split}: expected 512 x 30-s epochs/window, got {eeg_shape[2]}"
        )
    if label_shape != (
        eeg_shape[0],
        eeg_shape[2],
    ):
        raise RuntimeError(
            f"{split}: labels {label_shape} do not match EEG {eeg_shape}"
        )
    if len(rows) != eeg_shape[0]:
        raise RuntimeError(
            f"{split}: index has {len(rows)} rows but EEG has {eeg_shape[0]} windows"
        )

    eeg = np.memmap(
        split_dir / "synthesized_eeg.mmap",
        dtype=np.dtype(
            metadata.get(
                "eeg_dtype",
                "float16",
            )
        ),
        mode="r",
        shape=eeg_shape,
    )

    stage = np.memmap(
        split_dir / "labels.mmap",
        dtype=np.dtype(
            metadata.get(
                "label_dtype",
                "int8",
            )
        ),
        mode="r",
        shape=label_shape,
    )

    # Same logic as Keondo: keep only subjects that have epoch-level arousal labels.
    roster = sorted(
        {
            str(row["subject_id"])
            for row in rows
            if arousal_path(
                arousal_root,
                str(row["subject_id"]),
            ).is_file()
        }
    )

    selected_subjects = subsample(
        roster,
        subject_cap,
    )
    selected_set = set(
        selected_subjects
    )

    arousal_cache: dict[str, np.ndarray] = {}

    row_parts = []
    epoch_parts = []
    arousal_parts = []
    wake_parts = []
    sleep_parts = []
    subject_parts = []

    for row_pos, row in enumerate(rows):
        sid = str(
            row["subject_id"]
        )
        if sid not in selected_set:
            continue

        if sid not in arousal_cache:
            arr = np.load(
                arousal_path(
                    arousal_root,
                    sid,
                ),
                allow_pickle=False,
            )
            arr = np.asarray(
                arr,
                dtype=np.float32,
            ).reshape(-1)

            if not np.isin(
                arr,
                [0.0, 1.0],
            ).all():
                raise RuntimeError(
                    f"{sid}: arousal labels are not binary 0/1"
                )
            arousal_cache[sid] = arr

        arousal_all = arousal_cache[sid]
        window_index = int(
            row["window_index"]
        )

        start = (
            window_index
            * 512
        )
        end = start + 512

        # Keondo pads arousal labels past the available annotation with zeros.
        if len(arousal_all) < end:
            padded = np.zeros(
                end,
                dtype=np.float32,
            )
            padded[:len(arousal_all)] = (
                arousal_all
            )
            arousal_window = (
                padded[start:end]
            )
        else:
            arousal_window = (
                arousal_all[start:end]
            )

        stage_window = np.asarray(
            stage[row_pos],
            dtype=np.int64,
        )

        # Our staging cache is already mapped to:
        # Wake=0, Light=1, Deep=2, REM=3.
        valid = (
            (stage_window >= 0)
            & (stage_window <= 3)
        )

        if not valid.any():
            continue

        ep = np.flatnonzero(
            valid
        ).astype(np.int32)

        s = stage_window[
            valid
        ]

        ar = arousal_window[
            valid
        ].astype(
            np.float32,
            copy=False,
        )

        wk = (
            s == WAKE_STAGE
        ).astype(np.float32)

        slp = (
            s != WAKE_STAGE
        )

        row_parts.append(
            np.full(
                len(ep),
                row_pos,
                dtype=np.int32,
            )
        )
        epoch_parts.append(ep)
        arousal_parts.append(ar)
        wake_parts.append(wk)
        sleep_parts.append(slp)
        subject_parts.append(
            np.full(
                len(ep),
                sid,
                dtype=object,
            )
        )

    if not row_parts:
        raise RuntimeError(
            f"{split}: no subjects had both generated EEG and arousal labels"
        )

    return SplitTable(
        split=split,
        eeg=eeg,
        stage=stage,
        rows=rows,
        selected_subjects=selected_subjects,
        row_indices=np.concatenate(
            row_parts
        ),
        epoch_indices=np.concatenate(
            epoch_parts
        ),
        arousal=np.concatenate(
            arousal_parts
        ),
        wake=np.concatenate(
            wake_parts
        ),
        sleep=np.concatenate(
            sleep_parts
        ),
        subject_ids=np.concatenate(
            subject_parts
        ),
    )


def epoch_features(
    table: SplitTable,
) -> torch.Tensor:
    """
    Keondo's epoch_features adapted to PaSL.

    His generic implementation concatenates all spectrogram frames inside
    one 30-s epoch. PaSL already uses 30-s frames, so each epoch contributes
    exactly one 256-bin column -> X shape (epochs, 256).
    """
    # np advanced indexing with a memmap returns a materialized array.
    x = table.eeg[
        table.row_indices,
        :,
        table.epoch_indices,
    ]

    x = np.asarray(
        x,
        dtype=np.float32,
    )

    if x.ndim != 2 or x.shape[1] != 256:
        raise RuntimeError(
            f"{table.split}: expected epoch features (N,256), got {x.shape}"
        )

    return torch.from_numpy(x)


def fit_linear_probe(
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    pos_weight,
    epochs: int = 20,
    lr: float = 1e-2,
    batch: int = 65536,
):
    """
    Mirror Keondo's _fit_linear_probe for arousal.

    - train-set standardisation
    - Linear(D -> 2): [arousal, wake]
    - Adam(lr=1e-2, weight_decay=1e-4)
    - BCEWithLogitsLoss with one pos_weight per head
    - 20 epochs
    - epoch-level batches of 65,536
    """
    mu = X.mean(
        0,
        keepdim=True,
    )
    sd = X.std(
        0,
        keepdim=True,
    ).clamp(
        min=1e-6
    )

    Xn = (
        X - mu
    ) / sd

    y = y.float()
    if y.dim() == 1:
        y = y[:, None]

    head = ArousalWakeLinear(
        input_dim=X.shape[1]
    ).to(device)

    opt = torch.optim.Adam(
        head.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    pw = torch.as_tensor(
        np.atleast_1d(
            np.asarray(
                pos_weight,
                dtype=np.float32,
            )
        ),
        device=device,
    )

    lossf = torch.nn.BCEWithLogitsLoss(
        pos_weight=pw,
        reduction="none",
    )

    n = Xn.shape[0]

    for epoch in range(epochs):
        perm = torch.randperm(n)

        total_loss = 0.0
        total_seen = 0

        for s in range(
            0,
            n,
            batch,
        ):
            idx = perm[
                s:s + batch
            ]

            xb = Xn[
                idx
            ].to(device)

            yb = y[
                idx
            ].to(device)

            loss = lossf(
                head(xb),
                yb,
            )  # (batch, 2)

            # Exact no-mask branch from Keondo:
            # mean each head separately, then sum both head losses.
            loss = loss.mean(
                0
            )

            opt.zero_grad()
            loss.sum().backward()
            opt.step()

            bs = int(
                len(idx)
            )
            total_loss += (
                float(
                    loss.sum().item()
                )
                * bs
            )
            total_seen += bs

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"BCE={total_loss / max(total_seen, 1):.6f}",
            flush=True,
        )

    head.eval()
    return head, (
        mu,
        sd,
    )


@torch.no_grad()
def probs(
    head: ArousalWakeLinear,
    X: torch.Tensor,
    mu: torch.Tensor,
    sd: torch.Tensor,
    device: torch.device,
):
    """
    Same chunk size and sigmoid probability conversion as Keondo.
    """
    out = []

    for s in range(
        0,
        X.shape[0],
        65536,
    ):
        xb = (
            (
                X[
                    s:s + 65536
                ]
                - mu
            )
            / sd
        ).to(device)

        out.append(
            torch.sigmoid(
                head(xb)
            ).cpu()
        )

    p = torch.cat(
        out
    ).numpy()

    return (
        p[:, 0],
        p[:, 1],
    )


def counts(
    pred: np.ndarray,
    gt: np.ndarray,
):
    tp = int(
        (pred & gt).sum()
    )
    return (
        tp,
        int(
            gt.sum()
        ),
        int(
            pred.sum()
        ),
    )


def f1_of(
    tp: int,
    n_gt: int,
    n_pred: int,
):
    recall = (
        tp
        / max(
            n_gt,
            1,
        )
    )
    precision = (
        tp
        / max(
            n_pred,
            1,
        )
    )

    f1 = (
        2
        * precision
        * recall
        / max(
            precision + recall,
            1e-9,
        )
    )

    return (
        f1,
        precision,
        recall,
    )


def balanced_acc(
    pred: np.ndarray,
    gt: np.ndarray,
) -> float:
    tpr = (
        (pred & gt).sum()
        / max(
            gt.sum(),
            1,
        )
    )
    tnr = (
        (~pred & ~gt).sum()
        / max(
            (~gt).sum(),
            1,
        )
    )

    return float(
        (
            tpr + tnr
        )
        / 2
    )


def auroc(
    gt: np.ndarray,
    score: np.ndarray,
):
    try:
        return float(
            roc_auc_score(
                gt,
                score,
            )
        )
    except Exception:
        return None


def reading(
    pred: np.ndarray,
    gt: np.ndarray,
    score: np.ndarray,
) -> dict:
    tp, n_gt, n_pred = counts(
        pred,
        gt,
    )
    f1, precision, recall = f1_of(
        tp,
        n_gt,
        n_pred,
    )

    return {
        "f1": float(f1),
        "precision": float(
            precision
        ),
        "recall": float(
            recall
        ),
        "acc": float(
            (
                pred == gt
            ).mean()
        ),
        "auroc": auroc(
            gt,
            score,
        ),
        "epochs": int(
            gt.shape[0]
        ),
        "positives": int(
            n_gt
        ),
        "predicted": int(
            n_pred
        ),
    }


def line(
    tag: str,
    result: dict,
    threshold: float,
) -> str:
    auc = (
        f'  AUROC {result["auroc"]:.3f}'
        if result[
            "auroc"
        ] is not None
        else ""
    )

    return (
        f"[arousal] {tag} @thr={threshold:.2f}: "
        f'F1 {result["f1"]:.3f}  '
        f'precision {result["precision"]:.3f}  '
        f'recall {result["recall"]:.3f}  '
        f'acc {result["acc"]:.3f}'
        f"{auc}  "
        f'({result["positives"]}/{result["epochs"]} positive epochs, '
        f'{result["predicted"]} predicted)'
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Keondo-style arousal linear probe on pre-generated "
            "PaSL epoch146 synthesized EEG."
        )
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
    )
    parser.add_argument(
        "--arousal-root",
        type=Path,
        default=DEFAULT_AROUSAL_ROOT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
    )

    # Match Keondo's eval_spec.py defaults.
    parser.add_argument(
        "--probe-train-subjects",
        type=int,
        default=300,
        help="0 = all; Keondo default = 300",
    )
    parser.add_argument(
        "--probe-val-subjects",
        type=int,
        default=150,
        help="0 = all; Keondo default = 150",
    )
    parser.add_argument(
        "--test-subjects",
        type=int,
        default=0,
        help="0 = full test split, matching Keondo",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=65536,
        help="Epoch-level probe batch size, matching Keondo",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    set_seed(
        args.seed
    )

    device = torch.device(
        f"cuda:{args.gpu}"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )
    print(
        "PaSL synthesized EEG:",
        args.data_root,
    )
    print(
        "Arousal labels:",
        args.arousal_root,
    )

    train = build_split_table(
        args.data_root,
        args.arousal_root,
        "train",
        args.probe_train_subjects,
    )
    val = build_split_table(
        args.data_root,
        args.arousal_root,
        "val",
        args.probe_val_subjects,
    )
    test = build_split_table(
        args.data_root,
        args.arousal_root,
        "test",
        args.test_subjects,
    )

    print()
    print(
        "[arousal] "
        f"{len(train.selected_subjects)} train / "
        f"{len(val.selected_subjects)} val / "
        f"{len(test.selected_subjects)} test subjects; "
        "30s PaSL spectrogram frames, 30s labels and scoring; "
        "two-head probe (arousal, wake)"
    )

    print(
        "[arousal] epochs: "
        f"train={len(train.arousal):,} "
        f"val={len(val.arousal):,} "
        f"test={len(test.arousal):,}"
    )

    # ------------------------------------------------------------
    # Train: exact Keondo-style two-head probe
    # ------------------------------------------------------------
    X_train = epoch_features(
        train
    )

    Y_train = torch.from_numpy(
        np.stack(
            [
                train.arousal,
                train.wake,
            ],
            axis=1,
        )
    )

    pos_ar = float(
        train.arousal.mean()
    )
    pos_wk = float(
        train.wake.mean()
    )

    pos_weight = [
        (
            1 - pos_ar
        )
        / max(
            pos_ar,
            1e-6,
        ),
        (
            1 - pos_wk
        )
        / max(
            pos_wk,
            1e-6,
        ),
    ]

    head, (
        mu,
        sd,
    ) = fit_linear_probe(
        X=X_train,
        y=Y_train,
        device=device,
        pos_weight=pos_weight,
        epochs=args.epochs,
        lr=args.lr,
        batch=args.batch_size,
    )

    wake_arousal_rate = (
        float(
            train.arousal[
                ~train.sleep
            ].mean()
        )
        if (
            ~train.sleep
        ).any()
        else 0.0
    )

    print(
        f"[arousal] probe trained on "
        f"{X_train.shape[0]:,} epochs of "
        f"1 x {X_train.shape[1]} bins "
        f"({pos_ar:.1%} arousal, "
        f"{pos_wk:.1%} wake; "
        f"{wake_arousal_rate:.1%} of wake epochs carry an arousal label)"
    )

    del (
        X_train,
        Y_train,
    )

    # ------------------------------------------------------------
    # Validation: choose thresholds exactly like Keondo
    # ------------------------------------------------------------
    X_val = epoch_features(
        val
    )
    val_ar_p, val_wk_p = probs(
        head,
        X_val,
        mu,
        sd,
        device,
    )
    del X_val

    val_ar = (
        val.arousal > 0.5
    )
    val_wk = (
        val.wake > 0.5
    )
    val_slp = val.sleep.astype(
        bool
    )

    grid = np.arange(
        0.1,
        0.91,
        0.05,
    )

    arousal_threshold = max(
        grid,
        key=lambda t: f1_of(
            *counts(
                val_ar_p[
                    val_slp
                ]
                >= t,
                val_ar[
                    val_slp
                ],
            )
        )[0],
    )

    wake_threshold = max(
        grid,
        key=lambda t: balanced_acc(
            val_wk_p >= t,
            val_wk,
        ),
    )

    arousal_threshold = float(
        arousal_threshold
    )
    wake_threshold = float(
        wake_threshold
    )

    print(
        f"[arousal] validation thresholds: "
        f"arousal={arousal_threshold:.2f} "
        f"(max GT-sleep F1), "
        f"wake={wake_threshold:.2f} "
        f"(max balanced accuracy)"
    )

    # ------------------------------------------------------------
    # Test: same three readings as Keondo
    # ------------------------------------------------------------
    X_test = epoch_features(
        test
    )
    test_ar_p, test_wk_p = probs(
        head,
        X_test,
        mu,
        sd,
        device,
    )
    del X_test

    gt_raw = (
        test.arousal > 0.5
    )
    gt_wake = (
        test.wake > 0.5
    )
    gt_slp = test.sleep.astype(
        bool
    )

    gt_sleep_only = (
        gt_raw
        & gt_slp
    )

    pred_ar = (
        test_ar_p
        >= arousal_threshold
    )
    pred_wake = (
        test_wk_p
        >= wake_threshold
    )

    # 1) Guideline-consistent main reading: remove GT Wake epochs.
    sleep_result = reading(
        pred_ar[
            gt_slp
        ],
        gt_raw[
            gt_slp
        ],
        test_ar_p[
            gt_slp
        ],
    )

    # 2) End-to-end reading: predicted sleep gates predicted arousal.
    gated_result = reading(
        pred_ar
        & ~pred_wake,
        gt_sleep_only,
        test_ar_p
        * (
            1 - test_wk_p
        ),
    )

    # 3) Legacy reading: every epoch, raw annotation, no gating.
    legacy_result = reading(
        pred_ar,
        gt_raw,
        test_ar_p,
    )

    wake_result = reading(
        pred_wake,
        gt_wake,
        test_wk_p,
    )
    wake_result[
        "balanced_acc"
    ] = balanced_acc(
        pred_wake,
        gt_wake,
    )

    n_wake = int(
        (
            ~gt_slp
        ).sum()
    )
    n_pos_wake = int(
        (
            gt_raw
            & ~gt_slp
        ).sum()
    )

    print()
    print(
        line(
            "GT-sleep epochs only ",
            sleep_result,
            arousal_threshold,
        )
    )
    print(
        line(
            "gated by predicted sleep",
            gated_result,
            arousal_threshold,
        )
    )
    print(
        line(
            "all epochs (legacy)   ",
            legacy_result,
            arousal_threshold,
        )
    )

    wake_auc = (
        f'  AUROC {wake_result["auroc"]:.3f}'
        if wake_result[
            "auroc"
        ] is not None
        else ""
    )

    print(
        f"[arousal] wake head @thr={wake_threshold:.2f}: "
        f'acc {wake_result["acc"]:.3f}  '
        f'balanced {wake_result["balanced_acc"]:.3f}  '
        f'F1 {wake_result["f1"]:.3f}'
        f"{wake_auc}  "
        f"[{n_wake}/{len(gt_slp)} epochs are GT wake; "
        f"{n_pos_wake} raw arousal positives fall inside Wake epochs]"
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    label_meta = load_arousal_meta(
        args.arousal_root
    )

    checkpoint = {
        "model_state_dict": (
            head.state_dict()
        ),
        "model_config": (
            head.model_config()
        ),
        "mean": mu,
        "std": sd,
        "arousal_threshold": (
            arousal_threshold
        ),
        "wake_threshold": (
            wake_threshold
        ),
        "pos_weight": (
            pos_weight
        ),
        "training_config": {
            "implementation": (
                "Keondo eval_spec.py arousal probe"
            ),
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": 1e-4,
            "optimizer": "Adam",
            "loss": (
                "BCEWithLogitsLoss"
            ),
            "batch_size_epochs": (
                args.batch_size
            ),
            "probe_train_subjects": (
                len(
                    train.selected_subjects
                )
            ),
            "probe_val_subjects": (
                len(
                    val.selected_subjects
                )
            ),
            "test_subjects": (
                len(
                    test.selected_subjects
                )
            ),
            "seed": args.seed,
        },
    }

    torch.save(
        checkpoint,
        args.output_dir
        / "arousal_linear_probe.pt",
    )

    results = {
        "task": "arousal",
        "implementation": (
            "Keondo-style eval_spec.py arousal protocol adapted to "
            "PaSL 30-s synthesized EEG bins"
        ),
        "input": (
            "PaSL synthesized EEG spectrogram"
        ),
        "feature_dim": 256,
        "spectrogram_frame_sec": 30,
        "arousal_label_sec": (
            AROUSAL_LABEL_SEC
        ),
        "arousal_min_sec": (
            label_meta.get(
                "min_sec"
            )
        ),
        "probe": (
            "Linear(256 -> 2 heads: arousal, wake)"
        ),
        "probe_parameters": (
            count_parameters(
                head
            )
        ),
        "optimizer": "Adam",
        "loss": (
            "BCEWithLogitsLoss"
        ),
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": 1e-4,
        "batch_size_epochs": (
            args.batch_size
        ),
        "arousal_pos_weight": float(
            pos_weight[0]
        ),
        "wake_pos_weight": float(
            pos_weight[1]
        ),
        "arousal_threshold": (
            arousal_threshold
        ),
        "wake_threshold": (
            wake_threshold
        ),
        "arousal_train_subjects": (
            len(
                train.selected_subjects
            )
        ),
        "arousal_val_subjects": (
            len(
                val.selected_subjects
            )
        ),
        "arousal_test_subjects": (
            len(
                test.selected_subjects
            )
        ),
        "arousal_wake_epochs_excluded": (
            n_wake
        ),
        "arousal_positives_in_wake": (
            n_pos_wake
        ),
        "wake_frac_test": float(
            gt_wake.mean()
        ),
    }

    for key in (
        "f1",
        "precision",
        "recall",
        "acc",
        "auroc",
        "epochs",
        "positives",
    ):
        results[
            f"arousal_sleep_{key}"
        ] = sleep_result[key]

        results[
            f"arousal_gated_{key}"
        ] = gated_result[key]

    for key in (
        "f1",
        "precision",
        "recall",
        "acc",
        "auroc",
    ):
        results[
            f"arousal_frame_{key}"
        ] = legacy_result[key]

    results[
        "arousal_epochs_test"
    ] = legacy_result[
        "epochs"
    ]
    results[
        "arousal_epochs_positive"
    ] = legacy_result[
        "positives"
    ]

    for key in (
        "f1",
        "precision",
        "recall",
        "acc",
        "balanced_acc",
        "auroc",
    ):
        results[
            f"wake_{key}"
        ] = wake_result[key]

    results[
        "wake_epochs_test"
    ] = wake_result[
        "epochs"
    ]

    with (
        args.output_dir
        / "results_arousal.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
            allow_nan=True,
        )

    np.savez_compressed(
        args.output_dir
        / "test_predictions.npz",
        subject_id=(
            test.subject_ids.astype(str)
        ),
        arousal_true=(
            gt_raw.astype(np.uint8)
        ),
        wake_true=(
            gt_wake.astype(np.uint8)
        ),
        sleep_true=(
            gt_slp.astype(np.uint8)
        ),
        p_arousal=test_ar_p,
        p_wake=test_wk_p,
        pred_arousal=(
            pred_ar.astype(np.uint8)
        ),
        pred_wake=(
            pred_wake.astype(np.uint8)
        ),
    )

    print()
    print(
        "Results:",
        args.output_dir
        / "results_arousal.json",
    )
    print(
        "Checkpoint:",
        args.output_dir
        / "arousal_linear_probe.pt",
    )


if __name__ == "__main__":
    main()
