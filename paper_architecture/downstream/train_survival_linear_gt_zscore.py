from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


FEATURE_ROOT = Path(
    "outputs/downstream_survival_features_gt"
)

LABEL_ROOT = Path(
    "outputs/downstream_survival_labels"
)

OUTPUT_DIR = Path(
    "outputs/survival_linear_gt_zscore"
)

OUTCOMES = [
    "angina",
    "cvd_death",
    "chf",
    "chd_death",
    "mi",
    "stroke",
]

SEED = 42
EPOCHS = 20
BATCH_SIZE = 8
LR = 1e-2
WEIGHT_DECAY = 1e-4
ETA_MIN = 1e-8


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SurvivalLinear(nn.Module):
    """
    One subject-level feature vector -> six Cox risk scores.
    """

    def __init__(
        self,
        input_dim: int = 256,
        num_outcomes: int = 6,
    ) -> None:
        super().__init__()

        self.head = nn.Linear(
            input_dim,
            num_outcomes,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.head(x)


def cox_ph_loss(
    hazards: torch.Tensor,
    event_times: torch.Tensor,
    is_event: torch.Tensor,
) -> torch.Tensor:
    """
    Same multi-outcome Cox PH formulation used by SleepMaMi.

    hazards     : (B, 6)
    event_times : (B, 6)
    is_event    : (B, 6)
    """

    sorted_times, sorted_idx = (
        event_times.sort(
            dim=0,
            descending=True,
        )
    )

    hazards = torch.gather(
        hazards,
        dim=0,
        index=sorted_idx,
    )

    is_event = torch.gather(
        is_event,
        dim=0,
        index=sorted_idx,
    )

    log_cumulative_hazard = (
        torch.logcumsumexp(
            hazards.float(),
            dim=0,
        )
    )

    losses = -(
        hazards
        - log_cumulative_hazard
    ) * is_event

    label_loss = (
        losses.sum(dim=0)
        / (
            is_event.sum(dim=0)
            + 1e-9
        )
    )

    return label_loss.mean()


def harrell_cindex(
    risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
) -> float:
    """
    Harrell C-index.

    Higher model score = higher risk = expected earlier event.
    """

    n = len(risk)

    i, j = np.triu_indices(
        n,
        k=1,
    )

    ti = time[i]
    tj = time[j]

    ei = event[i].astype(bool)
    ej = event[j].astype(bool)

    ri = risk[i]
    rj = risk[j]

    # i had an observed event before j
    case_i = (
        (ti < tj)
        & ei
    )

    # j had an observed event before i
    case_j = (
        (tj < ti)
        & ej
    )

    # Same follow-up time:
    # informative only when exactly one is an event.
    tied_time = (
        (ti == tj)
        & (ei != ej)
    )

    expected_i = (
        case_i
        | (
            tied_time
            & ei
        )
    )

    expected_j = (
        case_j
        | (
            tied_time
            & ej
        )
    )

    valid = (
        expected_i
        | expected_j
    )

    if valid.sum() == 0:
        return 0.5

    higher_correct = np.zeros(
        len(i),
        dtype=np.float64,
    )

    higher_correct[
        expected_i
    ] = (
        ri[expected_i]
        > rj[expected_i]
    )

    higher_correct[
        expected_j
    ] = (
        rj[expected_j]
        > ri[expected_j]
    )

    predicted_tie = (
        ri == rj
    )

    score = (
        higher_correct[valid].sum()
        + 0.5
        * predicted_tie[valid].sum()
    )

    return float(
        score / valid.sum()
    )


def load_split(
    split: str,
):
    features = np.load(
        FEATURE_ROOT
        / f"{split}_features.npy"
    ).astype(
        np.float32
    )

    with (
        FEATURE_ROOT
        / f"{split}_subjects.txt"
    ).open(
        "r",
        encoding="utf-8",
    ) as f:
        subjects = [
            line.strip()
            for line in f
            if line.strip()
        ]

    label_path = (
        LABEL_ROOT
        / f"{split}.csv"
    )

    labels = {}

    with label_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            labels[
                row["subject_id"]
            ] = row

    if len(subjects) != len(features):
        raise RuntimeError(
            f"{split}: subject/features mismatch"
        )

    missing = [
        subject
        for subject in subjects
        if subject not in labels
    ]

    if missing:
        print(
            f"{split}: excluding "
            f"{len(missing)} subjects "
            f"without survival labels:"
        )
        for subject in missing:
            print("  ", subject)

    keep = [
        i
        for i, subject in enumerate(subjects)
        if subject in labels
    ]

    features = features[keep]

    subjects = [
        subjects[i]
        for i in keep
    ]

    events = np.zeros(
        (
            len(subjects),
            len(OUTCOMES),
        ),
        dtype=np.float32,
    )

    times = np.zeros_like(
        events
    )

    for subject_index, subject in enumerate(
        subjects
    ):
        row = labels[subject]

        for outcome_index, outcome in enumerate(
            OUTCOMES
        ):
            events[
                subject_index,
                outcome_index,
            ] = float(
                row[
                    f"{outcome}_event"
                ]
            )

            times[
                subject_index,
                outcome_index,
            ] = float(
                row[
                    f"{outcome}_time"
                ]
            )

    return (
        features,
        times,
        events,
        subjects,
    )


@torch.no_grad()
def predict(
    model: nn.Module,
    features: np.ndarray,
    device: torch.device,
) -> np.ndarray:

    model.eval()

    x = torch.from_numpy(
        features
    ).to(device)

    risk = model(x)

    return (
        risk
        .float()
        .cpu()
        .numpy()
    )


def compute_cindices(
    risk: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
):
    result = {}

    for k, outcome in enumerate(
        OUTCOMES
    ):
        result[outcome] = (
            harrell_cindex(
                risk=risk[:, k],
                time=times[:, k],
                event=events[:, k],
            )
        )

    return result


def print_cindices(
    title: str,
    metrics: dict,
) -> None:

    print()
    print(title)
    print("-" * 50)

    for outcome in OUTCOMES:
        print(
            f"{outcome:12s}: "
            f"{metrics[outcome]:.4f}"
        )

    print(
        f"{'MEAN':12s}: "
        f"{np.mean(list(metrics.values())):.4f}"
    )


def main():
    seed_everything(
        SEED
    )

    OUTPUT_DIR.mkdir(
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

    (
        x_train,
        t_train,
        e_train,
        train_subjects,
    ) = load_split(
        "train"
    )

    (
        x_val,
        t_val,
        e_val,
        val_subjects,
    ) = load_split(
        "val"
    )

    (
        x_test,
        t_test,
        e_test,
        test_subjects,
    ) = load_split(
        "test"
    )

    # ------------------------------------------------------------
    # Train-only per-frequency standardization
    # ------------------------------------------------------------
    feature_mean = x_train.mean(
        axis=0,
        keepdims=True,
        dtype=np.float64,
    ).astype(np.float32)

    feature_std = x_train.std(
        axis=0,
        keepdims=True,
        dtype=np.float64,
    ).astype(np.float32)

    feature_std = np.maximum(
        feature_std,
        1e-6,
    )

    x_train = (
        (x_train - feature_mean)
        / feature_std
    ).astype(np.float32)

    x_val = (
        (x_val - feature_mean)
        / feature_std
    ).astype(np.float32)

    x_test = (
        (x_test - feature_mean)
        / feature_std
    ).astype(np.float32)

    print()
    print("Feature standardization: train-only per-frequency z-score")
    print("Train feature mean:", float(x_train.mean()))
    print("Train feature std :", float(x_train.std()))
    print("Val feature mean  :", float(x_val.mean()))
    print("Test feature mean :", float(x_test.mean()))

    print(
        "Train:",
        x_train.shape,
    )
    print(
        "Val  :",
        x_val.shape,
    )
    print(
        "Test :",
        x_test.shape,
    )

    dataset = TensorDataset(
        torch.from_numpy(
            x_train
        ),
        torch.from_numpy(
            t_train
        ),
        torch.from_numpy(
            e_train
        ),
    )

    generator = torch.Generator()
    generator.manual_seed(
        SEED
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        generator=generator,
    )

    model = SurvivalLinear().to(
        device
    )

    print(
        "Trainable parameters:",
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        ),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=(
            0.9,
            0.99,
        ),
        weight_decay=WEIGHT_DECAY,
    )

    total_steps = (
        EPOCHS
        * len(loader)
    )

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=ETA_MIN,
        )
    )

    best_val_cindex = -float(
        "inf"
    )

    best_epoch = None

    print()
    print(
        "=" * 72
    )
    print(
        "SURVIVAL LINEAR PROBE"
    )
    print(
        "=" * 72
    )
    print(
        "Input        : 256"
    )
    print(
        "Outputs      : 6 risk scores"
    )
    print(
        "Loss         : Cox PH"
    )
    print(
        "Epochs       :",
        EPOCHS,
    )
    print(
        "Batch size   :",
        BATCH_SIZE,
        "subjects",
    )
    print(
        "Initial LR   :",
        LR,
    )
    print(
        "Final LR     :",
        ETA_MIN,
    )
    print(
        "Weight decay :",
        WEIGHT_DECAY,
    )

    for epoch in range(
        EPOCHS
    ):
        model.train()

        total_loss = 0.0
        batches = 0

        for (
            features,
            event_times,
            is_event,
        ) in loader:

            features = features.to(
                device
            )

            event_times = (
                event_times.to(
                    device
                )
            )

            is_event = is_event.to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            hazards = model(
                features
            )

            loss = cox_ph_loss(
                hazards,
                event_times,
                is_event,
            )

            loss.backward()

            optimizer.step()
            scheduler.step()

            total_loss += (
                loss.item()
            )

            batches += 1

        val_risk = predict(
            model,
            x_val,
            device,
        )

        val_metrics = (
            compute_cindices(
                val_risk,
                t_val,
                e_val,
            )
        )

        mean_val = float(
            np.mean(
                list(
                    val_metrics.values()
                )
            )
        )

        print()
        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
            f" | train loss="
            f"{total_loss / batches:.6f}"
            f" | lr="
            f"{optimizer.param_groups[0]['lr']:.3e}"
            f" | val mean C-index="
            f"{mean_val:.4f}"
        )

        for outcome in OUTCOMES:
            print(
                f"  {outcome:12s} "
                f"{val_metrics[outcome]:.4f}"
            )

        if mean_val > best_val_cindex:
            best_val_cindex = (
                mean_val
            )

            best_epoch = (
                epoch + 1
            )

            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "val_cindex": (
                        val_metrics
                    ),
                    "mean_val_cindex": (
                        mean_val
                    ),
                    "input_dim": 256,
                    "num_outcomes": 6,
                    "outcomes": OUTCOMES,
                },
                OUTPUT_DIR
                / "checkpoint_best.pt",
            )

    print()
    print(
        "=" * 72
    )
    print(
        "BEST VALIDATION CHECKPOINT"
    )
    print(
        "=" * 72
    )
    print(
        "Epoch:",
        best_epoch,
    )
    print(
        "Mean val C-index:",
        f"{best_val_cindex:.4f}",
    )

    checkpoint = torch.load(
        OUTPUT_DIR
        / "checkpoint_best.pt",
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    test_risk = predict(
        model,
        x_test,
        device,
    )

    test_metrics = (
        compute_cindices(
            test_risk,
            t_test,
            e_test,
        )
    )

    print_cindices(
        "FINAL TEST C-INDEX",
        test_metrics,
    )

    result = {
        "best_epoch": (
            best_epoch
        ),
        "best_val_mean_cindex": (
            best_val_cindex
        ),
        "test_cindex": (
            test_metrics
        ),
        "test_mean_cindex": float(
            np.mean(
                list(
                    test_metrics.values()
                )
            )
        ),
        "train_subjects": (
            len(train_subjects)
        ),
        "val_subjects": (
            len(val_subjects)
        ),
        "test_subjects": (
            len(test_subjects)
        ),
        "settings": {
            "input_dim": 256,
            "outputs": 6,
            "epochs": EPOCHS,
            "batch_size": (
                BATCH_SIZE
            ),
            "lr": LR,
            "weight_decay": (
                WEIGHT_DECAY
            ),
            "optimizer": "AdamW",
            "betas": [
                0.9,
                0.99,
            ],
            "scheduler": (
                "CosineAnnealingLR"
            ),
            "eta_min": ETA_MIN,
            "loss": "Cox PH",
            "seed": SEED,
        },
    }

    with (
        OUTPUT_DIR
        / "test_metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print()
    print(
        "Saved:",
        OUTPUT_DIR
        / "test_metrics.json",
    )


if __name__ == "__main__":
    main()
