from __future__ import annotations

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
)


IGNORE_INDEX = -100

CLASS_NAMES = (
    "Wake",
    "Light",
    "Deep",
    "REM",
)


def flatten_valid_predictions(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    logits:
        (B, 512, 4)

    labels:
        (B, 512)

    Returns
    -------
    y_true, y_pred:
        Flattened valid 30-second epochs only.
    """

    if logits.ndim != 3:
        raise ValueError(
            f"Expected logits (B,T,C), got {tuple(logits.shape)}"
        )

    if labels.ndim != 2:
        raise ValueError(
            f"Expected labels (B,T), got {tuple(labels.shape)}"
        )

    pred = logits.argmax(
        dim=-1
    )

    mask = (
        labels != ignore_index
    )

    y_true = (
        labels[mask]
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64)
    )

    y_pred = (
        pred[mask]
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64)
    )

    return y_true, y_pred


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Compute pooled epoch-level sleep staging metrics.
    """

    y_true = np.asarray(
        y_true,
        dtype=np.int64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.int64,
    )

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: "
            f"{y_true.shape} vs {y_pred.shape}"
        )

    if y_true.size == 0:
        raise ValueError(
            "No valid sleep-stage labels."
        )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3],
        average="macro",
        zero_division=0,
    )

    kappa = cohen_kappa_score(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3],
    )

    f1_per_class = f1_score(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3],
        average=None,
        zero_division=0,
    )

    result = {
        "num_epochs": int(
            y_true.size
        ),
        "accuracy": float(
            accuracy
        ),
        "macro_f1": float(
            macro_f1
        ),
        "kappa": float(
            kappa
        ),
    }

    for idx, name in enumerate(
        CLASS_NAMES
    ):
        result[
            f"f1_{name.lower()}"
        ] = float(
            f1_per_class[idx]
        )

    return result


class MetricAccumulator:
    """
    Accumulates predictions over an entire validation/test set.
    """

    def __init__(
        self,
        *,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:

        self.ignore_index = int(
            ignore_index
        )

        self.true_parts = []
        self.pred_parts = []

    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:

        y_true, y_pred = (
            flatten_valid_predictions(
                logits,
                labels,
                ignore_index=self.ignore_index,
            )
        )

        self.true_parts.append(
            y_true
        )

        self.pred_parts.append(
            y_pred
        )

    def compute(
        self,
    ) -> dict:

        if not self.true_parts:
            raise RuntimeError(
                "MetricAccumulator is empty."
            )

        y_true = np.concatenate(
            self.true_parts
        )

        y_pred = np.concatenate(
            self.pred_parts
        )

        return compute_metrics(
            y_true,
            y_pred,
        )

    def reset(
        self,
    ) -> None:

        self.true_parts.clear()
        self.pred_parts.clear()
