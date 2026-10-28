"""
Fit one linear Cox probe on the SHHS1 wav2sleep-heldout comparison split.

Outcomes:
- all-cause death
- CVD death
- incident stroke

This version is self-contained and does not depend on eval_spec.py/train_utils.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------
# Paths / endpoints
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ROSTER = (
    ROOT
    / "data"
    / "shhs_meta_overlap_wav2sleep"
)

DEFAULT_CSV = Path(
    "/hdd2/kdpark/sleep_datasets/"
    "#_SHHS/datasets/"
    "shhs-cvd-summary-dataset-0.20.0.csv"
)

ENDPOINTS = (
    "all_cause_death",
    "cvd_death",
    "stroke",
)


# ---------------------------------------------------------------------
# CoxPH loss
# ---------------------------------------------------------------------

def _cox_ph_loss_1d(
    risk: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
) -> torch.Tensor:
    """
    Negative Cox proportional-hazards partial log likelihood.

    risk:
        Larger value = larger predicted hazard.

    time:
        Follow-up/event time.

    event:
        1 = event observed
        0 = censored

    The risk set for an event at time t contains subjects with T >= t.
    """

    risk = risk.reshape(-1)
    time = time.reshape(-1)
    event = event.reshape(-1) > 0.5

    event_indices = torch.nonzero(
        event,
        as_tuple=False,
    ).flatten()

    # Small virtual batches may contain no event.
    # Keep a differentiable zero so backward() still works.
    if event_indices.numel() == 0:
        return risk.sum() * 0.0

    terms = []

    for idx in event_indices:

        at_risk = time >= time[idx]

        log_risk_sum = torch.logsumexp(
            risk[at_risk],
            dim=0,
        )

        terms.append(
            risk[idx] - log_risk_sum
        )

    return -torch.stack(terms).mean()


# ---------------------------------------------------------------------
# Harrell C-index
# ---------------------------------------------------------------------

def _harrell_c_index(
    risk,
    time,
    event,
) -> float:
    """
    Harrell concordance index.

    Larger predicted risk should correspond to an earlier observed event.

    A subject with an observed event can be compared with subjects whose
    observed follow-up time is strictly greater.
    """

    risk = np.asarray(
        risk,
        dtype=np.float64,
    ).reshape(-1)

    time = np.asarray(
        time,
        dtype=np.float64,
    ).reshape(-1)

    event = np.asarray(
        event,
        dtype=np.float64,
    ).reshape(-1) > 0.5

    if not (
        len(risk)
        == len(time)
        == len(event)
    ):
        raise ValueError(
            "risk/time/event lengths differ"
        )

    concordant = 0.0
    comparable = 0

    for i in range(len(risk)):

        if not event[i]:
            continue

        later = time > time[i]

        n_later = int(
            np.sum(later)
        )

        if n_later == 0:
            continue

        other_risk = risk[later]

        comparable += n_later

        concordant += float(
            np.sum(
                risk[i] > other_risk
            )
        )

        concordant += 0.5 * float(
            np.sum(
                risk[i] == other_risk
            )
        )

    if comparable == 0:
        return float("nan")

    return float(
        concordant / comparable
    )


# ---------------------------------------------------------------------
# NumPy -> Torch
# ---------------------------------------------------------------------

def _tensor(
    array: np.ndarray,
) -> torch.Tensor:

    array = np.ascontiguousarray(
        array,
        dtype=np.float32,
    )

    return torch.from_numpy(array)


# ---------------------------------------------------------------------
# Roster
# ---------------------------------------------------------------------

def read_roster(
    root: Path,
    split: str,
) -> list[str]:

    path = (
        root
        / f"{split}_list.txt"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Roster not found: {path}"
        )

    ids = [
        line.strip().split("/")[-1]
        for line
        in path.read_text().splitlines()
        if line.strip()
    ]

    if len(ids) != len(set(ids)):
        raise ValueError(
            f"duplicate IDs in {split} roster"
        )

    return ids


# ---------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------

def read_features(
    root: Path,
) -> dict[str, np.ndarray]:

    rows: dict[
        str,
        np.ndarray,
    ] = {}

    for split in (
        "train",
        "val",
        "test",
    ):

        paths = sorted(
            root.glob(
                f"{split}_shard*.npz"
            )
        )

        if not paths:
            raise FileNotFoundError(
                f"No {split} feature shards "
                f"found under {root}"
            )

        for path in paths:

            with np.load(path) as data:

                subjects = data[
                    "subjects"
                ]

                features = data[
                    "features"
                ]

                if (
                    len(subjects)
                    != len(features)
                ):
                    raise ValueError(
                        f"row mismatch in {path}"
                    )

                for subject, feature in zip(
                    subjects.tolist(),
                    features,
                ):

                    subject = str(subject)

                    if subject in rows:
                        raise ValueError(
                            "duplicate feature "
                            f"for {subject}"
                        )

                    feature = np.asarray(
                        feature,
                        dtype=np.float32,
                    )

                    if not np.isfinite(
                        feature
                    ).all():
                        raise ValueError(
                            "non-finite feature "
                            f"for {subject}"
                        )

                    rows[subject] = feature

    return rows


# ---------------------------------------------------------------------
# Survival outcome construction
# ---------------------------------------------------------------------

def outcome(
    row: pd.Series,
) -> tuple[
    np.ndarray,
    np.ndarray,
] | None:

    cens = row.get(
        "censdate"
    )

    if (
        pd.isna(cens)
        or float(cens) <= 0
    ):
        return None

    cens = float(cens)

    vital = row.get(
        "vital"
    )

    if (
        pd.isna(vital)
        or float(vital)
        not in (0.0, 1.0)
    ):
        return None

    # SHHS:
    # vital == 0 means death
    # vital == 1 means alive/censored
    events = [
        float(
            float(vital) == 0.0
        )
    ]

    times = [
        cens
    ]

    for (
        event_col,
        time_col,
    ) in (
        (
            "cvd_death",
            "cvd_dthdt",
        ),
        (
            "stroke",
            "stk_date",
        ),
    ):

        raw_event = row.get(
            event_col
        )

        raw_time = row.get(
            time_col
        )

        event = float(
            pd.notna(raw_event)
            and float(raw_event) >= 1.0
        )

        event_time = (
            float(raw_time)
            if pd.notna(raw_time)
            else np.nan
        )

        # Event marked but no valid event time:
        # treat as censored at censdate.
        if (
            not np.isfinite(
                event_time
            )
            or event_time <= 0
        ):
            event = 0.0
            event_time = cens

        events.append(
            event
        )

        times.append(
            event_time
        )

    return (
        np.asarray(
            events,
            dtype=np.float32,
        ),
        np.asarray(
            times,
            dtype=np.float32,
        ),
    )


# ---------------------------------------------------------------------
# Prepare train / val / test
# ---------------------------------------------------------------------

def prepare_splits(
    roster_root: Path,
    feature_root: Path,
    csv_path: Path,
    common_feature_root: Path | None = None,
):

    features = read_features(
        feature_root
    )

    common_subjects = (
        set(
            read_features(
                common_feature_root
            )
        )
        if common_feature_root
        is not None
        else None
    )

    labels = pd.read_csv(
        csv_path
    )

    if "nsrrid" not in labels.columns:
        raise ValueError(
            "CSV has no nsrrid column"
        )

    labels = labels.set_index(
        "nsrrid"
    )

    splits = {}

    for split in (
        "train",
        "val",
        "test",
    ):

        roster = read_roster(
            roster_root,
            split,
        )

        x = []
        events = []
        times = []
        subjects = []

        missing_features = []
        missing_labels = []
        zero_followup = []
        excluded_by_common = []

        for subject in roster:

            if (
                common_subjects
                is not None
                and subject
                not in common_subjects
            ):
                excluded_by_common.append(
                    subject
                )
                continue

            if subject not in features:
                missing_features.append(
                    subject
                )
                continue

            # shhs1-200001 -> 200001
            sid = int(
                subject.rsplit(
                    "-",
                    1,
                )[-1]
            )

            if sid not in labels.index:
                missing_labels.append(
                    subject
                )
                continue

            row = labels.loc[sid]

            # Defensive check in case CSV contains duplicate nsrrid rows.
            if isinstance(
                row,
                pd.DataFrame,
            ):
                row = row.iloc[0]

            target = outcome(
                row
            )

            if target is None:
                zero_followup.append(
                    subject
                )
                continue

            subjects.append(
                subject
            )

            x.append(
                features[subject]
            )

            events.append(
                target[0]
            )

            times.append(
                target[1]
            )

        if not x:
            raise RuntimeError(
                f"No analyzable subjects "
                f"for split={split}"
            )

        splits[split] = {
            "subjects": subjects,
            "features": np.stack(
                x
            ).astype(
                np.float32
            ),
            "event": np.stack(
                events
            ).astype(
                np.float32
            ),
            "time": np.stack(
                times
            ).astype(
                np.float32
            ),
            "roster_n": len(
                roster
            ),
            "missing_features": (
                missing_features
            ),
            "missing_labels": (
                missing_labels
            ),
            "zero_followup": (
                zero_followup
            ),
            "excluded_by_common": (
                excluded_by_common
            ),
        }

    # No overlap between split subjects.
    for a, b in (
        (
            "train",
            "val",
        ),
        (
            "train",
            "test",
        ),
        (
            "val",
            "test",
        ),
    ):

        overlap = (
            set(
                splits[a][
                    "subjects"
                ]
            )
            & set(
                splits[b][
                    "subjects"
                ]
            )
        )

        if overlap:
            raise ValueError(
                f"{a}/{b} overlap: "
                f"{len(overlap)} subjects"
            )

    dims = {
        row[
            "features"
        ].shape[1]
        for row
        in splits.values()
    }

    if len(dims) != 1:
        raise ValueError(
            "inconsistent feature "
            f"dimensions: {dims}"
        )

    return splits


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--feature_dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--model_name",
        required=True,
    )

    parser.add_argument(
        "--roster_root",
        type=Path,
        default=DEFAULT_ROSTER,
    )

    parser.add_argument(
        "--survival_csv",
        type=Path,
        default=DEFAULT_CSV,
    )

    parser.add_argument(
        "--common_feature_dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--test_exclude_ids_file",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--beta2",
        type=float,
        default=0.99,
    )

    parser.add_argument(
        "--batch_size_per_device",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--world_size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--feature_normalization",
        choices=(
            "none",
            "train_zscore",
        ),
        default="none",
    )

    args = parser.parse_args()

    # --------------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------------

    torch.set_num_threads(
        2
    )

    torch.manual_seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    device = torch.device(
        args.device
    )

    # --------------------------------------------------------------
    # Data
    # --------------------------------------------------------------

    data = prepare_splits(
        args.roster_root,
        args.feature_dir,
        args.survival_csv,
        args.common_feature_dir,
    )

    print()
    print(
        "=" * 70
    )
    print(
        "SURVIVAL DATA"
    )
    print(
        "=" * 70
    )

    for split, row in data.items():

        print(
            f"{split.upper():5s} "
            f"roster={row['roster_n']}  "
            f"analyzable={len(row['subjects'])}  "
            f"missing_features="
            f"{len(row['missing_features'])}  "
            f"missing_labels="
            f"{len(row['missing_labels'])}  "
            f"zero_followup="
            f"{len(row['zero_followup'])}  "
            f"excluded_common="
            f"{len(row['excluded_by_common'])}"
        )

        print(
            "       events:",
            {
                ENDPOINTS[j]: int(
                    row[
                        "event"
                    ][:, j].sum()
                )
                for j
                in range(
                    len(
                        ENDPOINTS
                    )
                )
            },
        )

    print()
    print(
        "Feature dimension:",
        data[
            "train"
        ][
            "features"
        ].shape[1],
    )

    # --------------------------------------------------------------
    # Optional normalization
    # --------------------------------------------------------------

    if (
        args.feature_normalization
        == "train_zscore"
    ):

        mu = (
            data[
                "train"
            ][
                "features"
            ].mean(
                axis=0,
                keepdims=True,
            )
        )

        sd = (
            data[
                "train"
            ][
                "features"
            ].std(
                axis=0,
                keepdims=True,
            )
        )

        sd = np.clip(
            sd,
            1e-6,
            None,
        )

    else:

        mu = 0.0
        sd = 1.0

    for row in data.values():

        row["x_t"] = _tensor(
            (
                row[
                    "features"
                ]
                - mu
            )
            / sd
        ).to(
            device
        )

        row["event_t"] = _tensor(
            row[
                "event"
            ]
        ).to(
            device
        )

        row["time_t"] = _tensor(
            row[
                "time"
            ]
        ).to(
            device
        )

    # --------------------------------------------------------------
    # Linear Cox head
    # --------------------------------------------------------------

    feature_dim = (
        data[
            "train"
        ][
            "features"
        ].shape[1]
    )

    head = torch.nn.Linear(
        feature_dim,
        len(
            ENDPOINTS
        ),
    ).to(
        device
    )

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.lr,
        betas=(
            0.9,
            args.beta2,
        ),
        weight_decay=(
            args.weight_decay
        ),
    )

    global_batch = (
        args.batch_size_per_device
        * args.world_size
    )

    n_train = len(
        data[
            "train"
        ][
            "subjects"
        ]
    )

    steps_per_epoch = int(
        np.ceil(
            n_train
            / global_batch
        )
    )

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=(
                steps_per_epoch
                * args.epochs
                + 1
            ),
            eta_min=1e-8,
        )
    )

    print()
    print(
        "=" * 70
    )
    print(
        "TRAIN COX PROBE"
    )
    print(
        "=" * 70
    )
    print(
        f"Linear({feature_dim} -> "
        f"{len(ENDPOINTS)})"
    )
    print(
        f"epochs={args.epochs}"
    )
    print(
        f"lr={args.lr}"
    )
    print(
        "batch_size_per_device="
        f"{args.batch_size_per_device}"
    )
    print(
        f"virtual_world_size="
        f"{args.world_size}"
    )
    print(
        f"global_batch="
        f"{global_batch}"
    )
    print(
        "feature_normalization="
        f"{args.feature_normalization}"
    )
    print()

    best_score = -float(
        "inf"
    )
    best_epoch = None
    best_state = None

    # --------------------------------------------------------------
    # Train
    # --------------------------------------------------------------

    for epoch in range(
        args.epochs
    ):

        head.train()

        permutation = (
            np.random.permutation(
                n_train
            )
        )

        epoch_losses = []

        for start in range(
            0,
            n_train,
            global_batch,
        ):

            group = permutation[
                start:
                start
                + global_batch
            ]

            rank_losses = []

            for rank in range(
                args.world_size
            ):

                lo = (
                    rank
                    * args.batch_size_per_device
                )

                hi = (
                    (rank + 1)
                    * args.batch_size_per_device
                )

                indices = group[
                    lo:hi
                ]

                if len(
                    indices
                ) == 0:
                    continue

                idx = torch.as_tensor(
                    indices,
                    device=device,
                    dtype=torch.long,
                )

                risk = head(
                    data[
                        "train"
                    ][
                        "x_t"
                    ][
                        idx
                    ]
                )

                endpoint_losses = []

                for j in range(
                    len(
                        ENDPOINTS
                    )
                ):

                    endpoint_losses.append(
                        _cox_ph_loss_1d(
                            risk[
                                :, j
                            ],
                            data[
                                "train"
                            ][
                                "time_t"
                            ][
                                idx,
                                j
                            ],
                            data[
                                "train"
                            ][
                                "event_t"
                            ][
                                idx,
                                j
                            ],
                        )
                    )

                rank_losses.append(
                    torch.stack(
                        endpoint_losses
                    ).mean()
                )

            if not rank_losses:
                continue

            loss = torch.stack(
                rank_losses
            ).mean()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            scheduler.step()

            epoch_losses.append(
                float(
                    loss.detach().cpu()
                )
            )

        # ----------------------------------------------------------
        # Validation C-index
        # ----------------------------------------------------------

        head.eval()

        with torch.inference_mode():

            val_risk = (
                head(
                    data[
                        "val"
                    ][
                        "x_t"
                    ]
                )
                .cpu()
                .numpy()
            )

        val_scores = []

        for j, name in enumerate(
            ENDPOINTS
        ):

            score = _harrell_c_index(
                val_risk[
                    :, j
                ],
                data[
                    "val"
                ][
                    "time"
                ][
                    :, j
                ],
                data[
                    "val"
                ][
                    "event"
                ][
                    :, j
                ],
            )

            val_scores.append(
                score
            )

        val_macro = float(
            np.nanmean(
                val_scores
            )
        )

        mean_loss = (
            float(
                np.mean(
                    epoch_losses
                )
            )
            if epoch_losses
            else float(
                "nan"
            )
        )

        print(
            f"Epoch "
            f"{epoch + 1}/{args.epochs}"
            f" | loss={mean_loss:.6f}"
            f" | val macro C-index="
            f"{val_macro:.5f}"
            f" | all-cause="
            f"{val_scores[0]:.5f}"
            f" | CVD="
            f"{val_scores[1]:.5f}"
            f" | stroke="
            f"{val_scores[2]:.5f}",
            flush=True,
        )

        if (
            np.isfinite(
                val_macro
            )
            and val_macro
            > best_score
        ):

            best_score = (
                val_macro
            )

            best_epoch = (
                epoch + 1
            )

            best_state = {
                key: value
                .detach()
                .cpu()
                .clone()
                for key, value
                in head
                .state_dict()
                .items()
            }

    if best_state is None:
        raise RuntimeError(
            "No valid validation "
            "C-index was obtained."
        )

    # --------------------------------------------------------------
    # Restore best epoch
    # --------------------------------------------------------------

    head.load_state_dict(
        best_state
    )

    head.eval()

    # --------------------------------------------------------------
    # Test
    # --------------------------------------------------------------

    with torch.inference_mode():

        test_risk = (
            head(
                data[
                    "test"
                ][
                    "x_t"
                ]
            )
            .cpu()
            .numpy()
        )

    endpoints = {}

    for j, name in enumerate(
        ENDPOINTS
    ):

        cindex = float(
            _harrell_c_index(
                test_risk[
                    :, j
                ],
                data[
                    "test"
                ][
                    "time"
                ][
                    :, j
                ],
                data[
                    "test"
                ][
                    "event"
                ][
                    :, j
                ],
            )
        )

        endpoints[
            name
        ] = {
            "cindex": cindex,
            "events": {
                split: int(
                    row[
                        "event"
                    ][
                        :, j
                    ].sum()
                )
                for split, row
                in data.items()
            },
        }

    macro_cindex = float(
        np.nanmean(
            [
                result[
                    "cindex"
                ]
                for result
                in endpoints.values()
            ]
        )
    )

    # --------------------------------------------------------------
    # Results JSON
    # --------------------------------------------------------------

    result = {
        "model_name": (
            args.model_name
        ),
        "feature_dir": str(
            args.feature_dir.resolve()
        ),
        "roster_root": str(
            args.roster_root.resolve()
        ),
        "survival_csv": str(
            args.survival_csv.resolve()
        ),
        "endpoints": (
            endpoints
        ),
        "macro_cindex": (
            macro_cindex
        ),
        "splits": {
            split: {
                "roster_n": (
                    row[
                        "roster_n"
                    ]
                ),
                "analyzable_n": len(
                    row[
                        "subjects"
                    ]
                ),
                "missing_features": (
                    row[
                        "missing_features"
                    ]
                ),
                "missing_labels": (
                    row[
                        "missing_labels"
                    ]
                ),
                "zero_followup": (
                    row[
                        "zero_followup"
                    ]
                ),
                "excluded_by_common": (
                    row[
                        "excluded_by_common"
                    ]
                ),
            }
            for split, row
            in data.items()
        },
        "common_feature_dir": (
            str(
                args
                .common_feature_dir
                .resolve()
            )
            if args.common_feature_dir
            else None
        ),
        "head": (
            "linear Cox; mean loss "
            "over three endpoints "
            "and two virtual ranks"
        ),
        "seed": (
            args.seed
        ),
        "best_epoch": (
            best_epoch
        ),
        "best_val_macro_cindex": (
            best_score
        ),
        "epochs": (
            args.epochs
        ),
        "lr": (
            args.lr
        ),
        "weight_decay": (
            args.weight_decay
        ),
        "beta2": (
            args.beta2
        ),
        "batch_size_per_device": (
            args.batch_size_per_device
        ),
        "world_size": (
            args.world_size
        ),
        "feature_normalization": (
            args.feature_normalization
        ),
        "age_sex_input": False,
    }

    # --------------------------------------------------------------
    # Optional sensitivity exclusion
    # --------------------------------------------------------------

    if (
        args.test_exclude_ids_file
        is not None
    ):

        excluded = {
            line.strip()
            for line
            in args
            .test_exclude_ids_file
            .read_text()
            .splitlines()
            if line.strip()
        }

        subjects = data[
            "test"
        ][
            "subjects"
        ]

        mask = np.asarray(
            [
                subject
                not in excluded
                for subject
                in subjects
            ]
        )

        if (
            not mask.any()
            or mask.all()
        ):
            raise ValueError(
                "test exclusion must "
                "remove some, but not "
                "all, subjects"
            )

        result[
            "test_exclusion_sensitivity"
        ] = {
            "excluded_ids_file": str(
                args
                .test_exclude_ids_file
                .resolve()
            ),
            "excluded_n": int(
                (~mask).sum()
            ),
            "retained_n": int(
                mask.sum()
            ),
            "endpoints": {
                name: {
                    "cindex": float(
                        _harrell_c_index(
                            test_risk[
                                mask,
                                j
                            ],
                            data[
                                "test"
                            ][
                                "time"
                            ][
                                mask,
                                j
                            ],
                            data[
                                "test"
                            ][
                                "event"
                            ][
                                mask,
                                j
                            ],
                        )
                    ),
                    "events": int(
                        data[
                            "test"
                        ][
                            "event"
                        ][
                            mask,
                            j
                        ].sum()
                    ),
                }
                for j, name
                in enumerate(
                    ENDPOINTS
                )
            },
        }

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            result,
            indent=2,
        )
    )

    # --------------------------------------------------------------
    # Final display
    # --------------------------------------------------------------

    print()
    print(
        "=" * 70
    )
    print(
        "FINAL TEST RESULTS"
    )
    print(
        "=" * 70
    )

    summary = {
        "test_n": len(
            data[
                "test"
            ][
                "subjects"
            ]
        ),
        "best_epoch": (
            best_epoch
        ),
        "best_val_macro_cindex": (
            best_score
        ),
        "events": {
            name: values[
                "events"
            ][
                "test"
            ]
            for name, values
            in endpoints.items()
        },
        "cindex": {
            name: values[
                "cindex"
            ]
            for name, values
            in endpoints.items()
        },
        "macro": (
            macro_cindex
        ),
    }

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print()
    print(
        "[saved]",
        args.output,
    )


if __name__ == "__main__":
    main()
