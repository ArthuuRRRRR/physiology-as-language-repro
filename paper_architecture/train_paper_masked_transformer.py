"""Train the paper-oriented respiration-to-EEG masked Transformer.

Multi-dataset version.

Inputs are already prepared:
    respiration : (N, 64, 2400) float32
    EEG tokens  : (N, 8, 64)     uint16

EEG tokenization is NOT performed during Transformer training.
The frozen multi-dataset VQGAN was already used during dataset construction.

Architecture:
    - respiration segments projected 2400 -> 768
    - 8 encoder blocks
    - 8 decoder blocks
    - 8 attention heads
    - 8192 EEG-token vocabulary
    - MAGE-style variable masking by default
    - optional fixed high-mask training ablation
    - masked-token cross entropy

Optimization follows the Physiology-as-Language reproduction:
    - AdamW
    - peak LR 1.125e-4
    - weight decay 0.05
    - 40 warm-up epochs
    - cosine decay
    - effective batch size 192 by default
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import (
    DataLoader,
    Dataset,
    Subset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from paper_architecture import (
    PaperMaskedRespirationToEEGTransformer,
)


DEFAULT_DATA_ROOT = Path(
    "outputs/paper_transformer_data"
)

DEFAULT_OUTPUT_DIR = Path(
    "outputs/"
    "paper_masked_transformer_multidataset"
)

ARCHITECTURE_NAME = (
    "paper_masked_transformer_"
    "mage_reproduction_multidataset"
)

DATA_FORMAT = (
    "precomputed_respiration_and_"
    "vqgan_tokens_mmap_v1"
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# Dataset
# ============================================================

class PaperTransformerMMapDataset(
    Dataset
):
    """
    Dataset backed directly by the mmap files created by
    build_transformer_dataset.py.

    Each example:
        respiration : (64, 2400) float32
        eeg_tokens  : (8, 64) int64
    """

    def __init__(
        self,
        data_root: Path,
        split: str,
    ) -> None:
        super().__init__()

        self.data_root = Path(
            data_root
        )

        self.split = split

        self.split_dir = (
            self.data_root
            / split
        )

        metadata_path = (
            self.split_dir
            / "metadata.json"
        )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata not found: "
                f"{metadata_path}"
            )

        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            self.metadata = json.load(
                file
            )

        self.num_samples = int(
            self.metadata[
                "num_samples"
            ]
        )

        resp_info = self.metadata[
            "respiration"
        ]

        token_info = self.metadata[
            "eeg_tokens"
        ]

        self.resp_shape = tuple(
            int(value)
            for value
            in resp_info["shape"]
        )

        self.token_shape = tuple(
            int(value)
            for value
            in token_info["shape"]
        )

        expected_resp_shape = (
            self.num_samples,
            64,
            2400,
        )

        expected_token_shape = (
            self.num_samples,
            8,
            64,
        )

        if (
            self.resp_shape
            != expected_resp_shape
        ):
            raise ValueError(
                "Unexpected respiration "
                f"shape for {split}: "
                f"{self.resp_shape}. "
                f"Expected "
                f"{expected_resp_shape}."
            )

        if (
            self.token_shape
            != expected_token_shape
        ):
            raise ValueError(
                "Unexpected EEG-token "
                f"shape for {split}: "
                f"{self.token_shape}. "
                f"Expected "
                f"{expected_token_shape}."
            )

        self.resp_path = (
            self.split_dir
            / resp_info["path"]
        )

        self.token_path = (
            self.split_dir
            / token_info["path"]
        )

        if not self.resp_path.exists():
            raise FileNotFoundError(
                self.resp_path
            )

        if not self.token_path.exists():
            raise FileNotFoundError(
                self.token_path
            )

        self._respiration = None
        self._tokens = None

    def __len__(self) -> int:
        return self.num_samples

    def _ensure_open(self) -> None:
        """
        Open memmaps lazily.

        This is safer when DataLoader uses multiple workers.
        """

        if self._respiration is None:
            self._respiration = (
                np.memmap(
                    self.resp_path,
                    dtype="float32",
                    mode="r",
                    shape=self.resp_shape,
                )
            )

        if self._tokens is None:
            self._tokens = (
                np.memmap(
                    self.token_path,
                    dtype="uint16",
                    mode="r",
                    shape=self.token_shape,
                )
            )

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Tensor]:

        self._ensure_open()

        # Explicit copies avoid writable-memmap warnings
        # when converting NumPy arrays to Torch tensors.

        respiration = np.array(
            self._respiration[index],
            dtype=np.float32,
            copy=True,
        )

        eeg_tokens = np.array(
            self._tokens[index],
            dtype=np.int64,
            copy=True,
        )

        return {
            "respiration": (
                torch.from_numpy(
                    respiration
                )
            ),
            "eeg_tokens": (
                torch.from_numpy(
                    eeg_tokens
                )
            ),
        }

    def __getstate__(self):
        """
        Prevent an already-open mmap object from being
        propagated into DataLoader worker processes.
        """

        state = self.__dict__.copy()

        state["_respiration"] = None
        state["_tokens"] = None

        return state


# ============================================================
# Respiration normalization
# ============================================================

def standardize_respiration(
    respiration: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """
    Normalize once over the complete 256-minute window.

    This preserves the behavior of the previous
    paper reproduction training pipeline.
    """

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


# ============================================================
# Scheduler
# ============================================================

def create_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int,
):
    """
    Linear warm-up followed by cosine decay.
    """

    if not (
        0
        <= warmup_epochs
        < epochs
    ):
        raise ValueError(
            "warmup_epochs must be "
            "in [0, epochs)"
        )

    def multiplier(
        epoch_index: int,
    ) -> float:

        if (
            warmup_epochs > 0
            and epoch_index
            < warmup_epochs
        ):
            return (
                epoch_index + 1
            ) / warmup_epochs

        decay_epochs = max(
            epochs
            - warmup_epochs,
            1,
        )

        progress = (
            epoch_index
            - warmup_epochs
        ) / decay_epochs

        progress = min(
            max(
                progress,
                0.0,
            ),
            1.0,
        )

        return (
            0.5
            * (
                1.0
                + math.cos(
                    math.pi
                    * progress
                )
            )
        )

    return (
        torch.optim.lr_scheduler
        .LambdaLR(
            optimizer,
            lr_lambda=multiplier,
        )
    )


# ============================================================
# One epoch
# ============================================================

def run_epoch(
    model: (
        PaperMaskedRespirationToEEGTransformer
    ),
    loader: DataLoader,
    device: torch.device,
    optimizer: (
        torch.optim.Optimizer
        | None
    ) = None,
    scaler=None,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    max_batches: int | None = None,
    log_every: int = 200,
    training_mask_ratio: float | None = None,
) -> dict[str, float | int]:

    training = (
        optimizer is not None
    )

    model.train(
        training
    )

    if training:
        optimizer.zero_grad(
            set_to_none=True
        )

    batches_to_run = len(
        loader
    )

    if max_batches is not None:
        batches_to_run = min(
            batches_to_run,
            max_batches,
        )

    total_loss = 0.0
    total_correct = 0
    total_top5 = 0
    total_positions = 0

    total_mask_ratio = 0.0

    processed_batches = 0

    successful_updates = 0
    skipped_updates = 0

    predicted_codes_seen = (
        torch.zeros(
            model.codebook_size,
            dtype=torch.bool,
            device=device,
        )
    )

    target_codes_seen = (
        torch.zeros_like(
            predicted_codes_seen
        )
    )

    for batch_index, batch in enumerate(
        loader,
        start=1,
    ):
        if (
            batch_index
            > batches_to_run
        ):
            break

        respiration = (
            batch[
                "respiration"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        respiration = (
            standardize_respiration(
                respiration
            )
        )

        eeg_tokens = (
            batch[
                "eeg_tokens"
            ]
            .to(
                device,
                non_blocking=True,
            )
            .long()
        )

        # Training:
        #   - default: paper/MAGE truncated-Gaussian masking;
        #   - optional ablation: fixed high masking ratio supplied with
        #     --training-mask-ratio (for example 0.95).
        #
        # Validation:
        # all EEG tokens masked, i.e. respiration-only generation.
        if training:
            mask_ratio = training_mask_ratio
        else:
            mask_ratio = 1.0

        with torch.set_grad_enabled(
            training
        ):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(
                    device.type
                    == "cuda"
                ),
            ):
                outputs = model(
                    respiration=(
                        respiration
                    ),
                    eeg_tokens=(
                        eeg_tokens
                    ),
                    mask_ratio=(
                        mask_ratio
                    ),
                )

                loss = outputs[
                    "loss"
                ]

        if not torch.isfinite(
            loss
        ):
            raise ValueError(
                "Non-finite loss detected"
            )

        if training:
            if scaler is None:
                raise ValueError(
                    "GradScaler required "
                    "during training."
                )

            scaler.scale(
                loss
                / gradient_accumulation_steps
            ).backward()

            should_update = (
                batch_index
                % gradient_accumulation_steps
                == 0
                or batch_index
                == batches_to_run
            )

            if should_update:
                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=(
                        max_grad_norm
                    ),
                )

                scale_before = (
                    scaler.get_scale()
                )

                scaler.step(
                    optimizer
                )

                scaler.update()

                scale_after = (
                    scaler.get_scale()
                )

                if (
                    scale_after
                    < scale_before
                ):
                    skipped_updates += 1

                else:
                    successful_updates += 1

                optimizer.zero_grad(
                    set_to_none=True
                )

        with torch.no_grad():
            logits = outputs[
                "logits"
            ]

            targets = outputs[
                "targets"
            ]

            mask = outputs[
                "mask"
            ]

            masked_logits = (
                logits[mask]
            )

            masked_targets = (
                targets[mask]
            )

            predictions = (
                masked_logits
                .argmax(
                    dim=-1
                )
            )

            top5_predictions = (
                masked_logits
                .topk(
                    k=5,
                    dim=-1,
                )
                .indices
            )

            positions = (
                masked_targets
                .numel()
            )

            total_loss += (
                float(
                    loss.item()
                )
                * positions
            )

            total_correct += int(
                (
                    predictions
                    == masked_targets
                )
                .sum()
                .item()
            )

            total_top5 += int(
                (
                    top5_predictions
                    == masked_targets
                    .unsqueeze(-1)
                )
                .any(
                    dim=-1
                )
                .sum()
                .item()
            )

            total_positions += (
                positions
            )

            total_mask_ratio += float(
                outputs[
                    "mask_ratio"
                ]
            )

            processed_batches += 1

            predicted_codes_seen[
                predictions.unique()
            ] = True

            target_codes_seen[
                masked_targets.unique()
            ] = True

        if (
            log_every > 0
            and batch_index
            % log_every
            == 0
        ):
            mode = (
                "train"
                if training
                else "validation"
            )

            print(
                f"  {mode} "
                f"batch {batch_index}"
                f"/{batches_to_run}"
                f" | CE="
                f"{total_loss / total_positions:.4f}"
                f" | acc="
                f"{total_correct / total_positions:.4f}"
                f" | top5="
                f"{total_top5 / total_positions:.4f}"
                f" | mask="
                f"{total_mask_ratio / processed_batches:.4f}",
                flush=True,
            )

    if processed_batches == 0:
        raise RuntimeError(
            "No batch was processed."
        )

    if (
        training
        and successful_updates
        == 0
    ):
        raise RuntimeError(
            "Every optimizer update "
            "was skipped."
        )

    return {
        "loss": (
            total_loss
            / total_positions
        ),

        "cross_entropy_loss": (
            total_loss
            / total_positions
        ),

        "accuracy": (
            total_correct
            / total_positions
        ),

        "top5_accuracy": (
            total_top5
            / total_positions
        ),

        "mask_ratio": (
            total_mask_ratio
            / processed_batches
        ),

        "unique_predicted_codes": (
            int(
                predicted_codes_seen
                .sum()
                .item()
            )
        ),

        "unique_target_codes": (
            int(
                target_codes_seen
                .sum()
                .item()
            )
        ),

        "positions": (
            total_positions
        ),

        "batches": (
            processed_batches
        ),

        "optimizer_updates": (
            successful_updates
        ),

        "skipped_optimizer_updates": (
            skipped_updates
        ),
    }


# ============================================================
# Checkpoint
# ============================================================

def make_checkpoint(
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_validation_loss,
    train_metrics,
    validation_metrics,
    data_root,
    args,
):
    return {
        "architecture": (
            ARCHITECTURE_NAME
        ),

        "data_format": (
            DATA_FORMAT
        ),

        "data_root": str(
            data_root
        ),

        "epoch": int(
            epoch
        ),

        "best_validation_loss": (
            float(
                best_validation_loss
            )
        ),

        "model_config": (
            model.get_config()
        ),

        "model_state_dict": (
            model.state_dict()
        ),

        "optimizer_state_dict": (
            optimizer.state_dict()
        ),

        "scheduler_state_dict": (
            scheduler.state_dict()
        ),

        "scaler_state_dict": (
            scaler.state_dict()
        ),

        "train_metrics": (
            train_metrics
        ),

        "validation_metrics": (
            validation_metrics
        ),

        "training_config": {
            "epochs": (
                args.epochs
            ),

            "batch_size": (
                args.batch_size
            ),

            "gradient_accumulation_steps": (
                args.gradient_accumulation_steps
            ),

            "effective_batch_size": (
                args.batch_size
                * args.gradient_accumulation_steps
            ),

            "lr": (
                args.lr
            ),

            "weight_decay": (
                args.weight_decay
            ),

            "warmup_epochs": (
                args.warmup_epochs
            ),

            "training_mask_ratio": (
                args.training_mask_ratio
            ),

            "temporal_attention_mask": (
                args.temporal_attention_mask
            ),

            "seed": (
                args.seed
            ),
        },
    }


def load_resume_checkpoint(
    checkpoint_path: Path,
    model: (
        PaperMaskedRespirationToEEGTransformer
    ),
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
) -> tuple[int, float]:

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if (
        checkpoint.get(
            "architecture"
        )
        != ARCHITECTURE_NAME
    ):
        raise ValueError(
            "Resume checkpoint has "
            "an incompatible architecture."
        )

    if (
        checkpoint.get(
            "data_format"
        )
        != DATA_FORMAT
    ):
        raise ValueError(
            "Resume checkpoint uses "
            "a different data format."
        )

    if (
        checkpoint.get(
            "model_config"
        )
        != model.get_config()
    ):
        raise ValueError(
            "Resume checkpoint model "
            "configuration does not match."
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    scheduler.load_state_dict(
        checkpoint[
            "scheduler_state_dict"
        ]
    )

    if (
        "scaler_state_dict"
        in checkpoint
    ):
        scaler.load_state_dict(
            checkpoint[
                "scaler_state_dict"
            ]
        )

    completed_epoch = int(
        checkpoint[
            "epoch"
        ]
    )

    best_validation_loss = float(
        checkpoint[
            "best_validation_loss"
        ]
    )

    return (
        completed_epoch + 1,
        best_validation_loss,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train the paper masked "
            "respiration-to-EEG Transformer "
            "using precomputed multi-dataset "
            "respiration and EEG tokens."
        )
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=(
            DEFAULT_DATA_ROOT
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            DEFAULT_OUTPUT_DIR
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
    )

    # Safe default for a ~24 GB GPU.
    #
    # 4 × 48 = effective batch size 192,
    # matching the paper configuration.
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=48,
    )

    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1.125e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--overfit-samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=None,
        help=(
            "Load model weights only from an existing checkpoint. "
            "For shared temporal positions, respiration and EEG "
            "time embeddings are averaged."
        ),
    )

    parser.add_argument(
        "--shared-temporal-position",
        action="store_true",
        help=(
            "Use the same 64-position temporal embedding "
            "for respiration tokens and EEG time columns."
        ),
    )
    parser.add_argument(
        "--temporal-alignment-tag",
        action="store_true",
        help=(
            "Keep the original separate temporal positional embeddings "
            "and add one learned shared alignment tag for each "
            "4-minute respiration/EEG interval."
        ),
    )

    parser.add_argument(
        "--temporal-attention-mask",
        action="store_true",
        help=(
            "Apply a hard cross-modal temporal attention mask. "
            "Respiration token t can exchange cross-modal information "
            "only with EEG tokens from temporal column t, while "
            "respiration-respiration and EEG-EEG attention stay global. "
            "This changes attention routing only: no new parameters and "
            "no auxiliary loss are added."
        ),
    )

    parser.add_argument(
        "--training-mask-ratio",
        type=float,
        default=None,
        help=(
            "Optional fixed EEG masking ratio used only during training. "
            "If omitted, use the paper/MAGE truncated-Gaussian masking. "
            "For the high-mask ablation, use e.g. 0.95. "
            "Validation remains fully masked at 1.0."
        ),
    )
    parser.add_argument(
        "--mask-ratio-mu",
        type=float,
        default=0.55,
        help=(
            "Mean/center of the truncated-Gaussian MAGE masking distribution. "
            "Paper default: 0.55. Use 0.85 for the high-mask variable ablation."
        ),
    )

    args = parser.parse_args()
    if (
        args.shared_temporal_position
        and args.temporal_alignment_tag
    ):
        raise ValueError(
            "--shared-temporal-position and "
            "--temporal-alignment-tag are separate "
            "ablations and cannot be enabled together."
        )

    # --------------------------------------------------------
    # Validate arguments
    # --------------------------------------------------------

    if (
        args.epochs < 1
        or args.batch_size < 1
    ):
        raise ValueError(
            "epochs and batch-size "
            "must be positive."
        )

    if (
        args.gradient_accumulation_steps
        < 1
    ):
        raise ValueError(
            "gradient-accumulation-steps "
            "must be positive."
        )

    if not (
        0
        <= args.warmup_epochs
        < args.epochs
    ):
        raise ValueError(
            "warmup-epochs must be "
            "smaller than epochs."
        )

    if (
        args.training_mask_ratio
        is not None
        and not (
            0.5
            <= args.training_mask_ratio
            <= 1.0
        )
    ):
        raise ValueError(
            "--training-mask-ratio must be in [0.5, 1.0] "
            "for the current MAGE masking configuration."
        )

    if (
        args.overfit_samples
        is not None
        and args.overfit_samples
        < 1
    ):
        raise ValueError(
            "overfit-samples must "
            "be positive."
        )

    # --------------------------------------------------------
    # Reproducibility / device
    # --------------------------------------------------------

    set_seed(
        args.seed
    )

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision(
            "high"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
        flush=True,
    )

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(
                device
            ),
            flush=True,
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = (
        PaperTransformerMMapDataset(
            args.data_root,
            "train",
        )
    )

    validation_dataset = (
        PaperTransformerMMapDataset(
            args.data_root,
            "val",
        )
    )

    print(
        "Training samples:",
        len(train_dataset),
        flush=True,
    )

    print(
        "Validation samples:",
        len(validation_dataset),
        flush=True,
    )

    if (
        args.overfit_samples
        is not None
    ):
        train_count = min(
            args.overfit_samples,
            len(train_dataset),
        )

        validation_count = min(
            args.overfit_samples,
            len(validation_dataset),
        )

        train_dataset = Subset(
            train_dataset,
            range(
                train_count
            ),
        )

        validation_dataset = Subset(
            validation_dataset,
            range(
                validation_count
            ),
        )

        print(
            "OVERFIT MODE",
            flush=True,
        )

        print(
            "Train subset:",
            len(train_dataset),
            flush=True,
        )

        print(
            "Validation subset:",
            len(validation_dataset),
            flush=True,
        )

    generator = torch.Generator()

    generator.manual_seed(
        args.seed
    )

    pin_memory = (
        device.type
        == "cuda"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=(
            args.batch_size
        ),
        shuffle=True,
        num_workers=(
            args.num_workers
        ),
        pin_memory=(
            pin_memory
        ),
        persistent_workers=(
            args.num_workers > 0
        ),
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=(
            args.batch_size
        ),
        shuffle=False,
        num_workers=(
            args.num_workers
        ),
        pin_memory=(
            pin_memory
        ),
        persistent_workers=(
            args.num_workers > 0
        ),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = (
        PaperMaskedRespirationToEEGTransformer(
            shared_temporal_position=(
                args.shared_temporal_position
            ),
            temporal_alignment_tag=(
                args.temporal_alignment_tag
            ),
            temporal_attention_mask=(
                args.temporal_attention_mask
            ),
            mask_ratio_mu=(
                args.mask_ratio_mu
            ),
        )
        .to(device)
    )

    if args.initial_checkpoint is not None:
        if not args.initial_checkpoint.exists():
            raise FileNotFoundError(
                args.initial_checkpoint
            )

        initial_checkpoint = torch.load(
            args.initial_checkpoint,
            map_location="cpu",
            weights_only=False,
        )

        if "model_state_dict" not in initial_checkpoint:
            raise KeyError(
                "Initial checkpoint is missing model_state_dict"
            )

        initial_state = dict(
            initial_checkpoint["model_state_dict"]
        )

        if args.shared_temporal_position:
            for module_name in (
                "encoder",
                "decoder",
            ):
                respiration_key = (
                    f"{module_name}.respiration_position"
                )

                eeg_time_key = (
                    f"{module_name}.eeg_time_position"
                )

                shared_key = (
                    f"{module_name}.shared_time_position"
                )

                if respiration_key not in initial_state:
                    raise KeyError(
                        f"Missing {respiration_key} "
                        "in initial checkpoint"
                    )

                if eeg_time_key not in initial_state:
                    raise KeyError(
                        f"Missing {eeg_time_key} "
                        "in initial checkpoint"
                    )

                respiration_position = (
                    initial_state.pop(
                        respiration_key
                    )
                )

                eeg_time_position = (
                    initial_state.pop(
                        eeg_time_key
                    )
                )

                # EEG time:
                # (1, 1, 64, 768)
                # -> (1, 64, 768)
                eeg_time_position = (
                    eeg_time_position.squeeze(1)
                )

                if (
                    respiration_position.shape
                    != eeg_time_position.shape
                ):
                    raise ValueError(
                        "Temporal-position shape mismatch: "
                        f"{tuple(respiration_position.shape)} vs "
                        f"{tuple(eeg_time_position.shape)}"
                    )

                initial_state[shared_key] = (
                    respiration_position
                    + eeg_time_position
                ) / 2.0

                print(
                    f"Converted {module_name} temporal positions:",
                    tuple(
                        initial_state[
                            shared_key
                        ].shape
                    ),
                )
        if args.temporal_alignment_tag:
            initial_config = initial_checkpoint.get(
                "model_config",
                {},
            )

            # This ablation must start from the original
            # separate-position paper baseline, not from
            # the previous shared-time experiment.
            if initial_config.get(
                "shared_temporal_position",
                False,
            ):
                raise ValueError(
                    "The temporal-alignment-tag ablation must "
                    "be initialized from a baseline checkpoint "
                    "with shared_temporal_position=False."
                )

            current_state = model.state_dict()

            for module_name in (
                "encoder",
                "decoder",
            ):
                alignment_key = (
                    f"{module_name}."
                    "temporal_alignment_tag"
                )

                if alignment_key not in initial_state:
                    initial_state[alignment_key] = (
                        current_state[
                            alignment_key
                        ].clone()
                    )

                    print(
                        "Initialized "
                        f"{alignment_key} "
                        "from zeros",
                        flush=True,
                    )

        model.load_state_dict(
            initial_state,
            strict=True,
        )

        print(
            "Initial checkpoint loaded:",
            args.initial_checkpoint,
        )

        print(
            "Initial checkpoint epoch:",
            int(
                initial_checkpoint.get(
                    "epoch",
                    -1,
                )
            ) + 1,
        )

        if args.shared_temporal_position:
            print(
                "Temporal initialization:",
                "mean(respiration_position, eeg_time_position)",
            )

    encoder_blocks = len(
        model.encoder.blocks.layers
    )

    decoder_blocks = len(
        model.decoder.blocks.layers
    )

    total_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    print(
        "Architecture:",
        ARCHITECTURE_NAME,
        flush=True,
    )

    print(
        "Encoder blocks:",
        encoder_blocks,
        flush=True,
    )

    print(
        "Decoder blocks:",
        decoder_blocks,
        flush=True,
    )

    print(
        "Embedding dim:",
        model.get_config()[
            "embedding_dim"
        ],
        flush=True,
    )

    print(
        "Attention heads:",
        model.get_config()[
            "num_heads"
        ],
        flush=True,
    )

    print(
        "Codebook size:",
        model.codebook_size,
        flush=True,
    )

    print(
        "Shared temporal position:",
        model.get_config()[
            "shared_temporal_position"
        ],
        flush=True,
    )

    print(
        "Temporal alignment tag:",
        model.get_config()[
            "temporal_alignment_tag"
        ],
        flush=True,
    )

    print(
        "Temporal attention mask:",
        model.get_config()[
            "temporal_attention_mask"
        ],
        flush=True,
    )

    print(
        "Training mask ratio:",
        (
            args.training_mask_ratio
            if args.training_mask_ratio
            is not None
            else "paper/MAGE variable"
        ),
        flush=True,
    )

    print(
        "Total parameters:",
        f"{total_parameters:,}",
        flush=True,
    )

    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}",
        flush=True,
    )

    print(
        "Batch size:",
        args.batch_size,
        flush=True,
    )

    print(
        "Gradient accumulation:",
        args.gradient_accumulation_steps,
        flush=True,
    )

    print(
        "Effective batch size:",
        (
            args.batch_size
            * args.gradient_accumulation_steps
        ),
        flush=True,
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=(
            args.weight_decay
        ),
    )

    scheduler = create_scheduler(
        optimizer,
        epochs=args.epochs,
        warmup_epochs=(
            args.warmup_epochs
        ),
    )

    scaler = (
        torch.amp.GradScaler(
            "cuda",
            enabled=(
                device.type
                == "cuda"
            ),
        )
    )

    # --------------------------------------------------------
    # Output safety
    # --------------------------------------------------------

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    latest_path = (
        args.output_dir
        / "checkpoint_latest.pt"
    )

    best_path = (
        args.output_dir
        / "checkpoint_best.pt"
    )

    history_path = (
        args.output_dir
        / "history.jsonl"
    )

    if (
        latest_path.exists()
        and args.resume_checkpoint
        is None
    ):
        raise FileExistsError(
            f"{latest_path} already exists. "
            "Use --resume-checkpoint or "
            "a different --output-dir "
            "to avoid overwriting a run."
        )

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    start_epoch = 0

    best_validation_loss = (
        float("inf")
    )

    if (
        args.resume_checkpoint
        is not None
    ):
        (
            start_epoch,
            best_validation_loss,
        ) = load_resume_checkpoint(
            checkpoint_path=(
                args.resume_checkpoint
            ),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )

        print(
            "Resumed from:",
            args.resume_checkpoint,
            flush=True,
        )

        print(
            "Next epoch:",
            start_epoch + 1,
            flush=True,
        )

        print(
            "Best validation CE:",
            best_validation_loss,
            flush=True,
        )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch_index in range(
        start_epoch,
        args.epochs,
    ):
        epoch_number = (
            epoch_index + 1
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        print()
        print(
            "=" * 72,
            flush=True,
        )

        print(
            f"Epoch "
            f"{epoch_number}/"
            f"{args.epochs}"
            f" | lr="
            f"{current_lr:.8g}",
            flush=True,
        )

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            gradient_accumulation_steps=(
                args.gradient_accumulation_steps
            ),
            max_grad_norm=(
                args.max_grad_norm
            ),
            max_batches=(
                args.max_train_batches
            ),
            log_every=(
                args.log_every
            ),
            training_mask_ratio=(
                args.training_mask_ratio
            ),
        )

        validation_metrics = run_epoch(
            model=model,
            loader=(
                validation_loader
            ),
            device=device,
            optimizer=None,
            scaler=None,
            gradient_accumulation_steps=1,
            max_grad_norm=(
                args.max_grad_norm
            ),
            max_batches=(
                args.max_val_batches
            ),
            log_every=(
                args.log_every
            ),
            training_mask_ratio=None,
        )

        validation_loss = float(
            validation_metrics[
                "loss"
            ]
        )

        improved = (
            validation_loss
            < best_validation_loss
        )

        if improved:
            best_validation_loss = (
                validation_loss
            )

        # Prepare LR for next epoch.
        scheduler.step()

        checkpoint = make_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch_index,
            best_validation_loss=(
                best_validation_loss
            ),
            train_metrics=(
                train_metrics
            ),
            validation_metrics=(
                validation_metrics
            ),
            data_root=(
                args.data_root
            ),
            args=args,
        )

        torch.save(
            checkpoint,
            latest_path,
        )

        if improved:
            torch.save(
                checkpoint,
                best_path,
            )

        history_entry = {
            "epoch": (
                epoch_number
            ),

            "lr": (
                current_lr
            ),

            "train": (
                train_metrics
            ),

            "validation": (
                validation_metrics
            ),

            "best_validation_loss": (
                best_validation_loss
            ),

            "best": (
                improved
            ),
        }

        with history_path.open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                json.dumps(
                    history_entry
                )
                + "\n"
            )

        print()
        print(
            "Train"
            f" | CE="
            f"{train_metrics['loss']:.4f}"
            f" | acc="
            f"{train_metrics['accuracy']:.4f}"
            f" | top5="
            f"{train_metrics['top5_accuracy']:.4f}"
            f" | mask="
            f"{train_metrics['mask_ratio']:.4f}"
            f" | pred_codes="
            f"{train_metrics['unique_predicted_codes']}"
            f" | target_codes="
            f"{train_metrics['unique_target_codes']}",
            flush=True,
        )

        print(
            "Validation"
            f" | CE="
            f"{validation_metrics['loss']:.4f}"
            f" | acc="
            f"{validation_metrics['accuracy']:.4f}"
            f" | top5="
            f"{validation_metrics['top5_accuracy']:.4f}"
            f" | mask="
            f"{validation_metrics['mask_ratio']:.4f}"
            f" | pred_codes="
            f"{validation_metrics['unique_predicted_codes']}"
            f" | target_codes="
            f"{validation_metrics['unique_target_codes']}",
            flush=True,
        )

        print(
            "Best validation CE:",
            f"{best_validation_loss:.4f}",
            (
                " [NEW BEST]"
                if improved
                else ""
            ),
            flush=True,
        )

        print(
            "Latest checkpoint:",
            latest_path,
            flush=True,
        )

        if improved:
            print(
                "Best checkpoint:",
                best_path,
                flush=True,
            )

    print()
    print(
        "=" * 72,
        flush=True,
    )

    print(
        "TRAINING COMPLETE",
        flush=True,
    )

    print(
        "Best validation CE:",
        best_validation_loss,
        flush=True,
    )

    print(
        "Best checkpoint:",
        best_path,
        flush=True,
    )


if __name__ == "__main__":
    main()
