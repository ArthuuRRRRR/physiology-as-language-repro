"""2-GPU DDP trainer for the paper-oriented respiration-to-EEG masked Transformer.

This file intentionally reuses the scientific components from
paper_architecture/train_paper_masked_transformer.py and changes only the
training infrastructure needed for DistributedDataParallel (DDP).

Scientific behavior preserved:
    - precomputed respiration : (N, 64, 2400) float32
    - precomputed EEG tokens  : (N, 8, 64) uint16
    - paper architecture (768 dim, 8 encoder, 8 decoder, 8 heads)
    - MAGE-style variable masking by default
    - masked-token cross entropy
    - AdamW
    - LR 1.125e-4
    - weight decay 0.05
    - 40 warm-up epochs
    - cosine decay
    - target effective batch size 192

Default 2-GPU DDP configuration:
    4 samples / GPU x 2 GPUs x 24 accumulation = 192 effective batch

Launch with:
    CUDA_VISIBLE_DEVICES=1,2 \
    torchrun --standalone --nproc_per_node=2 \
    paper_architecture/train_paper_masked_transformer_ddp.py ...

Notes:
    - DDP changes sample ordering and floating-point reduction order, so a run
      will not be bit-for-bit identical to the single-GPU run.
    - Checkpoints save the UNWRAPPED model state_dict (no "module." prefix), so
      they remain compatible with the existing paper evaluation scripts.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler, Subset
from torch.utils.data.distributed import DistributedSampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from paper_architecture import (  # noqa: E402
    PaperMaskedRespirationToEEGTransformer,
)
from paper_architecture.train_paper_masked_transformer import (  # noqa: E402
    ARCHITECTURE_NAME,
    DATA_FORMAT,
    PaperTransformerMMapDataset,
    create_scheduler,
    standardize_respiration,
)


DEFAULT_DATA_ROOT = Path("outputs/paper_transformer_data")
DEFAULT_OUTPUT_DIR = Path(
    "outputs/paper_masked_transformer_multidataset_ddp"
)


# ============================================================
# Distributed helpers
# ============================================================

def setup_distributed() -> tuple[int, int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this DDP trainer.")

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            "This script must be launched with torchrun. Example:\n"
            "CUDA_VISIBLE_DEVICES=1,2 torchrun --standalone "
            "--nproc_per_node=2 "
            "paper_architecture/train_paper_masked_transformer_ddp.py ..."
        )

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    return rank, world_size, local_rank, device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def main_print(rank: int, *args, **kwargs) -> None:
    if is_main_process(rank):
        print(*args, **kwargs)


def seed_before_model_init(seed: int) -> None:
    """All ranks initialize identically; DDP later broadcasts rank-0 weights."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_rank_runtime(seed: int, rank: int) -> None:
    """Give each rank an independent dropout/masking RNG stream."""
    runtime_seed = seed + rank
    random.seed(runtime_seed)
    np.random.seed(runtime_seed)
    torch.manual_seed(runtime_seed)
    torch.cuda.manual_seed_all(runtime_seed)


class DistributedEvalSampler(Sampler[int]):
    """Partition evaluation data across ranks without padding or duplication."""

    def __init__(
        self,
        dataset,
        num_replicas: int,
        rank: int,
    ) -> None:
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

    def __iter__(self):
        return iter(
            range(
                self.rank,
                len(self.dataset),
                self.num_replicas,
            )
        )

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.rank >= n:
            return 0
        return (
            (n - 1 - self.rank)
            // self.num_replicas
            + 1
        )


def all_reduce_metrics(
    *,
    total_loss: float,
    total_correct: int,
    total_top5: int,
    total_positions: int,
    total_mask_ratio: float,
    processed_batches: int,
    predicted_codes_seen: torch.Tensor,
    target_codes_seen: torch.Tensor,
    successful_updates: int,
    skipped_updates: int,
    device: torch.device,
) -> dict[str, float | int]:
    scalars = torch.tensor(
        [
            total_loss,
            float(total_correct),
            float(total_top5),
            float(total_positions),
            total_mask_ratio,
            float(processed_batches),
        ],
        dtype=torch.float64,
        device=device,
    )

    dist.all_reduce(
        scalars,
        op=dist.ReduceOp.SUM,
    )

    predicted_codes_int = predicted_codes_seen.to(
        dtype=torch.int32
    )
    target_codes_int = target_codes_seen.to(
        dtype=torch.int32
    )

    dist.all_reduce(
        predicted_codes_int,
        op=dist.ReduceOp.MAX,
    )
    dist.all_reduce(
        target_codes_int,
        op=dist.ReduceOp.MAX,
    )

    update_counts = torch.tensor(
        [
            successful_updates,
            skipped_updates,
        ],
        dtype=torch.int64,
        device=device,
    )

    # Every DDP rank performs the same optimizer-step schedule.
    # MAX avoids reporting world_size x the true number of updates.
    dist.all_reduce(
        update_counts,
        op=dist.ReduceOp.MAX,
    )

    (
        global_total_loss,
        global_correct,
        global_top5,
        global_positions,
        global_mask_ratio,
        global_batches,
    ) = scalars.tolist()

    global_positions_int = int(round(global_positions))
    global_batches_int = int(round(global_batches))

    if global_positions_int <= 0:
        raise RuntimeError("No positions were aggregated.")

    if global_batches_int <= 0:
        raise RuntimeError("No batches were aggregated.")

    return {
        "loss": (
            global_total_loss
            / global_positions_int
        ),
        "cross_entropy_loss": (
            global_total_loss
            / global_positions_int
        ),
        "accuracy": (
            global_correct
            / global_positions_int
        ),
        "top5_accuracy": (
            global_top5
            / global_positions_int
        ),
        "mask_ratio": (
            global_mask_ratio
            / global_batches_int
        ),
        "unique_predicted_codes": int(
            (predicted_codes_int > 0)
            .sum()
            .item()
        ),
        "unique_target_codes": int(
            (target_codes_int > 0)
            .sum()
            .item()
        ),
        "positions": global_positions_int,
        "batches": global_batches_int,
        "optimizer_updates": int(
            update_counts[0].item()
        ),
        "skipped_optimizer_updates": int(
            update_counts[1].item()
        ),
    }


# ============================================================
# DDP epoch
# ============================================================

def run_epoch_ddp(
    *,
    model,
    raw_model: PaperMaskedRespirationToEEGTransformer,
    loader: DataLoader,
    device: torch.device,
    rank: int,
    optimizer: torch.optim.Optimizer | None = None,
    scaler=None,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    max_batches: int | None = None,
    log_every: int = 200,
    training_mask_ratio: float | None = None,
) -> dict[str, float | int]:
    training = optimizer is not None

    model.train(training)

    if training:
        optimizer.zero_grad(
            set_to_none=True
        )

    batches_to_run = len(loader)

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

    predicted_codes_seen = torch.zeros(
        raw_model.codebook_size,
        dtype=torch.bool,
        device=device,
    )

    target_codes_seen = torch.zeros_like(
        predicted_codes_seen
    )

    for batch_index, batch in enumerate(
        loader,
        start=1,
    ):
        if batch_index > batches_to_run:
            break

        respiration = (
            batch["respiration"]
            .to(
                device,
                non_blocking=True,
            )
        )

        respiration = standardize_respiration(
            respiration
        )

        eeg_tokens = (
            batch["eeg_tokens"]
            .to(
                device,
                non_blocking=True,
            )
            .long()
        )

        # Training:
        #   - default: paper/MAGE truncated-Gaussian masking
        #   - optional fixed mask ratio
        # Validation:
        #   - full masking (respiration-only generation)
        if training:
            mask_ratio = training_mask_ratio
        else:
            mask_ratio = 1.0

        should_update = (
            training
            and (
                batch_index
                % gradient_accumulation_steps
                == 0
                or batch_index
                == batches_to_run
            )
        )

        # On accumulation microbatches, suppress DDP gradient all-reduce.
        # The final microbatch of each accumulation group performs the sync.
        sync_context = nullcontext()

        if (
            training
            and isinstance(model, DDP)
            and not should_update
        ):
            sync_context = model.no_sync()

        with sync_context:
            with torch.set_grad_enabled(
                training
            ):
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=(
                        device.type == "cuda"
                    ),
                ):
                    outputs = model(
                        respiration=respiration,
                        eeg_tokens=eeg_tokens,
                        mask_ratio=mask_ratio,
                    )

                    loss = outputs["loss"]

            if not torch.isfinite(loss):
                raise ValueError(
                    "Non-finite loss detected"
                )

            if training:
                if scaler is None:
                    raise ValueError(
                        "GradScaler required during training."
                    )

                scaler.scale(
                    loss
                    / gradient_accumulation_steps
                ).backward()

        if should_update:
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_grad_norm,
            )

            scale_before = scaler.get_scale()

            scaler.step(optimizer)
            scaler.update()

            scale_after = scaler.get_scale()

            if scale_after < scale_before:
                skipped_updates += 1
            else:
                successful_updates += 1

            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.no_grad():
            logits = outputs["logits"]
            targets = outputs["targets"]
            mask = outputs["mask"]

            masked_logits = logits[mask]
            masked_targets = targets[mask]

            predictions = (
                masked_logits
                .argmax(dim=-1)
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
                float(loss.item())
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
                .any(dim=-1)
                .sum()
                .item()
            )

            total_positions += positions

            total_mask_ratio += float(
                outputs["mask_ratio"]
            )

            processed_batches += 1

            predicted_codes_seen[
                predictions.unique()
            ] = True

            target_codes_seen[
                masked_targets.unique()
            ] = True

        if (
            is_main_process(rank)
            and log_every > 0
            and batch_index % log_every == 0
        ):
            mode = (
                "train"
                if training
                else "validation"
            )

            print(
                f"  {mode} rank0 batch "
                f"{batch_index}/{batches_to_run}"
                f" | local_CE="
                f"{total_loss / total_positions:.4f}"
                f" | local_acc="
                f"{total_correct / total_positions:.4f}"
                f" | local_top5="
                f"{total_top5 / total_positions:.4f}"
                f" | local_mask="
                f"{total_mask_ratio / processed_batches:.4f}",
                flush=True,
            )

    if processed_batches == 0:
        raise RuntimeError(
            "No batch was processed."
        )

    if (
        training
        and successful_updates == 0
    ):
        raise RuntimeError(
            "Every optimizer update was skipped."
        )

    return all_reduce_metrics(
        total_loss=total_loss,
        total_correct=total_correct,
        total_top5=total_top5,
        total_positions=total_positions,
        total_mask_ratio=total_mask_ratio,
        processed_batches=processed_batches,
        predicted_codes_seen=predicted_codes_seen,
        target_codes_seen=target_codes_seen,
        successful_updates=successful_updates,
        skipped_updates=skipped_updates,
        device=device,
    )


# ============================================================
# Initial checkpoint compatibility
# ============================================================

def load_initial_weights(
    *,
    model: PaperMaskedRespirationToEEGTransformer,
    checkpoint_path: Path,
    shared_temporal_position: bool,
    temporal_alignment_tag: bool,
    rank: int,
) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            checkpoint_path
        )

    initial_checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" not in initial_checkpoint:
        raise KeyError(
            "Initial checkpoint is missing model_state_dict"
        )

    initial_state = dict(
        initial_checkpoint[
            "model_state_dict"
        ]
    )

    if shared_temporal_position:
        for module_name in (
            "encoder",
            "decoder",
        ):
            respiration_key = (
                f"{module_name}."
                "respiration_position"
            )

            eeg_time_key = (
                f"{module_name}."
                "eeg_time_position"
            )

            shared_key = (
                f"{module_name}."
                "shared_time_position"
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
                .squeeze(1)
            )

            if (
                respiration_position.shape
                != eeg_time_position.shape
            ):
                raise ValueError(
                    "Temporal-position shape mismatch: "
                    f"{tuple(respiration_position.shape)} "
                    "vs "
                    f"{tuple(eeg_time_position.shape)}"
                )

            initial_state[shared_key] = (
                respiration_position
                + eeg_time_position
            ) / 2.0

            main_print(
                rank,
                f"Converted {module_name} temporal positions:",
                tuple(
                    initial_state[
                        shared_key
                    ].shape
                ),
                flush=True,
            )

    if temporal_alignment_tag:
        initial_config = (
            initial_checkpoint.get(
                "model_config",
                {},
            )
        )

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

                main_print(
                    rank,
                    "Initialized "
                    f"{alignment_key} "
                    "from zeros",
                    flush=True,
                )

    model.load_state_dict(
        initial_state,
        strict=True,
    )

    main_print(
        rank,
        "Initial checkpoint loaded:",
        checkpoint_path,
        flush=True,
    )

    main_print(
        rank,
        "Initial checkpoint epoch:",
        int(
            initial_checkpoint.get(
                "epoch",
                -1,
            )
        ) + 1,
        flush=True,
    )

    if shared_temporal_position:
        main_print(
            rank,
            "Temporal initialization:",
            "mean(respiration_position, eeg_time_position)",
            flush=True,
        )


# ============================================================
# Checkpoint
# ============================================================

def make_checkpoint_ddp(
    *,
    model: PaperMaskedRespirationToEEGTransformer,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_validation_loss: float,
    train_metrics,
    validation_metrics,
    data_root: Path,
    args,
    world_size: int,
) -> dict:
    effective_batch_size = (
        args.batch_size
        * world_size
        * args.gradient_accumulation_steps
    )

    return {
        "architecture": ARCHITECTURE_NAME,
        "data_format": DATA_FORMAT,
        "data_root": str(data_root),
        "epoch": int(epoch),
        "best_validation_loss": float(
            best_validation_loss
        ),
        "model_config": model.get_config(),
        # Unwrapped state dict => compatible with existing evaluators.
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "scheduler_state_dict": (
            scheduler.state_dict()
        ),
        "scaler_state_dict": (
            scaler.state_dict()
        ),
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "training_config": {
            "epochs": args.epochs,
            "distributed": True,
            "world_size": world_size,
            "batch_size_per_gpu": (
                args.batch_size
            ),
            "global_physical_batch_size": (
                args.batch_size
                * world_size
            ),
            "gradient_accumulation_steps": (
                args.gradient_accumulation_steps
            ),
            "effective_batch_size": (
                effective_batch_size
            ),
            "lr": args.lr,
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
            "seed": args.seed,
        },
    }


def load_resume_checkpoint_ddp(
    *,
    checkpoint_path: Path,
    model: PaperMaskedRespirationToEEGTransformer,
    optimizer,
    scheduler,
    scaler,
) -> tuple[int, float]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if (
        checkpoint.get("architecture")
        != ARCHITECTURE_NAME
    ):
        raise ValueError(
            "Resume checkpoint has "
            "an incompatible architecture."
        )

    if (
        checkpoint.get("data_format")
        != DATA_FORMAT
    ):
        raise ValueError(
            "Resume checkpoint uses "
            "a different data format."
        )

    if (
        checkpoint.get("model_config")
        != model.get_config()
    ):
        raise ValueError(
            "Resume checkpoint model "
            "configuration does not match."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"],
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

    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(
            checkpoint[
                "scaler_state_dict"
            ]
        )

    completed_epoch = int(
        checkpoint["epoch"]
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

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the paper masked respiration-to-EEG "
            "Transformer using 2-GPU PyTorch DDP and "
            "precomputed respiration / EEG-token mmap files."
        )
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help=(
            "Physical batch size PER GPU. "
            "Default: 4."
        ),
    )

    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=24,
        help=(
            "With 2 GPUs and batch-size 4, "
            "24 gives effective batch 192."
        ),
    )

    parser.add_argument(
        "--expected-effective-batch-size",
        type=int,
        default=192,
        help=(
            "Safety check. Training aborts if "
            "batch_size * world_size * accumulation "
            "does not equal this value."
        ),
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
        help=(
            "DataLoader workers PER RANK."
        ),
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
        help=(
            "Maximum training batches PER RANK."
        ),
    )

    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help=(
            "Maximum validation batches PER RANK."
        ),
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
            "Load model weights only from an "
            "existing paper checkpoint."
        ),
    )

    parser.add_argument(
        "--shared-temporal-position",
        action="store_true",
    )

    parser.add_argument(
        "--temporal-alignment-tag",
        action="store_true",
    )

    parser.add_argument(
        "--temporal-attention-mask",
        action="store_true",
    )

    parser.add_argument(
        "--training-mask-ratio",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--mask-ratio-mu",
        type=float,
        default=0.55,
    )

    return parser


def main() -> None:
    rank = -1

    try:
        (
            rank,
            world_size,
            local_rank,
            device,
        ) = setup_distributed()

        parser = build_parser()
        args = parser.parse_args()

        if world_size != 2:
            raise RuntimeError(
                "This file is configured for exactly 2 GPUs. "
                f"torchrun reported world_size={world_size}."
            )

        if (
            args.shared_temporal_position
            and args.temporal_alignment_tag
        ):
            raise ValueError(
                "--shared-temporal-position and "
                "--temporal-alignment-tag are separate "
                "ablations and cannot be enabled together."
            )

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
                "--training-mask-ratio must be "
                "in [0.5, 1.0]."
            )

        if (
            args.overfit_samples
            is not None
            and args.overfit_samples < 1
        ):
            raise ValueError(
                "overfit-samples must be positive."
            )

        effective_batch_size = (
            args.batch_size
            * world_size
            * args.gradient_accumulation_steps
        )

        if (
            effective_batch_size
            != args.expected_effective_batch_size
        ):
            raise ValueError(
                "Effective batch-size safety check failed: "
                f"{args.batch_size} per GPU "
                f"x {world_size} GPUs "
                f"x {args.gradient_accumulation_steps} accumulation "
                f"= {effective_batch_size}, expected "
                f"{args.expected_effective_batch_size}."
            )

        seed_before_model_init(
            args.seed
        )

        torch.set_float32_matmul_precision(
            "high"
        )

        main_print(
            rank,
            "Device mode: 2-GPU DDP",
            flush=True,
        )

        main_print(
            rank,
            "World size:",
            world_size,
            flush=True,
        )

        if is_main_process(rank):
            for gpu_index in range(
                world_size
            ):
                print(
                    f"GPU {gpu_index}:",
                    torch.cuda.get_device_name(
                        gpu_index
                    ),
                    flush=True,
                )

        # ----------------------------------------------------
        # Dataset
        # ----------------------------------------------------

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
                range(train_count),
            )

            validation_dataset = Subset(
                validation_dataset,
                range(validation_count),
            )

            main_print(
                rank,
                "OVERFIT MODE",
                flush=True,
            )

        main_print(
            rank,
            "Training samples:",
            len(train_dataset),
            flush=True,
        )

        main_print(
            rank,
            "Validation samples:",
            len(validation_dataset),
            flush=True,
        )

        if (
            len(train_dataset)
            % world_size
            != 0
        ):
            main_print(
                rank,
                "WARNING: training sample count is not "
                "divisible by world_size; DistributedSampler "
                "will pad a small number of samples.",
                flush=True,
            )

        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=False,
        )

        validation_sampler = (
            DistributedEvalSampler(
                validation_dataset,
                num_replicas=world_size,
                rank=rank,
            )
        )

        pin_memory = True

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=(
                args.num_workers > 0
            ),
        )

        validation_loader = DataLoader(
            validation_dataset,
            batch_size=args.batch_size,
            sampler=validation_sampler,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=(
                args.num_workers > 0
            ),
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        raw_model = (
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

        if (
            args.initial_checkpoint
            is not None
        ):
            load_initial_weights(
                model=raw_model,
                checkpoint_path=(
                    args.initial_checkpoint
                ),
                shared_temporal_position=(
                    args.shared_temporal_position
                ),
                temporal_alignment_tag=(
                    args.temporal_alignment_tag
                ),
                rank=rank,
            )

        encoder_blocks = len(
            raw_model.encoder.blocks.layers
        )

        decoder_blocks = len(
            raw_model.decoder.blocks.layers
        )

        total_parameters = sum(
            parameter.numel()
            for parameter
            in raw_model.parameters()
        )

        trainable_parameters = sum(
            parameter.numel()
            for parameter
            in raw_model.parameters()
            if parameter.requires_grad
        )

        main_print(
            rank,
            "Architecture:",
            ARCHITECTURE_NAME,
            flush=True,
        )

        main_print(
            rank,
            "Encoder blocks:",
            encoder_blocks,
            flush=True,
        )

        main_print(
            rank,
            "Decoder blocks:",
            decoder_blocks,
            flush=True,
        )

        main_print(
            rank,
            "Embedding dim:",
            raw_model.get_config()[
                "embedding_dim"
            ],
            flush=True,
        )

        main_print(
            rank,
            "Attention heads:",
            raw_model.get_config()[
                "num_heads"
            ],
            flush=True,
        )

        main_print(
            rank,
            "Codebook size:",
            raw_model.codebook_size,
            flush=True,
        )

        main_print(
            rank,
            "Shared temporal position:",
            raw_model.get_config()[
                "shared_temporal_position"
            ],
            flush=True,
        )

        main_print(
            rank,
            "Temporal alignment tag:",
            raw_model.get_config()[
                "temporal_alignment_tag"
            ],
            flush=True,
        )

        main_print(
            rank,
            "Temporal attention mask:",
            raw_model.get_config()[
                "temporal_attention_mask"
            ],
            flush=True,
        )

        main_print(
            rank,
            "Training mask ratio:",
            (
                args.training_mask_ratio
                if args.training_mask_ratio
                is not None
                else "paper/MAGE variable"
            ),
            flush=True,
        )

        main_print(
            rank,
            "Total parameters:",
            f"{total_parameters:,}",
            flush=True,
        )

        main_print(
            rank,
            "Trainable parameters:",
            f"{trainable_parameters:,}",
            flush=True,
        )

        main_print(
            rank,
            "Batch size per GPU:",
            args.batch_size,
            flush=True,
        )

        main_print(
            rank,
            "Global physical batch:",
            args.batch_size
            * world_size,
            flush=True,
        )

        main_print(
            rank,
            "Gradient accumulation:",
            args.gradient_accumulation_steps,
            flush=True,
        )

        main_print(
            rank,
            "Effective batch size:",
            effective_batch_size,
            flush=True,
        )

        # Wrap only after all initial-weight conversion is complete.
        model = DDP(
            raw_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )

        # Independent stochastic masks/dropout across ranks.
        seed_rank_runtime(
            args.seed,
            rank,
        )

        # ----------------------------------------------------
        # Optimizer / scheduler
        # ----------------------------------------------------

        optimizer = torch.optim.AdamW(
            raw_model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        scheduler = create_scheduler(
            optimizer,
            epochs=args.epochs,
            warmup_epochs=(
                args.warmup_epochs
            ),
        )

        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=True,
        )

        # ----------------------------------------------------
        # Output / resume
        # ----------------------------------------------------

        if is_main_process(rank):
            args.output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

        dist.barrier()

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

        # Shared filesystem: all ranks see the same checkpoint path.
        if (
            latest_path.exists()
            and args.resume_checkpoint
            is None
        ):
            raise FileExistsError(
                f"{latest_path} already exists. "
                "Use --resume-checkpoint or "
                "a different --output-dir."
            )

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
            ) = load_resume_checkpoint_ddp(
                checkpoint_path=(
                    args.resume_checkpoint
                ),
                model=raw_model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
            )

            # DDP replicas must start from exactly the same resumed model.
            for parameter in raw_model.parameters():
                dist.broadcast(
                    parameter.data,
                    src=0,
                )

            main_print(
                rank,
                "Resumed from:",
                args.resume_checkpoint,
                flush=True,
            )

            main_print(
                rank,
                "Next epoch:",
                start_epoch + 1,
                flush=True,
            )

            main_print(
                rank,
                "Best validation CE:",
                best_validation_loss,
                flush=True,
            )

        dist.barrier()

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        for epoch_index in range(
            start_epoch,
            args.epochs,
        ):
            train_sampler.set_epoch(
                epoch_index
            )

            epoch_number = (
                epoch_index + 1
            )

            current_lr = (
                optimizer
                .param_groups[0]["lr"]
            )

            main_print(
                rank,
                "",
                flush=True,
            )

            main_print(
                rank,
                "=" * 72,
                flush=True,
            )

            main_print(
                rank,
                f"Epoch "
                f"{epoch_number}/"
                f"{args.epochs}"
                f" | lr="
                f"{current_lr:.8g}",
                flush=True,
            )

            train_metrics = run_epoch_ddp(
                model=model,
                raw_model=raw_model,
                loader=train_loader,
                device=device,
                rank=rank,
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

            # Validation uses the raw local replica: no DDP forward
            # synchronization is needed, and metrics are all-reduced below.
            validation_metrics = run_epoch_ddp(
                model=raw_model,
                raw_model=raw_model,
                loader=validation_loader,
                device=device,
                rank=rank,
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
                validation_metrics["loss"]
            )

            improved = (
                validation_loss
                < best_validation_loss
            )

            if improved:
                best_validation_loss = (
                    validation_loss
                )

            scheduler.step()

            if is_main_process(rank):
                checkpoint = (
                    make_checkpoint_ddp(
                        model=raw_model,
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
                        world_size=(
                            world_size
                        ),
                    )
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
                    "epoch": epoch_number,
                    "lr": current_lr,
                    "train": train_metrics,
                    "validation": (
                        validation_metrics
                    ),
                    "best_validation_loss": (
                        best_validation_loss
                    ),
                    "best": improved,
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

                print(
                    "",
                    flush=True,
                )

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

            dist.barrier()

        main_print(
            rank,
            "",
            flush=True,
        )

        main_print(
            rank,
            "=" * 72,
            flush=True,
        )

        main_print(
            rank,
            "TRAINING COMPLETE",
            flush=True,
        )

        main_print(
            rank,
            "Best validation CE:",
            best_validation_loss,
            flush=True,
        )

        main_print(
            rank,
            "Best checkpoint:",
            best_path,
            flush=True,
        )

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
