"""Two-GPU DDP trainer for the multi-dataset EEG VQGAN.

Fidelity target versus the current single-GPU configuration:
    single GPU: batch 30 x grad accumulation 4 = effective batch 120
    two GPUs:   batch 15/GPU x 2 GPUs x grad accumulation 4 = 120

The discriminator uses BatchNorm2d in the original implementation.  DDP
would otherwise compute BN statistics independently on 15 samples/GPU.
Therefore the discriminator is converted to SyncBatchNorm so its BN
statistics are computed over the same global physical batch of 30.

The VQGAN itself is unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from scripts.train_vqgan_multidataset import (
    ManifestEEGDataset,
    amp_context,
    build_scaler,
    checkpoint_payload,
    compute_generator_losses,
    load_normalization,
    save_reconstruction,
    set_requires_grad,
    validate_epoch,
)
from src.models.discriminator import (
    PatchDiscriminator,
    discriminator_hinge_loss,
)
from src.models.vqgan import VQGAN


def unwrap(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, DDP) else module


def setup_ddp() -> tuple[int, int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    if "RANK" not in os.environ or "LOCAL_RANK" not in os.environ:
        raise RuntimeError(
            "Launch with torchrun, for example: "
            "torchrun --standalone --nproc_per_node=2 "
            "scripts/train_vqgan_multidataset_ddp.py ..."
        )

    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    if world_size != 2:
        raise RuntimeError(
            f"This fidelity configuration expects exactly 2 GPUs; got {world_size}."
        )

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return rank, local_rank, world_size, device


def reduce_train_metrics(
    totals: dict[str, float],
    total_examples: int,
    device: torch.device,
) -> tuple[dict[str, float], int]:
    names = (
        "generator",
        "discriminator",
        "reconstruction",
        "vq",
        "correlation",
        "adversarial",
    )

    values = torch.tensor(
        [totals[name] for name in names] + [float(total_examples)],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)

    global_examples = int(values[-1].item())
    metrics = {
        name: values[i].item() / global_examples
        for i, name in enumerate(names)
    }
    return metrics, global_examples


def train_epoch_ddp(
    model: DDP,
    discriminator: DDP,
    loader: DataLoader,
    optimizer_vqgan: torch.optim.Optimizer,
    optimizer_disc: torch.optim.Optimizer,
    scaler_vqgan,
    scaler_disc,
    device: torch.device,
    epoch: int,
    corr_weight: float,
    adv_weight: float,
    discriminator_start_epoch: int,
    gradient_accumulation_steps: int,
    amp_dtype: str,
    log_every: int,
    max_batches: int | None,
    rank: int,
) -> dict[str, float]:
    model.train()
    discriminator.train()

    adversarial_active = epoch >= discriminator_start_epoch

    total_batches = len(loader)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)
    if total_batches <= 0:
        raise RuntimeError("No training batch is available")

    totals = {
        "generator": 0.0,
        "discriminator": 0.0,
        "reconstruction": 0.0,
        "vq": 0.0,
        "correlation": 0.0,
        "adversarial": 0.0,
    }

    total_examples = 0
    optimizer_updates = 0
    pending = 0
    group_size = 0

    for batch_index, batch in enumerate(loader):
        if batch_index >= total_batches:
            break

        if pending == 0:
            group_size = min(
                gradient_accumulation_steps,
                total_batches - batch_index,
            )
            optimizer_vqgan.zero_grad(set_to_none=True)
            optimizer_disc.zero_grad(set_to_none=True)

        # Synchronize DDP gradients only on the final micro-batch of each
        # accumulation group. This changes communication cost, not the gradient.
        sync_now = (pending + 1) == group_size

        eeg = batch["eeg_spectrogram"].unsqueeze(1).to(
            device,
            non_blocking=True,
        )
        local_batch_size = eeg.shape[0]

        # -------------------------
        # Discriminator backward
        # -------------------------
        if adversarial_active:
            set_requires_grad(discriminator, True)

            # This pass is inference-only for the VQGAN, so bypass its DDP
            # wrapper; no gradient synchronization is needed here.
            with torch.no_grad(), amp_context(device, amp_dtype):
                fake_eeg, _, _ = unwrap(model)(eeg)

            disc_sync_context = (
                nullcontext() if sync_now else discriminator.no_sync()
            )
            with disc_sync_context:
                with amp_context(device, amp_dtype):
                    discriminator_loss = discriminator_hinge_loss(
                        discriminator(eeg),
                        discriminator(fake_eeg.detach()),
                    )

                scaler_disc.scale(
                    discriminator_loss / group_size
                ).backward()
        else:
            discriminator_loss = eeg.new_zeros(())

        # -------------------------
        # Generator / VQGAN backward
        # -------------------------
        set_requires_grad(discriminator, False)

        model_sync_context = (
            nullcontext() if sync_now else model.no_sync()
        )

        with model_sync_context:
            with amp_context(device, amp_dtype):
                _, generator_losses, _ = compute_generator_losses(
                    model=model,
                    # Avoid invoking the discriminator DDP reducer while the
                    # discriminator parameters are frozen. SyncBatchNorm
                    # remains active in the underlying module.
                    discriminator=unwrap(discriminator),
                    eeg=eeg,
                    corr_weight=corr_weight,
                    adv_weight=adv_weight,
                    adversarial_active=adversarial_active,
                )

            scaler_vqgan.scale(
                generator_losses["generator"] / group_size
            ).backward()

        set_requires_grad(discriminator, True)

        pending += 1

        if pending == group_size:
            if adversarial_active:
                scaler_disc.step(optimizer_disc)
                scaler_disc.update()

            scaler_vqgan.step(optimizer_vqgan)
            scaler_vqgan.update()

            optimizer_updates += 1
            pending = 0

        totals["discriminator"] += (
            float(discriminator_loss.detach()) * local_batch_size
        )
        for name, value in generator_losses.items():
            totals[name] += float(value.detach()) * local_batch_size

        total_examples += local_batch_size

        if (
            rank == 0
            and log_every > 0
            and (batch_index + 1) % log_every == 0
        ):
            print(
                f"  train batch {batch_index + 1}/{total_batches} | "
                f"recon={totals['reconstruction'] / total_examples:.4f} | "
                f"corr={totals['correlation'] / total_examples:.4f} | "
                f"vq={totals['vq'] / total_examples:.4f}",
                flush=True,
            )

    metrics, global_examples = reduce_train_metrics(
        totals,
        total_examples,
        device,
    )

    metrics["examples"] = float(global_examples)
    # One DDP iteration corresponds to one GLOBAL batch (15 + 15 = 30).
    metrics["batches"] = float(total_batches)
    metrics["optimizer_updates"] = float(optimizer_updates)
    metrics["adversarial_active"] = float(adversarial_active)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-GPU DDP training for the multi-dataset VQGAN."
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("outputs/vqgan_multidataset_preprocessed"),
    )
    parser.add_argument("--train-manifest", type=Path, default=None)
    parser.add_argument("--val-manifest", type=Path, default=None)
    parser.add_argument("--normalization-file", type=Path, default=None)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/vqgan_multidataset_ddp"),
    )

    parser.add_argument("--epochs", type=int, default=200)

    # PER-GPU batch. With exactly 2 GPUs this reproduces global batch 30.
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
    )

    parser.add_argument("--lr", type=float, default=4.8e-5)
    parser.add_argument("--disc-lr", type=float, default=4.8e-5)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--corr-weight", type=float, default=0.1)
    parser.add_argument("--adv-weight", type=float, default=0.01)

    parser.add_argument(
        "--discriminator-start-epoch",
        type=int,
        default=1,
    )

    # This is per process. 4 x 2 GPUs = 8 workers total, matching the
    # previous single-GPU run's total worker count.
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--amp-dtype",
        choices=["bfloat16", "float16", "none"],
        default="bfloat16",
    )

    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)

    args = parser.parse_args()

    rank, local_rank, world_size, device = setup_ddp()

    try:
        if args.batch_size <= 0:
            raise ValueError("batch-size must be positive")
        if args.gradient_accumulation_steps <= 0:
            raise ValueError(
                "gradient-accumulation-steps must be positive"
            )
        if args.epochs <= 0:
            raise ValueError("epochs must be positive")

        if args.train_manifest is None:
            args.train_manifest = (
                args.data_root / "train_manifest.jsonl"
            )
        if args.val_manifest is None:
            args.val_manifest = (
                args.data_root / "val_manifest.jsonl"
            )
        if args.normalization_file is None:
            args.normalization_file = (
                args.data_root / "normalization.json"
            )

        global_batch_size = args.batch_size * world_size
        effective_batch_size = (
            global_batch_size
            * args.gradient_accumulation_steps
        )

        # Hard guard: refuse to silently change the current optimization
        # semantics (single GPU batch 30, accumulation 4, effective 120).
        if global_batch_size != 30:
            raise ValueError(
                "Expected global physical batch size 30 "
                f"(got {global_batch_size}). "
                "With 2 GPUs use --batch-size 15."
            )
        if effective_batch_size != 120:
            raise ValueError(
                "Expected effective batch size 120 "
                f"(got {effective_batch_size}). "
                "Use --gradient-accumulation-steps 4."
            )

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        # Keep exactly the same TF32 behavior as the original trainer.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        normalization, min_db, max_db = load_normalization(
            args.normalization_file
        )

        train_dataset = ManifestEEGDataset(
            args.train_manifest,
            min_db,
            max_db,
        )
        validation_dataset = ManifestEEGDataset(
            args.val_manifest,
            min_db,
            max_db,
        )

        # For the current 25,188-sample train set:
        # 25,188 / 2 = 12,594 samples/rank.
        # batch 15 + drop_last=True -> 839 batches/rank and 9 dropped/rank,
        # i.e. 25,170 samples used and 18 dropped globally: exactly the same
        # count as single-GPU batch 30 + drop_last=True.
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=False,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            drop_last=True,
        )

        # Preserve validation semantics exactly: rank 0 alone validates the
        # complete validation set with global batch 30 and no distributed
        # padding/duplication.
        validation_loader = None
        if rank == 0:
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=global_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=args.num_workers > 0,
                drop_last=False,
            )

        model_raw = VQGAN().to(device)

        discriminator_raw = PatchDiscriminator().to(device)

        # Critical fidelity step:
        # original discriminator = BatchNorm2d with physical batch 30.
        # DDP = 15 samples/GPU, so SyncBatchNorm restores BN statistics over
        # the same global physical batch of 30.
        discriminator_raw = nn.SyncBatchNorm.convert_sync_batchnorm(
            discriminator_raw
        )

        optimizer_vqgan = torch.optim.Adam(
            model_raw.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
        )
        optimizer_disc = torch.optim.Adam(
            discriminator_raw.parameters(),
            lr=args.disc_lr,
            betas=(args.beta1, args.beta2),
        )

        fp16_scaling = args.amp_dtype == "float16"
        scaler_vqgan = build_scaler(fp16_scaling)
        scaler_disc = build_scaler(fp16_scaling)

        start_epoch = 1
        best_validation_loss = float("inf")

        if args.resume is not None:
            checkpoint = torch.load(
                args.resume,
                map_location=device,
            )

            model_raw.load_state_dict(
                checkpoint["model_state_dict"],
                strict=True,
            )
            # BatchNorm2d and SyncBatchNorm have compatible state-dict
            # parameter/buffer names.
            discriminator_raw.load_state_dict(
                checkpoint["discriminator_state_dict"],
                strict=True,
            )

            optimizer_vqgan.load_state_dict(
                checkpoint["optimizer_vqgan_state_dict"]
            )
            optimizer_disc.load_state_dict(
                checkpoint["optimizer_disc_state_dict"]
            )

            if "scaler_vqgan_state_dict" in checkpoint:
                scaler_vqgan.load_state_dict(
                    checkpoint["scaler_vqgan_state_dict"]
                )
            if "scaler_disc_state_dict" in checkpoint:
                scaler_disc.load_state_dict(
                    checkpoint["scaler_disc_state_dict"]
                )

            start_epoch = int(checkpoint["epoch"]) + 1
            best_validation_loss = float(
                checkpoint.get(
                    "best_validation_loss",
                    float("inf"),
                )
            )

        model = DDP(
            model_raw,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )

        discriminator = DDP(
            discriminator_raw,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )

        if rank == 0:
            args.output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )
        dist.barrier()

        latest_path = (
            args.output_dir / "checkpoint_latest.pt"
        )
        best_path = (
            args.output_dir / "checkpoint_best.pt"
        )

        if rank == 0:
            print("Device mode: 2-GPU DDP")
            print(f"World size: {world_size}")
            print(
                f"GPU 0: {torch.cuda.get_device_name(0)}"
            )
            print(
                f"GPU 1: {torch.cuda.get_device_name(1)}"
            )
            print(f"Training samples: {len(train_dataset)}")
            print(
                f"Validation samples: {len(validation_dataset)}"
            )
            print(
                f"Training datasets: "
                f"{train_dataset.dataset_counts()}"
            )
            print(
                f"Validation datasets: "
                f"{validation_dataset.dataset_counts()}"
            )
            print(
                f"Normalization: [{min_db:.4f}, "
                f"{max_db:.4f}] dB"
            )
            print(
                f"Per-GPU batch size: {args.batch_size}"
            )
            print(
                f"Global physical batch size: "
                f"{global_batch_size}"
            )
            print(
                f"Gradient accumulation: "
                f"{args.gradient_accumulation_steps}"
            )
            print(
                f"Effective batch size: "
                f"{effective_batch_size}"
            )
            print(f"AMP dtype: {args.amp_dtype}")
            print(f"Epochs: {args.epochs}")
            print(f"Learning rate: {args.lr}")
            print(
                f"Correlation weight: {args.corr_weight}"
            )
            print(
                f"Adversarial weight: {args.adv_weight}"
            )
            print(
                "Discriminator starts at epoch: "
                f"{args.discriminator_start_epoch}"
            )
            print(
                "VQGAN parameters: "
                f"{sum(p.numel() for p in model_raw.parameters()):,}"
            )
            print(
                "Discriminator BatchNorm: SyncBatchNorm "
                "(global batch statistics)"
            )

            run_config = vars(args).copy()
            run_config.update(
                {
                    "data_root": str(args.data_root),
                    "train_manifest": str(
                        args.train_manifest
                    ),
                    "val_manifest": str(
                        args.val_manifest
                    ),
                    "normalization_file": str(
                        args.normalization_file
                    ),
                    "output_dir": str(
                        args.output_dir
                    ),
                    "resume": (
                        None
                        if args.resume is None
                        else str(args.resume)
                    ),
                    "min_db": min_db,
                    "max_db": max_db,
                    "normalization": normalization,
                    "ddp_world_size": world_size,
                    "per_gpu_batch_size": args.batch_size,
                    "global_batch_size": global_batch_size,
                    "effective_batch_size": (
                        effective_batch_size
                    ),
                    "sync_batchnorm_discriminator": True,
                }
            )

            with (
                args.output_dir / "run_config.json"
            ).open(
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    run_config,
                    handle,
                    indent=2,
                )

        for epoch in range(
            start_epoch,
            args.epochs + 1,
        ):
            # Required for a fresh deterministic distributed shuffle each epoch.
            train_sampler.set_epoch(epoch)

            epoch_start = time.monotonic()

            train_metrics = train_epoch_ddp(
                model=model,
                discriminator=discriminator,
                loader=train_loader,
                optimizer_vqgan=optimizer_vqgan,
                optimizer_disc=optimizer_disc,
                scaler_vqgan=scaler_vqgan,
                scaler_disc=scaler_disc,
                device=device,
                epoch=epoch,
                corr_weight=args.corr_weight,
                adv_weight=args.adv_weight,
                discriminator_start_epoch=(
                    args.discriminator_start_epoch
                ),
                gradient_accumulation_steps=(
                    args.gradient_accumulation_steps
                ),
                amp_dtype=args.amp_dtype,
                log_every=args.log_every,
                max_batches=args.max_train_batches,
                rank=rank,
            )

            # All ranks have identical synchronized model parameters here.
            dist.barrier()

            if rank == 0:
                assert validation_loader is not None

                # Validate the raw rank-0 VQGAN. This avoids any DDP
                # communication and preserves the original validation set.
                validation_metrics = validate_epoch(
                    model=model_raw,
                    loader=validation_loader,
                    device=device,
                    corr_weight=args.corr_weight,
                    amp_dtype=args.amp_dtype,
                    max_batches=args.max_val_batches,
                )

                elapsed_minutes = (
                    time.monotonic()
                    - epoch_start
                ) / 60.0

                improved = (
                    validation_metrics["loss"]
                    < best_validation_loss
                )

                if improved:
                    best_validation_loss = (
                        validation_metrics["loss"]
                    )

                payload = checkpoint_payload(
                    epoch=epoch,
                    args=args,
                    min_db=min_db,
                    max_db=max_db,
                    model=model_raw,
                    discriminator=discriminator_raw,
                    optimizer_vqgan=optimizer_vqgan,
                    optimizer_disc=optimizer_disc,
                    scaler_vqgan=scaler_vqgan,
                    scaler_disc=scaler_disc,
                    train_metrics=train_metrics,
                    validation_metrics=validation_metrics,
                    best_validation_loss=best_validation_loss,
                )

                # Correct DDP metadata while preserving the checkpoint's
                # model/discriminator key format (no "module." prefix).
                payload["batch_size"] = global_batch_size
                payload["per_gpu_batch_size"] = (
                    args.batch_size
                )
                payload["world_size"] = world_size
                payload["global_batch_size"] = (
                    global_batch_size
                )
                payload["effective_batch_size"] = (
                    effective_batch_size
                )
                payload["sync_batchnorm_discriminator"] = True

                torch.save(
                    payload,
                    latest_path,
                )

                if improved:
                    torch.save(
                        payload,
                        best_path,
                    )

                print(
                    f"Epoch {epoch:03d} | "
                    f"G={train_metrics['generator']:.4f} | "
                    f"D={train_metrics['discriminator']:.4f} | "
                    f"train_MAE="
                    f"{train_metrics['reconstruction']:.4f} | "
                    f"val={validation_metrics['loss']:.4f} | "
                    f"val_MAE="
                    f"{validation_metrics['reconstruction']:.4f} | "
                    f"val_corr_loss="
                    f"{validation_metrics['correlation']:.4f} | "
                    f"val_codes="
                    f"{int(validation_metrics['unique_codes'])} | "
                    f"minutes={elapsed_minutes:.1f}",
                    flush=True,
                )

            # Rank 1 waits while rank 0 validates/checkpoints.
            dist.barrier()

        if rank == 0:
            if not best_path.exists():
                raise RuntimeError(
                    "No best checkpoint was produced"
                )

            best_checkpoint = torch.load(
                best_path,
                map_location=device,
            )

            model_raw.load_state_dict(
                best_checkpoint["model_state_dict"],
                strict=True,
            )

            save_reconstruction(
                model=model_raw,
                dataset=validation_dataset,
                device=device,
                output_path=(
                    args.output_dir
                    / "reconstruction_best.png"
                ),
            )

            print(
                f"Best epoch: "
                f"{best_checkpoint['epoch']}"
            )
            print(
                f"Best validation loss: "
                f"{best_validation_loss:.4f}"
            )

        dist.barrier()

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
