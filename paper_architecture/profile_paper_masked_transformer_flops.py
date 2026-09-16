"""Profile one training epoch of the paper cross-modal Transformer.

This script measures only the respiration-to-EEG Transformer:
  * respiration projection;
  * joint encoder;
  * decoder;
  * MLM head and masked-token cross-entropy;
  * backward pass.

It deliberately excludes the VQGAN, data loading, validation, gradient
clipping, and the AdamW parameter update.
"""

from __future__ import annotations

import argparse
import math
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from paper_architecture import PaperMaskedRespirationToEEGTransformer


DEFAULT_NUM_TRAINING_SAMPLES = 12_694


@contextmanager
def force_math_attention(device: torch.device):
    """Expose attention matrix multiplications to torch.profiler.

    Flash/memory-efficient attention kernels can hide their internal FLOPs
    from ``torch.profiler(with_flops=True)``. The math backend is therefore
    used during profiling. This changes the kernel implementation, not the
    mathematical operation count.
    """

    if device.type != "cuda":
        yield
        return

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(SDPBackend.MATH):
            yield
    except ImportError:
        # Compatibility with older PyTorch releases.
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_math=True,
            enable_mem_efficient=False,
        ):
            yield


def autocast_context(device: torch.device, enabled: bool):
    if device.type == "cuda" and enabled:
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )
    return nullcontext()


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def transformer_layer_flops(
    *,
    sequence_length: int,
    embedding_dim: int,
    feedforward_dim: int,
) -> int:
    """Dense forward FLOPs for one TransformerEncoderLayer.

    Convention: one multiplication and one addition count as two FLOPs.
    LayerNorm, GELU, dropout, softmax, masking and other elementwise
    operations are excluded from this analytical count.
    """

    n = sequence_length
    d = embedding_dim
    f = feedforward_dim

    # Q, K, V and attention output projections.
    attention_projections = 8 * n * d * d

    # QK^T and attention-probability x V.
    attention_products = 4 * n * n * d

    # d -> f and f -> d feed-forward projections.
    feedforward = 4 * n * d * f

    return attention_projections + attention_products + feedforward


def analytical_forward_flops(config: dict[str, int | float | bool]) -> dict[str, int]:
    """Calculate dense model-forward FLOPs for one sample."""

    respiration_samples = int(config["respiration_samples"])
    num_respiration_tokens = int(config["num_respiration_tokens"])
    eeg_grid_height = int(config["eeg_grid_height"])
    eeg_grid_width = int(config["eeg_grid_width"])
    codebook_size = int(config["codebook_size"])
    embedding_dim = int(config["embedding_dim"])
    num_encoder_layers = int(config["num_encoder_layers"])
    num_decoder_layers = int(config["num_decoder_layers"])
    mlp_ratio = float(config["mlp_ratio"])
    mask_ratio_min = float(config["mask_ratio_min"])

    num_eeg_tokens = eeg_grid_height * eeg_grid_width
    feedforward_dim = int(embedding_dim * mlp_ratio)

    # MAGE physically removes a fixed mask_ratio_min fraction before the
    # encoder. In this implementation: 512 EEG tokens -> 256 kept tokens.
    num_dropped = math.ceil(num_eeg_tokens * mask_ratio_min)
    num_kept_eeg = num_eeg_tokens - num_dropped
    encoder_length = num_respiration_tokens + num_kept_eeg
    decoder_length = num_respiration_tokens + num_eeg_tokens

    respiration_projection = (
        2
        * num_respiration_tokens
        * respiration_samples
        * embedding_dim
    )

    encoder_blocks = num_encoder_layers * transformer_layer_flops(
        sequence_length=encoder_length,
        embedding_dim=embedding_dim,
        feedforward_dim=feedforward_dim,
    )

    encoder_to_decoder = (
        2 * encoder_length * embedding_dim * embedding_dim
    )

    decoder_blocks = num_decoder_layers * transformer_layer_flops(
        sequence_length=decoder_length,
        embedding_dim=embedding_dim,
        feedforward_dim=feedforward_dim,
    )

    mlm_dense = 2 * num_eeg_tokens * embedding_dim * embedding_dim
    vocabulary_projection = (
        2 * num_eeg_tokens * embedding_dim * codebook_size
    )

    components = {
        "respiration_projection": respiration_projection,
        "encoder_blocks": encoder_blocks,
        "encoder_to_decoder": encoder_to_decoder,
        "decoder_blocks": decoder_blocks,
        "mlm_dense": mlm_dense,
        "vocabulary_projection": vocabulary_projection,
    }
    components["total"] = sum(components.values())
    components["encoder_length"] = encoder_length
    components["decoder_length"] = decoder_length
    components["feedforward_dim"] = feedforward_dim
    return components


def profiler_total_flops(profiler) -> int:
    return int(
        sum(event.flops or 0 for event in profiler.key_averages())
    )


def profile_forward(
    *,
    model: torch.nn.Module,
    respiration: torch.Tensor,
    eeg_tokens: torch.Tensor,
    mask_ratio: float | None,
    device: torch.device,
    amp: bool,
) -> int:
    model.train()

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with force_math_attention(device):
        with profile(
            activities=activities,
            with_flops=True,
            record_shapes=False,
            profile_memory=False,
        ) as profiler:
            with torch.no_grad():
                with autocast_context(device, amp):
                    model(
                        respiration=respiration,
                        eeg_tokens=eeg_tokens,
                        mask_ratio=mask_ratio,
                    )

            if device.type == "cuda":
                torch.cuda.synchronize()

    return profiler_total_flops(profiler)


def profile_training_step(
    *,
    model: torch.nn.Module,
    respiration: torch.Tensor,
    eeg_tokens: torch.Tensor,
    mask_ratio: float | None,
    device: torch.device,
    amp: bool,
) -> int:
    """Profile forward, masked-token CE and backward, without optimizer."""

    model.train()
    model.zero_grad(set_to_none=True)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with force_math_attention(device):
        with profile(
            activities=activities,
            with_flops=True,
            record_shapes=False,
            profile_memory=False,
        ) as profiler:
            with autocast_context(device, amp):
                outputs = model(
                    respiration=respiration,
                    eeg_tokens=eeg_tokens,
                    mask_ratio=mask_ratio,
                )
                loss = outputs["loss"]

            loss.backward()

            if device.type == "cuda":
                torch.cuda.synchronize()

    model.zero_grad(set_to_none=True)
    return profiler_total_flops(profiler)


def format_count(value: float) -> str:
    units = (
        (1e15, "PFLOPs"),
        (1e12, "TFLOPs"),
        (1e9, "GFLOPs"),
        (1e6, "MFLOPs"),
    )
    absolute_value = abs(value)
    for scale, label in units:
        if absolute_value >= scale:
            return f"{value / scale:.4f} {label}"
    return f"{value:.0f} FLOPs"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Profile forward and training FLOPs for the paper cross-modal "
            "respiration-to-EEG Transformer."
        )
    )
    parser.add_argument("--num-training-samples", type=int, default=DEFAULT_NUM_TRAINING_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mask-ratio", type=float, default=None)
    parser.add_argument("--mask-ratio-mu", type=float, default=0.55)
    parser.add_argument("--shared-temporal-position", action="store_true")
    parser.add_argument("--temporal-alignment-tag", action="store_true")
    parser.add_argument("--temporal-attention-mask", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--skip-profiler", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.num_training_samples < 1:
        raise ValueError("--num-training-samples must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.mask_ratio is not None and not 0.5 <= args.mask_ratio <= 1.0:
        raise ValueError("--mask-ratio must be in [0.5, 1.0]")
    if args.shared_temporal_position and args.temporal_alignment_tag:
        raise ValueError(
            "--shared-temporal-position and --temporal-alignment-tag "
            "cannot be enabled together"
        )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    use_cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device("cuda" if use_cuda else "cpu")
    amp = use_cuda and not args.no_amp

    model = PaperMaskedRespirationToEEGTransformer(
        mask_ratio_mu=args.mask_ratio_mu,
        shared_temporal_position=args.shared_temporal_position,
        temporal_alignment_tag=args.temporal_alignment_tag,
        temporal_attention_mask=args.temporal_attention_mask,
    ).to(device)

    config = model.get_config()
    total_parameters, trainable_parameters = count_parameters(model)
    analytical = analytical_forward_flops(config)

    batch_size = args.batch_size
    respiration = torch.randn(
        batch_size,
        int(config["num_respiration_tokens"]),
        int(config["respiration_samples"]),
        device=device,
    )
    eeg_tokens = torch.randint(
        low=0,
        high=int(config["codebook_size"]),
        size=(
            batch_size,
            int(config["eeg_grid_height"]),
            int(config["eeg_grid_width"]),
        ),
        dtype=torch.long,
        device=device,
    )

    print("\nCross-modal Transformer FLOPs")
    print("=" * 64)
    print(f"Device:                  {device}")
    print(f"AMP:                     {amp}")
    print(f"Profile batch size:      {batch_size}")
    print(f"Training samples/epoch:  {args.num_training_samples:,}")
    print(f"Parameters:              {total_parameters:,}")
    print(f"Trainable parameters:    {trainable_parameters:,}")
    print(f"Encoder sequence length: {analytical['encoder_length']}")
    print(f"Decoder sequence length: {analytical['decoder_length']}")
    print(f"Feed-forward dimension:  {analytical['feedforward_dim']}")

    analytical_forward = analytical["total"]
    analytical_training = 3 * analytical_forward
    analytical_epoch = analytical_training * args.num_training_samples

    print("\nAnalytical dense FLOPs (1 multiply + 1 add = 2 FLOPs)")
    print("-" * 64)
    print(f"Forward/sample:          {format_count(analytical_forward)}")
    print(f"Training/sample (~3x):   {format_count(analytical_training)}")
    print(f"Training/epoch (~3x):    {format_count(analytical_epoch)}")

    print("\nAnalytical forward breakdown per sample")
    print("-" * 64)
    for name in (
        "respiration_projection",
        "encoder_blocks",
        "encoder_to_decoder",
        "decoder_blocks",
        "mlm_dense",
        "vocabulary_projection",
    ):
        print(f"{name:26s} {format_count(analytical[name])}")

    if args.skip_profiler:
        print("\nProfiler skipped by --skip-profiler.")
        return

    # One unprofiled warm-up avoids including first-use kernel setup.
    model.zero_grad(set_to_none=True)
    with force_math_attention(device):
        with autocast_context(device, amp):
            warmup_outputs = model(
                respiration=respiration,
                eeg_tokens=eeg_tokens,
                mask_ratio=args.mask_ratio,
            )
        warmup_outputs["loss"].backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    model.zero_grad(set_to_none=True)

    measured_forward_batch = profile_forward(
        model=model,
        respiration=respiration,
        eeg_tokens=eeg_tokens,
        mask_ratio=args.mask_ratio,
        device=device,
        amp=amp,
    )
    measured_training_batch = profile_training_step(
        model=model,
        respiration=respiration,
        eeg_tokens=eeg_tokens,
        mask_ratio=args.mask_ratio,
        device=device,
        amp=amp,
    )

    measured_forward_sample = measured_forward_batch / batch_size
    measured_training_sample = measured_training_batch / batch_size
    measured_epoch = measured_training_sample * args.num_training_samples

    print("\nPyTorch profiler FLOPs")
    print("-" * 64)
    print(f"Forward/profiled batch:  {format_count(measured_forward_batch)}")
    print(f"Forward/sample:          {format_count(measured_forward_sample)}")
    print(f"Training/profiled batch: {format_count(measured_training_batch)}")
    print(f"Training/sample:         {format_count(measured_training_sample)}")
    print(f"Training/epoch:          {format_count(measured_epoch)}")

    if measured_forward_sample > 0:
        backward_ratio = (
            measured_training_sample / measured_forward_sample
        )
        print(f"Training/forward ratio:  {backward_ratio:.4f}x")

    print("\nScope")
    print("-" * 64)
    print("Included: Transformer forward, masked-token CE, backward.")
    print("Excluded: VQGAN, data loading, validation, gradient clipping,")
    print("          AdamW update and checkpoint I/O.")
    print("Note: analytical FLOPs are the primary architecture-level count;")
    print("      profiler FLOPs are a runtime cross-check and may omit some")
    print("      fused or elementwise operations depending on PyTorch.")


if __name__ == "__main__":
    main()
