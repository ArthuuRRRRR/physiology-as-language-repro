import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import (
    DataLoader,
    Dataset,
)

from src.data.dataset import PhysiologyPairDataset
from src.models.masked_transformer import (
    RespirationToEEGTransformer,
)
from src.models.mage_transformer import (
    MAGERespirationToEEGTransformer,
)
from src.models.vqgan import VQGAN


DEFAULT_TRANSFORMER_CHECKPOINT = Path(
    "outputs/masked_transformer_joint_long/"
    "held_out_fold_0/checkpoint_best.pt"
)

DEFAULT_VQGAN_CHECKPOINT = Path(
    "outputs/vqgan_adversarial_w0001/"
    "held_out_fold_0/checkpoint_best.pt"
)

DEFAULT_DATA_ROOT = Path(
    "outputs/shhs_preprocessed"
)


class ShuffledRespirationDataset(Dataset):
    """
    Replace each target's respiration with respiration
    from another evaluated participant.

    A deterministic circular permutation is selected so
    that no EEG target keeps respiration from the same
    participant.
    """

    def __init__(
        self,
        base_dataset,
        number_of_samples,
    ):
        self.base_dataset = base_dataset

        self.number_of_samples = min(
            number_of_samples,
            len(base_dataset),
        )

        if self.number_of_samples < 2:
            raise ValueError(
                "At least two samples are required "
                "for shuffled respiration"
            )

        self.target_indices = list(
            range(self.number_of_samples)
        )

        patient_ids = []

        for index in self.target_indices:
            sample = base_dataset[index]

            if "patient_id" not in sample:
                raise KeyError(
                    "patient_id is required for "
                    "participant-wise shuffling"
                )

            patient_ids.append(
                str(sample["patient_id"])
            )

        selected_shift = None

        # Find a circular shift for which every source
        # belongs to a different participant.
        for shift in range(
            1,
            self.number_of_samples,
        ):
            valid = all(
                patient_ids[index]
                != patient_ids[
                    (
                        index + shift
                    )
                    % self.number_of_samples
                ]
                for index in range(
                    self.number_of_samples
                )
            )

            if valid:
                selected_shift = shift
                break

        if selected_shift is None:
            raise RuntimeError(
                "Could not create a participant-wise "
                "respiration derangement"
            )

        self.source_indices = [
            (
                index + selected_shift
            )
            % self.number_of_samples
            for index in self.target_indices
        ]

        self.patient_ids = patient_ids
        self.shift = selected_shift

        print(
            "Respiration circular shift:",
            self.shift,
        )

    def __len__(self):
        return self.number_of_samples

    def __getitem__(self, index):
        target_index = self.target_indices[
            index
        ]

        source_index = self.source_indices[
            index
        ]

        target_sample = dict(
            self.base_dataset[target_index]
        )

        source_sample = self.base_dataset[
            source_index
        ]

        target_patient = str(
            target_sample["patient_id"]
        )

        source_patient = str(
            source_sample["patient_id"]
        )

        if target_patient == source_patient:
            raise RuntimeError(
                "Respiration source and EEG target "
                "belong to the same participant"
            )

        target_sample["respiration"] = (
            source_sample["respiration"]
        )

        target_sample[
            "respiration_source_patient_id"
        ] = source_patient

        return target_sample


def load_state_dict(
    checkpoint,
    possible_keys,
):
    """
    Retrieve a model state dictionary from a checkpoint.
    """

    for key in possible_keys:
        if key in checkpoint:
            return checkpoint[key]

    raise KeyError(
        "No model state dictionary found. "
        f"Available keys: {list(checkpoint.keys())}"
    )


def standardize_respiration(
    respiration,
    eps=1e-6,
):
    """
    Apply the same global per-window normalization
    used during Transformer training.
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
    ) / (std + eps)


def decode_token_ids(
    vqgan,
    token_ids,
):
    """
    Convert an EEG token grid into a reconstructed
    EEG spectrogram using the frozen VQGAN decoder.

    token_ids:
        (B, 8, 64)

    output:
        (B, 1, 256, 512)
    """

    codebook = F.normalize(
        vqgan.quantizer.codebook.weight,
        p=2,
        dim=1,
    )

    quantized = F.embedding(
        token_ids,
        codebook,
    )

    # (B, 8, 64, 32)
    # -> (B, 32, 8, 64)
    quantized = (
        quantized
        .permute(0, 3, 1, 2)
        .contiguous()
    )

    return vqgan.decode(quantized)


def sample_correlation(
    target,
    prediction,
    eps=1e-8,
):
    """
    Pearson correlation calculated separately
    for every spectrogram.
    """

    target = target.flatten(
        start_dim=1
    )

    prediction = prediction.flatten(
        start_dim=1
    )

    target = target - target.mean(
        dim=1,
        keepdim=True,
    )

    prediction = prediction - prediction.mean(
        dim=1,
        keepdim=True,
    )

    numerator = (
        target * prediction
    ).sum(dim=1)

    denominator = (
        torch.sqrt(
            (target ** 2).sum(dim=1)
            + eps
        )
        * torch.sqrt(
            (prediction ** 2).sum(dim=1)
            + eps
        )
    )

    return numerator / denominator


def temporal_correlation(
    target,
    prediction,
    eps=1e-8,
):
    """
    Measure temporal agreement after removing each
    frequency bin's temporal mean.
    """

    target = target - target.mean(
        dim=-1,
        keepdim=True,
    )

    prediction = prediction - prediction.mean(
        dim=-1,
        keepdim=True,
    )

    return sample_correlation(
        target,
        prediction,
        eps=eps,
    )


def sample_snr(
    target,
    prediction,
    eps=1e-8,
):
    """
    Signal-to-noise ratio in dB for each sample.
    """

    target = target.flatten(
        start_dim=1
    )

    prediction = prediction.flatten(
        start_dim=1
    )

    signal_power = (
        target ** 2
    ).mean(dim=1)

    noise_power = (
        (target - prediction) ** 2
    ).mean(dim=1)

    return 10.0 * torch.log10(
        (signal_power + eps)
        / (noise_power + eps)
    )


def save_comparison(
    ground_truth,
    oracle_reconstruction,
    predicted_reconstruction,
    output_path,
    zero_respiration,
    shuffle_respiration,
):
    """
    Save a visual comparison for the first sample.
    """

    ground_truth = (
        ground_truth[0, 0]
        .detach()
        .cpu()
        .numpy()
    )

    oracle_reconstruction = (
        oracle_reconstruction[0, 0]
        .detach()
        .cpu()
        .numpy()
    )

    predicted_reconstruction = (
        predicted_reconstruction[0, 0]
        .detach()
        .cpu()
        .numpy()
    )

    difference = np.abs(
        ground_truth
        - predicted_reconstruction
    )

    if shuffle_respiration:
        prediction_title = (
            "EEG reconstructed from "
            "another participant's respiration"
        )

    elif zero_respiration:
        prediction_title = (
            "EEG reconstructed with "
            "zero respiration"
        )

    else:
        prediction_title = (
            "EEG reconstructed from "
            "correct synchronized respiration"
        )

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(14, 12),
    )

    images = [
        ground_truth,
        oracle_reconstruction,
        predicted_reconstruction,
        difference,
    ]

    titles = [
        "Ground-truth EEG spectrogram",
        (
            "Frozen VQGAN reconstruction "
            "from true EEG tokens"
        ),
        prediction_title,
        (
            "Absolute difference: "
            "ground truth vs prediction"
        ),
    ]

    for index, axis in enumerate(axes):
        image_kwargs = {
            "aspect": "auto",
            "origin": "lower",
        }

        if index < 3:
            image_kwargs["vmin"] = 0
            image_kwargs["vmax"] = 1

        axis.imshow(
            images[index],
            **image_kwargs,
        )

        axis.set_title(
            titles[index]
        )

        axis.set_xlabel(
            "Time (30-second epochs)"
        )

        axis.set_ylabel(
            "Frequency bins"
        )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the joint respiration-to-EEG "
            "Masked Transformer."
        )
    )

    parser.add_argument(
        "--transformer-checkpoint",
        type=Path,
        default=(
            DEFAULT_TRANSFORMER_CHECKPOINT
        ),
    )

    parser.add_argument(
        "--vqgan-checkpoint",
        type=Path,
        default=DEFAULT_VQGAN_CHECKPOINT,
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    respiration_group = (
        parser.add_mutually_exclusive_group()
    )

    respiration_group.add_argument(
        "--zero-respiration",
        action="store_true",
        help=(
            "Replace normalized respiration "
            "with zeros."
        ),
    )

    respiration_group.add_argument(
        "--shuffle-respiration",
        action="store_true",
        help=(
            "Replace respiration with respiration "
            "from another validation participant."
        ),
    )

    args = parser.parse_args()

    if args.max_samples < 1:
        raise ValueError(
            "max-samples must be at least 1"
        )

    if not (
        args.transformer_checkpoint.exists()
    ):
        raise FileNotFoundError(
            "Transformer checkpoint not found: "
            f"{args.transformer_checkpoint}"
        )

    if not args.vqgan_checkpoint.exists():
        raise FileNotFoundError(
            "VQGAN checkpoint not found: "
            f"{args.vqgan_checkpoint}"
        )

    transformer_checkpoint = torch.load(
        args.transformer_checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    held_out_fold = int(
        transformer_checkpoint[
            "held_out_fold"
        ]
    )

    min_db = float(
        transformer_checkpoint["min_db"]
    )

    max_db = float(
        transformer_checkpoint["max_db"]
    )

    if args.output_dir is None:
        if args.shuffle_respiration:
            suffix = (
                "evaluation_shuffled_respiration"
            )

        elif args.zero_respiration:
            suffix = (
                "evaluation_zero_respiration"
            )

        else:
            suffix = "evaluation"

        output_dir = (
            args.transformer_checkpoint.parent
            / suffix
        )

    else:
        output_dir = args.output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)
    print(
        "Held-out fold:",
        held_out_fold,
    )
    print(
        "Checkpoint epoch:",
        transformer_checkpoint.get(
            "epoch"
        ),
    )
    print(
        "Architecture:",
        transformer_checkpoint.get(
            "architecture",
            "unknown",
        ),
    )
    print(
        "Zero respiration:",
        args.zero_respiration,
    )
    print(
        "Shuffled respiration:",
        args.shuffle_respiration,
    )

    validation_directory = (
        args.data_root
        / f"fold_{held_out_fold}"
    )

    dataset = PhysiologyPairDataset(
        validation_directory,
        min_db=min_db,
        max_db=max_db,
    )

    if args.shuffle_respiration:
        dataset = ShuffledRespirationDataset(
            base_dataset=dataset,
            number_of_samples=(
                args.max_samples
            ),
        )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    architecture = transformer_checkpoint.get(
        "architecture",
        "joint_concatenated_transformer",
    )

    if architecture == "mage_visible_encoder_decoder":
        transformer = (
            MAGERespirationToEEGTransformer()
            .to(device)
        )

    elif architecture in (
        None,
        "joint_concatenated_transformer",
    ):
        transformer = (
            RespirationToEEGTransformer()
            .to(device)
        )

    else:
        raise ValueError(
            "Unsupported Transformer architecture: "
            f"{architecture}"
        )

    transformer_state = load_state_dict(
        transformer_checkpoint,
        possible_keys=[
            "model_state_dict",
            "transformer_state_dict",
        ],
    )

    transformer.load_state_dict(
        transformer_state,
        strict=True,
    )

    transformer.eval()

    vqgan_checkpoint = torch.load(
        args.vqgan_checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    vqgan = VQGAN().to(device)

    vqgan_state = load_state_dict(
        vqgan_checkpoint,
        possible_keys=[
            "model_state_dict",
            "vqgan_state_dict",
            "generator_state_dict",
        ],
    )

    vqgan.load_state_dict(
        vqgan_state,
        strict=True,
    )

    vqgan.eval()

    for parameter in vqgan.parameters():
        parameter.requires_grad = False

    mask_token_id = getattr(
        transformer,
        "mask_token_id",
        8192,
    )

    total_cross_entropy = 0.0
    total_positions = 0
    total_correct = 0
    total_top5 = 0

    total_mae = 0.0
    total_correlation = 0.0
    total_temporal_correlation = 0.0
    total_snr = 0.0

    total_oracle_mae = 0.0
    total_oracle_correlation = 0.0
    total_oracle_temporal_correlation = 0.0

    target_codes = set()
    predicted_codes = set()

    evaluated_samples = 0
    comparison_saved = False

    with torch.inference_mode():
        for batch in loader:
            if (
                evaluated_samples
                >= args.max_samples
            ):
                break

            respiration = (
                batch["respiration"]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            eeg = (
                batch["eeg_spectrogram"]
                .unsqueeze(1)
                .to(
                    device,
                    non_blocking=True,
                )
            )

            # Shuffled samples are normalized only after
            # the complete respiration window is replaced.
            respiration = (
                standardize_respiration(
                    respiration
                )
            )

            if args.zero_respiration:
                respiration = torch.zeros_like(
                    respiration
                )

            _, true_tokens, _ = vqgan.encode(
                eeg
            )

            masked_tokens = torch.full_like(
                true_tokens,
                fill_value=mask_token_id,
            )

            logits = transformer(
                respiration,
                masked_tokens,
            )

            targets_flat = true_tokens.flatten(
                start_dim=1
            )

            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    logits.shape[-1],
                ),
                targets_flat.reshape(-1),
                reduction="sum",
            )

            predicted_flat = logits.argmax(
                dim=-1
            )

            predicted_tokens = (
                predicted_flat.view_as(
                    true_tokens
                )
            )

            top5_predictions = logits.topk(
                k=5,
                dim=-1,
            ).indices

            correct = (
                predicted_flat
                == targets_flat
            )

            top5_correct = (
                top5_predictions
                == targets_flat.unsqueeze(-1)
            ).any(dim=-1)

            oracle_reconstruction = (
                decode_token_ids(
                    vqgan,
                    true_tokens,
                )
            )

            predicted_reconstruction = (
                decode_token_ids(
                    vqgan,
                    predicted_tokens,
                )
            )

            batch_size = eeg.shape[0]
            positions = targets_flat.numel()

            total_cross_entropy += (
                loss.item()
            )

            total_positions += positions

            total_correct += (
                correct.sum().item()
            )

            total_top5 += (
                top5_correct.sum().item()
            )

            total_mae += (
                torch.abs(
                    eeg
                    - predicted_reconstruction
                )
                .flatten(start_dim=1)
                .mean(dim=1)
                .sum()
                .item()
            )

            total_correlation += (
                sample_correlation(
                    eeg,
                    predicted_reconstruction,
                )
                .sum()
                .item()
            )

            total_temporal_correlation += (
                temporal_correlation(
                    eeg,
                    predicted_reconstruction,
                )
                .sum()
                .item()
            )

            total_snr += (
                sample_snr(
                    eeg,
                    predicted_reconstruction,
                )
                .sum()
                .item()
            )

            total_oracle_mae += (
                torch.abs(
                    eeg
                    - oracle_reconstruction
                )
                .flatten(start_dim=1)
                .mean(dim=1)
                .sum()
                .item()
            )

            total_oracle_correlation += (
                sample_correlation(
                    eeg,
                    oracle_reconstruction,
                )
                .sum()
                .item()
            )

            total_oracle_temporal_correlation += (
                temporal_correlation(
                    eeg,
                    oracle_reconstruction,
                )
                .sum()
                .item()
            )

            target_codes.update(
                true_tokens
                .detach()
                .cpu()
                .reshape(-1)
                .tolist()
            )

            predicted_codes.update(
                predicted_tokens
                .detach()
                .cpu()
                .reshape(-1)
                .tolist()
            )

            if not comparison_saved:
                save_comparison(
                    ground_truth=eeg,
                    oracle_reconstruction=(
                        oracle_reconstruction
                    ),
                    predicted_reconstruction=(
                        predicted_reconstruction
                    ),
                    output_path=(
                        output_dir
                        / "reconstruction_comparison.png"
                    ),
                    zero_respiration=(
                        args.zero_respiration
                    ),
                    shuffle_respiration=(
                        args.shuffle_respiration
                    ),
                )

                comparison_saved = True

            evaluated_samples += batch_size

            if (
                evaluated_samples % 25
                == 0
            ):
                print(
                    "Evaluated:",
                    evaluated_samples,
                )

    metrics = {
        "transformer_checkpoint": str(
            args.transformer_checkpoint
        ),
        "vqgan_checkpoint": str(
            args.vqgan_checkpoint
        ),
        "held_out_fold": held_out_fold,
        "checkpoint_epoch": (
            transformer_checkpoint.get(
                "epoch"
            )
        ),
        "zero_respiration": (
            args.zero_respiration
        ),
        "shuffle_respiration": (
            args.shuffle_respiration
        ),
        "evaluated_samples": (
            evaluated_samples
        ),
        "cross_entropy": (
            total_cross_entropy
            / total_positions
        ),
        "token_accuracy": (
            total_correct
            / total_positions
        ),
        "token_top5_accuracy": (
            total_top5
            / total_positions
        ),
        "predicted_eeg_mae": (
            total_mae
            / evaluated_samples
        ),
        "predicted_eeg_correlation": (
            total_correlation
            / evaluated_samples
        ),
        "predicted_eeg_temporal_correlation": (
            total_temporal_correlation
            / evaluated_samples
        ),
        "predicted_eeg_snr_db": (
            total_snr
            / evaluated_samples
        ),
        "oracle_vqgan_mae": (
            total_oracle_mae
            / evaluated_samples
        ),
        "oracle_vqgan_correlation": (
            total_oracle_correlation
            / evaluated_samples
        ),
        "oracle_vqgan_temporal_correlation": (
            total_oracle_temporal_correlation
            / evaluated_samples
        ),
        "unique_target_codes": len(
            target_codes
        ),
        "unique_predicted_codes": len(
            predicted_codes
        ),
    }

    metrics_path = (
        output_dir / "metrics.json"
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    print()
    print(
        "Evaluated samples:",
        evaluated_samples,
    )
    print(
        "Cross-entropy:",
        f"{metrics['cross_entropy']:.4f}",
    )
    print(
        "Token accuracy:",
        f"{metrics['token_accuracy']:.4f}",
    )
    print(
        "Token top-5 accuracy:",
        (
            f"{metrics['token_top5_accuracy']:.4f}"
        ),
    )
    print(
        "Predicted EEG MAE:",
        f"{metrics['predicted_eeg_mae']:.4f}",
    )
    print(
        "Predicted EEG correlation:",
        (
            f"{metrics['predicted_eeg_correlation']:.4f}"
        ),
    )
    print(
        "Predicted EEG temporal correlation:",
        (
            f"{metrics['predicted_eeg_temporal_correlation']:.4f}"
        ),
    )
    print(
        "Predicted EEG SNR:",
        (
            f"{metrics['predicted_eeg_snr_db']:.2f} dB"
        ),
    )
    print(
        "Oracle VQGAN MAE:",
        f"{metrics['oracle_vqgan_mae']:.4f}",
    )
    print(
        "Oracle VQGAN correlation:",
        (
            f"{metrics['oracle_vqgan_correlation']:.4f}"
        ),
    )
    print(
        "Oracle temporal correlation:",
        (
            f"{metrics['oracle_vqgan_temporal_correlation']:.4f}"
        ),
    )
    print(
        "Unique target codes:",
        metrics["unique_target_codes"],
    )
    print(
        "Unique predicted codes:",
        metrics["unique_predicted_codes"],
    )
    print(
        "Comparison saved:",
        output_dir
        / "reconstruction_comparison.png",
    )
    print(
        "Metrics saved:",
        metrics_path,
    )


if __name__ == "__main__":
    main()