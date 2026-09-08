"""Evaluate a multi-dataset EEG VQGAN checkpoint by split and dataset."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.models.vqgan import VQGAN


class ManifestEEGDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        split: str,
        min_db: float,
        max_db: float,
    ) -> None:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        self.split = split
        self.min_db = float(min_db)
        self.max_db = float(max_db)
        self.entries: list[dict] = []

        with manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if "path" not in entry or "dataset" not in entry:
                    raise KeyError(
                        f"Invalid entry at {manifest_path}:{line_number}"
                    )
                self.entries.append(entry)

        if not self.entries:
            raise RuntimeError(f"Empty manifest: {manifest_path}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        entry = self.entries[index]
        path = Path(entry["path"])
        if not path.is_absolute():
            path = PROJECT_ROOT / path

        if not path.exists():
            raise FileNotFoundError(f"Sample not found: {path}")

        with np.load(path, allow_pickle=False) as sample:
            if "eeg_spectrogram_db" not in sample:
                raise KeyError(f"eeg_spectrogram_db missing from {path}")
            eeg_db = sample["eeg_spectrogram_db"].astype(
                np.float32,
                copy=False,
            )

        if eeg_db.shape != (256, 512):
            raise ValueError(f"Invalid shape {eeg_db.shape} in {path}")
        if not np.isfinite(eeg_db).all():
            raise ValueError(f"Non-finite spectrogram in {path}")

        eeg = (eeg_db - self.min_db) / (self.max_db - self.min_db)
        eeg = np.clip(eeg, 0.0, 1.0).astype(np.float32, copy=False)

        return {
            "eeg": torch.from_numpy(eeg.copy()),
            "dataset": str(entry["dataset"]),
            "split": self.split,
            "path": str(path),
        }


class MetricAccumulator:
    def __init__(self, codebook_size: int = 8192) -> None:
        self.samples = 0
        self.mae_sum = 0.0
        self.correlation_sum = 0.0
        self.temporal_correlation_sum = 0.0
        self.snr_sum = 0.0
        self.used_codes = torch.zeros(codebook_size, dtype=torch.bool)

    def add(
        self,
        mae: float,
        correlation: float,
        temporal_correlation: float,
        snr: float,
        indices: Tensor,
    ) -> None:
        self.samples += 1
        self.mae_sum += mae
        self.correlation_sum += correlation
        self.temporal_correlation_sum += temporal_correlation
        self.snr_sum += snr
        codes = torch.unique(indices.detach().cpu()).long()
        self.used_codes[codes] = True

    def result(self) -> dict[str, float | int]:
        if self.samples == 0:
            raise RuntimeError("Cannot summarize an empty metric accumulator")
        return {
            "samples": self.samples,
            "mae": self.mae_sum / self.samples,
            "correlation": self.correlation_sum / self.samples,
            "temporal_correlation": (
                self.temporal_correlation_sum / self.samples
            ),
            "snr_db": self.snr_sum / self.samples,
            "unique_codes": int(self.used_codes.sum()),
        }


def pearson_per_sample(x: Tensor, y: Tensor, eps: float = 1e-8) -> Tensor:
    x = x.float().flatten(start_dim=1)
    y = y.float().flatten(start_dim=1)
    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    numerator = (x * y).sum(dim=1)
    denominator = torch.sqrt(x.square().sum(dim=1) + eps) * torch.sqrt(
        y.square().sum(dim=1) + eps
    )
    return numerator / denominator.clamp_min(eps)


def temporal_pearson_per_sample(
    x: Tensor,
    y: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    # Mean spectral power at every 30-second time position.
    x_time = x.float().mean(dim=2).squeeze(1)
    y_time = y.float().mean(dim=2).squeeze(1)
    x_time = x_time - x_time.mean(dim=1, keepdim=True)
    y_time = y_time - y_time.mean(dim=1, keepdim=True)
    numerator = (x_time * y_time).sum(dim=1)
    denominator = torch.sqrt(x_time.square().sum(dim=1) + eps) * torch.sqrt(
        y_time.square().sum(dim=1) + eps
    )
    return numerator / denominator.clamp_min(eps)


def snr_per_sample(target: Tensor, prediction: Tensor, eps: float = 1e-8):
    signal_power = target.float().square().mean(dim=(1, 2, 3))
    noise_power = (target.float() - prediction.float()).square().mean(
        dim=(1, 2, 3)
    )
    return 10.0 * torch.log10((signal_power + eps) / (noise_power + eps))


def amp_context(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "none":
        return nullcontext()
    dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def save_example(
    target: np.ndarray,
    reconstruction: np.ndarray,
    dataset: str,
    split: str,
    source_path: str,
    output_dir: Path,
) -> str:
    difference = np.abs(target - reconstruction)
    figure, axes = plt.subplots(3, 1, figsize=(14, 9))
    axes[0].imshow(target, aspect="auto", origin="lower", vmin=0, vmax=1)
    axes[0].set_title("Ground-truth normalized EEG spectrogram")
    axes[1].imshow(
        reconstruction,
        aspect="auto",
        origin="lower",
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("VQGAN reconstruction")
    axes[2].imshow(difference, aspect="auto", origin="lower", vmin=0)
    axes[2].set_title("Absolute difference")
    for axis in axes:
        axis.set_xlabel("Time (30-second epochs)")
        axis.set_ylabel("Frequency bins")
    figure.suptitle(f"{split} / {dataset}\n{source_path}", fontsize=10)
    plt.tight_layout()

    output_path = output_dir / f"reconstruction_{safe_name(split)}_{safe_name(dataset)}.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return str(output_path)


@torch.no_grad()
def evaluate_manifest(
    model: VQGAN,
    dataset: ManifestEEGDataset,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: str,
    output_dir: Path,
    max_samples_per_dataset: int | None,
    log_every: int,
) -> dict[str, dict]:
    model.eval()
    accumulators: dict[str, MetricAccumulator] = defaultdict(MetricAccumulator)
    example_paths: dict[str, str] = {}
    evaluated_total = 0

    for batch in loader:
        target = batch["eeg"].unsqueeze(1).to(device, non_blocking=True)

        with amp_context(device, amp_dtype):
            reconstruction, indices, _ = model(target)

        mae = (reconstruction.float() - target.float()).abs().mean(
            dim=(1, 2, 3)
        )
        correlation = pearson_per_sample(reconstruction, target)
        temporal_correlation = temporal_pearson_per_sample(
            reconstruction,
            target,
        )
        snr = snr_per_sample(target, reconstruction)

        for item_index, dataset_name in enumerate(batch["dataset"]):
            accumulator = accumulators[dataset_name]
            if (
                max_samples_per_dataset is not None
                and accumulator.samples >= max_samples_per_dataset
            ):
                continue

            accumulator.add(
                mae=float(mae[item_index]),
                correlation=float(correlation[item_index]),
                temporal_correlation=float(temporal_correlation[item_index]),
                snr=float(snr[item_index]),
                indices=indices[item_index],
            )
            evaluated_total += 1

            if dataset_name not in example_paths:
                example_paths[dataset_name] = save_example(
                    target=target[item_index, 0].float().cpu().numpy(),
                    reconstruction=(
                        reconstruction[item_index, 0].float().cpu().numpy()
                    ),
                    dataset=dataset_name,
                    split=dataset.split,
                    source_path=batch["path"][item_index],
                    output_dir=output_dir,
                )

        if log_every > 0 and evaluated_total % log_every < target.shape[0]:
            print(f"Evaluated {evaluated_total} samples", flush=True)

        if max_samples_per_dataset is not None:
            manifest_datasets = {entry["dataset"] for entry in dataset.entries}
            if all(
                accumulators[name].samples >= max_samples_per_dataset
                for name in manifest_datasets
            ):
                break

    results = {
        name: accumulator.result()
        for name, accumulator in sorted(accumulators.items())
        if accumulator.samples > 0
    }
    for name, path in example_paths.items():
        results[name]["reconstruction_plot"] = path
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the shared multi-dataset EEG VQGAN."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/vqgan_multidataset/checkpoint_best.pt"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("outputs/vqgan_multidataset_preprocessed"),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["val", "test"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/vqgan_multidataset_evaluation"),
    )
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--amp-dtype",
        choices=["bfloat16", "float16", "none"],
        default="bfloat16",
    )
    parser.add_argument("--max-samples-per-dataset", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    checkpoint = torch.load(args.checkpoint, map_location=device)
    min_db = float(checkpoint["min_db"])
    max_db = float(checkpoint["max_db"])
    model = VQGAN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Normalization: [{min_db:.4f}, {max_db:.4f}] dB")

    all_results: dict[str, dict] = {}
    for split in args.splits:
        manifest_path = args.data_root / f"{split}_manifest.jsonl"
        dataset = ManifestEEGDataset(
            manifest_path=manifest_path,
            split=split,
            min_db=min_db,
            max_db=max_db,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
            drop_last=False,
        )

        print(f"\n{split.upper()}: {len(dataset)} manifest samples")
        split_results = evaluate_manifest(
            model=model,
            dataset=dataset,
            loader=loader,
            device=device,
            amp_dtype=args.amp_dtype,
            output_dir=args.output_dir,
            max_samples_per_dataset=args.max_samples_per_dataset,
            log_every=args.log_every,
        )
        all_results[split] = split_results

        for dataset_name, metrics in split_results.items():
            print(
                f"  {dataset_name} | n={metrics['samples']} | "
                f"MAE={metrics['mae']:.4f} | "
                f"corr={metrics['correlation']:.4f} | "
                f"temporal_corr={metrics['temporal_correlation']:.4f} | "
                f"SNR={metrics['snr_db']:.2f} dB | "
                f"codes={metrics['unique_codes']}"
            )

    output = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "min_db": min_db,
        "max_db": max_db,
        "amp_dtype": args.amp_dtype,
        "results": all_results,
    }
    metrics_path = args.output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(f"\nMetrics saved: {metrics_path}")


if __name__ == "__main__":
    main()
