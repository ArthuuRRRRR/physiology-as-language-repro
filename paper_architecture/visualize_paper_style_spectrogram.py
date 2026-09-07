#!/usr/bin/env python3
"""
Visualize EEG spectrograms in a paper-like style.

This script supports two modes:

1) Spectrogram-matrix mode:
   - Input is already a 2D spectrogram array (freq x time).
   - Useful for GT / oracle / predicted spectrograms coming from the VQGAN pipeline.

2) Raw-waveform mode:
   - Input is a 1D EEG waveform.
   - The script computes a multitaper spectrogram using non-overlapping windows,
     similarly to the display logic found in Keondo's loader.

Typical use cases:
- Re-display GT / oracle / predicted spectrograms with cleaner paper-style rendering.
- Produce a figure that visually matches the paper much more closely.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

try:
    from mne.time_frequency import psd_array_multitaper
except ImportError:
    psd_array_multitaper = None


# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------
def load_array(path: str, key: Optional[str] = None) -> np.ndarray:
    """
    Load a .npy or .npz array.

    If .npz is provided and key is None:
      - if there is a single array, load it automatically
      - otherwise raise an error
    """
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()

    if suffix == ".npy":
        arr = np.load(path_obj)
        return np.asarray(arr)

    if suffix == ".npz":
        data = np.load(path_obj)
        if key is not None:
            if key not in data:
                raise KeyError(f"Key '{key}' not found in {path}")
            return np.asarray(data[key])

        keys = list(data.keys())
        if len(keys) == 1:
            return np.asarray(data[keys[0]])

        raise ValueError(
            f"{path} contains multiple arrays {keys}. "
            f"Please provide --gt-key / --oracle-key / --pred-key."
        )

    raise ValueError(f"Unsupported file type: {path}")


# ---------------------------------------------------------------------
# Spectrogram computation
# ---------------------------------------------------------------------
def compute_multitaper_spectrogram(
    eeg_signal: np.ndarray,
    sfreq: float,
    window_sec: float = 5.0,
    fmin: float = 0.5,
    fmax: float = 32.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a non-overlapping multitaper spectrogram from a raw EEG waveform.

    Returns
    -------
    spec_db : np.ndarray
        Shape (freq_bins, time_bins), in dB.
    freqs : np.ndarray
        Frequency axis in Hz.
    """
    if psd_array_multitaper is None:
        raise ImportError(
            "mne is required for raw-waveform spectrogram computation. "
            "Install it with: pip install mne"
        )

    eeg_signal = np.asarray(eeg_signal).squeeze()
    if eeg_signal.ndim != 1:
        raise ValueError(
            f"Raw waveform must be 1D after squeeze, got shape {eeg_signal.shape}"
        )

    window_samples = int(round(window_sec * sfreq))
    if window_samples <= 0:
        raise ValueError("window_sec * sfreq must be > 0")

    num_windows = len(eeg_signal) // window_samples
    if num_windows == 0:
        raise ValueError("Signal too short for the requested window size")

    trimmed = eeg_signal[: num_windows * window_samples]
    chunks = trimmed.reshape(num_windows, window_samples)

    all_psd = []
    freqs = None

    for chunk in chunks:
        psd, freqs = psd_array_multitaper(
            chunk[np.newaxis, :],
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            adaptive=True,
            normalization="full",
            verbose=False,
        )
        psd = psd[0]  # (freq_bins,)
        psd_db = 10.0 * np.log10(np.maximum(psd, 1e-12))
        all_psd.append(psd_db)

    # time is columns
    spec_db = np.stack(all_psd, axis=1)  # (freq_bins, time_bins)
    return spec_db, freqs


# ---------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------
def percentile_normalize(
    spec: np.ndarray,
    pmin: float = 1.0,
    pmax: float = 99.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> tuple[np.ndarray, float, float]:
    """
    Normalize a spectrogram to [0, 1] using percentile clipping.
    """
    spec = np.asarray(spec, dtype=np.float32)

    if vmin is None:
        vmin = float(np.percentile(spec, pmin))
    if vmax is None:
        vmax = float(np.percentile(spec, pmax))

    if vmax <= vmin:
        vmax = vmin + 1e-8

    spec_norm = (spec - vmin) / (vmax - vmin)
    spec_norm = np.clip(spec_norm, 0.0, 1.0)
    return spec_norm, vmin, vmax


# ---------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------
def parse_intervals(interval_string: Optional[str]) -> list[tuple[float, float]]:
    """
    Parse intervals like:
      "60:120,150:210,240:300"
    into:
      [(60,120), (150,210), (240,300)]
    """
    if not interval_string:
        return []

    intervals = []
    for item in interval_string.split(","):
        item = item.strip()
        if not item:
            continue
        start_str, end_str = item.split(":")
        intervals.append((float(start_str), float(end_str)))
    return intervals


def draw_slow_wave_boxes(
    ax: plt.Axes,
    intervals_min: list[tuple[float, float]],
    y_bottom: float,
    box_height_hz: float = 1.8,
    linewidth: float = 2.0,
):
    """
    Draw black boxes near the bottom of the spectrogram, similar to the paper figure.
    """
    for start_min, end_min in intervals_min:
        rect = plt.Rectangle(
            (start_min, y_bottom),
            end_min - start_min,
            box_height_hz,
            fill=False,
            edgecolor="black",
            linewidth=linewidth,
        )
        ax.add_patch(rect)


def render_panel(
    ax: plt.Axes,
    spec_img: np.ndarray,
    title: str,
    total_minutes: float,
    fmin_plot: float,
    fmax_plot: float,
    cmap: str,
    slow_wave_intervals: list[tuple[float, float]],
):
    ax.imshow(
        spec_img,
        origin="lower",
        aspect="auto",
        extent=[0, total_minutes, fmin_plot, fmax_plot],
        cmap=cmap,
    )
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Frequency (Hz)", fontsize=12)
    ax.set_xlabel("Time (min)", fontsize=12)

    if slow_wave_intervals:
        draw_slow_wave_boxes(
            ax=ax,
            intervals_min=slow_wave_intervals,
            y_bottom=max(fmin_plot, 0.5),
        )


def prepare_spectrogram_input(
    path: str,
    kind: str,
    key: Optional[str],
    sfreq: Optional[float],
    window_sec: float,
    fmin: float,
    fmax: float,
) -> tuple[np.ndarray, float, float]:
    """
    Returns:
      raw_spec_db : 2D array (freq x time)
      inferred_frame_sec : time duration represented by one column
      inferred_fmax_plot : upper frequency axis
    """
    arr = load_array(path, key=key)

    if kind == "waveform":
        if sfreq is None:
            raise ValueError(
                f"{path}: --sfreq is required when kind=waveform"
            )
        spec_db, freqs = compute_multitaper_spectrogram(
            eeg_signal=arr,
            sfreq=sfreq,
            window_sec=window_sec,
            fmin=fmin,
            fmax=fmax,
        )
        inferred_frame_sec = window_sec
        inferred_fmax_plot = float(freqs[-1])
        return spec_db, inferred_frame_sec, inferred_fmax_plot

    if kind == "spectrogram":
        arr = np.asarray(arr)
        arr = np.squeeze(arr)

        if arr.ndim != 2:
            raise ValueError(
                f"{path}: spectrogram input must be 2D, got shape {arr.shape}"
            )
        return arr, -1.0, fmax

    raise ValueError(f"Unknown kind: {kind}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper-style EEG spectrogram figures."
    )

    # Required GT input
    parser.add_argument("--gt-path", required=True, help="Path to GT .npy or .npz")
    parser.add_argument(
        "--gt-kind",
        default="spectrogram",
        choices=["spectrogram", "waveform"],
        help="Whether GT input is already a spectrogram matrix or a raw waveform",
    )
    parser.add_argument("--gt-key", default=None, help="Optional npz key for GT")

    # Optional oracle
    parser.add_argument("--oracle-path", default=None, help="Optional oracle input")
    parser.add_argument(
        "--oracle-kind",
        default="spectrogram",
        choices=["spectrogram", "waveform"],
        help="Whether oracle input is a spectrogram or waveform",
    )
    parser.add_argument(
        "--oracle-key",
        default=None,
        help="Optional npz key for oracle input",
    )

    # Optional prediction
    parser.add_argument("--pred-path", default=None, help="Optional predicted input")
    parser.add_argument(
        "--pred-kind",
        default="spectrogram",
        choices=["spectrogram", "waveform"],
        help="Whether prediction input is a spectrogram or waveform",
    )
    parser.add_argument(
        "--pred-key",
        default=None,
        help="Optional npz key for prediction input",
    )

    # Spectrogram params
    parser.add_argument(
        "--sfreq",
        type=float,
        default=None,
        help="Sampling rate, required if any input kind=waveform",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=5.0,
        help="Window length (sec) for raw-waveform multitaper spectrogram",
    )
    parser.add_argument(
        "--frame-sec",
        type=float,
        default=30.0,
        help="Seconds per time-bin when input is already a spectrogram matrix "
             "(for your current VQGAN/paper pipeline, 30 sec is appropriate)",
    )
    parser.add_argument("--fmin", type=float, default=0.5)
    parser.add_argument("--fmax", type=float, default=32.0)

    # Display params
    parser.add_argument("--pmin", type=float, default=1.0)
    parser.add_argument("--pmax", type=float, default=99.0)
    parser.add_argument(
        "--shared-normalization",
        action="store_true",
        help="Use a common percentile normalization range across all panels",
    )
    parser.add_argument(
        "--cmap",
        default="turbo",
        help="Matplotlib colormap (turbo works well for paper-style EEG plots)",
    )
    parser.add_argument(
        "--slow-wave-intervals",
        default=None,
        help='Optional boxes in minutes, e.g. "60:120,150:210,240:300"',
    )

    # Output
    parser.add_argument(
        "--output-path",
        required=True,
        help="Where to save the rendered figure",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also display the figure interactively",
    )

    args = parser.parse_args()

    slow_wave_intervals = parse_intervals(args.slow_wave_intervals)

    # -----------------------------------------------------------------
    # Load inputs
    # -----------------------------------------------------------------
    panels_raw = []

    gt_raw, gt_frame_sec, gt_fmax_plot = prepare_spectrogram_input(
        path=args.gt_path,
        kind=args.gt_kind,
        key=args.gt_key,
        sfreq=args.sfreq,
        window_sec=args.window_sec,
        fmin=args.fmin,
        fmax=args.fmax,
    )
    gt_frame_sec = args.frame_sec if args.gt_kind == "spectrogram" else gt_frame_sec
    panels_raw.append(("Ground-truth EEG spectrogram", gt_raw, gt_frame_sec, gt_fmax_plot))

    if args.oracle_path:
        oracle_raw, oracle_frame_sec, oracle_fmax_plot = prepare_spectrogram_input(
            path=args.oracle_path,
            kind=args.oracle_kind,
            key=args.oracle_key,
            sfreq=args.sfreq,
            window_sec=args.window_sec,
            fmin=args.fmin,
            fmax=args.fmax,
        )
        oracle_frame_sec = (
            args.frame_sec if args.oracle_kind == "spectrogram" else oracle_frame_sec
        )
        panels_raw.append((
            "Frozen VQGAN reconstruction from true EEG tokens",
            oracle_raw,
            oracle_frame_sec,
            oracle_fmax_plot,
        ))

    pred_raw = None
    pred_frame_sec = None
    pred_fmax_plot = None
    if args.pred_path:
        pred_raw, pred_frame_sec, pred_fmax_plot = prepare_spectrogram_input(
            path=args.pred_path,
            kind=args.pred_kind,
            key=args.pred_key,
            sfreq=args.sfreq,
            window_sec=args.window_sec,
            fmin=args.fmin,
            fmax=args.fmax,
        )
        pred_frame_sec = (
            args.frame_sec if args.pred_kind == "spectrogram" else pred_frame_sec
        )
        panels_raw.append((
            "EEG reconstructed from respiration-predicted tokens",
            pred_raw,
            pred_frame_sec,
            pred_fmax_plot,
        ))

    # -----------------------------------------------------------------
    # Shared normalization if requested
    # -----------------------------------------------------------------
    if args.shared_normalization:
        all_values = np.concatenate([panel[1].ravel() for panel in panels_raw], axis=0)
        shared_vmin = float(np.percentile(all_values, args.pmin))
        shared_vmax = float(np.percentile(all_values, args.pmax))
    else:
        shared_vmin = None
        shared_vmax = None

    panels_norm = []
    for title, raw_spec, frame_sec, fmax_plot in panels_raw:
        spec_norm, _, _ = percentile_normalize(
            raw_spec,
            pmin=args.pmin,
            pmax=args.pmax,
            vmin=shared_vmin,
            vmax=shared_vmax,
        )
        panels_norm.append((title, spec_norm, frame_sec, fmax_plot))

    # Add absolute difference panel if GT and prediction are both available
    if pred_raw is not None:
        if gt_raw.shape != pred_raw.shape:
            raise ValueError(
                f"GT and prediction must have the same shape for difference plot: "
                f"{gt_raw.shape} vs {pred_raw.shape}"
            )
        diff_raw = np.abs(gt_raw - pred_raw)
        diff_norm, _, _ = percentile_normalize(
            diff_raw,
            pmin=args.pmin,
            pmax=args.pmax,
        )
        panels_norm.append((
            "Absolute difference: ground truth vs prediction",
            diff_norm,
            gt_frame_sec,
            gt_fmax_plot,
        ))

    # -----------------------------------------------------------------
    # Render
    # -----------------------------------------------------------------
    n_panels = len(panels_norm)
    fig_height = 3.4 * n_panels
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(16, fig_height),
        squeeze=False,
    )
    axes = axes[:, 0]

    for ax, (title, spec_img, frame_sec, fmax_plot) in zip(axes, panels_norm):
        total_minutes = (spec_img.shape[1] * frame_sec) / 60.0
        render_panel(
            ax=ax,
            spec_img=spec_img,
            title=title,
            total_minutes=total_minutes,
            fmin_plot=args.fmin,
            fmax_plot=fmax_plot,
            cmap=args.cmap,
            slow_wave_intervals=slow_wave_intervals,
        )

    plt.tight_layout()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Figure saved to: {output_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
