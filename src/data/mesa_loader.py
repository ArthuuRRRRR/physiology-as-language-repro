from pathlib import Path

import numpy as np
import pyedflib


DEFAULT_RESP_CHANNEL = "Abdo"
DEFAULT_EEG_CHANNEL = "EEG3"


def _find_channel(labels, channel_name):
    normalized = [label.strip().lower() for label in labels]
    target = channel_name.strip().lower()

    if target not in normalized:
        raise ValueError(
            f"Channel '{channel_name}' not found.\n"
            f"Available channels: {labels}"
        )

    return normalized.index(target)


def _eeg_to_volts(signal, unit):
    unit = unit.strip().lower()

    if unit in {"uv", "µv", "μv"}:
        return signal * 1e-6

    if unit == "mv":
        return signal * 1e-3

    if unit == "v":
        return signal

    raise ValueError(
        f"Unsupported EEG physical unit: '{unit}'"
    )


def load_mesa_pair(
    edf_path,
    resp_channel=DEFAULT_RESP_CHANNEL,
    eeg_channel=DEFAULT_EEG_CHANNEL,
):
    edf_path = Path(edf_path)

    if not edf_path.exists():
        raise FileNotFoundError(
            f"EDF file not found: {edf_path}"
        )

    edf = pyedflib.EdfReader(str(edf_path))

    try:
        labels = edf.getSignalLabels()

        resp_idx = _find_channel(
            labels,
            resp_channel,
        )

        eeg_idx = _find_channel(
            labels,
            eeg_channel,
        )

        respiration = edf.readSignal(
            resp_idx
        ).astype(np.float32)

        eeg = edf.readSignal(
            eeg_idx
        ).astype(np.float32)

        fs_resp = float(
            edf.getSampleFrequency(resp_idx)
        )

        fs_eeg = float(
            edf.getSampleFrequency(eeg_idx)
        )

        eeg_unit = edf.getPhysicalDimension(
            eeg_idx
        )

        eeg = _eeg_to_volts(
            eeg,
            eeg_unit,
        ).astype(np.float32)

        resp_duration = (
            len(respiration) / fs_resp
        )

        eeg_duration = (
            len(eeg) / fs_eeg
        )

        if abs(
            resp_duration - eeg_duration
        ) > 1.0:
            raise ValueError(
                "Respiration and EEG durations do not match: "
                f"{resp_duration:.2f}s vs "
                f"{eeg_duration:.2f}s"
            )

        return {
            "respiration": respiration,
            "eeg": eeg,
            "fs_resp": fs_resp,
            "fs_eeg": fs_eeg,
            "resp_label": labels[resp_idx],
            "eeg_label": labels[eeg_idx],
            "eeg_unit_original": eeg_unit,
            "duration_sec": min(
                resp_duration,
                eeg_duration,
            ),
        }

    finally:
        edf.close()