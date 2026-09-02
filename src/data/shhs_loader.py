from pathlib import Path

import numpy as np
import pyedflib


# Exact EEG target used in the paper:
# SHHS EDF label "EEG" corresponds to C4-A1.
DEFAULT_EEG_CHANNEL = "EEG"

# Reproduction choice:
# The paper uses breathing-belt signals but does not specify
# which SHHS belt channel was selected.
# We use the abdominal belt by default and keep it configurable.
DEFAULT_RESP_CHANNEL = "ABDO RES"


def _find_channel(labels, channel_name):
    if channel_name not in labels:
        raise ValueError(
            f"Channel '{channel_name}' not found.\n"
            f"Available channels: {labels}"
        )

    return labels.index(channel_name)


def load_shhs_pair(
    edf_path,
    resp_channel=DEFAULT_RESP_CHANNEL,
    eeg_channel=DEFAULT_EEG_CHANNEL,
):
    """
    Load synchronized raw respiration and EEG signals from SHHS.

    Default channels
    ----------------
    Respiration:
        ABDO RES
        Abdominal respiratory belt.

    EEG:
        EEG
        Corresponds to C4-A1 in SHHS, which is the EEG
        channel used in the Physiology as Language paper.

    Returns
    -------
    dict containing:
        respiration
        eeg
        fs_resp
        fs_eeg
        resp_label
        eeg_label
        duration_sec
    """

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

        respiration = (
            edf.readSignal(resp_idx)
            .astype(np.float32)
        )

        eeg = (
            edf.readSignal(eeg_idx)
            .astype(np.float32)
        )

        # Standardize EEG amplitude to volts.
        # SHHS stores EEG physical values in microvolts (uV).
        eeg_unit = edf.getPhysicalDimension(eeg_idx).strip()

        if eeg_unit.lower() in {"uv", "µv", "μv"}:
            eeg *= 1e-6
        elif eeg_unit.lower() == "mv":
            eeg *= 1e-3
        elif eeg_unit.lower() == "v":
            pass
        else:
            raise ValueError(
                f"Unsupported EEG physical unit: {eeg_unit}"
            )

        fs_resp = float(
            edf.getSampleFrequency(resp_idx)
        )

        fs_eeg = float(
            edf.getSampleFrequency(eeg_idx)
        )

        resp_duration = (
            len(respiration) / fs_resp
        )

        eeg_duration = (
            len(eeg) / fs_eeg
        )

        # Both signals should describe the same PSG night.
        if abs(resp_duration - eeg_duration) > 1.0:
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
            "eeg_unit": "V",
            "eeg_original_unit": eeg_unit,
            "duration_sec": min(
                resp_duration,
                eeg_duration,
            ),
        }

    finally:
        edf.close()
