from pathlib import Path

import numpy as np
import pyedflib


DEFAULT_RESP_CHANNEL = "Thor"
DEFAULT_EEG_CHANNEL = "EEG3"


def _find_channel(labels, channel_name):
    """
    Return the index of a channel from its EDF label.
    """
    if channel_name not in labels:
        raise ValueError(
            f"Channel '{channel_name}' not found.\n"
            f"Available channels: {labels}"
        )

    return labels.index(channel_name)


def load_mesa_pair(
    edf_path,
    resp_channel=DEFAULT_RESP_CHANNEL,
    eeg_channel=DEFAULT_EEG_CHANNEL,
):
    """
    Load synchronized respiration and EEG signals from one MESA EDF file.

    Parameters
    ----------
    edf_path : str or Path
        Path to the MESA EDF file.

    resp_channel : str
        Respiration channel name.
        Default: Thor.

    eeg_channel : str
        EEG channel name.
        Default: EEG3 (C4-M1 in MESA).

    Returns
    -------
    dict
        Dictionary containing:
        - respiration
        - eeg
        - fs_resp
        - fs_eeg
        - resp_label
        - eeg_label
        - duration_sec
    """

    edf_path = Path(edf_path)

    if not edf_path.exists():
        raise FileNotFoundError(f"EDF file not found: {edf_path}")

    edf = pyedflib.EdfReader(str(edf_path))

    try:
        labels = edf.getSignalLabels()

        resp_idx = _find_channel(labels, resp_channel)
        eeg_idx = _find_channel(labels, eeg_channel)

        respiration = edf.readSignal(resp_idx).astype(np.float32)
        eeg = edf.readSignal(eeg_idx).astype(np.float32)

        fs_resp = float(edf.getSampleFrequency(resp_idx))
        fs_eeg = float(edf.getSampleFrequency(eeg_idx))

        resp_duration = len(respiration) / fs_resp
        eeg_duration = len(eeg) / fs_eeg

        # Check that both signals cover the same recording duration
        if abs(resp_duration - eeg_duration) > 1.0:
            raise ValueError(
                "Respiration and EEG durations do not match: "
                f"{resp_duration:.2f}s vs {eeg_duration:.2f}s"
            )

        return {
            "respiration": respiration,
            "eeg": eeg,
            "fs_resp": fs_resp,
            "fs_eeg": fs_eeg,
            "resp_label": labels[resp_idx],
            "eeg_label": labels[eeg_idx],
            "duration_sec": min(resp_duration, eeg_duration),
        }

    finally:
        edf.close()
