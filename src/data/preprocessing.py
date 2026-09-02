import numpy as np
from mne.time_frequency import psd_array_multitaper
from scipy.interpolate import interp1d
from scipy.signal import resample_poly

EEG_EPOCH_SEC = 30
RESP_SEGMENT_SEC = 4 * 60
MODEL_WINDOW_SEC = 256 * 60
TARGET_RESP_FS = 10.0

def extract_synchronized_window(
    respiration,
    eeg,
    fs_resp,
    fs_eeg,
    start_sec,
    duration_sec=MODEL_WINDOW_SEC,
):
    """
    Extract the same time interval from respiration and EEG.
    """

    resp_start = int(start_sec * fs_resp)
    resp_end = int((start_sec + duration_sec) * fs_resp)

    eeg_start = int(start_sec * fs_eeg)
    eeg_end = int((start_sec + duration_sec) * fs_eeg)

    if resp_end > len(respiration):
        raise ValueError("Requested window exceeds respiration duration.")

    if eeg_end > len(eeg):
        raise ValueError("Requested window exceeds EEG duration.")

    resp_window = respiration[resp_start:resp_end]
    eeg_window = eeg[eeg_start:eeg_end]

    return resp_window, eeg_window


def segment_respiration(
    respiration,
    fs_resp,
    segment_sec=RESP_SEGMENT_SEC,
):
    """
    Split raw respiration into non-overlapping 4-minute segments.

    The paper keeps respiration as raw waveform and applies
    a learned linear projection later in the model.
    """

    samples_per_segment = int(segment_sec * fs_resp)

    n_segments = len(respiration) // samples_per_segment

    respiration = respiration[
        : n_segments * samples_per_segment
    ]

    segments = respiration.reshape(
        n_segments,
        samples_per_segment,
    )

    return segments

def resample_respiration(
    respiration,
    fs_resp,
    target_fs=TARGET_RESP_FS,
):
    """
    Resample respiration to a common sampling rate.

    This is a reproduction choice because the paper does not
    specify the respiration resampling rate.

    SHHS recordings are mostly 10 Hz, with a subset of SHHS2
    recorded at 8 Hz. We standardize respiration to 10 Hz so
    every 4-minute segment has the same number of samples.
    """

    if np.isclose(fs_resp, target_fs):
        return respiration.astype(np.float32), target_fs

    # SHHS case: 8 Hz -> 10 Hz = up 5 / down 4
    ratio = target_fs / fs_resp

    from fractions import Fraction

    frac = Fraction(ratio).limit_denominator(100)

    respiration = resample_poly(
        respiration,
        up=frac.numerator,
        down=frac.denominator,
    )

    return respiration.astype(np.float32), target_fs


def eeg_to_multitaper_spectrogram(
    eeg,
    fs_eeg,
    epoch_sec=EEG_EPOCH_SEC,
    fmin=0.5,
    fmax=32.0,
):
    """
    Convert EEG waveform into a multitaper PSD spectrogram.

    Output shape:
        frequency_bins x time_epochs
    """

    samples_per_epoch = int(epoch_sec * fs_eeg)

    n_epochs = len(eeg) // samples_per_epoch

    eeg = eeg[: n_epochs * samples_per_epoch]

    epochs = eeg.reshape(
        n_epochs,
        samples_per_epoch,
    )

    psd, freqs = psd_array_multitaper(
        epochs,
        sfreq=fs_eeg,
        fmin=fmin,
        fmax=fmax,
        verbose=False,
    )

    # MNE output:
    # time_epochs x frequency_bins

    # Desired spectrogram:
    # frequency_bins x time_epochs
    spectrogram = psd.T.astype(np.float32)

    return spectrogram, freqs.astype(np.float32)

def resample_frequency_axis(
    spectrogram,
    freqs,
    n_freq_bins=256,
):
    """
    Resample the frequency axis of a spectrogram to a fixed number
    of frequency bins using linear interpolation.

    Note
    ----
    The paper specifies a 256 x 512 VQGAN input but does not
    describe the exact frequency-resampling procedure.
    """

    target_freqs = np.linspace(
        freqs[0],
        freqs[-1],
        n_freq_bins,
        dtype=np.float32,
    )

    interpolator = interp1d(
        freqs,
        spectrogram,
        axis=0,
        kind="linear",
    )

    resized_spectrogram = interpolator(
        target_freqs
    ).astype(np.float32)

    return resized_spectrogram, target_freqs

def spectrogram_to_db(spectrogram):
    """
    Convert a PSD spectrogram to decibels without normalization.
    """

    eps = np.finfo(np.float32).tiny

    spectrogram_db = 10 * np.log10(
        np.maximum(spectrogram, eps)
    )

    return spectrogram_db.astype(np.float32)


def normalize_db_spectrogram(
    spectrogram_db,
    min_db,
    max_db,
):
    """
    Normalize an already converted dB spectrogram to [0, 1].

    min_db and max_db must be estimated from training data only.
    """

    if max_db <= min_db:
        raise ValueError(
            "max_db must be greater than min_db"
        )

    spectrogram_db = np.clip(
        spectrogram_db,
        min_db,
        max_db,
    )

    spectrogram_norm = (
        spectrogram_db - min_db
    ) / (
        max_db - min_db
    )

    return spectrogram_norm.astype(np.float32)


def normalize_spectrogram(
    spectrogram,
    min_db=-120.0,
    max_db=-70.0,
):
    """
    Compatibility function for the existing prototype pipeline.

    The default bounds remain provisional reproduction choices.
    """

    spectrogram_db = spectrogram_to_db(
        spectrogram
    )

    return normalize_db_spectrogram(
        spectrogram_db,
        min_db=min_db,
        max_db=max_db,
    )

def preprocess_pair(
    respiration,
    eeg,
    fs_resp,
    fs_eeg,
    start_sec,
    duration_sec=MODEL_WINDOW_SEC,
     output_scale="normalized",
):
    """
    Complete preprocessing of one synchronized respiration-EEG window.

    Returns
    -------
    resp_segments : np.ndarray
        Raw respiration segments.
        Expected working shape for 256 min: (64, samples_per_4min)

    eeg_spectrogram : np.ndarray
        Multitaper EEG spectrogram resampled on frequency axis.
        Expected shape: (256, 512)

    freqs : np.ndarray
        Final frequency axis.
    """

    resp_window, eeg_window = extract_synchronized_window(
        respiration=respiration,
        eeg=eeg,
        fs_resp=fs_resp,
        fs_eeg=fs_eeg,
        start_sec=start_sec,
        duration_sec=duration_sec,
    )

    resp_window, fs_resp = resample_respiration(
        resp_window,
        fs_resp,
    )

    resp_segments = segment_respiration(
        resp_window,
        fs_resp,
    )

    spectrogram, freqs = eeg_to_multitaper_spectrogram(
        eeg_window,
        fs_eeg,
    )

    spectrogram, freqs = resample_frequency_axis(
        spectrogram,
        freqs,
        n_freq_bins=256,
    )

    if output_scale == "db":
        spectrogram = spectrogram_to_db(
            spectrogram
        )

    elif output_scale == "normalized":
        spectrogram = normalize_spectrogram(
            spectrogram
        )

    else:
        raise ValueError(
            "output_scale must be 'db' or 'normalized'"
        )

    return resp_segments, spectrogram, freqs
