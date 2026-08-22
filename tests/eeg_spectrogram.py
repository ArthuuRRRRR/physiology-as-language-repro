import numpy as np
import pyedflib
import matplotlib.pyplot as plt

from mne.time_frequency import psd_array_multitaper


EDF_PATH = "/hdd2/kdpark/sleep_datasets/MESA_raw/polysomnography/edfs/mesa-sleep-3785.edf"

EEG_CH = 6          # EEG3 = C4-M1
EPOCH_SEC = 30

# For this first test only:
START_HOUR = 1
DURATION_HOURS = 1

FMIN = 0.5
FMAX = 32.0


# --------------------------------------------------
# 1. Load EEG
# --------------------------------------------------

edf = pyedflib.EdfReader(EDF_PATH)

fs = edf.getSampleFrequency(EEG_CH)
eeg = edf.readSignal(EEG_CH)

edf.close()

print("Sampling rate:", fs, "Hz")
print("Full EEG shape:", eeg.shape)


# --------------------------------------------------
# 2. Select only one hour for the smoke test
# --------------------------------------------------

start_sample = int(START_HOUR * 3600 * fs)
end_sample = int((START_HOUR + DURATION_HOURS) * 3600 * fs)

eeg_test = eeg[start_sample:end_sample]

print("Selected duration:", len(eeg_test) / fs / 60, "minutes")


# --------------------------------------------------
# 3. Cut into 30-second epochs
# --------------------------------------------------

samples_per_epoch = int(EPOCH_SEC * fs)

n_epochs = len(eeg_test) // samples_per_epoch

eeg_test = eeg_test[:n_epochs * samples_per_epoch]

epochs = eeg_test.reshape(n_epochs, samples_per_epoch)

print("Epochs shape:", epochs.shape)
print("Number of 30-s epochs:", n_epochs)


# --------------------------------------------------
# 4. Multitaper PSD
# --------------------------------------------------

psd, freqs = psd_array_multitaper(
    epochs,
    sfreq=fs,
    fmin=FMIN,
    fmax=FMAX,
    verbose=False
)

print("PSD shape:", psd.shape)
print("Frequency shape:", freqs.shape)
print("Frequency range:", freqs[0], "-", freqs[-1], "Hz")


# --------------------------------------------------
# 5. Convert power to dB
# --------------------------------------------------

psd_db = 10 * np.log10(psd + 1e-12)

# Current shape:
# epochs x frequencies
#
# For a spectrogram we want:
# frequencies x time

spectrogram = psd_db.T

print("Spectrogram shape:", spectrogram.shape)


# --------------------------------------------------
# 6. Display
# --------------------------------------------------

time_minutes = np.arange(n_epochs) * EPOCH_SEC / 60

plt.figure(figsize=(14, 6))

plt.imshow(
    spectrogram,
    aspect="auto",
    origin="lower",
    extent=[
        time_minutes[0],
        time_minutes[-1] + EPOCH_SEC / 60,
        freqs[0],
        freqs[-1]
    ]
)

plt.xlabel("Time (minutes)")
plt.ylabel("Frequency (Hz)")
plt.title("MESA EEG3 (C4-M1) - Multitaper Spectrogram")

plt.colorbar(label="Power (dB)")
plt.tight_layout()
plt.show()
