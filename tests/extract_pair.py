import pyedflib
import numpy as np
import matplotlib.pyplot as plt

EDF_PATH = "/hdd2/kdpark/sleep_datasets/MESA_raw/polysomnography/edfs/mesa-sleep-3785.edf"

RESP_CH = 10   # Thor
EEG_CH = 6     # EEG3 = C4-M1

edf = pyedflib.EdfReader(EDF_PATH)

resp = edf.readSignal(RESP_CH)
eeg = edf.readSignal(EEG_CH)

fs_resp = edf.getSampleFrequency(RESP_CH)
fs_eeg = edf.getSampleFrequency(EEG_CH)

print("\nPotential saturation:")

print(
    "Thor at min/max:",
    np.mean((resp <= resp.min()) | (resp >= resp.max())) * 100,
    "%"
)

print(
    "EEG at min/max:",
    np.mean((eeg <= eeg.min()) | (eeg >= eeg.max())) * 100,
    "%"
)

print("Respiration:")
print("  channel:", edf.getLabel(RESP_CH))
print("  fs:", fs_resp, "Hz")
print("  samples:", len(resp))
print("  duration:", len(resp) / fs_resp / 3600, "hours")
print("  min/max:", np.min(resp), np.max(resp))

print("\nEEG:")
print("  channel:", edf.getLabel(EEG_CH))
print("  fs:", fs_eeg, "Hz")
print("  samples:", len(eeg))
print("  duration:", len(eeg) / fs_eeg / 3600, "hours")
print("  min/max:", np.min(eeg), np.max(eeg))

print("\nSignal metadata:")

for ch in [RESP_CH, EEG_CH]:
    print(f"\n{edf.getLabel(ch)}")
    print("  physical dimension:", edf.getPhysicalDimension(ch))
    print("  physical min:", edf.getPhysicalMinimum(ch))
    print("  physical max:", edf.getPhysicalMaximum(ch))
    print("  digital min:", edf.getDigitalMinimum(ch))
    print("  digital max:", edf.getDigitalMaximum(ch))

edf.close()

# Display 2 minutes starting at hour 1
start_sec = 3600
duration_sec = 120

r0 = int(start_sec * fs_resp)
r1 = int((start_sec + duration_sec) * fs_resp)

e0 = int(start_sec * fs_eeg)
e1 = int((start_sec + duration_sec) * fs_eeg)

t_resp = np.arange(r1-r0) / fs_resp
t_eeg = np.arange(e1-e0) / fs_eeg

plt.figure(figsize=(14, 4))
plt.plot(t_resp, resp[r0:r1])
plt.title("Thoracic Respiration - 2 min")
plt.xlabel("Time (s)")
plt.show()

plt.figure(figsize=(14, 4))
plt.plot(t_eeg, eeg[e0:e1])
plt.title("EEG3 / C4-M1 - 2 min")
plt.xlabel("Time (s)")
plt.show()