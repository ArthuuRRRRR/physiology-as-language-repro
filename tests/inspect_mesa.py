import pyedflib

EDF_PATH = "/hdd2/kdpark/sleep_datasets/MESA_raw/polysomnography/edfs/mesa-sleep-3785.edf"

edf = pyedflib.EdfReader(EDF_PATH)

print(f"File: {EDF_PATH}")
print(f"Number of signals: {edf.signals_in_file}")
print(f"Duration: {edf.file_duration / 3600:.2f} hours")

print("\nChannels:")
for i, label in enumerate(edf.getSignalLabels()):
    fs = edf.getSampleFrequency(i)
    print(f"{i:02d} | {label:20s} | {fs:g} Hz")

edf.close()


