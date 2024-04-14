
import matplotlib.pyplot as plt
import numpy as np
import mne
from mne.preprocessing import ICA
from mne.preprocessing import create_eog_epochs, create_ecg_epochs
import matplotlib

# Load the EDF file
file_path = './chb01_01.edf'
raw = mne.io.read_raw_edf(file_path, preload=True)
raw.plot(block=True)

print(raw)
print(raw.info)
print(raw.info["bads"])
#plot -psd
raw.compute_psd(fmax=64).plot(picks="data", exclude="bads")
raw.plot(duration=5, n_channels=23)

montage = mne.channels.read_montage(kind='chb01_01.edf', ch_names=None, path='./chb01_01.edf', unit='m', transform=False)
print(montage)

raw.plot(block=True)


