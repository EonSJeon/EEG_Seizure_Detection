import matplotlib.pyplot as plt
import numpy as np
from mne_connectivity import envelope_correlation

import mne
from mne.preprocessing import compute_proj_ecg

# sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = './chb01_01.edf'
raw = mne.io.read_raw_edf(sample_data_raw_file)
print(raw)
print(raw.info)
print(raw.info["bads"])
epochs = mne.make_fixed_length_epochs(raw, duration=30, preload=True)

# Crop, resample, and pick channels
raw.crop(tmax=150)
raw.resample(100)
raw.pick_channels(["C3-P3"])

# Compute ECG projector
# ecg_proj, _ = compute_proj_ecg(raw, ch_name="C3-P3")  # ECG filter
# if ecg_proj:
#     raw.add_proj(ecg_proj)
#     raw.apply_proj()

# Drop bad epochs
epochs = mne.Epochs(raw, epochs.events, tmin=0, tmax=30, baseline=None, reject=None)

event_related_plot = epochs.plot_image(picks=["C3-P3"])