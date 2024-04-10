import mne
import matplotlib.pyplot as plt
import numpy as np

# Load the EDF file
file_path = './modified_data.edf'
raw = mne.io.read_raw_edf(file_path, preload=True)

# Assuming 'raw' is your loaded EEG data
data, times = raw[:]

# Select the first channel data
# Change 0 to another index to select a different channel
channel_data = data[0]  # First channel data
sampling_rate = int(raw.info['sfreq'])  # Get the sampling rate

# Generate the spectrogram
plt.figure(figsize=(10, 6))
# Using default normalization by removing norm=LogNorm()
Pxx, freqs, bins, im = plt.specgram(channel_data, NFFT=2048, Fs=sampling_rate, noverlap=1024, cmap='viridis')

# Limit the frequency to under 32 Hz for visualization
plt.ylim(0, 32)

plt.colorbar(im).set_label('Intensity (dB)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Spectrogram of the First Channel')
plt.show()
