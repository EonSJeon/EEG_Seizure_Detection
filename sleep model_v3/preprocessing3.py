

import mne
import os
import numpy as np
import pandas as pd
from scipy.stats import entropy
import matplotlib.pyplot as plt

# Constants and configurations
ORDEDRED_CHANNEL_PAIRS = [
    ('Fp1', 'F7'), ('F7', 'T7'), ('T7', 'P7'), ('P7', 'O1'),
    ('Fp1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
    ('Fz', 'Cz'), ('Cz', 'Pz'),
    ('Fp2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
    ('Fp2', 'F8'), ('F8', 'T8'), ('T8', 'P8'), ('P8', 'O2')
]
VALID_LABELS = {'W': np.array([1,0,0]), '1': np.array([0,1,0]), '2': np.array([0,0,1]), '3': np.array([0,0,1])}
FREQ_BANDS = {'delta':(0.5,4), 'theta':(4,8), 'alpha':(8,12), 'beta':(12,30), 'gamma':(30,120)}
F_SAM=256

root_path = '/Users/jeonsang-eon/sleep_data/'
dst_path='/Users/jeonsang-eon/sleep_data_processed4'
label_file_path = os.path.join(root_path, 'sourcedata')

window_duration = 30  # seconds

def re_reference_eeg(raw):
    new_data = np.empty((len(ORDEDRED_CHANNEL_PAIRS), len(raw.times)))
    new_channel_names = []
    for idx, (ch1, ch2) in enumerate(ORDEDRED_CHANNEL_PAIRS):
        if ch1 in raw.ch_names and ch2 in raw.ch_names:
            ch1_data = raw.get_data(picks=ch1)
            ch2_data = raw.get_data(picks=ch2)
            ref_data = ch1_data - ch2_data
            new_data[idx, :] = ref_data
            new_channel_names.append(f'{ch1}-{ch2}')
    return mne.io.RawArray(new_data, mne.create_info(ch_names=new_channel_names, sfreq=raw.info['sfreq'], ch_types='eeg')), new_channel_names



import numpy as np
from scipy.stats import entropy

def win2features(win, sfreq):
    """Extract PSD features using FFT from each channel of an epoch.

    Parameters:
    win (mne.Epochs): The MNE Epochs object containing EEG data.
    sfreq (float): Sampling frequency of the data.
    FREQ_BANDS (dict): Frequency bands to analyze (e.g., {'delta': (1, 4), 'theta': (4, 8), ...}).

    Returns:
    numpy.ndarray: Array of extracted features for each channel.
    """
    # Extract data from the epoch, ensuring correct shape
    data = win.get_data()[0]  # Shape (n_channels, n_times)

    # Initialize list to store features for all channels
    feature_lists = []

    # Compute FFT for each channel
    for i in range(data.shape[0]):  # Assuming data.shape[0] is the number of channels
        # FFT
        fft_values = np.fft.rfft(data[i])
        fft_freqs = np.fft.rfftfreq(len(data[i]), d=1/sfreq)

        feature_list = []

        # Loop through each predefined frequency band
        for band in FREQ_BANDS:
            fmin, fmax = FREQ_BANDS[band]
            freq_indices = np.where((fft_freqs >= fmin) & (fft_freqs <= fmax))[0]
            band_fft = np.abs(fft_values[freq_indices])

            # Calculate mean spectral power
            mean_spectral_power = np.mean(band_fft**2)

            # Calculate spectral entropy
            normalized_band_fft = band_fft / np.sum(band_fft)
            spectral_entropy = entropy(normalized_band_fft, base=10)

            # Calculate mean spectral amplitude
            mean_spectrum_amplitude = np.mean(band_fft)

            # Append features to channel's feature list
            feature_list.append([mean_spectral_power, spectral_entropy, mean_spectrum_amplitude])

        # Append the feature list of the current channel to the overall feature lists
        feature_lists.append(feature_list)

    # Convert list to numpy array for easier processing later
    feature_lists = np.array(feature_lists)

    # Mean and standard deviation for normalization
    mean = np.mean(feature_lists, axis=0)
    std = np.std(feature_lists, axis=0)

    # Normalization
    normalized_feature_lists = (feature_lists - mean) / std

    return normalized_feature_lists

sub_nums = [
    2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 17, 18, 19, 20,
    21, 22, 23, 25, 26, 27, 28, 29, 30,
    31, 32, 33
]


for i in sub_nums:
    feature_data=[]
    labels=[]
    sub_name = f'sub-{i:02d}'
    sub_label_file_name = f'{sub_name}-sleep-stage.tsv'
    sub_label_file_path = os.path.join(label_file_path, sub_label_file_name)
    sub_data_dir_path = os.path.join(root_path, sub_name)
    sub_dst_path=os.path.join(dst_path,sub_name)
    
    labels_data = pd.read_csv(
        sub_label_file_path,
        delimiter='\t',
        usecols=['session', 'epoch_start_time_sec', '30-sec_epoch_sleep_stage']
    )
    valid_labels_data = labels_data[labels_data['30-sec_epoch_sleep_stage'].isin(VALID_LABELS.keys())]
    unique_sessions = valid_labels_data['session'].unique()

    if len(valid_labels_data) != 0:
        for session_name in unique_sessions:
            sub_data_file_name = f"{sub_name}_{session_name}_eeg.vhdr"
            sub_data_file_path = os.path.join(sub_data_dir_path, sub_data_file_name)

            raw = mne.io.read_raw_brainvision(sub_data_file_path, preload=True, verbose='ERROR')
            raw.notch_filter(freqs=60, notch_widths=1, fir_design='firwin')

            # Adjusted for proper EEG re-referencing
            channel_adjusted_raw, new_channel_names = re_reference_eeg(raw)
            windows = mne.make_fixed_length_epochs(channel_adjusted_raw, duration=window_duration)

            valid_labels_data_by_session = valid_labels_data[valid_labels_data['session'] == session_name]
            for _, row in valid_labels_data_by_session.iterrows():
                start_time = int(row['epoch_start_time_sec'])  # Ensure start_time is an integer
                idx = start_time // window_duration  # Use integer division

                label = VALID_LABELS[row['30-sec_epoch_sleep_stage']]
                window = windows[idx]  # Access the epoch
                print(f"Index: {idx}, Label: {label}")
            
                print(f'Epoch {idx} shape: {window.get_data().shape}') #(1, 18, 150000)
                
                feature_data.append(win2features(window, F_SAM))
                labels.append(label)

    # Check if the destination directory exists, if not, create it
    if not os.path.exists(sub_dst_path):
        os.makedirs(sub_dst_path)
        print(f"Created directory: {sub_dst_path}")

    # Save feature data and labels to .npy files
    feature_file_path = os.path.join(sub_dst_path, f"{sub_name}_features.npy")
    labels_file_path = os.path.join(sub_dst_path, f"{sub_name}_labels.npy")

    np.save(feature_file_path, np.array(feature_data))
    np.save(labels_file_path, np.array(labels))
    print(f"Saved features and labels for {sub_name} to {sub_dst_path}.")

