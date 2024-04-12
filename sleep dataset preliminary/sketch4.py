import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import mne

def re_reference_eeg(raw, desired_channel_pairs):
    new_data = []
    new_channel_names = []
    for ch1, ch2 in desired_channel_pairs:
        if ch1 in raw.ch_names and ch2 in raw.ch_names:
            ch1_data, ch2_data = raw[ch1][0], raw[ch2][0]
            ref_data = ch1_data - ch2_data
            new_data.append(ref_data)
            new_channel_names.append(f'{ch1}-{ch2}')
    return np.array(new_data), new_channel_names

def preprocess_eeg_to_spectrogram(file_path, output_dir, desired_channel_pairs, segment_length=30, fs=64, dpi=100):
    raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose='ERROR')
    raw.resample(fs)
    new_data, new_channel_names = re_reference_eeg(raw, desired_channel_pairs)

    samples_per_segment = fs * segment_length
    n_segments = new_data.shape[2] // samples_per_segment

    for i in range(n_segments):
        start_sample = i * samples_per_segment
        end_sample = start_sample + samples_per_segment
        segment_data = new_data[:, :, start_sample:end_sample]

        for j, channel_data in enumerate(segment_data):
            channel_data = channel_data.squeeze()  # Ensure data is 1D
            print(f'channel_data_shape:{channel_data.shape}')
            f, t, Sxx = spectrogram(channel_data, fs=fs, nperseg=256, noverlap=256-12)
            
            print(Sxx.shape)
            
            plt.figure(figsize=(1, 0.8), dpi=dpi)
            plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='gray')
            plt.ylim(0, 32)
            plt.show()
            # plt.axis('off')

            temp_img_path = os.path.join(output_dir, f"temp_spectrogram_{i}_{j}.png")
            plt.savefig(temp_img_path, bbox_inches='tight', pad_inches=0)
            plt.close()

            img = plt.imread(temp_img_path)
            cropped_img = img[14:90, 11:71]

            cropped_img_path = os.path.join(output_dir, f"cropped_spectrogram_{start_sample//fs}s_channel{j}_rev.png")
            plt.imsave(cropped_img_path, img)

            os.remove(temp_img_path)
            print(f"Processed and saved: {cropped_img_path}")

desired_channel_pairs = [
    ('Fp1', 'F7'), ('F7', 'T7'), ('T7', 'P7'), ('P7', 'O1'), 
    ('Fp1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'), 
    ('Fz', 'Cz'), ('Cz', 'Pz'), 
    ('Fp2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
    ('Fp2', 'F8'), ('F8', 'T8'), ('T8', 'P8'), ('P8', 'O2')
]

file_path = '/Users/jeonsang-eon/sleep_data/sub-01/sub-01_task-rest_run-1_eeg.vhdr'
output_dir = '/Users/jeonsang-eon/sleep_data/sub-01/'

os.makedirs(output_dir, exist_ok=True)
preprocess_eeg_to_spectrogram(file_path, output_dir, desired_channel_pairs)
