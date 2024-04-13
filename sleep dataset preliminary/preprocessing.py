import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import mne
import json

F_SAM=64
SEG_DUR=30
ORDEDRED_CHANNEL_PAIRS = [
    ('Fp1', 'F7'), ('F7', 'T7'), ('T7', 'P7'), ('P7', 'O1'), 
    ('Fp1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'), 
    ('Fz', 'Cz'), ('Cz', 'Pz'), 
    ('Fp2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
    ('Fp2', 'F8'), ('F8', 'T8'), ('T8', 'P8'), ('P8', 'O2')
]
def re_reference_eeg(raw):
    new_data = []
    new_channel_names = []
    for ch1, ch2 in ORDEDRED_CHANNEL_PAIRS:
        if ch1 in raw.ch_names and ch2 in raw.ch_names:
            ch1_data, ch2_data = raw[ch1][0], raw[ch2][0]
            ref_data = ch1_data - ch2_data
            new_data.append(ref_data)
            new_channel_names.append(f'{ch1}-{ch2}')
    return np.array(new_data), new_channel_names

class TrainSpec:
    def __init__(self, channels, directory, spec_data, start_time, parent_file, duration=30, label=None):
        self.channels = channels
        self.directory = directory
        self.spec_data = spec_data
        self.start_time = start_time
        self.parent_file = parent_file
        self.duration = duration
        self.label = label

    def save(self):
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        file_name = f"spec_{self.start_time}s.npy"
        np.save(os.path.join(self.directory, file_name), self.spec_data)
        meta_data = {
            'channels': self.channels,
            'start_time': self.start_time,
            'duration': self.duration,
            'parent_file': self.parent_file,
            'label': self.label
        }
        with open(os.path.join(self.directory, f"meta_{self.start_time}s.json"), 'w') as f:
            json.dump(meta_data, f)
        print(f"Saved spectrogram and metadata for start time {self.start_time}s with label {self.label}.")

def preprocess(sub_name, session_name, data_file_path, label_file_path, output_dir_path):
    try:
        raw = mne.io.read_raw_brainvision(data_file_path, preload=True, verbose='ERROR')
        raw.resample(F_SAM)
    except Exception as e:
        print(f"Failed to load or resample the EEG data: {e}")
        return

    new_data, new_channel_names = re_reference_eeg(raw)
    samples_per_segment = F_SAM * SEG_DUR

    for i in range(new_data.shape[2] // samples_per_segment):
        start_sample = i * samples_per_segment
        end_sample = start_sample + samples_per_segment
        segment_data = new_data[:, :, start_sample:end_sample].squeeze()

        specPack = []
        for channel_data in segment_data:
            f, t, Sxx = spectrogram(channel_data, fs=F_SAM, nperseg=256, noverlap=256-15)
            specPack.append(Sxx)

        if len(specPack) > 0:
            specPack = np.array(specPack)
            start_time = start_sample // F_SAM

            labels = pd.read_csv(
                        label_file_path, 
                        delimiter='\t', 
                        usecols=['session', 'epoch_start_time_sec', '30-sec_epoch_sleep_stage']
                        )
            # Find the label for the segment based on start time
            # Correct the condition to filter the DataFrame for the label
            label_row = labels[(labels['epoch_start_time_sec'] == start_time) & (labels['session'] == session_name)]

            # Check if the label_row is empty and assign the label accordingly
            label = label_row['30-sec_epoch_sleep_stage'].iloc[0] if not label_row.empty else 'Unknown'

            # Save
            if not os.path.exists(output_dir_path):
                os.makedirs(output_dir_path)
            npy_file_name = f"{sub_name}_{session_name}_{start_time}s.npy"
            np.save(os.path.join(output_dir_path, npy_file_name), specPack)
            meta_data = {
                'channels': ORDEDRED_CHANNEL_PAIRS,
                'start_time': start_time,
                'duration': SEG_DUR,
                'parent_file': data_file_path,
                'label': label
            }

            meta_file_name=f"{sub_name}_{session_name}_{start_time}s.json"
            with open(os.path.join(output_dir_path, meta_file_name), 'w') as f:
                json.dump(meta_data, f)
            print(f"Saved spectrogram and metadata for start time {start_time}s with label {label}.")

        else:
            print(f"No data processed for segment starting at {start_sample/F_SAM} seconds.")
    


       


    







# root_dir='/Users/jeonsang-eon/sleep_data/'



# label_file_path = '/Users/jeonsang-eon/sleep_data/sourcedata/sub-01-sleep-stage.tsv'  # Path to the uploaded label file
# file_path = '/Users/jeonsang-eon/sleep_data/sub-01/sub-01_task-rest_run-1_eeg.vhdr'
# output_dir = '/Users/jeonsang-eon/sleep_data_processed/sub-01/'

# os.makedirs(output_dir, exist_ok=True)
# preprocess_eeg_to_spectrogram(file_path, output_dir, label_file_path, desired_channel_pairs)

####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import spectrogram
# import mne
# import json
# import pandas as pd

# desired_channel_pairs = [
#     ('Fp1', 'F7'), ('F7', 'T7'), ('T7', 'P7'), ('P7', 'O1'), 
#     ('Fp1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'), 
#     ('Fz', 'Cz'), ('Cz', 'Pz'), 
#     ('Fp2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
#     ('Fp2', 'F8'), ('F8', 'T8'), ('T8', 'P8'), ('P8', 'O2')
# ]





# def load_labels(label_path):
#     # Load labels assuming the file has columns ['start_time', 'stage']
#     return pd.read_csv(label_path, delimiter='\t')

# def find_label_for_segment(labels, start_time, duration):
#     # Assuming labels are provided in seconds similar to the start_time
#     label = labels[(labels['start_time'] <= start_time) & (labels['start_time'] + labels['duration'] > start_time)]
#     return label['stage'].values[0] if not label.empty else None

# class TrainSpec:
#     def __init__(self, channels, directory, spec_data, start_time, parent_file, duration=30, label=None):
#         self.channels = channels
#         self.directory = directory
#         self.spec_data = spec_data
#         self.start_time = start_time
#         self.parent_file = parent_file
#         self.duration = duration
#         self.label = label  # Add label to the class

#     def save(self):
#         if not os.path.exists(self.directory):
#             os.makedirs(self.directory)
#         file_name = f"spec_{self.start_time}s.npy"
#         np.save(os.path.join(self.directory, file_name), self.spec_data)

#         # Include label in metadata
#         meta_data = {
#             'channels': self.channels,
#             'start_time': self.start_time,
#             'duration': self.duration,
#             'parent_file': self.parent_file,
#             'label': self.label  # Save the label in the metadata
#         }
#         with open(os.path.join(self.directory, f"meta_{self.start_time}s.json"), 'w') as f:
#             json.dump(meta_data, f)
#         print(f"Saved spectrogram and metadata for start time {self.start_time}s with label {self.label}.")

# def preprocess_eeg_to_spectrogram(file_path, output_dir, label_path, desired_channel_pairs, segment_length=30, fs=64):
#     labels = load_labels(label_path)
#     try:
#         raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose='ERROR')
#         raw.resample(fs)
#     except Exception as e:
#         print(f"Failed to load or resample the EEG data: {e}")
#         return

#     new_data, new_channel_names = re_reference_eeg(raw, desired_channel_pairs)
#     samples_per_segment = fs * segment_length

#     for i in range(new_data.shape[2] // samples_per_segment):
#         start_sample = i * samples_per_segment
#         end_sample = start_sample + samples_per_segment
#         segment_data = new_data[:, :, start_sample:end_sample].squeeze()

#         specPack = []
#         for channel_data in segment_data:
#             f, t, Sxx = spectrogram(channel_data, fs=fs, nperseg=256, noverlap=256-15)
#             specPack.append(Sxx)

#         if len(specPack) > 0:
#             specPack = np.array(specPack)
#             start_time = start_sample // fs
#             label = find_label_for_segment(labels, start_time, segment_length)
#             temp_spec = TrainSpec(new_channel_names, output_dir, specPack, start_time, file_path, label=label)
#             temp_spec.save()
#         else:
#             print(f"No data processed for segment starting at {start_sample/fs} seconds.")

# label_file_path = '/Users/jeonsang-eon/sleep_data/sourcedata/sub-01-sleep-stage.tsv'  # Path to the uploaded label file
# file_path = '/Users/jeonsang-eon/sleep_data/sub-01/sub-01_task-rest_run-1_eeg.vhdr'
# output_dir = '/Users/jeonsang-eon/sleep_data_processed/sub-01/'

# # Execute the preprocessing function with labels
# preprocess_eeg_to_spectrogram(file_path, output_dir, label_file_path, desired_channel_pairs)


########################################################################################################
########################################################################################################
########################################################################################################


# class TrainSpec:
#     def __init__(self, channels, directory, spec_data, start_time, parent_file, duration=30):
#         self.channels = channels
#         self.directory = directory
#         self.spec_data = spec_data
#         self.start_time = start_time
#         self.parent_file = parent_file
#         self.duration = duration

#     def save(self):
#         if not os.path.exists(self.directory):
#             os.makedirs(self.directory)
#         file_name = f"spec_{self.start_time}s.npy"
#         np.save(os.path.join(self.directory, file_name), self.spec_data)
#         meta_data = {
#             'channels': self.channels,
#             'start_time': self.start_time,
#             'duration': self.duration,
#             'parent_file': self.parent_file
#         }
#         with open(os.path.join(self.directory, f"meta_{self.start_time}s.json"), 'w') as f:
#             json.dump(meta_data, f)
#         print(f"Saved spectrogram and metadata for start time {self.start_time}s.")

# def preprocess_eeg_to_spectrogram(file_path, output_dir, desired_channel_pairs, segment_length=30, fs=64):
#     try:
#         raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose='ERROR')
#         raw.resample(fs)
#     except Exception as e:
#         print(f"Failed to load or resample the EEG data: {e}")
#         return

#     new_data, new_channel_names = re_reference_eeg(raw, desired_channel_pairs)
#     samples_per_segment = fs * segment_length
#     n_segments = new_data.shape[2] // samples_per_segment

#     for i in range(n_segments):
#         start_sample = i * samples_per_segment
#         end_sample = start_sample + samples_per_segment
#         segment_data = new_data[:, :, start_sample:end_sample].squeeze()

#         specPack = []
#         for channel_data in segment_data:
#             f, t, Sxx = spectrogram(channel_data, fs=fs, nperseg=256, noverlap=256-15)
#             specPack.append(Sxx)

#         if len(specPack) > 0:
#             specPack = np.array(specPack)
#             start_time = start_sample // fs
#             temp_spec = TrainSpec(new_channel_names, output_dir, specPack, start_time, file_path)
#             temp_spec.save()
#         else:
#             print(f"No data processed for segment starting at {start_sample/fs} seconds.")



# # Setup directories and file paths
# file_path = '/Users/jeonsang-eon/sleep_data/sub-01/sub-01_task-rest_run-1_eeg.vhdr'
# output_dir = '/Users/jeonsang-eon/sleep_data_processed/sub-01/'

# os.makedirs(output_dir, exist_ok=True)
# preprocess_eeg_to_spectrogram(file_path, output_dir, desired_channel_pairs)


# def preprocess_eeg_to_spectrogram(file_path, output_dir, desired_channel_pairs, segment_length=30, fs=64, dpi=100):
#     raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose='ERROR')
#     raw.resample(fs)
#     new_data, new_channel_names = re_reference_eeg(raw, desired_channel_pairs)
#     # new_data.shape = (18, 1, duration*fs)

#     samples_per_segment = fs * segment_length 
#     # samples_per_segment = 30 sec* 64 Hz = 1920
#     n_segments = new_data.shape[2] // samples_per_segment

#     for i in range(n_segments):
#         start_sample = i * samples_per_segment
#         end_sample = start_sample + samples_per_segment
#         segment_data = new_data[:, :, start_sample:end_sample]

#         specPack=np.array()
#         for j, channel_data in enumerate(segment_data):
#             channel_data = channel_data.squeeze()
#             f, t, Sxx = spectrogram(channel_data, fs=fs, nperseg=256, noverlap=256-15)
#             # Sxx.shape= (129 ,111)
#             specPack[j]=Sxx
        
#         tempSpec= TrainSpec()
#         tempSpec.dir=output_dir
#         tempSpec.specPack=specPack
#         tempSpec.channels=new_channel_names
#         tempSpec.startTime=start_sample//fs
#         tempSpec.parentFile=file_path

#         tempSpec.save(f"cropped_spectrogram_{start_sample//fs}s_channel{j}_rev.png")
            
            
            


# file_path = '/Users/jeonsang-eon/sleep_data/sub-01/sub-01_task-rest_run-1_eeg.vhdr'
# output_dir = '/Users/jeonsang-eon/sleep_data/sub-01/'

# os.makedirs(output_dir, exist_ok=True)
# preprocess_eeg_to_spectrogram(file_path, output_dir, desired_channel_pairs)


    # plt.figure(figsize=(1, 0.8), dpi=dpi)
    # plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='gray')
    # plt.ylim(0, 32)
    # plt.show()
    # plt.axis('off')

    # temp_img_path = os.path.join(output_dir, f"temp_spectrogram_{i}_{j}.png")
    # plt.savefig(temp_img_path, bbox_inches='tight', pad_inches=0)
    # plt.close()

    # img = plt.imread(temp_img_path)
    # cropped_img = img[14:90, 11:71]

    # cropped_img_path = os.path.join(output_dir, f"cropped_spectrogram_{start_sample//fs}s_channel{j}_rev.png")
    # plt.imsave(cropped_img_path, img)

    # os.remove(temp_img_path)
    # print(f"Processed and saved: {cropped_img_path}")