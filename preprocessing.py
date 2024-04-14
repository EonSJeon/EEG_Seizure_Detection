import mne 
import os
from data_management import get_folder_info
import matplotlib.pyplot as plt
import numpy as np

bands = {'Delta':(0.5,4), 'Theta':(4,8), 'Alpha':(8,12), 'Beta':(12,30), 'Gamma':(30,120)}
files_list = get_folder_info()

def find_seizure_info(file_name):
    num_seizures = 0
    seizure_start_times = []
    seizure_end_times = []

    patient_folder = file_name.split('_')[0]
    txt_file_name = f'{patient_folder}-summary.txt'

    txt_file_path = os.path.join(os.getcwd(), patient_folder, txt_file_name)

    found_target_file = False
    count = 0

    with open(txt_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('File Name: ' + file_name):
                found_target_file = True
            elif line.startswith('File Name:') and found_target_file:
                break
            elif found_target_file:
                if line.startswith('Number of Seizures in File:'):
                    num_seizures = int(line.split(':')[-1].strip())
                    count = 1 if num_seizures > 0 else 0
                elif line.startswith('Seizure {} Start Time:'.format(count)):
                    seizure_start_times.append(int(line.split(':')[-1].strip().split()[0]))
                elif line.startswith('Seizure {} End Time:'.format(count)):
                    seizure_end_times.append(int(line.split(':')[-1].strip().split()[0]))
                    count += 1
    seizure_info = list(zip(seizure_start_times, seizure_end_times))
    if found_target_file == True:
        return seizure_info
    else:
        print('File does not exist')
        return None

#print(find_seizure_info('chb16_17.edf'))

def preprocess(patient_num, file_name):
    patient_folder = f'chb{patient_num:02d}'
    #file_name = 'chb{:02d}_{:02d}.edf'.format(patient_num, file_num)
    patient_file_path = files_list[patient_folder][file_name]

    # Filtering 
    raw = mne.io.read_raw_edf(patient_file_path, preload=True)
    raw.filter(0.1,127)
    raw.notch_filter(freqs=60)
    fs = raw.info['sfreq']

    # Labeling 
    seizure_start_end_times = find_seizure_info(file_name)
 