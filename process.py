from data_management import get_folder_info
from preprocessing import preprocess
import mne 
import os
import matplotlib.pyplot as plt
import numpy as np


files_list = get_folder_info()

def process(patient_num):
    patient_folder = f'chb{patient_num:02d}'
    patient_file_list = sorted(list(files_list[patient_folder].keys()))

    y = []
    
    for patient_file in patient_file_list:
        label_list = preprocess(patient_num, patient_file)
        y.extend(label_list)

    print(y)
    
process(6)