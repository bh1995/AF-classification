# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 11:07:22 2021

@author: bjorn

script for loading ECG data
"""

import os
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import heartpy as hp
from scipy import signal
import pandas as pd


def load_challenge_data(filename):
    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)
    new_file = filename.replace('.mat','.hea')
    input_header_file = os.path.join(new_file)
    with open(input_header_file,'r') as f:
        header_data=f.readlines()
    return data, header_data

def common_member(a, b):
    a_set = set(a)
    b_set = set(b)
    if len(a_set.intersection(b_set)) > 0:
        return(True) 
    return(False) 

def import_key_data(path, fs_in=500, fs_out=300, n_second=10):
    possible_af_labels = ['164889003',	'AF', '1221', '153', '2', '15', '1514', '570', '3475']
    possible_normal_labels = ['426783006', 'NSR', '918', '4', '0', '80', '18092', '1752', '20846']
    gender=[]
    age=[]
    labels=[]
    ecg_filenames=[]
    data_af_list = []
    data_normal_list = []
    data_normal = None
    data_af = None
    for subdir, dirs, files in sorted(os.walk(path)):
        for filename in files:
            filepath = subdir + os.sep + filename
            if filepath.endswith(".mat"):
                data, header_data = load_challenge_data(filepath)
                # print(data.shape)
                lab = header_data[15][5:-1]
                # print(lab.split(','))
                # add AF data
                if common_member(lab.split(','), possible_af_labels):
                    # print('AF:', lab)
                    labels.append(header_data[15][5:-1])
                    ecg_filenames.append(filepath)
                    gender.append(header_data[14][6:-1])
                    age.append(header_data[13][6:-1])
                    for i in range(data.shape[0]):
                        sig = data[i]
                        secs = len(sig)/fs_in # Number of seconds in signal record
                        samps = secs*fs_out   # Number of samples to downsample
                        sig = signal.resample(sig, num=int(samps)) # resample signal to correct fs_out
                        # plt.plot(sig[0:10*300])
                        # plt.title('Raw AF signal, nr:'+str(i))
                        # plt.show()
                        sig = hp.filter_signal(sig, cutoff=[0.5, 40], sample_rate=fs_out, order=6, filtertype='bandpass')
                        sig = hp.remove_baseline_wander(sig, sample_rate=fs_out, cutoff=0.05) # remove baseline wander
                        # record = hp.filter_signal(record, cutoff=50, sample_rate=fs, filtertype='notch') # removes powerline interference at 50 Hz
                        # scaler = preprocessing.StandardScaler().fit(record.reshape(-1, 1))
                        # record = scaler.fit_transform(record.reshape(-1, 1))
                        sig = (sig-np.min(sig))/(np.max(sig)-np.min(sig)) # get data between [0,1]
                        # plt.plot(sig[0:10*300])
                        # plt.title('Processed AF signal, nr:'+str(i))
                        # plt.show()
                        seq_list = [sig[ii:ii + int(n_second*fs_out)] for ii in range(0, len(sig), int(n_second*fs_out))]
                        for seq in seq_list:
                            if seq.shape[0] == int(n_second*fs_out):
                              # continue
                                data_af_list.append(seq) 
                # add Normal data                
                if common_member(lab.split(','), possible_normal_labels):
                    # print('Normal:', lab)
                    labels.append(header_data[15][5:-1])
                    ecg_filenames.append(filepath)
                    gender.append(header_data[14][6:-1])
                    age.append(header_data[13][6:-1])
                    for i in range(data.shape[0]):
                        sig = data[i]
                        secs = len(sig)/fs_in # Number of seconds in signal record
                        samps = secs*fs_out   # Number of samples to downsample
                        sig = signal.resample(sig, num=int(samps)) # resample signal to correct fs_out
                        # plt.plot(sig[0:10*300])
                        # plt.title('Raw Normal signal, nr:'+str(i))
                        # plt.show()
                        sig = hp.filter_signal(sig, cutoff=[0.5, 40], sample_rate=fs_out, order=6, filtertype='bandpass')
                        sig = hp.remove_baseline_wander(sig, sample_rate=fs_out, cutoff=0.05) # remove baseline wander
                        # record = hp.filter_signal(record, cutoff=50, sample_rate=fs, filtertype='notch') # removes powerline interference at 50 Hz
                        # scaler = preprocessing.StandardScaler().fit(record.reshape(-1, 1))
                        # record = scaler.fit_transform(record.reshape(-1, 1))
                        sig = (sig-np.min(sig))/(np.max(sig)-np.min(sig)) # get data between [0,1]
                        # plt.plot(sig[0:10*300])
                        # plt.title('Processed Normal signal, nr:'+str(i))
                        # plt.show()
                        seq_list = [sig[ii:ii + int(n_second*fs_out)] for ii in range(0, len(sig), int(n_second*fs_out))]
                        for seq in seq_list:
                            if seq.shape[0] == int(n_second*fs_out):
                              # continue
                                data_normal_list.append(seq) 
                                            
    try:
        data_af = np.vstack(data_af_list)
        print('A total of', data_af.shape[0], 'AF signals were extracted.')  
    except:
        print('No AF signals were extracted!')
    
    try:
        data_normal = np.vstack(data_normal_list)
        print('A total of', data_normal.shape[0], 'Normal signals were extracted.') 
    except:
        print('No Normal signals were extracted!')
    
              
    return gender, age, labels, ecg_filenames, data_af, data_normal


path = 'C:/Users/bjorn/Documents/ecg_data/WFDB(2)'
gender, age, labels, ecg_filenames, data_af, data_normal = import_key_data(path, fs_in=1000)

# save af numpy array locally
p = 'C:/Users/bjorn/OneDrive/Dokument/projects/ecg_data/Training_2_processed/af2'
# open a binary file in write mode
file = open(p, "wb")
# save array to the file
np.save(file, data_af)
# close the file
file.close
# save normal numpy array locally
p = 'C:/Users/bjorn/OneDrive/Dokument/projects/ecg_data/Training_2_processed/normal2'
# open a binary file in write mode
file = open(p, "wb")
# save array to the file
np.save(file, data_normal)
# close the file
file.close


# load data saved in .csv format
df = pd.read_csv('C:/Users/bjorn/OneDrive/Dokument/projects/ecg_data/train2017/train2017_df_af_10s.csv')
data = df.values
data.shape
# save data
# open a binary file in write mode
file = open('C:/Users/bjorn/OneDrive/Dokument/projects/ecg_data/train2017/af8', "wb")
# save array to the file
np.save(file, data)
# close the file
file.close



