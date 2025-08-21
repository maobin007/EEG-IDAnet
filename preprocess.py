import os 
import sys
import scipy.signal as signal
current_path = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(os.path.split(current_path)[0])[0]
sys.path.append(current_path)
sys.path.append(rootPath)

import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from LoadData import Load_BCIC_2a, Load_BCIC_2b
from LoadData import load_data_LOSO
from scipy.stats import zscore
import mne
import gc



def standardize_data(X_train, X_test, axis=1):
    X_train = (X_train - np.mean(X_train, axis=axis)) / np.std(X_train, axis=axis)
    X_test = (X_test - np.mean(X_test, axis=axis)) / np.std(X_test, axis=axis)
    return X_train, X_test


def bandpass_filter(data, bandFiltCutF, fs, filtOrder=50, axis=1, filtType='filter'):
    a = [1]

    if (bandFiltCutF[0] == 0 or bandFiltCutF[0] is None) and (bandFiltCutF[1] is None or bandFiltCutF[1] == fs / 2.0):
        # no filter
        print("Not doing any filtering. Invalid cut-off specifications")
        return data
    elif bandFiltCutF[0] == 0 or bandFiltCutF[0] is None:
        # low-pass filter
        print("Using lowpass filter since low cut hz is 0 or None")
        h = signal.firwin(numtaps=filtOrder + 1, cutoff=bandFiltCutF[1], pass_zero="lowpass", fs=fs)
    elif (bandFiltCutF[1] is None) or (bandFiltCutF[1] == fs / 2.0):
        # high-pass filter
        print("Using highpass filter since high cut hz is None or nyquist freq")
        h = signal.firwin(numtaps=filtOrder + 1, cutoff=bandFiltCutF[0], pass_zero="highpass", fs=fs)
    else:
        h = signal.firwin(numtaps=filtOrder + 1, cutoff=bandFiltCutF, pass_zero="bandpass", fs=fs)

    if filtType == 'filtfilt':
        dataOut = signal.filtfilt(h, a, data, axis=axis)
    else:
        dataOut = signal.lfilter(h, a, data, axis=axis)

    return dataOut



#%%
def get_data(path, tmin=0., tmax=4., low_freq=4, high_freq=38, subject=None, LOSO=False, isStandard=True, data_type='BCICIV_2a'):
    # Load and split the dataset into training and testing 
    if LOSO:
        # Loading and Dividing of the data set based on the 
        # 'Leave One Subject Out' (LOSO) evaluation approach. 
        X_train, y_train, X_test, y_test, sfreq, channels = load_data_LOSO(path, subject, data_type=data_type, tmin=tmin, tmax=tmax, low_freq=low_freq, high_freq=high_freq)
    else:
        # Loading and Dividing of the data set based on the subject-specific (subject-dependent) approach.
        # In this approach, we used the same training and testing data as the original competition, 
        # i.e., trials in session 1 for training, and trials in session 2 for testing.  
        # path = path + 's{:}/'.format(subject)
        if data_type == 'BCICIV_2a':
            load_raw_data = Load_BCIC_2a(path, subject)
            eeg_data = load_raw_data.get_epochs_train(tmin=tmin, tmax=tmax, low_freq=low_freq, high_freq=high_freq, baseline=None)
            X_train, y_train = eeg_data['x_data'][:, :, :], eeg_data['y_labels']
            eeg_data = load_raw_data.get_epochs_test(tmin=tmin, tmax=tmax, low_freq=low_freq, high_freq=high_freq, baseline=None)
            X_test, y_test = eeg_data['x_data'][:, :, :], eeg_data['y_labels']
            sfreq = eeg_data['fs']
            channels = eeg_data['channels']
        elif data_type == 'BCICIV_2b':
            load_raw_data = Load_BCIC_2b(path, subject)
            eeg_data = load_raw_data.get_epochs_train(tmin=tmin, tmax=tmax, low_freq=low_freq, high_freq=high_freq, baseline=None)
            X_train, y_train = eeg_data['x_data'][:, :, :], eeg_data['y_labels']
            eeg_data = load_raw_data.get_epochs_test(tmin=tmin, tmax=tmax, low_freq=low_freq, high_freq=high_freq, baseline=None)
            X_test, y_test = eeg_data['x_data'][:, :, :], eeg_data['y_labels']
            sfreq = eeg_data['fs']
            channels = eeg_data['channels']
    n_tr, n_ch, T = X_train.shape
    X_train = X_train.reshape(n_tr, 1, n_ch, T)
    n_tr, n_ch, T = X_test.shape
    X_test = X_test.reshape(n_tr, 1, n_ch, T)
    # Standardize the data
    if (isStandard == True):
        X_train, X_test = standardize_data(X_train, X_test, axis=(0,1))

    return X_train, y_train, X_test, y_test



#%%
def BCIC_DataLoader(x_train, y_train, batch_size=64, num_workers=1, shuffle=True):
    # 将数据转换为TensorDataset类型
    x_train = torch.tensor(x_train)
    y_train = torch.tensor(y_train)
    dataset  = TensorDataset(x_train, y_train)
    # 分割数据，生成batch
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    # 函数返回值
    return dataloader
