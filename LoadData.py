import mne
import numpy as np
import scipy.io as scio



#%%
class Load_BCIC_2a():
    '''
    Subclass of LoadData for loading BCI Competition IV Dataset 2a.

    Methods:
        get_epochs(self, tmin=-0., tmax=2, baseline=None, downsampled=None)
    '''
    def __init__(self, data_path, persion):
        self.stimcodes_train=('769','770','771','772')
        self.stimcodes_test=('783')
        self.data_path = data_path
        self.persion = persion
        self.channels_to_remove = ['EOG-left', 'EOG-central', 'EOG-right']
        super(Load_BCIC_2a,self).__init__()

    def get_epochs_train(self, tmin=-0., tmax=2, low_freq=None, high_freq=None, baseline=None, downsampled=None):
        file_to_load = 'data_gdf/' + 'A0{}T.gdf'.format(self.persion)
        raw_data = mne.io.read_raw_gdf(self.data_path + file_to_load, preload=True)
        if low_freq and high_freq:
            raw_data.filter(l_freq=low_freq, h_freq=high_freq, method='iir', iir_params=dict(order=8, ftype='butter'), fir_window='hamming', verbose=True)
        if downsampled is not None:
            raw_data.resample(sfreq=downsampled)
        self.fs = raw_data.info.get('sfreq')
        events, event_ids = mne.events_from_annotations(raw_data)
        stims =[value for key, value in event_ids.items() if key in self.stimcodes_train]
        epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=tmin, tmax=tmax, event_repeated='drop',
                            baseline=baseline, preload=True, proj=False, reject_by_annotation=False)
        epochs = epochs.drop_channels(self.channels_to_remove)
        self.channels_name = epochs.info['ch_names']
        self.y_labels = epochs.events[:, -1] - min(epochs.events[:, -1])
        self.x_data = epochs.get_data()*1e6
        eeg_data={'x_data':self.x_data[:, :, :-1],
                  'y_labels':self.y_labels,
                  'fs':self.fs,
                  'channels':self.channels_name}
        return eeg_data

    def get_epochs_test(self, tmin=-0., tmax=2, low_freq=None, high_freq=None, baseline=None, downsampled=None):
        file_to_load = 'data_gdf/' + 'A0{}E.gdf'.format(self.persion)
        # file_to_load = 's{:}/'.format(self.persion) + 'A0' + self.persion + 'E.gdf'
        raw_data = mne.io.read_raw_gdf(self.data_path + file_to_load, preload=True)
        data_path_label = self.data_path + "label_mat/A0{}E.mat" .format(self.persion)
        mat_label = scio.loadmat(data_path_label)
        mat_label = mat_label['classlabel'][:,0]-1
        if (low_freq is not None) and (high_freq is not None):
            raw_data.filter(l_freq=low_freq, h_freq=high_freq, method='iir', iir_params=dict(order=8, ftype='butter'), fir_window='hamming', verbose=True)
            # raw_data.filter(l_freq=low_freq, h_freq=high_freq, method='fir', fir_window='blackman')
        if downsampled is not None:
            raw_data.resample(sfreq=downsampled)
        self.fs = raw_data.info.get('sfreq')
        events, event_ids = mne.events_from_annotations(raw_data)
        stims =[value for key, value in event_ids.items() if key in self.stimcodes_test]
        epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=tmin, tmax=tmax, event_repeated='drop',
                            baseline=baseline, preload=True, proj=False, reject_by_annotation=False)
        epochs = epochs.drop_channels(self.channels_to_remove)
        self.channels_name = epochs.info['ch_names']
        self.y_labels = epochs.events[:, -1] - min(epochs.events[:, -1]) + mat_label
        self.x_data = epochs.get_data()*1e6
        eeg_data={'x_data':self.x_data[:, :, :-1],
                  'y_labels':self.y_labels,
                  'fs':self.fs,
                  'channels': self.channels_name
                  }
        return eeg_data


#%%
class Load_BCIC_2b():
    '''
    Subclass of LoadData for loading BCI Competition IV Dataset 2b.

    Methods:
        get_epochs(self, tmin=-0., tmax=2, baseline=None, downsampled=None)
    '''
    def __init__(self, data_path, persion):
        self.stimcodes_train=('769','770')
        self.stimcodes_test=('783')
        self.data_path = data_path
        self.persion = persion
        self.channels_to_remove = ['EOG:ch01', 'EOG:ch02', 'EOG:ch03']
        self.train_name = ['1','2','3']
        self.test_name = ['4','5']
        super(Load_BCIC_2b,self).__init__()

    def get_epochs_train(self, tmin=-0., tmax=2, low_freq=None, high_freq=None, baseline=None, downsampled=None):
        x_data = []
        y_labels = []
        for session in self.train_name:
            file_to_load = 'data_gdf/' + 'B0{}0{}T.gdf'.format(self.persion, session)
            raw_data = mne.io.read_raw_gdf(self.data_path + file_to_load, preload=True)
            data_path_label = self.data_path + 'label_mat/' + 'B0{}0{}T.mat'.format(self.persion, session)
            mat_label = scio.loadmat(data_path_label)
            mat_label = mat_label['classlabel'][:,0]-1
            if low_freq and high_freq:
                raw_data.filter(l_freq=low_freq, h_freq=high_freq, method='iir',
                                iir_params=dict(order=4, ftype='butter'), fir_window='hamming', verbose=True)
            if downsampled is not None:
                raw_data.resample(sfreq=downsampled)
            self.fs = raw_data.info.get('sfreq')
            events, event_ids = mne.events_from_annotations(raw_data)
            stims =[value for key, value in event_ids.items() if key in self.stimcodes_train]
            epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=tmin, tmax=tmax, event_repeated='drop',
                                baseline=baseline, preload=True, proj=False, reject_by_annotation=False)
            epochs = epochs.drop_channels(self.channels_to_remove)
            self.channels_name = epochs.info['ch_names']
            self.y_labels = epochs.events[:, -1] - min(epochs.events[:, -1])
            self.x_data = epochs.get_data()*1e6
            x_data.extend(self.x_data[:, :, :-1])
            y_labels.extend(self.y_labels)

        x_data = np.array(x_data)
        y_labels = np.array(y_labels)
        eeg_data={'x_data':x_data,
                  'y_labels':y_labels,
                  'fs':self.fs,
                  'channels': self.channels_name}
        return eeg_data

    def get_epochs_test(self, tmin=-0., tmax=2, low_freq=None, high_freq=None, baseline=None, downsampled=None):
        x_data = []
        y_labels = []
        for session in self.test_name:
            file_to_load = 'data_gdf/' + 'B0{}0{}E.gdf'.format(self.persion, session)
            raw_data = mne.io.read_raw_gdf(self.data_path + file_to_load, preload=True)
            data_path_label = self.data_path + 'label_mat/' + 'B0{}0{}E.mat'.format(self.persion, session)
            mat_label = scio.loadmat(data_path_label)
            mat_label = mat_label['classlabel'][:,0]-1
            if (low_freq is not None) and (high_freq is not None):
                raw_data.filter(l_freq=low_freq, h_freq=high_freq, method='iir',
                                iir_params=dict(order=4, ftype='butter'), fir_window='hamming', verbose=True)
            if downsampled is not None:
                raw_data.resample(sfreq=downsampled)
            self.fs = raw_data.info.get('sfreq')
            events, event_ids = mne.events_from_annotations(raw_data)
            stims =[value for key, value in event_ids.items() if key in self.stimcodes_test]
            epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=tmin, tmax=tmax, event_repeated='drop',
                                baseline=baseline, preload=True, proj=False, reject_by_annotation=False)
            epochs = epochs.drop_channels(self.channels_to_remove)
            self.channels_name = epochs.info['ch_names']
            self.y_labels = epochs.events[:, -1] - min(epochs.events[:, -1]) + mat_label
            self.x_data = epochs.get_data()*1e6
            x_data.extend(self.x_data[:, :, :-1])
            y_labels.extend(self.y_labels)
        
        x_data = np.array(x_data)
        y_labels = np.array(y_labels)
        eeg_data={'x_data':x_data,
                  'y_labels':y_labels,
                  'fs':self.fs,
                  'channels': self.channels_name
                  }
        return eeg_data


# %%
def load_data_LOSO(data_path, subject, data_type='BCICIV_2a', tmin=0, tmax=4., low_freq=4, high_freq=38):
    """ Loading and Dividing of the data set based on the 'Leave One Subject Out' (LOSO) evaluation approach.
    LOSO is used for  Subject-independent evaluation.

        Parameters
        ----------
        data_path: string
            dataset path
            # Dataset BCI Competition IV-2a is available on
            # http://bnci-horizon-2020.eu/database/data-sets
        subject: int
            number of subject in [1, .. ,9]
            Here, the subject data is used  test the model and other subjects data
            for training
    """

    X_train, y_train = [], []
    for sub in range(1, 10):
        # path = data_path + 's' + str(sub) + '/'
        path = data_path
        if data_type == 'BCICIV_2a':
            load_raw_data = Load_BCIC_2a(path, sub)
            eeg_data1 = load_raw_data.get_epochs_train(tmin=tmin, tmax=tmax, low_freq=low_freq, high_freq=high_freq,
                                                       baseline=None)
            x1 = eeg_data1['x_data'][:, :, :]
            y1 = eeg_data1['y_labels']
            eeg_data2 = load_raw_data.get_epochs_test(tmin=tmin, tmax=tmax, low_freq=low_freq, high_freq=high_freq,
                                                      baseline=None)
            x2 = eeg_data2['x_data'][:, :, :]
            y2 = eeg_data2['y_labels']
            X = np.concatenate((x1, x2), axis=0)
            y = np.concatenate((y1, y2), axis=0)
            sfreq = eeg_data1['fs']
            channels = eeg_data1['channels']
        elif data_type == 'BCICIV_2b':
            load_raw_data = Load_BCIC_2b(path, sub)
            eeg_data1 = load_raw_data.get_epochs_train(tmin=tmin, tmax=tmax, low_freq=low_freq, high_freq=high_freq,
                                                       baseline=None)
            x1 = eeg_data1['x_data'][:, :, 50:]
            y1 = eeg_data1['y_labels']
            eeg_data2 = load_raw_data.get_epochs_test(tmin=tmin, tmax=tmax, low_freq=low_freq, high_freq=high_freq,
                                                      baseline=None)
            x2 = eeg_data2['x_data'][:, :, 50:]
            y2 = eeg_data2['y_labels']
            X = np.concatenate((x1, x2), axis=0)
            y = np.concatenate((y1, y2), axis=0)
            sfreq = eeg_data1['fs']
            channels = eeg_data1['channels']
        if (sub == subject):
            X_test = X
            y_test = y
        elif (X_train == []):
            X_train = X
            y_train = y
        else:
            X_train = np.concatenate((X_train, X), axis=0)
            y_train = np.concatenate((y_train, y), axis=0)

    return X_train, y_train, X_test, y_test, sfreq, channels

