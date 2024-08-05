from pathlib import Path
import numpy as np
import six.moves.urllib as urllib
import zipfile
from scipy import stats
from scipy import signal
import pandas as pd
import os

import torch
from torch.utils.data import Dataset


class PAMAP2(Dataset):
    r"""(protocol) https://archive.ics.uci.edu/ml/machine-learning-databases/00231/readme.pdf"""

    if 'SLURM_JOB_ID' in os.environ:
        dir = '/netscratch/geissler/BeyondConfusion/datasets/' + 'pamap2/' + 'dataset'
        dir = Path(dir)
    else:
        dir = Path(__file__).parent.joinpath('dataset')
    dir.mkdir(parents=True, exist_ok=True)
    print(dir)

    # Data url files
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip"
    preprocessed_url = "https://zenodo.org/record/834467/files/data03.zip"

    ACTIONS_IDX = {
        0: 'no_activity',
        1: 'lying',
        2: 'sitting',
        3: 'standing',
        4: 'walking',
        5: 'running',
        6: 'cycling',
        7: 'nordic_walking',
        # 9: 'watching_tv',
        # 10: 'computer_work',
        # 11: 'car_driving',
        12: 'ascending_stairs',
        13: 'descending_stairs',
        16: 'vaccuum_cleaning',
        17: 'ironing',
        # 18: 'folding_laundry',
        # 19: 'house_cleaning',
        # 20: 'playing_soccer',
        24: 'rope_jumping'
    }
    ACTIVITIES_MAP = dict(map(reversed, ACTIONS_IDX.items()))    # [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]

    ORIGINAL_FREQUENCY = 100    #    100Hz
    USERS = [1, 2, 3, 4, 5, 6, 7, 8]    # exclude user 9
    NB_CLASS = 12
    #####################################
    # Subject ID ,Sex, Age (years), Height (cm), Weight (kg), Resting HR (bpm), Max HR (bpm), Dominant hand
    # Note: Age, Height, Weight, Resting HR, Max HR

    def __init__(self, window_size=200, window_step=50, transform=None, users=[1, 3, 6], columns=None, train_users=None, frequency=None):
        """ PAMAP2 Dataset object. Data collected at 100Hz. Sensor positions: wrist on dominant arm, chest, ankle on dominant side"""
        headers = PAMAP2.headers()
        dataset_splits = {
            'train': [1, 2, 4, 5, 6, ],
            'val': [8, ],
            'test': [3, 7],
            'unseen_test': [3, 7],
            'full': [1, 2, 3, 4, 5, 6, 7, 8, ]}
        self.users = dataset_splits[users] if isinstance(users, str) else users
        train_users = dataset_splits[train_users] if isinstance(train_users, str) else train_users
        self.window_size = window_size
        self.window_step = window_step
        self.transform = transform
        self.frequency = frequency or PAMAP2.ORIGINAL_FREQUENCY
        frequency_factor = PAMAP2.ORIGINAL_FREQUENCY / self.frequency
        except_class = None

        save_at_original_file = self.download_unzip(url=PAMAP2.url)

        self.x_data = list()
        self.y_data = list()
        self.user_id_list = []
        # 'ankle_acc_16g_x', 'ankle_acc_16g_y', 'ankle_acc_16g_z'  # ankle acceleration; s. all: print(headers)
        columns = columns or ['hand_acc_16g_x', 'hand_acc_16g_y', 'hand_acc_16g_z', 
                              'hand_gyroscope_x', 'hand_gyroscope_y', 'hand_gyroscope_z', 
                              'hand_magnometer_x', 'hand_magnometer_y', 'hand_magnometer_z', 
                              #'hand_orientation_0', 'hand_orientation_1', 'hand_orientation_2', 'hand_orientation_3',
                              'chest_acc_16g_x', 'chest_acc_16g_y', 'chest_acc_16g_z', 
                              'chest_gyroscope_x', 'chest_gyroscope_y', 'chest_gyroscope_z', 
                              'chest_magnometer_x', 'chest_magnometer_y', 'chest_magnometer_z', 
                              #'chest_orientation_0', 'chest_orientation_1', 'chest_orientation_2', 'chest_orientation_3',
                              'ankle_acc_16g_x', 'ankle_acc_16g_y', 'ankle_acc_16g_z',  
                              'ankle_gyroscope_x', 'ankle_gyroscope_y', 'ankle_gyroscope_z', 
                              'ankle_magnometer_x', 'ankle_magnometer_y', 'ankle_magnometer_z', 
                              #'ankle_orientation_0', 'ankle_orientation_1', 'ankle_orientation_2', 'ankle_orientation_3'
                              ]   # hand acceleration
        
        columns_idx = [headers.index(c) for c in columns]
        #print(headers)
        # print(columns_idx, headers.index('activityID'))
        protocol = 'Protocol/'  # '' for both Optional + Protocol directories

        #####################################
        # mean & std
        self.mean_std = self.compute_mean_std(train_users, save_at_original_file, protocol, columns_idx, frequency_factor)
        #####################################

        for user_id in self.users:
            for trial_user_file in sorted(save_at_original_file.rglob(f"*{protocol}subject10{user_id}.dat")):
                trial_user_file_np = trial_user_file.with_name(f"{trial_user_file.stem}_c{len(columns_idx)}_resampled.npy")
                trial_user_file_label_np = trial_user_file.with_name(f"{trial_user_file.stem}_c{len(columns_idx)}_label.npy")
                # (Re)Load & Process data
                if trial_user_file_np.exists() and trial_user_file_label_np.exists():
                    values = np.load(trial_user_file_np)
                    labels = np.load(trial_user_file_label_np)
                else:
                    # A. Label B. Input: 1.Read file 2. Extract specific data 3. Remove missing data 4. Resample 5. Segmentation
                    user_trial_dataset = np.genfromtxt(trial_user_file, delimiter=None, skip_header=False)
                    # Labels
                    labels = user_trial_dataset[..., headers.index('activityID')]
                    labels = self.resamples(labels, factor=frequency_factor, neighbours=True)
                    np.save(trial_user_file_label_np, labels)
                    # Inputs
                    values = user_trial_dataset[..., columns_idx]
                    values = self.extra_interpolates_nan(values)                     # fix nan issues in case
                    values = self.resamples(values, factor=frequency_factor)         # 100Hz(original) to target frequency e.g. 50Hz
                    np.save(trial_user_file_np, values)

                values = (values - self.mean_std['mean']) / self.mean_std['std']
                values = self.sliding_window_np(values)

                labels = self.sliding_window_np(labels, flatten='majority')
                self.x_data.extend(values)
                self.y_data.extend(labels)
                self.user_id_list.extend([user_id] * len(values))

        self.x_data = [torch.from_numpy(window_x).float() for window_x in self.x_data]
        self.y_data = [int(window_lbl) for window_lbl in self.y_data]

        if except_class is not None:
            except_class = except_class if isinstance(except_class, (list, tuple)) else [except_class]
            used_classes = [k for k in self.ACTIONS_IDX.keys() if k not in except_class]
            used_data_idx = []
            # print(f"Used classes ({len(used_classes)}): {used_classes}")
            for ci_ in used_classes:
                class_idx = np.arange(len(self.y_data))[np.array(self.y_data) == ci_]
                used_data_idx.extend(class_idx)
            self.y_data = [self.y_data[idx] for idx in used_data_idx]
            self.x_data = [self.x_data[idx] for idx in used_data_idx]
            self.used_classes = used_classes

        # Map classes to [0..last_class]
        if except_class is not None:
            [self.ACTIONS_IDX.pop(c, 'not in') for c in except_class]
            self.ACTIVITIES_MAPV = dict(map(reversed, self.ACTIONS_IDX.items()))
        self.remap = dict(map(reversed, enumerate(self.ACTIONS_IDX.keys())))  # {n..m}->{0..N-1}
        self.y_data = [self.remap.get(y) for y in self.y_data]

        assert len(self.x_data) == len(self.y_data), f"Sizes missmatch: inputs {len(self.x_data)} >< labels {len(self.y_data)}!"

        # Info
        print('Classes - Counts', [o.tolist() for o in np.unique(self.y_data, return_counts=True)])
        self.labels = np.unique(self.y_data).tolist()
        self.features = len(columns)

    def compute_mean_std(self, lst_usr_train, save_at_original_file, protocol, columns_idx, frequency_factor, reload=True):
        assert lst_usr_train is not None, "The train_users is None!"
        path_mean_std_saved = self.dir.joinpath(f"mean_std_pamap2_users_{''.join([f'{i}' for i in sorted(lst_usr_train)])}.npz")
        if path_mean_std_saved.exists() and reload:
            return np.load(path_mean_std_saved, allow_pickle=True)

        lst_x = []
        for user_id in lst_usr_train:
            for trial_user_file in sorted(save_at_original_file.rglob(f"*{protocol}subject10{user_id}.dat")):
                user_trial_dataset = np.genfromtxt(trial_user_file, delimiter=None, skip_header=False)
                values = user_trial_dataset[..., columns_idx]
                values = self.extra_interpolates_nan(values)
                values = self.resamples(values, factor=frequency_factor)
                lst_x.extend(values)
        arr_train_x = np.asarray(lst_x)
        print(f"Users: {sorted(lst_usr_train)} -- Data: {arr_train_x.shape}", arr_train_x)

        mean = np.mean(arr_train_x, axis=0)
        std = np.std(arr_train_x, axis=0)
        np.savez(path_mean_std_saved, mean=mean, std=std)

        return {'mean': mean, 'std': std}

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, index):
        x = self.x_data[index]
        class_idx = self.y_data[index]
        if self.transform:
            x = self.transform(x)

        return x, class_idx

    def download_unzip(self, url):
        data_path = self.dir.joinpath('data')

        if not data_path.exists():
            path_to_zip_file = self.dir.joinpath('data.zip')

            # Download zip file with data
            if not path_to_zip_file.exists():
                print("Downloading data...")
                local_fn, headers = urllib.request.urlretrieve(url=url, filename=path_to_zip_file)
                # print(local_fn, headers)

            # Extract the zip file
            if not data_path.parent.joinpath('PAMAP2_Dataset').exists():
                with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
                    print("Extracting data...")
                    zip_ref.extractall(data_path.parent)

        return data_path.parent.joinpath('PAMAP2_Dataset')

    @staticmethod
    def extra_interpolates_nan(x):
        df = pd.DataFrame(x)
        df = df.interpolate(limit_direction='both')
        return df.to_numpy()

    def sliding_window_np(self, x, stride=1, offset=0, flatten=None):
        window_size = self.window_size
        window_step = self.window_step
        # def sliding_window(x, window_size, window_step, stride=1, offset=0, flatten=None):
        overall_window_size = (window_size - 1) * stride + 1
        num_windows = (x.shape[0] - offset - overall_window_size) // window_step + 1
        windows = []
        for i in range(num_windows):
            start_index = i * window_step + offset
            this_window = x[start_index: start_index + overall_window_size: stride]
            if flatten is not None and flatten == 'majority':
                this_window = stats.mode(this_window, keepdims=False)[0]
            windows.append(this_window)
        return windows  # np.array(windows)

    def resamples(self, x, factor, neighbours=False):
        size = round(factor * len(x))
        if neighbours:
            positions = np.linspace(0, len(x)-1, size)
            return x[positions.astype(int)]

        x_resampled = signal.resample(x, size)
        return x_resampled

    @staticmethod
    def headers():
        axes = ['x', 'y', 'z']
        IMUsensor_columns = ['temperature'] + \
                            ['acc_16g_' + i for i in axes] + \
                            ['acc_6g_' + i for i in axes] + \
                            ['gyroscope_' + i for i in axes] + \
                            ['magnometer_' + i for i in axes] + \
                            ['orientation_' + str(i) for i in range(4)]

        header = ["timestamp", "activityID", "heartrate"] + \
                 ["hand_" + s for s in IMUsensor_columns] + \
                 ["chest_" + s for s in IMUsensor_columns] + \
                 ["ankle_" + s for s in IMUsensor_columns]
        return header


if __name__ == '__main__':
    ds = PAMAP2(users='full', window_size=200, frequency=50, columns=None, train_users=[1, 3, 4, 5, 6, 8])
    print(f"ds size: {len(ds)}\n"
          f"data frequency {ds.frequency}")

    for i, (x, y) in enumerate(ds):
        print(x.shape, y)
        if np.isnan(x.numpy()).sum() > 0:
            print("Error ...")
            break
        if i > 5:
            break
    # print(ds.headers())
    print(f"Labels: {ds.labels}")
    print("Features: ", ds.features)
    print("#"*100)
