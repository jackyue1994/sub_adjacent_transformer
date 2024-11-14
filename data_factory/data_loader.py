import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import pickle


class PSMSegLoader(object):
    def __init__(self, data_path, win_size, step, ratio=1, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.ratio = ratio
        self.scaler = StandardScaler()
        # self.scaler2 = RobustScaler()
        data = pd.read_csv(os.path.join(data_path, 'train.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)

        self.scaler.fit(data)

        test_data = pd.read_csv(os.path.join(data_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)

        print(f'before transform: train data max: {np.max(data)}; min: {np.min(data)}')
        print(f'before transform: test data max: {np.max(test_data)}; min: {np.min(test_data)}')

        data = self.scaler.transform(data)
        self.test = self.scaler.transform(test_data)

        print(f'train data max: {np.max(data)}; min: {np.min(data)}')
        print(f'test data max: {np.max(self.test)}; min: {np.min(self.test)}')

        self.train = data
        self.val = self.test

        self.test_labels = pd.read_csv(os.path.join(data_path, 'test_label.csv')).values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (int(self.train.shape[0]*self.ratio) - self.win_size) // self.step + 1
        elif self.mode == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'val':
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'test':
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return (np.float32(self.test[
                               index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]),
                    np.float32(self.test_labels[
                               index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]))


class MSLSegLoader(object):
    def __init__(self, data_path, win_size, step, ratio=1, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.ratio = ratio
        self.scaler = StandardScaler()
        self.scaler2 = RobustScaler()

        files = os.listdir(data_path)
        train_file = [file for file in files if 'train' in file and '.npy' in file][0]
        test_file = [file for file in files if 'test' in file and '.npy' in file and 'label' not in file][0]
        test_label_file = [file for file in files if 'test' in file and '.npy' in file and 'label' in file][0]
        # len_file = [file for file in files if 'entity-len' in file and '.npy' in file]
        len_file = []

        data = np.load(os.path.join(data_path, train_file))
        test_data = np.load(os.path.join(data_path, test_file))

        print(f'before transform: train data max: {np.max(data)}; min: {np.min(data)}')  # , axis=0
        print(f'before transform: test data max: {np.max(test_data)}; min: {np.min(test_data)}')

        if len_file:
            print(f"{len_file[0]} is used...")
            entity_file = os.path.join(data_path, len_file[0])
            tmp = np.load(entity_file, allow_pickle=True).item()
            train_start_idx = tmp['train-start']
            train_end_idx = tmp['train-end']
            test_start_idx = tmp['test-start']
            test_end_idx = tmp['test-end']
        else:
            print("entity-len.npy is not found...")
            train_start_idx = test_start_idx = [0, ]
            train_end_idx = [data.shape[0], ]
            test_end_idx = [test_data.shape[0], ]

        # preprocess by each entity
        data_list = []
        test_data_list = []
        for i in range(len(train_start_idx)):
            train_start = train_start_idx[i]
            train_end = train_end_idx[i]
            test_start = test_start_idx[i]
            test_end = test_end_idx[i]

            data_i = data[train_start:train_end, :]
            test_data_i = test_data[test_start:test_end, :]

            all_data = np.concatenate([data_i, test_data_i], axis=0)
            std_list = all_data.std(axis=0)
            columns_to_keep = np.where(std_list > -1)[0]

            self.scaler.fit(data_i[:, columns_to_keep])
            self.scaler2.fit(data_i[:, columns_to_keep])

            data_i[:, columns_to_keep] = self.scaler.transform(data_i[:, columns_to_keep])
            test_data_i[:, columns_to_keep] = self.scaler2.transform(test_data_i[:, columns_to_keep])

            data_list.append(data_i)
            test_data_list.append(test_data_i)

        data = np.concatenate(data_list, axis=0)
        test_data = np.concatenate(test_data_list, axis=0)

        # check
        ind = np.where(np.max(data, axis=0) > 100)[0]
        ind2 = np.where(np.max(test_data, axis=0) > 100)[0]
        if ind.any() or ind2.any():
            print("There are large values in data or test_data. Please check...")

        self.train = data
        self.test = test_data
        self.val = test_data
        self.test_labels = np.load(os.path.join(data_path, test_label_file))

        print(f'after transform: train data max: {np.max(data)}; min: {np.min(data)}')
        print(f'after transform: test data max: {np.max(self.test)}; min: {np.min(self.test)}')

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.mode == "train":
            return (int(self.train.shape[0]*self.ratio) - self.win_size) // self.step + 1
        elif self.mode == 'train-nolap':
            return (self.test.shape[0] - self.win_size) // self.win_size + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'train-nolap':
            start = index // self.step * self.win_size
            return (np.float32(self.test[start:start + self.win_size]),
                    np.float32(self.test_labels[0:self.win_size]))
        elif self.mode == 'test':
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            start = index // self.step * self.win_size
            return np.float32(self.test[start:start + self.win_size]), np.float32(
                self.test_labels[start:start + self.win_size])


class SMAPSegLoader(object):
    def __init__(self, data_path, win_size, step, ratio=1, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.ratio = ratio
        self.scaler = StandardScaler()
        self.scaler2 = RobustScaler()

        files = os.listdir(data_path)
        train_file = [file for file in files if 'train' in file and '.npy' in file][0]
        test_file = [file for file in files if 'test' in file and '.npy' in file and 'label' not in file][0]
        test_label_file = [file for file in files if 'test' in file and '.npy' in file and 'label' in file][0]
        len_file = [file for file in files if 'entity-len' in file and '.npy' in file]

        data = np.load(os.path.join(data_path, train_file))
        test_data = np.load(os.path.join(data_path, test_file))

        print(f'before transform: train data max: {np.max(data)}; min: {np.min(data)}')  # , axis=0
        print(f'before transform: test data max: {np.max(test_data)}; min: {np.min(test_data)}')

        if len_file:
            print(f"{len_file[0]} is used...")
            entity_file = os.path.join(data_path, len_file[0])
            tmp = np.load(entity_file, allow_pickle=True).item()
            train_start_idx = tmp['train-start']
            train_end_idx = tmp['train-end']
            test_start_idx = tmp['test-start']
            test_end_idx = tmp['test-end']
        else:
            print("entity-len.npy is not found...")
            train_start_idx = test_start_idx = [0, ]
            train_end_idx = [data.shape[0], ]
            test_end_idx = [test_data.shape[0], ]

        # preprocess by each entity
        data_list = []
        test_data_list = []
        for i in range(len(train_start_idx)):
            train_start = train_start_idx[i]
            train_end = train_end_idx[i]
            test_start = test_start_idx[i]
            test_end = test_end_idx[i]

            data_i = data[train_start:train_end, :]
            test_data_i = test_data[test_start:test_end, :]

            all_data = np.concatenate([data_i, test_data_i], axis=0)
            std_list = all_data.std(axis=0)
            columns_to_keep = np.where(std_list > -1)[0]  # do not delete any columns

            self.scaler.fit(test_data_i[:, columns_to_keep])
            # self.scaler2.fit(all_data[:, columns_to_keep])

            data_i[:, columns_to_keep] = self.scaler.transform(data_i[:, columns_to_keep])
            test_data_i[:, columns_to_keep] = self.scaler.transform(test_data_i[:, columns_to_keep])

            data_list.append(data_i)
            test_data_list.append(test_data_i)

        data = np.concatenate(data_list, axis=0)
        test_data = np.concatenate(test_data_list, axis=0)

        # check
        ind = np.where(np.max(data, axis=0) > 100)[0]
        ind2 = np.where(np.max(test_data, axis=0) > 100)[0]
        if ind.any() or ind2.any():
            print("There are large values in data or test_data. Please check...")

        self.train = data
        self.test = test_data
        self.val = test_data
        self.test_labels = np.load(os.path.join(data_path, test_label_file))

        print(f'after transform: train data max: {np.max(data)}; min: {np.min(data)}')
        print(f'after transform: test data max: {np.max(self.test)}; min: {np.min(self.test)}')

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.mode == "train":
            return (int(self.train.shape[0]*self.ratio) - self.win_size) // self.step + 1
        elif self.mode == 'train-nolap':
            return (self.test.shape[0] - self.win_size) // self.win_size + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'train-nolap':
            start = index // self.step * self.win_size
            return (np.float32(self.test[start:start + self.win_size]),
                    np.float32(self.test_labels[0:self.win_size]))
        elif self.mode == 'test':
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            start = index // self.step * self.win_size
            return np.float32(self.test[start:start + self.win_size]), np.float32(
                self.test_labels[start:start + self.win_size])


class SMDSegLoader(object):
    def __init__(self, data_path, win_size, step, ratio=1, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.ratio = ratio
        self.scaler = StandardScaler()

        files = os.listdir(data_path)
        train_file = [file for file in files if 'train' in file and '.npy' in file][0]
        test_file = [file for file in files if 'test' in file and '.npy' in file and 'label' not in file][0]
        test_label_file = [file for file in files if 'test' in file and '.npy' in file and 'label' in file][0]
        len_file = [file for file in files if 'entity-len' in file and '.npy' in file]

        data = np.load(os.path.join(data_path, train_file))
        test_data = np.load(os.path.join(data_path, test_file))

        print(f'before transform: train data max: {np.max(data)}; min: {np.min(data)}')  # , axis=0
        print(f'before transform: test data max: {np.max(test_data)}; min: {np.min(test_data)}')

        if len_file:
            print(f"{len_file[0]} is used...")
            entity_file = os.path.join(data_path, len_file[0])
            tmp = np.load(entity_file, allow_pickle=True).item()
            train_start_idx = tmp['train-start']
            train_end_idx = tmp['train-end']
            test_start_idx = tmp['test-start']
            test_end_idx = tmp['test-end']
        else:
            print("entity-len.npy is not found...")
            train_start_idx = test_start_idx = [0, ]
            train_end_idx = [data.shape[0], ]
            test_end_idx = [test_data.shape[0], ]

        # preprocess by each entity
        data_list = []
        test_data_list = []
        for i in range(len(train_start_idx)):
            train_start = train_start_idx[i]
            train_end = train_end_idx[i]
            test_start = test_start_idx[i]
            test_end = test_end_idx[i]

            data_i = data[train_start:train_end, :]
            test_data_i = test_data[test_start:test_end, :]

            all_data = np.concatenate([data_i, test_data_i], axis=0)
            std_list = all_data.std(axis=0)
            columns_to_keep = np.where(std_list > -1)[0]
            # print((std_list > 1e-3).all())

            data_i[:, columns_to_keep] = self.scaler.fit_transform(data_i[:, columns_to_keep])
            test_data_i[:, columns_to_keep] = self.scaler.fit_transform(test_data_i[:, columns_to_keep])

            data_list.append(data_i)
            test_data_list.append(test_data_i)

        data = np.concatenate(data_list, axis=0)
        test_data = np.concatenate(test_data_list, axis=0)

        # check
        ind = np.where(np.max(data, axis=0) > 100)[0]
        ind2 = np.where(np.max(test_data, axis=0) > 100)[0]
        if ind.any() or ind2.any():
            print("There are large values in data or test_data. Please check...")

        self.train = data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        # self.val = self.test
        self.test_labels = np.load(os.path.join(data_path, test_label_file))

        print(f'after transform: train data max: {np.max(data)}; min: {np.min(data)}')
        print(f'after transform: test data max: {np.max(self.test)}; min: {np.min(self.test)}')

        print("train:", self.train.shape)
        print("test:", self.test.shape)

    def __len__(self):
        if self.mode == "train":
            return (int(self.train.shape[0]*self.ratio) - self.win_size) // self.step + 1
        elif self.mode == 'train-nolap':
            return (self.train.shape[0] - self.win_size) // self.win_size + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'train-nolap':
            start = index // self.step * self.win_size
            return (np.float32(self.train[start:start + self.win_size]),
                    np.float32(self.test_labels[0:self.win_size]))
        elif self.mode == 'test':
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            start = index // self.step * self.win_size
            return np.float32(self.test[start:start + self.win_size]), np.float32(
                self.test_labels[start:start + self.win_size])


class SWaTSegLoader(object):
    def __init__(self, data_path, win_size, step, ratio=1, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.ratio = ratio

        self.scaler = StandardScaler()

        files = os.listdir(data_path)
        train_file = [file for file in files if 'train' in file and '.npy' in file][0]
        test_file = [file for file in files if 'test' in file and '.npy' in file and 'label' not in file][0]
        test_label_file = [file for file in files if 'test' in file and '.npy' in file and 'label' in file][0]

        data = np.load(os.path.join(data_path, train_file))
        test_data = np.load(os.path.join(data_path, test_file))

        print(f'before transform: train data max: {np.max(data)}; min: {np.min(data)}')  # , axis=0
        print(f'before transform: test data max: {np.max(test_data)}; min: {np.min(test_data)}')

        data = self.scaler.fit_transform(data)
        test_data = self.scaler.fit_transform(test_data)

        # check
        ind = np.where(np.max(data, axis=0) > 100)[0]
        ind2 = np.where(np.max(test_data, axis=0) > 100)[0]
        if ind.any() or ind2.any():
            print("There are large values in data or test_data. Please check...")

        self.train = data
        self.test = test_data

        self.val = self.test
        self.test_labels = np.load(os.path.join(data_path, test_label_file))

        print(f'after transform: train data max: {np.max(data)}; min: {np.min(data)}')
        print(f'after transform: test data max: {np.max(self.test)}; min: {np.min(self.test)}')

        print("train:", self.train.shape)
        print("test:", self.test.shape)

    def __len__(self):
        if self.mode == "train":
            return (int(self.train.shape[0]*self.ratio) - self.win_size) // self.step + 1
        elif self.mode == 'train-nolap':
            return (self.train.shape[0] - self.win_size) // self.win_size + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'train-nolap':
            start = index // self.step * self.win_size
            return (np.float32(self.train[start:start + self.win_size]),
                    np.float32(self.test_labels[0:self.win_size]))
        elif self.mode == 'test':
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            start = index // self.step * self.win_size
            return np.float32(self.test[start:start + self.win_size]), np.float32(
                self.test_labels[start:start + self.win_size])


class WADISegLoader(object):
    def __init__(self, data_path, win_size, step, ratio=1, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.ratio = ratio

        self.scaler = StandardScaler()

        files = os.listdir(data_path)
        train_file = [file for file in files if 'train' in file and '.npy' in file][0]
        test_file = [file for file in files if 'test' in file and '.npy' in file and 'label' not in file][0]
        test_label_file = [file for file in files if 'test' in file and '.npy' in file and 'label' in file][0]

        data = np.load(os.path.join(data_path, train_file))
        test_data = np.load(os.path.join(data_path, test_file))
        # just in case
        data = np.nan_to_num(data)
        test_data = np.nan_to_num(test_data)

        print(f'before transform: train data max: {np.max(data)}; min: {np.min(data)}')  # , axis=0
        print(f'before transform: test data max: {np.max(test_data)}; min: {np.min(test_data)}')

        data = self.scaler.fit_transform(data)
        test_data = self.scaler.fit_transform(test_data)

        # check
        ind = np.where(np.max(data, axis=0) > 100)[0]
        ind2 = np.where(np.max(test_data, axis=0) > 100)[0]
        if ind.any() or ind2.any():
            print("There are large values in data or test_data. Please check...")

        self.train = data
        self.test = test_data

        self.val = self.test
        self.test_labels = np.load(os.path.join(data_path, test_label_file))

        print(f'after transform: train data max: {np.max(data)}; min: {np.min(data)}')
        print(f'after transform: test data max: {np.max(self.test)}; min: {np.min(self.test)}')

        print("train:", self.train.shape)
        print("test:", self.test.shape)

    def __len__(self):
        if self.mode == "train":
            return (int(self.train.shape[0]*self.ratio) - self.win_size) // self.step + 1
        elif self.mode == 'train-nolap':
            return (self.test.shape[0] - self.win_size) // self.win_size + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'train-nolap':
            start = index // self.step * self.win_size
            return (np.float32(self.test[start:start + self.win_size]),
                    np.float32(self.test_labels[0:self.win_size]))
        elif self.mode == 'test':
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            start = index // self.step * self.win_size
            return np.float32(self.test[start:start + self.win_size]), np.float32(
                self.test_labels[start:start + self.win_size])


class NIPSSegLoader(object):
    def __init__(self, data_path, win_size, step, ratio=1, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.ratio = ratio

        self.scaler = StandardScaler()

        files = os.listdir(data_path)
        train_file = [file for file in files if 'train' in file and '.npy' in file][0]
        test_file = [file for file in files if 'test' in file and '.npy' in file and 'label' not in file][0]
        test_label_file = [file for file in files if 'test' in file and '.npy' in file and 'label' in file][0]

        data = np.load(os.path.join(data_path, train_file))
        test_data = np.load(os.path.join(data_path, test_file))
        # just in case
        data = np.nan_to_num(data)
        test_data = np.nan_to_num(test_data)

        print(f'before transform: train data max: {np.max(data)}; min: {np.min(data)}')  # , axis=0
        print(f'before transform: test data max: {np.max(test_data)}; min: {np.min(test_data)}')

        data = self.scaler.fit_transform(data)
        test_data = self.scaler.fit_transform(test_data)

        # check
        ind = np.where(np.max(data, axis=0) > 100)[0]
        ind2 = np.where(np.max(test_data, axis=0) > 100)[0]
        if ind.any() or ind2.any():
            print("There are large values in data or test_data. Please check...")

        self.train = data
        self.test = test_data

        self.val = self.test
        self.test_labels = np.load(os.path.join(data_path, test_label_file))

        print(f'after transform: train data max: {np.max(data)}; min: {np.min(data)}')
        print(f'after transform: test data max: {np.max(self.test)}; min: {np.min(self.test)}')

        print("train:", self.train.shape)
        print("test:", self.test.shape)

    def __len__(self):
        if self.mode == "train":
            return (int(self.train.shape[0]*self.ratio) - self.win_size) // self.step + 1
        elif self.mode == 'train-nolap':
            return (self.test.shape[0] - self.win_size) // self.win_size + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'train-nolap':
            start = index // self.step * self.win_size
            return (np.float32(self.test[start:start + self.win_size]),
                    np.float32(self.test_labels[0:self.win_size]))
        elif self.mode == 'test':
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            start = index // self.step * self.win_size
            return np.float32(self.test[start:start + self.win_size]), np.float32(
                self.test_labels[start:start + self.win_size])


def get_loader_segment(data_path, batch_size, win_size=100, step=10, ratio=1, mode='train', dataset='KDD'):
    if dataset.lower().startswith('smd'):
        step = 10
        dataset = SMDSegLoader(data_path, win_size, step, ratio=ratio, mode=mode)
    elif dataset.lower().startswith('msl'):
        dataset = MSLSegLoader(data_path, win_size, 1, ratio=ratio, mode=mode)
    elif dataset.lower().startswith('smap'):
        dataset = SMAPSegLoader(data_path, win_size, 1, ratio=ratio, mode=mode)
    elif dataset.lower().startswith('psm'):
        dataset = PSMSegLoader(data_path, win_size, 1, ratio=ratio, mode=mode)
    elif dataset.lower().startswith('swat'):
        dataset = SWaTSegLoader(data_path, win_size, step, ratio=ratio, mode=mode)
    elif dataset.lower().startswith('wadi'):
        dataset = WADISegLoader(data_path, win_size, step, ratio=ratio, mode=mode)
    elif dataset.lower().startswith('nips'):
        dataset = NIPSSegLoader(data_path, win_size, 1, ratio=ratio, mode=mode)
    else:
        raise ValueError("Dataset setting error")

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader
