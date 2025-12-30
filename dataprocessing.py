import os, random, sys

import torch
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler

from torch.utils.data import Dataset
# from torch.nn.functional import normalize
from utils import *


class MultiviewData(Dataset):
    def __init__(self, db, device, path="datasets/", missing_rate=0.0):
        self.data_views = list()
        self.mean_vectors = list()  # [新增] 用于存储每个视图的均值

        # ==========================================
        # 第一部分：加载数据
        # ==========================================
        if db == "MSRCv1":
            mat = sio.loadmat(os.path.join(path, 'MSRCv1.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            scaler = MinMaxScaler()
            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "MNIST-USPS":
            mat = sio.loadmat(os.path.join(path, 'MNIST_USPS.mat'))
            X1 = mat['X1'].astype(np.float32)
            X2 = mat['X2'].astype(np.float32)
            self.data_views.append(X1.reshape(X1.shape[0], -1))
            self.data_views.append(X2.reshape(X2.shape[0], -1))
            self.num_views = len(self.data_views)
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "BDGP":
            mat = sio.loadmat(os.path.join(path, 'BDGP.mat'))
            X1 = mat['X1'].astype(np.float32)
            X2 = mat['X2'].astype(np.float32)
            self.data_views.append(X1)
            self.data_views.append(X2)
            self.num_views = len(self.data_views)
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "Fashion":
            mat = sio.loadmat(os.path.join(path, 'Fashion.mat'))
            # Fashion 需要 reshape 展平
            X1 = mat['X1'].reshape(mat['X1'].shape[0], mat['X1'].shape[1] * mat['X1'].shape[2]).astype(np.float32)
            X2 = mat['X2'].reshape(mat['X2'].shape[0], mat['X2'].shape[1] * mat['X2'].shape[2]).astype(np.float32)
            X3 = mat['X3'].reshape(mat['X3'].shape[0], mat['X3'].shape[1] * mat['X3'].shape[2]).astype(np.float32)
            self.data_views.append(X1)
            self.data_views.append(X2)
            self.data_views.append(X3)
            self.num_views = len(self.data_views)
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "COIL20":
            mat = sio.loadmat(os.path.join(path, 'COIL20.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            scaler = MinMaxScaler()
            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "hand":
            mat = sio.loadmat(os.path.join(path, 'handwritten.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            scaler = MinMaxScaler()
            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])
            self.labels = np.array(np.squeeze(mat['Y']) + 1).astype(np.int32)

        elif db == "scene":
            mat = sio.loadmat(os.path.join(path, 'Scene15.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            scaler = MinMaxScaler()
            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        else:
            raise NotImplementedError

        # ==========================================
        # 第二部分：数据预处理 (计算均值 + 生成掩码)
        # ==========================================

        # 1. [新增] 计算每个视图的均值 (用于填充)
        # 必须在转 GPU 之前计算
        for idx in range(self.num_views):
            # 计算均值 (numpy操作)
            mean_vec = np.mean(self.data_views[idx], axis=0)
            # 转 tensor 并存起来
            self.mean_vectors.append(torch.from_numpy(mean_vec).to(device))

        # 2. 生成缺失掩码
        self.missing_rate = missing_rate
        num_samples = self.labels.shape[0]
        self.masks = np.ones((num_samples, self.num_views), dtype=np.float32)

        if missing_rate > 0:
            print(f"正在模拟缺失数据 (均值填充模式)... 缺失率: {missing_rate}")
            for i in range(num_samples):
                while True:
                    rand_probs = np.random.rand(self.num_views)
                    sample_mask = (rand_probs > missing_rate).astype(np.float32)
                    if np.sum(sample_mask) > 0:
                        self.masks[i] = sample_mask
                        break

        # 转 GPU
        self.masks = torch.from_numpy(self.masks).to(device)
        for idx in range(self.num_views):
            self.data_views[idx] = torch.from_numpy(self.data_views[idx]).to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sub_data_views = list()
        mask = self.masks[index]

        for view_idx in range(self.num_views):
            data_view = self.data_views[view_idx]

            if mask[view_idx] == 1:
                # 如果存在，返回真实数据
                sub_data_views.append(data_view[index])
            else:
                # [关键修改] 如果缺失，返回该视图的“平均值”，而不是全 0
                sub_data_views.append(self.mean_vectors[view_idx])

        return sub_data_views, self.labels[index], mask


def get_multiview_data(mv_data, batch_size):
    num_views = len(mv_data.data_views)
    num_samples = len(mv_data.labels)
    num_clusters = len(np.unique(mv_data.labels))

    mv_data_loader = torch.utils.data.DataLoader(
        mv_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    return mv_data_loader, num_views, num_samples, num_clusters


def get_all_multiview_data(mv_data):
    num_views = len(mv_data.data_views)
    num_samples = len(mv_data.labels)
    num_clusters = len(np.unique(mv_data.labels))

    mv_data_loader = torch.utils.data.DataLoader(
        mv_data,
        batch_size=num_samples,
        shuffle=True,
        drop_last=True,
    )

    return mv_data_loader, num_views, num_samples, num_clusters