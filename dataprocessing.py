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
        if db == "YouTubeface_sel":
            # 加载数据
            mat = sio.loadmat(os.path.join(path, 'YouTubeface_sel.mat'))
            X_data = mat['X']

            # [修正点] 您的数据形状是 (5, 1)，所以视图数是行数 (shape[0])
            self.num_views = X_data.shape[0]

            for idx in range(self.num_views):
                # [修正点] 使用 [idx, 0] 读取数据，而不是 [0, idx]
                view_data = X_data[idx, 0]

                # 确保数据类型为 float32
                self.data_views.append(view_data.astype(np.float32))

            # [强烈推荐] 开启归一化，因为特征维度差异大，且数值范围可能不同
            scaler = MinMaxScaler()
            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])

            # 处理标签
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

            # 2. 数据归一化 (强烈建议保留)
            scaler = MinMaxScaler()
            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])
            # 3. 处理标签 Y
            # 您的 Keys 包含 'Y'，直接读取即可
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)


        elif db == "cifar10":
            mat = sio.loadmat(os.path.join(path, 'cifar10.mat'))

            # 1. 读取 Data
            X_data = mat['data']
            # 结构是 (3, 1)，所以视图数是 3
            self.num_views = X_data.shape[0]

            for idx in range(self.num_views):
                # 读取第 idx 个视图的数据
                # 原始形状是 (特征, 样本)，例如 (512, 50000)
                view_data = X_data[idx, 0]

                # [关键操作] 转置！变成 (50000, 512) 以适配 PyTorch
                view_data = view_data.T

                # 确保是 float32
                self.data_views.append(view_data.astype(np.float32))

            # 归一化 (深度特征数值差异大，推荐开启)
            scaler = MinMaxScaler()
            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])

            # 2. 读取 Label
            # 结构是 (3, 1)，我们取第 0 个块即可
            Y = mat['truelabel'][0, 0]
            # 原始是 1-10，PyTorch 需要 0-9，所以要 -1
            self.labels = np.array(np.squeeze(Y)).astype(np.int32) - 1

        elif db == "cifar100":
            mat = sio.loadmat(os.path.join(path, 'cifar100.mat'))

            # 1. 读取 Data
            X_data = mat['data']
            # 结构是 (3, 1)，所以视图数是 3
            self.num_views = X_data.shape[0]

            for idx in range(self.num_views):
                # 读取第 idx 个视图的数据
                # 原始形状是 (特征, 样本)，例如 (512, 50000)
                view_data = X_data[idx, 0]

                # [关键操作] 转置！变成 (50000, 512) 以适配 PyTorch
                view_data = view_data.T

                # 确保是 float32
                self.data_views.append(view_data.astype(np.float32))

            # 归一化 (深度特征数值差异大，推荐开启)
            scaler = MinMaxScaler()
            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])

            # 2. 读取 Label
            # 结构是 (3, 1)，我们取第 0 个块即可
            Y = mat['truelabel'][0, 0]
            # 原始是 1-10，PyTorch 需要 0-9，所以要 -1
            self.labels = np.array(np.squeeze(Y)).astype(np.int32) - 1


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


        elif db == "NoisyMNIST30000":

            mat = sio.loadmat(os.path.join(path, 'NoisyMNIST30000.mat'))


            X1 = mat['X1'].astype(np.float32)

            X2 = mat['X2'].astype(np.float32)

            self.data_views.append(X1)

            self.data_views.append(X2)


            self.num_views = len(self.data_views)

            # 归一化 (配合 main.py 的 normalized=True)

            scaler = MinMaxScaler()

            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])

            # 读取标签

            # 假设标签变量名为 'Y' 或 'truelabel'



            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

            # 如果标签是 1-10，需要转为 0-9

            if np.min(self.labels) == 1:
                self.labels = self.labels - 1

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