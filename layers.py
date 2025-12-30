import torch
import torch.nn as nn

# --- 原型层定义 ---
class PrototypeLayer(nn.Module):
    def __init__(self, input_dim, num_clusters, alpha=1.0):
        super(PrototypeLayer, self).__init__()
        self.alpha = alpha
        self.cluster_centers = nn.Parameter(torch.Tensor(num_clusters, input_dim))
        nn.init.xavier_normal_(self.cluster_centers.data)

    def forward(self, x):
        norm_squared = torch.sum((x.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        q = 1.0 / (1.0 + norm_squared / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = torch.t(torch.t(q) / torch.sum(q, 1))
        return q

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, feature_dim, dims):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential()
        for i in range(len(dims)+1):
            if i == 0:
                self.encoder.add_module('Linear%d' % i,  nn.Linear(input_dim, dims[i]))
            elif i == len(dims):
                self.encoder.add_module('Linear%d' % i, nn.Linear(dims[i-1], feature_dim))
            else:
                self.encoder.add_module('Linear%d' % i, nn.Linear(dims[i-1], dims[i]))
            self.encoder.add_module('relu%d' % i, nn.ReLU())

    def forward(self, x):
        return self.encoder(x)


class AutoDecoder(nn.Module):
    def __init__(self, input_dim, feature_dim, dims):
        super(AutoDecoder, self).__init__()
        self.decoder = nn.Sequential()
        dims = list(reversed(dims))
        for i in range(len(dims)+1):
            if i == 0:
                self.decoder.add_module('Linear%d' % i,  nn.Linear(feature_dim, dims[i]))
            elif i == len(dims):
                self.decoder.add_module('Linear%d' % i, nn.Linear(dims[i-1], input_dim))
            else:
                self.decoder.add_module('Linear%d' % i, nn.Linear(dims[i-1], dims[i]))
            self.decoder.add_module('relu%d' % i, nn.ReLU())

    def forward(self, x):
        return self.decoder(x)


class CVCLNetwork(nn.Module):
    def __init__(self, num_views, input_sizes, dims, dim_high_feature, dim_low_feature, num_clusters):
        super(CVCLNetwork, self).__init__()
        self.encoders = list()
        self.decoders = list()
        for idx in range(num_views):
            self.encoders.append(AutoEncoder(input_sizes[idx], dim_high_feature, dims))
            self.decoders.append(AutoDecoder(input_sizes[idx], dim_high_feature, dims))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

        # [修改点 1] 替换 MLP 为 映射层 + 原型层
        self.feature_map = nn.Linear(dim_high_feature, dim_low_feature)
        self.prototype_layer = PrototypeLayer(dim_low_feature, num_clusters)

    def forward(self, data_views):
        lbps = list()
        dvs = list()
        features = list()

        num_views = len(data_views)
        for idx in range(num_views):
            data_view = data_views[idx]
            high_features = self.encoders[idx](data_view)

            # [修改点 2] 前向传播逻辑
            low_feature = self.feature_map(high_features)  # 降维
            label_probs = self.prototype_layer(low_feature)  # 计算概率

            data_view_recon = self.decoders[idx](high_features)

            features.append(low_feature)  # 保存低维特征供初始化用
            lbps.append(label_probs)
            dvs.append(data_view_recon)

        return lbps, dvs, features
