import time

import torch
import torch.nn as nn
from loss import *
from metrics import *
from dataprocessing import *
from sklearn.cluster import KMeans


# --- [新增] 生成对抗掩码 ---
def generate_adversarial_mask(data, gradients, mask_ratio=0.2):
    saliency = torch.abs(gradients)
    batch_size = data.shape[0]
    saliency_flat = saliency.view(batch_size, -1)

    k = int(data.shape[1] * mask_ratio)
    if k == 0: return torch.ones_like(data)

    # 找到前 k 大的梯度值作为阈值
    topk_values, _ = torch.topk(saliency_flat, k, dim=1)
    threshold = topk_values[:, -1].unsqueeze(1)

    # 遮挡梯度最大的部分 (显著性区域)
    mask = (saliency < threshold).float()
    return mask


# --- [新增] 初始化中心 ---
def initialize_centers(network_model, mv_data, batch_size, device='cuda'):
    print("正在使用 K-Means 初始化原型中心 (已过滤缺失样本)...")
    from sklearn.cluster import KMeans
    import numpy as np

    network_model.eval()
    # 接收 masks
    data_loader, num_views, num_samples, _ = get_multiview_data(mv_data, batch_size)

    valid_features = []
    with torch.no_grad():
        for batch_idx, (sub_data_views, _, masks) in enumerate(data_loader):
            _, _, features = network_model(sub_data_views)

            # 遍历每个视角
            for v in range(num_views):
                # features[v]: [batch, dim]
                # masks[:, v]: [batch]

                # 找出 mask 为 1 的索引
                valid_idx = torch.nonzero(masks[:, v]).squeeze()

                # 如果这个 batch 里该视角有有效数据
                if valid_idx.numel() > 0:
                    # 只取有效特征
                    feat = features[v][valid_idx]
                    if feat.dim() == 1:  # 防止只有一个样本时维度不对
                        feat = feat.unsqueeze(0)
                    valid_features.append(feat.cpu().numpy())

    # 拼接所有有效的特征
    if len(valid_features) > 0:
        all_features = np.concatenate(valid_features, axis=0)
        print(f"用于初始化的有效样本数: {all_features.shape[0]}")

        kmeans = KMeans(n_clusters=network_model.prototype_layer.cluster_centers.shape[0], n_init=20)
        kmeans.fit(all_features)
        predicted_centers = kmeans.cluster_centers_

        network_model.prototype_layer.cluster_centers.data = torch.tensor(predicted_centers).to(device)
        print("初始化完成。")
    else:
        print("错误：没有找到有效特征用于初始化！")


def pre_train(network_model, mv_data, batch_size, epochs, optimizer):
    t = time.time()
    # 1. 修改解包，接收 masks
    mv_data_loader, num_views, num_samples, _ = get_multiview_data(mv_data, batch_size)

    pre_train_loss_values = np.zeros(epochs, dtype=np.float64)

    # 2. 关键修改：reduction='none'
    # 原来是默认求平均，现在我们要拿到每个样本的具体误差，方便后续过滤
    criterion = torch.nn.MSELoss(reduction='none')

    for epoch in range(epochs):
        total_loss = 0.
        # 解包拿到 masks
        for batch_idx, (sub_data_views, _, masks) in enumerate(mv_data_loader):
            _, dvs, _ = network_model(sub_data_views)
            loss_list = list()

            for idx in range(num_views):
                # 计算重建误差
                # criterion 输出 shape: [batch_size, feature_dim]
                recon_loss = criterion(dvs[idx], sub_data_views[idx])

                # 对特征维度求平均，得到每个样本的重建误差
                # shape 变为: [batch_size]
                recon_loss = recon_loss.mean(dim=1)

                # 获取当前视角的 mask (1表示存在，0表示缺失)
                current_mask = masks[:, idx]

                # 3. 核心逻辑：只保留 mask=1 的样本误差
                # 计算有效样本数量
                valid_count = torch.sum(current_mask)

                if valid_count > 0:
                    # 误差 * 掩码 = 只剩下有效样本的误差（无效的变0）
                    # 然后除以有效样本数，求平均
                    loss_view = torch.sum(recon_loss * current_mask) / valid_count
                    loss_list.append(loss_view)

            # 只有当 loss_list 不为空时才反向传播
            if len(loss_list) > 0:
                loss = sum(loss_list)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        pre_train_loss_values[epoch] = total_loss
        if epoch % 10 == 0 or epoch == epochs - 1:
            print('Pre-training, epoch {}, Loss:{:.7f}'.format(epoch, total_loss / num_samples))

    print("Pre-training finished.")
    print("Total time elapsed: {:.4f}s".format(time.time() - t))

    return pre_train_loss_values


def contrastive_train(network_model, mv_data, mvc_loss, batch_size, lmd, beta, temperature_l, normalized, epoch,
                      optimizer):
    network_model.train()
    mv_data_loader, num_views, num_samples, num_clusters = get_multiview_data(mv_data, batch_size)
    criterion = torch.nn.MSELoss()
    total_loss = 0.

    # 设置遮挡比例
    MASK_RATIO = 0.

    for batch_idx, (sub_data_views, _, mask) in enumerate(mv_data_loader):

        # === 阶段 1: 寻找弱点 (生成对抗掩码) ===
        clean_inputs = []
        for v in range(num_views):
            # 开启梯度记录
            clean_inputs.append(sub_data_views[v].clone().detach().requires_grad_(True))

        # 预先跑一次模型，只为了算梯度
        lbps, _, _ = network_model(clean_inputs)

        # 只计算对比损失来寻找显著性区域
        temp_loss_list = []
        for i in range(num_views):
            for j in range(i + 1, num_views):
                temp_loss_list.append(lmd * mvc_loss.forward_label(lbps[i], lbps[j], temperature_l, normalized))

        # 反向传播获取梯度
        network_model.zero_grad()
        if len(temp_loss_list) > 0:
            sum(temp_loss_list).backward()

        # 生成带遮挡的输入
        masked_inputs = []
        for v in range(num_views):
            grad = clean_inputs[v].grad
            if grad is not None:
                mask = generate_adversarial_mask(sub_data_views[v], grad, MASK_RATIO)
                # 应用掩码
                masked_inputs.append(sub_data_views[v] * mask.detach())
            else:
                masked_inputs.append(sub_data_views[v])

        # === 阶段 2: 正式训练 (使用遮挡数据 + 原型约束) ===
        # 注意：现在输入的是 masked_inputs
        lbps, dvs, _ = network_model(masked_inputs)

        loss_list = list()
        for i in range(num_views):
            for j in range(i + 1, num_views):
                # 对比损失 (让被遮挡的样本依然能分类一致)
                loss_list.append(lmd * mvc_loss.forward_label(lbps[i], lbps[j], temperature_l, normalized))
                # 熵正则化
                loss_list.append(beta * mvc_loss.forward_prob(lbps[i], lbps[j]))

            # [关键改进] 重建损失：尝试从“遮挡图像”恢复出“原始图像”
            # 这比恢复遮挡图像本身更高级 (类似于 MAE 思想)
            loss_list.append(criterion(sub_data_views[i], dvs[i]))


        loss = sum(loss_list)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 10 == 0:
        print('Epoch {}, Loss:{:.7f}'.format(epoch, total_loss / num_samples))

    return total_loss

def inference(network_model, mv_data, batch_size):
    network_model.eval()
    # 1. 接收 mask
    mv_data_loader, num_views, num_samples, _ = get_multiview_data(mv_data, batch_size)

    soft_vector = []
    pred_vectors = []
    labels_vector = []
    for v in range(num_views):
        pred_vectors.append([])

    for batch_idx, (sub_data_views, sub_labels, masks) in enumerate(mv_data_loader):
        with torch.no_grad():
            lbps, _, _ = network_model(sub_data_views)

            # --- [关键修改] 动态加权求和 ---
            # 初始化总概率和计数器
            lbp_sum = torch.zeros_like(lbps[0])
            valid_mask_sum = torch.zeros(lbps[0].shape[0], 1).to(lbps[0].device)

            for idx in range(num_views):
                # 获取当前视角的 mask: [batch, 1]
                current_mask = masks[:, idx].unsqueeze(1)

                # 只累加存在的视图的预测概率
                lbp_sum += lbps[idx] * current_mask
                valid_mask_sum += current_mask

                # 记录单视角的预测 (可选，仅作参考)
                pred_label = torch.argmax(lbps[idx], dim=1)
                pred_vectors[idx].extend(pred_label.detach().cpu().numpy())

            # 计算平均概率 (避免除以0，加上一个极小值)
            lbp = lbp_sum / (valid_mask_sum + 1e-8)
            # ---------------------------

        soft_vector.extend(lbp.detach().cpu().numpy())
        labels_vector.extend(sub_labels)

    for idx in range(num_views):
        pred_vectors[idx] = np.array(pred_vectors[idx])

    actual_num_samples = len(soft_vector)
    labels_vector = np.array(labels_vector).reshape(actual_num_samples)
    total_pred = np.argmax(np.array(soft_vector), axis=1)

    return total_pred, pred_vectors, labels_vector

def valid(network_model, mv_data, batch_size):

    total_pred, pred_vectors, labels_vector = inference(network_model, mv_data, batch_size)
    num_views = len(mv_data.data_views)

    print("Clustering results on cluster assignments of each view:")
    for idx in range(num_views):
        acc, nmi, pur, ari = calculate_metrics(labels_vector,  pred_vectors[idx])
        print('ACC{} = {:.4f} NMI{} = {:.4f} PUR{} = {:.4f} ARI{}={:.4f}'.format(idx+1, acc,
                                                                                 idx+1, nmi,
                                                                                 idx+1, pur,
                                                                                 idx+1, ari))

    print("Clustering results on semantic labels: " + str(labels_vector.shape[0]))
    acc, nmi, pur, ari = calculate_metrics(labels_vector, total_pred)
    print('ACC = {:.4f} NMI = {:.4f} PUR = {:.4f} ARI={:.4f}'.format(acc, nmi, pur, ari))

    return acc, nmi, pur, ari
