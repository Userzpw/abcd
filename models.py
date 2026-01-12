import time

import torch
import torch.nn as nn
from loss import *
from metrics import *
from dataprocessing import *
from sklearn.cluster import KMeans


# --- [新增] 生成对抗掩码 ---
def generate_adversarial_mask(data, gradients, mask_ratio=0.2):
    """
    生成对抗掩码：遮挡梯度最大的特征区域
    """
    saliency = torch.abs(gradients)
    if saliency.sum() == 0:
        return generate_random_mask(data, mask_ratio)
    batch_size = data.shape[0]
    saliency_flat = saliency.view(batch_size, -1)

    k = int(data.shape[1] * mask_ratio)
    if k == 0: return torch.ones_like(data)

    # 找到前 k 大的梯度值作为阈值
    topk_values, _ = torch.topk(saliency_flat, k, dim=1)
    # 取第 k 大的值作为阈值，这里要注意维度的对齐
    threshold = topk_values[:, -1].unsqueeze(1)

    # 遮挡梯度最大的部分 (显著性区域大于阈值的置为0，其余为1)
    mask = (saliency < threshold).float()
    return mask



# --- [新增] 生成随机掩码 (Random Masking) ---
def generate_random_mask(data, mask_ratio=0.5):
    """
    生成随机二值掩码
    :param data: 输入数据 [batch_size, feature_dim]
    :param mask_ratio: 遮挡比例 (0.0 - 1.0), 例如 0.5 表示随机遮挡 50% 的特征
    :return: mask tensor (1 表示保留, 0 表示遮挡)
    """
    if mask_ratio <= 0:
        return torch.ones_like(data)


    # 生成与数据同形状的随机概率矩阵 [0, 1)
    probs = torch.rand_like(data)
    # 大于 mask_ratio 的位置保留(1)，小于等于的遮挡(0)
    mask = (probs > mask_ratio).float()

    return mask


# --- [修改] 对比训练函数 (集成对抗掩码) ---
def contrastive_train(network_model, mv_data, mvc_loss, batch_size, lmd, beta,
                      temperature_l, normalized, epoch, optimizer, mask_ratio=0.5):
    network_model.train()
    mv_data_loader, num_views, num_samples, num_clusters = get_multiview_data(mv_data, batch_size)

    criterion_recon = torch.nn.MSELoss(reduction='none')
    total_loss = 0.

    # === [策略] 设定对抗攻击的预热期 ===
    # 前 50 个 Epoch 用随机掩码 (温和)
    # 50 个 Epoch 后用对抗掩码 (激进)
    WARMUP_EPOCHS = 50
    use_adversarial = epoch >= WARMUP_EPOCHS

    for batch_idx, (sub_data_views, _, masks) in enumerate(mv_data_loader):

        final_inputs = []  # 最终喂给网络训练的输入

        # ==========================================
        # 分支 A: 对抗掩码生成 (需要计算输入梯度)
        # ==========================================
        if use_adversarial:
            # 1. 复制一份数据，开启梯度记录
            adv_data_views = []
            for v in range(num_views):
                # .detach().clone() 断开历史图，.requires_grad_(True) 开启输入梯度
                d = sub_data_views[v].detach().clone().requires_grad_(True)
                adv_data_views.append(d)

            # 2. 试探性前向传播 (为了骗取梯度)
            # 这里我们只关心 Loss 对输入的梯度，不需要更新网络参数
            lbps, dvs, _ = network_model(adv_data_views)

            # 3. 计算一个临时 Loss (只为了反向传播求梯度)
            # 这里简单用对比损失之和即可，目的是找到“最影响聚类”的特征
            adv_loss = 0.
            for i in range(num_views):
                for j in range(i + 1, num_views):
                    mask_pair = masks[:, i] * masks[:, j]
                    if torch.sum(mask_pair) > 0:
                        # 简单计算，不用太精确，只要梯度方向对就行
                        adv_loss += mvc_loss.forward_prob(lbps[i], lbps[j])

            # 4. 反向传播，获取输入梯度
            network_model.zero_grad()  # 清空网络权重梯度 (安全起见)
            if adv_loss != 0:
                adv_loss.backward()

            # 5. 生成对抗掩码并应用
            for v in range(num_views):
                if adv_data_views[v].grad is not None:
                    # 获取梯度
                    data_grad = adv_data_views[v].grad.data
                    # 生成对抗掩码
                    adv_mask = generate_adversarial_mask(sub_data_views[v], data_grad, mask_ratio)
                    # 应用掩码
                    final_inputs.append(sub_data_views[v] * adv_mask)
                else:
                    # 如果没有梯度 (比如该视图所有样本都缺失)，则保持原样或随机掩码
                    final_inputs.append(sub_data_views[v] * generate_random_mask(sub_data_views[v], mask_ratio))

            # 清空梯度，防止影响后续真正的训练
            network_model.zero_grad()

        # ==========================================
        # 分支 B: 随机掩码 (预热期或降级方案)
        # ==========================================
        else:
            for v in range(num_views):
                rnd_mask = generate_random_mask(sub_data_views[v], mask_ratio)
                final_inputs.append(sub_data_views[v] * rnd_mask)

        # ==========================================
        # 正式训练 (Real Training Step)
        # ==========================================
        # 使用处理好(被遮挡)的 final_inputs 进行真正的前向传播
        lbps, dvs, _ = network_model(final_inputs)

        loss_list = list()

        # ... (以下是你原本的 Loss 计算代码，保持不变) ...
        # --- A. 对比损失 ---
        for i in range(num_views):
            for j in range(i + 1, num_views):
                mask_i = masks[:, i]
                mask_j = masks[:, j]
                mask_pair = mask_i * mask_j
                if torch.sum(mask_pair) > 0:
                    valid_indices = torch.nonzero(mask_pair).squeeze()
                    if valid_indices.dim() == 0: valid_indices = valid_indices.unsqueeze(0)

                    lbp_i_valid = lbps[i][valid_indices]
                    lbp_j_valid = lbps[j][valid_indices]

                    loss_contrast = mvc_loss.forward_label(lbp_i_valid, lbp_j_valid, temperature_l, normalized)
                    loss_prob = mvc_loss.forward_prob(lbp_i_valid, lbp_j_valid)

                    loss_list.append(lmd * loss_contrast)
                    loss_list.append(beta * loss_prob)

            # --- B. 重建损失 ---
            recon_loss = criterion_recon(dvs[i], sub_data_views[i]).mean(dim=1)
            current_mask = masks[:, i]
            if torch.sum(current_mask) > 0:
                loss_view = torch.sum(recon_loss * current_mask) / torch.sum(current_mask)
                loss_list.append(loss_view)

        # 反向传播更新参数
        if len(loss_list) > 0:
            loss = sum(loss_list)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    # 打印部分增加信息
    if epoch % 10 == 0:
        mode = "Adversarial" if use_adversarial else "Random"
        print('Epoch {} [{} Mask], Loss:{:.7f}'.format(epoch, mode, total_loss / num_samples))

    return total_loss


# ... (保留 initialize_centers, pre_train 等其他函数不变) ...

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

def inference(network_model, mv_data, batch_size):
    network_model.eval()
    # 1. 接收 mask
    mv_data_loader, num_views, num_samples, _ = get_multiview_data(mv_data, batch_size)

    soft_vector = []
    # 初始化为 list of lists
    pred_vectors = [[] for _ in range(num_views)]
    labels_vector = []

    for batch_idx, (sub_data_views, sub_labels, masks) in enumerate(mv_data_loader):
        with torch.no_grad():
            lbps, _, _ = network_model(sub_data_views)

            # --- 动态加权求和 ---
            lbp_sum = torch.zeros_like(lbps[0])
            valid_mask_sum = torch.zeros(lbps[0].shape[0], 1).to(lbps[0].device)

            for idx in range(num_views):
                # 获取当前视角的 mask
                current_mask = masks[:, idx].unsqueeze(1)

                # 累加
                lbp_sum += lbps[idx] * current_mask
                valid_mask_sum += current_mask

                # [修复点1] 记录单视角预测 -> 显式转为 cpu numpy 再转 list，确保是标量列表
                pred_label = torch.argmax(lbps[idx], dim=1)
                pred_vectors[idx].extend(pred_label.cpu().numpy().flatten().tolist())

            # 计算平均概率
            lbp = lbp_sum / (valid_mask_sum + 1e-8)
            # ---------------------------

        # [修复点2] 统一转 list
        soft_vector.extend(lbp.cpu().numpy().tolist())

        # [修复点3] 安全处理 Labels，防止 Tensor/Numpy 混用
        if isinstance(sub_labels, torch.Tensor):
            labels_vector.extend(sub_labels.cpu().numpy().flatten().tolist())
        else:
            labels_vector.extend(np.array(sub_labels).flatten().tolist())

    # [修复点4] 最终生成数组时，再次强制 flatten，确保是一维向量
    labels_vector = np.array(labels_vector).flatten()
    total_pred = np.argmax(np.array(soft_vector), axis=1)

    # 处理每个视角的预测向量
    for idx in range(num_views):
        pred_vectors[idx] = np.array(pred_vectors[idx]).flatten()

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
