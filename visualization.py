import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from dataprocessing import get_multiview_data


# ==========================================
# 核心工具：提取特征
# ==========================================

def extract_common_features(network_model, mv_data, batch_size, device='cuda'):
    """
    提取【融合后】的特征 (使用 Mask 过滤掉无效数据)
    """
    network_model.eval()
    data_loader, num_views, num_samples, _ = get_multiview_data(mv_data, batch_size)

    all_features = []
    all_labels = []

    print("正在提取融合特征...")
    with torch.no_grad():
        for batch_idx, (sub_data_views, sub_labels, masks) in enumerate(data_loader):
            masks = masks.to(device)
            _, _, features = network_model(sub_data_views)

            # --- 融合逻辑 ---
            stacked_features = torch.stack(features, dim=1)
            masks_expanded = masks.unsqueeze(2)
            sum_features = torch.sum(stacked_features * masks_expanded, dim=1)
            valid_counts = torch.sum(masks_expanded, dim=1)
            valid_counts[valid_counts == 0] = 1.0
            avg_features = sum_features / valid_counts

            all_features.append(avg_features.cpu().numpy())
            all_labels.append(sub_labels)

    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)


def extract_view_features(network_model, mv_data, view_idx, batch_size, device='cuda'):
    """
    提取【单个视图】的特征 (不进行过滤，直接看原始效果，包含均值填充的噪声)
    """
    network_model.eval()
    data_loader, _, _, _ = get_multiview_data(mv_data, batch_size)

    all_features = []
    all_labels = []

    # print(f"正在提取 View {view_idx+1} 特征...")
    with torch.no_grad():
        for batch_idx, (sub_data_views, sub_labels, _) in enumerate(data_loader):
            _, _, features = network_model(sub_data_views)
            # 直接取指定视图的特征
            feat = features[view_idx]
            all_features.append(feat.cpu().numpy())
            all_labels.append(sub_labels)

    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)


# ==========================================
# 绘图函数 1: 简单的 t-SNE (只画融合结果)
# ==========================================

def plot_tsne(network_model, mv_data, batch_size, device='cuda', save_name='tsne_result.png'):
    features, labels = extract_common_features(network_model, mv_data, batch_size, device)

    print("正在运行 t-SNE...")
    tsne = TSNE(n_components=2, init='pca', random_state=42)
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    # 尝试获取 tab10 颜色，如果报错则回退
    try:
        colors = plt.cm.get_cmap('tab10', len(unique_labels))
    except:
        colors = plt.cm.get_cmap('jet', len(unique_labels))

    for i, label in enumerate(unique_labels):
        indices = labels == label
        plt.scatter(features_2d[indices, 0], features_2d[indices, 1],
                    color=colors(i), label=f'Cluster {label}', s=10, alpha=0.7)

    plt.title('t-SNE: Fused Features', fontsize=15)
    plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    print(f"图片已保存: {save_name}")
    # plt.close() # 如果不想在窗口弹出，可以取消注释


# ==========================================
# 绘图函数 2: 混淆矩阵
# ==========================================

def plot_confusion(y_true, y_pred, save_name='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix', fontsize=15)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Cluster')
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    print(f"图片已保存: {save_name}")


# ==========================================
# 绘图函数 3: 多视图 t-SNE 对比 (2x2 大图)
# ==========================================

def plot_multiview_tsne(network_model, mv_data, batch_size, device='cuda', save_name='multiview_tsne.png'):
    print("开始生成多视图 t-SNE 对比图 (比较耗时，请稍等)...")

    views_features = []
    labels = None

    # 1. 获取单视图特征
    num_views = len(mv_data.data_views)
    for v in range(num_views):
        feat, lbl = extract_view_features(network_model, mv_data, v, batch_size, device)
        views_features.append(feat)
        if labels is None: labels = lbl

    # 2. 获取融合特征
    fusion_feat, _ = extract_common_features(network_model, mv_data, batch_size, device)
    views_features.append(fusion_feat)

    titles = [f'View {i + 1} (Single)' for i in range(num_views)] + ['Fusion (Ours)']

    # 布局设置
    rows, cols = 2, 2
    if num_views > 3: rows, cols = 2, 3

    plt.figure(figsize=(16, 12))
    tsne = TSNE(n_components=2, init='pca', random_state=42)
    unique_labels = np.unique(labels)
    try:
        colors = plt.cm.get_cmap('tab10', len(unique_labels))
    except:
        colors = plt.cm.get_cmap('jet', len(unique_labels))

    for i, feats in enumerate(views_features):
        # 如果视图太多，超过了子图数量，就不画了
        if i >= rows * cols: break

        ax = plt.subplot(rows, cols, i + 1)
        print(f"  正在计算子图 {i + 1}/{len(views_features)}: {titles[i]}...")

        X_2d = tsne.fit_transform(feats)

        for l_idx, label in enumerate(unique_labels):
            indices = labels == label
            # 单视图的点画小一点，透明一点，因为可能有重叠
            ax.scatter(X_2d[indices, 0], X_2d[indices, 1],
                       color=colors(l_idx), s=5, alpha=0.6)

        ax.set_title(titles[i], fontsize=16, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    print(f"对比图已保存: {save_name}")


# ==========================================
# 绘图函数 4: 指标对比柱状图
# ==========================================

def plot_metric_comparison(metrics_dict, save_name='metric_comparison.png'):
    labels = ['ACC', 'NMI', 'PUR', 'ARI']
    x = np.arange(len(labels))
    width = 0.2

    plt.figure(figsize=(12, 6))
    num_items = len(metrics_dict)
    start_offset = - (num_items * width) / 2 + width / 2

    # 颜色列表
    color_list = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#c2c2f0']

    for i, (name, values) in enumerate(metrics_dict.items()):
        offset = start_offset + i * width
        c = color_list[i % len(color_list)]
        plt.bar(x + offset, values, width, label=name, color=c, edgecolor='white')

    plt.ylabel('Score')
    plt.title('Performance Comparison: Single Views vs. Fusion', fontsize=16)
    plt.xticks(x, labels, fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    print(f"指标图已保存: {save_name}")