import argparse
import warnings

from models import *
from layers import *
from loss import *
from visualization import plot_metric_comparison, plot_multiview_tsne, plot_all_views_tsne
from metrics import calculate_metrics as clustering
import torch
import scipy.io as sio
import os
# --- [新增] 强制设置 joblib 的核心数，防止它去查询系统报错 ---
os.environ['LOKY_MAX_CPU_COUNT'] = '4'
# -------------------------------------------------------

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='CVCLNet')
parser.add_argument('--mask_ratio', type=float, default=0.2, help='Adversarial masking ratio')
parser.add_argument('--missing_rate', type=float, default=0.5, help='缺失率 (0.0 - 0.9)')
parser.add_argument('--load_model', default=False, help='Testing if True or training.')
parser.add_argument('--save_model', default=False, help='Saving the model after training.')

parser.add_argument('--db', type=str, default='cifar100',
                    choices=['MSRCv1', 'MNIST-USPS', 'COIL20', 'scene', 'hand', 'Fashion', 'BDGP','YouTubeface_sel', 'cifar10', 'cifar100'],
                    help='dataset name')
parser.add_argument('--seed', type=int, default=10, help='Initializing random seed.')
parser.add_argument("--mse_epochs", default=200, help='Number of epochs to pre-training.')
parser.add_argument("--con_epochs", default=100, help='Number of epochs to fine-tuning.')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0005, help='Initializing learning rate.')
parser.add_argument('--weight_decay', type=float, default=0., help='Initializing weight decay.')
parser.add_argument("--temperature_l", type=float, default=1.0)
parser.add_argument('--batch_size', default=100, type=int,
                    help='The total number of samples must be evenly divisible by batch_size.')
parser.add_argument('--normalized', type=bool, default=False)
parser.add_argument('--gpu', default='0', type=str, help='GPU device idx.')

args = parser.parse_args()


# torch.cuda.set_device(0)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":

    if args.db == "MSRCv1":
        # db checked 97.62
        args.learning_rate = 0.0005
        args.batch_size = 35
        # args.con_epochs = 400
        args.con_epochs = 200
        args.seed = 10
        args.normalized = False

        dim_high_feature = 2000
        dim_low_feature = 1024
        dims = [256, 512]
        lmd = 0.01
        beta = 0.005

    elif args.db == "MNIST-USPS":
        # db checked 99.7
        args.learning_rate = 0.0001
        args.batch_size = 50
        args.seed = 10
        args.con_epochs = 200
        args.normalized = False

        dim_high_feature = 1500
        dim_low_feature = 1024
        dims = [256, 512, 1024]
        lmd = 0.05
        beta = 0.05

    elif args.db == "COIL20":
        # db checked 84.65
        args.learning_rate = 0.0005
        args.batch_size = 180
        args.seed = 50
        args.con_epochs = 400
        args.normalized = False

        dim_high_feature = 768
        dim_low_feature = 200
        dims = [256, 512, 1024, 2048]
        lmd = 0.01
        beta = 0.01

    elif args.db == "scene":
        # db checked 44.59
        args.learning_rate = 0.0005
        args.con_epochs = 100
        args.batch_size = 69
        args.seed = 10
        args.normalized = False

        dim_high_feature = 1500
        dim_low_feature = 256
        dims = [256, 512, 1024, 2048]
        lmd = 0.01
        beta = 0.05

    elif args.db == "hand":
        # db checked 96.85
        args.learning_rate = 0.0001
        args.batch_size = 200
        args.seed = 50
        args.con_epochs = 200
        args.normalized = True

        dim_high_feature = 1024
        dim_low_feature = 1024
        dims = [256, 512, 1024]
        lmd = 0.005
        beta = 0.001

    elif args.db == "Fashion":
        # db checked 99.31
        args.learning_rate = 0.0005
        args.batch_size = 100
        args.con_epochs = 20
        args.seed = 20
        args.normalized = True
        args.temperature_l = 0.5

        dim_high_feature = 2000
        dim_low_feature = 500
        dims = [256, 512]
        lmd = 0.005
        beta = 0.005

    elif args.db == "BDGP":
        # db checked 99.2
        args.learning_rate = 0.0001
        args.batch_size = 250
        args.seed = 10
        args.con_epochs = 100
        args.normalized = True

        dim_high_feature = 2000
        dim_low_feature = 1024
        dims = [256, 512]
        lmd = 0.01
        beta = 0.01

    elif args.db == "YouTubeface_sel":
        # 基础训练参数
        args.learning_rate = 0.0005
        args.batch_size = 5000  # 样本量大，建议 500~1000
        args.con_epochs = 100  # 建议跑满 100 轮以观察收敛
        args.seed = 10
        args.normalized = True  # 配合 dataprocessing 的归一化

        # --- 网络结构配置 (适配混合维度) ---

        # 1. 编码器结构 (dims):
        # 对于 64维视图：64 -> 256 -> 512 (温和扩张)
        # 对于 838维视图：838 -> 256 -> 512 (压缩后再提取)
        dims = [256, 512]

        # 2. 高层特征 (dim_high_feature):
        # 所有视图编码后统一映射到的维度，1024 是一个通用且足够大的值
        dim_high_feature = 1024

        # 3. 聚类特征 (dim_low_feature):
        # 最终用于聚类的低维空间，通常比 high 小
        dim_low_feature = 512

        # --- 损失权重 ---
        lmd = 0.01  # 聚类损失权重 (必须 > 0)
        beta = 0.01  # 熵正则化权重

    elif args.db == "cifar10":
        # 数据量 50000，特征维度大，建议使用较小的学习率和较大的 Batch Size
        args.learning_rate = 0.0005
        args.batch_size = 200  # 5万数据，Batch 设大点跑得快
        args.con_epochs = 100
        args.seed = 10
        args.normalized = True  # 开启归一化

        # --- 网络结构 ---
        # 输入维度分别是 512, 2048, 1024
        # 策略：先统一降维到 1024，再映射到 2000

        dim_high_feature = 2000
        dim_low_feature = 1024  # 聚类空间大一点，因为 CIFAR 语义复杂

        # 编码器层结构
        # 512 -> 1024
        # 2048 -> 1024
        # 1024 -> 1024
        dims = [1024]

        # 损失权重
        lmd = 0.01
        beta = 0.01


    elif args.db == "cifar100":

        # === 关键修改 1: 降低学习率 ===

        # CIFAR-100 很敏感，0.0005 可能太大导致震荡，改用 0.0001

        args.learning_rate = 0.0001

        args.batch_size = 100


        args.mse_epochs = 200

        args.con_epochs = 100

        args.seed = 10

        args.normalized = True


        args.temperature_l = 0.5

        # [重中之重] 网络容量

        # dim_high 设大一点，保留更多信息

        dim_high_feature = 2048

        # dim_low 不要压得太狠，1024 对于 100 个类来说是底线

        dim_low_feature = 1024

        # [核心] 加深网络层数！

        # 之前的层数太浅，无法从 512/1024 的输入中提取出抽象的语义

        # 尝试使用 3-4 层全连接网络

        dims = [2048, 2048]

        # === 关键修改 5: 调整损失权重 ===

        # lmd (对比损失): 降低！

        # 因为 View 1/3 很弱，如果 lmd 太大，它们会被强行拉向 View 2 的位置（而 View 2 初始也不准），导致集体坍塌。

        # 先让它们稍微自由一点。

        lmd = 0.01  # 之前可能是 0.01 或 0.05

        # beta (熵正则): 提高！

        # 强迫每个视图输出的概率分布更“尖锐”，不要大家都输出 0.01 的均匀分布

        beta = 0.005
    print("==========\nArgs:{}\n==========".format(args))
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mv_data = MultiviewData(args.db, device, missing_rate=args.missing_rate)
    num_views = len(mv_data.data_views)
    num_samples = mv_data.labels.size
    num_clusters = np.unique(mv_data.labels).size

    input_sizes = np.zeros(num_views, dtype=int)
    for idx in range(num_views):
        input_sizes[idx] = mv_data.data_views[idx].shape[1]
    print("每个视图的输入特征维度 (input_sizes):", input_sizes)

    t = time.time()
    # neural network architecture
    mnw = CVCLNetwork(num_views, input_sizes, dims, dim_high_feature, dim_low_feature, num_clusters)
    # filling it into GPU
    mnw = mnw.to(device)

    mvc_loss = DeepMVCLoss(args.batch_size, num_clusters)
    optimizer = torch.optim.Adam(mnw.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.load_model:
        state_dict = torch.load('./models/CVCL_pytorch_model_%s.pth' % args.db)
        mnw.load_state_dict(state_dict)


    else:

        # 1. 预训练 (Pre-train)

        pre_train_loss_values = pre_train(mnw, mv_data, args.batch_size, args.mse_epochs, optimizer)

        # --- [插入] 初始化原型中心 ---

        # 此时 AutoEncoder 已经预训练好了，提取的特征比较靠谱

        initialize_centers(mnw, mv_data, args.batch_size, device)

        # ---------------------------

        t = time.time()

        fine_tuning_loss_values = np.zeros(args.con_epochs, dtype=np.float64)

        # 2. 对抗训练 (Contrastive Train with Masking)

        # 这里调用的已经是我们修改过的带 Mask 的函数了

        for epoch in range(args.con_epochs):
            total_loss = contrastive_train(mnw, mv_data, mvc_loss, args.batch_size, lmd, beta,

                                           args.temperature_l, args.normalized, epoch, optimizer)

            fine_tuning_loss_values[epoch] = total_loss

        print("contrastive_train finished.")
        print("Total time elapsed: {:.2f}s".format(time.time() - t))

        if args.save_model:
            torch.save(mnw.state_dict(), './models/CVCL_pytorch_model_%s.pth' % args.db)

    acc, nmi, pur, ari = valid(mnw, mv_data, args.batch_size)
    with open('result_%s.txt' % args.db, 'a+') as f:
        f.write('{} \t {} \t {} \t {} \t {} \t {} \t {} \t {:.6f} \t {:.6f} \t {:.6f} \t {:.4f} \n'.format(
            dim_high_feature, dim_low_feature, args.seed, args.batch_size,
            args.learning_rate, lmd, beta, acc, nmi, pur, (time.time() - t)))
        f.flush()
        # ... (前面的代码：acc, nmi, pur, ari = valid(...)) ...
        # ... (前面的代码：写入 result.txt) ...



        print("\n========== 开始可视化 & 自动评估 ==========")
        # ... (之前的代码: valid, 写入 result.txt 等)

        # === 新增：可视化部分 ===
        print("\nStarting Visualization...")

        # 1. 可视化融合后的特征 (总结果)
        plot_multiview_tsne(mnw, mv_data, args.batch_size, device, save_path=f'tsne_fused_{args.db}.png')

        # 2. [新功能] 可视化每个视图单独的特征
        # 这将生成 tsne_view_1_[db].png, tsne_view_2_[db].png 等
        plot_all_views_tsne(mnw, mv_data, args.batch_size, device, prefix=f'tsne_view_{args.db}')

        # 3. 可视化 Loss 曲线
        if 'fine_tuning_loss_values' in locals():
            plot_metric_comparison(fine_tuning_loss_values, title=f"Contrastive Loss ({args.db})",
                                   save_path=f'loss_{args.db}.png')

    # dim_high_features = np.array([2000, 1500, 1024, 1000, 768, 512, 500, 256, 200], dtype=np.int32)
    # dim_low_features = np.array([2000, 1500, 1024, 1000, 768, 512, 500, 256, 200], dtype=np.int32)
    # seeds = np.array([10, 20, 50], dtype=np.int32)
    # # dims_layers = np.array([[256, 512, 1024]])
    # # dims_layers = np.array([[256, 512], [256, 512, 1024], [256, 512, 1024, 2048]])
    # dims_layers = [[256, 512], [256, 512, 1024], [256, 512, 1024, 2048]]
    # batch_sizes = np.array([20, 30, 50, 60], dtype=np.int32)
    # lambdas = np.array([0.005, 0.01, 0.05], dtype=np.float32)
    # betas = np.array([0.005, 0.01, 0.05], dtype=np.float32)
    # learning_rates = np.array([0.0001, 0.0005], dtype=np.float32)
    # for dh_idx in range(dim_high_features.shape[0]):
    #     dim_high_feature = dim_high_features[dh_idx]
    #     for dl_idx in range(dh_idx, dim_low_features.shape[0]):
    #         dim_low_feature = dim_low_features[dl_idx]
    #         for sd_idx in range(seeds.shape[0]):
    #             seed = seeds[sd_idx]
    #             for dim_idx in range(len(dims_layers)):
    #                 dims = np.array(dims_layers[dim_idx])
    #                 for bs_idx in range(batch_sizes.shape[0]):
    #                     batch_size = int(batch_sizes[bs_idx])
    #                     for lmd_idx in range(lambdas.shape[0]):
    #                         lmd = lambdas[lmd_idx]
    #                         for beta_idx in range(betas.shape[0]):
    #                             beta = betas[beta_idx]
    #                             for lr_idx in range(learning_rates.shape[0]):
    #                                 learning_rate = learning_rates[lr_idx]
    #
    #                                 set_seed(args.seed)
    #                                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #                                 mv_data = MultiviewData(args.db, device)
    #                                 num_views = len(mv_data.data_views)
    #                                 num_samples = mv_data.labels.size
    #                                 num_clusters = np.unique(mv_data.labels).size
    #
    #                                 input_sizes = np.zeros(num_views, dtype=int)
    #                                 for idx in range(num_views):
    #                                     input_sizes[idx] = mv_data.data_views[idx].shape[1]
    #
    #                                 t = time.time()
    #                                 # neural network architecture
    #                                 mnw = CVCLNetwork(num_views, input_sizes, dims, dim_high_feature,
    #                                                   dim_low_feature, num_clusters)
    #                                 # filling it into GPU
    #                                 mnw = mnw.to(device)
    #
    #                                 mvc_loss = DeepMVCLoss(batch_size, num_clusters)
    #                                 optimizer = torch.optim.Adam(mnw.parameters(), lr=learning_rate,
    #                                                              weight_decay=args.weight_decay)
    #                                 pre_train(mnw, mv_data, batch_size, args.mse_epochs, optimizer)
    #
    #                                 for epoch in range(args.con_epochs):
    #                                     total_loss = contrastive_train(mnw, mv_data, mvc_loss, batch_size, lmd,
    #                                                                    beta, args.temperature_l, args.normalized,
    #                                                                    epoch, optimizer)
    #
    #                                 print("contrastive_train finished.")
    #                                 print("Total time elapsed: {:.2f}s".format(time.time() - t))
    #
    #                                 acc, nmi, pur, ari = valid(mnw, mv_data, batch_size)
    #                                 with open(args.db + '_result.txt', 'a+') as f:
    #                                     f.write('{} \t {} \t {} \t {} \t {} \t {:.4f} \t {:.3f} \t {:.3f} \t {:.6f} '
    #                                             '\t {:.6f} \t {:.6f} \t {:.6f} \t {:.4f} \n'.format(
    #                                         dim_idx, dim_high_feature, dim_low_feature, seed, batch_size,
    #                                         learning_rate, lmd, beta, acc, nmi, pur, ari, (time.time() - t)))
    #                                     f.flush()
