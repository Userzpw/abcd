import argparse
import warnings

from models import *
from layers import *
from loss import *
from visualization import plot_metric_comparison, plot_multiview_tsne
from metrics import calculate_metrics as clustering
import torch
import scipy.io as sio
import os
# --- [新增] 强制设置 joblib 的核心数，防止它去查询系统报错 ---
os.environ['LOKY_MAX_CPU_COUNT'] = '4'
# -------------------------------------------------------

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='CVCLNet')
parser.add_argument('--missing_rate', type=float, default=0.5, help='缺失率 (0.0 - 0.9)')
parser.add_argument('--load_model', default=False, help='Testing if True or training.')
parser.add_argument('--save_model', default=False, help='Saving the model after training.')

parser.add_argument('--db', type=str, default='Fashion',
                    choices=['MSRCv1', 'MNIST-USPS', 'COIL20', 'scene', 'hand', 'Fashion', 'BDGP'],
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
print("==========\nArgs:{}\n==========".format(args))

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
        args.con_epochs = 10
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

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mv_data = MultiviewData(args.db, device, missing_rate=args.missing_rate)
    num_views = len(mv_data.data_views)
    num_samples = mv_data.labels.size
    num_clusters = np.unique(mv_data.labels).size

    input_sizes = np.zeros(num_views, dtype=int)
    for idx in range(num_views):
        input_sizes[idx] = mv_data.data_views[idx].shape[1]

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

        # 1. 拿到预测结果
        total_pred, pred_vectors, labels_vector = inference(mnw, mv_data, args.batch_size)

        # 2. 自动算分
        metrics_data = {}

        # 算单视图
        for i, view_pred in enumerate(pred_vectors):
            # --- [关键修改] ---
            # 你的 metrics.py 定义是 calculate_metrics(label, pred) -> (True, Pred)
            # 所以这里要把 labels_vector (真实值) 放在前面！
            acc, nmi, pur, ari = clustering(labels_vector, view_pred)
            # -----------------

            metrics_data[f'View {i + 1}'] = [acc, nmi, pur, ari]
            print(f"View {i + 1}: ACC={acc:.4f}, NMI={nmi:.4f}")

        # 算融合
        # --- [关键修改] 同样要把 labels_vector 放在前面 ---
        acc, nmi, pur, ari = clustering(labels_vector, total_pred)
        # -----------------------------------------------

        metrics_data['Fusion'] = [acc, nmi, pur, ari]
        print(f"Fusion: ACC={acc:.4f}, NMI={nmi:.4f}")

        # 3. 自动画图
        # 确保 visualization.py 里的函数已经 import 进来了
        try:
            plot_metric_comparison(metrics_data, save_name=f'compare_metrics_{args.db}.png')
            plot_multiview_tsne(mnw, mv_data, args.batch_size, device, save_name=f'compare_tsne_{args.db}.png')
            print("可视化完成！请查看生成的 png 图片。")
        except NameError:
            print("提示：请确保在 main.py 开头导入了 plot_metric_comparison 和 plot_multiview_tsne")

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
