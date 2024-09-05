import os
import ast
import torch
import random
import argparse
import numpy as np


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from data_loader.loader import Loader
from core import Base, train, train_stage1, train_stage2, test
from tools import make_dirs, Logger, os_walk, time_now
import warnings
warnings.filterwarnings("ignore")

best_mAP = 0
best_rank1 = 0
def seed_torch(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(config):
    global best_mAP
    global best_rank1

    loaders = Loader(config)
    model = Base(config)

    make_dirs(model.output_path)
    make_dirs(model.save_model_path)
    make_dirs(model.save_logs_path)

    logger = Logger(os.path.join(os.path.join(config.output_path, 'logs/'), 'log.txt'))
    logger('\n' * 3)
    logger(config)

    if config.mode == 'train':
        if config.resume_train_epoch >= 0:
            model.resume_model(config.resume_train_epoch)
            start_train_epoch = config.resume_train_epoch
        else:
            start_train_epoch = 0

        if config.auto_resume_training_from_lastest_step:
            root, _, files = os_walk(model.save_model_path)
            if len(files) > 0:
                indexes = []
                for file in files:
                    indexes.append(int(file.replace('.pth', '').split('_')[-1]))
                indexes = sorted(list(set(indexes)), reverse=False)
                model.resume_model(indexes[-1])
                start_train_epoch = indexes[-1]
                logger('Time: {}, automatically resume training from the latest step (model {})'.format(time_now(),
                                    indexes[-1]))

        print('Start the 3st Stage Training')
        print('Extracting Text Features')

        num_classes = model.model.module.num_classes
        batch = config.batch_size
        i_ter = num_classes // batch
        left = num_classes - batch * (num_classes // batch)
        if left != 0:
            i_ter = i_ter + 1
        text_features_rgb = []
        text_features_ir = []
        with torch.no_grad():
            for i in range(i_ter):
                if i + 1 != i_ter:
                    l_list = torch.arange(i * batch, (i + 1) * batch)
                else:
                    l_list = torch.arange(i * batch, num_classes)
                text_feature_rgb = model.model(label1=l_list, get_text=True)
                text_feature_ir = model.model(label2=l_list, get_text=True)
                text_features_rgb.append(text_feature_rgb.cpu())
                text_features_ir.append(text_feature_ir.cpu())
            text_features_rgb = torch.cat(text_features_rgb, 0)
            text_features_ir = torch.cat(text_features_ir, 0)
        print('Text Features Extracted, Start Training')

        def draw_tsne(rgb, ir):
            tsne = TSNE(n_components=2, init='pca', random_state=42, n_iter=1000)
            all_features = np.concatenate((rgb, ir), axis=0)
            X_tsne = tsne.fit_transform(all_features)
            X_tsne_rgb = X_tsne[:len(rgb)]
            X_tsne_ir = X_tsne[len(rgb):]

            # 创建一个颜色列表，长度与特征点数量相同
            colors = plt.cm.rainbow(np.linspace(0, 1, len(rgb)))

            plt.figure(figsize=(10, 5))
            # 对于rgb特征，使用五角星形状，颜色从颜色列表中获取
            plt.scatter(X_tsne_rgb[:, 0], X_tsne_rgb[:, 1] + 0.5, c=colors, label='rgb', marker='*')

            # 对于ir特征，使用默认的圆形形状，颜色从颜色列表中获取
            plt.scatter(X_tsne_ir[:, 0], X_tsne_ir[:, 1], c=colors, label='ir', marker='s')

            plt.legend()
            plt.show()

            print('t-SNE finished!')

        draw_tsne(text_features_rgb, text_features_ir)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=str, default='cuda')
    parser.add_argument('--mode', type=str, default='train', help='train, test')
    parser.add_argument('--test_mode', default='all', type=str, help='all or indoor')
    parser.add_argument('--gall_mode', default='single', type=str, help='single or multi')
    parser.add_argument('--regdb_test_mode', default='v-t', type=str, help='')
    parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
    parser.add_argument('--sysu_data_path', type=str, default='E:/hhj/SYSU-MM01-PART/')
    parser.add_argument('--regdb_data_path', type=str, default='/opt/data/private/data/RegDB/')
    parser.add_argument('--trial', default=1, type=int, help='trial (only for RegDB dataset)')
    parser.add_argument('--batch-size', default=32, type=int, metavar='B', help='training batch size')
    parser.add_argument('--img_w', default=144, type=int, metavar='imgw', help='img width')
    parser.add_argument('--img_h', default=288, type=int, metavar='imgh', help='img height')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--pid_num', type=int, default=395)
    parser.add_argument('--learning_rate', type=float, default=0.0003)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--milestones', nargs='+', type=int, default=[40, 70],
                        help='milestones for the learning rate decay')

    parser.add_argument('--stage1_batch-size', default=32, type=int, metavar='B', help='training batch size')
    parser.add_argument('--stage1_learning_rate', type=float, default=0.0003)
    parser.add_argument('--stage2_learning_rate', type=float, default=0.0003)
    parser.add_argument('--stage1_weight_decay', type=float, default=1e-4)
    parser.add_argument('--stage1_lr_min', type=float, default=1e-6)
    parser.add_argument('--stage1_warmup_lr_init', type=float, default=0.00001)
    parser.add_argument('--stage1_warmup_epochs', type=int, default=5)
    parser.add_argument('--stage1_train_epochs', type=int, default=60)

    parser.add_argument('--lambda1', type=float, default=0.15)
    parser.add_argument('--lambda2', type=float, default=0.05)
    parser.add_argument('--lambda3', type=float, default=0.1)

    parser.add_argument('--num_pos', default=4, type=int,
                        help='num of pos per identity in each modality')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='num of pos per identity in each modality')
    parser.add_argument('--output_path', type=str, default='models/base/',
                        help='path to save related informations')
    parser.add_argument('--max_save_model_num', type=int, default=1, help='0 for max num is infinit')
    parser.add_argument('--resume_train_epoch', type=int, default=-1, help='-1 for no resuming')
    parser.add_argument('--auto_resume_training_from_lastest_step', type=ast.literal_eval, default=True)
    parser.add_argument('--total_train_epoch', type=int, default=120)
    parser.add_argument('--eval_epoch', type=int, default=1)
    parser.add_argument('--resume_test_model', type=int, default=119, help='-1 for no resuming')

    config = parser.parse_args()
    seed_torch(config.seed)
    main(config)
