import os
import ast
import torch
import random
import argparse
import numpy as np


from data_loader.loader_trans import Loader
from core import test
from core.base_trans import Base
from core.train_trans import train, train_stage1
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

    loader = loaders.get_train_loader()
    import pickle
    data = loader.samplers

    # for i, (input1_0, input2, label1, label2) in enumerate(loader):
    #     print(i, input1_0.shape, input2.shape, label1.shape, label2.shape)

if __name__ == '__main__':
    # set multi-processing start method
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=str, default='cuda')
    parser.add_argument('--mode', type=str, default='train', help='train, test')
    parser.add_argument('--test_mode', default='all', type=str, help='all or indoor')
    parser.add_argument('--gall_mode', default='single', type=str, help='single or multi')
    parser.add_argument('--regdb_test_mode', default='v-t', type=str, help='')
    parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
    # parser.add_argument('--sysu_data_path', type=str, default='E:/hhj/SYSU-MM01-PART/')
    parser.add_argument('--sysu_data_path', type=str, default='D:/hhj/SYSU-MM01/')
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
    parser.add_argument('--num_workers', default=4, type=int,
                        help='num of pos per identity in each modality')
    # parser.add_argument('--output_path', type=str, default='models/base/',
    #                     help='path to save related informations')
    parser.add_argument('--output_path', type=str, default='D:/PretrainModel/CSDN/models/base/',
                        help='path to save related informations')
    parser.add_argument('--max_save_model_num', type=int, default=1, help='0 for max num is infinit')
    parser.add_argument('--resume_train_epoch', type=int, default=-1, help='-1 for no resuming')
    parser.add_argument('--auto_resume_training_from_lastest_step', type=ast.literal_eval, default=True)
    parser.add_argument('--total_train_epoch', type=int, default=120)
    parser.add_argument('--eval_epoch', type=int, default=1)
    parser.add_argument('--resume_test_model', type=int, default=105, help='-1 for no resuming')

    config = parser.parse_args()
    seed_torch(config.seed)
    main(config)