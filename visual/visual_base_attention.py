import os
import torch
import torch.nn as nn

from bisect import bisect_right
from .visual_model_attention import Model
from network.lr import CosineLRScheduler
from tools import os_walk, CrossEntropyLabelSmooth, SupConLoss, TripletLoss_WRT, hcc_euc, hcc_kl, hcc_kl_3, ptcc, ptcc_3


class Base:
    def __init__(self, config):
        self.config = config

        self.pid_num = config.pid_num

        self.max_save_model_num = config.max_save_model_num
        self.output_path = config.output_path
        self.save_model_path = os.path.join(self.output_path, 'models/')
        self.save_logs_path = os.path.join(self.output_path, 'logs/')

        self.learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay
        self.milestones = config.milestones

        self.img_h = config.img_h
        self.img_w = config.img_w

        self.stage1_learning_rate = config.stage1_learning_rate
        self.stage2_learning_rate = config.stage2_learning_rate
        self.stage1_weight_decay = config.stage1_weight_decay
        self.stage1_train_epochs = config.stage1_train_epochs
        self.stage1_lr_min = config.stage1_lr_min
        self.stage1_warmup_lr_init = config.stage1_warmup_lr_init
        self.stage1_warmup_epochs = config.stage1_warmup_epochs

        self._init_device()
        self._init_model()

    def _init_device(self):
        self.device = torch.device('cuda')

    def _init_model(self):

        self.model = Model(self.pid_num, self.img_h, self.img_w)
        self.model = nn.DataParallel(self.model).to(self.device)


    def save_model(self, save_epoch, is_best):
        if is_best:
            model_file_path = os.path.join(self.save_model_path, 'model_{}.pth'.format(save_epoch))
            torch.save(self.model.state_dict(), model_file_path)

        if self.max_save_model_num > 0:
            root, _, files = os_walk(self.save_model_path)
            for file in files:
                if '.pth' not in file:
                    files.remove(file)
            if len(files) > 1 * self.max_save_model_num:
                file_iters = sorted([int(file.replace('.pth', '').split('_')[1]) for file in files], reverse=False)

                model_file_path = os.path.join(root, 'model_{}.pth'.format(file_iters[0]))
                os.remove(model_file_path)

    def resume_last_model(self):
        root, _, files = os_walk(self.save_model_path)
        for file in files:
            if '.pth' not in file:
                files.remove(file)
        if len(files) > 0:
            indexes = []
            for file in files:
                indexes.append(int(file.replace('.pth', '').split('_')[-1]))
            indexes = sorted(list(set(indexes)), reverse=False)
            self.resume_model(indexes[-1])
            start_train_epoch = indexes[-1]
            return start_train_epoch
        else:
            return 0

    def resume_model(self, resume_epoch):
        model_path = os.path.join(self.save_model_path, 'model_{}.pth'.format(resume_epoch))
        self.model.load_state_dict(torch.load(model_path), strict=False)
        print('Successfully resume model from {}'.format(model_path))

    def set_train(self):
        self.model = self.model.train()

        self.training = True

    def set_eval(self):
        self.model = self.model.eval()

        self.training = False

