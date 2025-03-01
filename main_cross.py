
import os
import ast
import torch
import random
import argparse
import numpy as np


from data_loader.loader_cross import Loader
from core import test
from core.base_cross import Base
from core.train_cross import train, train_stage1, train_stage1_3share, train_stage1_randomcolor, train_2rgb
from tools import make_dirs, Logger, os_walk, time_now
import warnings
warnings.filterwarnings("ignore")


best_mAP = 0
best_rank1 = 0
best_rank10 = 0
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
    global best_rank10

    loaders = Loader(config)
    model = Base(config)

    make_dirs(model.output_path)
    make_dirs(model.save_model_path)
    make_dirs(model.save_logs_path)

    logger = Logger(os.path.join(os.path.join(config.output_path, 'logs/'), 'log.txt'))
    logger('\n' * 3)
    logger(config)

    logger('Start Time: {})'.format(time_now()))

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

        # logger('Start the 1st Stage of Training')
        # logger('Extracting Image Features')
        #
        # visible_image_features = []
        # visible_labels = []
        # infrared_image_features = []
        # infrared_labels = []
        #
        # with torch.no_grad():
        #     for i, data in enumerate(loaders.get_train_normal_loader()):
        #         rgb_imgs, rgb_pids = data[0].to(model.device), data[2].to(model.device)
        #         ir_imgs, ir_pids = data[1].to(model.device), data[3].to(model.device)
        #         rgb_image_features_proj = model.model(x1=rgb_imgs, get_image=True)
        #         ir_image_features_proj = model.model(x2=ir_imgs, get_image=True)
        #         for i, j, img_feat1, img_feat2 in zip(rgb_pids, ir_pids, rgb_image_features_proj, ir_image_features_proj):
        #             visible_labels.append(i)
        #             visible_image_features.append(img_feat1.cpu())
        #             infrared_labels.append(j)
        #             infrared_image_features.append(img_feat2.cpu())
        #     visible_labels_list = torch.stack(visible_labels, dim=0).cuda()
        #     infrared_labels_list = torch.stack(infrared_labels, dim=0).cuda()
        #     visible_image_features_list = torch.stack(visible_image_features, dim=0).cuda()
        #     infrared_image_features_list = torch.stack(infrared_image_features, dim=0).cuda()
        #     batch = config.stage1_batch_size
        #     # num_image = infrared_labels_list.shape[0]
        #     num_image = visible_labels_list.shape[0]
        #     i_ter = num_image // batch
        # del visible_labels, visible_image_features, infrared_labels, infrared_image_features
        # logger('Image Features Extracted, Start Training')
        #
        # model._init_optimizer_stage1()
        #
        # for current_epoch in range(start_train_epoch, config.stage1_train_epochs):
        #     model.model_lr_scheduler_stage1.step(current_epoch)
        #     _, result = train_stage1_3share(model, num_image, i_ter, batch, visible_labels_list,
        #                              visible_image_features_list, infrared_labels_list, infrared_image_features_list)
        #     logger('Time: {}; Epoch: {}; LR: {}; {}'.format(time_now(), current_epoch,
        #                                                     model.model_lr_scheduler_stage1._get_lr
        #                                                     (current_epoch)[0], result))
        # load_name = 'backup_3/model_stage1_3share_prompt.pth'
        load_name = 'end_sysu/model_stage2_only_stage2_64.pth'
        model_path = os.path.join(r'D:/PretrainModel/CSDN/models/base/models', load_name)
        trained_model_state_dict = torch.load(model_path)
        model.model.module.prompt_learner.load_state_dict({k.replace('module.prompt_learner.', ''): v for k, v in trained_model_state_dict.items() if k.startswith('module.prompt_learner.')})
        print(f'Load the prompt_learner end,load_name: {load_name}')
        # save_name = f'stage1_part/{config.num_part}.pth'
        # model_file_path = os.path.join(model.save_model_path, save_name)
        # torch.save(model.model.state_dict(), model_file_path)
        # logger(f'The 1st Stage of Trained,asave_name: {save_name}')


        logger('Start the 3st Stage Training')
        logger('Extracting Text Features')

        num_classes = model.model.module.num_classes
        batch = config.batch_size
        i_ter = num_classes // batch
        left = num_classes - batch * (num_classes // batch)
        if left != 0:
            i_ter = i_ter + 1
        text_features = []
        with torch.no_grad():
            for i in range(i_ter):
                if i + 1 != i_ter:
                    l_list = torch.arange(i * batch, (i + 1) * batch)
                else:
                    l_list = torch.arange(i * batch, num_classes)
                # text_feature = model.model(label=l_list, get_fusion_text=True)
                text_feature = model.model(label=l_list, get_text=True)
                text_features.append(text_feature.cpu())
            text_features = torch.cat(text_features, 0).to(model.device)
        logger('Text Features Extracted, Start Training')

        model._init_optimizer_stage3()

        best_epoch=0
        part_text = model.model(get_part_text=True)
        for current_epoch in range(start_train_epoch, config.total_train_epoch):
            model.model_lr_scheduler_stage3.step(current_epoch)

            # _, result = train(model, loaders, text_features, config)
            _, result = train_2rgb(model, loaders, text_features, part_text, config)
            logger('Time: {}; Epoch: {}; LR, {}; {}'.format(time_now(), current_epoch,
                                                            model.model_lr_scheduler_stage3.get_lr()[0], result))

            if current_epoch + 1 >= 1 and (current_epoch + 1) % config.eval_epoch == 0:
                cmc, mAP, mINP = test(model, loaders, config)
                rank_1_10_20 = [cmc[0], cmc[9], cmc[19]]
                if cmc[0] + mAP > best_rank1 + best_mAP:
                    is_best_result = True
                    best_rank1 = cmc[0]
                    best_rank10 = cmc[9]
                    best_mAP = mAP
                    best_epoch = current_epoch
                else:
                    is_best_result = False
                # is_best_result = (cmc[0] >= best_rank1)
                # best_rank1 = max(cmc[0], best_rank1)
                model.save_model(current_epoch, is_best_result)
                logger('Time: {}; Test on Dataset: {}, \nmINP: {} \nmAP: {} \nRank_1_10_20: {}'.format(time_now(),
                                                                                            config.dataset,
                                                                                            mINP, mAP, rank_1_10_20))
                logger(f'now best result: [{best_rank1} {best_rank10}] {best_mAP} in epoch {best_epoch}\n')

    elif config.mode == 'test':
        loader = loaders.get_train_loader()
        # model.resume_model(config.resume_test_model)
        model_path = os.path.join(r'D:\PretrainModel\CSDN\models\testModel', 'model_101_v18.pth')
        model.model.load_state_dict(torch.load(model_path))
        cmc, mAP, mINP = test(model, loaders, config)
        rank_1_10_20 = [cmc[0], cmc[9], cmc[19]]
        logger('Time: {}; Test on Dataset: {}, \nmINP: {} \nmAP: {} \nRank_1_10_20: {}'.format(time_now(),
                                                                                       config.dataset,
                                                                                       mINP, mAP, rank_1_10_20))

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=str, default='cuda')
    parser.add_argument('--mode', type=str, default='test', help='train, test')
    parser.add_argument('--test_mode', default='all', type=str, help='all or indoor')
    parser.add_argument('--gall_mode', default='single', type=str, help='single or multi')
    parser.add_argument('--regdb_test_mode', default='v-t', type=str, help='')
    parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
    # parser.add_argument('--sysu_data_path', type=str, default='E:/hhj/SYSU-MM01-PART/')
    parser.add_argument('--sysu_data_path', type=str, default='E:/hhj/SYSU-MM01/')
    parser.add_argument('--regdb_data_path', type=str, default='D:/Re-ID_Data/person/Infrared/RegDB/')
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
    parser.add_argument('--num_workers', default=0, type=int,
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
