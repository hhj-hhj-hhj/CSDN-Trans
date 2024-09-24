import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from visual.visual_base import Base
from PIL import Image
# from torchvision.models import resnet50
# model = resnet50(pretrained=True)


import argparse
import random
import os
import ast
# 使用预训练模型
def main(config):
    base = Base(config)
    model_trans = base.model
    # model_trans.eval()

    image_path = r"E:\hhj\SYSU-MM01\cam2\0001\0006.jpg"
    image = Image.open(image_path)

    # model_path = r'D:/PretrainModel/CSDN/models/testModel/model_105_V1_trans.pth'
    # model_path = r'D:/PretrainModel/CSDN/models/testModel/model_91_only3.pth'
    # model_path = r'D:/PretrainModel/CSDN/models/testModel/model_106.pth'
    # model_path = r'D:/PretrainModel/CSDN/models/testModel/model_86_one_prompt.pth'

    # model_trans.load_state_dict(torch.load(model_path), strict=False)

    import torchvision.transforms as transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_test_rgb = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((config.img_h, config.img_w)),
        transforms.ToTensor(),
        normalize])

    img_tensor = transform_test_rgb(image.copy())
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(base.device)

    input_tensor = img_tensor

    # input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    target_layers = [model_trans.module.image_attention_fusion]
    cam = GradCAM(model=model_trans, target_layers=target_layers)

    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]

    img = np.array(image)
    img = cv2.resize(img, (config.img_w, config.img_h))

    visualization = show_cam_on_image(img.astype(dtype=np.float32)/255.0, grayscale_cam, use_rgb=True)

    plt.imshow(visualization)
    plt.axis('off')
    plt.show()



if __name__ == '__main__':
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=str, default='cuda')
    parser.add_argument('--mode', type=str, default='train', help='train, test')
    parser.add_argument('--test_mode', default='all', type=str, help='all or indoor')
    parser.add_argument('--gall_mode', default='single', type=str, help='single or multi')
    parser.add_argument('--regdb_test_mode', default='v-t', type=str, help='')
    parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
    # parser.add_argument('--sysu_data_path', type=str, default='E:/hhj/SYSU-MM01-PART/')
    parser.add_argument('--sysu_data_path', type=str, default='E:/hhj/SYSU-MM01/')
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