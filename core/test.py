
import numpy as np
import torch
from tools import eval_regdb, eval_sysu
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 添加这行代码以便更好地调试 CUDA 错误

def test(base, loader, config):
    base.set_eval()
    print('Extracting Query Feature...')
    ptr = 0
    # query_feat = np.zeros((loader.n_query, 3072))
    # query_feat = np.zeros((loader.n_query, 5120))
    query_feat = np.zeros((loader.n_query, 4096))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(loader.query_loader):
            batch_num = input.size(0)
            input = input.to(base.device)
            feat = base.model(x2=input)

            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num

    print('Extracting Gallery Feature...')

    if loader.dataset == 'sysu':
        all_cmc = 0
        all_mAP = 0
        all_mINP = 0
        for i in range(10):
            ptr = 0
            gall_loader = loader.gallery_loaders[i]
            # gall_feat = np.zeros((loader.n_gallery, 3072))
            # gall_feat = np.zeros((loader.n_gallery, 5120))
            gall_feat = np.zeros((loader.n_gallery, 4096))
            with torch.no_grad():
                for batch_idx, (input, label) in enumerate(gall_loader):
                    batch_num = input.size(0)
                    input = input.to(base.device)
                    feat = base.model(x1=input)

                    gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                    ptr = ptr + batch_num
            distmat = np.matmul(query_feat, np.transpose(gall_feat))
            cmc, mAP, mINP = eval_sysu(-distmat, loader.query_label, loader.gall_label, loader.query_cam,
                                       loader.gall_cam)
            all_cmc += cmc
            all_mAP += mAP
            all_mINP += mINP
        all_cmc /= 10.0
        all_mAP /= 10.0
        all_mINP /= 10.0

    elif loader.dataset == 'regdb':
        gall_loader = loader.gallery_loaders
        gall_feat = np.zeros((loader.n_gallery, 3072))
        ptr = 0
        with torch.no_grad():
            for batch_idx, (input, label) in enumerate(gall_loader):
                batch_num = input.size(0)
                input = input.to(base.device)
                feat = base.model(x1=input)

                gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()

                ptr = ptr + batch_num
        if config.regdb_test_mode == 't-v':
            distmat = np.matmul(query_feat, np.transpose(gall_feat))
            cmc, mAP, mINP = eval_regdb(-distmat, loader.query_label, loader.gall_label)
        else:
            distmat = np.matmul(gall_feat, np.transpose(query_feat))
            cmc, mAP, mINP = eval_regdb(-distmat, loader.gall_label, loader.query_label)

        all_cmc, all_mAP, all_mINP = cmc, mAP, mINP


    return all_cmc, all_mAP, all_mINP