import torch
from tools import MultiItemAverageMeter

def train_stage1_randomcolor(base, data_loader):
    base.set_train()
    meter = MultiItemAverageMeter()
    # iter_list = torch.randperm(num_image).to(base.device)
    for i, data in enumerate(data_loader):
        rgb_img, ir_img = data[0].to(base.device), data[1].to(base.device)
        rgb_target, ir_target = data[2].to(base.device).long(), data[3].to(base.device).long()
        with torch.no_grad():
            rgb_image_features = base.model(x1=rgb_img, get_image=True)
            ir_image_features = base.model(x2=ir_img, get_image=True)
        rgb_text_features = base.model(label=rgb_target, get_text=True)
        loss_i2t_rgb = base.con_creiteron(rgb_image_features, rgb_text_features, rgb_target, rgb_target)
        loss_i2t_ir = base.con_creiteron(ir_image_features, rgb_text_features, ir_target, ir_target)
        loss_i2t = loss_i2t_rgb + loss_i2t_ir

        loss_t2i_rgb = base.con_creiteron(rgb_text_features, rgb_image_features, rgb_target, rgb_target)
        loss_t2i_ir = base.con_creiteron(rgb_text_features, ir_image_features, ir_target, ir_target)
        loss_t2i = loss_t2i_rgb + loss_t2i_ir

        loss = loss_i2t + loss_t2i
        base.model_optimizer_stage1.zero_grad()
        loss.backward()
        base.model_optimizer_stage1.step()

        meter.update({'loss_i2t': loss_i2t.data,
                      'loss_t2i': loss_t2i.data,})
    return meter.get_val(), meter.get_str()

def train_stage1(base, num_image, i_ter, batch, visible_labels_list, visible_image_features_list,
                   infrared_image_features_list):
    base.set_train()
    meter = MultiItemAverageMeter()
    iter_list = torch.randperm(num_image).to(base.device)
    for i in range(i_ter):
        # print(f"this is the {i}/{i_ter} iteration")
        b_list = iter_list[i*batch: (i+1)*batch]
        rgb_target = visible_labels_list[b_list].long()
        # ir_target = visible_labels_list[b_list].long()
        rgb_image_features = visible_image_features_list[b_list]
        ir_image_features = infrared_image_features_list[b_list]
        rgb_text_features = base.model(label1=rgb_target, get_text=True)
        # image_features = torch.cat([rgb_image_features, ir_image_features], dim=0)
        # text_features = torch.cat([rgb_text_features, rgb_text_features], dim=0)
        # target = torch.cat([rgb_target, rgb_target], dim=0)
        # loss_i2t = base.con_creiteron(image_features, text_features, target, target)
        # loss_t2i = base.con_creiteron(text_features, image_features, target, target)

        rgb_loss_i2t = base.con_creiteron(rgb_image_features, rgb_text_features, rgb_target, rgb_target)
        ir_loss_i2t = base.con_creiteron(ir_image_features, rgb_text_features, rgb_target, rgb_target)
        loss_i2t = rgb_loss_i2t + ir_loss_i2t

        rgb_loss_t2i = base.con_creiteron(rgb_text_features, rgb_image_features, rgb_target, rgb_target)
        ir_loss_t2i = base.con_creiteron(rgb_text_features, ir_image_features, rgb_target, rgb_target)
        loss_t2i = rgb_loss_t2i + ir_loss_t2i

        loss = loss_i2t + loss_t2i
        base.model_optimizer_stage1.zero_grad()
        loss.backward()
        base.model_optimizer_stage1.step()

        meter.update({'loss_i2t': loss_i2t.data,
                      'loss_t2i': loss_t2i.data,})
        # if (i + 1) % 200 == 0:
        #     print(f'stage1: iter:[{i + 1}/{i_ter}] loss_i2t:{loss_i2t.data}  loss_t2i:{loss_t2i.data}')

    return meter.get_val(), meter.get_str()

def train_stage1_3share(base, num_image, i_ter, batch, visible_labels_list, visible_image_features_list, infrared_labels_list, infrared_image_features_list, is_common=False):
    base.set_train()
    meter = MultiItemAverageMeter()
    rgb_iter_list = torch.randperm(num_image).to(base.device)
    # ir_iter_list = torch.randperm(num_image).to(base.device)
    ir_iter_list = rgb_iter_list
    for i in range(i_ter):
        # print(f"this is the {i}/{i_ter} iteration")
        rgb_b_list = rgb_iter_list[i*batch: (i+1)*batch]
        ir_b_list = ir_iter_list[i*batch: (i+1)*batch]
        rgb_target = visible_labels_list[rgb_b_list].long()
        ir_target = infrared_labels_list[ir_b_list].long()
        rgb_image_features = visible_image_features_list[rgb_b_list]
        ir_image_features = infrared_image_features_list[ir_b_list]
        if is_common:
            image_features = torch.cat([rgb_image_features, ir_image_features], dim=0)
            target = torch.cat([rgb_target, ir_target], dim=0)
            text_features = base.model(label=target, get_text=True)
            loss_i2t = base.con_creiteron(image_features, text_features, target, target)
            loss_t2i = base.con_creiteron(text_features, image_features, target, target)
        else:
            rgb_text_features = base.model(label1=rgb_target, get_text=True)
            ir_text_features = base.model(label2=ir_target, get_text=True)
            rgb_loss_i2t = base.con_creiteron(rgb_image_features, rgb_text_features, rgb_target, rgb_target)
            ir_loss_i2t = base.con_creiteron(ir_image_features, ir_text_features, ir_target, ir_target)
            loss_i2t = rgb_loss_i2t + ir_loss_i2t

            rgb_loss_t2i = base.con_creiteron(rgb_text_features, rgb_image_features, rgb_target, rgb_target)
            ir_loss_t2i = base.con_creiteron(ir_text_features, ir_image_features, ir_target, ir_target)
            loss_t2i = rgb_loss_t2i + ir_loss_t2i

        loss = loss_i2t + loss_t2i
        base.model_optimizer_stage1.zero_grad()
        loss.backward()
        base.model_optimizer_stage1.step()

        meter.update({'loss_i2t': loss_i2t.data,
                      'loss_t2i': loss_t2i.data,})
        # if (i + 1) % 200 == 0:
        #     print(f'stage1: iter:[{i + 1}/{i_ter}] loss_i2t:{loss_i2t.data}  loss_t2i:{loss_t2i.data}')

    return meter.get_val(), meter.get_str()

def train_stage2(base, num_image, i_ter, batch, labels_list, image_features_list):
    base.set_train()
    meter = MultiItemAverageMeter()
    iter_list = torch.randperm(num_image).to(base.device)
    for i in range(i_ter):
        # print(f"this is the {i}/{i_ter} iteration")
        b_list = iter_list[i*batch: (i+1)*batch]
        target = labels_list[b_list].long()
        image_features = image_features_list[b_list]
        text_features = base.model(label=target, get_text=True)
        loss_i2t = base.con_creiteron(image_features, text_features, target, target)
        loss_t2i = base.con_creiteron(text_features, image_features, target, target)


        loss = loss_i2t + loss_t2i
        base.model_optimizer_stage1.zero_grad()
        loss.backward()
        base.model_optimizer_stage1.step()

        meter.update({'loss_i2t': loss_i2t.data,
                      'loss_t2i': loss_t2i.data,})
        # if (i + 1) % 200 == 0:
        #     print(f'stage1: iter:[{i + 1}/{i_ter}] loss_i2t:{loss_i2t.data}  loss_t2i:{loss_t2i.data}')

    return meter.get_val(), meter.get_str()

# 只有一种rgb图像
def train(base, loaders, text_features, config):

    base.set_train()
    meter = MultiItemAverageMeter()
    loader = loaders.get_train_loader()
    for iter, (input1_0, input2, label1, label2) in enumerate(loader):
        # print(f"now is {i}/{len(loader)} step")
        # if i == 10:
        #     break
        rgb_imgs1, rgb_pids = input1_0, label1
        ir_imgs, ir_pids = input2, label2
        rgb_imgs1, rgb_pids = rgb_imgs1.to(base.device), rgb_pids.to(base.device).long()
        ir_imgs, ir_pids = ir_imgs.to(base.device), ir_pids.to(base.device).long()

        rgb_imgs = rgb_imgs1
        pids = torch.cat([rgb_pids, ir_pids], dim=0)

        features, cls_score, pp = base.model(x1=rgb_imgs, x2=ir_imgs)

        n = features[1].shape[0] // 2
        rgb_attn_features = features[1][:n]
        ir_attn_features = features[1][n:]
        rgb_logits = rgb_attn_features @ text_features.t()
        ir_logits = ir_attn_features @ text_features.t()

        ide_loss = base.pid_creiteron(cls_score[0], pids)
        ide_loss_proj = base.pid_creiteron(cls_score[1], pids)
        triplet_loss = base.tri_creiteron(features[0].squeeze(), pids)
        triplet_loss_proj = base.tri_creiteron(features[1].squeeze(), pids)

        # loss_hcc_euc = base.criterion_hcc_euc(features[1], pids)
        # loss_hcc_kl = base.criterion_hcc_kl(cls_score[1], pids)


        rgb_i2t_ide_loss = base.pid_creiteron(rgb_logits, rgb_pids)
        ir_i2t_ide_loss = base.pid_creiteron(ir_logits, ir_pids)


        loss = ide_loss + ide_loss_proj + config.lambda1 * (triplet_loss + triplet_loss_proj) +  config.lambda2 * rgb_i2t_ide_loss + config.lambda3 * ir_i2t_ide_loss

        base.model_optimizer_stage3.zero_grad()
        loss.backward()
        base.model_optimizer_stage3.step()
        meter.update({'pid_loss': ide_loss.data,
                      'pid_loss_proj': ide_loss_proj.data,
                      'triplet_loss': triplet_loss.data,
                      'triplet_loss_proj': triplet_loss_proj.data,
                      'rgb_i2t_pid_loss': rgb_i2t_ide_loss.data,
                      'ir_i2t_pid_loss': ir_i2t_ide_loss.data,
                      # 'loss_hcc_euc': loss_hcc_euc.data,
                      #   'loss_hcc_kl': loss_hcc_kl.data,
                      # 'loss_pp_euc': loss_pp_euc.data,
                      })
        # if (iter + 1) % 200 == 0:
        #     print(f'Iteration: [{iter + 1}/{len(loader)}]  pid_loss: {ide_loss.data} pid_loss_proj: {ide_loss_proj.data}'\
        #           f' triplet_loss: {triplet_loss.data} triplet_loss_proj: {triplet_loss_proj.data} rgb_i2t_pid_loss: {rgb_i2t_ide_loss.data} ir_i2t_pid_loss: {ir_i2t_ide_loss.data}')
    return meter.get_val(), meter.get_str()

# 有两种rgb图像
def train_2rgb(base, loaders, text_features, config):

    base.set_train()
    meter = MultiItemAverageMeter()
    loader = loaders.get_train_loader()
    # for iter, (rgb_1, rgb_1_flip, rgb_2, rgb_2_flip, ir_1, ir_1_flip, rgb_pids, ir_pids) in enumerate(loader):
    for iter, (rgb_1, rgb_2, ir_1, rgb_pids, ir_pids) in enumerate(loader):
        # rgb_imgs1, rgb_imgs2, rgb_pids = input1_0, input1_1, label1
        # ir_imgs, ir_pids = input2, label2
        # rgb_1, rgb_1_flip, rgb_2, rgb_2_flip, ir_1, ir_1_flip = rgb_1.to(base.device), rgb_1_flip.to(base.device), rgb_2.to(base.device), rgb_2_flip.to(base.device), ir_1.to(base.device), ir_1_flip.to(base.device)
        rgb_1, rgb_2, ir_1 = rgb_1.to(base.device), rgb_2.to(base.device), ir_1.to(base.device)
        rgb_pids, ir_pids = rgb_pids.to(base.device).long(), ir_pids.to(base.device).long()

        rgb_imgs = torch.cat([rgb_1, rgb_2], dim=0)
        # rgb_imgs_flip = torch.cat([rgb_1_flip, rgb_2_flip], dim=0)
        pids = torch.cat([rgb_pids, rgb_pids, ir_pids], dim=0)

        # features, cls_score, part_features, cls_scores_part, attention_weight, attention_weight_flip = base.model(x1=rgb_imgs, x1_flip=rgb_imgs_flip, x2=ir_1, x2_flip=ir_1_flip, label=pids)
        features, cls_score, part_features, cls_scores_part, per_part_features = base.model(x1=rgb_imgs, x2=ir_1, label=pids)

        n = features[1].shape[0] // 3
        rgb_attn_features = features[1].narrow(0, 0, n)
        ir_attn_features = features[1].narrow(0, 2 * n, n)
        rgb_logits = rgb_attn_features @ text_features.t()
        ir_logits = ir_attn_features @ text_features.t()
        num_part = per_part_features.size(0)
        loss_ipc = 0
        for i in range(num_part):
            loss_ipc += base.IPC(per_part_features[i], rgb_pids)

        loss_ipc /= num_part

        loss_ipd = base.IPD(per_part_features, rgb_pids)

        ide_loss = base.pid_creiteron(cls_score[0], pids)
        ide_loss_proj = base.pid_creiteron(cls_score[1], pids)

        ide_loss_part = base.pid_creiteron(cls_scores_part, pids)

        triplet_loss = base.tri_creiteron(features[0].squeeze(), pids)
        triplet_loss_proj = base.tri_creiteron(features[1].squeeze(), pids)
        triplet_loss_part = base.tri_creiteron(part_features, pids)

        rgb_i2t_ide_loss = base.pid_creiteron(rgb_logits, rgb_pids)
        ir_i2t_ide_loss = base.pid_creiteron(ir_logits, ir_pids)

        # atten_loss = base.euclidean(attention_weight, attention_weight_flip_flip) # / (attention_weight.size(-1) * attention_weight.size(-2))

        # loss_kl = base.criterion_hcc_kl_3(cls_score[1], pids)
        # loss_kl_map = base.criterion_hcc_kl_3(cls_score[0], pids)
        # loss_kl_part = base.criterion_hcc_kl_3(cls_scores_part, pids)

        loss = ide_loss + ide_loss_proj + ide_loss_part + config.lambda1 * (triplet_loss + triplet_loss_proj + triplet_loss_part) + \
               config.lambda2 * rgb_i2t_ide_loss + config.lambda3 * ir_i2t_ide_loss + 0.15 * (loss_ipd + loss_ipc) #+ loss_kl + loss_kl_map + loss_kl_part

        base.model_optimizer_stage3.zero_grad()
        loss.backward()
        base.model_optimizer_stage3.step()
        meter.update({'pid_loss': ide_loss.data,
                      'pid_loss_proj': ide_loss_proj.data,
                      'pid_loss_part': ide_loss_part.data,
                      'triplet_loss': triplet_loss.data,
                      'triplet_loss_proj': triplet_loss_proj.data,
                      'triplet_loss_part': triplet_loss_part.data,
                      'rgb_i2t_pid_loss': rgb_i2t_ide_loss.data,
                      'ir_i2t_pid_loss': ir_i2t_ide_loss.data,
                      # 'loss_kl': loss_kl.data,
                      # 'loss_kl_map': loss_kl_map.data,
                      # 'loss_kl_part': loss_kl_part.data,
                      'loss_ipc': loss_ipc.data,
                      'loss_ipd': loss_ipd.data,
                      })
        # print(f"iter = {iter}")
        # if (iter + 1) % 200 == 0:
        #     print(f'Iteration [{iter + 1}/{len(loader)}] Loss: {meter.get_str()}\n')
        # if iter == 2:
        #     break
        # break
    return meter.get_val(), meter.get_str()






