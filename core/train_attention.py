import torch
from tools import MultiItemAverageMeter


def train_stage0(base, dataloader, text_features):
    base.set_train()
    meter = MultiItemAverageMeter()

    for i, data in enumerate(dataloader):
        # print(f"this is the {i}/{len(dataloader)} iteration")
        rgb_imgs, ir_imgs, label1, label2, shape_maps_rgb, shape_maps_ir = data
        rgb_imgs, ir_imgs = rgb_imgs.to(base.device), ir_imgs.to(base.device)
        # label1, label2 = label1.to(base.device).long(), label2.to(base.device).long()
        shape_maps_rgb, shape_maps_ir = shape_maps_rgb.to(base.device), shape_maps_ir.to(base.device)

        imgs = torch.cat([rgb_imgs, ir_imgs], dim=0)

        # target_t_p = torch.tensor([1]).to(base.device).long()
        # target_t_n = torch.tensor([2]).to(base.device).long()
        # target_t_p = target_t_p.repeat(rgb_imgs.size(0))
        # target_t_n = target_t_n.repeat(rgb_imgs.size(0))
        # target_t = torch.cat([target_t_p, target_t_n], dim=0)

        target_i = torch.tensor([0]).to(base.device).long()
        target_i = target_i.repeat(2 * rgb_imgs.size(0))
        shape_imgs = torch.cat([shape_maps_rgb, shape_maps_ir], dim=0)

        with torch.no_grad():
            image_maps = base.model(x1=imgs, get_map=True)
            shape_maps = base.model(x1=shape_imgs, get_map=True)

        fusion_map = base.model(img_map=image_maps, shape_map=shape_maps, get_atten=True)

        base.model.module.attnpool.requires_grad = False
        image_features = base.model(fusion_map=fusion_map, maps2feature=True)

        logit = image_features @ text_features.t()
        loss_i2t = base.pid_creiteron(logit, target_i)
        # loss_t2i = base.con_creiteron(text_features, image_features, target_t, target_i)
        loss_t2i = loss_i2t

        if torch.isnan(loss_i2t).any():
            print("loss_i2t 中检测到 NaN")
        if torch.isnan(loss_t2i).any():
            print("loss_t2i 中检测到 NaN")

        loss = loss_i2t + loss_t2i
        base.model_optimizer_stage2.zero_grad()
        loss.backward()
        base.model_optimizer_stage2.step()

        meter.update({'loss_i2t': loss_i2t.data,
                      'loss_t2i': loss_t2i.data, })
        if i % 10 == 0:
            print(f"[{i}/{len(dataloader)}] {meter.get_str()}")

    return meter.get_val(), meter.get_str()


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
        rgb_text_features = base.model(label1=rgb_target, get_text=True)
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
                      'loss_t2i': loss_t2i.data, })

    return meter.get_val(), meter.get_str()


def train_stage1(base, num_image, i_ter, batch, visible_labels_list, visible_image_features_list,
                 infrared_image_features_list):
    base.set_train()
    meter = MultiItemAverageMeter()
    iter_list = torch.randperm(num_image).to(base.device)
    for i in range(i_ter):
        # print(f"this is the {i}/{i_ter} iteration")
        b_list = iter_list[i * batch: (i + 1) * batch]
        rgb_target = visible_labels_list[b_list].long()
        # ir_target = visible_labels_list[b_list].long()
        rgb_image_features = visible_image_features_list[b_list]
        ir_image_features = infrared_image_features_list[b_list]
        rgb_text_features = base.model(label1=rgb_target, get_text=True)
        # ir_text_features = base.model(label2=ir_target, get_text=True)
        image_features = torch.cat([rgb_image_features, ir_image_features], dim=0)
        text_features = torch.cat([rgb_text_features, rgb_text_features], dim=0)
        # text_features = rgb_text_features
        target = torch.cat([rgb_target, rgb_target], dim=0)
        loss_i2t = base.con_creiteron(image_features, text_features, target, target)
        loss_t2i = base.con_creiteron(text_features, image_features, target, target)

        loss = loss_i2t + loss_t2i
        base.model_optimizer_stage1.zero_grad()
        loss.backward()
        base.model_optimizer_stage1.step()

        meter.update({'loss_i2t': loss_i2t.data,
                      'loss_t2i': loss_t2i.data, })

    return meter.get_val(), meter.get_str()


def train(base, loaders, text_features, config):
    base.set_train()
    meter = MultiItemAverageMeter()
    loader = loaders.get_train_loader()
    for i, (input1_0, input2, label1, label2) in enumerate(loader):
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
        loss_hcc_kl = base.criterion_hcc_kl(cls_score[1], pids)
        loss_pp_euc = 0
        for i in range(pp.size(1)):
            loss_pp_euc += base.criterion_pp(pp[:, i], pids) / pp.size(1)

        rgb_i2t_ide_loss = base.pid_creiteron(rgb_logits, rgb_pids)
        ir_i2t_ide_loss = base.pid_creiteron(ir_logits, ir_pids)

        # loss = ide_loss + ide_loss_proj + config.lambda1 * (triplet_loss + triplet_loss_proj) + \
        #        config.lambda2 * rgb_i2t_ide_loss + config.lambda3 * ir_i2t_ide_loss + (loss_hcc_euc + loss_hcc_kl) + loss_pp_euc * 0.15

        # loss = ide_loss + ide_loss_proj + config.lambda1 * (triplet_loss + triplet_loss_proj) + \
        #        config.lambda2 * rgb_i2t_ide_loss + config.lambda3 * ir_i2t_ide_loss + loss_pp_euc * 0.05 + (loss_hcc_euc + loss_hcc_kl)

        loss = ide_loss + ide_loss_proj + config.lambda1 * (triplet_loss + triplet_loss_proj) + \
               config.lambda2 * rgb_i2t_ide_loss + config.lambda3 * ir_i2t_ide_loss + loss_pp_euc * 0.05 + loss_hcc_kl

        # loss = ide_loss + ide_loss_proj + config.lambda1 * (triplet_loss + triplet_loss_proj) + \
        #        0.075 * rgb_i2t_ide_loss + 0.15 * ir_i2t_ide_loss + loss_pp_euc * 0.05 + (
        #                    loss_hcc_euc + loss_hcc_kl)

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
                      'loss_hcc_kl': loss_hcc_kl.data,
                      'loss_pp_euc': loss_pp_euc.data,
                      })
    return meter.get_val(), meter.get_str()







