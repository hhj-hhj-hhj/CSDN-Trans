import torch
from tools import MultiItemAverageMeter


def train_stage0(base, dataloader, rgb_mean_feature, ir_mean_feature, pid_center):
    base.set_train()
    meter = MultiItemAverageMeter()

    for iter, data in enumerate(dataloader):
        rgb_imgs, ir_imgs, label1, label2, shape_maps_rgb, shape_maps_ir = data
        rgb_imgs, ir_imgs = rgb_imgs.to(base.device), ir_imgs.to(base.device)
        label1, label2 = label1.to(base.device).long(), label2.to(base.device).long()
        shape_maps_rgb, shape_maps_ir = shape_maps_rgb.to(base.device), shape_maps_ir.to(base.device)

        # imgs = torch.cat([rgb_imgs, ir_imgs], dim=0)
        #
        # target_i = torch.tensor([0]).to(base.device).long()
        # target_i = target_i.repeat(2 * rgb_imgs.size(0))
        # shape_imgs = torch.cat([shape_maps_rgb, shape_maps_ir], dim=0)

        with torch.no_grad():
            # image_maps = base.model(x1=imgs, get_map=True)
            # shape_maps = base.model(x1=shape_imgs, get_map=True)
            rgb_img_maps = base.model(x1=rgb_imgs, get_map=True)
            ir_img_maps = base.model(x2=ir_imgs, get_map=True)
            rgb_shape_maps = base.model(x1=shape_maps_rgb, get_map=True)
            ir_shape_maps = base.model(x2=shape_maps_ir, get_map=True)

        rgb_fusion_map = base.model(img_map=rgb_img_maps, shape_map=rgb_shape_maps, get_atten=True)
        ir_fusion_map = base.model(img_map=ir_img_maps, shape_map=ir_shape_maps, get_atten=True)

        base.model.module.attnpool.requires_grad = False
        rgb_image_features = base.model(fusion_map=rgb_fusion_map, maps2feature=True)
        ir_image_features = base.model(fusion_map=ir_fusion_map, maps2feature=True)

        loss_rgb_center_ce = base.criterion_center_ce(rgb_image_features, rgb_mean_feature, label1)
        loss_ir_center_ce = base.criterion_center_ce(ir_image_features, ir_mean_feature, label2)

        loss_center_ce = loss_rgb_center_ce + loss_ir_center_ce

        base.model_optimizer_stage2.zero_grad()
        loss_center_ce.backward()
        base.model_optimizer_stage2.step()

        meter.update({'loss_rgb_center': loss_rgb_center_ce.data,
                      'loss_ir_center': loss_ir_center_ce.data,})
        # if (iter + 1) % 200 == 0:
        #     print(f"Iteration: [{iter + 1}/{len(dataloader)}] loss_rgb_center: {loss_rgb_center_ce.data:.4f} loss_ir_center: {loss_ir_center_ce.data:.4f}")

    return meter.get_val(), meter.get_str()


def train_stage1_randomcolor(base, data_loader):
    base.set_train()
    meter = MultiItemAverageMeter()
    # iter_list = torch.randperm(num_image).to(base.device)
    for iter, data in enumerate(data_loader):
        rgb_imgs, ir_imgs, label1, label2, shape_maps_rgb, shape_maps_ir = data
        rgb_imgs, ir_imgs = rgb_imgs.to(base.device), ir_imgs.to(base.device)
        label1, label2 = label1.to(base.device).long(), label2.to(base.device).long()
        shape_maps_rgb, shape_maps_ir = shape_maps_rgb.to(base.device), shape_maps_ir.to(base.device)

        with torch.no_grad():
            rgb_img_maps = base.model(x1=rgb_imgs, get_map=True)
            ir_img_maps = base.model(x2=ir_imgs, get_map=True)
            rgb_shape_maps = base.model(x1=shape_maps_rgb, get_map=True)
            ir_shape_maps = base.model(x2=shape_maps_ir, get_map=True)
            rgb_fusion_map = base.model(img_map=rgb_img_maps, shape_map=rgb_shape_maps, get_atten=True)
            ir_fusion_map = base.model(img_map=ir_img_maps, shape_map=ir_shape_maps, get_atten=True)
            rgb_image_features = base.model(fusion_map=rgb_fusion_map, maps2feature=True)
            ir_image_features = base.model(fusion_map=ir_fusion_map, maps2feature=True)

        rgb_text_features = base.model(label=label1, get_text=True)
        ir_text_features = base.model(label=label2, get_text=True)

        loss_i2t_rgb = base.con_creiteron(rgb_image_features, rgb_text_features, label1, label1)
        loss_i2t_ir = base.con_creiteron(ir_image_features, ir_text_features, label2, label2)
        loss_i2t = loss_i2t_rgb + loss_i2t_ir

        loss_t2i_rgb = base.con_creiteron(rgb_text_features, rgb_image_features, label1, label1)
        loss_t2i_ir = base.con_creiteron(rgb_text_features, ir_image_features, label2, label2)
        loss_t2i = loss_t2i_rgb + loss_t2i_ir

        loss = loss_i2t + loss_t2i
        base.model_optimizer_stage1.zero_grad()
        loss.backward()
        base.model_optimizer_stage1.step()

        meter.update({'loss_i2t': loss_i2t.data,
                      'loss_t2i': loss_t2i.data, })
        # if (iter + 1) % 200 == 0:
        #     print(f"Iteration: [{iter + 1}/{len(data_loader)}] loss_i2t: {loss_i2t.data} loss_t2i: {loss_t2i.data}")

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
        if (iter + 1) % 200 == 0:
            print(f'Iteration: [{iter + 1}/{len(loader)}]  {meter.get_str()}')
    return meter.get_val(), meter.get_str()

def train_2rgb(base, loaders, text_features, config):

    base.set_train()
    meter = MultiItemAverageMeter()
    loader = loaders.get_train_loader()
    for iter, (input1_0, input1_1, input2, label1, label2) in enumerate(loader):
        rgb_imgs1, rgb_imgs2, rgb_pids = input1_0, input1_1, label1
        ir_imgs, ir_pids = input2, label2
        rgb_imgs1, rgb_imgs2, rgb_pids = rgb_imgs1.to(base.device),  rgb_imgs2.to(base.device), \
                                        rgb_pids.to(base.device).long()
        ir_imgs, ir_pids = ir_imgs.to(base.device), ir_pids.to(base.device).long()

        rgb_imgs = torch.cat([rgb_imgs1, rgb_imgs2], dim=0)
        pids = torch.cat([rgb_pids, rgb_pids, ir_pids], dim=0)

        features, cls_score, pp = base.model(x1=rgb_imgs, x2=ir_imgs)

        n = features[1].shape[0] // 3
        rgb_attn_features = features[1].narrow(0, 0, n)
        ir_attn_features = features[1].narrow(0, 2 * n, n)
        rgb_logits = rgb_attn_features @ text_features.t()
        ir_logits = ir_attn_features @ text_features.t()

        ide_loss = base.pid_creiteron(cls_score[0], pids)
        ide_loss_proj = base.pid_creiteron(cls_score[1], pids)
        triplet_loss = base.tri_creiteron(features[0].squeeze(), pids)
        triplet_loss_proj = base.tri_creiteron(features[1].squeeze(), pids)

        rgb_i2t_ide_loss = base.pid_creiteron(rgb_logits, rgb_pids)
        ir_i2t_ide_loss = base.pid_creiteron(ir_logits, ir_pids)

        loss_hcc_kl = base.criterion_hcc_kl_3(cls_score[1], pids)
        loss_hcc_kl_map = base.criterion_hcc_kl_3(cls_score[0], pids)
        # loss_pp_euc = 0
        # for i in range(pp.size(1)):
        #     loss_pp_euc += base.criterion_pp_3(pp[:,i], pids) / pp.size(1)

        loss = ide_loss + ide_loss_proj + config.lambda1 * (triplet_loss + triplet_loss_proj) + \
               config.lambda2 * rgb_i2t_ide_loss + config.lambda3 * ir_i2t_ide_loss + loss_hcc_kl + loss_hcc_kl_map # + loss_pp_euc * 0.05

        base.model_optimizer_stage3.zero_grad()
        loss.backward()
        base.model_optimizer_stage3.step()
        meter.update({'pid_loss': ide_loss.data,
                      'pid_loss_proj': ide_loss_proj.data,
                      'triplet_loss': triplet_loss.data,
                      'triplet_loss_proj': triplet_loss_proj.data,
                      'rgb_i2t_pid_loss': rgb_i2t_ide_loss.data,
                      'ir_i2t_pid_loss': ir_i2t_ide_loss.data,
                      'loss_hcc_kl': loss_hcc_kl.data,
                      'loss_hcc_kl_map': loss_hcc_kl_map.data,
                      # 'loss_pp_euc': loss_pp_euc,
                      })
        # print(f"iter = {iter}")
        # if (iter + 1) % 200 == 0:
        #     print(f'Iteration [{iter + 1}/{len(loader)}] Loss: {meter.get_str()}')
        # if iter == 3:
        #     break
        # break
    return meter.get_val(), meter.get_str()

