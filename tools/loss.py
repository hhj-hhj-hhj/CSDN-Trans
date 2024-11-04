import torch.nn as nn
import torch


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = targets.long()
        size = log_probs.size()
        targets = torch.zeros((size[0], size[1])).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)

        if self.use_gpu:
            targets = targets.to(torch.device('cuda'))
        targets = (1 - self.epsilon) * targets + self.epsilon / size[1]
        loss = (-targets * log_probs).mean(0).sum()
        return loss

class SupConLoss(nn.Module):
    def __init__(self, device):
        super(SupConLoss, self).__init__()
        self.device = device
        self.temperature = 1.0
    def forward(self, text_features, image_features, t_label, i_targets):
        batch_size = text_features.shape[0]
        batch_size_N = image_features.shape[0]
        mask = torch.eq(t_label.unsqueeze(1).expand(batch_size, batch_size_N), \
            i_targets.unsqueeze(0).expand(batch_size,batch_size_N)).float().to(self.device)

        logits = torch.div(torch.matmul(text_features, image_features.T),self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - mean_log_prob_pos.mean()

        return loss

def normalize(x, axis=-1):
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def pdist_torch(emb1, emb2):
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim=1, keepdim=True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    dist_mtx = dist_mtx.clamp(min=1e-12).sqrt()
    return dist_mtx

def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W

class TripletLoss_WRT(nn.Module):

    def __init__(self):
        super(TripletLoss_WRT, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)

        return loss



def compute_dist_euc(x1,x2,p1,p2):
    m, n = x1.shape[0], x2.shape[0]
    dist = torch.pow(x1, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(x2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(x1, x2.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()
    mask = p1.expand(n, m).t().eq(p2.expand(m, n))
    return dist, mask

def compute_dist_kl(x1,x2,p1,p2):
    n1, c = x1.shape
    n2, c = x2.shape
    pr1 = x1.expand(n2, n1, c).detach()
    pr2 = x2.expand(n1, n2, c).detach()
    x1 = x1.clamp(1e-9).log().expand(n2, n1, c)
    x2 = x2.clamp(1e-9).log().expand(n1, n2, c)
    dist = (pr2 * x2.detach() - pr2 * x1.permute(1, 0, 2)).sum(dim=2) + \
           (pr1 * x1.detach() - pr1 * x2.permute(1, 0, 2)).sum(dim=2).t()
    mask = p1.expand(n2, n1).t().eq(p2.expand(n1, n2))
    return dist, mask

class hcc_euc(nn.Module):
    def __init__(self, margin_euc=0.6):
        super(hcc_euc, self).__init__()
        self.margin_euc = margin_euc

    def forward(self, x, pids):
        margin = self.margin_euc

        p = len(pids.unique())
        c = x.shape[-1]
        pidhc = pids.reshape(2*p, -1)[:, 0]# pid编号
        hcen = x.reshape(2*p, -1, c).mean(dim=1)# 每个pid对应的中心，C维

        dist, mask = compute_dist_euc(x, hcen, pids, pidhc)
        loss = []
        n, m = dist.shape
        for i in range(n // 2):
            loss.append(dist[i][m // 2:][mask[i][m // 2:]])
        for i in range(n // 2, n):
            loss.append(dist[i][:m // 2][mask[i][:m // 2]])
        loss1 = torch.cat(loss).mean()
        dist, mask = compute_dist_euc(x, hcen, pids, pidhc)
        loss = []
        n, m = dist.shape
        for i in range(n):
            loss.append((margin - dist[i][mask[i] == 0]).clamp(0))
        loss2 = torch.cat(loss).mean()
        return loss1 + loss2

class hcc_kl(nn.Module):
    def __init__(self, margin_kl=6):
        super(hcc_kl, self).__init__()
        self.margin_kl = margin_kl

    def forward(self, x, pids):
        margin = self.margin_kl
        x = x.softmax(dim=-1)

        p = len(pids.unique())
        c = x.shape[-1]
        pidhc = pids.reshape(2*p, -1)[:, 0]# pid编号
        hcen = x.reshape(2*p, -1, c).mean(dim=1)# 每个pid对应的中心，C维

        dist, mask = compute_dist_kl(x, hcen, pids, pidhc)
        loss = []
        n, m = dist.shape
        for i in range(n // 2):
            loss.append(dist[i][m // 2:][mask[i][m // 2:]])
        for i in range(n // 2, n):
            loss.append(dist[i][:m // 2][mask[i][:m // 2]])
        loss1 = torch.cat(loss).mean()
        dist, mask = compute_dist_kl(x, hcen, pids, pidhc)
        loss = []
        n, m = dist.shape
        for i in range(n):
            loss.append((margin - dist[i][mask[i] == 0]).clamp(0))
        loss2 = torch.cat(loss).mean()
        return loss1 + loss2

class hcc_kl_3(nn.Module):
    def __init__(self, margin_kl=6.0, k1=1.0, k2=1.0):
        super(hcc_kl_3, self).__init__()
        self.margin_kl = margin_kl
        self.k1 = k1
        self.k2 = k2

    def forward(self, x, pids):
        margin = self.margin_kl
        x = x.softmax(dim=-1)

        p = len(pids.unique())
        c = x.shape[-1]
        pidhc = pids.reshape(3*p, -1)[:, 0]# pid编号
        hcen = x.reshape(3*p, -1, c).mean(dim=1)# 每个pid对应的中心，C维

        dist, mask = compute_dist_kl(x, hcen, pids, pidhc)
        n, m = dist.shape
        mid_n = n // 3 * 2
        mid_m = m // 3 * 2
        # loss = []
        # for i in range(mid_n):
        #     loss.append(dist[i][mid_m:][mask[i][mid_m:]])
        # for i in range(mid_n, n):
        #     loss.append(dist[i][:mid_m][mask[i][:mid_m]])
        # loss1_2 = torch.cat(loss).mean()
        loss1 = torch.cat([dist[:mid_n, mid_m:][mask[:mid_n, mid_m:]], dist[mid_n:, :mid_m][mask[mid_n:, :mid_m]]]).mean()
        # loss = []
        # n, m = dist.shape
        # for i in range(n):
        #     loss.append((margin - dist[i][mask[i] == 0]).clamp(0))
        # loss2_2 = torch.cat(loss).mean()
        loss2 = (margin - dist[mask == 0]).clamp(0).mean()
        loss_all = self.k1 * loss1 + self.k2 * loss2
        return loss_all

class ptcc(nn.Module):
    def __init__(self, margin_euc=0.3):
        super(ptcc, self).__init__()
        self.margin_euc = margin_euc

    def forward(self, x, pids):

        p = len(pids.unique())
        c = x.shape[-1]
        pidhc = pids.reshape(2*p, -1)[:, 0]# pid编号
        hcen = x.reshape(2*p, -1, c).mean(dim=1)# 每个pid对应的中心，C维

        dist, mask = compute_dist_euc(x, hcen, pids, pidhc)
        loss = []
        n, m = dist.shape
        for i in range(n // 2):
            loss.append(dist[i][m // 2:][mask[i][m // 2:]])
        for i in range(n // 2, n):
            loss.append(dist[i][:m // 2][mask[i][:m // 2]])
        loss = torch.cat(loss).mean()
        return loss

class ptcc_3(nn.Module):
    def __init__(self, margin_euc=0.3):
        super(ptcc_3, self).__init__()
        self.margin_euc = margin_euc

    def forward(self, x, pids):

        p = len(pids.unique())
        c = x.shape[-1]
        pidhc = pids.reshape(3*p, -1)[:, 0]# pid编号
        hcen = x.reshape(3*p, -1, c).mean(dim=1)# 每个pid对应的中心，C维

        dist, mask = compute_dist_euc(x, hcen, pids, pidhc)
        loss = []
        n, m = dist.shape
        mid_n = n // 3 * 2
        mid_m = m // 3 * 2
        for i in range(mid_n):
            loss.append(dist[i][mid_m:][mask[i][mid_m:]])
        for i in range(mid_n, n):
            loss.append(dist[i][:mid_m][mask[i][:mid_m]])
        loss = torch.cat(loss).mean()
        return loss

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=1.0):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, f_v, f_i, labels):
        """
        Args:
            f_v: 可见光图像特征，形状为 (B, C)，B 是批量大小
            f_i: 红外图像特征，形状为 (B, C)
            labels: 图像的标签，形状为 (B,)
        """
        B, D = f_v.shape
        # 计算相似度矩阵
        sim_v2i = torch.matmul(f_v, f_i.T) / self.temperature
        sim_v2v = torch.matmul(f_v, f_v.T) / self.temperature
        sim_i2i = torch.matmul(f_i, f_i.T) / self.temperature

        # 数值稳定性处理
        sim_v2i = sim_v2i - torch.max(sim_v2i, dim=1, keepdim=True)[0].detach()
        sim_v2v = sim_v2v - torch.max(sim_v2v, dim=1, keepdim=True)[0].detach()
        sim_i2i = sim_i2i - torch.max(sim_i2i, dim=1, keepdim=True)[0].detach()

        # 指数距离
        exp_sim_v2i = torch.exp(sim_v2i)
        exp_sim_v2v = torch.exp(sim_v2v)
        exp_sim_i2v = torch.exp(sim_v2i.T)  # 转置来对应公式中的 i2v
        exp_sim_i2i = torch.exp(sim_i2i)

        p = len(labels.unique())
        num_perid = B // p
        mask = torch.zeros((B, B), device=f_v.device, dtype=torch.bool)

        loss_v2i = torch.zeros(B, device=f_v.device, dtype=torch.float32)
        loss_i2v = torch.zeros(B, device=f_v.device, dtype=torch.float32)

        for i in range(p):
            st = i * num_perid
            ed = st + num_perid
            mask[st:ed, st:ed] = 1  # 构造 mask，表示正样本对的位置
            # 构造 mask，表示正样本对的位置
            mask_v2i = mask[st:ed, :]
            mask_i2v = mask.T[st:ed, :]
            mask_v2v = torch.logical_not(mask_v2i)
            mask_i2i = torch.logical_not(mask_i2v)

            # 计算 Lv2ice
            lv2ice_numerator = (mask_v2i * exp_sim_v2i[st:ed]).sum(dim=1)
            lv2ice_denominator = (mask_v2v * exp_sim_v2v[st:ed]).sum(dim=1) + (mask_v2i * exp_sim_v2i[st:ed]).sum(dim=1)
            lv2ice = -torch.log(lv2ice_numerator / lv2ice_denominator)
            lv2ice = lv2ice / mask_v2i.sum(1)
            loss_v2i[st:ed] += lv2ice

            # 计算 Li2vce
            li2vce_numerator = (mask_i2v * exp_sim_i2v[st:ed]).sum(dim=1)
            li2vce_denominator = (mask_i2i * exp_sim_i2i[st:ed]).sum(dim=1) + (mask_i2v * exp_sim_i2v[st:ed]).sum(dim=1)
            li2vce = -torch.log(li2vce_numerator / li2vce_denominator)
            li2vce = li2vce / mask_i2v.sum(1)
            loss_i2v[st:ed] += li2vce

            mask[st:ed, st:ed] = 0  # 恢复 mask

        # 总损失
        loss_v2i = loss_v2i.mean()
        loss_i2v = loss_i2v.mean()
        loss = loss_v2i + loss_i2v
        return loss

