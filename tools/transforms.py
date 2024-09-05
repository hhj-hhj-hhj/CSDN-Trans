import torch
import numpy as np
import math

class RandomColoring_tensor(object):
    def __init__(self, p=0.5, sl=0.02, sh=0.4, r1=0.3, r2=3.33, is_rgb=True):
        self.p = p
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.r2 = r2
        self.is_rgb = is_rgb

    def __call__(self, img):
        is_rgb = self.is_rgb
        if torch.rand(1) >= self.p: return img
        # cvt = RGB_HSV()
        # img = cvt.rgb_to_hsv(img.unsqueeze(0))[0]
        for attempt in range(5):
            area = img.size()[1] * img.size()[2]
            area = torch.tensor(area, device=img.device)
            target_area = (self.sl + torch.rand(1,device=img.device) * (self.sh - self.sl))* area
            # target_area = target_area.to(img.device)
            aspect_ratio = self.r1 + torch.rand(1,device=img.device)  * (self.r2 - self.r1)
            aspect_ratio = torch.tensor(aspect_ratio, device=img.device)
            # aspect_ratio = aspect_ratio.to(img.device)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < img.size()[2] and h < img.size()[1]:
                x1 = torch.randint(0, img.size()[1] - h, (1,),device=img.device)
                y1 = torch.randint(0, img.size()[2] - w, (1,),device=img.device)
                if is_rgb:
                    img[0, x1:x1 + h, y1:y1 + w] = torch.rand(1,device=img.device)
                    img[1, x1:x1 + h, y1:y1 + w] = img[1, x1:x1 + h, y1:y1 + w] * 0.5 + 0.5 * torch.rand(1,device=img.device)
                    img[2, x1:x1 + h, y1:y1 + w] = img[2, x1:x1 + h, y1:y1 + w] * 0.9 + 0.1 * (
                                1 + torch.rand(1,device=img.device)  * (1. / img[2, x1:x1 + h, y1:y1 + w].max() - 1))
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = torch.rand(1,device=img.device)
                    img[1, x1:x1 + h, y1:y1 + w] = img[1, x1:x1 + h, y1:y1 + w] * 0.5 + 0.5 * torch.rand(1,device=img.device)
                    img[2, x1:x1 + h, y1:y1 + w] = img[2, x1:x1 + h, y1:y1 + w] * 0.5 + 0.5 * torch.rand(1,device=img.device)
        # img = cvt.hsv_to_rgb(img.unsqueeze(0))[0]
        return img

class RandomColoring(object):
    def __init__(self, p=0.5, sl=0.02, sh=0.4, r1=0.3, r2=3.33, is_rgb=True):
        self.p = p
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.r2 = r2
        self.is_rgb = is_rgb
    def __call__(self, img):
        is_rgb = self.is_rgb
        if np.random.uniform(0, 1) >= self.p:return img
        cvt = RGB_HSV()
        img = cvt.rgb_to_hsv(img.unsqueeze(0))[0]
        for attempt in range(5):
            area = img.size()[1] * img.size()[2]
            target_area = np.random.uniform(self.sl, self.sh) * area
            #用np.random和测试的random.random分开
            aspect_ratio = np.random.uniform(self.r1, self.r2)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < img.size()[2] and h < img.size()[1]:
                x1 = np.random.randint(0, img.size()[1] - h)
                y1 = np.random.randint(0, img.size()[2] - w)
                if is_rgb:
                    img[0, x1:x1 + h, y1:y1 + w] = np.random.uniform(0, 1)
                    img[1, x1:x1 + h, y1:y1 + w] = img[1, x1:x1 + h, y1:y1 + w] * 0.5 + 0.5 * np.random.uniform(0, 1)# RGB饱和度高，现在要降一些，适应IR
                    img[2, x1:x1 + h, y1:y1 + w] = img[2, x1:x1 + h, y1:y1 + w] * 0.9 + 0.1 * np.random.uniform(1, 1./(img[2, x1:x1 + h, y1:y1 + w].max()))# RGB亮度没有很高，现在要高一些，适应IR
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = np.random.uniform(0, 1) #(img[0, x1:x1 + h, y1:y1 + w] + np.random.uniform(0, 1))%1
                    img[1, x1:x1 + h, y1:y1 + w] = img[1, x1:x1 + h, y1:y1 + w] * 0.5 + 0.5 * np.random.uniform(0, 1)
                    img[2, x1:x1 + h, y1:y1 + w] = img[2, x1:x1 + h, y1:y1 + w] * 0.5 + 0.5 * np.random.uniform(0, 1)
        img = cvt.hsv_to_rgb(img.unsqueeze(0))[0]
        return img

class RGB_HSV(torch.nn.Module):
    def __init__(self, eps=1e-8):
        super(RGB_HSV, self).__init__()
        self.eps = eps

    def rgb_to_hsv(self, img):
        # img: (B, C, H, W)
        # img = img / 255.0
        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)

        hue[img[:, 2] == img.max(1)[0]] = 4.0 + ((img[:, 0] - img[:, 1]) / (img.max(1)[0] - img.min(1)[0] + self.eps))[
            img[:, 2] == img.max(1)[0]]
        hue[img[:, 1] == img.max(1)[0]] = 2.0 + ((img[:, 2] - img[:, 0]) / (img.max(1)[0] - img.min(1)[0] + self.eps))[
            img[:, 1] == img.max(1)[0]]
        hue[img[:, 0] == img.max(1)[0]] = (0.0 + ((img[:, 1] - img[:, 2]) / (img.max(1)[0] - img.min(1)[0] + self.eps))[
            img[:, 0] == img.max(1)[0]]) % 6

        hue[img.min(1)[0] == img.max(1)[0]] = 0.0
        hue = hue / 6

        saturation = (img.max(1)[0] - img.min(1)[0]) / (img.max(1)[0] + self.eps)
        saturation[img.max(1)[0] == 0] = 0

        value = img.max(1)[0]

        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)
        hsv = torch.cat([hue, saturation, value], dim=1)
        return hsv

    def hsv_to_rgb(self, hsv):
        h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]
        # 对出界值的处理
        h = h % 1
        s = torch.clamp(s, 0, 1)
        v = torch.clamp(v, 0, 1)

        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)

        hi = torch.floor(h * 6)
        f = h * 6 - hi
        p = v * (1 - s)
        q = v * (1 - (f * s))
        t = v * (1 - ((1 - f) * s))

        hi0 = hi == 0
        hi1 = hi == 1
        hi2 = hi == 2
        hi3 = hi == 3
        hi4 = hi == 4
        hi5 = hi == 5

        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]

        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]

        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]

        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]

        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]

        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]

        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)
        return rgb