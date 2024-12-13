import matplotlib.pyplot as plt
import torch
import cv2
import re
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
# from part.part_base import Base
from PIL import Image
# from part.part_base_attention import Base as BaseAtt

def create_gradient_mask(part_img, now_part, back_ground, shape, blur_kernel=(15, 15)):
    # 初始化掩码
    part_mask = np.zeros(shape[:2], dtype=np.float32)
    # 创建布尔掩码
    now_part_mask = np.all(part_img == now_part, axis=-1)
    back_ground_mask = np.all(part_img == back_ground, axis=-1)
    # 创建距离场（逐渐变化的值）
    now_part_dist = cv2.distanceTransform((now_part_mask).astype(np.uint8), cv2.DIST_L2, 5)
    other_part_dist = cv2.distanceTransform((~(now_part_mask | back_ground_mask)).astype(np.uint8), cv2.DIST_L2, 5)
    # 归一化距离场
    now_part_dist = now_part_dist / now_part_dist.max() if now_part_dist.max() > 0 else now_part_dist
    other_part_dist = other_part_dist / other_part_dist.max() if other_part_dist.max() > 0 else other_part_dist
    # 生成渐进掩码值
    part_mask[now_part_mask] = 0.1 + 0.8 * now_part_dist[now_part_mask]  # now_part区域逐渐递增到1.0
    part_mask[~(now_part_mask | back_ground_mask)] = 0.05 + 0.35 * other_part_dist[~(now_part_mask | back_ground_mask)]  # 其他区域逐渐递增到0.5
    part_mask[back_ground_mask] = np.random.uniform(0.0, 0.05, size=back_ground_mask.sum())  # 背景区域固定为最低范围

    # 添加轻微随机噪声
    noise = np.random.uniform(-0.05, 0.05, part_mask.shape)
    part_mask += noise
    # 应用高斯模糊
    part_mask = cv2.GaussianBlur(part_mask, blur_kernel, 0)
    return part_mask

# 可视化热力图
def my_show_cam_on_image(img, mask, use_rgb=True):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255.0
    if use_rgb:
        heatmap = heatmap[..., ::-1]  # BGR to RGB
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

idx= '0001'
id = '0091'
image_path = "E:/hhj/SYSU-MM01/cam2/" + id + "/" + idx +".jpg"
image = Image.open(image_path)

part_image_path = image_path.replace('SYSU-MM01', 'SYSU-MM01-output')
part_image_path = re.sub(r'/(\d+).jpg$', r'/rgb_\1.png', part_image_path)
part_image = Image.open(part_image_path).convert('RGB')
print(part_image.size)

part_list = [[192,0,128],[128,0,128],[64,128,128],[192,128,128]]
img = np.array(image)
part_img = np.array(part_image)
# 创建初始化掩码和给出部位的颜色
part_num = 3
part_mask = np.zeros(part_img.shape[:2], dtype=np.float32)
now_part = part_list[part_num]
back_ground = [0, 0, 0]
# 创建布尔掩码
now_part_mask = np.all(part_img == now_part, axis=-1)
back_ground_mask = np.all(part_img == back_ground, axis=-1)
# 生成随机化的掩码
part_mask = create_gradient_mask(part_img, now_part, back_ground, img.shape)

visualization = show_cam_on_image(img.astype(dtype=np.float32)/255.0, part_mask, use_rgb=True)
# visualization.reshape()
print(visualization.shape)
print(img.shape)

# 对visualization进行变形
visualization_resized = cv2.resize(visualization, (144, 288))
# 保存图片
# 将图像从 RGB 转换为 BGR
visualization_resized_bgr = cv2.cvtColor(visualization_resized, cv2.COLOR_RGB2BGR)
output_path = f"./save/{part_num+1}/{id}.jpg"
cv2.imwrite(output_path, visualization_resized_bgr)

plt.imshow(visualization_resized)
plt.axis('off')
plt.show()
print('----------------------')

# print('----------------------')
# target_layers = [model_trans.module.image_encoder[-1][-1]]
# cam = GradCAM(model=model_trans, target_layers=target_layers)
#
# grayscale_cam = cam(input_tensor=input_tensor, targets=None)
# grayscale_cam = grayscale_cam[0, :]
#
# img = np.array(image)
# img = cv2.resize(img, (config.img_w, config.img_h))
#
# partization = show_cam_on_image(img.astype(dtype=np.float32)/255.0, grayscale_cam, use_rgb=True)
#
# plt.imshow(partization)
# plt.axis('off')
# plt.show()

