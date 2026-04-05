# -*- coding: utf-8 -*-
"""
推理阶段注意力热力图叠加与保存。
将 patch 级注意力 [H_p, W_p] 上采样到图像尺寸，用 jet 色图叠加到原图并保存。

热力图特征来源（在 models/clip_vit.py get_spatial_attention 中）：
- source='pre_transformer'（默认）：Transformer 前的 patch 特征 → 投影 → 与各类文本的余弦相似度 → 每 patch 取最大。
  空间更局部，热力图更集中在物体上。
- source='final'：Transformer 最后一层 ln_post 后的 patch 特征，同上投影与相似度。与预测一致但更平滑。
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ImageNet 标准归一化参数（与 data_loader 中 test transform 一致）
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def denormalize_tensor(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """将 [C,H,W] 或 [B,C,H,W] 的归一化 tensor 反归一化到 [0,1] 范围。"""
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    t = tensor.cpu().float().numpy()
    mean = mean.reshape(1, 3, 1, 1)
    std = std.reshape(1, 3, 1, 1)
    t = t * std + mean
    t = np.clip(t, 0.0, 1.0)
    return t


def attention_to_heatmap_rgba(att_np, alpha=0.5):
    """
    将 2D 注意力图转为 RGBA 热力图（蓝-绿-黄-红），用于叠加。
    att_np: (H, W), 会先做 min-max 归一化
    alpha: 热力图通道透明度
    returns: (H, W, 4) float in [0,1]
    """
    att_min, att_max = att_np.min(), att_np.max()
    if att_max - att_min > 1e-6:
        att_norm = (att_np - att_min) / (att_max - att_min)
    else:
        att_norm = np.zeros_like(att_np)
    cmap = plt.get_cmap('jet')
    heatmap_rgb = cmap(att_norm)[:, :, :3]
    heatmap_alpha = (att_norm * alpha).reshape(att_norm.shape[0], att_norm.shape[1], 1)
    return np.concatenate([heatmap_rgb, heatmap_alpha], axis=-1)


def save_heatmap_overlay(
    img_tensor,
    attention_2d,
    save_path,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
    heatmap_alpha=0.5,
    image_size=224,
):
    """
    将注意力图叠加到原图并保存为 PNG。
    Args:
        img_tensor: [3, H, W] 或 [1, 3, H, W]，已归一化的图像
        attention_2d: [h, w] numpy 或 tensor，patch 级注意力
        save_path: 保存路径
        heatmap_alpha: 热力图叠加强度 (0~1)
        image_size: 输出图像边长（与原图一致，通常 224）
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    if torch.is_tensor(attention_2d):
        attention_2d = attention_2d.cpu().float().numpy()
    if torch.is_tensor(img_tensor):
        img_np = denormalize_tensor(img_tensor, mean, std)
    else:
        img_np = np.asarray(img_tensor)
    if img_np.ndim == 4:
        img_np = img_np[0]
    img_np = np.transpose(img_np, (1, 2, 0))
    img_np = np.ascontiguousarray(img_np)
    h_in, w_in = img_np.shape[:2]
    att_h, att_w = attention_2d.shape
    heatmap_resized = np.array(
        np.repeat(
            np.repeat(attention_2d, (h_in + att_h - 1) // att_h, axis=0)[:h_in],
            (w_in + att_w - 1) // att_w,
            axis=1,
        )[:h_in, :w_in]
    )
    heatmap_rgba = attention_to_heatmap_rgba(heatmap_resized, alpha=heatmap_alpha)
    img_rgb = img_np.squeeze()
    if img_rgb.ndim == 2:
        img_rgb = np.stack([img_rgb] * 3, axis=-1)
    overlay = heatmap_rgba[:, :, :3] * heatmap_rgba[:, :, 3:4] + img_rgb * (1.0 - heatmap_rgba[:, :, 3:4])
    overlay = np.clip(overlay, 0.0, 1.0)
    plt.imsave(save_path, overlay)
    plt.close('all')


def _resize_2d_numpy(arr, out_h, out_w):
    """用 repeat 将 (h,w) 放大到 (out_h, out_w)。"""
    h, w = arr.shape
    if h == out_h and w == out_w:
        return arr
    scale_h = (out_h + h - 1) // h
    scale_w = (out_w + w - 1) // w
    out = np.repeat(np.repeat(arr, scale_h, axis=0)[:out_h], scale_w, axis=1)[:out_h, :out_w]
    return out


def save_heatmap_overlay_resize(
    img_tensor,
    attention_2d,
    save_path,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
    heatmap_alpha=0.5,
    image_size=224,
):
    """
    将 [h,w] 注意力图放大到 image_size，再叠加到原图保存。
    与参考图一致：热力图与图像同尺寸，蓝/紫为低注意力，黄/红为高注意力。
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    if torch.is_tensor(attention_2d):
        attention_2d = attention_2d.cpu().float().numpy()
    if torch.is_tensor(img_tensor):
        img_np = denormalize_tensor(img_tensor, mean, std)
    else:
        img_np = np.asarray(img_tensor, dtype=np.float32)
    if img_np.ndim == 4:
        img_np = img_np[0]
    img_np = np.transpose(img_np, (1, 2, 0))
    img_np = np.ascontiguousarray(img_np)
    if img_np.shape[0] != image_size or img_np.shape[1] != image_size:
        h, w = img_np.shape[:2]
        scale_h = (image_size + h - 1) // h
        scale_w = (image_size + w - 1) // w
        big = np.repeat(np.repeat(img_np, scale_h, axis=0)[:image_size], scale_w, axis=1)[:image_size, :image_size]
        img_np = big
    img_rgb = img_np.squeeze()
    if img_rgb.ndim == 2:
        img_rgb = np.stack([img_rgb] * 3, axis=-1)
    att_resized = _resize_2d_numpy(attention_2d.astype(np.float32), image_size, image_size)
    att_min, att_max = att_resized.min(), att_resized.max()
    if att_max - att_min > 1e-6:
        att_norm = (att_resized - att_min) / (att_max - att_min)
    else:
        att_norm = np.zeros_like(att_resized)
    cmap = plt.get_cmap('jet')
    heatmap_rgb = cmap(att_norm)[:, :, :3]
    overlay = heatmap_rgb * heatmap_alpha + img_rgb * (1.0 - heatmap_alpha)
    overlay = np.clip(overlay, 0.0, 1.0)
    plt.imsave(save_path, overlay)
    plt.close('all')
