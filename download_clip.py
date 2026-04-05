#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预下载 CLIP 权重到默认缓存目录（~/.cache/clip 或 Windows 下 %USERPROFILE%\\.cache\\clip）。
首次运行训练/测试时若不指定 --clip-path，也会自动下载；本脚本用于提前下载或检查缓存。
"""
import argparse
import clip

def main():
    parser = argparse.ArgumentParser(description='预下载 CLIP 模型')
    parser.add_argument('--model', type=str, default='ViT-B/16',
                        choices=list(clip.available_models()),
                        help='要下载的模型名，默认 ViT-B/16')
    args = parser.parse_args()
    print(f'正在下载 {args.model}（若已缓存则跳过）...')
    model, preprocess = clip.load(args.model, device='cpu', jit=False)
    print(f'完成。模型已缓存，后续可不指定 --clip-path 直接运行训练/测试。')

if __name__ == '__main__':
    main()
