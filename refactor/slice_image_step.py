#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像切分模块
负责将长图切分为多个切片
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple


def slice_image(image_path: str, slice_height: int, overlap: int, output_images_dir: Path) -> Tuple[np.ndarray, List[Dict]]:
    """
    切分长图
    
    Args:
        image_path: 图像路径
        slice_height: 切片高度
        overlap: 重叠区域像素
        output_images_dir: 输出图像目录
        
    Returns:
        original_image: 原始图像
        slices_info: 切片信息列表，包含切片图像和位置信息
    """
    # 读取原始图像
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"无法读取图像: {image_path}")
        
    h, w, c = original_image.shape
    print(f"原始图像尺寸: {w} x {h}")
    
    if h <= slice_height:
        # 图像高度小于等于切片高度，不需要切分
        return original_image, [{
            'slice': original_image,
            'start_y': 0,
            'end_y': h,
            'slice_index': 0
        }]
    
    slices_info = []
    current_y = 0
    slice_index = 0
    
    while current_y < h:
        # 计算当前切片的结束位置
        end_y = min(current_y + slice_height, h)
        
        # 提取切片
        slice_img = original_image[current_y:end_y, :, :]
        
        # 保存切片信息
        slice_info = {
            'slice': slice_img,
            'start_y': current_y,
            'end_y': end_y,
            'slice_index': slice_index
        }
        slices_info.append(slice_info)
        
        # 保存切片图像
        slice_path = output_images_dir / f"slice_{slice_index:03d}.jpg"
        cv2.imwrite(str(slice_path), slice_img)
        print(f"保存切片 {slice_index}: {slice_path}")
        
        # 计算下一个切片的起始位置
        if end_y >= h:
            break
        current_y = end_y - overlap
        slice_index += 1
        
    return original_image, slices_info


def perform_image_slicing(ocr_instance, image_path):
    """
    执行图像切分步骤
    
    Args:
        ocr_instance: LongImageOCR实例
        image_path: 图像路径
        
    Returns:
        tuple: (original_image, slices_info)
    """
    # 1. 切分图像
    print("步骤1: 切分图像...")
    original_image, slices_info = slice_image(
        image_path, 
        ocr_instance.slice_height, 
        ocr_instance.overlap, 
        ocr_instance.output_images_dir
    )
    # 保存原图到实例属性中，供其他方法使用
    ocr_instance.original_image = original_image
    print(f"共切分为 {len(slices_info)} 个切片")
    
    return original_image, slices_info