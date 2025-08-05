"""
图像处理工具函数
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def save_debug_image(image: np.ndarray, filename: str, output_dir: str = "./output_images/debug"):
    """
    保存调试图像
    
    Args:
        image: 要保存的图像
        filename: 文件名
        output_dir: 输出目录
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    full_path = output_path / filename
    cv2.imwrite(str(full_path), image)


def draw_boxes_on_image(image: np.ndarray, boxes: list, 
                       color: Tuple[int, int, int] = (0, 255, 0),
                       thickness: int = 2) -> np.ndarray:
    """
    在图像上绘制边界框
    
    Args:
        image: 原始图像
        boxes: 边界框列表
        color: 绘制颜色 (B, G, R)
        thickness: 线条粗细
        
    Returns:
        绘制了边界框的图像副本
    """
    result_img = image.copy()
    
    for box in boxes:
        if isinstance(box[0], list):
            # OCR格式: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            points = np.array(box, dtype=np.int32)
            cv2.polylines(result_img, [points], True, color, thickness)
        else:
            # xywh格式: (x, y, w, h)
            x, y, w, h = box
            cv2.rectangle(result_img, (int(x), int(y)), 
                         (int(x + w), int(y + h)), color, thickness)
    
    return result_img


def crop_image_region(image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """
    裁剪图像区域
    
    Args:
        image: 原始图像
        x, y: 左上角坐标
        w, h: 宽度和高度
        
    Returns:
        裁剪后的图像区域
    """
    # 确保坐标在图像范围内
    x = max(0, x)
    y = max(0, y)
    x2 = min(image.shape[1], x + w)
    y2 = min(image.shape[0], y + h)
    
    return image[y:y2, x:x2]


def resize_image_keeping_aspect_ratio(image: np.ndarray, 
                                    max_width: Optional[int] = None,
                                    max_height: Optional[int] = None) -> np.ndarray:
    """
    保持纵横比缩放图像
    
    Args:
        image: 原始图像
        max_width: 最大宽度
        max_height: 最大高度
        
    Returns:
        缩放后的图像
    """
    h, w = image.shape[:2]
    
    if max_width is None and max_height is None:
        return image
    
    # 计算缩放比例
    scale = 1.0
    if max_width is not None:
        scale = min(scale, max_width / w)
    if max_height is not None:
        scale = min(scale, max_height / h)
    
    if scale >= 1.0:
        return image
    
    # 计算新尺寸
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 缩放图像
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized


def merge_images_vertically(images: list, gap: int = 10) -> np.ndarray:
    """
    垂直合并多个图像
    
    Args:
        images: 图像列表
        gap: 图像之间的间隔（像素）
        
    Returns:
        合并后的图像
    """
    if not images:
        return np.array([])
    
    # 获取最大宽度
    max_width = max(img.shape[1] for img in images)
    
    # 计算总高度
    total_height = sum(img.shape[0] for img in images) + gap * (len(images) - 1)
    
    # 创建结果图像（白色背景）
    result = np.ones((total_height, max_width, 3), dtype=np.uint8) * 255
    
    # 合并图像
    current_y = 0
    for img in images:
        h, w = img.shape[:2]
        # 居中放置
        x_offset = (max_width - w) // 2
        result[current_y:current_y + h, x_offset:x_offset + w] = img
        current_y += h + gap
    
    return result