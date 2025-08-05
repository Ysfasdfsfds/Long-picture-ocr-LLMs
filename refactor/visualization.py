"""
可视化模块
用于绘制和保存各种可视化结果
"""

import cv2
from typing import Optional, Tuple, List, Dict
import numpy as np


def visualize_selected_box(selected_box: Optional[Tuple], slices_info: List[Dict], 
                          original_image: Optional[np.ndarray]) -> None:
    """
    将选中的框绘制到切片和原图上并保存
    
    Args:
        selected_box: 选中的框 (x, y, w, h, slice_idx) 或 None
        slices_info: 所有切片信息列表
        original_image: 原始图像
    """
    if selected_box is None:
        print("没有选中的框需要可视化")
        return
    
    x, y, w, h, slice_idx = selected_box
    
    # 获取对应切片的图像
    for slice_info in slices_info:
        if slice_info['slice_index'] == slice_idx:
            slice_img = slice_info['slice']
            # 在切片图像上绘制矩形
            slice_with_box = slice_img.copy()
            cv2.rectangle(slice_with_box, (x, y - slice_info['start_y']), 
                         (x + w, y + h - slice_info['start_y']), (0, 0, 255), 2)
            cv2.imwrite(f"output_images/selected_box_slice_{slice_idx}.jpg", slice_with_box)
            print(f"已将selected_box绘制到切片{slice_idx}图像并保存")
            
            # 在原图上绘制矩形
            if original_image is not None:
                original_with_box = original_image.copy()
                cv2.rectangle(original_with_box, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.imwrite("output_images/selected_box_original.jpg", original_with_box)
                print("已将selected_box绘制到原图并保存")
            break


def visualize_ocr_and_avatar_boxes(slice_img: np.ndarray, ocr_boxes: List, 
                                 avatar_boxes: List, start_y: int, 
                                 output_path: str) -> None:
    """
    在切片图像上绘制OCR框（绿色）和头像框（红色）
    
    Args:
        slice_img: 切片图像
        ocr_boxes: OCR检测框列表（原图坐标）
        avatar_boxes: 头像检测框列表 [(x, y, w, h), ...]（原图坐标）
        start_y: 切片在原图中的起始y坐标
        output_path: 输出文件路径
    """
    img_with_boxes = slice_img.copy()
    
    # 画红色框：头像框
    for (x, y, w, h) in avatar_boxes:
        # 转换为切片坐标
        y_in_slice = y - start_y
        cv2.rectangle(img_with_boxes, (x, y_in_slice), 
                     (x + w, y_in_slice + h), (0, 0, 255), 2)  # 红色BGR
    
    # 画绿色框：OCR框
    for box in ocr_boxes:
        # box: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        pts = [(int(pt[0]), int(pt[1] - start_y)) for pt in box]  # 转回切片坐标
        for i in range(4):
            pt1 = pts[i]
            pt2 = pts[(i + 1) % 4]
            cv2.line(img_with_boxes, pt1, pt2, (0, 255, 0), 2)  # 绿色BGR
    
    cv2.imwrite(output_path, img_with_boxes)
    print(f"已保存可视化结果到: {output_path}")