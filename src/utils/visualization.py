"""
可视化工具模块
用于生成调试和展示用的可视化图像
"""

import cv2
import numpy as np
import logging
from typing import List, Optional, Tuple
from pathlib import Path

from ..models.slice_info import SliceInfo
from ..utils.config import Config

logger = logging.getLogger(__name__)


class Visualizer:
    """可视化工具类"""
    
    def __init__(self, config: Config):
        """
        初始化可视化工具
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.output_dir = Path(config.output.output_images_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def visualize_selected_box(self, selected_box: Optional[Tuple], 
                             slice_infos: List[SliceInfo],
                             original_image: np.ndarray,
                             output_file: str = "selected_box_visualization.jpg"):
        """
        可视化选中的目标框
        
        Args:
            selected_box: 选中的框 (x, y, w, h, slice_idx)
            slice_infos: 切片信息列表
            original_image: 原始图像
            output_file: 输出文件名
        """
        if selected_box is None:
            logger.warning("没有选中的框可以可视化")
            return
        
        x, y, w, h, slice_idx = selected_box
        
        # 在原图上绘制
        vis_img = original_image.copy()
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(vis_img, f"Selected Box (Slice {slice_idx})", 
                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 保存原图可视化
        output_path = self.output_dir / output_file
        cv2.imwrite(str(output_path), vis_img)
        logger.info(f"选中框可视化已保存到: {output_path}")
        
        # 也在对应切片上绘制
        if 0 <= slice_idx < len(slice_infos):
            slice_info = slice_infos[slice_idx]
            slice_vis = slice_info.image.copy()
            
            # 转换为切片坐标
            slice_y = y - slice_info.start_y
            cv2.rectangle(slice_vis, (x, slice_y), (x + w, slice_y + h), (0, 0, 255), 3)
            
            slice_output = self.output_dir / f"selected_box_slice_{slice_idx}.jpg"
            cv2.imwrite(str(slice_output), slice_vis)
            logger.info(f"切片{slice_idx}的选中框可视化已保存到: {slice_output}")
    
    def visualize_avatars_on_image(self, image: np.ndarray, avatar_boxes: List[Tuple],
                                  output_file: str = "avatars_marked.jpg"):
        """
        在图像上可视化头像框
        
        Args:
            image: 图像
            avatar_boxes: 头像框列表 [(x, y, w, h), ...]
            output_file: 输出文件名
        """
        vis_img = image.copy()
        
        for idx, box in enumerate(avatar_boxes):
            x, y, w, h = box
            # 绘制矩形
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # 添加编号
            cv2.putText(vis_img, f"{idx}:y={y}", (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        output_path = self.output_dir / output_file
        cv2.imwrite(str(output_path), vis_img)
        logger.info(f"头像框可视化已保存到: {output_path}")
    
    def visualize_ocr_results(self, image: np.ndarray, ocr_boxes: List,
                            texts: List[str], scores: List[float],
                            output_file: str = "ocr_results.jpg"):
        """
        可视化OCR识别结果
        
        Args:
            image: 图像
            ocr_boxes: OCR边界框列表
            texts: 识别的文本列表
            scores: 置信度分数列表
            output_file: 输出文件名
        """
        vis_img = image.copy()
        
        for box, text, score in zip(ocr_boxes, texts, scores):
            # 绘制多边形
            points = np.array(box, dtype=np.int32)
            cv2.polylines(vis_img, [points], True, (0, 255, 0), 2)
            
            # 添加文本和分数
            x_min = min(pt[0] for pt in box)
            y_min = min(pt[1] for pt in box)
            label = f"{text[:10]}... ({score:.2f})" if len(text) > 10 else f"{text} ({score:.2f})"
            cv2.putText(vis_img, label, (x_min, y_min - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        output_path = self.output_dir / output_file
        cv2.imwrite(str(output_path), vis_img)
        logger.info(f"OCR结果可视化已保存到: {output_path}")
    
    def create_process_summary_image(self, original_image: np.ndarray,
                                   slice_infos: List[SliceInfo],
                                   ocr_count: int, avatar_count: int,
                                   output_file: str = "process_summary.jpg"):
        """
        创建处理过程总结图像
        
        Args:
            original_image: 原始图像
            slice_infos: 切片信息列表
            ocr_count: OCR识别结果数量
            avatar_count: 头像检测数量
            output_file: 输出文件名
        """
        h, w = original_image.shape[:2]
        
        # 创建缩略图
        scale = min(800 / w, 1.0)
        new_w = int(w * scale)
        new_h = int(h * scale)
        thumbnail = cv2.resize(original_image, (new_w, new_h))
        
        # 在缩略图上绘制切片边界
        for slice_info in slice_infos:
            y1 = int(slice_info.start_y * scale)
            y2 = int(slice_info.end_y * scale)
            cv2.line(thumbnail, (0, y1), (new_w, y1), (0, 255, 0), 2)
            cv2.putText(thumbnail, f"Slice {slice_info.slice_index}", 
                       (10, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 添加统计信息
        info_height = 100
        result = np.ones((new_h + info_height, new_w, 3), dtype=np.uint8) * 255
        result[:new_h] = thumbnail
        
        # 添加文本信息
        y_offset = new_h + 30
        cv2.putText(result, f"Total Slices: {len(slice_infos)}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(result, f"OCR Results: {ocr_count}", 
                   (10, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(result, f"Avatars Detected: {avatar_count}", 
                   (10, y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        output_path = self.output_dir / output_file
        cv2.imwrite(str(output_path), result)
        logger.info(f"处理总结图像已保存到: {output_path}")