"""
图像切片模块
负责将长图切分为多个切片
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
import logging

from ..models.slice_info import SliceInfo
from ..utils.config import Config

logger = logging.getLogger(__name__)


class ImageSlicer:
    """图像切片器"""
    
    def __init__(self, config: Config):
        """
        初始化图像切片器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.slice_height = config.image.slice_height
        self.overlap = config.image.overlap
        self.output_dir = Path(config.output.output_images_dir)
    
    def slice_image(self, image_path: str) -> Tuple[np.ndarray, List[SliceInfo]]:
        """
        切分长图
        
        Args:
            image_path: 图像路径
            
        Returns:
            (original_image, slice_infos): 原始图像和切片信息列表
        """
        # 读取原始图像
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        h, w, c = original_image.shape
        logger.info(f"原始图像尺寸: {w} x {h}")
        
        # 如果图像高度小于等于切片高度，不需要切分
        if h <= self.slice_height:
            slice_info = SliceInfo(
                slice_index=0,
                start_y=0,
                end_y=h,
                image=original_image.copy()
            )
            self._save_slice(slice_info)
            return original_image, [slice_info]
        
        # 切分图像
        slice_infos = self._perform_slicing(original_image)
        
        logger.info(f"共切分为 {len(slice_infos)} 个切片")
        return original_image, slice_infos
    
    def _perform_slicing(self, image: np.ndarray) -> List[SliceInfo]:
        """
        执行图像切分
        
        Args:
            image: 原始图像
            
        Returns:
            切片信息列表
        """
        h, w, c = image.shape
        slice_infos = []
        current_y = 0
        slice_index = 0
        
        while current_y < h:
            # 计算当前切片的结束位置
            end_y = min(current_y + self.slice_height, h)
            
            # 提取切片
            slice_img = image[current_y:end_y, :, :].copy()
            
            # 创建切片信息
            slice_info = SliceInfo(
                slice_index=slice_index,
                start_y=current_y,
                end_y=end_y,
                image=slice_img
            )
            slice_infos.append(slice_info)
            
            # 保存切片图像
            self._save_slice(slice_info)
            
            # 计算下一个切片的起始位置
            if end_y >= h:
                break
            current_y = end_y - self.overlap
            slice_index += 1
        
        return slice_infos
    
    def _save_slice(self, slice_info: SliceInfo):
        """
        保存切片图像
        
        Args:
            slice_info: 切片信息
        """
        slice_path = self.output_dir / f"slice_{slice_info.slice_index:03d}.jpg"
        cv2.imwrite(str(slice_path), slice_info.image)
        logger.debug(f"保存切片 {slice_info.slice_index}: {slice_path}")
    
    def get_slices_to_process(self, slice_infos: List[SliceInfo]) -> List[SliceInfo]:
        """
        根据切片数量决定处理策略
        
        Args:
            slice_infos: 所有切片信息列表
            
        Returns:
            需要处理的切片列表
        """
        total_slices = len(slice_infos)
        
        if total_slices == 1:
            # 只有一个切片，处理所有切片
            slices_to_process = slice_infos
            logger.info("只有一个切片，将处理所有切片")
        elif total_slices == 2:
            # 两个切片，选择第一个切片
            slices_to_process = slice_infos[:1]
            logger.info("有2个切片，将只处理第一个切片")
        else:
            # 大于等于3个切片，排除开始和结束的切片，只处理中间切片
            slices_to_process = slice_infos[1:-1]
            logger.info(f"共有{total_slices}个切片，将处理中间{len(slices_to_process)}个切片")
        
        return slices_to_process