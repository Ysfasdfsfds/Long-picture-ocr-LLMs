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
        self.slice_width = config.image.slice_width
        self.overlap = config.image.overlap
        self.x_overlap = config.image.x_overlap
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
        
        # 如果图像高度小于等于切片高度且宽度小于等于切片宽度，不需要切分
        if h <= self.slice_height and w <= self.slice_width:
            slice_info = SliceInfo(
                slice_index=0,
                x_index=0,
                y_index=0,
                start_x=0,
                end_x=w,
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
        执行图像切分（支持x-y双向切片）
        
        Args:
            image: 原始图像
            
        Returns:
            切片信息列表
        """
        h, w, c = image.shape
        slice_infos = []
        slice_index = 0
        
        # 确定x方向切片策略
        if w <= self.slice_width:
            # 宽度小于等于切片宽度，只在y方向切片
            logger.info(f"图像宽度 {w} <= {self.slice_width}，只进行y方向切片")
            x_slices = [(0, w)]  # 只有一个x切片：从0到w
        else:
            # 宽度大于切片宽度，在x方向切片
            logger.info(f"图像宽度 {w} > {self.slice_width}，进行x-y双向切片")
            x_slices = self._calculate_x_slices(w)
        
        # 遍历x方向切片
        for x_idx, (start_x, end_x) in enumerate(x_slices):
            # 遍历y方向切片
            current_y = 0
            y_idx = 0
            
            while current_y < h:
                # 计算当前切片的结束位置
                end_y = min(current_y + self.slice_height, h)
                
                # 提取切片
                slice_img = image[current_y:end_y, start_x:end_x, :].copy()
                
                # 创建切片信息
                slice_info = SliceInfo(
                    slice_index=slice_index,
                    x_index=x_idx,
                    y_index=y_idx,
                    start_x=start_x,
                    end_x=end_x,
                    start_y=current_y,
                    end_y=end_y,
                    image=slice_img
                )
                slice_infos.append(slice_info)
                
                # 保存切片图像
                self._save_slice(slice_info)
                
                # 计算下一个y方向切片的起始位置
                if end_y >= h:
                    break
                current_y = end_y - self.overlap
                y_idx += 1
                slice_index += 1
        
        return slice_infos
    
    def _calculate_x_slices(self, total_width: int) -> List[Tuple[int, int]]:
        """
        计算x方向的切片位置
        
        Args:
            total_width: 图像总宽度
            
        Returns:
            x切片位置列表，每个元素为(start_x, end_x)
        """
        x_slices = []
        current_x = 0
        
        while current_x < total_width:
            # 计算当前切片的结束位置
            end_x = min(current_x + self.slice_width, total_width)
            
            x_slices.append((current_x, end_x))
            
            # 计算下一个切片的起始位置
            if end_x >= total_width:
                break
            current_x = end_x - self.x_overlap
        
        logger.info(f"x方向切片计算结果: {x_slices}")
        return x_slices
    
    def _save_slice(self, slice_info: SliceInfo):
        """
        保存切片图像
        
        Args:
            slice_info: 切片信息
        """
        # 新文件命名格式：slice_x{x_idx}_y{y_idx}.jpg（例如：slice_x0_y0.jpg）
        slice_path = self.output_dir / f"slice_x{slice_info.x_index:02d}_y{slice_info.y_index:02d}.jpg"
        cv2.imwrite(str(slice_path), slice_info.image)
        logger.debug(f"保存切片 x{slice_info.x_index}_y{slice_info.y_index} (索引{slice_info.slice_index}): {slice_path}")
    
    def get_slices_to_process(self, slice_infos: List[SliceInfo]) -> List[SliceInfo]:
        """
        根据切片数量决定处理策略（已弃用，建议使用AvatarDetector中的方法）
        
        Args:
            slice_infos: 所有切片信息列表
            
        Returns:
            需要处理的切片列表
        """
        # 这个方法保持向后兼容，但建议使用AvatarDetector中的对应方法
        logger.warning("ImageSlicer.get_slices_to_process已弃用，建议使用AvatarDetector中的对应方法")
        
        total_slices = len(slice_infos)
        
        if total_slices == 1:
            slices_to_process = slice_infos
            logger.info("只有一个切片，将处理所有切片")
        elif total_slices == 2:
            slices_to_process = slice_infos[:1]
            logger.info("有2个切片，将只处理第一个切片")
        else:
            slices_to_process = slice_infos[1:-1]
            logger.info(f"共有{total_slices}个切片，将处理中间{len(slices_to_process)}个切片")
        
        return slices_to_process