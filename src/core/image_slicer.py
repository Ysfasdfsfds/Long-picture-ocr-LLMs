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
            # TODO(human): 实现详细的错误诊断逻辑
            error_msg = self._diagnose_image_read_failure(image_path)
            raise ValueError(f"无法读取图像: {image_path}. {error_msg}")
        
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
    
    def _diagnose_image_read_failure(self, image_path: str) -> str:
        """诊断图像读取失败的具体原因，返回描述性错误信息"""
        import os
        from pathlib import Path
        
        # 检查文件是否存在
        if not os.path.exists(image_path):
            return "文件不存在，请检查路径是否正确"
        
        # 检查是否是文件(不是目录)
        if not os.path.isfile(image_path):
            return "路径指向的不是文件"
        
        # 检查文件权限
        if not os.access(image_path, os.R_OK):
            return "文件无读取权限"
        
        # 检查文件大小
        file_size = os.path.getsize(image_path)
        if file_size == 0:
            return "文件为空"
        
        # 检查文件扩展名
        file_ext = Path(image_path).suffix.lower()
        supported_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
        if file_ext not in supported_formats:
            return f"不支持的文件格式: {file_ext}，支持的格式: {', '.join(supported_formats)}"
        
        # 检查文件是否损坏(尝试读取文件头)
        try:
            with open(image_path, 'rb') as f:
                header = f.read(10)
                if len(header) < 4:
                    return "文件头信息不完整，可能文件损坏"
        except Exception as e:
            return f"无法读取文件: {str(e)}"
        
        return "文件格式可能不被OpenCV支持或文件已损坏"