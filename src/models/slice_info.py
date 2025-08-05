"""
切片信息数据模型
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class SliceInfo:
    """切片信息"""
    slice_index: int
    start_y: int
    end_y: int
    image: np.ndarray
    
    @property
    def height(self) -> int:
        """切片高度"""
        return self.end_y - self.start_y
    
    @property
    def shape(self) -> tuple:
        """切片图像形状"""
        return self.image.shape
    
    def contains_y(self, y: float) -> bool:
        """判断Y坐标是否在切片范围内"""
        return self.start_y <= y <= self.end_y
    
    def to_dict(self) -> dict:
        """转换为字典格式（不包含图像数据）"""
        return {
            'slice_index': self.slice_index,
            'start_y': self.start_y,
            'end_y': self.end_y,
            'height': self.height
        }