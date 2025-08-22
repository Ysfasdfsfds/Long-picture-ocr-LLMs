"""
切片信息数据模型
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class SliceInfo:
    """切片信息"""
    slice_index: int  # 保持向后兼容的综合索引
    x_index: int      # x方向索引（列索引）
    y_index: int      # y方向索引（行索引）
    start_x: int      # x方向起始坐标
    end_x: int        # x方向结束坐标
    start_y: int      # y方向起始坐标
    end_y: int        # y方向结束坐标
    image: np.ndarray
    
    @property
    def width(self) -> int:
        """切片宽度"""
        return self.end_x - self.start_x
    
    @property
    def height(self) -> int:
        """切片高度"""
        return self.end_y - self.start_y
    
    @property
    def shape(self) -> tuple:
        """切片图像形状"""
        return self.image.shape
    
    @property
    def is_left_column(self) -> bool:
        """判断是否为左侧第一列切片"""
        return self.x_index == 0
    
    def contains_point(self, x: float, y: float) -> bool:
        """判断坐标点是否在切片范围内"""
        return (self.start_x <= x <= self.end_x and 
                self.start_y <= y <= self.end_y)
    
    def contains_y(self, y: float) -> bool:
        """判断Y坐标是否在切片范围内（保持向后兼容）"""
        return self.start_y <= y <= self.end_y
    
    def to_dict(self) -> dict:
        """转换为字典格式（不包含图像数据）"""
        return {
            'slice_index': self.slice_index,
            'x_index': self.x_index,
            'y_index': self.y_index,
            'start_x': self.start_x,
            'end_x': self.end_x,
            'start_y': self.start_y,
            'end_y': self.end_y,
            'width': self.width,
            'height': self.height,
            'is_left_column': self.is_left_column
        }