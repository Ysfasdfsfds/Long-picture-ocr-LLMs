"""
OCR结果数据模型
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any


@dataclass
class OCRItem:
    """单个OCR识别项"""
    text: str
    box: List[List[float]]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    score: float
    slice_index: int
    original_text: Optional[str] = None  # 标记前的原始文本
    is_virtual: bool = False  # 是否为虚拟添加的项（如虚拟昵称）
    
    def get_center_y(self) -> float:
        """获取中心Y坐标"""
        return sum(pt[1] for pt in self.box) / 4
    
    def get_min_y(self) -> float:
        """获取最小Y坐标（为了与原版兼容，返回左下角Y坐标）"""
        # 原版get_box_y_min返回的是box[3][1]（左下角Y坐标）
        # 这不是真正的最小Y，但为了保持兼容性，我们使用相同的逻辑
        return self.box[3][1]
    
    def get_max_y(self) -> float:
        """获取最大Y坐标"""
        return max(pt[1] for pt in self.box)
    
    def get_min_x(self) -> float:
        """获取最小X坐标"""
        return min(pt[0] for pt in self.box)
    
    def get_max_x(self) -> float:
        """获取最大X坐标"""
        return max(pt[0] for pt in self.box)
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            'text': self.text,
            'box': self.box,
            'score': self.score,
            'slice_index': self.slice_index,
            'original_text': self.original_text,
            'virtual': self.is_virtual
        }


@dataclass
class AvatarItem:
    """头像检测项"""
    box: tuple  # (x, y, w, h)
    slice_index: int
    
    @property
    def x(self) -> float:
        return self.box[0]
    
    @property
    def y(self) -> float:
        return self.box[1]
    
    @property
    def width(self) -> float:
        return self.box[2]
    
    @property
    def height(self) -> float:
        return self.box[3]
    
    @property
    def center_x(self) -> float:
        return self.x + self.width / 2
    
    @property
    def center_y(self) -> float:
        return self.y + self.height / 2
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            'box': self.box,
            'slice_index': self.slice_index,
            'center_x': self.center_x,
            'center_y': self.center_y
        }


@dataclass
class SliceOCRResult:
    """切片OCR处理结果"""
    slice_index: int
    start_y: int
    end_y: int
    ocr_items: List[OCRItem] = field(default_factory=list)
    avatar_items: List[AvatarItem] = field(default_factory=list)
    
    def add_ocr_item(self, text: str, box: List[List[float]], score: float, 
                     is_virtual: bool = False):
        """添加OCR项"""
        item = OCRItem(
            text=text,
            box=box,
            score=score,
            slice_index=self.slice_index,
            is_virtual=is_virtual
        )
        self.ocr_items.append(item)
        return item
    
    def add_avatar_item(self, box: tuple):
        """添加头像项"""
        item = AvatarItem(box=box, slice_index=self.slice_index)
        self.avatar_items.append(item)
        return item
    
    def sort_by_y(self):
        """按Y坐标排序 - 使用中心Y坐标，与原版保持一致"""
        self.ocr_items.sort(key=lambda x: x.get_center_y())
        self.avatar_items.sort(key=lambda x: x.center_y)
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            'slice_index': self.slice_index,
            'start_y': self.start_y,
            'end_y': self.end_y,
            'ocr_items': [item.to_dict() for item in self.ocr_items],
            'avatar_items': [item.to_dict() for item in self.avatar_items]
        }