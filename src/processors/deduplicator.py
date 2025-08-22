"""
去重处理器模块
处理OCR结果和头像位置的去重
"""

import logging
from typing import List, Tuple, Dict, Any

from ..models.ocr_result import OCRItem, AvatarItem
from ..utils.config import Config
from ..utils.type_converter import get_box_bounds, convert_box_format

logger = logging.getLogger(__name__)


class Deduplicator:
    """去重处理器"""
    
    def __init__(self, config: Config):
        """
        初始化去重处理器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.ocr_iou_threshold = config.deduplication.ocr_iou_threshold
        self.avatar_iou_threshold = config.deduplication.avatar_iou_threshold
    
    def deduplicate(self, ocr_items: List[OCRItem], 
                   avatar_items: List[AvatarItem]) -> Tuple[List[OCRItem], List[AvatarItem]]:
        """
        对OCR结果和头像位置进行去重
        
        Args:
            ocr_items: OCR识别项列表
            avatar_items: 头像项列表
            
        Returns:
            (去重后的OCR项列表, 去重后的头像项列表)
        """
        logger.info("开始去重处理...")
        
        # OCR去重
        deduplicated_ocr = self._deduplicate_ocr(ocr_items)
        logger.info(f"OCR去重: {len(ocr_items)} -> {len(deduplicated_ocr)}")
        
        # 头像去重
        deduplicated_avatars = self._deduplicate_avatars(avatar_items)
        logger.info(f"头像去重: {len(avatar_items)} -> {len(deduplicated_avatars)}")
        
        return deduplicated_ocr, deduplicated_avatars
    
    def _deduplicate_ocr(self, ocr_items: List[OCRItem]) -> List[OCRItem]:
        """
        OCR结果去重
        
        Args:
            ocr_items: OCR项列表
            
        Returns:
            去重后的OCR项列表
        """
        if not ocr_items:
            return []
        
        deduplicated = []
        
        for current_item in ocr_items:
            is_duplicate = False
            
            # 与已添加的项比较
            for i, existing_item in enumerate(deduplicated):
                iou = self._calculate_box_iou(current_item.box, existing_item.box)
                
                if iou > self.ocr_iou_threshold:
                    is_duplicate = True
                    # 保留置信度更高的结果
                    if current_item.score > existing_item.score:
                        deduplicated[i] = current_item
                        logger.debug(f"替换重复OCR (IoU={iou:.3f}): "
                                   f"'{existing_item.text}' -> '{current_item.text}'")
                    else:
                        logger.debug(f"跳过重复OCR (IoU={iou:.3f}): '{current_item.text}'")
                    break
            
            if not is_duplicate:
                deduplicated.append(current_item)
        
        return deduplicated
    
    def _deduplicate_avatars(self, avatar_items: List[AvatarItem]) -> List[AvatarItem]:
        """
        头像位置去重
        
        Args:
            avatar_items: 头像项列表
            
        Returns:
            去重后的头像项列表
        """
        if not avatar_items:
            return []
        
        deduplicated = []
        
        for current_item in avatar_items:
            is_duplicate = False
            
            # 检查是否与已存在的头像重复
            for i, existing_item in enumerate(deduplicated):
                iou = self._calculate_box_iou(
                    convert_box_format(current_item.box, 'points'),
                    convert_box_format(existing_item.box, 'points')
                )
                
                if iou > self.avatar_iou_threshold:
                    is_duplicate = True
                    
                    # 保留面积更大的头像
                    if current_item.area > existing_item.area:
                        deduplicated[i] = current_item
                        logger.debug(f"替换重复头像 (IoU={iou:.3f}): "
                                   f"slice_{existing_item.slice_index} -> "
                                   f"slice_{current_item.slice_index}")
                    else:
                        logger.debug(f"跳过重复头像 (IoU={iou:.3f}): "
                                   f"slice_{current_item.slice_index}")
                    break
            
            if not is_duplicate:
                deduplicated.append(current_item)
        
        return deduplicated
    
    def _calculate_box_iou(self, box1: Any, box2: Any) -> float:
        """
        计算两个矩形框的IoU
        
        Args:
            box1, box2: 边界框
            
        Returns:
            IoU值 (0-1)
        """
        try:
            # 获取边界值
            b1 = get_box_bounds(box1)
            b2 = get_box_bounds(box2)
            
            # 计算交集
            inter_x_min = max(b1['min_x'], b2['min_x'])
            inter_y_min = max(b1['min_y'], b2['min_y'])
            inter_x_max = min(b1['max_x'], b2['max_x'])
            inter_y_max = min(b1['max_y'], b2['max_y'])
            
            if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
                return 0.0
            
            inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
            
            # 计算并集
            area1 = (b1['max_x'] - b1['min_x']) * (b1['max_y'] - b1['min_y'])
            area2 = (b2['max_x'] - b2['min_x']) * (b2['max_y'] - b2['min_y'])
            union_area = area1 + area2 - inter_area
            
            return inter_area / union_area if union_area > 0 else 0.0
            
        except Exception as e:
            logger.error(f"计算IoU时出错: {e}")
            return 0.0