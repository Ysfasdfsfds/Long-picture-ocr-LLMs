"""
OCR引擎封装模块
提供统一的OCR接口
"""

import cv2
import numpy as np
from typing import List, Optional, Dict, Any
from rapidocr import RapidOCR
import logging

from ..models.ocr_result import OCRItem, SliceOCRResult
from ..models.slice_info import SliceInfo
from ..utils.config import Config

logger = logging.getLogger(__name__)


class OCREngine:
    """OCR引擎封装类"""
    
    def __init__(self, config: Config):
        """
        初始化OCR引擎
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.engine = RapidOCR(config_path=config.ocr.config_path)
        self.text_score_threshold = config.ocr.text_score_threshold
    
    def process_slice(self, slice_info: SliceInfo, 
                     save_visualization: bool = True) -> SliceOCRResult:
        """
        处理单个切片的OCR识别
        
        Args:
            slice_info: 切片信息
            save_visualization: 是否保存可视化结果
            
        Returns:
            切片OCR结果
        """
        logger.info(f"处理切片 {slice_info.slice_index}...")
        
        # 创建结果对象
        result = SliceOCRResult(
            slice_index=slice_info.slice_index,
            start_y=slice_info.start_y,
            end_y=slice_info.end_y
        )
        
        # 进行OCR识别
        slice_img_rgb = cv2.cvtColor(slice_info.image, cv2.COLOR_BGR2RGB)
        ocr_result = self.engine(slice_img_rgb)
        
        # 保存可视化结果
        if save_visualization:
            vis_path = f"{self.config.output.output_images_dir}/slice_ocr_result_{slice_info.slice_index}.jpg"
            ocr_result.vis(vis_path)
        
        # 处理OCR结果
        if ocr_result.boxes is not None and ocr_result.txts is not None:
            self._process_ocr_items(ocr_result, slice_info, result)
        else:
            logger.warning(f"切片 {slice_info.slice_index} 未检测到文本")
        
        # 排序结果
        result.sort_by_y()
        
        logger.info(f"切片 {slice_info.slice_index} 处理完成，"
                   f"检测到 {len(result.ocr_items)} 个文本")
        
        return result
    
    def _process_ocr_items(self, ocr_result: Any, slice_info: SliceInfo, 
                          result: SliceOCRResult):
        """
        处理OCR识别项
        
        Args:
            ocr_result: RapidOCR的识别结果
            slice_info: 切片信息
            result: 切片OCR结果对象
        """
        for box, txt, score in zip(ocr_result.boxes, ocr_result.txts, ocr_result.scores):
            # 过滤低置信度结果
            if score < self.text_score_threshold:
                continue
            
            # 转换坐标到原图坐标系
            adjusted_box = self._adjust_box_to_original(box, slice_info.start_y)
            
            # 添加OCR项
            result.add_ocr_item(
                text=txt,
                box=adjusted_box,
                score=score
            )
        
        logger.debug(f"切片 {slice_info.slice_index} 过滤后保留 "
                    f"{len(result.ocr_items)} 个文本")
    
    def _adjust_box_to_original(self, box: List[List[float]], 
                               start_y: int) -> List[List[float]]:
        """
        将切片坐标转换为原图坐标
        
        Args:
            box: 切片中的边界框坐标
            start_y: 切片在原图中的起始Y坐标
            
        Returns:
            原图坐标系中的边界框
        """
        adjusted_box = []
        for point in box:
            adjusted_point = [point[0], point[1] + start_y]
            adjusted_box.append(adjusted_point)
        return adjusted_box
    
    def batch_process_slices(self, slice_infos: List[SliceInfo]) -> List[SliceOCRResult]:
        """
        批量处理多个切片
        
        Args:
            slice_infos: 切片信息列表
            
        Returns:
            切片OCR结果列表
        """
        results = []
        for slice_info in slice_infos:
            result = self.process_slice(slice_info)
            results.append(result)
        return results