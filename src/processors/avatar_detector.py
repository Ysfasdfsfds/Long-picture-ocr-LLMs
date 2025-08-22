"""
头像检测器模块
负责检测聊天界面中的头像位置
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict
from pathlib import Path

from ..models.slice_info import SliceInfo
from ..models.ocr_result import AvatarItem, SliceOCRResult
from ..utils.config import Config

logger = logging.getLogger(__name__)


class AvatarDetector:
    """头像检测器"""
    
    def __init__(self, config: Config):
        """
        初始化头像检测器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.debug_dir = Path(config.output.debug_dir)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        
        # 头像检测参数
        self.binary_threshold = config.image.binary_threshold
        self.gaussian_blur_size = config.image.gaussian_blur_size
        self.square_ratio_min = config.avatar.square_ratio_min
        self.square_ratio_max = config.avatar.square_ratio_max
        self.strict_square_ratio_min = config.avatar.strict_square_ratio_min
        self.strict_square_ratio_max = config.avatar.strict_square_ratio_max
        self.iou_threshold = config.avatar.iou_threshold
        self.x_crop_offset = config.avatar.x_crop_offset
        
        # 存储每个切片的x_crop值
        self.slice_x_crop_values: Dict[int, Optional[Tuple]] = {}
    
    def detect_avatars(self, slice_info: SliceInfo, x_crop: Optional[int] = None) -> List[AvatarItem]:
        """
        检测切片中的头像
        
        Args:
            slice_info: 切片信息
            x_crop: X方向裁剪值（如果为None，则自动计算）
            
        Returns:
            检测到的头像列表
        """
        logger.info(f"开始检测切片 {slice_info.slice_index} 的头像...")
        
        # 如果提供了x_crop，使用它；否则使用原图
        if x_crop is not None:
            slice_img = slice_info.image[0:slice_info.image.shape[0], 0:x_crop]
            logger.debug(f"使用x_crop={x_crop}进行裁剪")
        else:
            slice_img = slice_info.image
            logger.debug("未进行x裁剪，使用原始图像")
        
        # 预处理图像
        binary = self._preprocess_image(slice_img)
        
        # 提取轮廓和外接矩形
        rects = self._extract_contours_and_rects(binary, slice_img)
        
        # 应用NMS去除重叠框
        nms_rects = self._apply_nms(rects)
        
        # 合并相邻的框
        merge_threshold = self._calculate_merge_threshold(rects)
        merged_rects = self._merge_nearby_boxes(nms_rects, merge_threshold)
        
        # # 保存调试图像
        # if self.config.output.debug_dir:
        #     self._save_debug_image(slice_img, merged_rects, slice_info.slice_index)
        
        # 转换为AvatarItem对象
        avatar_items = []
        for rect in merged_rects:
            # 将Y坐标转换为原图坐标
            x, y, w, h = rect
            adjusted_rect = (x, y + slice_info.start_y, w, h)
            avatar_item = AvatarItem(box=adjusted_rect, slice_index=slice_info.slice_index)
            avatar_items.append(avatar_item)
        
        logger.info(f"切片 {slice_info.slice_index} 检测到 {len(avatar_items)} 个头像")
        return avatar_items
    
    def calculate_x_crop(self, slice_infos: List[SliceInfo]) -> Optional[int]:
        """
        基于多个切片计算合适的x_crop值
        
        Args:
            slice_infos: 切片信息列表
            
        Returns:
            计算得到的x_crop值
        """
        # 根据切片数量决定处理哪些切片
        slices_to_process = self._get_slices_to_process(slice_infos)
        
        # 处理选定的切片，收集target_box
        all_boxes = []
        for slice_info in slices_to_process:
            target_box = self._find_target_box_for_slice(slice_info)
            if target_box is not None:
                # 将坐标转换为原图坐标
                x, y, w, h = target_box
                adjusted_box = (x, y + slice_info.start_y, w, h, slice_info.slice_index)
                all_boxes.append(adjusted_box)
                self.slice_x_crop_values[slice_info.slice_index] = adjusted_box[:4]
        
        # 找到最合适的target_box
        selected_box = self._find_best_target_box(all_boxes)
        
        if selected_box:
            x, y, w, h, _ = selected_box
            x_crop = x + w + self.x_crop_offset
            logger.info(f"计算得到的x_crop值: {x_crop}")
            return x_crop
        else:
            logger.warning("未找到合适的target_box")
            return None
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """预处理图像：转灰度、模糊、二值化"""
        # 转灰度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊去噪
        blurred = cv2.GaussianBlur(gray, self.gaussian_blur_size, 0)
        
        # 二值化
        _, binary = cv2.threshold(blurred, self.binary_threshold, 255, cv2.THRESH_BINARY)
        
        # 反转颜色（头像通常是深色背景）
        binary = 255 - binary
        
        return binary
    
    def _extract_contours_and_rects(self, binary_img: np.ndarray, 
                                   original_img: np.ndarray) -> List[Tuple]:
        """提取轮廓并计算外接矩形"""
        # 查找轮廓
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 计算所有轮廓的外接矩形
        rects = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            rects.append((x, y, w, h))
        
        # 按面积从大到小排序
        rects = sorted(rects, key=lambda box: box[2] * box[3], reverse=True)
        
        return rects
    
    def _find_target_box_for_slice(self, slice_info: SliceInfo) -> Optional[Tuple]:
        """为单个切片找到target_box"""
        # 预处理图像
        binary = self._preprocess_image(slice_info.image)
        
        # 提取轮廓和外接矩形
        rects = self._extract_contours_and_rects(binary, slice_info.image)
        
        if not rects:
            return None
        
        # 使用改进的算法找到最合适的框
        target_box = self._select_target_box(rects)
        
        return target_box
    
    def _select_target_box(self, rects: List[Tuple]) -> Optional[Tuple]:
        """选择最合适的target_box"""
        if not rects:
            return None
        
        # 判断是否为严格正方形
        def is_strict_square(r):
            ratio = r[2] / r[3] if r[3] != 0 else 0
            return (self.strict_square_ratio_min <= ratio <= self.strict_square_ratio_max or 
                    self.strict_square_ratio_min <= (1/ratio) <= self.strict_square_ratio_max)
        
        # 获取前三个不同的x坐标值
        x_list = [r[0] for r in rects]
        unique_x = sorted(set(x_list))
        top_3_x = unique_x[:3]
        
        # 挑选出左侧前三的所有框
        left_top3_rects = [r for r in rects if r[0] in top_3_x]
        
        # 筛选掉不严格趋于正方形的框
        square_rects = [r for r in left_top3_rects if is_strict_square(r)]
        
        # 在剩余框中选择面积最大的作为target_box
        if square_rects:
            target_box = max(square_rects, key=lambda r: r[2] * r[3])
            return target_box
        else:
            # 降级处理：选择最左侧的框
            return min(rects, key=lambda r: r[0])
    
    def _find_best_target_box(self, all_boxes: List[Tuple]) -> Optional[Tuple]:
        """从所有框中找到最合适的目标框"""
        if not all_boxes:
            return None
        
        # 按x坐标排序
        all_boxes.sort(key=lambda box: box[0])
        
        # 对最左侧前20%的box进行操作
        left_20_percent_count = max(1, int(len(all_boxes) * 0.2))
        left_boxes = all_boxes[:left_20_percent_count]
        
        # 找到符合要求的框
        for box in left_boxes:
            x, y, w, h, slice_idx = box
            # 判断是否严格趋近于正方形
            aspect_ratio = w / h if h > 0 else 0
            is_square_like = self.strict_square_ratio_min <= aspect_ratio <= self.strict_square_ratio_max
            
            if is_square_like:
                return box
        
        return None
    
    def _get_slices_to_process(self, slice_infos: List[SliceInfo]) -> List[SliceInfo]:
        """根据切片数量决定处理策略，只选择左侧第一列切片"""
        # 首先筛选出左侧第一列的切片（x_index == 0）
        left_column_slices = [slice_info for slice_info in slice_infos if slice_info.is_left_column]
        
        if not left_column_slices:
            logger.warning("没有找到左侧第一列的切片，使用所有切片")
            left_column_slices = slice_infos
        
        logger.info(f"筛选出 {len(left_column_slices)} 个左侧第一列切片（共 {len(slice_infos)} 个切片）")
        
        # 对左侧第一列切片应用原有的y方向选择策略
        total_left_slices = len(left_column_slices)
        
        if total_left_slices == 1:
            selected_slices = left_column_slices
            logger.info("左侧第一列只有1个切片，处理所有")
        elif total_left_slices == 2:
            selected_slices = left_column_slices[:1]
            logger.info("左侧第一列有2个切片，只处理第一个")
        else:
            selected_slices = left_column_slices[1:-1]
            logger.info(f"左侧第一列有{total_left_slices}个切片，处理中间{len(selected_slices)}个")
        
        return selected_slices
    
    def _apply_nms(self, rects: List[Tuple]) -> List[Tuple]:
        """应用非最大抑制算法去除重叠框"""
        keep_rects = []
        for rect in rects:
            keep = True
            for kept_rect in keep_rects:
                if self._calculate_iou(rect, kept_rect) > self.iou_threshold:
                    keep = False
                    break
            if keep:
                keep_rects.append(rect)
        return keep_rects
    
    def _calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """计算两个框的IOU"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # 计算交集
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height
        
        # 计算并集
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0
        
        return inter_area / union_area
    
    def _calculate_merge_threshold(self, rects: List[Tuple]) -> float:
        """根据最大框计算合并阈值"""
        if rects:
            max_box = rects[0]  # 面积最大的框
            max_dim = max(max_box[2], max_box[3])  # 最长边
            merge_threshold = max_dim * self.config.image.merge_distance_factor
            return merge_threshold
        else:
            return 20
    
    def _merge_nearby_boxes(self, rects: List[Tuple], merge_threshold: float) -> List[Tuple]:
        """合并相邻的边界框"""
        merged_rects = []
        used = [False] * len(rects)
        
        for i in range(len(rects)):
            if used[i]:
                continue
            
            # 找到所有需要与当前框合并的框
            group = [rects[i]]
            used[i] = True
            
            for j in range(i + 1, len(rects)):
                if used[j]:
                    continue
                
                # 检查是否与组中任何一个框相邻
                should_add = False
                for box_in_group in group:
                    if self._should_merge(box_in_group, rects[j], merge_threshold):
                        should_add = True
                        break
                
                if should_add:
                    group.append(rects[j])
                    used[j] = True
            
            # 合并这一组框
            merged_box = self._merge_boxes(group)
            if merged_box:
                merged_rects.append(merged_box)
        
        return merged_rects
    
    def _should_merge(self, box1: Tuple, box2: Tuple, distance_threshold: float) -> bool:
        """判断两个框是否应该合并"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # 计算两个框中心点之间的距离
        center1_x, center1_y = x1 + w1 // 2, y1 + h1 // 2
        center2_x, center2_y = x2 + w2 // 2, y2 + h2 // 2
        distance = ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5
        
        # 如果距离小于阈值或IOU大于0.01，则合并
        return distance < distance_threshold or self._calculate_iou(box1, box2) > 0.01
    
    def _merge_boxes(self, boxes: List[Tuple]) -> Optional[Tuple]:
        """合并一组框为一个最小外接框"""
        if not boxes:
            return None
        
        min_x = min(box[0] for box in boxes)
        min_y = min(box[1] for box in boxes)
        max_x = max(box[0] + box[2] for box in boxes)
        max_y = max(box[1] + box[3] for box in boxes)
        
        return (min_x, min_y, max_x - min_x, max_y - min_y)
    
    def _save_debug_image(self, image: np.ndarray, rects: List[Tuple], slice_index: int):
        """保存调试图像"""
        debug_img = image.copy()
        for (x, y, w, h) in rects:
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        debug_path = self.debug_dir / f"avatars_slice_{slice_index}.jpg"
        cv2.imwrite(str(debug_path), debug_img)