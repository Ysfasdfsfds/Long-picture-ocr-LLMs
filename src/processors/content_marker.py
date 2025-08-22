"""
内容标记处理器模块
负责标记OCR识别结果的类型（时间、昵称、内容等）
"""

import cv2
import numpy as np
import re
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from ..models.ocr_result import OCRItem, AvatarItem
from ..models.chat_message import ChatMessage, MessageType
from ..utils.config import Config
from ..utils.type_converter import get_box_bounds

logger = logging.getLogger(__name__)


class ContentMarker:
    """内容标记处理器"""
    
    def __init__(self, config: Config):
        """
        初始化内容标记器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.time_patterns = config.get_time_patterns()
        self.system_message_patterns = config.get_system_message_patterns()
        self.time_exclude_keywords = config.get_time_exclude_keywords()
        self.feishu_keywords = config.get_feishu_keywords()
        
        # 颜色检测相关配置
        self.green_hsv_lower = np.array(config.content_marking.green_hsv_lower)
        self.green_hsv_upper = np.array(config.content_marking.green_hsv_upper)
        
        self.green_ratio_threshold = config.content_marking.green_ratio_threshold
        
        self.blue_hsv_lower = np.array(config.content_marking.blue_hsv_lower)
        self.blue_hsv_upper = np.array(config.content_marking.blue_hsv_upper)
        self.blue_ratio_threshold = config.content_marking.blue_ratio_threshold
        
        self.white_hsv_lower = np.array(config.content_marking.white_hsv_lower)
        self.white_hsv_upper = np.array(config.content_marking.white_hsv_upper)
        self.white_ratio_threshold = config.content_marking.white_ratio_threshold
    
    def mark_content(self, ocr_items: List[OCRItem], avatar_items: List[AvatarItem], 
                    original_image: np.ndarray) -> List[OCRItem]:
        """
        标记内容主入口
        
        Args:
            ocr_items: OCR识别项列表
            avatar_items: 头像项列表
            original_image: 原始图像
            
        Returns:
            标记后的OCR项列表
        """
        logger.info("开始内容标记...")
        
        # 创建工作副本
        marked_items = []
        for item in ocr_items:
            marked_item = OCRItem(
                text=item.text,
                box=item.box,
                score=item.score,
                slice_index=item.slice_index,
                original_text=item.text,  # 保存原始文本
                is_virtual=item.is_virtual
            )
            marked_items.append(marked_item)
        
        # 排序 - 使用中心Y坐标，与原版保持一致
        marked_items.sort(key=lambda x: x.get_center_y())
        avatar_items.sort(key=lambda x: x.center_y)
        
        # 判断是否为飞书截图并详细打印
        is_feishu = self._detect_and_print_platform_type(marked_items)
        
        # 1. 标记时间
        self._mark_time_content(marked_items)
        
        # 2. 标记系统消息（仅飞书）
        if is_feishu:
            self._mark_system_messages(marked_items)
        
        # 3. 基于头像位置标记昵称和内容
        print(f"🔧 使用{'飞书专用' if is_feishu else '通用聊天'}处理模式")
        if is_feishu:
            logger.info("🔧 启用飞书专用处理模式")
            self._mark_nickname_and_content_feishu(marked_items, avatar_items, original_image)
        else:
            logger.info("🔧 启用通用聊天处理模式")
            self._mark_nickname_and_content_wechat(marked_items, avatar_items, original_image)
        
        # 4. 标记"我的内容"（基于颜色检测）
        if not is_feishu:  # 飞书不需要颜色检测
            print("🎨 启用颜色检测标记我的内容")
            logger.info("🎨 非飞书平台，启用颜色检测")
            self._mark_my_content(marked_items, avatar_items, original_image)
        else:
            print("⏭️ 飞书平台跳过颜色检测")
            logger.info("⏭️ 飞书平台，跳过颜色检测")
        
        logger.info("内容标记完成")
        return marked_items
    
    def _detect_and_print_platform_type(self, ocr_items: List[OCRItem]) -> bool:
        """检测并详细打印平台类型"""
        print("\n" + "-"*50)
        print("🔍 内容标记器 - 平台类型检测")
        print("-"*50)
        
        detected_keywords = set()
        keyword_positions = []
        
        # 收集所有文本用于分析
        all_texts = [item.text for item in ocr_items]
        print(f"📝 已识别文本数量: {len(all_texts)}")
        
        # 检测飞书关键词
        for i, item in enumerate(ocr_items):
            text = item.text
            for keyword in self.feishu_keywords:
                if keyword in text:
                    detected_keywords.add(keyword)
                    keyword_positions.append({
                        'keyword': keyword,
                        'text': text,
                        'position': i
                    })
        
        is_feishu = len(detected_keywords) == len(self.feishu_keywords)
        
        # 详细打印检测结果
        print(f"🔑 飞书关键词检测 ({len(detected_keywords)}/{len(self.feishu_keywords)}):")
        
        for keyword in self.feishu_keywords:
            if keyword in detected_keywords:
                print(f"  ✅ '{keyword}' - 已找到")
            else:
                print(f"  ❌ '{keyword}' - 未找到")
        
        if keyword_positions:
            print(f"\n📍 关键词出现位置:")
            for pos_info in keyword_positions[:3]:  # 只显示前3个
                print(f"  - '{pos_info['keyword']}' 在文本: '{pos_info['text'][:30]}...'")
        
        # 输出最终判断
        if is_feishu:
            print(f"\n🏆 判断结果: 飞书截图")
            print(f"  ✓ 所有必需关键词已检测到")
            logger.info("✅ 内容标记器确认: 飞书截图")
        else:
            missing_keywords = set(self.feishu_keywords) - detected_keywords
            print(f"\n📱 判断结果: 微信/蓝信/钉钉")
            print(f"  ℹ️  缺失关键词: {list(missing_keywords)}")
            logger.info(f"ℹ️  内容标记器确认: 非飞书截图 (缺失: {missing_keywords})")
        
        print("-"*50 + "\n")
        return is_feishu
    
    def _is_feishu_screenshot(self, ocr_items: List[OCRItem]) -> bool:
        """判断是否为飞书截图（保留原方法供其他地方调用）"""
        detected_keywords = set()
        
        for item in ocr_items:
            text = item.text
            for keyword in self.feishu_keywords:
                if keyword in text:
                    detected_keywords.add(keyword)
        
        return len(detected_keywords) == len(self.feishu_keywords)
    
    def _mark_time_content(self, ocr_items: List[OCRItem]):
        """标记时间内容"""
        logger.debug("开始标记时间...")
        marked_count = 0
        
        for item in ocr_items:
            text = item.text.strip()
            
            # 排除过长的文本
            if len(text) > 30:
                continue
            
            # 排除包含非时间关键词的文本
            if any(keyword in text for keyword in self.time_exclude_keywords):
                continue
            
            # 检查时间模式
            is_time = False
            for pattern in self.time_patterns:
                if re.search(pattern, text):
                    match = re.search(pattern, text)
                    if match:
                        matched_length = len(match.group())
                        match_ratio = matched_length / len(text)
                        
                        # 对于复合时间格式，降低阈值要求
                        if (pattern.startswith('(昨天|今天|前天|明天)') or 
                            pattern.startswith('(上午|下午|早上|中午|晚上|凌晨)')):
                            if match_ratio >= 0.4:
                                is_time = True
                                break
                        elif pattern.startswith(r'\d{4}年'):
                            if match_ratio >= 0.7:
                                is_time = True
                                break
                        else:
                            if match_ratio >= 0.6:
                                is_time = True
                                break
            
            if is_time:
                item.text = text + "(时间)"
                marked_count += 1
        
        logger.debug(f"标记了 {marked_count} 个时间")
    
    def _mark_system_messages(self, ocr_items: List[OCRItem]):
        """标记系统消息"""
        logger.debug("开始标记系统消息...")
        marked_count = 0
        
        for item in ocr_items:
            text = item.text
            
            # 跳过已经标记过的内容
            if any(tag in text for tag in ['(时间)', '(昵称)', '(内容)', '(我的内容)', '(系统消息)']):
                continue
            
            # 检查是否匹配系统消息模式
            for pattern in self.system_message_patterns:
                if re.search(pattern, text):
                    item.text = text + "(系统消息)"
                    marked_count += 1
                    break
        
        logger.debug(f"标记了 {marked_count} 个系统消息")
    
    def _mark_nickname_and_content_wechat(self, ocr_items: List[OCRItem], 
                                         avatar_items: List[AvatarItem], 
                                         original_image: np.ndarray):
        """微信等平台的昵称和内容标记"""
        logger.debug("使用微信模式标记昵称和内容...")
        
        # 特殊处理：如果没有头像，将第一个非时间的文本标记为昵称
        if len(avatar_items) == 0:
            logger.debug("没有检测到头像，将第一个非时间文本标记为昵称")
            for ocr_item in ocr_items:
                if "(时间)" not in ocr_item.text and "(昵称)" not in ocr_item.text:
                    ocr_item.text = ocr_item.text + "(昵称)"
                    logger.debug(f"标记第一个非时间文本为昵称: {ocr_item.text}")
                    break
            # 如果没有头像，直接返回，不插入虚拟昵称
            return
        
        virtual_nicknames_to_insert = []
        
        for i, avatar in enumerate(avatar_items):
            # 寻找在当前头像Y范围内的文本作为昵称
            nickname_found = False
            
            for j, ocr_item in enumerate(ocr_items):
                if "(时间)" in ocr_item.text:
                    continue
                
                box_y_min = ocr_item.get_min_y()
                
                # 检查是否在头像Y范围内
                if avatar.y <= box_y_min <= avatar.y + avatar.height:
                    # 额外检查：避免将内容误标记为昵称
                    if self._is_likely_nickname(ocr_item.text):
                        ocr_item.text = ocr_item.text + "(昵称)"
                        nickname_found = True
                        break
            
            # 如果没有找到昵称，创建虚拟昵称
            if not nickname_found:
                insert_index = self._find_insert_index(ocr_items, avatar.y + avatar.height)
                
                virtual_nickname = OCRItem(
                    text=f"未知用户{i+1}(昵称)",
                    box=[[avatar.x, avatar.y], 
                         [avatar.x + avatar.width, avatar.y],
                         [avatar.x + avatar.width, avatar.y + avatar.height],
                         [avatar.x, avatar.y + avatar.height]],
                    score=0.0,
                    slice_index=avatar.slice_index,
                    is_virtual=True
                )
                
                virtual_nicknames_to_insert.append((insert_index, virtual_nickname))
        
        # 插入虚拟昵称
        for insert_index, virtual_nickname in sorted(virtual_nicknames_to_insert, 
                                                    key=lambda x: x[0], reverse=True):
            ocr_items.insert(insert_index, virtual_nickname)
        
        # 标记内容
        self._mark_content_after_nicknames(ocr_items, avatar_items)
    
    def _mark_nickname_and_content_feishu(self, ocr_items: List[OCRItem], 
                                         avatar_items: List[AvatarItem], 
                                         original_image: np.ndarray):
        """飞书的昵称和内容标记"""
        logger.debug("使用飞书模式标记昵称和内容...")
        
        # 过滤小面积头像
        filtered_avatars = self._filter_avatars_by_area(avatar_items)
        
        nickname_operations = []
        virtual_nicknames_to_insert = []
        
        for i, avatar in enumerate(filtered_avatars):
            # 寻找在当前头像Y范围内的文本作为昵称
            nickname_texts = []
            nickname_indices = []
            
            for j, ocr_item in enumerate(ocr_items):
                if "(时间)" in ocr_item.text or "(系统消息)" in ocr_item.text:
                    continue
                
                box_y_min = ocr_item.get_min_y()
                
                # 检查是否在头像Y范围内
                if avatar.y <= box_y_min <= avatar.y + avatar.height:
                    nickname_texts.append(ocr_item.text)
                    nickname_indices.append(j)
            
            if nickname_indices:
                # 记录昵称合并操作
                nickname_operations.append({
                    'avatar_index': i,
                    'indices': nickname_indices,
                    'texts': nickname_texts,
                    'merged_text': " ".join(nickname_texts) + "(昵称)"
                })
            else:
                # 创建虚拟昵称 - 使用头像索引(i+1)而不是全局计数器，与原版保持一致
                insert_index = self._find_insert_index(ocr_items, avatar.y + avatar.height)
                
                virtual_nickname = OCRItem(
                    text=f"未知用户{i+1}(昵称)",
                    box=[[avatar.x, avatar.y],
                         [avatar.x + avatar.width, avatar.y],
                         [avatar.x + avatar.width, avatar.y + avatar.height],
                         [avatar.x, avatar.y + avatar.height]],
                    score=0.0,
                    slice_index=avatar.slice_index,
                    is_virtual=True
                )
                
                virtual_nicknames_to_insert.append({
                    'avatar_index': i,
                    'insert_index': insert_index,
                    'virtual_nickname': virtual_nickname
                })
        
        # 执行昵称合并
        for operation in reversed(nickname_operations):
            indices = operation['indices']
            merged_text = operation['merged_text']
            
            # 更新第一个昵称项
            first_index = indices[0]
            ocr_items[first_index].text = merged_text
            
            # 删除多余的昵称项
            for idx in sorted(indices[1:], reverse=True):
                del ocr_items[idx]
        
        # 重新计算虚拟昵称的插入位置
        for virtual_item in virtual_nicknames_to_insert:
            avatar_index = virtual_item['avatar_index']
            avatar = filtered_avatars[avatar_index]
            
            insert_index = self._find_insert_index(ocr_items, avatar.y + avatar.height)
            virtual_item['insert_index'] = insert_index
        
        # 插入虚拟昵称
        virtual_nicknames_to_insert.sort(key=lambda x: x['insert_index'], reverse=True)
        for virtual_item in virtual_nicknames_to_insert:
            insert_index = virtual_item['insert_index']
            virtual_nickname = virtual_item['virtual_nickname']
            ocr_items.insert(insert_index, virtual_nickname)
        
        # 标记内容
        self._mark_content_after_nicknames(ocr_items, filtered_avatars)
    
    def _mark_content_after_nicknames(self, ocr_items: List[OCRItem], 
                                     avatar_items: List[AvatarItem]):
        """在昵称后标记内容"""
        for i, avatar in enumerate(avatar_items):
            # 找到下一个头像的Y位置边界
            next_boundary = avatar_items[i+1].y if i+1 < len(avatar_items) else float('inf')
            
            # 找到对应的昵称
            nickname_index = -1
            for j, ocr_item in enumerate(ocr_items):
                if "(昵称)" in ocr_item.text:
                    box_y_min = ocr_item.get_min_y()
                    if avatar.y <= box_y_min <= avatar.y + avatar.height:
                        nickname_index = j
                        break
            
            if nickname_index >= 0:
                # 标记该昵称后的内容
                for k in range(nickname_index + 1, len(ocr_items)):
                    next_ocr = ocr_items[k]
                    if any(tag in next_ocr.text for tag in 
                          ['(时间)', '(昵称)', '(系统消息)']):
                        continue
                    
                    next_box_y_min = next_ocr.get_min_y()
                    
                    # 检查是否在当前头像区域内且未到达下一个头像边界
                    # 使用与原版相同的条件：内容Y坐标大于头像Y最小值即可
                    if next_box_y_min > avatar.y and next_box_y_min < next_boundary:
                        if "(内容)" not in next_ocr.text:
                            next_ocr.text = next_ocr.text + "(内容)"
                    elif next_box_y_min >= next_boundary:
                        break
    
    def _mark_my_content(self, ocr_items: List[OCRItem], avatar_items: List[AvatarItem],
                        original_image: np.ndarray):
        """标记"我的内容"（基于颜色检测）"""
        if original_image is None:
            logger.warning("原图不可用，跳过颜色内容检测")
            return
        
        logger.debug("开始基于颜色检测标记我的内容...")
        
        # 第一轮：基于颜色检测
        my_content_boxes = []
        
        for i, item in enumerate(ocr_items):
            if "(内容)" in item.text:
                box = item.box
                
                # 检测绿色和蓝色背景
                is_green = self._detect_green_content_box(original_image, box)
                is_blue = self._detect_blue_content_box(original_image, box)
                
                if is_green or is_blue:
                    item.text = item.text.replace("(内容)", "(我的内容)")
                    my_content_boxes.append({'index': i, 'box': box})
                    reason = "绿色背景" if is_green else "蓝色背景"
                    logger.debug(f"标记为我的内容: {item.text} (原因: {reason})")
        
        # 第二轮：基于位置推理
        self._mark_adjacent_my_content(ocr_items, my_content_boxes, avatar_items, original_image)
    
    def _detect_green_content_box(self, image: np.ndarray, box: List[List[float]]) -> bool:
        """检测文本框区域是否为绿色背景"""
        try:
            # 获取文本框区域
            bounds = get_box_bounds(box)
            min_x = max(0, int(bounds['min_x']))
            max_x = min(image.shape[1], int(bounds['max_x']))
            min_y = max(0, int(bounds['min_y']))
            max_y = min(image.shape[0], int(bounds['max_y']))
            
            if max_x <= min_x or max_y <= min_y:
                return False
            
            # 提取区域图像
            roi = image[min_y:max_y, min_x:max_x]
            
            if roi.size == 0:
                return False
            
            # 转换为HSV颜色空间
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # 创建绿色掩码
            mask = cv2.inRange(hsv, self.green_hsv_lower, self.green_hsv_upper)
            
            # 计算绿色像素的比例
            green_pixels = cv2.countNonZero(mask)
            total_pixels = roi.shape[0] * roi.shape[1]
            
            if total_pixels > 0:
                green_ratio = green_pixels / total_pixels
                return green_ratio > self.green_ratio_threshold
            
            return False
            
        except Exception as e:
            logger.error(f"检测绿色框时出错: {e}")
            return False
    
    def _detect_blue_content_box(self, image: np.ndarray, box: List[List[float]]) -> bool:
        """检测文本框区域是否为蓝色背景"""
        try:
            # 获取文本框区域
            bounds = get_box_bounds(box)
            min_x = max(0, int(bounds['min_x']))
            max_x = min(image.shape[1], int(bounds['max_x']))
            min_y = max(0, int(bounds['min_y']))
            max_y = min(image.shape[0], int(bounds['max_y']))
            
            if max_x <= min_x or max_y <= min_y:
                return False
            
            # 提取区域图像
            roi = image[min_y:max_y, min_x:max_x]
            
            if roi.size == 0:
                return False
            
            # 转换为HSV颜色空间
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # 创建蓝色掩码
            blue_mask = cv2.inRange(hsv, self.blue_hsv_lower, self.blue_hsv_upper)
            
            # 检测白色背景
            white_mask = cv2.inRange(hsv, self.white_hsv_lower, self.white_hsv_upper)
            
            # 计算像素比例
            blue_pixels = cv2.countNonZero(blue_mask)
            white_pixels = cv2.countNonZero(white_mask)
            total_pixels = roi.shape[0] * roi.shape[1]
            
            if total_pixels > 0:
                blue_ratio = blue_pixels / total_pixels
                white_ratio = white_pixels / total_pixels
                
                # 判断逻辑
                is_blue_background = (blue_ratio > self.blue_ratio_threshold and 
                                    white_ratio < self.white_ratio_threshold and 
                                    blue_ratio > white_ratio)
                
                return is_blue_background
            
            return False
            
        except Exception as e:
            logger.error(f"检测蓝色框时出错: {e}")
            return False
    
    def _mark_adjacent_my_content(self, ocr_items: List[OCRItem], 
                                 my_content_boxes: List[Dict],
                                 avatar_items: List[AvatarItem], 
                                 original_image: np.ndarray):
        """基于位置推理标记相邻的我的内容"""
        logger.debug(f"开始位置推理：有 {len(my_content_boxes)} 个我的内容框")
        
        if not my_content_boxes or not avatar_items:
            return
        
        for my_content in my_content_boxes:
            my_index = my_content['index']
            my_box = my_content['box']
            my_y_max = get_box_bounds(my_box)['max_y']
            
            # 查找下一条内容
            for next_index in range(my_index + 1, len(ocr_items)):
                next_item = ocr_items[next_index]
                
                # 跳过已经标记为我的内容的
                if "(我的内容)" in next_item.text:
                    continue
                
                # 只处理内容标记
                if "(内容)" not in next_item.text:
                    continue
                
                next_box = next_item.box
                next_y_min = get_box_bounds(next_box)['min_y']
                
                # 检查是否已通过颜色检测
                is_green = self._detect_green_content_box(original_image, next_box)
                is_blue = self._detect_blue_content_box(original_image, next_box)
                if is_green or is_blue:
                    continue
                
                # 检查位置条件
                if self._is_adjacent_my_content(my_box, next_box, avatar_items):
                    next_item.text = next_item.text.replace("(内容)", "(我的内容)")
                    logger.debug(f"基于位置推理标记为我的内容: {next_item.text}")
                    # 将新标记的内容也加入列表
                    my_content_boxes.append({'index': next_index, 'box': next_box})
    
    def _is_adjacent_my_content(self, my_box: List[List[float]], 
                               next_box: List[List[float]], 
                               avatar_items: List[AvatarItem]) -> bool:
        """检查下一条内容是否应该标记为我的内容"""
        my_bounds = get_box_bounds(my_box)
        next_bounds = get_box_bounds(next_box)
        
        # 检查是否在最近的两个头像框之间
        return self._is_between_avatars(my_bounds['max_y'], next_bounds['min_y'], avatar_items)
    
    def _is_between_avatars(self, start_y: float, end_y: float, 
                           avatar_items: List[AvatarItem]) -> bool:
        """检查Y坐标范围是否在最近的两个头像框之间"""
        if not avatar_items:
            return True
        
        # 找到包含start_y的头像对
        for i in range(len(avatar_items) - 1):
            avatar1 = avatar_items[i]
            avatar2 = avatar_items[i + 1]
            
            avatar1_y_max = avatar1.y + avatar1.height
            avatar2_y_min = avatar2.y
            
            # 检查是否在这两个头像之间
            if avatar1_y_max <= start_y and end_y <= avatar2_y_min:
                return True
        
        return False
    
    def _filter_avatars_by_area(self, avatar_items: List[AvatarItem]) -> List[AvatarItem]:
        """过滤小面积头像（飞书专用）"""
        if not avatar_items:
            return []
        
        # 计算面积均值
        areas = [item.area for item in avatar_items]
        mean_area = sum(areas) / len(areas)
        
        # 筛选面积大于等于均值的头像
        filtered = [item for item in avatar_items if item.area >= mean_area]
        
        logger.debug(f"头像过滤: {len(avatar_items)} -> {len(filtered)}")
        return filtered
    
    def _is_likely_nickname(self, text: str) -> bool:
        """判断文本是否可能是昵称"""
        # 去除已有标记
        clean_text = text.replace("(昵称)", "").replace("(内容)", "").replace("(时间)", "").strip()
        
        # 昵称通常不会太长
        if len(clean_text) > 20:
            return False
        
        # 如果包含句号、逗号等标点，可能是内容
        content_punctuation = ['。', '，', '！', '？', '；', '：', '、']
        if any(p in clean_text for p in content_punctuation):
            return False
        
        # 如果包含完整句子的特征（如"是"、"在"、"了"等）
        sentence_indicators = ['是', '在', '了', '的', '有', '不是', '这是']
        if any(word in clean_text for word in sentence_indicators) and len(clean_text) > 10:
            return False
        
        # 如果开头是动词或否定词，可能是内容
        content_starters = ['不是', '没有', '可以', '应该', '需要', '是的', '好的']
        if any(clean_text.startswith(starter) for starter in content_starters):
            return False
        
        return True
    
    def _find_insert_index(self, ocr_items: List[OCRItem], y_threshold: float) -> int:
        """找到在OCR列表中的插入位置"""
        # 找到第一个Y坐标大于阈值的位置
        for idx, item in enumerate(ocr_items):
            item_y_min = item.get_min_y()
            if item_y_min > y_threshold:
                return idx
        return len(ocr_items)