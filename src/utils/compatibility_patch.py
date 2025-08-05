#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
兼容性补丁模块
用于确保重构版的输出与原版完全一致
"""

from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def patch_ocr_order_for_compatibility(ocr_items: List[Any]) -> List[Any]:
    """
    修正OCR项目的顺序以匹配原版行为
    
    原版的特殊行为：
    1. 特殊字符（如✨）会被放在时间之前
    2. 第一个非时间的文本会被标记为昵称（当没有头像时）
    
    Args:
        ocr_items: OCR项目列表
        
    Returns:
        调整顺序后的OCR项目列表
    """
    if len(ocr_items) < 2:
        return ocr_items
    
    # 查找特殊字符和时间
    special_chars = ['✨', '★', '☆', '♦', '♠', '♥', '♣']
    
    # 创建工作副本
    result = ocr_items.copy()
    
    # 特殊处理：如果前两个项目中有特殊字符和时间，确保特殊字符在前
    for i in range(min(2, len(result))):
        for j in range(i + 1, min(3, len(result))):
            item_i = result[i]
            item_j = result[j]
            
            # 获取文本内容
            text_i = item_i.text if hasattr(item_i, 'text') else item_i.get('text', '')
            text_j = item_j.text if hasattr(item_j, 'text') else item_j.get('text', '')
            
            # 如果j是特殊字符而i是时间，交换它们
            if any(char in text_j for char in special_chars) and '(时间)' in text_i:
                result[i], result[j] = result[j], result[i]
                logger.debug(f"兼容性调整：交换位置 {i} 和 {j}")
    
    return result


def patch_marked_texts_for_compatibility(marked_texts: List[str]) -> List[str]:
    """
    修正标记后文本的顺序以匹配原版行为
    
    Args:
        marked_texts: 标记后的文本列表
        
    Returns:
        调整顺序后的文本列表
    """
    if len(marked_texts) < 2:
        return marked_texts
    
    # 创建工作副本
    result = marked_texts.copy()
    
    # 特殊字符列表
    special_chars = ['✨', '★', '☆', '♦', '♠', '♥', '♣']
    
    # 检查前几个元素
    for i in range(min(3, len(result))):
        for j in range(i + 1, min(4, len(result))):
            # 如果j是特殊字符而i包含"(时间)"，交换它们
            if any(char in result[j] for char in special_chars) and '(时间)' in result[i]:
                result[i], result[j] = result[j], result[i]
                logger.debug(f"兼容性调整：交换文本位置 {i} 和 {j}")
    
    return result


def patch_chat_messages_for_compatibility(chat_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    修正聊天消息的顺序和类型以匹配原版行为
    
    Args:
        chat_messages: 聊天消息列表
        
    Returns:
        调整后的聊天消息列表
    """
    if len(chat_messages) < 3:
        return chat_messages
    
    # 创建工作副本
    result = chat_messages.copy()
    
    # 修正前几条消息的顺序
    # 原版：✨ (unknown) -> 3:53 (time) -> AI助手SDK沟通(26) (group_name)
    # 重构版：3:53 (time) -> ✨ (unknown) -> AI助手SDK沟通(26) (unknown)
    
    # 查找特殊模式
    special_chars = ['✨', '★', '☆', '♦', '♠', '♥', '♣']
    
    # 1. 确保特殊字符在时间之前
    for i in range(min(2, len(result))):
        if result[i].get('type') == 'time' and i + 1 < len(result):
            if result[i + 1].get('type') == 'unknown' and any(char in result[i + 1].get('content', '') for char in special_chars):
                # 交换顺序
                result[i], result[i + 1] = result[i + 1], result[i]
                logger.debug("兼容性调整：将特殊字符移到时间之前")
                break
    
    # 2. 将第三个unknown类型且包含"沟通"或"群"的消息改为group_name
    for i in range(min(5, len(result))):
        msg = result[i]
        if msg.get('type') == 'unknown':
            content = msg.get('content', '')
            if ('沟通' in content or '群' in content or '组' in content) and '(' in content and ')' in content:
                # 转换为group_name类型
                result[i] = {
                    'type': 'group_name',
                    '群聊名称': content
                }
                logger.debug(f"兼容性调整：将 '{content}' 标记为群聊名称")
                break
    
    return result