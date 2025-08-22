"""
聊天分析器模块
负责将标记后的OCR结果组织成结构化的聊天消息
"""

import re
import logging
from typing import List, Dict, Optional

from ..models.ocr_result import OCRItem
from ..models.chat_message import ChatMessage, ChatSession, MessageType
from ..utils.config import Config

logger = logging.getLogger(__name__)


class ChatAnalyzer:
    """聊天分析器"""
    
    def __init__(self, config: Config):
        """
        初始化聊天分析器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.time_patterns = config.get_time_patterns()
    
    def analyze(self, marked_ocr_items: List[OCRItem]) -> ChatSession:
        """
        分析标记后的OCR结果，组织成结构化聊天消息
        
        Args:
            marked_ocr_items: 标记后的OCR项列表
            
        Returns:
            聊天会话对象
        """
        logger.info("开始分析聊天消息...")
        
        # 提取所有标记后的文本
        marked_texts = [item.text for item in marked_ocr_items]
        
        # 分析昵称信息
        nickname_analysis = self._analyze_nicknames(marked_texts)
        
        # 创建聊天会话
        session = ChatSession(messages=[])
        
        # 组织聊天消息
        self._organize_messages(marked_texts, nickname_analysis, session)
        
        # 获取统计信息
        stats = session.get_statistics()
        logger.info(f"分析完成: 总消息数={stats['total']}, "
                   f"聊天消息={stats['chat']}, "
                   f"时间消息={stats['time']}, "
                   f"我的消息={stats['my_chat']}")
        
        return session
    
    def _organize_messages(self, marked_texts: List[str], 
                          nickname_analysis: Dict, 
                          session: ChatSession):
        """组织聊天消息"""
        i = 0
        
        while i < len(marked_texts):
            text = marked_texts[i].strip()
            if not text:
                i += 1
                continue
            
            # 处理时间标记
            if "(时间)" in text:
                time_text = text.replace("(时间)", "").strip()
                message = ChatMessage(type=MessageType.TIME, time=time_text)
                session.add_message(message)
                i += 1
                continue
            
            # 处理我的内容标记
            if "(我的内容)" in text:
                my_content_parts = []
                j = i
                
                # 收集连续的"我的内容"标记
                while j < len(marked_texts):
                    current_text = marked_texts[j].strip()
                    if not current_text:
                        j += 1
                        continue
                    
                    if "(我的内容)" in current_text:
                        content = current_text.replace("(我的内容)", "").strip()
                        if content:
                            my_content_parts.append(content)
                        j += 1
                    else:
                        break
                
                # 创建消息
                if my_content_parts:
                    combined_content = " ".join(my_content_parts)
                    message = ChatMessage(
                        type=MessageType.MY_CHAT,
                        nickname="我",
                        content=combined_content
                    )
                    session.add_message(message)
                
                i = j
                continue
            
            # 处理昵称标记
            if "(昵称)" in text:
                nickname = text.replace("(昵称)", "").strip()
                
                # 收集后续内容
                content_parts = []
                retract_messages = []
                j = i + 1
                
                while j < len(marked_texts):
                    next_text = marked_texts[j].strip()
                    if not next_text:
                        j += 1
                        continue
                    
                    # 遇到新的标记则停止
                    if ("(昵称)" in next_text or "(时间)" in next_text or 
                        "(我的内容)" in next_text):
                        break
                    
                    # 收集内容
                    if "(内容)" in next_text:
                        content = next_text.replace("(内容)", "").strip()
                        if content:
                            # 检查特殊内容
                            if "撤回了一条消息" in content:
                                retract_messages.append(content)
                            elif self._is_time_content(content):
                                # 时间内容
                                message = ChatMessage(type=MessageType.TIME, time=content)
                                session.add_message(message)
                            else:
                                content_parts.append(content)
                    else:
                        # 未标记的内容也收集
                        content_parts.append(next_text)
                    
                    j += 1
                
                # 创建消息
                if content_parts:
                    combined_content = " ".join(content_parts)
                    message = ChatMessage(
                        type=MessageType.CHAT,
                        nickname=nickname,
                        content=combined_content
                    )
                    session.add_message(message)
                else:
                    # 没有内容，检查是否为群聊名称
                    if self._is_group_name(nickname, i, nickname_analysis):
                        message = ChatMessage(
                            type=MessageType.GROUP_NAME,
                            content=nickname
                        )
                        session.add_message(message)
                
                # 添加撤回消息
                for retract_content in retract_messages:
                    message = ChatMessage(
                        type=MessageType.RETRACT_MESSAGE,
                        content=retract_content
                    )
                    session.add_message(message)
                
                i = j
                continue
            
            # 处理孤立的内容标记
            if "(内容)" in text:
                content = text.replace("(内容)", "").strip()
                
                if "撤回了一条消息" in content:
                    message = ChatMessage(
                        type=MessageType.RETRACT_MESSAGE,
                        content=content
                    )
                elif self._is_time_content(content):
                    message = ChatMessage(type=MessageType.TIME, time=content)
                else:
                    message = ChatMessage(
                        type=MessageType.CHAT,
                        nickname="未知",
                        content=content
                    )
                session.add_message(message)
                i += 1
                continue
            
            # 处理系统消息
            if "(系统消息)" in text:
                system_content = text.replace("(系统消息)", "").strip()
                message = ChatMessage(
                    type=MessageType.SYSTEM_MESSAGE,
                    content=system_content
                )
                session.add_message(message)
            else:
                # 未知内容
                message = ChatMessage(
                    type=MessageType.UNKNOWN,
                    content=text
                )
                session.add_message(message)
            
            i += 1
    
    def _analyze_nicknames(self, marked_texts: List[str]) -> Dict:
        """分析所有昵称的出现情况"""
        nickname_info = {}
        first_nickname_index = None
        
        for i, text in enumerate(marked_texts):
            text = text.strip()
            if "(昵称)" in text:
                nickname = text.replace("(昵称)", "").strip()
                
                if nickname not in nickname_info:
                    nickname_info[nickname] = {
                        'first_occurrence': i,
                        'count': 1,
                        'has_content': False
                    }
                    
                    if first_nickname_index is None:
                        first_nickname_index = i
                else:
                    nickname_info[nickname]['count'] += 1
                
                # 检查该昵称是否有对应内容
                j = i + 1
                has_content = False
                while j < len(marked_texts):
                    next_text = marked_texts[j].strip()
                    if not next_text:
                        j += 1
                        continue
                    
                    if ("(昵称)" in next_text or "(时间)" in next_text or 
                        "(我的内容)" in next_text):
                        break
                    
                    if "(内容)" in next_text:
                        content = next_text.replace("(内容)", "").strip()
                        if content:
                            has_content = True
                            break
                    
                    j += 1
                
                if has_content:
                    nickname_info[nickname]['has_content'] = True
        
        return {
            'nickname_info': nickname_info,
            'first_nickname_index': first_nickname_index
        }
    
    def _is_group_name(self, nickname: str, position: int, 
                      nickname_analysis: Dict) -> bool:
        """判断是否为群聊名称"""
        nickname_info = nickname_analysis['nickname_info']
        first_nickname_index = nickname_analysis['first_nickname_index']
        
        if nickname not in nickname_info:
            return False
        
        info = nickname_info[nickname]
        
        # 排除虚拟昵称（如"未知用户1"）
        if nickname.startswith("未知用户"):
            return False
        
        # 检查是否包含群聊特征（如括号内数字）
        import re
        has_group_pattern = bool(re.search(r'\(\d+\)', nickname))
        
        # 条件：是第一个出现的昵称，且只出现一次或没有内容，且有群聊特征
        is_first_nickname = (position == first_nickname_index)
        is_unique_or_no_content = (info['count'] == 1 or not info['has_content'])
        
        return is_first_nickname and is_unique_or_no_content and has_group_pattern
    
    def _is_time_content(self, text: str) -> bool:
        """检查文本是否为时间内容"""
        text = text.strip()
        
        # 排除过长的文本
        if len(text) > 30:
            return False
        
        # 排除包含非时间关键词的文本
        exclude_keywords = self.config.get_time_exclude_keywords()
        if any(keyword in text for keyword in exclude_keywords):
            return False
        
        # 检查时间模式
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
                            return True
                    elif pattern.startswith(r'\d{4}年'):
                        if match_ratio >= 0.7:
                            return True
                    else:
                        if match_ratio >= 0.6:
                            return True
        
        return False