"""
JSON导出器模块
负责将处理结果导出为JSON格式
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

from ..models.ocr_result import OCRItem, AvatarItem, SliceOCRResult
from ..models.chat_message import ChatSession
from ..utils.config import Config
from ..utils.type_converter import ensure_serializable
from ..utils.compatibility_patch import patch_marked_texts_for_compatibility, patch_chat_messages_for_compatibility

logger = logging.getLogger(__name__)


class JsonExporter:
    """JSON导出器"""
    
    def __init__(self, config: Config):
        """
        初始化导出器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.output_dir = Path(config.output.output_json_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_chat_messages(self, session: ChatSession, 
                           output_file: Optional[str] = None) -> str:
        """
        导出结构化的聊天消息
        
        Args:
            session: 聊天会话对象
            output_file: 输出文件名（可选）
            
        Returns:
            输出文件路径
        """
        if output_file is None:
            output_file = "structured_chat_messages.json"
        
        output_path = self.output_dir / output_file
        
        # 获取统计信息
        stats = session.get_statistics()
        
        # 准备导出数据
        chat_messages = [msg.to_dict() for msg in session.messages]
        
        # 应用兼容性补丁
        chat_messages = patch_chat_messages_for_compatibility(chat_messages)
        
        export_data = {
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "total_messages": len(chat_messages),
                "description": "结构化聊天消息 - 按昵称、内容、时间组织"
            },
            "chat_messages": chat_messages,
            "statistics": {
                "nickname_messages": len([msg for msg in chat_messages if msg.get('type') == 'chat']),
                "time_messages": len([msg for msg in chat_messages if msg.get('type') == 'time']),
                "my_messages": len([msg for msg in chat_messages if msg.get('type') == 'my_chat']),
                "group_name_messages": len([msg for msg in chat_messages if msg.get('type') == 'group_name']),
                "retract_messages": len([msg for msg in chat_messages if msg.get('type') == 'retract_message']),
                "unknown_messages": len([msg for msg in chat_messages if msg.get('type') == 'unknown'])
            }
        }
        
        # 确保数据可序列化
        export_data = ensure_serializable(export_data)
        
        # 保存到文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"结构化聊天消息已导出到: {output_path}")
        self._log_statistics(stats)
        
        return str(output_path)
    
    def export_marked_ocr_results(self, marked_items: List[OCRItem],
                                output_file: Optional[str] = None) -> str:
        """
        导出标记后的OCR结果
        
        Args:
            marked_items: 标记后的OCR项列表
            output_file: 输出文件名（可选）
            
        Returns:
            输出文件路径
        """
        if output_file is None:
            output_file = "marked_ocr_results_original.json"
        
        output_path = self.output_dir / output_file
        
        if not marked_items:
            logger.warning("没有标记后的OCR结果可导出")
            return ""
        
        # 提取所有text字段
        text_results = [item.text for item in marked_items]
        
        # 应用兼容性补丁
        text_results = patch_marked_texts_for_compatibility(text_results)
        
        # 统计各类标记
        time_count = len([text for text in text_results if "(时间)" in text])
        nickname_count = len([text for text in text_results if "(昵称)" in text])
        content_count = len([text for text in text_results if "(内容)" in text])
        my_content_count = len([text for text in text_results if "(我的内容)" in text])
        
        export_data = {
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "total_items": len(text_results),
                "description": "标记后的OCR文本结果 - 只包含文本内容"
            },
            "marked_texts": text_results,
            "statistics": {
                "time_items": time_count,
                "nickname_items": nickname_count,
                "content_items": content_count,
                "my_content_items": my_content_count,
                "unmarked_items": len(text_results) - time_count - nickname_count - content_count - my_content_count
            }
        }
        
        # 确保数据可序列化
        export_data = ensure_serializable(export_data)
        
        # 保存到文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"标记后的OCR结果已导出到: {output_path}")
        logger.info(f"  - 时间标记: {time_count} 项")
        logger.info(f"  - 昵称标记: {nickname_count} 项")
        logger.info(f"  - 内容标记: {content_count} 项")
        logger.info(f"  - 我的内容: {my_content_count} 项")
        logger.info(f"  - 未标记: {export_data['statistics']['unmarked_items']} 项")
        
        return str(output_path)
    
    def export_summary_data(self, ocr_items: List[OCRItem], 
                          avatar_items: List[AvatarItem],
                          marked_items: List[OCRItem],
                          output_file: Optional[str] = None) -> str:
        """
        导出汇总数据
        
        Args:
            ocr_items: 去重后的OCR项列表
            avatar_items: 去重后的头像项列表
            marked_items: 标记后的OCR项列表
            output_file: 输出文件名（可选）
            
        Returns:
            输出文件路径
        """
        if output_file is None:
            output_file = "summary_data_original.json"
        
        output_path = self.output_dir / output_file
        
        # 准备导出数据
        export_data = {
            'ocr_results_original': [item.to_dict() for item in ocr_items],
            'avatar_positions_original': [item.to_dict() for item in avatar_items],
            'marked_ocr_results_original': [item.to_dict() for item in marked_items],
            'statistics': {
                'total_ocr_items': len(ocr_items),
                'total_avatars': len(avatar_items),
                'total_marked_ocr_items': len(marked_items),
                'processed_slices': len(set(item.slice_index for item in ocr_items)) if ocr_items else 0
            }
        }
        
        # 确保数据可序列化
        export_data = ensure_serializable(export_data)
        
        # 保存到文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"汇总数据已导出到: {output_path}")
        return str(output_path)
    
    def _log_statistics(self, stats: Dict[str, int]):
        """记录统计信息"""
        logger.info(f"  - 普通聊天消息: {stats['chat']} 条")
        logger.info(f"  - 时间消息: {stats['time']} 条")
        logger.info(f"  - 我的消息: {stats['my_chat']} 条")
        logger.info(f"  - 群聊名称: {stats['group_name']} 条")
        logger.info(f"  - 撤回消息: {stats['retract_message']} 条")
        logger.info(f"  - 未知内容: {stats['unknown']} 条")