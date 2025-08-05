"""
聊天消息数据模型
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum


class MessageType(Enum):
    """消息类型枚举"""
    CHAT = "chat"
    MY_CHAT = "my_chat"
    TIME = "time"
    GROUP_NAME = "group_name"
    SYSTEM_MESSAGE = "system_message"
    RETRACT_MESSAGE = "retract_message"
    UNKNOWN = "unknown"


@dataclass
class ChatMessage:
    """聊天消息"""
    type: MessageType
    content: Optional[str] = None
    nickname: Optional[str] = None
    time: Optional[str] = None
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
        result = {"type": self.type.value}
        
        if self.type in [MessageType.CHAT, MessageType.MY_CHAT]:
            result["昵称"] = self.nickname or "未知"
            result["内容"] = self.content or ""
        elif self.type == MessageType.TIME:
            result["time"] = self.time or ""
        elif self.type == MessageType.GROUP_NAME:
            result["群聊名称"] = self.content or ""
        elif self.type == MessageType.SYSTEM_MESSAGE:
            result["content"] = self.content or ""
        elif self.type == MessageType.RETRACT_MESSAGE:
            result["撤回信息"] = self.content or ""
        else:
            result["content"] = self.content or ""
        
        return result
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ChatMessage':
        """从字典创建实例"""
        msg_type = MessageType(data.get("type", "unknown"))
        
        # 根据类型提取相应字段
        if msg_type in [MessageType.CHAT, MessageType.MY_CHAT]:
            return cls(
                type=msg_type,
                nickname=data.get("昵称"),
                content=data.get("内容")
            )
        elif msg_type == MessageType.TIME:
            return cls(
                type=msg_type,
                time=data.get("time")
            )
        elif msg_type == MessageType.GROUP_NAME:
            return cls(
                type=msg_type,
                content=data.get("群聊名称")
            )
        elif msg_type == MessageType.RETRACT_MESSAGE:
            return cls(
                type=msg_type,
                content=data.get("撤回信息")
            )
        else:
            return cls(
                type=msg_type,
                content=data.get("content")
            )


@dataclass
class ChatSession:
    """聊天会话"""
    messages: List[ChatMessage]
    group_name: Optional[str] = None
    
    def add_message(self, message: ChatMessage):
        """添加消息"""
        self.messages.append(message)
        
        # 如果是群名消息，更新群名
        if message.type == MessageType.GROUP_NAME:
            self.group_name = message.content
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "chat_messages": [msg.to_dict() for msg in self.messages],
            "metadata": {
                "total_messages": len(self.messages),
                "group_name": self.group_name
            }
        }
    
    def get_statistics(self) -> dict:
        """获取统计信息"""
        stats = {
            "total": len(self.messages),
            "chat": 0,
            "my_chat": 0,
            "time": 0,
            "group_name": 0,
            "system_message": 0,
            "retract_message": 0,
            "unknown": 0
        }
        
        for msg in self.messages:
            if msg.type == MessageType.CHAT:
                stats["chat"] += 1
            elif msg.type == MessageType.MY_CHAT:
                stats["my_chat"] += 1
            elif msg.type == MessageType.TIME:
                stats["time"] += 1
            elif msg.type == MessageType.GROUP_NAME:
                stats["group_name"] += 1
            elif msg.type == MessageType.SYSTEM_MESSAGE:
                stats["system_message"] += 1
            elif msg.type == MessageType.RETRACT_MESSAGE:
                stats["retract_message"] += 1
            else:
                stats["unknown"] += 1
        
        return stats