"""
配置管理模块
统一管理所有配置参数，支持从文件加载
"""

import os
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class OCRConfig:
    """OCR相关配置"""
    config_path: str = "default_rapidocr.yaml"
    text_score_threshold: float = 0.65
    

@dataclass
class ImageConfig:
    """图像处理相关配置"""
    slice_height: int = 1200
    overlap: int = 200
    binary_threshold: int = 230
    gaussian_blur_size: tuple = (5, 5)
    merge_distance_factor: float = 1.0  # 合并距离因子（相对于最大框尺寸）
    

@dataclass
class AvatarConfig:
    """头像检测相关配置"""
    square_ratio_min: float = 0.9
    square_ratio_max: float = 1.1
    strict_square_ratio_min: float = 0.8
    strict_square_ratio_max: float = 1.2
    iou_threshold: float = 0.0
    x_crop_offset: int = 3
    

@dataclass
class DeduplicationConfig:
    """去重相关配置"""
    ocr_iou_threshold: float = 0.65
    avatar_iou_threshold: float = 0.0
    

@dataclass
class ContentMarkingConfig:
    """内容标记相关配置"""
    green_hsv_lower: tuple = (35, 40, 40)
    green_hsv_upper: tuple = (85, 255, 255)
    green_ratio_threshold: float = 0.2
    
    blue_hsv_lower: tuple = (100, 30, 80)
    blue_hsv_upper: tuple = (130, 180, 255)
    blue_ratio_threshold: float = 0.3
    
    white_hsv_lower: tuple = (0, 0, 200)
    white_hsv_upper: tuple = (180, 30, 255)
    white_ratio_threshold: float = 0.5
    

@dataclass
class OutputConfig:
    """输出相关配置"""
    output_json_dir: str = "./output_json"
    output_images_dir: str = "./output_images"
    debug_dir: str = "./output_images/debug"
    

@dataclass
class Config:
    """主配置类，包含所有子配置"""
    ocr: OCRConfig = field(default_factory=OCRConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    avatar: AvatarConfig = field(default_factory=AvatarConfig)
    deduplication: DeduplicationConfig = field(default_factory=DeduplicationConfig)
    content_marking: ContentMarkingConfig = field(default_factory=ContentMarkingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    def __post_init__(self):
        """初始化后创建必要的目录"""
        for dir_path in [self.output.output_json_dir, 
                        self.output.output_images_dir,
                        self.output.debug_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_file(cls, config_file: str) -> 'Config':
        """从配置文件加载配置"""
        if not os.path.exists(config_file):
            print(f"配置文件 {config_file} 不存在，使用默认配置")
            return cls()
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # 递归创建配置对象
            config = cls()
            for section, values in config_data.items():
                if hasattr(config, section) and isinstance(values, dict):
                    section_config = getattr(config, section)
                    for key, value in values.items():
                        if hasattr(section_config, key):
                            setattr(section_config, key, value)
            
            return config
        except Exception as e:
            print(f"加载配置文件失败: {e}，使用默认配置")
            return cls()
    
    def to_file(self, config_file: str):
        """保存配置到文件"""
        config_data = {}
        for attr_name in ['ocr', 'image', 'avatar', 'deduplication', 
                         'content_marking', 'output']:
            attr = getattr(self, attr_name)
            config_data[attr_name] = {
                k: v for k, v in vars(attr).items() 
                if not k.startswith('_')
            }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
    
    def get_time_patterns(self) -> list:
        """获取时间匹配模式列表"""
        return [
            r'\d{4}年\d{1,2}月\d{1,2}日\d{1,2}:\d{2}',                          # 2025年6月17日9:10
            r'\d{4}年\d{1,2}月\d{1,2}日',                                       # 2025年6月17日
            r'(昨天|今天|前天|明天)(早上|上午|中午|下午|晚上|凌晨)?\d{1,2}:\d{2}',  # 昨天晚上6:23
            r'(上午|下午|早上|中午|晚上|凌晨)\d{1,2}:\d{2}',                     # 上午9:17
            r'\d{1,2}:\d{2}',                                                  # 5:51
            r'\d{4}-\d{1,2}-\d{1,2}',                                          # 年-月-日  
            r'\d{1,2}/\d{1,2}',                                                # 月/日
            r'\d{1,2}月\d{1,2}日',                                             # 月日
            r'\d{1,2}月\d{1,2}日\d{1,2}:\d{2}',                                # 7月28日14:53
            r'\d{1,2}:\d{2}:\d{2}',                                           # 时:分:秒
            r'(昨天|今天|前天|明天)',                                           # 相对日期
            r'周[一二三四五六日天]',                                            # 星期
        ]
    
    def get_system_message_patterns(self) -> list:
        """获取系统消息匹配模式列表"""
        return [
            r'欢迎.*加入',           # 欢迎xxx加入
            r'.*加入了群聊',         # xxx加入了群聊
            r'.*退出了群聊',         # xxx退出了群聊
            r'.*邀请.*加入了群聊',   # xxx邀请yyy加入了群聊
            r'.*撤回了.*条消息',     # xxx撤回了x条消息
            r'群聊名称.*更改为',     # 群聊名称更改为xxx
            r'.*开启了.*',          # xxx开启了群聊邀请确认
            r'.*关闭了.*',          # xxx关闭了群聊邀请确认
            r'.*设置.*为管理员',     # xxx设置yyy为管理员
            r'.*移除了.*的管理员身份', # xxx移除了yyy的管理员身份
            r'.*修改了群名称',       # xxx修改了群名称
            r'系统消息',            # 直接的系统消息标识
            r'以上是历史消息',       # 历史消息分隔符
        ]
    
    def get_time_exclude_keywords(self) -> list:
        """获取时间排除关键词列表"""
        return ['报送', '回执', '会议', '参加', '人员', '工作', '通知', '安排', 
                '要求', '地点', '内容', '完成', '需要', '前', '后', '开始', 
                '结束', '传包', '表格', '填写', '更新', '自测']
    
    def get_feishu_keywords(self) -> list:
        """获取飞书截图识别关键词"""
        return ["云文档", "文件", "消息", "Pin"]


# 创建全局默认配置实例
default_config = Config()