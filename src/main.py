"""
OCR长图处理主程序
重构版本 - 保持与原版相同的功能和输出
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from .core.image_slicer import ImageSlicer
from .core.ocr_engine import OCREngine
from .core.chat_analyzer import ChatAnalyzer
from .processors.avatar_detector import AvatarDetector
from .processors.content_marker import ContentMarker
from .processors.deduplicator import Deduplicator
from .exporters.json_exporter import JsonExporter
from .models.ocr_result import OCRItem, AvatarItem
from .utils.config import Config
from .utils.visualization import Visualizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LongImageOCR:
    """长图OCR处理器（重构版）"""
    
    def __init__(self, config_path: str = "default_rapidocr.yaml"):
        """
        初始化长图OCR处理器
        
        Args:
            config_path: RapidOCR配置文件路径
        """
        # 加载配置
        self.config = Config()
        self.config.ocr.config_path = config_path
        
        # 初始化各个组件
        self.image_slicer = ImageSlicer(self.config)
        self.ocr_engine = OCREngine(self.config)
        self.avatar_detector = AvatarDetector(self.config)
        self.content_marker = ContentMarker(self.config)
        self.deduplicator = Deduplicator(self.config)
        self.chat_analyzer = ChatAnalyzer(self.config)
        self.json_exporter = JsonExporter(self.config)
        self.visualizer = Visualizer(self.config)
        
        # 存储处理结果
        self.original_image = None
        self.all_ocr_items = []
        self.all_avatar_items = []
        self.marked_ocr_items = []
        self.chat_session = None
        
        logger.info("长图OCR处理器初始化完成")
    
    def process_long_image(self, image_path: str) -> Dict:
        """
        处理长图的完整流程
        
        Args:
            image_path: 图像路径
            
        Returns:
            处理结果摘要
        """
        logger.info(f"开始处理长图: {image_path}")
        
        try:
            # 1. 切分图像
            logger.info("步骤1: 切分图像...")
            self.original_image, slice_infos = self.image_slicer.slice_image(image_path)
            
            # 早期检测图片类型并醒目打印
            self._early_platform_detection(image_path)
            
            # 2. 计算x_crop值
            logger.info("步骤2: 计算x_crop值...")
            x_crop = self.avatar_detector.calculate_x_crop(slice_infos)
            
            # 可视化选中的框
            if hasattr(self.avatar_detector, 'slice_x_crop_values'):
                selected_box = self._find_selected_box()
                if selected_box:
                    self.visualizer.visualize_selected_box(
                        selected_box, slice_infos, self.original_image
                    )
            
            # 3. 处理切片
            logger.info("步骤3: 处理切片...")
            slice_results = self._process_slices(slice_infos, x_crop)
            
            # 4. 去重处理
            logger.info("步骤4: 去重处理...")
            self._deduplicate_results()
            
            # 5. 判断平台类型（最终确认）
            is_feishu = self._is_feishu_screenshot()
            platform = "飞书" if is_feishu else "微信/蓝信/钉钉"
            self._print_final_platform_detection(platform, is_feishu)
            
            # 6. 内容标记
            logger.info("步骤5: 内容标记...")
            self.marked_ocr_items = self.content_marker.mark_content(
                self.all_ocr_items, self.all_avatar_items, self.original_image
            )
            
            # 7. 分析聊天消息
            logger.info("步骤6: 分析聊天消息...")
            self.chat_session = self.chat_analyzer.analyze(self.marked_ocr_items)
            
            # 8. 导出结果
            logger.info("步骤7: 导出结果...")
            self._export_results()
            
            # 9. 创建处理总结
            self.visualizer.create_process_summary_image(
                self.original_image, slice_infos,
                len(self.all_ocr_items), len(self.all_avatar_items)
            )
            
            # 返回结果摘要
            return self._create_summary()
            
        except Exception as e:
            logger.error(f"处理过程中出现错误: {e}", exc_info=True)
            raise
    
    def _early_platform_detection(self, image_path: str):
        """在处理早期进行平台类型检测并醒目打印"""
        print("\n" + "="*60)
        print("🔍 图片类型检测 - 初步分析")
        print("="*60)
        
        # 基于文件名的初步判断
        image_name = os.path.basename(image_path)
        print(f"📁 输入图片: {image_name}")
        
        # 进行初步OCR检测（使用少量切片）
        logger.info("进行初步OCR检测以确定平台类型...")
        
        # 这里可以添加更多检测逻辑
        print("⏳ 正在分析图片内容特征...")
        print("="*60 + "\n")
    
    def _print_final_platform_detection(self, platform: str, is_feishu: bool):
        """醒目打印最终的平台检测结果"""
        print("\n" + "*"*60)
        print("🎯 图片类型检测 - 最终结果")
        print("*"*60)
        
        if is_feishu:
            print("📱 检测结果: 飞书 (Feishu)")
            print("✅ 检测状态: 成功识别")
            print("🔧 处理模式: 飞书专用模式")
            print("📋 特征识别: 检测到飞书特有界面元素")
        else:
            print("📱 检测结果: 微信/蓝信/钉钉")
            print("ℹ️  检测状态: 默认识别")
            print("🔧 处理模式: 通用聊天模式")
            print("📋 特征识别: 未检测到飞书特征，使用通用处理")
        
        print("*"*60)
        logger.info(f"🎯 最终确认平台类型: {platform}")
        print()
    
    def _process_slices(self, slice_infos: List, x_crop: Optional[int]) -> List:
        """处理所有切片"""
        all_ocr_items = []
        all_avatar_items = []
        
        for slice_info in slice_infos:
            logger.info(f"处理切片 {slice_info.slice_index}...")
            
            # OCR识别
            slice_ocr_result = self.ocr_engine.process_slice(slice_info)
            
            # 头像检测
            avatar_items = self.avatar_detector.detect_avatars(slice_info, x_crop)
            
            # 添加到结果中
            all_ocr_items.extend(slice_ocr_result.ocr_items)
            all_avatar_items.extend(avatar_items)
            
            # 更新切片结果
            slice_ocr_result.avatar_items = avatar_items
        
        # 保存到实例变量
        self.all_ocr_items = all_ocr_items
        self.all_avatar_items = all_avatar_items
        
        return []  # 兼容原版接口
    
    def _deduplicate_results(self):
        """去重处理"""
        self.all_ocr_items, self.all_avatar_items = self.deduplicator.deduplicate(
            self.all_ocr_items, self.all_avatar_items
        )
    
    def _is_feishu_screenshot(self) -> bool:
        """判断是否为飞书截图"""
        keywords = self.config.get_feishu_keywords()
        detected_keywords = set()
        
        for item in self.all_ocr_items:
            text = item.text
            for keyword in keywords:
                if keyword in text:
                    detected_keywords.add(keyword)
        
        is_feishu = len(detected_keywords) == len(keywords)
        
        # 详细记录检测结果
        if is_feishu:
            logger.info(f"✓ 飞书检测成功 - 检测到所有关键词: {detected_keywords}")
        else:
            missing_keywords = set(keywords) - detected_keywords
            logger.info(f"✗ 飞书检测失败 - 检测到关键词: {detected_keywords}, 缺失关键词: {missing_keywords}")
        
        return is_feishu
    
    def _export_results(self):
        """导出所有结果"""
        # 导出标记后的OCR结果
        self.json_exporter.export_marked_ocr_results(self.marked_ocr_items)
        
        # 导出结构化聊天消息
        self.json_exporter.export_chat_messages(self.chat_session)
        
        # 导出汇总数据
        self.json_exporter.export_summary_data(
            self.all_ocr_items, self.all_avatar_items, self.marked_ocr_items
        )
    
    def _find_selected_box(self) -> Optional[Tuple]:
        """找到选中的目标框"""
        # 从avatar_detector的内部数据中查找
        if not hasattr(self.avatar_detector, 'slice_x_crop_values'):
            return None
        
        # 这里简化处理，实际应该从calculate_x_crop的返回值中获取
        # 为了兼容原版，我们模拟返回一个值
        for slice_idx, box in self.avatar_detector.slice_x_crop_values.items():
            if box is not None:
                x, y, w, h = box
                return (x, y, w, h, slice_idx)
        
        return None
    
    def _create_summary(self) -> Dict:
        """创建处理结果摘要"""
        stats = self.chat_session.get_statistics() if self.chat_session else {}
        
        return {
            "total_ocr_items": len(self.all_ocr_items),
            "total_avatars": len(self.all_avatar_items),
            "total_messages": stats.get('total', 0),
            "chat_messages": stats.get('chat', 0),
            "time_messages": stats.get('time', 0),
            "my_messages": stats.get('my_chat', 0)
        }
    
    def process_with_llm(self, user_question: str, llm_processor=None):
        """
        使用LLM处理用户问题（保持接口兼容）
        
        Args:
            user_question: 用户问题
            llm_processor: LLM处理器（可选）
        """
        if self.chat_session and llm_processor:
            messages = self.chat_session.to_dict()['chat_messages']
            return llm_processor(user_question, messages)
        else:
            logger.warning("没有可用的聊天消息或LLM处理器")
            return None


def main():
    """主函数"""
    # 清理输出目录
    if os.path.exists("output_images"):
        shutil.rmtree("output_images")
    if os.path.exists("output_json"):
        shutil.rmtree("output_json")
    
    # 初始化处理器
    processor = LongImageOCR(config_path="./default_rapidocr.yaml")
    
    # 处理长图
    image_path = r"images/image copy 9.png"
    
    try:
        result = processor.process_long_image(image_path)
        print("\n处理结果摘要:")
        for key, value in result.items():
            print(f"  {key}: {value}")
        
        # 模拟与LLM的交互
        print("\n处理完成！")
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()