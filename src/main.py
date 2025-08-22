"""
OCR长图处理主程序
重构版本 - 保持与原版相同的功能和输出
"""

import os
import shutil
import logging
import time
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
        
        # 记录总体开始时间
        total_start_time = time.time()
        step_times = {}
        
        try:
            # 1. 切分图像
            step_start = time.time()
            logger.info("步骤1: 切分图像...")
            self.original_image, slice_infos = self.image_slicer.slice_image(image_path)
            step_times['step1_slice_image'] = time.time() - step_start
            logger.info(f"步骤1 完成，耗时: {step_times['step1_slice_image']:.2f}秒")
            
            # 早期检测图片类型并醒目打印
            self._early_platform_detection(image_path)
            
            # 2. 计算x_crop值
            step_start = time.time()
            logger.info("步骤2: 计算x_crop值...")
            x_crop = self.avatar_detector.calculate_x_crop(slice_infos)
            step_times['step2_calculate_x_crop'] = time.time() - step_start
            logger.info(f"步骤2 完成，耗时: {step_times['step2_calculate_x_crop']:.2f}秒")
            
            # 可视化选中的框
            if hasattr(self.avatar_detector, 'slice_x_crop_values'):
                selected_box = self._find_selected_box()
                if selected_box:
                    self.visualizer.visualize_selected_box(
                        selected_box, slice_infos, self.original_image
                    )
            
            # 3. 处理切片
            step_start = time.time()
            logger.info("步骤3: 处理切片...")
            slice_stats = self._process_slices(slice_infos, x_crop)
            step_times['step3_process_slices'] = time.time() - step_start
            step_times['step3_details'] = slice_stats  # 保存详细统计
            logger.info(f"步骤3 完成，耗时: {step_times['step3_process_slices']:.2f}秒")
            
            # 4. 去重处理
            step_start = time.time()
            logger.info("步骤4: 去重处理...")
            self._deduplicate_results()
            step_times['step4_deduplicate'] = time.time() - step_start
            logger.info(f"步骤4 完成，耗时: {step_times['step4_deduplicate']:.2f}秒")
            
            # 4.5 生成原图OCR可视化（使用去重后的数据）
            use_detailed_timing = getattr(self.config, 'enable_detailed_ocr_timing', True)
            if use_detailed_timing and self.all_ocr_items:
                try:
                    logger.info("开始生成原图OCR可视化（使用去重后数据）...")
                    self.ocr_engine.visualize_full_image_ocr_results(
                        self.original_image, self.all_ocr_items
                    )
                    logger.info("原图OCR可视化完成")
                except Exception as vis_e:
                    logger.warning(f"原图OCR可视化失败: {vis_e}")
            
            # 5. 判断平台类型（最终确认）
            step_start = time.time()
            is_feishu = self._is_feishu_screenshot()
            platform = "飞书" if is_feishu else "微信/蓝信/钉钉"
            self._print_final_platform_detection(platform, is_feishu)
            step_times['step5_platform_detection'] = time.time() - step_start
            logger.info(f"步骤5 (平台检测) 完成，耗时: {step_times['step5_platform_detection']:.2f}秒")
            
            # 6. 内容标记
            step_start = time.time()
            logger.info("步骤6: 内容标记...")
            self.marked_ocr_items = self.content_marker.mark_content(
                self.all_ocr_items, self.all_avatar_items, self.original_image
            )
            step_times['step6_content_marking'] = time.time() - step_start
            logger.info(f"步骤6 完成，耗时: {step_times['step6_content_marking']:.2f}秒")
            
            # 7. 分析聊天消息
            step_start = time.time()
            logger.info("步骤7: 分析聊天消息...")
            self.chat_session = self.chat_analyzer.analyze(self.marked_ocr_items)
            step_times['step7_chat_analysis'] = time.time() - step_start
            logger.info(f"步骤7 完成，耗时: {step_times['step7_chat_analysis']:.2f}秒")
            
            # 8. 导出结果
            step_start = time.time()
            logger.info("步骤8: 导出结果...")
            self._export_results()
            
            # 导出详细的时间统计数据
            self._export_timing_details(step_times)
            
            step_times['step8_export_results'] = time.time() - step_start
            logger.info(f"步骤8 完成，耗时: {step_times['step8_export_results']:.2f}秒")
            
            # 9. 创建处理总结
            step_start = time.time()
            self.visualizer.create_process_summary_image(
                self.original_image, slice_infos,
                len(self.all_ocr_items), len(self.all_avatar_items)
            )
            step_times['step9_create_summary'] = time.time() - step_start
            logger.info(f"步骤9 (创建总结) 完成，耗时: {step_times['step9_create_summary']:.2f}秒")
            
            # 计算总体执行时间
            total_time = time.time() - total_start_time
            step_times['total_time'] = total_time
            
            # 打印详细的时间统计
            self._print_timing_summary(step_times)
            
            # 返回结果摘要（包含时间信息）
            return self._create_summary(step_times)
            
        except Exception as e:
            total_time = time.time() - total_start_time
            logger.error(f"处理过程中出现错误: {e}，总耗时: {total_time:.2f}秒", exc_info=True)
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
    
    def _process_slices(self, slice_infos: List, x_crop: Optional[int]) -> Dict[str, float]:
        """处理所有切片，支持详细计时模式"""
        all_ocr_items = []
        all_avatar_items = []
        
        # 详细时间统计
        slice_times = {}
        total_ocr_time = 0
        total_avatar_time = 0
        
        logger.info(f"开始处理 {len(slice_infos)} 个切片...")
        
        # 检查是否启用详细计时模式
        use_detailed_timing = getattr(self.config, 'enable_detailed_ocr_timing', True)
        
        if use_detailed_timing:
            logger.info("🔍 启用详细OCR计时模式 - 将分别记录Detection和Recognition时间")
        
        for slice_info in slice_infos:
            slice_idx = slice_info.slice_index
            slice_start_time = time.time()
            
            logger.info(f"处理切片 {slice_idx}...")
            
            # OCR识别 - 根据模式选择不同的处理方法
            ocr_start_time = time.time()
            if use_detailed_timing:
                # 使用详细计时的OCR处理
                slice_ocr_result = self.ocr_engine.process_slice_with_detailed_timing(slice_info)
            else:
                # 使用常规OCR处理
                slice_ocr_result = self.ocr_engine.process_slice(slice_info)
            ocr_time = time.time() - ocr_start_time
            total_ocr_time += ocr_time
            
            # 头像检测 - 单独计时
            avatar_start_time = time.time()
            avatar_items = self.avatar_detector.detect_avatars(slice_info, x_crop)
            avatar_time = time.time() - avatar_start_time
            total_avatar_time += avatar_time
            
            # 记录单个切片的详细时间
            slice_total_time = time.time() - slice_start_time
            slice_times[f'slice_{slice_idx}'] = {
                'total_time': slice_total_time,
                'ocr_time': ocr_time,
                'avatar_time': avatar_time,
                'ocr_items_count': len(slice_ocr_result.ocr_items),
                'avatar_items_count': len(avatar_items)
            }
            
            logger.info(f"切片 {slice_idx} 完成: OCR={ocr_time:.2f}s, 头像={avatar_time:.2f}s, 总计={slice_total_time:.2f}s")
            
            # 添加到结果中
            all_ocr_items.extend(slice_ocr_result.ocr_items)
            all_avatar_items.extend(avatar_items)
            
            # 更新切片结果
            slice_ocr_result.avatar_items = avatar_items
        
        # 如果启用了详细计时，导出详细的计时记录
        if use_detailed_timing:
            self.ocr_engine.export_slice_timing_records()
        
        # 保存到实例变量
        self.all_ocr_items = all_ocr_items
        self.all_avatar_items = all_avatar_items
        
        # 原图OCR可视化移至去重处理之后进行
        # 汇总统计
        summary_stats = {
            'total_slices': len(slice_infos),
            'total_ocr_time': total_ocr_time,
            'total_avatar_time': total_avatar_time,
            'average_ocr_time': total_ocr_time / len(slice_infos) if slice_infos else 0,
            'average_avatar_time': total_avatar_time / len(slice_infos) if slice_infos else 0,
            'slice_details': slice_times,
            'detailed_timing_enabled': use_detailed_timing
        }
        
        # 打印详细统计
        self._print_slice_timing_summary(summary_stats)
        
        return summary_stats
    
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
    
    def _print_timing_summary(self, step_times: Dict[str, float]):
        """打印详细的时间统计摘要"""
        print("\n" + "⏱️ "*30)
        print("⏱️  处理时间统计摘要")
        print("⏱️ "*30)
        
        # 打印各个步骤的时间
        step_names = {
            'step1_slice_image': '步骤1: 切分图像',
            'step2_calculate_x_crop': '步骤2: 计算x_crop值',
            'step3_process_slices': '步骤3: 处理切片(OCR+头像检测)',
            'step4_deduplicate': '步骤4: 去重处理',
            'step5_platform_detection': '步骤5: 平台类型检测',
            'step6_content_marking': '步骤6: 内容标记',
            'step7_chat_analysis': '步骤7: 分析聊天消息',
            'step8_export_results': '步骤8: 导出结果',
            'step9_create_summary': '步骤9: 创建处理总结'
        }
        
        for step_key, step_name in step_names.items():
            if step_key in step_times:
                time_value = step_times[step_key]
                print(f"📊 {step_name}: {time_value:.2f}秒")
        
        print(f"\n🏁 总体执行时间: {step_times['total_time']:.2f}秒")
        
        # 计算最耗时的步骤（只比较数值类型的步骤时间）
        step_only_times = {k: v for k, v in step_times.items() 
                          if k.startswith('step') and isinstance(v, (int, float))}
        if step_only_times:
            slowest_step = max(step_only_times, key=step_only_times.get)
            slowest_time = step_only_times[slowest_step]
            slowest_name = step_names.get(slowest_step, slowest_step)
            print(f"🐌 最耗时步骤: {slowest_name} ({slowest_time:.2f}秒)")
        
        print("⏱️ "*30 + "\n")
    
    def _print_slice_timing_summary(self, slice_stats: Dict):
        """打印切片处理的详细时间统计"""
        print("\n" + "🔍"*30)
        print("🔍 步骤3: 切片处理详细统计")
        print("🔍"*30)
        
        # 基本统计
        total_slices = slice_stats['total_slices']
        total_ocr_time = slice_stats['total_ocr_time']
        total_avatar_time = slice_stats['total_avatar_time']
        avg_ocr_time = slice_stats['average_ocr_time']
        avg_avatar_time = slice_stats['average_avatar_time']
        
        print(f"📊 切片总数: {total_slices}")
        print(f"🔤 OCR识别总耗时: {total_ocr_time:.2f}秒 (平均: {avg_ocr_time:.2f}秒/片)")
        print(f"👤 头像检测总耗时: {total_avatar_time:.2f}秒 (平均: {avg_avatar_time:.2f}秒/片)")
        
        # 分析最耗时的切片
        slice_details = slice_stats['slice_details']
        if slice_details:
            # 按总时间排序找最耗时的切片
            sorted_slices = sorted(slice_details.items(), 
                                 key=lambda x: x[1]['total_time'], reverse=True)
            
            slowest_slice = sorted_slices[0]
            slice_name = slowest_slice[0]
            slice_data = slowest_slice[1]
            
            print(f"🐌 最耗时切片: {slice_name}")
            print(f"   ├─ 总耗时: {slice_data['total_time']:.2f}秒")
            print(f"   ├─ OCR耗时: {slice_data['ocr_time']:.2f}秒")
            print(f"   ├─ 头像检测: {slice_data['avatar_time']:.2f}秒")
            print(f"   ├─ OCR识别数: {slice_data['ocr_items_count']}个")
            print(f"   └─ 头像数量: {slice_data['avatar_items_count']}个")
            
            # 显示前3个最耗时的切片概览
            print(f"\n📈 耗时TOP3切片:")
            for i, (slice_name, slice_data) in enumerate(sorted_slices[:3]):
                print(f"   {i+1}. {slice_name}: {slice_data['total_time']:.2f}s "
                      f"(OCR: {slice_data['ocr_time']:.2f}s, "
                      f"头像: {slice_data['avatar_time']:.2f}s)")
        
        print("🔍"*30 + "\n")
    
    def _create_summary(self, step_times: Dict[str, float] = None) -> Dict:
        """创建处理结果摘要"""
        stats = self.chat_session.get_statistics() if self.chat_session else {}
        
        summary = {
            "total_ocr_items": len(self.all_ocr_items),
            "total_avatars": len(self.all_avatar_items),
            "total_messages": stats.get('total', 0),
            "chat_messages": stats.get('chat', 0),
            "time_messages": stats.get('time', 0),
            "my_messages": stats.get('my_chat', 0)
        }
        
        # 添加时间统计信息（只保留关键数据，避免过于冗长）
        if step_times:
            # 只保留基本的步骤时间，不包含详细的切片数据
            clean_step_times = {}
            for k, v in step_times.items():
                if k != 'total_time' and not k.endswith('_details') and isinstance(v, (int, float)):
                    clean_step_times[k] = round(v, 2)
            
            summary['timing'] = {
                'total_time': round(step_times.get('total_time', 0), 2),
                'step_times': clean_step_times
            }
            
            # 如果有步骤3的详细统计，只保留汇总信息
            if 'step3_details' in step_times:
                step3_details = step_times['step3_details']
                summary['timing']['step3_summary'] = {
                    'total_slices': step3_details.get('total_slices', 0),
                    'total_ocr_time': round(step3_details.get('total_ocr_time', 0), 2),
                    'total_avatar_time': round(step3_details.get('total_avatar_time', 0), 2),
                    'average_ocr_time': round(step3_details.get('average_ocr_time', 0), 3),
                    'average_avatar_time': round(step3_details.get('average_avatar_time', 0), 3)
                }
        
        return summary
    
    def _export_timing_details(self, step_times: Dict):
        """导出详细的时间统计数据到JSON文件"""
        import json
        from pathlib import Path
        
        # 确保输出目录存在
        output_dir = Path("output_json")
        output_dir.mkdir(exist_ok=True)
        
        # 准备完整的时间统计数据
        timing_data = {
            "timestamp": str(time.strftime("%Y-%m-%d %H:%M:%S")),
            "total_execution_time": step_times.get('total_time', 0),
            "step_summary": {},
            "step3_detailed_analysis": None
        }
        
        # 添加各步骤的基本时间统计
        step_names = {
            'step1_slice_image': '步骤1: 切分图像',
            'step2_calculate_x_crop': '步骤2: 计算x_crop值',
            'step3_process_slices': '步骤3: 处理切片(OCR+头像检测)',
            'step4_deduplicate': '步骤4: 去重处理',
            'step5_platform_detection': '步骤5: 平台类型检测',
            'step6_content_marking': '步骤6: 内容标记',
            'step7_chat_analysis': '步骤7: 分析聊天消息',
            'step8_export_results': '步骤8: 导出结果',
            'step9_create_summary': '步骤9: 创建处理总结'
        }
        
        for step_key, step_name in step_names.items():
            if step_key in step_times and isinstance(step_times[step_key], (int, float)):
                timing_data["step_summary"][step_key] = {
                    "name": step_name,
                    "time_seconds": round(step_times[step_key], 3),
                    "percentage": round((step_times[step_key] / step_times.get('total_time', 1)) * 100, 1)
                }
        
        # 添加步骤3的详细分析
        if 'step3_details' in step_times:
            step3_details = step_times['step3_details']
            
            # 基本汇总
            timing_data["step3_detailed_analysis"] = {
                "summary": {
                    "total_slices": step3_details.get('total_slices', 0),
                    "total_ocr_time": round(step3_details.get('total_ocr_time', 0), 3),
                    "total_avatar_time": round(step3_details.get('total_avatar_time', 0), 3),
                    "average_ocr_time": round(step3_details.get('average_ocr_time', 0), 3),
                    "average_avatar_time": round(step3_details.get('average_avatar_time', 0), 3),
                    "ocr_percentage": round((step3_details.get('total_ocr_time', 0) / step3_details.get('total_ocr_time', 1) + step3_details.get('total_avatar_time', 0)) * 100, 1) if (step3_details.get('total_ocr_time', 0) + step3_details.get('total_avatar_time', 0)) > 0 else 0
                },
                "slice_details": {}
            }
            
            # 每个切片的详细数据
            slice_details = step3_details.get('slice_details', {})
            for slice_name, slice_data in slice_details.items():
                timing_data["step3_detailed_analysis"]["slice_details"][slice_name] = {
                    "total_time": round(slice_data.get('total_time', 0), 3),
                    "ocr_time": round(slice_data.get('ocr_time', 0), 3),
                    "avatar_time": round(slice_data.get('avatar_time', 0), 3),
                    "ocr_items_count": slice_data.get('ocr_items_count', 0),
                    "avatar_items_count": slice_data.get('avatar_items_count', 0),
                    "efficiency": {
                        "ocr_items_per_second": round(slice_data.get('ocr_items_count', 0) / max(slice_data.get('ocr_time', 0.001), 0.001), 2),
                        "total_items_per_second": round((slice_data.get('ocr_items_count', 0) + slice_data.get('avatar_items_count', 0)) / max(slice_data.get('total_time', 0.001), 0.001), 2)
                    }
                }
            
            # 添加性能分析
            if slice_details:
                sorted_slices = sorted(slice_details.items(), key=lambda x: x[1]['total_time'], reverse=True)
                timing_data["step3_detailed_analysis"]["performance_analysis"] = {
                    "slowest_slice": {
                        "name": sorted_slices[0][0],
                        "time": round(sorted_slices[0][1]['total_time'], 3),
                        "reason": "most_time_consuming"
                    },
                    "fastest_slice": {
                        "name": sorted_slices[-1][0],
                        "time": round(sorted_slices[-1][1]['total_time'], 3),
                        "reason": "least_time_consuming"
                    },
                    "top_3_slowest": [
                        {
                            "name": slice_name,
                            "time": round(slice_data['total_time'], 3),
                            "ocr_percentage": round((slice_data['ocr_time'] / slice_data['total_time']) * 100, 1) if slice_data['total_time'] > 0 else 0
                        }
                        for slice_name, slice_data in sorted_slices[:3]
                    ]
                }
        
        # 导出到文件
        output_file = output_dir / "timing_analysis.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(timing_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"详细时间统计已导出到: {output_file}")
        print(f"📁 详细时间统计已保存到: {output_file}")
    
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