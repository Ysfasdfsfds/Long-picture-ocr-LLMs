#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
长图OCR处理脚本
实现长图切分、OCR识别、结果整合和可视化
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any
from rapidocr import RapidOCR
import math
import shutil
from LLM_run import process_with_llm
import re
from sklearn.cluster import DBSCAN
from process_avatar import process_avatar,process_avatar_v2
from refactor.slice_image_step import perform_image_slicing
from refactor.x_croped_calculator import calculate_x_croped
from refactor.slice_processor import process_single_slice, collect_results_to_original_coords
from refactor.visualization import visualize_selected_box
from refactor.chat_exporter import export_structured_chat_messages

class LongImageOCR:
    def __init__(self, config_path: str = "default_rapidocr.yaml"):
        """
        初始化长图OCR处理器
        
        Args:
            config_path: RapidOCR配置文件路径
        """
        self.engine = RapidOCR(config_path=config_path)
        self.slice_height = 1200  # 切片高度
        self.overlap = 200  # 重叠区域像素
        self.text_score_threshold = 0.65  # 文本识别置信度阈值
        self.original_image = None  # 存储原始图像
        
        # 汇总字段：存储原图坐标系统中的所有结果
        self.all_ocr_results_original = []  # 所有OCR结果(原图坐标)
        self.all_avatar_positions_original = []  # 所有头像位置(原图坐标)
        self.marked_ocr_results_original = []  # 标记后的OCR结果(原图坐标)
        self.structured_chat_messages = []  # 结构化的聊天消息
        
        # 创建输出目录
        self.output_json_dir = Path("./output_json")
        self.output_images_dir = Path("./output_images")
        self.output_json_dir.mkdir(exist_ok=True)
        self.output_images_dir.mkdir(exist_ok=True)
        
    def process_slices(self, slices_info: List[Dict]) -> List[Dict]:
        """
        对所有切片进行OCR处理和聊天消息分析
        
        Args:
            slices_info: 切片信息列表
            
        Returns:
            slice_results: 每个切片的OCR和聊天分析结果列表
        """
        slice_results = []
        
        # 初始化汇总字段：存储原图坐标系统中的所有OCR结果和头像位置
        all_ocr_results_original = []  # 所有OCR结果(原图坐标)
        all_avatar_positions_original = []  # 所有头像位置(原图坐标)

        #-------------------------------------------------------------
        #基于所有切片，计算x_croped
        x_croped, selected_box, slice_x_croped_final = calculate_x_croped(slices_info)
        
        # 将selected_box画到原图中并保存
        visualize_selected_box(selected_box, slices_info, self.original_image)
        #计算x_croped的值，到这为止，x_croped的值已经计算出来了
        #-------------------------------------------------------------
             
        index = 0
        for slice_info in slices_info:
            # 处理单个切片
            slice_result = process_single_slice(slice_info, self.engine, 
                                              self.text_score_threshold, 
                                              x_croped, index)
            
            # 收集结果到原图坐标系统
            collect_results_to_original_coords(slice_result, all_ocr_results_original, 
                                             all_avatar_positions_original)
            
            # 添加到切片结果列表
            slice_results.append(slice_result)
            print(f"切片 {slice_result['slice_index']} 添加到结果列表")
            index += 1
        
        
        # 去重处理
        print(f"\n=== 开始去重处理 ===")
        from refactor.deduplication import deduplicate_results
        deduplicated_ocr, deduplicated_avatars = deduplicate_results(all_ocr_results_original, all_avatar_positions_original)
        
        # 判断当前图片是否为飞书截图
        is_feishu_screenshot = self._is_feishu_screenshot(deduplicated_ocr)
        print(f"\n=== 基于去重后数据重新标记 ===")
        from refactor.remark_content import remark_content_with_deduplicated_data_feishu, remark_content_with_deduplicated_data_wechat
        if is_feishu_screenshot:
            print("检测到飞书截图")
            marked_ocr_results = remark_content_with_deduplicated_data_feishu(deduplicated_ocr, deduplicated_avatars, self.original_image)
        else:
            print("执行微信 蓝信 钉钉 的标记")
            # 基于去重后的数据重新标记
            marked_ocr_results = remark_content_with_deduplicated_data_wechat(deduplicated_ocr, deduplicated_avatars, self.original_image)

        
        
        # 保存汇总结果到类属性
        self.all_ocr_results_original = deduplicated_ocr
        self.all_avatar_positions_original = deduplicated_avatars
        self.marked_ocr_results_original = marked_ocr_results
        
        # 保存标记后的OCR结果到JSON文件
        self._export_marked_ocr_results()
        pass
        
        # 整理并导出结构化的聊天消息
        LLM_input = export_structured_chat_messages(self.marked_ocr_results_original, self.output_json_dir)
        # 保存到类属性
        self.structured_chat_messages = LLM_input.get('chat_messages', [])
        
        # 输出汇总统计信息
        print(f"\n=== 去重后汇总统计 ===")
        print(f"原图坐标系统中的OCR结果总数: {len(deduplicated_ocr)} (去重前: {len(all_ocr_results_original)})")
        print(f"原图坐标系统中的头像位置总数: {len(deduplicated_avatars)} (去重前: {len(all_avatar_positions_original)})")
        print(f"标记后的OCR结果总数: {len(marked_ocr_results)}")
        
        # 按切片显示详细信息  
        processed_slices = set(item['slice_index'] for item in deduplicated_ocr)
        for slice_idx in sorted(processed_slices):
            ocr_count = len([item for item in deduplicated_ocr if item['slice_index'] == slice_idx])
            avatar_count = len([item for item in deduplicated_avatars if item['slice_index'] == slice_idx])
            print(f"切片 {slice_idx}: OCR结果 {ocr_count} 个, 头像位置 {avatar_count} 个")
        
        return LLM_input
    
    
    
    
    def process_long_image(self, image_path: str) -> Dict:
        """
        处理长图的完整流程
        
        Args:
            image_path: 图像路径
            
        Returns:
            处理结果摘要
        """
        print(f"开始处理长图: {image_path}")
        
        # 1. 切分图像
        original_image, slices_info = perform_image_slicing(self, image_path)
        
        # 2. OCR处理和切片级聊天分析
        print("步骤2: OCR处理和切片级聊天分析...")
        LLMs_input = self.process_slices(slices_info)
        print(f"LLMs_input: {LLMs_input["chat_messages"]}")
      

        #将结果输入到ollama模型
        print("步骤7: 将结果输入到ollama模型...")
        while True:
            user_question = input("请输入你想问的问题（输入'退出'结束）：")
            if user_question.strip() in ["退出", "q", "Q", "exit"]:
                print("已退出与ollama模型的交互。")
                break
            process_with_llm(user_question, LLMs_input["chat_messages"])


    def _is_feishu_screenshot(self, ocr_results):
        """
        判断图片是否为飞书截图
        
        Args:
            ocr_results: OCR识别结果列表
            
        Returns:
            bool: 是否为飞书截图
        """
        keywords = ["云文档", "文件", "消息", "Pin"]
        detected_keywords = set()
        
        for item in ocr_results:
            text = item.get('text', '')
            for keyword in keywords:
                if keyword in text:
                    detected_keywords.add(keyword)
        
        # 如果检测到所有关键词，则判断为飞书截图
        is_feishu = len(detected_keywords) == len(keywords)
        
        if is_feishu:
            print("检测到飞书截图")
        else:
            print(f"未检测到飞书截图，仅检测到以下关键词: {detected_keywords}")
            
        return is_feishu
    
    def get_summary_data(self) -> Dict:
        """
        获取汇总数据：原图坐标系统中的所有OCR结果、头像位置和标记后的OCR结果
        
        Returns:
            包含OCR结果、头像位置和标记后OCR结果的字典
        """
        return {
            'ocr_results_original': self.all_ocr_results_original,
            'avatar_positions_original': self.all_avatar_positions_original,
            'marked_ocr_results_original': self.marked_ocr_results_original,
            'statistics': {
                'total_ocr_items': len(self.all_ocr_results_original),
                'total_avatars': len(self.all_avatar_positions_original),
                'total_marked_ocr_items': len(self.marked_ocr_results_original),
                'processed_slices': len(set(item['slice_index'] for item in self.all_ocr_results_original)) if self.all_ocr_results_original else 0
            }
        }
    
    def export_summary_data(self, output_path: str = None) -> str:
        """
        导出汇总数据到JSON文件
        
        Args:
            output_path: 输出文件路径，如果为None则使用默认路径
            
        Returns:
            实际的输出文件路径
        """
        if output_path is None:
            output_path = self.output_json_dir / "summary_data_original.json"
        
        summary_data = self.get_summary_data()
        
        # 转换numpy类型为Python原生类型以便JSON序列化
        import json
        import numpy as np
        
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            return obj
        
        # 递归转换所有numpy类型
        def deep_convert(data):
            if isinstance(data, dict):
                return {k: deep_convert(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [deep_convert(v) for v in data]
            else:
                return convert_numpy(data)
        
        summary_data = deep_convert(summary_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        print(f"汇总数据已导出到: {output_path}")
        return str(output_path)
    
    def _export_marked_ocr_results(self, output_path: str = None) -> str:
        """
        导出标记后的OCR结果到JSON文件
        
        Args:
            output_path: 输出文件路径，如果为None则使用默认路径
            
        Returns:
            实际的输出文件路径
        """
        if output_path is None:
            output_path = self.output_json_dir / "marked_ocr_results_original.json"
        
        if not self.marked_ocr_results_original:
            print("没有标记后的OCR结果可导出")
            return ""
        
        import json
        from datetime import datetime
        
        # 提取所有text字段
        text_results = [item.get('text', '') for item in self.marked_ocr_results_original]
        
        export_data = {
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "total_items": len(text_results),
                "description": "标记后的OCR文本结果 - 只包含文本内容"
            },
            "marked_texts": text_results
        }
        
        # 按类型分类统计
        time_count = len([text for text in text_results if "(时间)" in text])
        nickname_count = len([text for text in text_results if "(昵称)" in text])
        content_count = len([text for text in text_results if "(内容)" in text])
        my_content_count = len([text for text in text_results if "(我的内容)" in text])
        
        export_data["statistics"] = {
            "time_items": time_count,
            "nickname_items": nickname_count,
            "content_items": content_count,
            "my_content_items": my_content_count,
            "unmarked_items": len(text_results) - time_count - nickname_count - content_count - my_content_count
        }
        
        # 转换numpy类型为Python原生类型以便JSON序列化
        import numpy as np
        
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            return obj
        
        # 递归转换所有numpy类型
        def deep_convert(data):
            if isinstance(data, dict):
                return {k: deep_convert(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [deep_convert(v) for v in data]
            else:
                return convert_numpy(data)
        
        export_data = deep_convert(export_data)
        
        # 保存到文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        print(f"标记后的OCR结果已导出到: {output_path}")
        print(f"  - 时间标记: {time_count} 项")
        print(f"  - 昵称标记: {nickname_count} 项") 
        print(f"  - 内容标记: {content_count} 项")
        print(f"  - 我的内容: {my_content_count} 项")
        print(f"  - 未标记: {export_data['statistics']['unmarked_items']} 项")
        
        return str(output_path)



def main():
    """主函数"""
    # 初始化处理器
    processor = LongImageOCR(config_path="./default_rapidocr.yaml")
    
    # 处理长图
    # image_path = r"images/image copy 9.png"
    image_path = r"images/feishu.png"
    
    try:
        result = processor.process_long_image(image_path)
        print("\n处理结果摘要:")
        for key, value in result.items():
            print(f"  {key}: {value}")
        
        # 导出汇总数据
        processor.export_summary_data()
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if os.path.exists("output_images"):
        shutil.rmtree("output_images")
    if os.path.exists("output_json"):
        shutil.rmtree("output_json")
    main()