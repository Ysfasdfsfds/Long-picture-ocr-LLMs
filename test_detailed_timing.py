#!/usr/bin/env python3
"""
详细OCR计时功能测试脚本
演示如何使用新的详细计时功能来分别记录detection和recognition的推理时间
"""

import sys
import os
import time
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.main import LongImageOCR
from src.utils.config import Config

def test_detailed_ocr_timing():
    """测试详细OCR计时功能"""
    print("🚀 开始测试详细OCR计时功能")
    print("="*60)
    
    # 创建配置，启用详细计时
    config = Config()
    config.enable_detailed_ocr_timing = True  # 启用详细计时
    
    print(f"📋 配置信息:")
    print(f"  - 详细计时模式: {'✅ 启用' if config.enable_detailed_ocr_timing else '❌ 禁用'}")
    print(f"  - OCR配置文件: {config.ocr.config_path}")
    print(f"  - 文本置信度阈值: {config.ocr.text_score_threshold}")
    print(f"  - 切片高度: {config.image.slice_height}px")
    print(f"  - 重叠区域: {config.image.overlap}px")
    print()
    
    # 初始化处理器
    try:
        processor = LongImageOCR(config_path="./default_rapidocr.yaml")
        # 手动设置配置
        processor.config = config
        print("✅ OCR处理器初始化成功")
    except Exception as e:
        print(f"❌ OCR处理器初始化失败: {e}")
        return
    
    # 查找测试图片
    test_image_dirs = ["images", "test_images", "."]
    test_image_patterns = ["*.png", "*.jpg", "*.jpeg"]
    
    test_image_path = None
    for test_dir in test_image_dirs:
        if not os.path.exists(test_dir):
            continue
        for pattern in test_image_patterns:
            images = list(Path(test_dir).glob(pattern))
            if images:
                test_image_path = str(images[0])  # 使用第一个找到的图片
                break
        if test_image_path:
            break
    
    if not test_image_path:
        print("❌ 未找到测试图片，请将图片放在以下目录之一：")
        for test_dir in test_image_dirs:
            print(f"   - {test_dir}/")
        return
    
    print(f"📸 测试图片: {test_image_path}")
    print()
    
    # 处理图片
    try:
        print("🔄 开始处理...")
        start_time = time.time()
        
        result = processor.process_long_image(test_image_path)
        
        end_time = time.time()
        total_processing_time = end_time - start_time
        
        print(f"\n✅ 处理完成！总耗时: {total_processing_time:.2f}秒")
        print()
        
        # 显示处理结果摘要
        print("📊 处理结果摘要:")
        for key, value in result.items():
            if key != 'timing':  # timing信息太详细，单独处理
                print(f"  - {key}: {value}")
        print()
        
        # 显示计时信息
        if 'timing' in result:
            timing = result['timing']
            print("⏱️  总体计时统计:")
            print(f"  - 总执行时间: {timing.get('total_time', 0):.2f}秒")
            
            if 'step3_summary' in timing:
                step3 = timing['step3_summary']
                print(f"  - 总切片数: {step3.get('total_slices', 0)}")
                print(f"  - OCR总耗时: {step3.get('total_ocr_time', 0):.2f}秒")
                print(f"  - 平均OCR时间: {step3.get('average_ocr_time', 0):.3f}秒/片")
                print(f"  - 头像检测总耗时: {step3.get('total_avatar_time', 0):.2f}秒")
                print(f"  - 平均头像检测时间: {step3.get('average_avatar_time', 0):.3f}秒/片")
        print()
        
        # 检查详细计时文件是否生成
        timing_file = Path("output_json/slice_ocr_detailed_timing.json")
        if timing_file.exists():
            print("✅ 详细计时记录文件已生成:")
            print(f"   📁 {timing_file}")
            
            # 简单统计文件内容
            import json
            try:
                with open(timing_file, 'r', encoding='utf-8') as f:
                    timing_data = json.load(f)
                
                if 'summary' in timing_data:
                    summary = timing_data['summary']
                    print(f"\n🔍 详细计时分析 (来自{timing_file.name}):")
                    
                    if 'timing_summary' in summary:
                        ts = summary['timing_summary']
                        print(f"  📊 时间分布统计:")
                        print(f"    - Detection总时间: {ts.get('total_detection_time', 0):.3f}秒")
                        print(f"    - Recognition总时间: {ts.get('total_recognition_time', 0):.3f}秒")
                        print(f"    - 预处理总时间: {ts.get('total_preprocessing_time', 0):.3f}秒")
                        print(f"    - 后处理总时间: {ts.get('total_postprocessing_time', 0):.3f}秒")
                        print()
                        print(f"    - 平均Detection时间: {ts.get('average_detection_time', 0):.3f}秒/片")
                        print(f"    - 平均Recognition时间: {ts.get('average_recognition_time', 0):.3f}秒/片")
                    
                    if 'performance_analysis' in summary:
                        pa = summary['performance_analysis']
                        if 'time_distribution' in pa:
                            td = pa['time_distribution']
                            print(f"  📈 时间占比分析:")
                            print(f"    - Detection占比: {td.get('detection_percentage', 0):.1f}%")
                            print(f"    - Recognition占比: {td.get('recognition_percentage', 0):.1f}%")
                            print(f"    - 预处理占比: {td.get('preprocessing_percentage', 0):.1f}%")
                            print(f"    - 后处理占比: {td.get('postprocessing_percentage', 0):.1f}%")
                        
                        if 'slowest_slice' in pa and 'fastest_slice' in pa:
                            print(f"  🐌 最慢切片: slice_{pa['slowest_slice']['slice_index']} ({pa['slowest_slice']['total_time']:.3f}s)")
                            print(f"  🏃 最快切片: slice_{pa['fastest_slice']['slice_index']} ({pa['fastest_slice']['total_time']:.3f}s)")
                    
                    if 'slice_details' in timing_data:
                        slice_count = len(timing_data['slice_details'])
                        print(f"  📋 记录了 {slice_count} 个切片的详细计时数据")
            
            except Exception as e:
                print(f"  ⚠️  读取详细计时文件出错: {e}")
        else:
            print("⚠️  详细计时记录文件未生成")
        
        print()
        print("🎉 测试完成！")
        print("\n💡 提示:")
        print("  - 详细计时数据已保存到 output_json/slice_ocr_detailed_timing.json")
        print("  - 可以通过修改 Config.enable_detailed_ocr_timing = False 来禁用详细计时")
        print("  - 详细计时会略微增加处理时间，但能提供更精确的性能分析")
    
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def test_comparison():
    """对比测试：普通模式 vs 详细计时模式"""
    print("\n" + "="*60)
    print("🔄 开始对比测试：普通模式 vs 详细计时模式")
    print("="*60)
    
    # 查找测试图片
    test_image_dirs = ["images", "test_images", "."]
    test_image_patterns = ["*.png", "*.jpg", "*.jpeg"]
    
    test_image_path = None
    for test_dir in test_image_dirs:
        if not os.path.exists(test_dir):
            continue
        for pattern in test_image_patterns:
            images = list(Path(test_dir).glob(pattern))
            if images:
                test_image_path = str(images[0])  # 使用第一个找到的图片
                break
        if test_image_path:
            break
    
    if not test_image_path:
        print("❌ 未找到测试图片，跳过对比测试")
        return
    
    results = {}
    
    # 测试两种模式
    for mode_name, enable_timing in [("普通模式", False), ("详细计时模式", True)]:
        print(f"\n🧪 测试 {mode_name}...")
        
        config = Config()
        config.enable_detailed_ocr_timing = enable_timing
        
        try:
            processor = LongImageOCR(config_path="./default_rapidocr.yaml")
            processor.config = config
            
            start_time = time.time()
            result = processor.process_long_image(test_image_path)
            end_time = time.time()
            
            results[mode_name] = {
                'total_time': end_time - start_time,
                'result': result
            }
            
            print(f"  ✅ {mode_name}完成，耗时: {results[mode_name]['total_time']:.2f}秒")
            
        except Exception as e:
            print(f"  ❌ {mode_name}测试失败: {e}")
            results[mode_name] = None
    
    # 对比结果
    print(f"\n📊 对比结果:")
    if all(results.values()):
        normal_time = results["普通模式"]['total_time']
        detailed_time = results["详细计时模式"]['total_time']
        overhead = detailed_time - normal_time
        overhead_percent = (overhead / normal_time) * 100
        
        print(f"  - 普通模式耗时: {normal_time:.2f}秒")
        print(f"  - 详细计时模式耗时: {detailed_time:.2f}秒")
        print(f"  - 时间开销: {overhead:.2f}秒 ({overhead_percent:.1f}%)")
        
        if overhead_percent < 5:
            print(f"  📈 结论: 详细计时的性能开销很小 (<5%)")
        elif overhead_percent < 15:
            print(f"  📈 结论: 详细计时的性能开销适中 (<15%)")
        else:
            print(f"  📈 结论: 详细计时的性能开销较高 (>15%)")
    else:
        print("  ⚠️  对比测试不完整，无法得出结论")

if __name__ == "__main__":
    print("🔍 详细OCR计时功能测试")
    print("本脚本将测试新增的OCR详细计时功能，分别记录每个切片的detection和recognition推理时间")
    print()
    
    # 基本测试
    test_detailed_ocr_timing()
    
    # 对比测试（可选）
    user_input = input("\n是否进行对比测试？(y/n，默认n): ").lower().strip()
    if user_input in ['y', 'yes']:
        test_comparison()
    
    print(f"\n✨ 所有测试完成！")