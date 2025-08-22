#!/usr/bin/env python3
"""
OCR结果可视化功能测试脚本
测试每个切片和原图的OCR结果可视化功能
"""

import sys
import os
import time
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.main import LongImageOCR
from src.utils.config import Config

def test_ocr_visualization():
    """测试OCR结果可视化功能"""
    print("🎨 开始测试OCR结果可视化功能")
    print("="*60)
    
    # 创建配置，启用详细计时（这样会同时生成可视化）
    config = Config()
    config.enable_detailed_ocr_timing = True  # 启用详细计时和可视化
    
    print(f"📋 配置信息:")
    print(f"  - 详细计时模式: {'✅ 启用' if config.enable_detailed_ocr_timing else '❌ 禁用'}")
    print(f"  - OCR配置文件: {config.ocr.config_path}")
    print(f"  - 文本置信度阈值: {config.ocr.text_score_threshold}")
    print()
    
    # 初始化处理器
    try:
        processor = LongImageOCR(config_path="./default_rapidocr.yaml")
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
        print("🔄 开始OCR处理和可视化...")
        start_time = time.time()
        
        result = processor.process_long_image(test_image_path)
        
        end_time = time.time()
        total_processing_time = end_time - start_time
        
        print(f"\n✅ 处理完成！总耗时: {total_processing_time:.2f}秒")
        print()
        
        # 显示处理结果摘要
        print("📊 处理结果摘要:")
        print(f"  - 总OCR项数: {result.get('total_ocr_items', 0)}")
        print(f"  - 总头像数: {result.get('total_avatars', 0)}")
        print(f"  - 聊天消息数: {result.get('chat_messages', 0)}")
        print(f"  - 我的消息数: {result.get('my_messages', 0)}")
        print()
        
        # 检查生成的可视化文件
        print("📁 检查生成的可视化文件:")
        
        # 1. 切片级别的可视化文件
        debug_dir = Path("output_images/debug")
        if debug_dir.exists():
            slice_vis_files = list(debug_dir.glob("slice_*_ocr_results.jpg"))
            print(f"   🔍 切片OCR可视化: {len(slice_vis_files)} 个文件")
            for i, vis_file in enumerate(sorted(slice_vis_files)[:5]):  # 只显示前5个
                file_size = vis_file.stat().st_size / 1024  # KB
                print(f"     {i+1}. {vis_file.name} ({file_size:.1f} KB)")
            if len(slice_vis_files) > 5:
                print(f"     ... 还有 {len(slice_vis_files) - 5} 个文件")
        else:
            print("   ❌ 未找到切片可视化目录")
        
        # 2. 原图级别的可视化文件
        full_vis_file = Path("output_images/full_image_ocr_results.jpg")
        if full_vis_file.exists():
            file_size = full_vis_file.stat().st_size / 1024 / 1024  # MB
            print(f"   🖼️  原图OCR可视化: {full_vis_file.name} ({file_size:.2f} MB)")
        else:
            print("   ❌ 未找到原图OCR可视化文件")
        
        # 3. 详细计时数据
        timing_file = Path("output_json/slice_ocr_detailed_timing.json")
        if timing_file.exists():
            file_size = timing_file.stat().st_size / 1024  # KB
            print(f"   📊 详细计时数据: {timing_file.name} ({file_size:.1f} KB)")
        
        print()
        
        # 显示可视化功能的效果说明
        print("🎨 可视化功能说明:")
        print("   ✅ 切片可视化:")
        print("     - 每个切片单独显示OCR结果")
        print("     - 绿色框：检测边界")
        print("     - 黑色文字：识别结果（框上方）")
        print("     - 绿色标签：序号和置信度（框下方）")
        print()
        print("   ✅ 原图可视化:")
        print("     - 在完整原图上显示所有OCR结果")
        print("     - 统一的颜色方案和布局")
        print("     - 适合查看整体识别效果")
        print()
        
        # 提供查看建议
        print("💡 查看建议:")
        print("   1. 打开切片可视化文件查看每个切片的详细识别效果")
        print("   2. 打开原图可视化文件查看整体识别分布")
        print("   3. 对比原图和可视化结果验证识别准确性")
        print()
        
        print("🎉 OCR可视化测试完成！")
    
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def check_visualization_quality():
    """检查可视化质量"""
    print("\n" + "="*60)
    print("🔍 检查可视化质量")
    print("="*60)
    
    # 检查生成的文件
    files_to_check = [
        ("切片可视化目录", Path("output_images/debug")),
        ("原图可视化", Path("output_images/full_image_ocr_results.jpg")),
        ("详细计时数据", Path("output_json/slice_ocr_detailed_timing.json"))
    ]
    
    all_good = True
    
    for file_desc, file_path in files_to_check:
        if file_path.is_dir():
            # 检查目录中的文件
            files = list(file_path.glob("*.jpg"))
            if files:
                total_size = sum(f.stat().st_size for f in files) / 1024 / 1024  # MB
                print(f"✅ {file_desc}: {len(files)} 个文件，总大小 {total_size:.2f} MB")
            else:
                print(f"❌ {file_desc}: 目录存在但无文件")
                all_good = False
        elif file_path.exists():
            file_size = file_path.stat().st_size / 1024  # KB
            print(f"✅ {file_desc}: 存在 ({file_size:.1f} KB)")
        else:
            print(f"❌ {file_desc}: 不存在")
            all_good = False
    
    if all_good:
        print("\n🎊 所有可视化文件都已正确生成！")
    else:
        print("\n⚠️  部分可视化文件缺失，请检查处理过程是否有错误")
    
    # 给出使用建议
    print(f"\n📖 使用说明:")
    print(f"   - 切片可视化: output_images/debug/slice_*_ocr_results.jpg")  
    print(f"   - 原图可视化: output_images/full_image_ocr_results.jpg")
    print(f"   - 可以用图像查看器打开这些文件查看OCR识别效果")

if __name__ == "__main__":
    print("🎨 OCR结果可视化功能测试")
    print("本脚本将测试新增的OCR结果可视化功能，包括切片和原图的可视化")
    print()
    
    # 基本测试
    test_ocr_visualization()
    
    # 质量检查
    check_visualization_quality()
    
    print(f"\n✨ 测试完成！")