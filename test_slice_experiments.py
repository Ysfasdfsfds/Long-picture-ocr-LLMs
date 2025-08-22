#!/usr/bin/env python3
"""
切片尺寸参数实验测试脚本
测试小规模配置组合验证功能正确性
"""

import os
import sys

# 修改实验运行器的配置，只测试少量组合
def test_slice_experiments():
    """测试切片实验功能"""
    print("🧪 开始测试切片尺寸参数实验功能...")
    
    # 导入实验运行器
    from run_experiments import OCRExperimentRunner
    
    # 创建测试用的实验运行器
    runner = OCRExperimentRunner()
    
    # 只测试2个OCR配置 × 2个切片尺寸 = 4种组合
    test_configs = [
        {"limit_type": "max", "limit_side_len": 1200, "slice_height": 600, "slice_width": 600},
        {"limit_type": "max", "limit_side_len": 1200, "slice_height": 800, "slice_width": 800},
        {"limit_type": "min", "limit_side_len": 1200, "slice_height": 600, "slice_width": 600},
        {"limit_type": "min", "limit_side_len": 1200, "slice_height": 800, "slice_width": 800}
    ]
    
    # 替换实验配置
    runner.experiment_configs = test_configs
    runner.experiments_dir = runner.experiments_dir.parent / "test_experiments_results"
    
    # 使用测试图片
    image_path = "/home/kylin/桌面/Long-picture-ocr-LLMs-main_a/images/image copy 18.png"
    
    if not os.path.exists(image_path):
        print(f"❌ 测试图片不存在: {image_path}")
        return False
    
    try:
        print(f"📸 使用测试图片: {os.path.basename(image_path)}")
        print(f"🔢 测试配置数量: {len(test_configs)}")
        
        # 运行测试实验
        runner.run_all_experiments(image_path)
        
        print("\n✅ 切片实验功能测试成功！")
        print(f"📁 测试结果保存在: {runner.experiments_dir}")
        return True
        
    except Exception as e:
        print(f"\n❌ 切片实验功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_slice_experiments()
    if success:
        print("\n🎉 准备运行完整实验 (108种配置)...")
        input("按Enter键继续，或Ctrl+C取消...")
        
        print("\n🚀 开始运行完整实验...")
        os.system("python run_experiments.py")
    else:
        print("\n⚠️  测试失败，请检查配置后重试")
        sys.exit(1)