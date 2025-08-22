#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专门用于调试RapidOCR内部det和rec的测试脚本
设置justMyCode=false后可以逐步调试到ONNX推理层
"""

import cv2
import numpy as np
from rapidocr import RapidOCR

def test_rapidocr_debug():
    """测试RapidOCR的det和rec推理过程"""
    
    # 初始化RapidOCR
    print("🔧 初始化RapidOCR...")
    ocr_engine = RapidOCR(config_path="./default_rapidocr.yaml")
    
    # 读取测试图像
    image_path = "images/feishu01.png"
    print(f"📖 读取测试图像: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取图像: {image_path}")
        return
    
    print(f"✅ 图像尺寸: {img.shape}")
    
    # 转换为RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    print("🎯 开始OCR推理...")
    print("💡 在这里设置断点，然后用F11步进到RapidOCR内部!")
    
    # ===== 关键调试点 =====
    # 在这行设置断点，启用justMyCode=false后可以进入det和rec内部
    result = ocr_engine(img_rgb)  # ← 设置断点在这里！
    # =====================
    
    print("🎉 OCR推理完成!")
    
    # 输出结果
    if result and hasattr(result, 'txts') and result.txts:
        print(f"📝 识别到 {len(result.txts)} 个文本:")
        for i, (txt, score) in enumerate(zip(result.txts, result.scores)):
            print(f"  {i+1}. [{score:.3f}] {txt}")
    else:
        print("⚠️ 未识别到文本")
    
    return result

def test_step_by_step_debug():
    """分步调试det和rec"""
    print("\n" + "="*50)
    print("🔍 分步调试模式")
    print("="*50)
    
    # 1. 创建RapidOCR实例
    ocr_engine = RapidOCR(config_path="./default_rapidocr.yaml")
    
    # 2. 读取图像
    image_path = "images/feishu01.png"
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    print("📍 步骤1: 准备调用文字检测(det)...")
    # 这里可以设置断点，然后步进到det内部
    det_result = ocr_engine.text_det(img_rgb)  # ← det调试点
    
    if det_result.boxes is not None:
        print(f"✅ 检测到 {len(det_result.boxes)} 个文字区域")
        
        print("📍 步骤2: 准备调用文字识别(rec)...")
        # 获取裁剪图像
        crop_imgs = []
        for box in det_result.boxes:
            # 这里简化处理，实际会有复杂的裁剪逻辑
            crop_imgs.append(img_rgb)  # 简化版本
        
        if crop_imgs:
            from rapidocr.ch_ppocr_rec import TextRecInput
            rec_input = TextRecInput(img=crop_imgs[:1])  # 只处理第一个
            
            # 这里可以设置断点，然后步进到rec内部
            rec_result = ocr_engine.text_rec(rec_input)  # ← rec调试点
            
            if rec_result.txts:
                print(f"✅ 识别结果: {rec_result.txts[0]}")
            else:
                print("❌ 识别失败")
    else:
        print("❌ 未检测到文字区域")

if __name__ == "__main__":
    print("🚀 RapidOCR Debug 测试")
    print("="*50)
    print("📋 调试说明:")
    print("1. 确保已设置 justMyCode: false")
    print("2. 在标记的位置设置断点")
    print("3. 使用F11逐步进入RapidOCR内部")
    print("4. 可以调试到det和rec的ONNX推理过程")
    print("="*50)
    
    # 完整流程测试
    test_rapidocr_debug()
    
    # 分步测试
    test_step_by_step_debug()