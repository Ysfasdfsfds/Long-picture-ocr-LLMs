# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个OCR长图智能处理系统，专门用于处理长聊天截图的OCR识别与分析工具。系统能够自动识别微信、飞书、钉钉、蓝信等主流聊天工具的截图，将长图中的聊天记录转换为结构化数据，并支持通过大语言模型(LLM)进行智能问答。

## 运行和开发命令

### 基本运行
```bash
# 运行重构版主控制器（推荐）
python controller_refactored.py

# 使用Python API
python -c "from src.main import LongImageOCR; processor = LongImageOCR(); processor.process_long_image('images/your_image.png')"
```

### 依赖安装
```bash
# 基础OCR依赖
pip install rapidocr-onnxruntime opencv-python numpy pillow

# LLM功能依赖
pip install requests

# 可选：安装Ollama进行本地LLM问答
# Linux/MacOS: curl -fsSL https://ollama.ai/install.sh | sh
# ollama pull qwen3:8b
```

### 测试和验证
```bash
# 运行重构对比测试（验证新旧版本一致性）
python compare_versions.py

# 运行重构测试脚本
python test_refactoring.py
```

## 代码架构

### 分层架构设计
系统采用分层架构，核心处理流程如下：

1. **用户输入层**: 长图文件 + 配置参数
2. **主控制器层**: controller_refactored.py（流程控制 + LLM集成）
3. **核心处理层**: ImageSlicer（图像切片）→ OCREngine（OCR识别）→ ChatAnalyzer（聊天分析）
4. **处理器层**: AvatarDetector（头像检测）→ ContentMarker（内容标记）→ Deduplicator（去重处理）
5. **输出层**: JsonExporter（JSON导出）+ Visualizer（可视化输出）

### 目录结构
```
src/
├── core/                   # 核心模块
│   ├── image_slicer.py     # 图像切片处理
│   ├── ocr_engine.py       # OCR引擎封装(基于RapidOCR)
│   └── chat_analyzer.py    # 聊天消息分析
├── processors/             # 处理器模块  
│   ├── avatar_detector.py  # 头像检测
│   ├── content_marker.py   # 内容标记
│   └── deduplicator.py     # 去重处理
├── models/                 # 数据模型(dataclass)
│   ├── ocr_result.py       # OCR结果模型
│   ├── chat_message.py     # 聊天消息模型
│   └── slice_info.py       # 切片信息模型
├── exporters/              # 导出器
│   └── json_exporter.py    # JSON导出
├── utils/                  # 工具模块
│   ├── config.py           # 配置管理(dataclass)
│   ├── image_utils.py      # 图像工具
│   ├── type_converter.py   # 类型转换
│   ├── compatibility_patch.py # 兼容性补丁
│   └── visualization.py    # 可视化
└── main.py                 # 主程序入口
```

### 核心组件说明

#### 配置系统 (src/utils/config.py)
- 使用dataclass实现强类型配置管理
- 包含OCRConfig、ImageConfig、AvatarConfig、ContentConfig、OutputConfig
- 支持默认值和配置验证

#### 数据模型 (src/models/)
- OCRItem: OCR识别结果项
- ChatMessage: 聊天消息（支持chat/time/group_name等类型）
- SliceInfo: 切片信息
- 全部使用dataclass，提供类型安全和序列化支持

#### 核心处理流程
1. **ImageSlicer**: 将长图按高度切片（默认1200px，重叠200px）
2. **OCREngine**: 基于RapidOCR进行文字识别
3. **AvatarDetector**: 检测聊天头像（正方形比例检测）
4. **ContentMarker**: 根据颜色和位置标记内容类型（时间/昵称/内容）
5. **Deduplicator**: 去除重叠区域的重复识别结果
6. **ChatAnalyzer**: 分析并构建聊天会话结构

## 配置文件

### default_rapidocr.yaml
RapidOCR的配置文件，主要参数：
- text_score: 0.5 (文本置信度阈值)
- limit_side_len: 1200 (图像最大边长)
- thresh: 0.3 (检测阈值)
- box_thresh: 0.5 (框体阈值)

### 代码中的主要配置项
```python
# 图像处理配置
slice_height = 1200        # 切片高度
overlap = 200             # 重叠区域大小
binary_threshold = 230    # 二值化阈值

# OCR配置  
text_score_threshold = 0.65  # 文本置信度阈值

# 头像检测配置
square_ratio_min = 0.9    # 正方形比例最小值
square_ratio_max = 1.1    # 正方形比例最大值

# 内容标记配置
green_ratio_threshold = 0.2  # 绿色背景比例阈值（微信）
blue_ratio_threshold = 0.3   # 蓝色背景比例阈值（飞书）
```

## 输入输出

### 输入
- 支持格式: PNG, JPG, JPEG
- 推荐尺寸: 宽度 < 2000px，高度不限
- 内容要求: 清晰的聊天界面截图
- 默认输入路径: images/

### 输出目录结构
```
output_images/              # 图像输出
├── slice_*.jpg            # 切片图像
├── slice_ocr_result_*.jpg # OCR识别结果可视化
├── process_summary.jpg    # 处理流程总结图
└── debug/                 # 调试图像
    └── avatars_slice_*.jpg

output_json/               # JSON输出
├── marked_ocr_results_original.json    # 标记后的OCR结果
├── structured_chat_messages.json       # 结构化聊天消息
└── summary_data_original.json          # 处理摘要
```

## LLM集成

### Ollama集成
- 默认模型: qwen3:8b
- API端点: http://localhost:11434/api/generate
- 功能: 对识别的聊天记录进行智能问答
- 提供结构化引用和总结功能

### 问答模式
系统会根据OCR识别的聊天记录，提供基于事实的问答服务，包括：
- 消息检索和引用
- 时间线分析
- 说话人信息提取
- 上下文关联分析

## 重构特性

这是原controller.py的重构版本，主要改进：
- 模块化架构（300+行单文件 → 分层模块）
- 消除全局变量，使用依赖注入
- 完整类型注解支持
- 数据类模型替代字典操作
- 兼容性补丁确保与原版输出100%一致

## 支持的聊天平台

- 微信（绿色气泡检测）
- 飞书（蓝色背景检测）  
- 钉钉
- 蓝信
- 可通过修改ContentMarker扩展更多平台