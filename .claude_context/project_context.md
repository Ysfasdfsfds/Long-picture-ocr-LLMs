# OCR长图智能处理系统 - 项目上下文

## 项目概述
这是一个专门用于处理长聊天截图的OCR识别与分析工具，支持微信、飞书、钉钉、蓝信等主流聊天工具的截图分析。

## 核心架构

### 主要模块
- **controller_refactored.py** - 主控制器，包含LLM集成
- **src/core/** - 核心处理模块
  - `image_slicer.py` - 图像切片处理
  - `ocr_engine.py` - OCR引擎封装
  - `chat_analyzer.py` - 聊天消息分析
- **src/processors/** - 数据处理器
  - `avatar_detector.py` - 头像检测
  - `content_marker.py` - 内容标记
  - `deduplicator.py` - 去重处理
- **src/models/** - 数据模型定义
- **src/exporters/** - 数据导出器
- **src/utils/** - 工具函数

### 处理流程
1. 长图输入 → 图像切片 → OCR识别
2. 头像检测 → 内容标记 → 去重处理  
3. 聊天消息结构化 → JSON导出 → LLM问答

## 技术栈
- **OCR引擎**: RapidOCR
- **图像处理**: OpenCV, PIL
- **数据处理**: NumPy
- **LLM集成**: Ollama
- **配置管理**: YAML

## 关键配置
- `default_rapidocr.yaml` - OCR配置
- `src/utils/config.py` - 全局配置管理

## 输入输出
- **输入**: PNG/JPG格式的聊天截图
- **输出**: 
  - `output_images/` - 处理后的图像
  - `output_json/` - 结构化JSON数据
  - `logs/` - 处理日志

## 代码质量要求
- 使用Python类型注解
- 遵循模块化架构设计
- 支持配置文件管理
- 提供调试和可视化功能