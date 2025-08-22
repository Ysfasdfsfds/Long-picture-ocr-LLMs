# OCR长图智能处理系统

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20MacOS-lightgrey.svg)](https://github.com)

## 🚀 项目概述

OCR长图智能处理系统是一个专门用于处理长聊天截图的OCR识别与分析工具。系统能够自动识别微信、飞书、钉钉、蓝信等主流聊天工具的截图，将长图中的聊天记录转换为结构化数据，并支持通过大语言模型进行智能问答。

### 核心功能

- **🖼️ 智能长图切片**：自动将超长聊天截图切分为多个片段，支持重叠区域处理
- **📝 高精度OCR识别**：基于RapidOCR引擎，准确识别中英文混合文本
- **👤 头像智能检测**：自动定位和识别聊天头像位置，用于消息归属判断
- **🏷️ 内容智能标记**：自动识别时间戳、昵称、聊天内容、系统消息等元素
- **🔍 去重处理**：智能去除重叠区域的重复识别结果
- **📊 结构化输出**：将识别结果整理为JSON格式的结构化数据
- **🤖 LLM集成**：支持接入Ollama等本地大模型进行智能问答
- **🎯 多平台适配**：自动识别并适配微信、飞书、钉钉、蓝信等不同平台的界面特征

### 技术特点

- **模块化架构**：采用分层架构设计，核心功能解耦，易于扩展
- **类型安全**：全面使用Python类型注解，提供更好的IDE支持
- **配置灵活**：支持YAML配置文件，参数可调节
- **调试友好**：提供丰富的调试输出和可视化功能
- **性能优化**：支持批量处理，内存使用优化

## 🏗️ 系统架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        用户输入层                             │
│                   (长图文件 + 配置参数)                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      主控制器层                               │
│                 (controller_refactored.py)                   │
│                    - 流程控制                                 │
│                    - LLM集成                                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      核心处理层                               │
│  ┌──────────────┬──────────────┬──────────────┐            │
│  │ ImageSlicer  │  OCREngine   │ ChatAnalyzer │            │
│  │   图像切片    │   OCR识别     │   聊天分析    │            │
│  └──────────────┴──────────────┴──────────────┘            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      处理器层                                 │
│  ┌─────────────┬─────────────┬──────────────┐              │
│  │AvatarDetector│ContentMarker│ Deduplicator │              │
│  │  头像检测     │  内容标记    │    去重处理   │              │
│  └─────────────┴─────────────┴──────────────┘              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       输出层                                  │
│  ┌──────────────────┬────────────────────┐                 │
│  │   JsonExporter   │    Visualizer      │                 │
│  │    JSON导出       │    可视化输出       │                 │
│  └──────────────────┴────────────────────┘                 │
└─────────────────────────────────────────────────────────────┘
```

### 目录结构

```
ocr_long_picture-main/
├── src/                        # 源代码目录
│   ├── core/                   # 核心模块
│   │   ├── image_slicer.py     # 图像切片处理
│   │   ├── ocr_engine.py       # OCR引擎封装
│   │   └── chat_analyzer.py    # 聊天消息分析
│   ├── processors/             # 处理器模块
│   │   ├── avatar_detector.py  # 头像检测
│   │   ├── content_marker.py   # 内容标记
│   │   └── deduplicator.py     # 去重处理
│   ├── models/                 # 数据模型
│   │   ├── ocr_result.py       # OCR结果模型
│   │   ├── chat_message.py     # 聊天消息模型
│   │   └── slice_info.py       # 切片信息模型
│   ├── exporters/              # 导出器
│   │   ├── json_exporter.py    # JSON导出
│   │   └── visualization.py    # 可视化
│   ├── utils/                  # 工具模块
│   │   ├── config.py           # 配置管理
│   │   ├── image_utils.py      # 图像工具
│   │   └── type_converter.py   # 类型转换
│   └── main.py                 # 主程序入口
├── images/                     # 输入图像目录
├── output_images/              # 输出图像目录
│   ├── debug/                  # 调试图像
│   └── slice_*.jpg            # 切片图像
├── output_json/                # JSON输出目录
├── logs/                       # 日志目录
├── controller_refactored.py    # 主控制器（支持LLM）
├── default_rapidocr.yaml       # OCR配置文件
├── LICENSE                     # Apache 2.0许可证
└── README.md                   # 本文档
```

## 🔧 安装指南

### 系统要求

- Python 3.8 或更高版本
- 操作系统：Windows / Linux / MacOS
- 内存：建议 4GB 以上
- 存储：至少 1GB 可用空间

### 依赖安装

1. **克隆项目**
```bash
git clone https://github.com/yourusername/ocr_long_picture.git
cd ocr_long_picture
```

2. **创建虚拟环境（推荐）**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/MacOS  
source venv/bin/activate
```

3. **安装依赖包**
```bash
pip install rapidocr-onnxruntime
pip install opencv-python
pip install numpy
pip install pillow
pip install requests  # 如需使用LLM功能
```

### Ollama安装（可选，用于LLM功能）

如果需要使用智能问答功能，请安装Ollama：

```bash
# Linux/MacOS
curl -fsSL https://ollama.ai/install.sh | sh

# 下载并启动模型
ollama pull qwen3:8b
ollama serve
```

## 📖 使用说明

### 快速开始

1. **基础使用 - 处理单张长图**

```bash
python controller_refactored.py
```

默认处理 `images/image copy 8.png` 文件。

2. **Python API调用**

```python
from src.main import LongImageOCR

# 初始化处理器
processor = LongImageOCR(config_path="./default_rapidocr.yaml")

# 处理长图
result = processor.process_long_image("images/your_image.png")

# 获取结构化聊天数据
chat_messages = processor.chat_session.to_dict()
```

### 命令行参数

修改 `controller_refactored.py` 中的参数：

```python
# 指定输入图像
image_path = r"images/your_image.png"

# 指定OCR配置文件
processor = LongImageOCR(config_path="./custom_config.yaml")

# 指定LLM模型（如使用Ollama）
model_name = "qwen3:8b"  # 可选: qwen3:8b, llama2, gemma等
```

### 配置选项

主要配置项（通过代码中的Config类设置）：

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

## 📁 输入输出说明

### 输入格式

- **支持格式**：PNG, JPG, JPEG
- **推荐尺寸**：宽度 < 2000px，高度不限
- **内容要求**：清晰的聊天界面截图

### 输出结构

#### 1. 图像输出 (`output_images/`)

- `slice_*.jpg` - 切片图像
- `slice_ocr_result_*.jpg` - OCR识别结果可视化
- `process_summary.jpg` - 处理流程总结图
- `debug/avatars_slice_*.jpg` - 头像检测调试图

#### 2. JSON输出 (`output_json/`)

**marked_ocr_results_original.json** - 标记后的OCR结果
```json
[
  {
    "text": "张三(昵称)",
    "box": [[100, 200], [300, 200], [300, 250], [100, 250]],
    "score": 0.98,
    "type": "nickname"
  }
]
```

**structured_chat_messages.json** - 结构化聊天消息
```json
{
  "chat_messages": [
    {
      "type": "chat",
      "昵称": "张三",
      "内容": "你好，请问今天的会议几点开始？"
    },
    {
      "type": "time",
      "time": "2025年1月6日 14:30"
    }
  ]
}
```

**summary_data_original.json** - 处理摘要
```json
{
  "total_ocr_items": 156,
  "total_avatars": 23,
  "total_messages": 45,
  "processing_time": 3.2
}
```

## 🤖 LLM集成使用

### 智能问答功能

系统集成了大语言模型，可以对识别的聊天记录进行智能问答：

```python
# 处理完图像后，自动进入问答模式
user_question = "张三提到了什么时间开会？"

# 系统会自动：
# 1. 分析聊天记录
# 2. 查找相关信息
# 3. 生成结构化回答
```

### 问答示例

**问**：李四说了什么关于项目进度的内容？

**答**：
```
引用内容：
- 说话人：李四
- 内容：项目进度已完成80%，预计下周三可以提交初版
- 时间：2025年1月6日 15:42
- 上下文指向分析：回复张三关于项目进度的询问

总结：李四汇报项目已完成80%，预计下周三提交初版。
```

## 🎯 使用场景

1. **会议记录整理**：将聊天中的会议讨论整理成结构化记录
2. **客服对话分析**：分析客服聊天记录，提取关键信息
3. **社交数据挖掘**：从群聊记录中提取话题、观点等信息
4. **合规审查**：检查聊天记录中的敏感信息
5. **知识提取**：从技术讨论中提取问题和解决方案

## ❓ 常见问题

### Q1: 如何处理超大图片（高度>10000px）？

A: 系统会自动进行切片处理，无需手动干预。可通过调整`slice_height`参数优化性能。

### Q2: OCR识别准确率不高怎么办？

A: 
1. 确保输入图片清晰，避免模糊或压缩过度
2. 调整`text_score_threshold`参数（降低阈值可能识别更多文本）
3. 检查RapidOCR配置文件中的参数设置

### Q3: 如何添加新的聊天平台支持？

A: 在`src/utils/config.py`中添加平台关键词，并在`content_marker.py`中实现相应的标记逻辑。

### Q4: 处理速度慢怎么优化？

A: 
1. 减小`slice_height`值（但可能影响上下文连续性）
2. 调整`overlap`值（减小重叠区域）
3. 使用更快的OCR模型配置

### Q5: 如何批量处理多张图片？

A: 编写简单的批处理脚本：
```python
import os
from src.main import LongImageOCR

processor = LongImageOCR()
for img in os.listdir("images"):
    if img.endswith(('.png', '.jpg')):
        processor.process_long_image(f"images/{img}")
```

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出建议！

### 贡献流程

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 开发规范

- 遵循PEP 8代码风格
- 添加类型注解
- 编写清晰的文档字符串
- 提供单元测试（如适用）

## 📄 许可证

本项目采用 Apache License 2.0 许可证。详见 [LICENSE](LICENSE) 文件。

基于 RapidOCR 项目，版权所有 (c) 2021 RapidOCR Authors。

## 🙏 致谢

- [RapidOCR](https://github.com/RapidAI/RapidOCR) - 提供高性能OCR引擎
- [OpenCV](https://opencv.org/) - 图像处理库
- [Ollama](https://ollama.ai/) - 本地大语言模型运行框架

## 📧 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 Issue: [GitHub Issues](https://github.com/yourusername/ocr_long_picture/issues)
- 邮件联系: your-email@example.com

---

**注意**：本工具仅供学习和研究使用，请勿用于非法用途。使用时请遵守相关法律法规和平台使用条款。