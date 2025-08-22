# OCR长图智能处理系统 - Experiments版本

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20MacOS-lightgrey.svg)](https://github.com)
[![Version](https://img.shields.io/badge/version-Experiments-orange.svg)](https://github.com)

> **⚠️ 注意：这是实验版本 (Experiments Version)**  
> 本版本专门用于测试和对比不同OCR参数配置的效果，包含完整的参数矩阵实验功能，用于选择最适合特定场景的参数组合。如需稳定版本，请使用主分支。

## 🚀 项目概述

OCR长图智能处理系统是一个专门用于处理长聊天截图的OCR识别与分析工具。系统能够自动识别微信、飞书、钉钉、蓝信等主流聊天工具的截图，将长图中的聊天记录转换为结构化数据，并支持通过大语言模型进行智能问答。

**实验版本特色功能：**
- 🔬 **参数矩阵实验**：自动测试多种OCR参数组合
- 📊 **性能对比分析**：详细的处理时间、准确率、内存使用对比
- 🎯 **最优参数推荐**：基于实验结果推荐最佳参数配置
- 📈 **可视化报告**：生成详细的实验结果分析报告

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

## 🔬 实验功能 (Experiments版本特有)

### 参数矩阵实验

本版本新增的核心功能：通过 `run_experiments.py` 自动测试多种参数组合，找出最优配置。

#### 实验参数配置

系统会自动测试以下参数组合：

**OCR检测参数 (RapidOCR配置):**
- `limit_type`: max/min (图像尺寸限制类型)
- `limit_side_len`: 200, 400, 600, 800, 1000, 1200 (边长限制值)

**图像切片参数:**
- 切片尺寸: 400x400, 600x600, 800x800, 1000x1000, 1200x1200

总计 **60种参数组合** 的全面对比测试。

#### 运行实验

```bash
# 运行完整参数矩阵实验
python run_experiments.py

# 实验过程中可以查看进度
tail -f experiments_results/comparison_report.txt

# 实验完成后查看结果摘要
cat experiments_results/summary_report.json
```

#### 实验输出

每个实验配置会生成：
- `experiments_results/{config_name}/`
  - `experiment_result.json` - 实验数据
  - `timing_report.txt` - 性能报告
  - `config.yaml` - 使用的参数配置
  - `output_images/` - 处理结果图像
  - `output_json/` - 结构化数据

全局分析报告：
- `summary_report.json` - 所有实验汇总
- `comparison_report.txt` - 参数对比分析
- `recognition_detailed_report.txt` - 识别准确率分析

#### 实验结果分析

基于实验结果，你可以：
1. **性能优化**：找出处理速度最快的参数组合
2. **准确率优化**：找出识别准确率最高的配置
3. **内存优化**：找出内存使用最少的参数
4. **平衡配置**：找出速度、准确率、内存的最佳平衡点

## 📖 使用说明

### 快速开始

1. **实验模式 - 寻找最优参数**

```bash
# 先运行参数实验（推荐）
python run_experiments.py
# 根据实验结果选择最适合的参数配置
```

2. **基础使用 - 处理单张长图**

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

### 生产应用场景
1. **会议记录整理**：将聊天中的会议讨论整理成结构化记录
2. **客服对话分析**：分析客服聊天记录，提取关键信息
3. **社交数据挖掘**：从群聊记录中提取话题、观点等信息
4. **合规审查**：检查聊天记录中的敏感信息
5. **知识提取**：从技术讨论中提取问题和解决方案

### 实验版本专用场景
1. **参数调优**：为特定应用场景找出最优OCR参数
2. **性能基准测试**：建立不同配置的性能基准
3. **算法对比研究**：研究不同参数对识别效果的影响
4. **硬件配置优化**：为特定硬件找出最适合的参数配置
5. **业务场景适配**：针对特定聊天平台或图像特征调优参数

## ❓ 常见问题

### 实验版本专有问题

### Q1: 如何选择最优的实验参数？

A: 运行 `python run_experiments.py` 后，参考以下指标：
- **速度优先**：查看 `timing_report.txt` 中处理时间最短的配置
- **准确率优先**：查看 `recognition_detailed_report.txt` 中置信度最高的配置
- **内存优先**：选择切片尺寸较小的配置（如400x400）
- **平衡配置**：推荐 `max_1200_slice_800x800` 作为起点

### Q2: 实验需要多长时间？

A: 60种参数组合的完整实验约需要：
- 小图片（< 5MB）：30-60分钟  
- 中等图片（5-20MB）：1-3小时
- 大图片（> 20MB）：3-8小时

可以通过修改 `run_experiments.py` 中的参数列表缩减实验范围。

### Q3: 如何分析实验结果？

A: 查看关键文件：
```bash
# 查看最快配置
grep -A5 "处理时间最短" experiments_results/comparison_report.txt

# 查看最准确配置  
grep -A5 "平均置信度最高" experiments_results/comparison_report.txt

# 查看综合排名
cat experiments_results/summary_report.json | jq '.ranking'
```

### 通用问题

### Q4: 如何处理超大图片（高度>10000px）？

A: 系统会自动进行切片处理，无需手动干预。实验版本可测试不同切片参数的效果。

### Q5: OCR识别准确率不高怎么办？

A: **实验版本优势**：
1. 运行参数实验找出最优配置
2. 对比不同`limit_side_len`的效果
3. 测试不同切片尺寸对边界识别的影响
4. 查看实验报告中的准确率分析

### Q6: 处理速度慢怎么优化？

A: **实验版本方法**：
1. 查看 `experiments_results/comparison_report.txt` 中的性能排名
2. 选择处理时间最短的参数配置
3. 根据硬件配置选择合适的切片尺寸

### Q7: 如何批量处理多张图片？

A: 结合实验结果的批处理脚本：
```python
import os
from src.main import LongImageOCR

# 使用实验得出的最优配置
processor = LongImageOCR(config_path="experiments_results/best_config.yaml")

for img in os.listdir("images"):
    if img.endswith(('.png', '.jpg')):
        processor.process_long_image(f"images/{img}")
```

## 🔄 版本说明

### Experiments版本 vs 主版本

| 功能 | Experiments版本 | 主版本 |
|------|-----------------|--------|
| 基础OCR处理 | ✅ | ✅ |
| LLM智能问答 | ✅ | ✅ |
| C++高性能版本 | ✅ | ✅ |
| **参数矩阵实验** | ✅ **独有** | ❌ |
| **性能对比分析** | ✅ **独有** | ❌ |
| **最优参数推荐** | ✅ **独有** | ❌ |
| 稳定性 | 实验性 | 稳定 |
| 生产推荐 | 研究/调优 | 生产部署 |

### 使用建议

- **研究和调优阶段**：使用本 Experiments 版本
- **生产部署阶段**：基于实验结果切换到主版本
- **参数优化需求**：使用本版本找出最优配置后应用到主版本

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出建议！

### Experiments版本特殊贡献

- **参数组合扩展**：在 `run_experiments.py` 中添加新的参数测试
- **评估指标改进**：提升实验结果的评估维度
- **可视化增强**：改进实验结果的展示方式
- **性能基准**：贡献不同硬件环境的基准数据

### 贡献流程

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/ExperimentFeature`)
3. 提交更改 (`git commit -m 'Add experiment feature'`)
4. 推送到分支 (`git push origin feature/ExperimentFeature`)
5. 开启 Pull Request

### 开发规范

- 遵循PEP 8代码风格
- 添加类型注解
- 编写清晰的文档字符串
- **实验相关代码需包含详细的性能注释**

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