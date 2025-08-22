# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个OCR长图智能处理系统，专门用于处理聊天截图的OCR识别与分析。系统将长聊天截图切分、识别并转换为结构化数据，支持通过大语言模型进行智能问答。

## 核心架构

项目采用分层模块化架构：

### 主要组件
- **ImageSlicer** (`src/core/image_slicer.py`) - 图像切片处理
- **OCREngine** (`src/core/ocr_engine.py`) - 基于RapidOCR的文字识别
- **ChatAnalyzer** (`src/core/chat_analyzer.py`) - 聊天消息分析
- **AvatarDetector** (`src/processors/avatar_detector.py`) - 头像检测
- **ContentMarker** (`src/processors/content_marker.py`) - 内容智能标记
- **Deduplicator** (`src/processors/deduplicator.py`) - 去重处理
- **JsonExporter** (`src/exporters/json_exporter.py`) - 结构化数据导出

### 数据流程
1. 输入长图 → ImageSlicer切片
2. 每个切片 → OCREngine文字识别
3. OCR结果 → AvatarDetector检测头像
4. 结合结果 → ContentMarker智能标记（昵称、时间、内容等）
5. 标记结果 → Deduplicator去重
6. 最终数据 → ChatAnalyzer分析并JsonExporter导出

## 常用开发命令

### 运行系统
```bash
# 运行主处理程序（包含LLM问答功能）
python controller_refactored.py

# 或仅运行核心OCR处理
python -m src.main

# 运行实验和性能测试
python run_experiments.py

# 运行特定实验配置
python test_slice_experiments.py

# 调试和可视化测试
python test_ocr_visualization.py
python test_detailed_timing.py
python test_rapidocr_debug.py
```

### 环境管理和清理
```bash
# 清理输出目录（重新开始处理时使用）
rm -rf output_images/ output_json/ experiments_results/

# 清理特定实验结果
rm -rf experiments_results/max_1200_slice_400x400/

# 验证Python环境和RapidOCR导入
python -c "print('✅ Python解释器可正常使用'); import rapidocr; print('✅ RapidOCR可正常导入')"

# 检查可用测试图像
ls images/
```

### 依赖安装
```bash
# 基础依赖
pip install rapidocr-onnxruntime opencv-python numpy pillow requests

# 可选：安装完整的RapidOCR项目（用于高级功能）
cd RapidOCR-main/python
pip install -e .
```

### C++版本构建（高性能版本）
```bash
# 切换到C++项目目录
cd cpp_project

# 创建构建目录
mkdir -p build && cd build

# 生成构建文件并编译
cmake .. && make -j$(nproc)

# 运行C++版本
./bin/ocr_long_image ../images/test.png
```

### LLM集成（可选）
```bash
# 安装Ollama用于本地LLM
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen3:8b
ollama serve
```

### 实验和性能分析
```bash
# 运行完整参数对比实验
python run_experiments.py

# 查看实验结果摘要
cat experiments_results/summary_report.json

# 查看详细识别报告
cat experiments_results/recognition_detailed_report.txt

# 分析特定实验配置的时间性能
python test_detailed_timing.py
```

## 核心配置和参数

### RapidOCR引擎配置（default_rapidocr.yaml）
```yaml
Global:
    text_score: 0.5              # OCR文本置信度阈值
    max_side_len: 2000           # 图像处理最大边长
    min_side_len: 30             # 最小边长限制

Det:
    limit_side_len: 1200         # 检测模型边长限制
    box_thresh: 0.5              # 文本框检测阈值
    thresh: 0.3                  # 二值化阈值
    unclip_ratio: 1.6            # 文本框扩展比例
```

### 系统参数配置（src/utils/config.py）
这是通过Python类定义的配置，需要在代码中修改：
```python
@dataclass
class Config:
    # 图像切片设置
    slice_height: int = 1200     # 图像切片高度
    overlap: int = 200           # 切片重叠区域
    binary_threshold: int = 230  # 二值化阈值
    
    # OCR过滤设置
    text_score_threshold: float = 0.65  # OCR结果过滤阈值
    
    # 头像检测设置
    square_ratio_min: float = 0.9    # 正方形检测最小比例
    square_ratio_max: float = 1.1    # 正方形检测最大比例
    min_avatar_area: int = 400       # 最小头像区域
    
    # 平台颜色检测阈值
    green_ratio_threshold: float = 0.2   # 微信绿色比例阈值
    blue_ratio_threshold: float = 0.3    # 飞书蓝色比例阈值
```

### 如何修改配置
```bash
# 修改RapidOCR配置
vim default_rapidocr.yaml

# 修改系统配置（需要编辑Python代码）
vim src/utils/config.py
# 在Config类中调整相应参数值

# 或者在运行时动态修改
python -c "
from src.utils.config import Config
config = Config(slice_height=800, overlap=100)
# 使用自定义配置
"
```

### 关键参数调优指南
- **性能优化**：
  - 减小`slice_height` (800-1000) 提升速度，但可能影响上下文连续性
  - 减小`overlap` (100-150) 减少重叠处理，提升速度
  - 调整`limit_side_len` (800-1200) 控制OCR处理精度
- **准确性优化**：
  - 调低`text_score_threshold` (0.5-0.6) 识别更多文本，但可能增加噪声
  - 调高`box_thresh` (0.6-0.7) 提高文本框检测准确性
  - 增加`overlap` (300-400) 提高边界文本识别
- **内存优化**：
  - 调整`slice_height` 和`overlap`平衡内存使用和处理质量
  - 使用较小的`max_side_len` 减少内存占用

## 输入输出目录

### 输入文件
- `images/` - 输入图像目录（支持PNG/JPG格式）
  - 微信聊天截图：`image.png`, `image copy*.png`
  - 飞书聊天截图：`feishu*.png`
  - 测试用例：`test_new_image.png`, `current_test.png`

### 输出文件
- `output_images/` - 输出图像和调试图像
  - `slice_x*_y*.jpg` - 按坐标命名的切片图像
  - `debug/` - 详细调试可视化
    - `slice_*_detection_boxes.jpg` - 文本框检测结果
    - `slice_*_ocr_results.jpg` - OCR识别结果可视化
  - `full_image_ocr_results.jpg` - 完整OCR结果图
  - `process_summary.jpg` - 处理流程汇总
  
- `output_json/` - 结构化JSON输出
  - `structured_chat_messages.json` - 最终聊天消息结构
  - `marked_ocr_results_original.json` - 标记后的OCR结果
  - `summary_data_original.json` - 处理摘要统计
  - `slice_ocr_detailed_timing.json` - 详细时间分析
  - `timing_analysis.json` - 性能分析数据

### 实验结果目录
- `experiments_results/` - 实验和对比测试结果
  - `{limit_type}_{limit_value}_slice_{width}x{height}/` - 各参数配置的结果
    - `config.yaml` - 实验配置
    - `experiment_result.json` - 实验结果
    - `timing_report.txt` - 时间性能报告
    - `output_images/` - 实验处理图像
    - `output_json/` - 实验JSON输出
  - `summary_report.json` - 全部实验汇总
  - `comparison_report.txt` - 参数对比分析
  - `recognition_detailed_report.txt` - 识别准确率报告

## 开发注意事项

### 代码风格约定
- 使用Python类型注解（必须）
- 采用dataclass定义配置类
- 模块化设计，严格遵循单一职责原则
- 详细的logging输出便于调试
- 使用docstring描述所有公共方法

### 扩展聊天平台支持
在`src/processors/content_marker.py`中添加新的平台识别逻辑：
```python
# 添加新平台关键词到平台检测
platform_keywords = {
    'wechat': ['微信', '昵称'],
    'feishu': ['飞书', '回复'],
    'dingtalk': ['钉钉'],
    'new_platform': ['新平台关键词']  # 新增平台
}
```

### 实验和性能监控
项目支持详细的性能分析和参数调优实验：

#### 运行实验
```bash
# 运行完整参数矩阵实验
python run_experiments.py
# 实验会自动测试不同的limit_side_len和切片尺寸组合

# 查看实验进度（实验运行时）
tail -f experiments_results/comparison_report.txt
```

#### 分析实验结果
```bash
# 查看实验汇总
cat experiments_results/summary_report.json

# 查看性能对比
cat experiments_results/comparison_report.txt

# 查看识别准确率分析
cat experiments_results/recognition_detailed_report.txt

# 分析特定配置的详细数据
cat experiments_results/max_1200_slice_400x400/experiment_result.json
```

#### 实验配置含义
实验目录名格式：`{limit_type}_{limit_value}_slice_{width}x{height}`
- `limit_type`: max/min - RapidOCR的limit_side_len限制类型
- `limit_value`: 200/400/600/800/1000/1200 - 边长限制值
- `width`x`height`: 切片尺寸，如400x400, 800x800, 1200x1200

#### 性能指标解读
- **处理时间**: 总OCR处理时间，不含图像IO和分析
- **识别项目数**: 检测到的文本框数量
- **平均置信度**: OCR识别结果的平均可信度
- **内存使用**: 处理过程中的峰值内存占用

## 常见问题和故障排除

### 环境问题
**Q: ImportError: No module named 'rapidocr'**
```bash
# 确保安装了正确的RapidOCR包
pip install rapidocr-onnxruntime
# 或者安装完整版本
cd RapidOCR-main/python && pip install -e .
```

**Q: OpenCV相关错误**
```bash
# Linux下安装OpenCV
sudo apt install libopencv-dev python3-opencv
# 或使用pip
pip install opencv-python opencv-contrib-python
```

**Q: 中文字体显示问题**
```bash
# 确保系统安装了中文字体
sudo apt install fonts-wqy-zenhei fonts-wqy-microhei  # Ubuntu
# Windows下复制字体文件到项目目录：ShanHaiJiGuSongKe-JianFan-2.ttf
```

### 处理问题
**Q: OCR识别准确率低**
- 检查输入图片清晰度，避免过度压缩
- 调整`text_score_threshold`参数（降低到0.5-0.6）
- 调整RapidOCR配置中的`box_thresh`参数
- 确保图片中文字大小合适（不要太小）

**Q: 处理速度慢**
- 减小`slice_height`参数（800-1000）
- 减小`overlap`参数（100-150）
- 调小`limit_side_len`参数（800-1000）
- 考虑使用C++版本获得更好性能

**Q: 内存不足**
- 减小`slice_height`参数
- 减小切片尺寸（使用600x600而不是1200x1200）
- 清理输出目录释放空间：`rm -rf output_*/ experiments_results/`

### 输出问题
**Q: 生成的JSON文件为空或格式错误**
- 检查输入图片是否包含可识别的文本
- 确保图片格式支持（PNG/JPG）
- 检查`text_score_threshold`是否过高
- 查看debug目录中的可视化结果排查问题

**Q: 实验结果不一致**
- 确保每次实验前清理输出目录
- 检查实验配置文件是否正确
- 验证RapidOCR版本一致性
- 使用相同的测试图片进行对比

### C++版本问题
**Q: C++编译失败**
```bash
# 确保安装了必要的依赖
sudo apt install build-essential cmake libopencv-dev  # Ubuntu
brew install cmake opencv  # macOS
# Windows需要Visual Studio或MinGW环境
```

**Q: C++运行时找不到库**
```bash
# 设置动态库路径
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
# 或在CMake中设置RPATH
```

## 支持的聊天平台

系统自动识别并适配：微信、飞书、钉钉、蓝信等主流聊天工具界面特征。
- **微信**：绿色气泡检测 (`green_ratio_threshold: 0.2`)
- **飞书**：蓝色背景识别 (`blue_ratio_threshold: 0.3`)
- **通用**：基于头像和文本框位置关系的智能推断