# OCR长图智能处理系统 Python到C++完整重写计划

## 项目概述

本计划旨在将现有的Python OCR长图智能处理系统完全重写为高性能的C++实现，同时保持与原Python版本的逻辑严格等价。该系统专门用于处理聊天截图的OCR识别与分析，支持微信、飞书、钉钉等多个平台。

## 一、技术栈对照表

| 功能模块 | Python技术栈 | C++技术栈 | 说明 |
|---------|------------|----------|------|
| 图像处理 | OpenCV-Python + NumPy | OpenCV C++ | 直接使用cv::Mat替代numpy数组 |
| OCR引擎 | RapidOCR Python | RapidOCR C++ SDK | 需要集成ONNX Runtime C++ |
| JSON处理 | json标准库 | nlohmann/json | 业界标准的C++ JSON库 |
| HTTP客户端 | requests | cpp-httplib | 轻量级HTTP库 |
| 文件系统 | pathlib | std::filesystem | C++17标准库 |
| 日志系统 | logging | spdlog | 高性能C++日志库 |
| 正则表达式 | re | std::regex | C++11标准库 |
| 数据结构 | dataclass | struct/class | 使用C++结构体和类 |

## 二、项目架构设计

### 2.1 目录结构

```
ocr_cpp_project/
├── CMakeLists.txt                 # 主构建文件
├── README.md                       # 项目说明
├── plan.md                         # 本重写计划
├── config/
│   ├── default_config.json        # 默认配置文件
│   └── rapidocr_config.yaml       # RapidOCR配置
├── src/
│   ├── main.cpp                   # 程序入口
│   ├── controller.h               # 主控制器头文件
│   ├── controller.cpp             # 主控制器实现
│   ├── core/                      # 核心功能模块
│   │   ├── image_slicer.h
│   │   ├── image_slicer.cpp
│   │   ├── ocr_engine.h
│   │   ├── ocr_engine.cpp
│   │   ├── chat_analyzer.h
│   │   └── chat_analyzer.cpp
│   ├── models/                    # 数据模型
│   │   ├── slice_info.h
│   │   ├── ocr_result.h
│   │   ├── chat_message.h
│   │   └── types.h               # 公共类型定义
│   ├── processors/                # 处理器模块
│   │   ├── avatar_detector.h
│   │   ├── avatar_detector.cpp
│   │   ├── content_marker.h
│   │   ├── content_marker.cpp
│   │   ├── deduplicator.h
│   │   └── deduplicator.cpp
│   ├── exporters/                 # 导出器模块
│   │   ├── json_exporter.h
│   │   └── json_exporter.cpp
│   └── utils/                     # 工具类
│       ├── config.h
│       ├── config.cpp
│       ├── visualization.h
│       ├── visualization.cpp
│       ├── logger.h               # 日志封装
│       └── logger.cpp
├── third_party/                   # 第三方库
│   ├── rapidocr/
│   ├── nlohmann_json/
│   ├── spdlog/
│   └── httplib/
├── tests/                         # 单元测试
│   ├── test_image_slicer.cpp
│   ├── test_ocr_engine.cpp
│   └── CMakeLists.txt
└── output/                        # 输出目录
    ├── images/
    ├── json/
    └── logs/
```

## 三、核心模块详细设计

### 3.1 数据模型 (models/)

#### SliceInfo (slice_info.h)
```cpp
struct SliceInfo {
    int slice_index;
    int start_y;
    int end_y;
    cv::Mat image;
    
    SliceInfo(int idx, int sy, int ey, const cv::Mat& img);
};
```

#### OCRItem (ocr_result.h)
```cpp
class OCRItem {
public:
    std::string text;
    std::vector<std::vector<float>> box;  // [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    float score;
    int slice_index;
    std::optional<std::string> original_text;
    bool is_virtual = false;
    
    float get_center_y() const;
    float get_min_y() const;
    float get_max_y() const;
    float get_min_x() const;
    float get_max_x() const;
    nlohmann::json to_json() const;
};
```

#### AvatarItem (ocr_result.h)
```cpp
struct AvatarItem {
    std::tuple<int, int, int, int> box;  // (x, y, width, height)
    int slice_index;
    
    nlohmann::json to_json() const;
};
```

#### ChatMessage (chat_message.h)
```cpp
enum class MessageType {
    CHAT,
    TIME,
    MY_CHAT,
    GROUP_NAME,
    SYSTEM_MESSAGE,
    RETRACT_MESSAGE,
    UNKNOWN
};

class ChatMessage {
public:
    MessageType type;
    std::optional<std::string> nickname;
    std::optional<std::string> content;
    std::optional<std::string> time;
    
    nlohmann::json to_json() const;
};
```

### 3.2 配置管理 (utils/config.h)

```cpp
class Config {
public:
    struct OCRConfig {
        std::string config_path = "default_rapidocr.yaml";
        float text_score_threshold = 0.5f;
    };
    
    struct ImageConfig {
        int slice_height = 900;
        int overlap = 100;
        int binary_threshold = 127;
        std::pair<int, int> gaussian_blur_size = {5, 5};
        float merge_distance_factor = 1.5f;
    };
    
    struct AvatarConfig {
        float square_ratio_min = 0.8f;
        float square_ratio_max = 1.2f;
        float strict_square_ratio_min = 0.85f;
        float strict_square_ratio_max = 1.15f;
        float iou_threshold = 0.1f;
        int x_crop_offset = 20;
    };
    
    // 其他配置子结构...
    
    OCRConfig ocr;
    ImageConfig image;
    AvatarConfig avatar;
    
    static Config from_file(const std::string& config_file);
    void to_file(const std::string& config_file) const;
    std::vector<std::string> get_time_patterns() const;
    std::vector<std::string> get_feishu_keywords() const;
};
```

### 3.3 核心引擎实现

#### ImageSlicer (core/image_slicer.h)
```cpp
class ImageSlicer {
private:
    Config config_;
    int slice_height_;
    int overlap_;
    std::filesystem::path output_dir_;
    
public:
    explicit ImageSlicer(const Config& config);
    
    std::pair<cv::Mat, std::vector<SliceInfo>> slice_image(const std::string& image_path);
    
private:
    std::vector<SliceInfo> perform_slicing(const cv::Mat& image);
    void save_slice(const SliceInfo& slice_info);
};
```

#### OCREngine (core/ocr_engine.h)
```cpp
class OCREngine {
private:
    Config config_;
    std::unique_ptr<RapidOCR> engine_;  // RapidOCR C++ SDK
    float text_score_threshold_;
    
public:
    explicit OCREngine(const Config& config);
    
    SliceOCRResult process_slice(const SliceInfo& slice_info, bool save_visualization = true);
    std::vector<SliceOCRResult> batch_process_slices(const std::vector<SliceInfo>& slice_infos);
    
private:
    void process_ocr_items(const RapidOCRResult& ocr_result, 
                          const SliceInfo& slice_info, 
                          SliceOCRResult& result);
    std::vector<std::vector<float>> adjust_box_to_original(
        const std::vector<std::vector<float>>& box, int start_y);
};
```

#### ChatAnalyzer (core/chat_analyzer.h)
```cpp
class ChatAnalyzer {
private:
    Config config_;
    std::vector<std::regex> time_patterns_;
    
public:
    explicit ChatAnalyzer(const Config& config);
    
    ChatSession analyze(const std::vector<OCRItem>& marked_ocr_items);
    
private:
    NicknameAnalysis analyze_nicknames(const std::vector<std::string>& marked_texts);
    void organize_messages(const std::vector<std::string>& marked_texts,
                          const NicknameAnalysis& nickname_analysis,
                          ChatSession& session);
    bool is_time_content(const std::string& text) const;
    bool is_group_name(const std::string& nickname, int position,
                      const NicknameAnalysis& nickname_analysis) const;
};
```

### 3.4 处理器模块

#### AvatarDetector (processors/avatar_detector.h)
```cpp
class AvatarDetector {
private:
    Config config_;
    std::filesystem::path debug_dir_;
    std::map<int, std::optional<std::tuple<int, int, int, int>>> slice_x_crop_values_;
    
    // 配置参数
    int binary_threshold_;
    cv::Size gaussian_blur_size_;
    float square_ratio_min_;
    float square_ratio_max_;
    float iou_threshold_;
    int x_crop_offset_;
    
public:
    explicit AvatarDetector(const Config& config);
    
    std::vector<AvatarItem> detect_avatars(const SliceInfo& slice_info, 
                                          std::optional<int> x_crop = std::nullopt);
    std::optional<int> calculate_x_crop(const std::vector<SliceInfo>& slice_infos);
    
private:
    cv::Mat preprocess_image(const cv::Mat& image);
    std::vector<cv::Rect> extract_contours_and_rects(const cv::Mat& binary_img,
                                                     const cv::Mat& original_img);
    std::vector<cv::Rect> apply_nms(const std::vector<cv::Rect>& rects);
    float calculate_iou(const cv::Rect& box1, const cv::Rect& box2);
    std::vector<cv::Rect> merge_nearby_boxes(const std::vector<cv::Rect>& rects,
                                            float merge_threshold);
    void save_debug_image(const cv::Mat& image, const std::vector<cv::Rect>& rects,
                         int slice_index);
};
```

#### ContentMarker (processors/content_marker.h)
```cpp
class ContentMarker {
private:
    Config config_;
    std::vector<std::regex> time_patterns_;
    std::vector<std::regex> system_patterns_;
    
public:
    explicit ContentMarker(const Config& config);
    
    std::vector<OCRItem> mark_content(const std::vector<OCRItem>& ocr_items,
                                     const std::vector<AvatarItem>& avatar_items,
                                     const cv::Mat& original_image);
    
private:
    void mark_nickname_and_content_wechat(std::vector<OCRItem>& ocr_items,
                                         const std::vector<AvatarItem>& avatar_items);
    void mark_nickname_and_content_feishu(std::vector<OCRItem>& ocr_items);
    bool is_time_text(const std::string& text);
    bool is_system_message(const std::string& text);
};
```

### 3.5 主控制器

#### LongImageOCR (controller.h)
```cpp
class LongImageOCR {
private:
    Config config_;
    std::unique_ptr<ImageSlicer> image_slicer_;
    std::unique_ptr<OCREngine> ocr_engine_;
    std::unique_ptr<AvatarDetector> avatar_detector_;
    std::unique_ptr<ContentMarker> content_marker_;
    std::unique_ptr<Deduplicator> deduplicator_;
    std::unique_ptr<ChatAnalyzer> chat_analyzer_;
    std::unique_ptr<JsonExporter> json_exporter_;
    std::unique_ptr<Visualizer> visualizer_;
    
    // 处理结果
    cv::Mat original_image_;
    std::vector<OCRItem> all_ocr_items_;
    std::vector<AvatarItem> all_avatar_items_;
    std::vector<OCRItem> marked_ocr_items_;
    std::unique_ptr<ChatSession> chat_session_;
    
public:
    explicit LongImageOCR(const std::string& config_path = "default_rapidocr.yaml");
    
    std::map<std::string, int> process_long_image(const std::string& image_path);
    std::string process_with_llm(const std::string& user_question,
                                const std::function<std::string(const std::string&, 
                                                               const std::string&)>& llm_processor = nullptr);
    
private:
    void early_platform_detection(const std::string& image_path);
    void print_final_platform_detection(const std::string& platform, bool is_feishu);
    std::vector<void> process_slices(const std::vector<SliceInfo>& slice_infos,
                                    std::optional<int> x_crop);
    void deduplicate_results();
    bool is_feishu_screenshot();
    void export_results();
    std::optional<std::tuple<int, int, int, int, int>> find_selected_box();
    std::map<std::string, int> create_summary();
};
```

## 四、关键技术实现细节

### 4.1 RapidOCR C++集成

```cpp
// ocr_engine.cpp 关键实现
#include <RapidOCR.h>

OCREngine::OCREngine(const Config& config) 
    : config_(config),
      text_score_threshold_(config.ocr.text_score_threshold) {
    
    // 初始化RapidOCR
    engine_ = std::make_unique<RapidOCR>();
    engine_->LoadModel(config.ocr.config_path);
}

SliceOCRResult OCREngine::process_slice(const SliceInfo& slice_info, 
                                       bool save_visualization) {
    // 转换图像格式
    cv::Mat rgb_image;
    cv::cvtColor(slice_info.image, rgb_image, cv::COLOR_BGR2RGB);
    
    // 执行OCR
    auto ocr_result = engine_->Detect(rgb_image);
    
    // 处理结果
    SliceOCRResult result(slice_info.slice_index, slice_info.start_y, slice_info.end_y);
    
    if (ocr_result.boxes && ocr_result.texts) {
        process_ocr_items(ocr_result, slice_info, result);
    }
    
    // 保存可视化结果
    if (save_visualization) {
        std::string vis_path = config_.output.output_images_dir + 
                              "/slice_ocr_result_" + 
                              std::to_string(slice_info.slice_index) + ".jpg";
        engine_->Visualize(vis_path, rgb_image, ocr_result);
    }
    
    return result;
}
```

### 4.2 中文正则表达式处理

```cpp
// chat_analyzer.cpp 中文处理示例
bool ChatAnalyzer::is_time_content(const std::string& text) const {
    // UTF-8编码的中文字符处理
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    std::wstring wide_text = converter.from_bytes(text);
    
    // 预编译的正则表达式
    static const std::vector<std::wregex> patterns = {
        std::wregex(L"\\d{4}年\\d{1,2}月\\d{1,2}日\\d{1,2}:\\d{2}"),
        std::wregex(L"(今天|昨天|前天|明天)(上午|下午|晚上)?\\d{1,2}:\\d{2}"),
        // 更多模式...
    };
    
    for (const auto& pattern : patterns) {
        if (std::regex_search(wide_text, pattern)) {
            return true;
        }
    }
    
    return false;
}
```

### 4.3 LLM集成 (Ollama API)

```cpp
// controller.cpp LLM处理
#include <httplib.h>

std::string process_with_ollama(const std::string& user_question,
                               const std::string& chat_messages) {
    httplib::Client cli("localhost", 11434);
    
    nlohmann::json payload = {
        {"model", "qwen3:8b"},
        {"prompt", "用户问题：" + user_question},
        {"system", generate_system_prompt(chat_messages)},
        {"stream", false}
    };
    
    auto res = cli.Post("/api/generate", 
                       payload.dump(), 
                       "application/json");
    
    if (res && res->status == 200) {
        auto response_json = nlohmann::json::parse(res->body);
        return response_json["response"];
    }
    
    return "处理失败";
}
```

### 4.4 内存管理策略

1. **智能指针使用**
   - 使用`std::unique_ptr`管理独占资源
   - 使用`std::shared_ptr`处理共享数据
   - RAII原则管理OpenCV Mat对象

2. **内存池优化**
   ```cpp
   class OCRItemPool {
   private:
       std::vector<std::unique_ptr<OCRItem>> pool_;
       std::queue<OCRItem*> available_;
       
   public:
       OCRItem* acquire();
       void release(OCRItem* item);
   };
   ```

3. **移动语义优化**
   ```cpp
   class SliceInfo {
   public:
       SliceInfo(SliceInfo&& other) noexcept
           : slice_index(other.slice_index),
             start_y(other.start_y),
             end_y(other.end_y),
             image(std::move(other.image)) {}
   };
   ```

## 五、并行处理优化

### 5.1 多线程图像切片处理

```cpp
#include <thread>
#include <future>

std::vector<SliceOCRResult> OCREngine::batch_process_slices(
    const std::vector<SliceInfo>& slice_infos) {
    
    const size_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::future<SliceOCRResult>> futures;
    
    for (const auto& slice_info : slice_infos) {
        futures.push_back(
            std::async(std::launch::async, 
                      [this, &slice_info]() {
                          return process_slice(slice_info);
                      })
        );
    }
    
    std::vector<SliceOCRResult> results;
    for (auto& future : futures) {
        results.push_back(future.get());
    }
    
    return results;
}
```

### 5.2 OpenCV GPU加速

```cpp
#ifdef OPENCV_GPU_SUPPORT
#include <opencv2/cudaimgproc.hpp>

cv::Mat AvatarDetector::preprocess_image_gpu(const cv::Mat& image) {
    cv::cuda::GpuMat gpu_image, gpu_gray, gpu_blurred, gpu_binary;
    
    gpu_image.upload(image);
    cv::cuda::cvtColor(gpu_image, gpu_gray, cv::COLOR_BGR2GRAY);
    cv::cuda::GaussianBlur(gpu_gray, gpu_blurred, gaussian_blur_size_, 0);
    cv::cuda::threshold(gpu_blurred, gpu_binary, binary_threshold_, 255, cv::THRESH_BINARY);
    
    cv::Mat result;
    gpu_binary.download(result);
    return 255 - result;
}
#endif
```

## 六、构建配置 (CMakeLists.txt)

```cmake
cmake_minimum_required(VERSION 3.14)
project(OCRLongImage VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找依赖
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

# 第三方库
add_subdirectory(third_party/nlohmann_json)
add_subdirectory(third_party/spdlog)
add_subdirectory(third_party/httplib)
add_subdirectory(third_party/rapidocr)

# 源文件
file(GLOB_RECURSE SOURCES 
    src/*.cpp
    src/core/*.cpp
    src/processors/*.cpp
    src/exporters/*.cpp
    src/utils/*.cpp
)

# 可执行文件
add_executable(ocr_long_image ${SOURCES})

# 链接库
target_link_libraries(ocr_long_image
    PRIVATE
    ${OpenCV_LIBS}
    Threads::Threads
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    httplib::httplib
    rapidocr
)

# 包含目录
target_include_directories(ocr_long_image
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)

# 编译选项
if(MSVC)
    target_compile_options(ocr_long_image PRIVATE /W4)
else()
    target_compile_options(ocr_long_image PRIVATE -Wall -Wextra -pedantic)
endif()

# 优化选项
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(ocr_long_image PRIVATE -O3 -march=native)
endif()

# 测试
enable_testing()
add_subdirectory(tests)
```

## 七、测试策略

### 7.1 单元测试框架

使用Google Test框架进行单元测试：

```cpp
// test_image_slicer.cpp
#include <gtest/gtest.h>
#include "core/image_slicer.h"

TEST(ImageSlicerTest, SliceSmallImage) {
    Config config;
    config.image.slice_height = 900;
    config.image.overlap = 100;
    
    ImageSlicer slicer(config);
    auto [image, slices] = slicer.slice_image("test_image.png");
    
    EXPECT_EQ(slices.size(), 1);
    EXPECT_EQ(slices[0].start_y, 0);
}

TEST(ImageSlicerTest, SliceLargeImage) {
    // 测试大图切片
}
```

### 7.2 集成测试

```cpp
// test_integration.cpp
TEST(IntegrationTest, ProcessWeChatScreenshot) {
    LongImageOCR processor("test_config.yaml");
    auto result = processor.process_long_image("wechat_test.png");
    
    EXPECT_GT(result["total_ocr_items"], 0);
    EXPECT_GT(result["chat_messages"], 0);
}
```

### 7.3 性能测试

```cpp
// benchmark.cpp
#include <benchmark/benchmark.h>

static void BM_OCRProcessing(benchmark::State& state) {
    OCREngine engine(Config());
    SliceInfo slice(0, 0, 900, test_image);
    
    for (auto _ : state) {
        auto result = engine.process_slice(slice);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_OCRProcessing);
```

## 八、部署和打包

### 8.1 Docker容器化

```dockerfile
FROM ubuntu:20.04

# 安装依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制源码
COPY . /app
WORKDIR /app

# 构建
RUN mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

# 运行
CMD ["./build/ocr_long_image"]
```

### 8.2 跨平台支持

- **Windows**: 使用Visual Studio 2019+和vcpkg管理依赖
- **macOS**: 使用Homebrew安装依赖
- **Linux**: 使用系统包管理器或从源码编译

## 九、性能对比预期

| 指标 | Python版本 | C++版本(预期) | 提升比例 |
|-----|-----------|--------------|---------|
| 单张图片处理时间 | 3-5秒 | 0.5-1秒 | 3-10x |
| 内存占用 | 500MB | 150MB | 3.3x |
| 并发处理能力 | 单线程 | 多线程 | N倍 |
| OCR准确率 | 95% | 95% | 相同 |

## 十、开发里程碑

### Phase 1: 基础架构 (第1-2周)
- [x] 项目结构搭建
- [ ] CMake配置
- [ ] 基础数据模型
- [ ] 配置管理系统
- [ ] 日志系统

### Phase 2: 核心功能 (第3-4周)
- [ ] ImageSlicer实现
- [ ] OCREngine集成
- [ ] AvatarDetector实现

### Phase 3: 业务逻辑 (第5-6周)
- [ ] ContentMarker实现
- [ ] ChatAnalyzer实现
- [ ] Deduplicator实现

### Phase 4: 集成测试 (第7周)
- [ ] 主控制器集成
- [ ] LLM接口实现
- [ ] 端到端测试

### Phase 5: 优化部署 (第8周)
- [ ] 性能优化
- [ ] Docker容器化
- [ ] 文档完善

## 十一、注意事项

1. **编码问题**: 确保所有字符串处理使用UTF-8编码
2. **内存安全**: 使用智能指针和RAII避免内存泄漏
3. **异常处理**: 使用try-catch处理所有可能的异常
4. **日志记录**: 保持与Python版本相同的日志级别和格式
5. **配置兼容**: 确保配置文件格式与Python版本兼容

## 十二、后续优化方向

1. **GPU加速**: 利用CUDA/OpenCL加速图像处理
2. **模型优化**: 使用TensorRT优化OCR模型推理
3. **分布式处理**: 支持多机分布式处理大批量图片
4. **Web服务化**: 提供RESTful API接口
5. **实时处理**: 支持视频流实时OCR

---

本计划将确保C++重写版本与Python原版在功能上完全等价，同时提供显著的性能提升。整个重写过程预计需要8周时间完成。