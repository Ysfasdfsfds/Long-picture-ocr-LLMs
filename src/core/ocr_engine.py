"""
OCR引擎封装模块
提供统一的OCR接口，支持分离detection和recognition的时间统计
"""

import cv2
import numpy as np
import time
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from rapidocr import RapidOCR
from rapidocr.main import RapidOCR as RapidOCREngine
from rapidocr.ch_ppocr_det import TextDetector, TextDetOutput
from rapidocr.ch_ppocr_rec import TextRecognizer, TextRecOutput, TextRecInput
from rapidocr.ch_ppocr_cls import TextClassifier, TextClsOutput
from rapidocr.utils.parse_parameters import ParseParams
from rapidocr.utils import LoadImage, resize_image_within_bounds, get_rotate_crop_image
import copy
import logging
from PIL import Image, ImageDraw, ImageFont
import io

from ..models.ocr_result import OCRItem, SliceOCRResult
from ..models.slice_info import SliceInfo
from ..utils.config import Config
from .recognition_profiler import RecognitionProfiler, RecognitionTiming
from .text_recognizer_wrapper import TextRecognizerWrapper
from .text_detector_wrapper import TextDetectorWrapper

logger = logging.getLogger(__name__)


class OCREngine:
    """OCR引擎封装类，支持分离detection和recognition的详细计时"""
    
    def __init__(self, config: Config):
        """
        初始化OCR引擎
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.engine = RapidOCR(config_path=config.ocr.config_path)
        self.text_score_threshold = config.ocr.text_score_threshold
        
        # 初始化详细计时的RapidOCR组件
        self._init_detailed_ocr_components(config.ocr.config_path)
        
        # 存储每个切片的详细计时数据
        self.slice_timing_records = {}
        
        # 确保输出目录存在
        self.timing_output_dir = Path("output_json")
        self.timing_output_dir.mkdir(exist_ok=True)
    
    def _init_detailed_ocr_components(self, config_path: str):
        """
        初始化用于详细计时的OCR组件
        """
        # 加载配置
        root_dir = Path(__file__).resolve().parent / ".." / ".."
        default_cfg_path = Path("/home/kylin/miniconda3/envs/long_ocr_LLm/lib/python3.12/site-packages/rapidocr/config.yaml")
        
        if config_path and Path(config_path).exists():
            cfg = ParseParams.load(config_path)
        else:
            cfg = ParseParams.load(default_cfg_path)
        
        # 初始化各个组件
        cfg.Det.engine_cfg = cfg.EngineConfig[cfg.Det.engine_type.value]
        self.text_det = TextDetector(cfg.Det)
        
        cfg.Cls.engine_cfg = cfg.EngineConfig[cfg.Cls.engine_type.value]
        self.text_cls = TextClassifier(cfg.Cls)
        
        cfg.Rec.engine_cfg = cfg.EngineConfig[cfg.Rec.engine_type.value]
        cfg.Rec.font_path = cfg.Global.font_path
        self.text_rec = TextRecognizer(cfg.Rec)
        
        self.load_img = LoadImage()
        self.max_side_len = cfg.Global.max_side_len
        self.min_side_len = cfg.Global.min_side_len
        self.width_height_ratio = cfg.Global.width_height_ratio
        self.min_height = cfg.Global.min_height
        
        self.cfg = cfg
        
        # 初始化性能分析器
        self.recognition_profiler = RecognitionProfiler()
        
        # 包装TextDetector以支持详细性能分析
        if hasattr(self, 'text_det'):
            self.text_det = TextDetectorWrapper(self.text_det, profiling_enabled=True)
        
        # 包装TextRecognizer以支持详细性能分析
        if hasattr(self, 'text_rec'):
            self.text_rec = TextRecognizerWrapper(self.text_rec, profiling_enabled=True)
        
        logger.info("详细计时OCR组件初始化完成（包含Recognition性能分析）")
    
    def process_slice(self, slice_info: SliceInfo, 
                     save_visualization: bool = True) -> SliceOCRResult:
        """
        处理单个切片的OCR识别（使用原有引擎，保持兼容性）
        
        Args:
            slice_info: 切片信息
            save_visualization: 是否保存可视化结果
            
        Returns:
            切片OCR结果
        """
        logger.info(f"处理切片 {slice_info.slice_index}...")
        
        # 创建结果对象
        result = SliceOCRResult(
            slice_index=slice_info.slice_index,
            start_y=slice_info.start_y,
            end_y=slice_info.end_y
        )
        
        # 进行OCR识别
        slice_img_rgb = cv2.cvtColor(slice_info.image, cv2.COLOR_BGR2RGB)
        ocr_result = self.engine(slice_img_rgb)
        
        # # 保存可视化结果
        # if save_visualization:
        #     vis_path = f"{self.config.output.output_images_dir}/slice_ocr_result_{slice_info.slice_index}.jpg"
        #     ocr_result.vis(vis_path)
        
        # 处理OCR结果
        if ocr_result.boxes is not None and ocr_result.txts is not None:
            self._process_ocr_items(ocr_result, slice_info, result)
        else:
            logger.warning(f"切片 {slice_info.slice_index} 未检测到文本")
        
        # 排序结果
        result.sort_by_y()
        
        # logger.info(f"切片 {slice_info.slice_index} 处理完成，"
        #            f"检测到 {len(result.ocr_items)} 个文本")
        
        return result
    
    def process_slice_with_detailed_timing(self, slice_info: SliceInfo, 
                                         save_visualization: bool = True) -> SliceOCRResult:
        """
        处理单个切片的OCR识别，记录详细的detection和recognition时间
        
        Args:
            slice_info: 切片信息
            save_visualization: 是否保存可视化结果
            
        Returns:
            切片OCR结果，包含详细计时信息
        """
        slice_idx = slice_info.slice_index
        logger.info(f"开始详细计时处理切片 {slice_idx}...")
        
        # 初始化计时记录
        timing_record = {
            "slice_index": slice_idx,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "image_shape": slice_info.image.shape,
            "start_y": slice_info.start_y,
            "end_y": slice_info.end_y,
            # OCR预处理阶段
            "ocr_preprocessing_time": 0.0,
            # Detection阶段细分
            "detection_time": 0.0,
            "detection_detailed": {},
            # Recognition前的裁剪预处理
            "crop_preprocessing_time": 0.0,
            # Recognition阶段细分
            "recognition_time": 0.0,
            "recognition_detailed": {},
            # 最终后处理
            "final_postprocessing_time": 0.0,
            # 可选阶段
            "classification_time": 0.0,
            # 兼容性字段（保持向后兼容）
            "detection_postprocessing_time": 0.0,
            "recognition_preprocessing_time": 0.0,
            "recognition_postprocessing_time": 0.0,
            "preprocessing_time": 0.0,
            "postprocessing_time": 0.0,
            "total_ocr_time": 0.0,
            "detected_boxes_count": 0,
            "recognized_texts_count": 0,
            "final_texts_count": 0
        }
        
        # 记录总体开始时间
        total_start_time = time.time()
        
        # 创建结果对象
        result = SliceOCRResult(
            slice_index=slice_idx,
            start_y=slice_info.start_y,
            end_y=slice_info.end_y
        )
        
        try:
            # 1. OCR图像预处理（Detection和Recognition共用）
            ocr_preprocess_start = time.time()
            slice_img_rgb = cv2.cvtColor(slice_info.image, cv2.COLOR_BGR2RGB)
            
            # 图像尺寸调整（模拟RapidOCR的预处理）
            img, ratio_h, ratio_w = resize_image_within_bounds(
                slice_img_rgb, self.min_side_len, self.max_side_len
            )
            op_record = {"preprocess": {"ratio_h": ratio_h, "ratio_w": ratio_w}}
            
            timing_record["ocr_preprocessing_time"] = time.time() - ocr_preprocess_start
            
            # 2. 文本检测 (Detection) - 使用详细性能分析
            detection_start = time.time()
            if hasattr(self.text_det, 'get_last_detailed_timing'):
                det_res = self.text_det(img)
                detailed_detection_timing = self.text_det.get_last_detailed_timing()
                
                # 记录详细Detection时间
                timing_record["detection_detailed"] = detailed_detection_timing
                timing_record["detection_time"] = detailed_detection_timing.get('total_time', time.time() - detection_start)
            else:
                det_res = self.text_det(img)
                timing_record["detection_time"] = time.time() - detection_start
                timing_record["detection_detailed"] = {}
            timing_record["detected_boxes_count"] = len(det_res.boxes) if det_res.boxes is not None else 0
            
            logger.info(f"切片 {slice_idx} - Detection完成: {timing_record['detection_time']:.3f}s, 检测到 {timing_record['detected_boxes_count']} 个文本框")
            
            # 可视化detection结果
            if det_res.boxes is not None and len(det_res.boxes) > 0:
                self._visualize_detection_results(img, det_res, slice_idx)
            
            if det_res.boxes is None or len(det_res.boxes) == 0:
                logger.warning(f"切片 {slice_idx} 未检测到文本框")
                timing_record["total_ocr_time"] = time.time() - total_start_time
                self.slice_timing_records[slice_idx] = timing_record
                return result
            
            # 3. Recognition前预处理：切割文本区域图像
            crop_preprocess_start = time.time()
            img_crop_list = []
            for box in det_res.boxes:
                tmp_box = copy.deepcopy(box)
                img_crop = get_rotate_crop_image(img, tmp_box)
                img_crop_list.append(img_crop)
            timing_record["crop_preprocessing_time"] = time.time() - crop_preprocess_start
            
            # 4. 文本方向分类 (Classification) - 可选
            if hasattr(self.cfg.Global, 'use_cls') and self.cfg.Global.use_cls:
                cls_start = time.time()
                cls_res = self.text_cls(img_crop_list)
                timing_record["classification_time"] = time.time() - cls_start
                if cls_res.img_list is not None:
                    img_crop_list = cls_res.img_list
                logger.info(f"切片 {slice_idx} - Classification完成: {timing_record['classification_time']:.3f}s")
            
            # 5. 文本识别 (Recognition) - 使用详细性能分析
            recognition_start = time.time()
            rec_input = TextRecInput(img=img_crop_list, return_word_box=False)
            
            # 使用包装器获取详细时间信息
            if hasattr(self.text_rec, 'get_last_detailed_timing'):
                rec_res = self.text_rec(rec_input)
                detailed_recognition_timing = self.text_rec.get_last_detailed_timing()
                
                # 记录详细Recognition时间
                timing_record["recognition_detailed"] = detailed_recognition_timing
                timing_record["recognition_time"] = detailed_recognition_timing.get('total_time', time.time() - recognition_start)
            else:
                rec_res = self.text_rec(rec_input)
                timing_record["recognition_time"] = time.time() - recognition_start
                timing_record["recognition_detailed"] = {}
            
            timing_record["recognized_texts_count"] = len(rec_res.txts) if rec_res.txts is not None else 0
            
            logger.info(f"切片 {slice_idx} - Recognition完成: {timing_record['recognition_time']:.3f}s, 识别到 {timing_record['recognized_texts_count']} 个文本")
            
            # 记录详细性能信息到日志
            if timing_record.get("detection_detailed"):
                det_detailed = timing_record["detection_detailed"]
                logger.debug(f"  Detection详细 - 预处理: {det_detailed.get('preprocessing_time', 0):.3f}s, "
                           f"推理: {det_detailed.get('inference_time', 0):.3f}s, "
                           f"后处理: {det_detailed.get('postprocessing_time', 0):.3f}s")
            
            if timing_record.get("recognition_detailed"):
                rec_detailed = timing_record["recognition_detailed"]
                logger.debug(f"  Recognition详细 - 前处理: {rec_detailed.get('preprocessing', {}).get('total', 0):.3f}s, "
                           f"推理: {rec_detailed.get('forward', {}).get('inference_time', 0):.3f}s, "
                           f"后处理: {rec_detailed.get('postprocessing', {}).get('total', 0):.3f}s")
            
            # 6. 最终后处理（坐标转换、过滤等）
            final_postprocess_start = time.time()
            
            if det_res.boxes is not None and rec_res.txts is not None and rec_res.scores is not None:
                # 调整坐标回原图坐标系
                dt_boxes_array = np.array(det_res.boxes).astype(np.float32)
                
                # 应用预处理的逆变换
                for op in reversed(list(op_record.keys())):
                    v = op_record[op]
                    if "preprocess" in op:
                        ratio_h = v.get("ratio_h", 1.0)
                        ratio_w = v.get("ratio_w", 1.0)
                        dt_boxes_array[:, :, 0] *= ratio_w
                        dt_boxes_array[:, :, 1] *= ratio_h
                
                # 保存用于可视化的切片坐标
                vis_boxes_slice = []
                vis_texts = []
                vis_scores = []
                
                # 过滤和处理OCR结果
                for box, txt, score in zip(dt_boxes_array, rec_res.txts, rec_res.scores):
                    # 过滤低置信度结果
                    if score < self.text_score_threshold:
                        continue
                    
                    # 保存切片坐标用于可视化（转换为整数）
                    vis_boxes_slice.append([[int(x), int(y)] for x, y in box.tolist()])
                    vis_texts.append(txt)
                    vis_scores.append(score)
                    
                    # 转换坐标到原图坐标系
                    adjusted_box = self._adjust_box_to_original(box.tolist(), slice_info.start_x, slice_info.start_y)
                    
                    # 添加OCR项
                    result.add_ocr_item(
                        text=txt,
                        box=adjusted_box,
                        score=score
                    )
            
            # 排序结果
            result.sort_by_y()
            timing_record["final_texts_count"] = len(result.ocr_items)
            timing_record["final_postprocessing_time"] = time.time() - final_postprocess_start
            
            # 可视化OCR结果（如果有结果的话）
            if vis_boxes_slice:
                try:
                    # 使用切片坐标进行可视化
                    self._visualize_ocr_results(img, vis_boxes_slice, vis_texts, vis_scores, slice_idx)
                except Exception as vis_e:
                    logger.warning(f"切片 {slice_idx} OCR可视化失败: {vis_e}")
            
        except Exception as e:
            logger.error(f"切片 {slice_idx} 处理出错: {e}", exc_info=True)
        
        # 计算总时间
        timing_record["total_ocr_time"] = time.time() - total_start_time
        
        # 计算兼容性字段（保持向后兼容）
        timing_record["detection_postprocessing_time"] = timing_record["crop_preprocessing_time"]
        timing_record["recognition_preprocessing_time"] = timing_record["crop_preprocessing_time"]
        timing_record["recognition_postprocessing_time"] = timing_record["final_postprocessing_time"]
        timing_record["preprocessing_time"] = (timing_record["ocr_preprocessing_time"] + 
                                              timing_record["crop_preprocessing_time"])
        timing_record["postprocessing_time"] = timing_record["final_postprocessing_time"]
        
        # 存储计时记录
        self.slice_timing_records[slice_idx] = timing_record
        
        # 打印详细计时信息
        self._print_slice_timing_detail(timing_record)
        
        logger.info(f"切片 {slice_idx} 详细计时处理完成，最终识别到 {timing_record['final_texts_count']} 个文本")
        
        return result
    
    def _print_slice_timing_detail(self, timing_record: Dict):
        """打印单个切片的详细计时信息"""
        idx = timing_record["slice_index"]
        total_time = timing_record['total_ocr_time']
        
        print(f"\n🔍 切片 {idx} 详细计时分析:")
        print(f"  📊 总耗时: {total_time:.3f}秒")
        
        # OCR预处理阶段
        ocr_pre = timing_record['ocr_preprocessing_time']
        print(f"  🔧 OCR预处理: {ocr_pre:.3f}秒 ({ocr_pre/total_time*100:.1f}%)")
        
        # Detection阶段
        det_time = timing_record['detection_time']
        print(f"  🔍 Detection阶段: {det_time:.3f}秒 ({det_time/total_time*100:.1f}%)")
        if timing_record.get('detection_detailed'):
            det_detailed = timing_record['detection_detailed']
            det_prep = det_detailed.get('preprocessing_time', 0)
            det_inf = det_detailed.get('inference_time', 0)
            det_post = det_detailed.get('postprocessing_time', 0)
            print(f"    ├─ 预处理: {det_prep:.3f}秒 ({det_prep/total_time*100:.1f}%)")
            print(f"    ├─ 推理: {det_inf:.3f}秒 ({det_inf/total_time*100:.1f}%)")
            print(f"    └─ 后处理: {det_post:.3f}秒 ({det_post/total_time*100:.1f}%)")
        
        # 裁剪预处理
        crop_pre = timing_record['crop_preprocessing_time']
        print(f"  ✂️  裁剪预处理: {crop_pre:.3f}秒 ({crop_pre/total_time*100:.1f}%)")
        
        # 可选的分类阶段
        if timing_record['classification_time'] > 0:
            cls_time = timing_record['classification_time']
            print(f"  🔄 方向分类: {cls_time:.3f}秒 ({cls_time/total_time*100:.1f}%)")
        
        # Recognition阶段
        rec_time = timing_record['recognition_time']
        print(f"  📝 Recognition阶段: {rec_time:.3f}秒 ({rec_time/total_time*100:.1f}%)")
        if timing_record.get('recognition_detailed'):
            rec_detailed = timing_record['recognition_detailed']
            rec_prep_total = rec_detailed.get('preprocessing', {}).get('total', 0)
            rec_inf = rec_detailed.get('forward', {}).get('inference_time', 0)
            rec_post_total = rec_detailed.get('postprocessing', {}).get('total', 0)
            print(f"    ├─ 预处理: {rec_prep_total:.3f}秒 ({rec_prep_total/total_time*100:.1f}%)")
            print(f"    ├─ 推理: {rec_inf:.3f}秒 ({rec_inf/total_time*100:.1f}%)")
            print(f"    └─ 后处理: {rec_post_total:.3f}秒 ({rec_post_total/total_time*100:.1f}%)")
        
        # 最终后处理
        final_post = timing_record['final_postprocessing_time']
        print(f"  🔄 最终后处理: {final_post:.3f}秒 ({final_post/total_time*100:.1f}%)")
        
        print(f"  📈 结果统计: 检测框{timing_record['detected_boxes_count']} → 识别文本{timing_record['recognized_texts_count']} → 最终文本{timing_record['final_texts_count']}")
    
    def export_slice_timing_records(self):
        """导出所有切片的详细计时记录到JSON文件"""
        if not self.slice_timing_records:
            logger.warning("没有切片计时记录可导出")
            return
        
        # 计算汇总统计
        summary_stats = self._calculate_timing_summary()
        
        # 准备导出数据
        export_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": summary_stats,
            "slice_details": self.slice_timing_records
        }
        
        # 导出到文件
        output_file = self.timing_output_dir / "slice_ocr_detailed_timing.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"切片详细计时记录已导出到: {output_file}")
        print(f"📁 切片详细计时记录已保存到: {output_file}")
        
        return output_file
    
    def _calculate_timing_summary(self) -> Dict:
        """计算所有切片的计时汇总统计"""
        if not self.slice_timing_records:
            return {}
        
        records = list(self.slice_timing_records.values())
        total_slices = len(records)
        
        # 汇总时间 - 细分统计
        total_ocr_preprocessing = sum(r["ocr_preprocessing_time"] for r in records)
        total_detection = sum(r["detection_time"] for r in records)
        total_crop_preprocessing = sum(r["crop_preprocessing_time"] for r in records)
        total_recognition = sum(r["recognition_time"] for r in records)
        total_final_postprocessing = sum(r["final_postprocessing_time"] for r in records)
        total_classification = sum(r["classification_time"] for r in records)
        total_ocr = sum(r["total_ocr_time"] for r in records)
        
        # Detection细分统计
        total_det_preprocessing = sum(r.get("detection_detailed", {}).get("preprocessing_time", 0) for r in records)
        total_det_inference = sum(r.get("detection_detailed", {}).get("inference_time", 0) for r in records)
        total_det_postprocessing = sum(r.get("detection_detailed", {}).get("postprocessing_time", 0) for r in records)
        
        # Recognition细分统计
        total_rec_preprocessing = sum(r.get("recognition_detailed", {}).get("preprocessing", {}).get("total", 0) for r in records)
        total_rec_inference = sum(r.get("recognition_detailed", {}).get("forward", {}).get("inference_time", 0) for r in records)
        total_rec_postprocessing = sum(r.get("recognition_detailed", {}).get("postprocessing", {}).get("total", 0) for r in records)
        
        # 兼容性字段
        total_preprocessing = sum(r["preprocessing_time"] for r in records)
        total_postprocessing = sum(r["postprocessing_time"] for r in records)
        
        # 汇总数量
        total_detected_boxes = sum(r["detected_boxes_count"] for r in records)
        total_recognized_texts = sum(r["recognized_texts_count"] for r in records)
        total_final_texts = sum(r["final_texts_count"] for r in records)
        
        # 找出最耗时和最快的切片
        slowest_slice = max(records, key=lambda x: x["total_ocr_time"])
        fastest_slice = min(records, key=lambda x: x["total_ocr_time"])
        
        summary = {
            "total_slices": total_slices,
            "detailed_timing_summary": {
                # OCR预处理阶段
                "total_ocr_preprocessing_time": round(total_ocr_preprocessing, 3),
                
                # Detection阶段细分
                "total_detection_time": round(total_detection, 3),
                "total_detection_preprocessing_time": round(total_det_preprocessing, 3),
                "total_detection_inference_time": round(total_det_inference, 3),
                "total_detection_postprocessing_time": round(total_det_postprocessing, 3),
                
                # 裁剪预处理
                "total_crop_preprocessing_time": round(total_crop_preprocessing, 3),
                
                # Recognition阶段细分
                "total_recognition_time": round(total_recognition, 3),
                "total_recognition_preprocessing_time": round(total_rec_preprocessing, 3),
                "total_recognition_inference_time": round(total_rec_inference, 3),
                "total_recognition_postprocessing_time": round(total_rec_postprocessing, 3),
                
                # 最终后处理
                "total_final_postprocessing_time": round(total_final_postprocessing, 3),
                
                # 可选阶段
                "total_classification_time": round(total_classification, 3),
                
                # 总计
                "total_ocr_time": round(total_ocr, 3),
                
                # 平均时间
                "average_ocr_preprocessing_time": round(total_ocr_preprocessing / total_slices, 3),
                "average_detection_time": round(total_detection / total_slices, 3),
                "average_detection_preprocessing_time": round(total_det_preprocessing / total_slices, 3),
                "average_detection_inference_time": round(total_det_inference / total_slices, 3),
                "average_detection_postprocessing_time": round(total_det_postprocessing / total_slices, 3),
                "average_crop_preprocessing_time": round(total_crop_preprocessing / total_slices, 3),
                "average_recognition_time": round(total_recognition / total_slices, 3),
                "average_recognition_preprocessing_time": round(total_rec_preprocessing / total_slices, 3),
                "average_recognition_inference_time": round(total_rec_inference / total_slices, 3),
                "average_recognition_postprocessing_time": round(total_rec_postprocessing / total_slices, 3),
                "average_final_postprocessing_time": round(total_final_postprocessing / total_slices, 3),
                "average_classification_time": round(total_classification / total_slices, 3),
                "average_total_time": round(total_ocr / total_slices, 3)
            },
            "legacy_timing_summary": {
                # 兼容性字段
                "total_preprocessing_time": round(total_preprocessing, 3),
                "total_detection_time": round(total_detection, 3),
                "total_classification_time": round(total_classification, 3),
                "total_recognition_time": round(total_recognition, 3),
                "total_postprocessing_time": round(total_postprocessing, 3),
                "total_ocr_time": round(total_ocr, 3),
                
                "average_preprocessing_time": round(total_preprocessing / total_slices, 3),
                "average_detection_time": round(total_detection / total_slices, 3),
                "average_classification_time": round(total_classification / total_slices, 3),
                "average_recognition_time": round(total_recognition / total_slices, 3),
                "average_postprocessing_time": round(total_postprocessing / total_slices, 3),
                "average_total_time": round(total_ocr / total_slices, 3)
            },
            "processing_summary": {
                "total_detected_boxes": total_detected_boxes,
                "total_recognized_texts": total_recognized_texts,
                "total_final_texts": total_final_texts,
                "average_boxes_per_slice": round(total_detected_boxes / total_slices, 2),
                "average_texts_per_slice": round(total_final_texts / total_slices, 2),
                "detection_efficiency": round(total_detected_boxes / total_detection if total_detection > 0 else 0, 2),
                "recognition_efficiency": round(total_final_texts / total_recognition if total_recognition > 0 else 0, 2)
            },
            "performance_analysis": {
                "slowest_slice": {
                    "slice_index": slowest_slice["slice_index"],
                    "total_time": round(slowest_slice["total_ocr_time"], 3),
                    "detection_time": round(slowest_slice["detection_time"], 3),
                    "recognition_time": round(slowest_slice["recognition_time"], 3)
                },
                "fastest_slice": {
                    "slice_index": fastest_slice["slice_index"],
                    "total_time": round(fastest_slice["total_ocr_time"], 3),
                    "detection_time": round(fastest_slice["detection_time"], 3),
                    "recognition_time": round(fastest_slice["recognition_time"], 3)
                },
                "detailed_time_distribution": {
                    "ocr_preprocessing_percentage": round((total_ocr_preprocessing / total_ocr) * 100, 1) if total_ocr > 0 else 0,
                    "detection_percentage": round((total_detection / total_ocr) * 100, 1) if total_ocr > 0 else 0,
                    "detection_preprocessing_percentage": round((total_det_preprocessing / total_ocr) * 100, 1) if total_ocr > 0 else 0,
                    "detection_inference_percentage": round((total_det_inference / total_ocr) * 100, 1) if total_ocr > 0 else 0,
                    "detection_postprocessing_percentage": round((total_det_postprocessing / total_ocr) * 100, 1) if total_ocr > 0 else 0,
                    "crop_preprocessing_percentage": round((total_crop_preprocessing / total_ocr) * 100, 1) if total_ocr > 0 else 0,
                    "recognition_percentage": round((total_recognition / total_ocr) * 100, 1) if total_ocr > 0 else 0,
                    "recognition_preprocessing_percentage": round((total_rec_preprocessing / total_ocr) * 100, 1) if total_ocr > 0 else 0,
                    "recognition_inference_percentage": round((total_rec_inference / total_ocr) * 100, 1) if total_ocr > 0 else 0,
                    "recognition_postprocessing_percentage": round((total_rec_postprocessing / total_ocr) * 100, 1) if total_ocr > 0 else 0,
                    "final_postprocessing_percentage": round((total_final_postprocessing / total_ocr) * 100, 1) if total_ocr > 0 else 0,
                    "classification_percentage": round((total_classification / total_ocr) * 100, 1) if total_ocr > 0 else 0
                },
                "legacy_time_distribution": {
                    "detection_percentage": round((total_detection / total_ocr) * 100, 1) if total_ocr > 0 else 0,
                    "recognition_percentage": round((total_recognition / total_ocr) * 100, 1) if total_ocr > 0 else 0,
                    "preprocessing_percentage": round((total_preprocessing / total_ocr) * 100, 1) if total_ocr > 0 else 0,
                    "postprocessing_percentage": round((total_postprocessing / total_ocr) * 100, 1) if total_ocr > 0 else 0
                }
            }
        }
        
        return summary
    
    def _visualize_ocr_results(self, img: np.ndarray, boxes: List, texts: List, 
                              scores: List, slice_idx: int, 
                              save_path: Optional[str] = None) -> np.ndarray:
        """
        可视化OCR结果，绘制检测框（绿色）和识别文字（黑色，在框上方）
        
        Args:
            img: 输入图像 (RGB格式)
            boxes: 检测框坐标列表
            texts: 识别文字列表  
            scores: 置信度分数列表
            slice_idx: 切片索引
            save_path: 保存路径，如果为None则自动生成
            
        Returns:
            绘制了OCR结果的图像
        """
        if not boxes or not texts:
            logger.warning(f"切片 {slice_idx} 没有OCR结果可可视化")
            return img
        
        # 复制图像用于绘制
        vis_img = img.copy()
        
        # 转换为PIL Image以便更好地处理中文字体
        pil_img = Image.fromarray(vis_img)
        draw = ImageDraw.Draw(pil_img)
        
        # 尝试加载中文字体
        try:
            # 尝试几个常见的中文字体路径
            font_paths = [
                "/home/kylin/桌面/Long-picture-ocr-LLMs-main_a/ShanHaiJiGuSongKe-JianFan-2.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 
                "/System/Library/Fonts/PingFang.ttc",  # macOS
                "C:/Windows/Fonts/simhei.ttf",  # Windows
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"  # Ubuntu
            ]
            font = None
            for font_path in font_paths:
                try:
                    if Path(font_path).exists():
                        font = ImageFont.truetype(font_path, size=20)
                        break
                except:
                    continue
            
            if font is None:
                font = ImageFont.load_default()
                logger.warning("使用默认字体，可能无法正确显示中文")
        except Exception as e:
            font = ImageFont.load_default()
            logger.warning(f"字体加载失败: {e}，使用默认字体")
        
        # 绘制每个检测框和对应的识别文字
        for i, (box, text, score) in enumerate(zip(boxes, texts, scores)):
            # 将box转换为整数坐标
            if isinstance(box, np.ndarray):
                box = box.tolist()
            box = [[int(x), int(y)] for x, y in box]
            
            # 绘制绿色检测框 - 使用polygon绘制封闭的四边形
            box_coords = [(x, y) for x, y in box]
            draw.polygon(box_coords, outline=(0, 255, 0), width=2)
            
            # 计算文字位置（框的上方）
            box_top_y = min([y for x, y in box])
            box_left_x = min([x for x, y in box]) 
            box_right_x = max([x for x, y in box])
            box_center_x = (box_left_x + box_right_x) // 2
            
            # 获取文字尺寸
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # 文字位置：在框上方，居中对齐，留有边距
            text_x = max(0, box_center_x - text_width // 2)
            text_y = max(0, box_top_y - text_height - 5)
            
            # 确保文字不会超出图像边界
            text_x = min(text_x, pil_img.width - text_width)
            
            # 绘制文字背景（白色半透明）
            bg_margin = 2
            bg_coords = [
                text_x - bg_margin, text_y - bg_margin,
                text_x + text_width + bg_margin, text_y + text_height + bg_margin
            ]
            draw.rectangle(bg_coords, fill=(255, 255, 255, 200))
            
            # 绘制黑色文字
            draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)
            
            # 在框的左下角绘制序号和置信度
            info_text = f"{i}:{score:.3f}"
            info_bbox = draw.textbbox((0, 0), info_text, font=font)
            info_width = info_bbox[2] - info_bbox[0]
            info_height = info_bbox[3] - info_bbox[1]
            
            box_bottom_y = max([y for x, y in box])
            info_x = box_left_x
            info_y = min(pil_img.height - info_height, box_bottom_y + 2)
            
            # 绘制信息背景
            info_bg_coords = [
                info_x - 2, info_y - 2,
                info_x + info_width + 2, info_y + info_height + 2
            ]
            draw.rectangle(info_bg_coords, fill=(0, 255, 0, 200))
            
            # 绘制白色信息文字
            draw.text((info_x, info_y), info_text, fill=(255, 255, 255), font=font)
        
        # 转换回numpy数组
        vis_img = np.array(pil_img)
        
        # 保存图像
        if save_path is None:
            # 确保输出目录存在
            debug_dir = Path("output_images/debug")
            debug_dir.mkdir(parents=True, exist_ok=True)
            save_path = debug_dir / f"slice_{slice_idx}_ocr_results.jpg"
        
        # 转换回BGR格式用于保存
        vis_img_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(save_path), vis_img_bgr)
        
        logger.info(f"切片 {slice_idx} 的OCR可视化已保存到: {save_path}")
        print(f"🎨 切片 {slice_idx} OCR可视化图已保存: {save_path}")
        print(f"   📊 检测框数量: {len(boxes)}")
        print(f"   📝 识别文字数量: {len(texts)}")
        if scores:
            print(f"   📈 平均置信度: {np.mean(scores):.3f}")
            print(f"   🏆 最高置信度: {np.max(scores):.3f}")
            print(f"   📉 最低置信度: {np.min(scores):.3f}")
        
        return vis_img
    
    def visualize_full_image_ocr_results(self, original_image: np.ndarray, 
                                       all_ocr_items: List, save_path: Optional[str] = None) -> np.ndarray:
        """
        在原图上可视化所有OCR结果
        
        Args:
            original_image: 原始图像 (BGR格式)
            all_ocr_items: 所有OCR项的列表
            save_path: 保存路径，如果为None则自动生成
            
        Returns:
            绘制了所有OCR结果的原图
        """
        if not all_ocr_items:
            logger.warning("没有OCR结果可在原图上可视化")
            return original_image
        
        # 转换为RGB格式
        img_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # 准备可视化数据
        boxes = []
        texts = []
        scores = []
        
        for item in all_ocr_items:
            boxes.append(item.box)
            texts.append(item.text)
            scores.append(item.score)
        
        # 转换为PIL Image以便更好地处理中文字体
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        
        # 尝试加载中文字体
        try:
            font_paths = [
                "/home/kylin/桌面/Long-picture-ocr-LLMs-main_a/ShanHaiJiGuSongKe-JianFan-2.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 
                "/System/Library/Fonts/PingFang.ttc",  # macOS
                "C:/Windows/Fonts/simhei.ttf",  # Windows
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"  # Ubuntu
            ]
            font = None
            for font_path in font_paths:
                try:
                    if Path(font_path).exists():
                        font = ImageFont.truetype(font_path, size=16)  # 原图用较小字体
                        break
                except:
                    continue
            
            if font is None:
                font = ImageFont.load_default()
                logger.warning("使用默认字体，可能无法正确显示中文")
        except Exception as e:
            font = ImageFont.load_default()
            logger.warning(f"字体加载失败: {e}，使用默认字体")
        
        logger.info(f"开始在原图上绘制 {len(boxes)} 个OCR结果...")
        
        # 绘制每个OCR结果
        for i, (box, text, score) in enumerate(zip(boxes, texts, scores)):
            try:
                # 将box转换为整数坐标
                if isinstance(box, np.ndarray):
                    box = box.tolist()
                box = [[int(x), int(y)] for x, y in box]
                
                # 绘制绿色检测框 - 使用polygon绘制封闭的四边形
                box_coords = [(x, y) for x, y in box]
                draw.polygon(box_coords, outline=(0, 255, 0), width=2)
                
                # 计算文字位置（框的上方）
                box_top_y = min([y for x, y in box])
                box_left_x = min([x for x, y in box]) 
                box_right_x = max([x for x, y in box])
                box_center_x = (box_left_x + box_right_x) // 2
                
                # 获取文字尺寸
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # 文字位置：在框上方，居中对齐，留有边距
                text_x = max(0, box_center_x - text_width // 2)
                text_y = max(0, box_top_y - text_height - 3)
                
                # 确保文字不会超出图像边界
                text_x = min(text_x, pil_img.width - text_width)
                
                # 绘制文字背景（白色半透明）
                bg_margin = 1
                bg_coords = [
                    text_x - bg_margin, text_y - bg_margin,
                    text_x + text_width + bg_margin, text_y + text_height + bg_margin
                ]
                draw.rectangle(bg_coords, fill=(255, 255, 255, 180))
                
                # 绘制黑色文字
                draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)
                
            except Exception as e:
                logger.warning(f"绘制第 {i} 个OCR结果时出错: {e}")
                continue
        
        # 转换回numpy数组
        vis_img = np.array(pil_img)
        
        # 保存图像
        if save_path is None:
            save_path = Path("output_images") / "full_image_ocr_results.jpg"
            save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换回BGR格式用于保存
        vis_img_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(save_path), vis_img_bgr)
        
        logger.info(f"原图OCR可视化已保存到: {save_path}")
        print(f"🎨 原图OCR可视化已保存: {save_path}")
        print(f"   📊 总OCR结果数量: {len(all_ocr_items)}")
        if scores:
            print(f"   📈 平均置信度: {np.mean(scores):.3f}")
            print(f"   🏆 最高置信度: {np.max(scores):.3f}")
            print(f"   📉 最低置信度: {np.min(scores):.3f}")
        
        return vis_img_bgr
    
    def _visualize_detection_results(self, img: np.ndarray, det_res: TextDetOutput, 
                                   slice_idx: int, save_path: Optional[str] = None) -> np.ndarray:
        """
        可视化detection结果，绘制检测框和置信度分数
        
        Args:
            img: 输入图像 (RGB格式)
            det_res: detection结果
            slice_idx: 切片索引
            save_path: 保存路径，如果为None则自动生成
            
        Returns:
            绘制了检测框的图像
        """
        if det_res.boxes is None or det_res.scores is None:
            logger.warning(f"切片 {slice_idx} 没有检测结果可可视化")
            return img
        
        # 复制图像用于绘制
        vis_img = img.copy()
        
        # 绘制每个检测框和分数
        for i, (box, score) in enumerate(zip(det_res.boxes, det_res.scores)):
            # 将box转换为整数坐标
            box = np.array(box, dtype=np.int32)
            
            # 选择颜色 (根据置信度分数)
            if score >= 0.9:
                color = (0, 255, 0)  # 高置信度 - 绿色
            elif score >= 0.7:
                color = (0, 255, 255)  # 中等置信度 - 黄色
            else:
                color = (0, 0, 255)  # 低置信度 - 红色
            
            # 绘制检测框
            cv2.polylines(vis_img, [box], True, color, 2)
            
            # 在框的左上角附近绘制序号和置信度
            text_pos = (int(box[0][0]), int(box[0][1]) - 5)
            text = f"{i}:{score:.3f}"
            
            # 绘制文本背景
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(vis_img, 
                         (text_pos[0], text_pos[1] - text_size[1] - 2),
                         (text_pos[0] + text_size[0], text_pos[1] + 2),
                         color, -1)
            
            # 绘制白色文字
            cv2.putText(vis_img, text, text_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 保存图像
        if save_path is None:
            # 确保输出目录存在
            debug_dir = Path("output_images/debug")
            debug_dir.mkdir(parents=True, exist_ok=True)
            save_path = debug_dir / f"slice_{slice_idx}_detection_boxes.jpg"
        
        # 转换回BGR格式用于保存
        vis_img_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(save_path), vis_img_bgr)
        
        logger.info(f"切片 {slice_idx} 的detection可视化已保存到: {save_path}")
        print(f"🎨 Detection可视化图已保存: {save_path}")
        print(f"   📊 检测框数量: {len(det_res.boxes)}")
        print(f"   📈 平均置信度: {np.mean(det_res.scores):.3f}")
        print(f"   🏆 最高置信度: {np.max(det_res.scores):.3f}")
        print(f"   📉 最低置信度: {np.min(det_res.scores):.3f}")
        
        return vis_img
    
    def _process_ocr_items(self, ocr_result: Any, slice_info: SliceInfo, 
                          result: SliceOCRResult):
        """
        处理OCR识别项
        
        Args:
            ocr_result: RapidOCR的识别结果
            slice_info: 切片信息
            result: 切片OCR结果对象
        """
        for box, txt, score in zip(ocr_result.boxes, ocr_result.txts, ocr_result.scores):
            # 过滤低置信度结果
            if score < self.text_score_threshold:
                continue
            
            # 转换坐标到原图坐标系
            adjusted_box = self._adjust_box_to_original(box, slice_info.start_x, slice_info.start_y)
            
            # 添加OCR项
            result.add_ocr_item(
                text=txt,
                box=adjusted_box,
                score=score
            )
        
        logger.debug(f"切片 {slice_info.slice_index} 过滤后保留 "
                    f"{len(result.ocr_items)} 个文本")
    
    def _adjust_box_to_original(self, box: List[List[float]], 
                               start_x: int = 0, start_y: int = 0) -> List[List[float]]:
        """
        将切片坐标转换为原图坐标
        
        Args:
            box: 切片中的边界框坐标
            start_x: 切片在原图中的起始X坐标
            start_y: 切片在原图中的起始Y坐标
            
        Returns:
            原图坐标系中的边界框
        """
        adjusted_box = []
        for point in box:
            adjusted_point = [point[0] + start_x, point[1] + start_y]
            adjusted_box.append(adjusted_point)
        return adjusted_box
    
    def batch_process_slices(self, slice_infos: List[SliceInfo]) -> List[SliceOCRResult]:
        """
        批量处理多个切片
        
        Args:
            slice_infos: 切片信息列表
            
        Returns:
            切片OCR结果列表
        """
        results = []
        for slice_info in slice_infos:
            result = self.process_slice(slice_info)
            results.append(result)
        return results