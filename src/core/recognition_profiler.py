"""
Recognition性能分析器
用于详细分析OCR Recognition阶段的时间分解
"""

import time
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class RecognitionTiming:
    """Recognition各阶段时间记录"""
    input_boxes_count: int = 0
    
    # 预处理时间
    resize_norm_time: float = 0.0
    batch_prepare_time: float = 0.0
    preprocessing_total: float = 0.0
    
    # Forward推理时间
    inference_time: float = 0.0
    
    # 后处理时间
    argmax_time: float = 0.0
    ctc_decode_time: float = 0.0
    duplicate_removal_time: float = 0.0
    confidence_calc_time: float = 0.0
    text_assembly_time: float = 0.0
    word_segmentation_time: float = 0.0
    postprocessing_total: float = 0.0
    
    # 总时间
    total_recognition_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'input_boxes_count': self.input_boxes_count,
            'preprocessing': {
                'resize_norm_time': round(self.resize_norm_time, 6),
                'batch_prepare_time': round(self.batch_prepare_time, 6),
                'total': round(self.preprocessing_total, 6)
            },
            'forward': {
                'inference_time': round(self.inference_time, 6)
            },
            'postprocessing': {
                'argmax_time': round(self.argmax_time, 6),
                'ctc_decode_time': round(self.ctc_decode_time, 6),
                'duplicate_removal_time': round(self.duplicate_removal_time, 6),
                'confidence_calc_time': round(self.confidence_calc_time, 6),
                'text_assembly_time': round(self.text_assembly_time, 6),
                'word_segmentation_time': round(self.word_segmentation_time, 6),
                'total': round(self.postprocessing_total, 6)
            },
            'total_recognition_time': round(self.total_recognition_time, 6)
        }

class RecognitionProfiler:
    """Recognition性能分析器"""
    
    def __init__(self):
        self.enabled = True
        
    def profile_recognition(self, text_recognizer, rec_input) -> Tuple[Any, RecognitionTiming]:
        """
        分析Recognition性能
        
        Args:
            text_recognizer: TextRecognizer实例
            rec_input: TextRecInput输入
            
        Returns:
            (recognition_result, timing)
        """
        if not self.enabled:
            # 如果不启用性能分析，直接调用原方法
            start = time.time()
            result = text_recognizer(rec_input)
            total_time = time.time() - start
            
            timing = RecognitionTiming()
            timing.total_recognition_time = total_time
            return result, timing
        
        timing = RecognitionTiming()
        
        # 记录输入文本框数量
        img_list = [rec_input.img] if isinstance(rec_input.img, np.ndarray) else rec_input.img
        timing.input_boxes_count = len(img_list)
        
        total_start = time.time()
        
        try:
            # 调用包装的recognition方法
            result = self._profile_recognition_detailed(text_recognizer, rec_input, timing)
            
            timing.total_recognition_time = time.time() - total_start
            
            # 计算总时间
            timing.preprocessing_total = timing.resize_norm_time + timing.batch_prepare_time
            timing.postprocessing_total = (timing.argmax_time + timing.ctc_decode_time + 
                                         timing.duplicate_removal_time + timing.confidence_calc_time + 
                                         timing.text_assembly_time + timing.word_segmentation_time)
            
            logger.debug(f"Recognition profiling completed: {timing.input_boxes_count} boxes, "
                        f"total: {timing.total_recognition_time:.3f}s")
            
            return result, timing
            
        except Exception as e:
            logger.error(f"Recognition profiling failed: {e}")
            # 降级到普通模式
            result = text_recognizer(rec_input)
            timing.total_recognition_time = time.time() - total_start
            return result, timing
    
    def _profile_recognition_detailed(self, text_recognizer, rec_input, timing: RecognitionTiming):
        """详细分析Recognition各个阶段"""
        
        # 从rec_input提取参数
        img_list = [rec_input.img] if isinstance(rec_input.img, np.ndarray) else rec_input.img
        return_word_box = rec_input.return_word_box
        
        # 模拟原始__call__方法的逻辑，但添加时间记录点
        
        # 1. 预处理阶段
        preprocess_start = time.time()
        
        width_list = [img.shape[1] / float(img.shape[0]) for img in img_list]
        indices = np.argsort(np.array(width_list))
        
        img_num = len(img_list)
        rec_res = [("", 0.0)] * img_num
        
        batch_num = text_recognizer.rec_batch_num
        elapse = 0
        
        batch_prepare_total = 0
        resize_norm_total = 0
        inference_total = 0
        postprocess_total = 0
        
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            
            # 批处理准备
            batch_start = time.time()
            
            imgC, imgH, imgW = text_recognizer.rec_image_shape[:3]
            max_wh_ratio = imgW / imgH
            wh_ratio_list = []
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
                wh_ratio_list.append(wh_ratio)
            
            batch_prepare_total += time.time() - batch_start
            
            # 图像归一化
            norm_batch_start = time.time()
            norm_img_batch = []
            for ino in range(beg_img_no, end_img_no):
                norm_img = text_recognizer.resize_norm_img(img_list[indices[ino]], max_wh_ratio)
                norm_img_batch.append(norm_img[np.newaxis, :])
            norm_img_batch = np.concatenate(norm_img_batch).astype(np.float32)
            resize_norm_total += time.time() - norm_batch_start
            
            # 2. Forward推理
            forward_start = time.time()
            preds = text_recognizer.session(norm_img_batch)
            inference_total += time.time() - forward_start
            
            # 3. 后处理
            postprocess_start = time.time()
            line_results, word_results = text_recognizer.postprocess_op(
                preds,
                return_word_box,
                wh_ratio_list=wh_ratio_list,
                max_wh_ratio=max_wh_ratio,
            )
            
            for rno, one_res in enumerate(line_results):
                if return_word_box:
                    rec_res[indices[beg_img_no + rno]] = (one_res, word_results[rno])
                    continue
                rec_res[indices[beg_img_no + rno]] = (one_res, None)
            
            postprocess_total += time.time() - postprocess_start
            elapse += time.time() - forward_start
        
        # 记录时间
        timing.batch_prepare_time = batch_prepare_total
        timing.resize_norm_time = resize_norm_total
        timing.inference_time = inference_total
        
        # 后处理时间暂时合并（详细分解需要修改rapidocr源码）
        timing.ctc_decode_time = postprocess_total
        
        all_line_results, all_word_results = list(zip(*rec_res))
        txts, scores = list(zip(*all_line_results))
        
        # 构造返回结果
        from rapidocr.ch_ppocr_rec.typings import TextRecOutput
        from rapidocr.utils.vis_res import VisRes
        
        return TextRecOutput(
            img_list,
            txts,
            scores,
            all_word_results,
            elapse,
            viser=VisRes(lang_type=text_recognizer.cfg.lang_type, 
                        font_path=text_recognizer.cfg.font_path),
        )