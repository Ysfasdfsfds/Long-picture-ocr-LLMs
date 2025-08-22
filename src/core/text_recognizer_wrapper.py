"""
TextRecognizer包装器
用于在不修改rapidocr源码的情况下添加详细性能分析
"""

import time
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

class TextRecognizerWrapper:
    """TextRecognizer包装器，添加详细性能分析"""
    
    def __init__(self, original_recognizer, profiling_enabled: bool = True):
        """
        初始化包装器
        
        Args:
            original_recognizer: 原始的TextRecognizer实例
            profiling_enabled: 是否启用性能分析
        """
        self.original_recognizer = original_recognizer
        self.profiling_enabled = profiling_enabled
        self.detailed_timing = {}
        
        # 转发所有属性到原始recognizer
        self.__dict__.update(original_recognizer.__dict__)
    
    def __getattr__(self, name):
        """转发属性访问到原始recognizer"""
        return getattr(self.original_recognizer, name)
    
    def __call__(self, rec_input, return_detailed_timing: bool = False):
        """
        包装的调用方法，添加详细时间记录
        
        Args:
            rec_input: TextRecInput输入
            return_detailed_timing: 是否返回详细时间信息
            
        Returns:
            如果return_detailed_timing=True: (result, detailed_timing)
            否则: result
        """
        if not self.profiling_enabled:
            result = self.original_recognizer(rec_input)
            if return_detailed_timing:
                return result, {}
            return result
        
        # 初始化详细时间记录
        detailed_timing = {
            'preprocessing': {
                'resize_norm_time': 0.0,
                'batch_prepare_time': 0.0,
                'total': 0.0
            },
            'forward': {
                'inference_time': 0.0
            },
            'postprocessing': {
                'argmax_time': 0.0,
                'ctc_decode_time': 0.0,
                'duplicate_removal_time': 0.0,
                'confidence_calc_time': 0.0,
                'text_assembly_time': 0.0,
                'word_segmentation_time': 0.0,
                'total': 0.0
            },
            'total_time': 0.0,
            'input_boxes_count': 0
        }
        
        total_start = time.time()
        
        try:
            # 执行带性能分析的recognition
            result = self._call_with_profiling(rec_input, detailed_timing)
            
            detailed_timing['total_time'] = time.time() - total_start
            detailed_timing['preprocessing']['total'] = (
                detailed_timing['preprocessing']['resize_norm_time'] + 
                detailed_timing['preprocessing']['batch_prepare_time']
            )
            detailed_timing['postprocessing']['total'] = (
                detailed_timing['postprocessing']['argmax_time'] +
                detailed_timing['postprocessing']['ctc_decode_time'] +
                detailed_timing['postprocessing']['duplicate_removal_time'] +
                detailed_timing['postprocessing']['confidence_calc_time'] +
                detailed_timing['postprocessing']['text_assembly_time'] +
                detailed_timing['postprocessing']['word_segmentation_time']
            )
            
            self.detailed_timing = detailed_timing
            
            if return_detailed_timing:
                return result, detailed_timing
            return result
            
        except Exception as e:
            logger.warning(f"Profiling failed, falling back to original: {e}")
            result = self.original_recognizer(rec_input)
            detailed_timing['total_time'] = time.time() - total_start
            
            if return_detailed_timing:
                return result, detailed_timing
            return result
    
    def _call_with_profiling(self, rec_input, timing_dict):
        """执行带性能分析的recognition"""
        
        # 提取输入参数
        img_list = [rec_input.img] if isinstance(rec_input.img, np.ndarray) else rec_input.img
        return_word_box = rec_input.return_word_box
        
        timing_dict['input_boxes_count'] = len(img_list)
        
        # 1. 预处理阶段
        preprocess_start = time.time()
        
        width_list = [img.shape[1] / float(img.shape[0]) for img in img_list]
        
        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))
        
        img_num = len(img_list)
        rec_res = [("", 0.0)] * img_num
        
        batch_num = self.original_recognizer.rec_batch_num
        elapse = 0
        
        # 分阶段时间累计
        resize_norm_total = 0
        batch_prepare_total = 0
        inference_total = 0
        postprocess_total = 0
        
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            
            # 批处理参数准备
            batch_prep_start = time.time()
            
            # Parameter Alignment for PaddleOCR
            imgC, imgH, imgW = self.original_recognizer.rec_image_shape[:3]
            max_wh_ratio = imgW / imgH
            wh_ratio_list = []
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
                wh_ratio_list.append(wh_ratio)
            
            batch_prepare_total += time.time() - batch_prep_start
            
            # 图像归一化和批处理准备
            norm_start = time.time()
            norm_img_batch = []
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.original_recognizer.resize_norm_img(img_list[indices[ino]], max_wh_ratio)
                norm_img_batch.append(norm_img[np.newaxis, :])
            norm_img_batch = np.concatenate(norm_img_batch).astype(np.float32)
            resize_norm_total += time.time() - norm_start
            
            # 2. Forward推理阶段
            inference_start = time.time()
            preds = self.original_recognizer.session(norm_img_batch)
            inference_total += time.time() - inference_start
            
            # 3. 后处理阶段
            postprocess_start = time.time()
            line_results, word_results = self.original_recognizer.postprocess_op(
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
            elapse += time.time() - inference_start
        
        # 记录各阶段时间
        timing_dict['preprocessing']['batch_prepare_time'] = batch_prepare_total
        timing_dict['preprocessing']['resize_norm_time'] = resize_norm_total
        timing_dict['forward']['inference_time'] = inference_total
        
        # 后处理时间（暂时无法细分，需要修改rapidocr源码）
        timing_dict['postprocessing']['ctc_decode_time'] = postprocess_total
        
        # 构造返回结果
        all_line_results, all_word_results = list(zip(*rec_res))
        txts, scores = list(zip(*all_line_results))
        
        from rapidocr.ch_ppocr_rec.typings import TextRecOutput
        from rapidocr.utils.vis_res import VisRes
        
        return TextRecOutput(
            img_list,
            txts,
            scores,
            all_word_results,
            elapse,
            viser=VisRes(lang_type=self.original_recognizer.cfg.lang_type, 
                        font_path=self.original_recognizer.cfg.font_path),
        )
    
    def get_last_detailed_timing(self) -> Dict:
        """获取最后一次调用的详细时间信息"""
        return self.detailed_timing.copy() if hasattr(self, 'detailed_timing') else {}