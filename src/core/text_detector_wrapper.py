"""
TextDetector包装器
用于在不修改rapidocr源码的情况下添加详细性能分析
"""

import time
import numpy as np
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class TextDetectorWrapper:
    """TextDetector包装器，添加详细性能分析"""
    
    def __init__(self, original_detector, profiling_enabled: bool = True):
        """
        初始化包装器
        
        Args:
            original_detector: 原始的TextDetector实例
            profiling_enabled: 是否启用性能分析
        """
        self.original_detector = original_detector
        self.profiling_enabled = profiling_enabled
        
        # 转发所有属性到原始detector（先做这个，避免覆盖我们的属性）
        self.__dict__.update(original_detector.__dict__)
        
        # 初始化我们自己的属性（放在update后面，确保不被覆盖）
        self.detailed_timing = {}
    
    def __getattr__(self, name):
        """转发属性访问到原始detector"""
        return getattr(self.original_detector, name)
    
    def __call__(self, img: np.ndarray, return_detailed_timing: bool = False):
        """
        包装的调用方法，添加详细时间记录
        
        Args:
            img: 输入图像
            return_detailed_timing: 是否返回详细时间信息
            
        Returns:
            如果return_detailed_timing=True: (result, detailed_timing)
            否则: result
        """
        if not self.profiling_enabled:
            result = self.original_detector(img)
            if return_detailed_timing:
                return result, {}
            return result
        
        # 初始化详细时间记录
        detailed_timing = {
            'preprocessing_time': 0.0,
            'inference_time': 0.0,
            'postprocessing_time': 0.0,
            'total_time': 0.0,
            'detected_boxes_count': 0
        }
        
        total_start = time.time()
        
        try:
            # 执行带性能分析的detection
            result = self._call_with_profiling(img, detailed_timing)
            
            detailed_timing['total_time'] = time.time() - total_start
            detailed_timing['detected_boxes_count'] = len(result.boxes) if result.boxes is not None else 0
            
            self.detailed_timing = detailed_timing
            
            if return_detailed_timing:
                return result, detailed_timing
            return result
            
        except Exception as e:
            logger.warning(f"Detection profiling failed, falling back to original: {e}")
            result = self.original_detector(img)
            detailed_timing['total_time'] = time.time() - total_start
            detailed_timing['detected_boxes_count'] = len(result.boxes) if result.boxes is not None else 0
            
            if return_detailed_timing:
                return result, detailed_timing
            return result
    
    def _call_with_profiling(self, img, timing_dict):
        """执行带性能分析的detection"""
        
        if img is None:
            raise ValueError("img is None")

        start_time = time.perf_counter()
        ori_img_shape = img.shape[0], img.shape[1]
        
        # 1. 预处理阶段
        preprocessing_start = time.time()
        
        # 获取预处理操作器
        preprocess_op = self.original_detector.get_preprocess(max(img.shape[0], img.shape[1]))
        
        # 执行预处理
        prepro_img = preprocess_op(img)
        if prepro_img is None:
            from rapidocr.ch_ppocr_det import TextDetOutput
            return TextDetOutput()
        
        timing_dict['preprocessing_time'] = time.time() - preprocessing_start
        timing_dict['preprocessed_image_shape'] = prepro_img.shape
        
        # 2. 推理阶段
        inference_start = time.time()
        preds = self.original_detector.session(prepro_img)
        timing_dict['inference_time'] = time.time() - inference_start
        
        # 3. 后处理阶段
        postprocessing_start = time.time()
        boxes, scores = self.original_detector.postprocess_op(preds, ori_img_shape)
        if len(boxes) < 1:
            from rapidocr.ch_ppocr_det import TextDetOutput
            timing_dict['postprocessing_time'] = time.time() - postprocessing_start
            return TextDetOutput()

        boxes = self.original_detector.sorted_boxes(boxes)
        timing_dict['postprocessing_time'] = time.time() - postprocessing_start
        
        # 构造返回结果
        elapse = time.perf_counter() - start_time
        from rapidocr.ch_ppocr_det import TextDetOutput
        return TextDetOutput(img, boxes, scores, elapse=elapse)
    
    def get_last_detailed_timing(self) -> Dict:
        """获取最后一次调用的详细时间信息"""
        return self.detailed_timing.copy() if hasattr(self, 'detailed_timing') else {}