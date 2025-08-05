"""
切片处理模块
处理单个切片的OCR识别、头像检测等功能
"""

import cv2
from typing import Dict, List, Optional, Tuple
from .process_avatar import process_avatar_v2


def process_single_slice(slice_info: Dict, engine, text_score_threshold: float, 
                        x_croped: Optional[int], index: int) -> Dict:
    """
    处理单个切片，包括OCR识别和头像检测
    
    Args:
        slice_info: 切片信息字典
        engine: OCR引擎实例
        text_score_threshold: 文本识别置信度阈值
        x_croped: x方向裁剪值
        index: 切片索引（用于保存文件）
        
    Returns:
        处理后的切片结果字典
    """
    slice_img = slice_info['slice']
    slice_index = slice_info['slice_index']
    start_y = slice_info['start_y']
    
    print(f"处理切片 {slice_index}...")
    
    # 进行OCR识别
    slice_img_rgb = cv2.cvtColor(slice_img, cv2.COLOR_BGR2RGB)
    result = engine(slice_img_rgb)
    result.vis(f"output_images/slice_ocr_result_{index}.jpg")
    
    # 过滤低置信度结果
    if result.boxes is not None and result.txts is not None:
        filtered_boxes = []
        filtered_txts = []
        filtered_scores = []
        
        for box, txt, score in zip(result.boxes, result.txts, result.scores):
            if score >= text_score_threshold:
                filtered_boxes.append(box)
                filtered_txts.append(txt)
                filtered_scores.append(score)
        
        print(f"切片 {slice_index} 过滤后结果: {[(txt, score) for txt, score in zip(filtered_txts, filtered_scores)]}")
        
        if not filtered_boxes:
            print(f"切片 {slice_index} 过滤后无有效文本")
            # 即使没有文本，也创建空的切片结果
            return create_empty_slice_result(slice_info)
        
        # 转换坐标到原图坐标系
        adjusted_boxes = []
        for box in filtered_boxes:
            # box 格式: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            adjusted_box = []
            for point in box:
                adjusted_point = [point[0], point[1] + start_y]
                adjusted_box.append(adjusted_point)
            adjusted_boxes.append(adjusted_box)
        
        # 构建该切片的OCR结果
        slice_ocr_result = {
            'boxes': adjusted_boxes,
            'txts': filtered_txts,
            'scores': filtered_scores,
            'image_shape': slice_img.shape
        }
        
        # 对头像裁图进行处理
        print(f"分析切片 {slice_index} 的聊天消息...")
        
        # 如果x_croped为None，使用原图像；否则进行裁剪
        if x_croped is not None:
            slice_img_for_avatar = slice_img[0:slice_img.shape[0], 0:x_croped]
            print(f"切片 {slice_index} 使用x_croped={x_croped}进行裁剪")
        else:
            slice_img_for_avatar = slice_img
            print(f"切片 {slice_index} 未进行x裁剪，使用原始图像")
        
        cv2.imwrite(f"./debug_images/slice_{slice_index:03d}_avatar.jpg", slice_img_for_avatar)
        
        # 输入的是slice_img的左侧头像截图，得到所有头像的外接矩形的坐标
        sliced_merged_result = process_avatar_v2(slice_img_for_avatar)
        
        # 将sliced_merged_result里的坐标还原到原图（y坐标加上start_y）
        if sliced_merged_result:
            restored_sliced_merged_result = []
            for (x, y, w, h) in sliced_merged_result:
                restored_box = (x, y + start_y, w, h)
                restored_sliced_merged_result.append(restored_box)
            sliced_merged_result = restored_sliced_merged_result
        
        # 对结果进行排序
        slice_ocr_result = sort_ocr_results(slice_ocr_result)
        if sliced_merged_result:
            sliced_merged_result = sorted(sliced_merged_result, key=lambda rect: rect[1])
        
        # 创建切片结果
        slice_result = {
            'slice_index': slice_index,
            'start_y': start_y,
            'end_y': slice_info['end_y'],
            'ocr_result': slice_ocr_result,
            'avatar_positions': sliced_merged_result if sliced_merged_result else [],
            'chat_result': None  # 现在基于去重后数据统一处理，这里暂时为None
        }
        
        print(f"切片 {slice_index} 处理完成")
        return slice_result
    else:
        print(f"切片 {slice_index} 未检测到文本")
        return create_empty_slice_result(slice_info)


def sort_ocr_results(ocr_result: Dict) -> Dict:
    """
    将OCR结果按从上到下排序
    
    Args:
        ocr_result: OCR结果字典，包含boxes、txts、scores
        
    Returns:
        排序后的OCR结果
    """
    if ocr_result['boxes']:
        # 获取每个box的最小y坐标（即最上方的点）
        box_with_index = []
        for idx, box in enumerate(ocr_result['boxes']):
            min_y = min(pt[1] for pt in box)
            box_with_index.append((min_y, idx, box))
        
        # 按min_y升序排序
        box_with_index.sort()
        
        # 重新排列boxes, txts, scores
        sorted_boxes = []
        sorted_txts = []
        sorted_scores = []
        for _, idx, box in box_with_index:
            sorted_boxes.append(ocr_result['boxes'][idx])
            sorted_txts.append(ocr_result['txts'][idx])
            sorted_scores.append(ocr_result['scores'][idx])
        
        ocr_result['boxes'] = sorted_boxes
        ocr_result['txts'] = sorted_txts
        ocr_result['scores'] = sorted_scores
    
    return ocr_result


def create_empty_slice_result(slice_info: Dict) -> Dict:
    """
    创建空的切片结果
    
    Args:
        slice_info: 切片信息
        
    Returns:
        空的切片结果字典
    """
    return {
        'slice_index': slice_info['slice_index'],
        'start_y': slice_info['start_y'],
        'end_y': slice_info['end_y'],
        'ocr_result': {
            'boxes': [],
            'txts': [],
            'scores': [],
            'image_shape': slice_info['slice'].shape
        },
        'avatar_positions': [],
        'chat_result': None
    }


def collect_results_to_original_coords(slice_result: Dict, all_ocr_results_original: List,
                                     all_avatar_positions_original: List) -> None:
    """
    将切片结果收集到原图坐标系统
    
    Args:
        slice_result: 切片处理结果
        all_ocr_results_original: 原图坐标系的OCR结果列表
        all_avatar_positions_original: 原图坐标系的头像位置列表
    """
    slice_index = slice_result['slice_index']
    ocr_result = slice_result['ocr_result']
    avatar_positions = slice_result['avatar_positions']
    
    # 收集OCR结果
    for idx, box in enumerate(ocr_result['boxes']):
        ocr_item_original = {
            'slice_index': slice_index,
            'box': box,  # 已经是原图坐标
            'text': ocr_result['txts'][idx],
            'score': ocr_result['scores'][idx]
        }
        all_ocr_results_original.append(ocr_item_original)
    
    # 收集头像位置
    if avatar_positions:
        for avatar_box in avatar_positions:
            x, y, w, h = avatar_box
            avatar_item_original = {
                'slice_index': slice_index,
                'box': (x, y, w, h),  # 已经是原图坐标
                'center_x': x + w/2,
                'center_y': y + h/2
            }
            all_avatar_positions_original.append(avatar_item_original)