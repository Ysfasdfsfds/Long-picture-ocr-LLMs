"""
x_croped 值计算模块
用于从切片中计算合适的 x_croped 裁剪值
"""

from typing import List, Dict, Tuple, Optional
from .process_avatar import preprocess_and_crop_image, slice_x_croped_values


def determine_slices_to_process(slices_info: List[Dict]) -> List[Dict]:
    """
    根据切片数量决定处理策略
    
    Args:
        slices_info: 所有切片信息列表
        
    Returns:
        需要处理的切片列表
    """
    total_slices = len(slices_info)
    if total_slices == 1:
        # 只有一个切片，处理所有切片
        slices_to_process = slices_info
        print("只有一个切片，将处理所有切片")
    elif total_slices == 2:
        # 两个切片，选择第一个切片
        slices_to_process = slices_info[:1]
        print("有2个切片，将只处理第一个切片")
    else:
        # 大于等于3个切片，排除开始和结束的切片，只处理中间切片
        slices_to_process = slices_info[1:-1]
        print(f"共有{total_slices}个切片，将处理中间{len(slices_to_process)}个切片（排除第一个和最后一个）")
    
    return slices_to_process


def find_best_target_box(all_boxes: List[Tuple]) -> Optional[Tuple]:
    """
    从所有框中找到最合适的目标框
    策略：选择最左侧前20%中趋近正方形的框
    
    Args:
        all_boxes: 所有框的列表，格式为 [(x, y, w, h, slice_idx), ...]
        
    Returns:
        选中的框 (x, y, w, h, slice_idx) 或 None
    """
    if not all_boxes:
        print("未找到任何target_box")
        return None
    
    # 2. 对最左侧前20%的box进行操作
    left_20_percent_count = max(1, int(len(all_boxes) * 0.2))
    left_boxes = all_boxes[:left_20_percent_count]
    print(f"最左侧前20%的box数量: {left_20_percent_count}")
    
    # 找到符合要求的框
    selected_box = None
    for i, (x, y, w, h, slice_idx) in enumerate(left_boxes):
        # 判断是否严格趋近于正方形（宽高比在0.8-1.2之间）
        aspect_ratio = w / h if h > 0 else 0
        is_square_like = 0.8 <= aspect_ratio <= 1.2
        
        print(f"第{i+1}个左侧框: x={x}, y={y}, w={w}, h={h}, 宽高比={aspect_ratio:.2f}, 是否趋近正方形={is_square_like}")
        
        if is_square_like:
            selected_box = (x, y, w, h, slice_idx)
            print(f"找到符合要求的框: 第{i+1}个左侧框，位于slice {slice_idx}")
            break
    
    if not selected_box:
        print("未找到符合要求的框（最左侧前20%中没有趋近正方形的框）")
    
    return selected_box


def calculate_x_croped(slices_info: List[Dict]) -> Tuple[Optional[int], Optional[Tuple], Dict[int, int]]:
    """
    计算 x_croped 值
    
    Args:
        slices_info: 所有切片信息列表
        
    Returns:
        tuple: (x_croped值, selected_box, slice_x_croped_final字典)
    """
    # 根据切片数量决定处理逻辑
    slices_to_process = determine_slices_to_process(slices_info)
    
    # 处理选定的切片
    for index, slice_info in enumerate(slices_to_process):
        img, binary, rects = preprocess_and_crop_image(slice_info['slice'], index, slice_info['start_y'])
    
    # 处理slice_x_croped_values中的所有target_box
    print("开始处理slice_x_croped_values中的target_box...")
    
    # 收集所有target_box并按x坐标排序
    all_boxes = []
    for slice_idx, target_box in slice_x_croped_values.items():
        if target_box is not None:
            # target_box是单个tuple (x, y, w, h)，不是list
            if isinstance(target_box, (list, tuple)) and len(target_box) == 4:
                x, y, w, h = target_box
                all_boxes.append((x, y, w, h, slice_idx))
    
    print(f"总共找到 {len(all_boxes)} 个target_box")
    
    # 按x坐标排序
    all_boxes.sort(key=lambda box: box[0])
    
    # 找到最合适的框
    selected_box = find_best_target_box(all_boxes)
    
    x_croped = None
    slice_x_croped_final = {}
    
    if selected_box:
        # 基于选中的框计算x_croped
        x, y, w, h, slice_idx = selected_box
        x_croped = x + w  # 使用框的右边界作为裁剪位置
        print(f"基于选中框计算的x_croped值: {x_croped}")
        
        # 创建单独的字典存储x_croped值，不覆盖原始target_box数据
        for slice_idx in slice_x_croped_values.keys():
            slice_x_croped_final[slice_idx] = x_croped
        print("已计算所有slice的x_croped值")
    else:
        print("未找到合适的框，x_croped设置为None")
    
    return x_croped, selected_box, slice_x_croped_final