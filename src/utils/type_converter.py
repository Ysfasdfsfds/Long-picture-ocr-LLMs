"""
类型转换工具模块
处理numpy类型到Python原生类型的转换
"""

import numpy as np
from typing import Any, Union, List, Dict


def convert_numpy_to_python(obj: Any) -> Any:
    """
    递归地将numpy类型转换为Python原生类型
    
    Args:
        obj: 需要转换的对象
        
    Returns:
        转换后的对象，确保所有numpy类型都被转换为Python原生类型
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_to_python(item) for item in obj)
    else:
        return obj


def ensure_serializable(data: Any) -> Any:
    """
    确保数据可以被JSON序列化
    
    Args:
        data: 需要处理的数据
        
    Returns:
        可序列化的数据
    """
    return convert_numpy_to_python(data)


def convert_box_format(box: Union[List[List[float]], tuple], 
                      to_format: str = 'xywh') -> Union[List[List[float]], tuple]:
    """
    转换边界框格式
    
    Args:
        box: 边界框，可以是以下格式之一：
            - OCR格式: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            - xywh格式: (x, y, w, h)
        to_format: 目标格式 ('xywh' 或 'points')
        
    Returns:
        转换后的边界框
    """
    if to_format == 'xywh':
        if isinstance(box, (tuple, list)) and len(box) == 4 and isinstance(box[0], (int, float)):
            # 已经是xywh格式
            return box
        elif isinstance(box[0], list):
            # OCR格式转xywh
            x_coords = [pt[0] for pt in box]
            y_coords = [pt[1] for pt in box]
            x_min = min(x_coords)
            y_min = min(y_coords)
            x_max = max(x_coords)
            y_max = max(y_coords)
            return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    elif to_format == 'points':
        if isinstance(box, (tuple, list)) and len(box) == 4 and isinstance(box[0], (int, float)):
            # xywh转points
            x, y, w, h = box
            return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
        elif isinstance(box[0], list):
            # 已经是points格式
            return box
    
    raise ValueError(f"不支持的转换格式: {to_format}")


def get_box_center(box: Union[List[List[float]], tuple]) -> tuple:
    """
    获取边界框的中心点坐标
    
    Args:
        box: 边界框
        
    Returns:
        (center_x, center_y)
    """
    if isinstance(box[0], list):
        # OCR格式
        x_coords = [pt[0] for pt in box]
        y_coords = [pt[1] for pt in box]
        return (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))
    else:
        # xywh格式
        x, y, w, h = box
        return (x + w / 2, y + h / 2)


def get_box_bounds(box: Union[List[List[float]], tuple]) -> Dict[str, float]:
    """
    获取边界框的边界值
    
    Args:
        box: 边界框
        
    Returns:
        包含min_x, min_y, max_x, max_y的字典
    """
    if isinstance(box[0], list):
        # OCR格式
        x_coords = [pt[0] for pt in box]
        y_coords = [pt[1] for pt in box]
        return {
            'min_x': min(x_coords),
            'min_y': min(y_coords),
            'max_x': max(x_coords),
            'max_y': max(y_coords)
        }
    else:
        # xywh格式
        x, y, w, h = box
        return {
            'min_x': x,
            'min_y': y,
            'max_x': x + w,
            'max_y': y + h
        }


def is_point_in_box(point: tuple, box: Union[List[List[float]], tuple]) -> bool:
    """
    判断点是否在边界框内
    
    Args:
        point: (x, y) 坐标
        box: 边界框
        
    Returns:
        是否在框内
    """
    bounds = get_box_bounds(box)
    x, y = point
    return (bounds['min_x'] <= x <= bounds['max_x'] and 
            bounds['min_y'] <= y <= bounds['max_y'])