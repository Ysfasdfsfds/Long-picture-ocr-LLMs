"""
内容标记处理相关函数
"""
from typing import List, Dict, Tuple, Any, Optional
import re
import cv2
import numpy as np
from pathlib import Path
import time
import random


def get_box_y_min(box):
    """获取box的中心Y坐标"""
    if isinstance(box[0], list):  # OCR box格式
        # return sum(pt[1] for pt in box) / 4
        return box[3][1]
    else:  # Avatar box格式 (x, y, w, h)
        return box[1] + box[3] / 2


def get_box_center_y(box):
    """获取box的中心Y坐标"""
    if isinstance(box[0], list):  # OCR box格式
        return sum(pt[1] for pt in box) / 4
        # return box[3][1]
    else:  # Avatar box格式 (x, y, w, h)
        return box[1] + box[3] / 2


def get_box_x_min(box):
    """获取文本框的最小X坐标"""
    if isinstance(box, (tuple, list)) and len(box) == 4 and isinstance(box[0], (int, float)):
        # 头像格式: (x, y, w, h)
        return box[0]
    else:
        # OCR格式: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        return min(point[0] for point in box)


def get_box_y_max(box):
    """获取文本框的最大Y坐标（底部）"""
    if isinstance(box, (tuple, list)) and len(box) == 4 and isinstance(box[0], (int, float)):
        # 头像格式: (x, y, w, h)
        return box[1] + box[3]  # y + h
    else:
        # OCR格式: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        return max(point[1] for point in box)


def get_avatar_y_min(box):
    """获取头像框的最小Y坐标（顶部）"""
    if isinstance(box, (tuple, list)) and len(box) == 4 and isinstance(box[0], (int, float)):
        # 头像格式: (x, y, w, h)
        return box[1]  # y
    else:
        # 如果是其他格式，返回最小Y
        return min(point[1] for point in box)


def detect_green_content_box(image: np.ndarray, box: List) -> bool:
    """
    检测文本框区域是否为绿色背景（本人消息框）
    
    Args:
        image: 原始图像
        box: 文本框坐标
        
    Returns:
        是否为绿色框
    """
    try:
        # 获取文本框区域
        points = np.array(box, dtype=np.int32)
        min_x = max(0, int(np.min(points[:, 0])))
        max_x = min(image.shape[1], int(np.max(points[:, 0])))
        min_y = max(0, int(np.min(points[:, 1])))
        max_y = min(image.shape[0], int(np.max(points[:, 1])))
        
        if max_x <= min_x or max_y <= min_y:
            return False
        
        # 提取区域图像
        roi = image[min_y:max_y, min_x:max_x]
        # 保存ROI到本地，用于调试
        debug_dir = Path("./output_images/debug")
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成唯一文件名，使用时间戳和随机数
        timestamp = int(time.time() * 1000)
        random_num = random.randint(1000, 9999)
        roi_filename = f"roi_{timestamp}_{random_num}.jpg"
        
        # 保存ROI图像
        roi_path = debug_dir / roi_filename
        cv2.imwrite(str(roi_path), roi)
        print(f"已保存ROI图像到: {roi_path}")
        
        if roi.size == 0:
            return False
        
        # 转换为HSV颜色空间进行绿色检测
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 定义绿色的HSV范围
        # 浅绿色范围（聊天界面中常见的绿色）
        lower_green1 = np.array([35, 40, 40])
        upper_green1 = np.array([85, 255, 255])
        
        # 创建绿色掩码
        mask = cv2.inRange(hsv, lower_green1, upper_green1)
        
        # 计算绿色像素的比例
        green_pixels = cv2.countNonZero(mask)
        total_pixels = roi.shape[0] * roi.shape[1]
        
        if total_pixels > 0:
            green_ratio = green_pixels / total_pixels
            # 如果绿色像素超过20%，认为是绿色框
            return green_ratio > 0.2
        
        return False
        
    except Exception as e:
        print(f"检测绿色框时出错: {e}")
        return False


def detect_blue_content_box(image: np.ndarray, box: List) -> bool:
    """
    检测文本框区域是否为蓝色背景（本人消息框）
    
    Args:
        image: 原始图像
        box: 文本框坐标
        
    Returns:
        是否为蓝色框
    """
    try:
        # 获取文本框区域
        points = np.array(box, dtype=np.int32)
        min_x = max(0, int(np.min(points[:, 0])))
        max_x = min(image.shape[1], int(np.max(points[:, 0])))
        min_y = max(0, int(np.min(points[:, 1])))
        max_y = min(image.shape[0], int(np.max(points[:, 1])))
        
        if max_x <= min_x or max_y <= min_y:
            return False
        
        # 提取区域图像
        roi = image[min_y:max_y, min_x:max_x]
        
        if roi.size == 0:
            return False
        
        # 转换为HSV颜色空间进行蓝色检测
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 定义蓝色的HSV范围（聊天气泡常见的淡蓝色）
        lower_blue = np.array([100, 30, 80])    # 降低饱和度下限，提高亮度下限
        upper_blue = np.array([130, 180, 255])  # 降低饱和度上限，避免深蓝色文字
        
        # 创建蓝色掩码
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # 额外检查：排除白色背景上的蓝色文字
        # 检测白色背景
        lower_white = np.array([0, 0, 200])     # 白色HSV范围
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # 计算像素比例
        blue_pixels = cv2.countNonZero(blue_mask)
        white_pixels = cv2.countNonZero(white_mask)
        total_pixels = roi.shape[0] * roi.shape[1]
        
        if total_pixels > 0:
            blue_ratio = blue_pixels / total_pixels
            white_ratio = white_pixels / total_pixels
            
            # 判断逻辑：
            # 1. 蓝色比例要足够高(>30%)
            # 2. 白色比例不能太高(<50%)，避免白底蓝字的情况
            # 3. 蓝色比例要大于白色比例，确保是蓝底而不是白底
            is_blue_background = (blue_ratio > 0.3 and 
                                white_ratio < 0.5 and 
                                blue_ratio > white_ratio)
            
            # print(f"蓝色检测 - 蓝色比例: {blue_ratio:.3f}, 白色比例: {white_ratio:.3f}, 判断为蓝色背景: {is_blue_background}")
            return is_blue_background
        
        return False
        
    except Exception as e:
        print(f"检测蓝色框时出错: {e}")
        return False


def mark_system_messages(ocr_results):
    """
    标记系统消息（如：欢迎..加入...等模式）
    
    Args:
        ocr_results: OCR结果列表
    """
    system_message_patterns = [
        r'欢迎.*加入',           # 欢迎xxx加入
        r'.*加入了群聊',         # xxx加入了群聊
        r'.*退出了群聊',         # xxx退出了群聊
        r'.*邀请.*加入了群聊',   # xxx邀请yyy加入了群聊
        r'.*撤回了.*条消息',     # xxx撤回了x条消息
        r'群聊名称.*更改为',     # 群聊名称更改为xxx
        r'.*开启了.*',          # xxx开启了群聊邀请确认
        r'.*关闭了.*',          # xxx关闭了群聊邀请确认
        r'.*设置.*为管理员',     # xxx设置yyy为管理员
        r'.*移除了.*的管理员身份', # xxx移除了yyy的管理员身份
        r'.*修改了群名称',       # xxx修改了群名称
        r'系统消息',            # 直接的系统消息标识
        r'以上是历史消息',       # 历史消息分隔符
    ]
    
    print("开始标记系统消息...")
    marked_count = 0
    
    for ocr_item in ocr_results:
        text = ocr_item['text']
        
        # 跳过已经标记过的内容
        if any(tag in text for tag in ['(时间)', '(昵称)', '(内容)', '(我的内容)', '(系统消息)']):
            continue
        
        # 检查是否匹配系统消息模式
        for pattern in system_message_patterns:
            if re.search(pattern, text):
                ocr_item['text'] = text + "(系统消息)"
                print(f"标记系统消息: {text[:50]}...")
                marked_count += 1
                break
    
    print(f"系统消息标记完成，共标记 {marked_count} 条")


def mark_time_content(ocr_results):
    """标记时间内容（使用严格的时间检测条件）"""
    # 时间模式列表
    time_patterns = [
        r'\d{4}年\d{1,2}月\d{1,2}日\d{1,2}:\d{2}',                          # 2025年6月17日9:10
        r'\d{4}年\d{1,2}月\d{1,2}日',                                       # 2025年6月17日
        r'(昨天|今天|前天|明天)(早上|上午|中午|下午|晚上|凌晨)?\d{1,2}:\d{2}',  # 昨天晚上6:23
        r'(上午|下午|早上|中午|晚上|凌晨)\d{1,2}:\d{2}',                     # 上午9:17
        r'\d{1,2}:\d{2}',                                                  # 5:51
        r'\d{4}-\d{1,2}-\d{1,2}',                                          # 年-月-日  
        r'\d{1,2}/\d{1,2}',                                                # 月/日
        r'\d{1,2}月\d{1,2}日',                                             # 月日
        r'\d{1,2}月\d{1,2}日\d{1,2}:\d{2}',                                # 7月28日14:53
        r'\d{1,2}:\d{2}:\d{2}',                                           # 时:分:秒
        r'(昨天|今天|前天|明天)',                                           # 相对日期
        r'周[一二三四五六日天]',                                            # 星期
    ]
    
    for ocr_item in ocr_results:
        text = ocr_item['text'].strip()
        
        # 排除过长的文本（超过30个字符的很可能不是纯时间）
        if len(text) > 30:
            continue
            
        # 排除包含明显非时间关键词的文本
        exclude_keywords = ['报送', '回执', '会议', '参加', '人员', '工作', '通知', '安排', '要求', '地点', '内容', 
                          '完成', '需要', '前', '后', '开始', '结束', '传包', '表格', '填写', '更新', '自测']
        if any(keyword in text for keyword in exclude_keywords):
            continue
        
        # 检查时间模式
        is_time = False
        for pattern in time_patterns:
            if re.search(pattern, text):
                match = re.search(pattern, text)
                if match:
                    matched_length = len(match.group())
                    match_ratio = matched_length / len(text)
                    
                    # 对于复合时间格式（如"昨天晚上6:23"），降低阈值要求
                    if pattern.startswith('(昨天|今天|前天|明天)') or pattern.startswith('(上午|下午|早上|中午|晚上|凌晨)'):
                        # 复合时间格式，匹配40%以上即可
                        if match_ratio >= 0.4:
                            is_time = True
                            break
                    else:
                        # 简单时间格式，仍然要求60%以上
                        if match_ratio >= 0.6:
                            is_time = True
                            break
        
        if is_time:
            ocr_item['text'] = text + "(时间)"
            print(f"标记时间: {ocr_item['text']}")
        else:
            print(f"跳过非纯时间文本: {text[:50]}...")


def is_between_avatars(start_y, end_y, avatars):
    """
    检查Y坐标范围是否在最近的两个头像框之间
    """
    if not avatars:
        return True  # 如果没有头像数据，允许通过
    
    # 按Y坐标排序头像
    sorted_avatars = sorted(avatars, 
                          key=lambda x: get_box_center_y(x['box']))
    
    # 找到包含start_y的头像对
    for i in range(len(sorted_avatars) - 1):
        avatar1 = sorted_avatars[i]
        avatar2 = sorted_avatars[i + 1]
        
        avatar1_y_max = get_box_y_max(avatar1['box'])
        avatar2_y_min = get_avatar_y_min(avatar2['box'])
        
        # 检查是否在这两个头像之间
        if avatar1_y_max <= start_y and end_y <= avatar2_y_min:
            return True
    
    return False


def is_adjacent_my_content(my_box, next_box, avatars):
    """
    检查下一条内容是否应该标记为我的内容
    
    条件：
    1. 处于最近的两个头像框之间
    （已取消X坐标对齐限制）
    """
    my_x_min = get_box_x_min(my_box)
    my_y_max = get_box_y_max(my_box)
    
    next_x_min = get_box_x_min(next_box)
    next_y_min = get_box_y_min(next_box)
    
    # 1. X坐标检查已取消 - 只要在头像范围内即可
    print(f"      X坐标信息: my_x={my_x_min}, next_x={next_x_min} (无限制)")
    
    # 2. 检查是否在最近的两个头像框之间
    print(f"      Y坐标检查: my_y_max={my_y_max}, next_y_min={next_y_min}")
    if not is_between_avatars(my_y_max, next_y_min, avatars):
        print(f"      ❌ 不在头像范围内")
        return False
    
    print(f"      ✅ 所有条件满足")
    return True


def mark_adjacent_my_content(ocr_results, my_content_boxes, avatar_positions, original_image):
    """
    基于位置推理标记相邻的我的内容
    
    逻辑：如果通过颜色检测确定该条是我的内容，如果下一条：
    1. 没有通过颜色检测
    2. 处于最近的两个头像框之间（y1max --- y2min）
    则下一条也是我的内容
    """
    # 使用传入的avatar_positions
    avatars = avatar_positions
    
    print(f"开始位置推理：有 {len(my_content_boxes)} 个我的内容框，{len(avatars)} 个头像")
    
    if not my_content_boxes or not avatars:
        print(f"跳过位置推理：my_content_boxes={len(my_content_boxes) if my_content_boxes else 0}, avatars={len(avatars) if avatars else 0}")
        return
    
    for my_content in my_content_boxes:
        my_index = my_content['index']
        my_box = my_content['box']
        my_x_min = get_box_x_min(my_box)
        my_y_max = get_box_y_max(my_box)
        
        print(f"处理我的内容[{my_index}]: '{ocr_results[my_index]['text'][:50]}...', x_min={my_x_min}, y_max={my_y_max}")
        
        # 查找下一条内容
        for next_index in range(my_index + 1, len(ocr_results)):
            next_item = ocr_results[next_index]
            
            # 跳过已经标记为我的内容的
            if "(我的内容)" in next_item['text']:
                print(f"  跳过[{next_index}]: 已是我的内容 - '{next_item['text'][:50]}...'")
                continue
            
            # 只处理内容标记
            if "(内容)" not in next_item['text']:
                print(f"  跳过[{next_index}]: 不是内容标记 - '{next_item['text'][:50]}...'")
                continue
            
            next_box = next_item['box']
            next_x_min = get_box_x_min(next_box)
            next_y_min = get_box_y_min(next_box)
            
            print(f"  检查[{next_index}]: '{next_item['text'][:50]}...', x_min={next_x_min}, y_min={next_y_min}")
            
            # 检查是否已通过颜色检测（如果已通过，跳过）
            try:
                is_green = detect_green_content_box(original_image, next_box)
                is_blue = detect_blue_content_box(original_image, next_box)
                if is_green or is_blue:
                    print(f"    跳过: 已通过颜色检测 (绿色={is_green}, 蓝色={is_blue})")
                    continue  # 已通过颜色检测，跳过
            except Exception as e:
                print(f"    颜色检测异常: {e}")
                pass
            
            # 检查位置条件
            print(f"    检查位置条件...")
            if is_adjacent_my_content(my_box, next_box, avatars):
                next_item['text'] = next_item['text'].replace("(内容)", "(我的内容)")
                print(f"✅ 基于位置推理标记为我的内容: {next_item['text']} (xmin对齐且在头像范围内)")
                # 将新标记的内容也加入列表，以便继续检查下一条
                my_content_boxes.append({'index': next_index, 'box': next_box})
            else:
                print(f"    ❌ 位置条件不满足")


def mark_green_content(ocr_results, avatar_positions, original_image):
    """标记绿色和蓝色背景的内容（基于颜色检测和位置推理）"""
    if original_image is None:
        print("原图不可用，跳过颜色内容检测")
        return
    
    # 第一轮：基于颜色检测标记我的内容
    my_content_boxes = []  # 存储已确认的我的内容框
    
    for i, ocr_item in enumerate(ocr_results):
        if "(内容)" in ocr_item['text']:
            box = ocr_item['box']
            
            # 基于绿色背景检测
            is_green = False
            try:
                is_green = detect_green_content_box(original_image, box)
            except Exception as e:
                print(f"绿色检测失败: {e}")
            
            # 基于蓝色背景检测
            is_blue = False
            try:
                is_blue = detect_blue_content_box(original_image, box)
            except Exception as e:
                print(f"蓝色检测失败: {e}")
            
            # 绿色或蓝色背景都标记为"我的内容"
            if is_green:
                ocr_item['text'] = ocr_item['text'].replace("(内容)", "(我的内容)")
                my_content_boxes.append({'index': i, 'box': box})
                print(f"标记为我的内容: {ocr_item['text']} (原因: 绿色背景)")
            elif is_blue:
                ocr_item['text'] = ocr_item['text'].replace("(内容)", "(我的内容)")
                my_content_boxes.append({'index': i, 'box': box})
                print(f"标记为我的内容: {ocr_item['text']} (原因: 蓝色背景)")
    
    # 第二轮：基于位置推理标记相邻的内容
    mark_adjacent_my_content(ocr_results, my_content_boxes, avatar_positions, original_image)


def mark_nickname_and_content_with_avatars_wechat(ocr_results, avatars, original_image):
    """基于头像位置标记昵称和内容"""
    # 一次性绘制所有头像框（只在第一次调用时执行）
    if original_image is not None:
        debug_img = original_image.copy()
        # 画出所有头像框
        for idx, avatar_item in enumerate(avatars):
            avatar_box_ = avatar_item['box']
            x, y, w, h = avatar_box_
            # 在原图上绘制矩形，使用红色(0,0,255)
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # 在框旁边添加索引编号
            cv2.putText(debug_img, f"{idx}:y={y}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 保存带有头像框的图像
        output_path = "output_images/avatars_marked_1.jpg"
        cv2.imwrite(output_path, debug_img)
        print(f"已将所有头像框绘制到原图并保存至 {output_path}")
    elif original_image is None:
        print("原图不可用，无法绘制头像框")
    
    # 收集需要插入的虚拟昵称
    virtual_nicknames_to_insert = []
    
    for i, avatar in enumerate(avatars):
        avatar_box = avatar['box']
        x_min, y_min, w, h = avatar_box
        y_max = y_min + h
        
        # 找到下一个头像的Y位置边界
        next_boundary = avatars[i+1]['box'][1] if i+1 < len(avatars) else float('inf')
        
        # 寻找在当前头像Y范围内的文本作为昵称
        nickname_found = False
        nickname_start_index = -1
        
        for j, ocr_item in enumerate(ocr_results):
            if "(时间)" in ocr_item['text']:  # 跳过已标记的时间
                continue
                
            box_y_min = get_box_y_min(ocr_item['box'])
            
            # 检查是否在头像Y范围内
            if y_min <= box_y_min <= y_max and not nickname_found:
                # 标记为昵称
                ocr_item['text'] = ocr_item['text'] + "(昵称)"
                print(f"标记昵称: {ocr_item['text']}")
                nickname_found = True
                nickname_start_index = j
                break
        
        # 如果没有找到昵称，记录需要插入虚拟昵称
        if not nickname_found:
            print(f"警告: 头像 {i} (y={y_min}-{y_max}) 附近未找到昵称，准备插入虚拟昵称")
            
            # 找到插入位置
            insert_index = len(ocr_results)  # 默认插入到末尾
            for idx, ocr_item in enumerate(ocr_results):
                item_y_min = get_box_y_min(ocr_item['box'])
                if item_y_min > y_max:
                    insert_index = idx
                    break
            
            # 创建虚拟昵称条目
            virtual_nickname = {
                'text': f"未知用户{i+1}(昵称)",
                'box': [[x_min, y_min], [x_min + w, y_min], [x_min + w, y_max], [x_min, y_max]],
                'confidence': 0.0,
                'slice_index': avatar.get('slice_index', -1),
                'virtual': True,
                'insert_index': insert_index  # 记录插入位置
            }
            
            virtual_nicknames_to_insert.append(virtual_nickname)
            print(f"准备插入虚拟昵称: {virtual_nickname['text']} 在位置 {insert_index}")
    
    # 统一插入虚拟昵称（按插入位置倒序插入，避免位置偏移）
    virtual_nicknames_to_insert.sort(key=lambda x: x['insert_index'], reverse=True)
    for virtual_nickname in virtual_nicknames_to_insert:
        insert_index = virtual_nickname['insert_index']
        del virtual_nickname['insert_index']  # 删除临时字段
        ocr_results.insert(insert_index, virtual_nickname)
        print(f"已插入虚拟昵称: {virtual_nickname['text']} 在位置 {insert_index}")
    
    # 现在标记内容（重新遍历，因为可能插入了虚拟昵称）
    for i, avatar in enumerate(avatars):
        avatar_box = avatar['box']
        x_min, y_min, w, h = avatar_box
        y_max = y_min + h
        
        # 找到下一个头像的Y位置边界
        next_boundary = avatars[i+1]['box'][1] if i+1 < len(avatars) else float('inf')
        
        # 找到对应的昵称
        nickname_index = -1
        for j, ocr_item in enumerate(ocr_results):
            if "(昵称)" in ocr_item['text']:
                box_y_min = get_box_y_min(ocr_item['box'])
                if y_min <= box_y_min <= y_max:
                    nickname_index = j
                    break
        
        if nickname_index >= 0:
            # 标记该昵称后的内容
            for k in range(nickname_index + 1, len(ocr_results)):
                next_ocr = ocr_results[k]
                if "(时间)" in next_ocr['text'] or "(昵称)" in next_ocr['text']:  # 跳过时间和昵称
                    continue
                
                next_box_y_min = get_box_y_min(next_ocr['box'])
                
                # 检查是否在当前头像区域内且未到达下一个头像边界
                if next_box_y_min > y_min and next_box_y_min < next_boundary:
                    if "(内容)" not in next_ocr['text']:  # 避免重复标记
                        next_ocr['text'] = next_ocr['text'] + "(内容)"
                        print(f"标记内容: {next_ocr['text']}")
                elif next_box_y_min >= next_boundary:
                    break


def mark_nickname_and_content_with_avatars_feishu(ocr_results, avatars, original_image):
    """基于头像位置标记昵称和内容"""

    # 计算所有头像的面积
    avatar_areas = []
    for avatar in avatars:
        x, y, w, h = avatar['box']
        area = w * h
        avatar_areas.append(area)
    
    # 计算面积均值
    if avatar_areas:
        mean_area = sum(avatar_areas) / len(avatar_areas)
        print(f"头像面积均值: {mean_area}")
        
        # 筛选面积大于等于均值的头像
        filtered_avatars = []
        for avatar in avatars:
            x, y, w, h = avatar['box']
            area = w * h
            if area >= mean_area:
                filtered_avatars.append(avatar)
            else:
                print(f"过滤掉小面积头像: 位置({x},{y}), 面积{area}")
        
        print(f"筛选前头像数量: {len(avatars)}, 筛选后头像数量: {len(filtered_avatars)}")
        avatars = filtered_avatars
    else:
        print("没有检测到头像")
    # 一次性绘制所有头像框（只在第一次调用时执行）
    if original_image is not None:
        debug_img = original_image.copy()
        # 画出所有头像框
        for idx, avatar_item in enumerate(avatars):
            avatar_box_ = avatar_item['box']
            x, y, w, h = avatar_box_
            # 在原图上绘制矩形，使用红色(0,0,255)
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # 在框旁边添加索引编号
            cv2.putText(debug_img, f"{idx}:y={y}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 保存带有头像框的图像
        output_path = "output_images/avatars_marked_1.jpg"
        cv2.imwrite(output_path, debug_img)
        print(f"已将所有头像框绘制到原图并保存至 {output_path}")
    elif original_image is None:
        print("原图不可用，无法绘制头像框")
    
    # 第一步：收集并处理所有昵称（合并连续的昵称文本）
    nickname_operations = []  # 存储昵称操作信息
    virtual_nicknames_to_insert = []
    
    for i, avatar in enumerate(avatars):
        avatar_box = avatar['box']
        x_min, y_min, w, h = avatar_box
        y_max = y_min + h
        
        # 寻找在当前头像Y范围内的文本作为昵称
        nickname_texts = []
        nickname_indices = []
        
        for j, ocr_item in enumerate(ocr_results):
            if "(时间)" in ocr_item['text'] or "(系统消息)" in ocr_item['text']:  # 跳过已标记的时间和系统消息
                continue
                
            box_y_min = get_box_y_min(ocr_item['box'])
            
            # 检查是否在头像Y范围内
            if y_min <= box_y_min <= y_max:
                nickname_texts.append(ocr_item['text'])
                nickname_indices.append(j)
        
        if nickname_indices:
            # 记录昵称合并操作
            nickname_operations.append({
                'avatar_index': i,
                'indices': nickname_indices,
                'texts': nickname_texts,
                'merged_text': " ".join(nickname_texts) + "(昵称)"
            })
            print(f"头像 {i}: 找到 {len(nickname_texts)} 个昵称片段，准备合并")
        else:
            # 记录需要插入虚拟昵称
            print(f"警告: 头像 {i} (y={y_min}-{y_max}) 附近未找到昵称，准备插入虚拟昵称")
            
            # 找到插入位置
            insert_index = len(ocr_results)  # 默认插入到末尾
            for idx, ocr_item in enumerate(ocr_results):
                item_y_min = get_box_y_min(ocr_item['box'])
                if item_y_min > y_max:
                    insert_index = idx
                    break
            
            virtual_nicknames_to_insert.append({
                'avatar_index': i,
                'insert_index': insert_index,
                'virtual_nickname': {
                    'text': f"未知用户{i+1}(昵称)",
                    'box': [[x_min, y_min], [x_min + w, y_min], [x_min + w, y_max], [x_min, y_max]],
                    'confidence': 0.0,
                    'slice_index': avatar.get('slice_index', -1),
                    'virtual': True
                }
            })
    
    # 第二步：执行昵称合并操作（从后往前处理，避免索引偏移）
    for operation in reversed(nickname_operations):
        indices = operation['indices']
        merged_text = operation['merged_text']
        
        # 更新第一个昵称项
        first_index = indices[0]
        ocr_results[first_index]['text'] = merged_text
        print(f"合并昵称: {merged_text} (来自 {len(indices)} 个文本片段)")
        
        # 删除多余的昵称项（从后往前删除）
        for idx in sorted(indices[1:], reverse=True):
            print(f"删除多余的昵称片段: {ocr_results[idx]['text']}")
            del ocr_results[idx]
    
    # 第三步：重新计算虚拟昵称的插入位置（因为OCR列表已被修改）
    for virtual_item in virtual_nicknames_to_insert:
        avatar_index = virtual_item['avatar_index']
        avatar_box = avatars[avatar_index]['box']
        x_min, y_min, w, h = avatar_box
        y_max = y_min + h
        
        # 重新计算插入位置
        insert_index = len(ocr_results)
        for idx, ocr_item in enumerate(ocr_results):
            item_y_min = get_box_y_min(ocr_item['box'])
            if item_y_min > y_max:
                insert_index = idx
                break
        
        virtual_item['insert_index'] = insert_index
    
    # 第四步：插入虚拟昵称（按插入位置倒序插入）
    virtual_nicknames_to_insert.sort(key=lambda x: x['insert_index'], reverse=True)
    for virtual_item in virtual_nicknames_to_insert:
        insert_index = virtual_item['insert_index']
        virtual_nickname = virtual_item['virtual_nickname']
        ocr_results.insert(insert_index, virtual_nickname)
        print(f"已插入虚拟昵称: {virtual_nickname['text']} 在位置 {insert_index}")
    
    # 现在标记内容（重新遍历，因为可能插入了虚拟昵称）
    for i, avatar in enumerate(avatars):
        avatar_box = avatar['box']
        x_min, y_min, w, h = avatar_box
        y_max = y_min + h
        
        # 找到下一个头像的Y位置边界
        next_boundary = avatars[i+1]['box'][1] if i+1 < len(avatars) else float('inf')
        
        # 找到对应的昵称
        nickname_index = -1
        for j, ocr_item in enumerate(ocr_results):
            if "(昵称)" in ocr_item['text']:
                box_y_min = get_box_y_min(ocr_item['box'])
                if y_min <= box_y_min <= y_max:
                    nickname_index = j
                    break
        
        if nickname_index >= 0:
            # 标记该昵称后的内容
            for k in range(nickname_index + 1, len(ocr_results)):
                next_ocr = ocr_results[k]
                if "(时间)" in next_ocr['text'] or "(昵称)" in next_ocr['text'] or "(系统消息)" in next_ocr['text']:  # 跳过时间、昵称和系统消息
                    continue
                
                next_box_y_min = get_box_y_min(next_ocr['box'])
                
                # 检查是否在当前头像区域内且未到达下一个头像边界
                if next_box_y_min > y_min and next_box_y_min < next_boundary:
                    if "(内容)" not in next_ocr['text']:  # 避免重复标记
                        next_ocr['text'] = next_ocr['text'] + "(内容)"
                        print(f"标记内容: {next_ocr['text']}")
                elif next_box_y_min >= next_boundary:
                    break


def remark_content_with_deduplicated_data_wechat(deduplicated_ocr, deduplicated_avatars, original_image):
    """
    基于去重后的数据重新标记内容
    
    Args:
        deduplicated_ocr: 去重后的OCR结果
        deduplicated_avatars: 去重后的头像位置
        original_image: 原始图像
        
    Returns:
        list: 标记后的OCR结果
    """
    print("开始重新标记...")
    
    # 创建工作副本，避免修改原始数据
    marked_results = []
    for ocr_item in deduplicated_ocr:
        marked_item = ocr_item.copy()
        marked_item['original_text'] = marked_item['text']  # 保存原始文本
        marked_results.append(marked_item)
    
    # 按Y坐标排序头像和OCR结果
    sorted_avatars = sorted(deduplicated_avatars, key=lambda x: x['center_y'])
    sorted_ocr = sorted(marked_results, key=lambda x: get_box_center_y(x['box']))
    
    print(f"处理 {len(sorted_avatars)} 个头像和 {len(sorted_ocr)} 个OCR结果")
    
    # 1. 首先标记时间
    mark_time_content(sorted_ocr)
    
    # 2. 基于头像位置标记昵称和内容
    mark_nickname_and_content_with_avatars_wechat(sorted_ocr, sorted_avatars, original_image)
    
    # 重新排序，因为可能插入了虚拟昵称
    sorted_ocr.sort(key=lambda x: get_box_center_y(x['box']))
    print(f"插入虚拟昵称后，OCR结果总数: {len(sorted_ocr)}")
    
    # 3. 标记绿色背景的内容为"我的内容"
    mark_green_content(sorted_ocr, sorted_avatars, original_image)
    
    print("重新标记完成")
    return sorted_ocr  # 返回处理后的sorted_ocr，包含虚拟昵称


def remark_content_with_deduplicated_data_feishu(deduplicated_ocr, deduplicated_avatars, original_image):
    """
    基于去重后的数据重新标记内容
    
    Args:
        deduplicated_ocr: 去重后的OCR结果
        deduplicated_avatars: 去重后的头像位置
        original_image: 原始图像
        
    Returns:
        list: 标记后的OCR结果
    """
    print("开始重新标记...")
    
    # 创建工作副本，避免修改原始数据
    marked_results = []
    for ocr_item in deduplicated_ocr:
        marked_item = ocr_item.copy()
        marked_item['original_text'] = marked_item['text']  # 保存原始文本
        marked_results.append(marked_item)
    
    # 按Y坐标排序头像和OCR结果
    sorted_avatars = sorted(deduplicated_avatars, key=lambda x: x['center_y'])
    sorted_ocr = sorted(marked_results, key=lambda x: get_box_center_y(x['box']))
    
    print(f"处理 {len(sorted_avatars)} 个头像和 {len(sorted_ocr)} 个OCR结果")
    
    # 1. 首先标记时间
    mark_time_content(sorted_ocr)
    
    # 2. 标记系统消息
    mark_system_messages(sorted_ocr)
    
    # 3. 基于头像位置标记昵称和内容
    mark_nickname_and_content_with_avatars_feishu(sorted_ocr, sorted_avatars, original_image)
    
    # 重新排序，因为可能插入了虚拟昵称
    sorted_ocr.sort(key=lambda x: get_box_center_y(x['box']))
    print(f"插入虚拟昵称后，OCR结果总数: {len(sorted_ocr)}")
    
    # 3. 标记绿色背景的内容为"我的内容"
    # mark_green_content(sorted_ocr, sorted_avatars, original_image)
    
    print("重新标记完成")
    return sorted_ocr  # 返回处理后的sorted_ocr，包含虚拟昵称