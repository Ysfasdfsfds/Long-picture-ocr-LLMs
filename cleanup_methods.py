#!/usr/bin/env python3
"""
临时脚本：删除已迁移的方法定义
"""

def cleanup_migrated_methods():
    file_path = '/home/ys/桌面/ocr_long_picture-main/long_image_ocr_opencv.py'
    
    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 要删除的方法列表
    methods_to_remove = [
        '_mark_nickname_and_content_with_avatars_wechat',
        '_mark_nickname_and_content_with_avatars_feishu', 
        '_mark_green_content',
        '_mark_adjacent_my_content',
        '_is_adjacent_my_content',
        '_is_between_avatars',
        '_get_box_x_min',
        '_get_box_y_max', 
        '_get_avatar_y_min'
    ]
    
    new_lines = []
    skip_lines = False
    current_method = None
    indentation_level = 0
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # 检查是否是要删除的方法开始
        if line.strip().startswith('def '):
            method_name = line.strip().split('(')[0].replace('def ', '')
            if method_name in methods_to_remove:
                skip_lines = True
                current_method = method_name
                indentation_level = len(line) - len(line.lstrip())
                # 添加注释说明方法已迁移
                new_lines.append(f"    # {method_name} 方法已移至 refactor/remark_content.py\n")
                i += 1
                continue
        
        # 如果正在跳过方法内容
        if skip_lines:
            # 检查是否到了方法结束（下一个方法开始或者类结束）
            if line.strip().startswith('def ') and len(line) - len(line.lstrip()) <= indentation_level:
                skip_lines = False
                current_method = None
                # 不跳过这一行，继续处理
            elif line.strip() == "" or line.strip().startswith('#') or line.strip().startswith('"""') or line.strip().startswith("'''"):
                # 跳过空行、注释和文档字符串
                pass
            elif len(line.strip()) > 0 and len(line) - len(line.lstrip()) <= indentation_level and not line.strip().startswith('"""') and not line.strip().startswith("'''"):
                # 到达同级或更外层的代码，方法结束
                skip_lines = False
                current_method = None
                # 不跳过这一行，继续处理
            else:
                # 继续跳过方法内部的行
                i += 1
                continue
        
        # 如果不在跳过状态，保留这一行
        if not skip_lines:
            new_lines.append(line)
        
        i += 1
    
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print("已删除已迁移的方法定义")

if __name__ == "__main__":
    cleanup_migrated_methods()