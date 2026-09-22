import re
import cv2
import numpy as np

def natural_sort_key(s):
    """
    自然排序键函数：将字符串中的数字部分转为整数进行排序。
    例如：img1, img2, img10 → img1, img2, img10
    """
    return [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', s)]

def generate_colors(n=20):
    """生成 n 种高对比颜色 (BGR)"""
    colors = []
    for i in range(n):
        hue = i / n
        rgb = cv2.cvtColor(np.uint8([[[hue * 180, 230, 230]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(map(int, rgb)))
    return colors

def paginate(items, per_page):
    """将列表分页，返回生成器，每页 per_page 个元素"""
    for i in range(0, len(items), per_page):
        yield items[i:i + per_page]
