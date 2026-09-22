# -*- coding: utf-8 -*-
"""
完整代码：YOLO 检测 + 过滤 + 异常标注 + 报告 + JSON 数据输出
主要功能：
  1. 加载 YOLO 模型，检测指定类别（如 'gujiline'）
  2. 对检测框进行三级过滤：
       a) 宽高比例及高度阈值过滤
       b) 剔除被完全包含的框（保留最大框）
       c) 对中心互相包含且垂直重叠的相邻框，保留宽度较大的
  3. 异常检测：相邻间隙过大、中心被包含
  4. 生成 HTML 报告（调用 report.py）
  5. 生成 JSON 数据：Array<{name, lines: [{left, right, width}]}>
  
修改说明：
  - 输入输出目录统一为 D:\\xz\\xqc\\crop3
  - 输出图片命名为 原文件名（不含扩展名）-detect.jpg
"""

import os
import cv2
import numpy as np
import json
from ultralytics import YOLO
from utils import natural_sort_key, generate_colors
from report import generate_report
from PIL import Image, ImageDraw, ImageFont


# ---------- 公共过滤条件函数 ----------
def width_valid(width, avg_width, min_ratio, max_ratio):
    """检查宽度是否在平均宽度的 [min_ratio, max_ratio] 范围内（第一步：合理宽度）"""
    return (width >= avg_width * min_ratio) & (width <= avg_width * max_ratio)


def height_valid(height, min_height):
    """检查高度是否大于等于最小高度阈值（第一步：合理高度）"""
    return height >= min_height


def is_box_valid(box, avg_width, min_ratio, max_ratio, min_height):
    """组合宽高检查，用于后续过滤"""
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    return width_valid(w, avg_width, min_ratio, max_ratio) and height_valid(h, min_height)


def filter_contained_boxes(boxes):
    """
    第二步：大小包含过滤。
    移除被其他框完全包含的框（保留面积最大的框）。
    思路：按面积降序排序，保留不被任何已保留框包含的框。
    boxes: (N,4) numpy array
    返回保留框的索引列表
    """
    if len(boxes) == 0:
        return []
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_idx = np.argsort(areas)[::-1]  # 面积从大到小
    keep = []
    for i in sorted_idx:
        box_i = boxes[i]
        contained = False
        for j in keep:
            box_j = boxes[j]
            # 若 box_i 被 box_j 完全包含（四边界均包含），则标记为被包含
            if (box_j[0] <= box_i[0] and box_j[1] <= box_i[1] and
                box_j[2] >= box_i[2] and box_j[3] >= box_i[3]):
                contained = True
                break
        if not contained:
            keep.append(i)
    return keep


def filter_adjacent_by_center(boxes, img_height, width_ratio_min, width_ratio_max, min_height_margin):
    """
    第三步：相互包含中线过滤。
    规则：当两个框的水平中心互相在对方的 x 区间内，且垂直方向有重叠时，
          认为它们是相邻的重复框，保留宽度较大的那个（需满足宽高有效条件）。
    返回保留框的索引列表
    """
    if len(boxes) <= 1:
        return list(range(len(boxes)))
    widths = boxes[:, 2] - boxes[:, 0]
    centers_x = (boxes[:, 0] + boxes[:, 2]) / 2
    avg_width_all = np.mean(widths)
    min_height = img_height - min_height_margin
    sorted_idx = np.argsort(centers_x)  # 按中心 x 排序
    keep = set(range(len(boxes)))
    for i in range(len(sorted_idx) - 1):
        idx_i = sorted_idx[i]
        idx_j = sorted_idx[i+1]
        box_i = boxes[idx_i]
        box_j = boxes[idx_j]
        ci = centers_x[idx_i]
        cj = centers_x[idx_j]
        # 中心互相包含
        contain_i = (box_i[0] <= cj <= box_i[2])
        contain_j = (box_j[0] <= ci <= box_j[2])
        if not (contain_i and contain_j):
            continue
        # 水平和垂直重叠检查（实际只有水平重叠是必须的，这里保留垂直判断）
        x_overlap = (box_i[0] < box_j[2]) and (box_j[0] < box_i[2])
        y_overlap = (box_i[1] < box_j[3]) and (box_j[1] < box_i[3])
        if not (x_overlap and y_overlap):
            continue
        # 保留宽度较大的
        if widths[idx_i] >= widths[idx_j]:
            keep_idx = idx_i
            del_idx = idx_j
        else:
            keep_idx = idx_j
            del_idx = idx_i
        # 确保保留的框符合宽高条件（防止误删）
        if is_box_valid(boxes[keep_idx], avg_width_all, width_ratio_min, width_ratio_max, min_height):
            keep.discard(del_idx)
    return [idx for idx in range(len(boxes)) if idx in keep]


# ---------- 异常检测 ----------
def detect_abnormalities(boxes, avg_width):
    """
    异常检测规则（两种）：
      1. 相邻列间距过大：当相邻框（按 x1 排序）的间隙 > 0.5 * avg_width 时，视为异常。
      2. 中心被包含：某框的中心 x 落在另一个框的 x 区间内，视为异常（可能表示重叠或漏检）。
    返回异常描述列表，每条为 (类型, 详情字符串)
    """
    if len(boxes) < 2:
        return []

    abnormalities = []
    # 按 x1 排序
    sorted_indices = np.argsort(boxes[:, 0])
    sorted_boxes = boxes[sorted_indices]

    # 异常1：相邻列间距
    for i in range(len(sorted_boxes) - 1):
        gap = sorted_boxes[i+1, 0] - sorted_boxes[i, 2]   # 下一框x1 - 当前框x2
        if gap > 0.5 * avg_width:
            abnormalities.append(("spacing", f"相邻列间隙 {gap:.1f} > {0.5*avg_width:.1f} (平均宽)"))

    # 异常2：中心被包含
    centers_x = (boxes[:, 0] + boxes[:, 2]) / 2
    for i in range(len(boxes)):
        for j in range(len(boxes)):
            if i == j:
                continue
            if boxes[j, 0] <= centers_x[i] <= boxes[j, 2]:
                abnormalities.append(("contain", f"框{i+1}的中心被框{j+1}的x区间包含"))
                break   # 每个框只记录一次被包含

    return abnormalities


def process_images(model, input_dir, output_dir, target_class, conf_thres,
                   width_ratio_min, width_ratio_max, min_height_margin,
                   edge_threshold=20, font_path="simhei.ttf"):
    """
    核心处理函数：
      1. 读取图片，YOLO 推理
      2. 应用三级过滤（合理宽高 → 大小包含 → 相互包含中线）
      3. 检测异常并在图上标注
      4. 绘制编号
      5. 返回 results_data（用于报告）和 json_data（用于 JSON 输出）
    
    输出图片命名：原文件名（不含扩展名）-detect.jpg
    """
    os.makedirs(output_dir, exist_ok=True)

    # 获取目标类别 ID
    class_names = model.names
    target_id = None
    for idx, name in class_names.items():
        if name == target_class:
            target_id = idx
            break
    if target_id is None:
        raise ValueError(f"类别 '{target_class}' 不存在，可用类别：{list(class_names.values())}")

    img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
    all_files = [f for f in os.listdir(input_dir) if f.lower().endswith(img_exts)]
    all_files.sort(key=natural_sort_key)
    if not all_files:
        print(f"❌ 在 {input_dir} 中没有图片")
        return [], []

    colors = generate_colors(20)
    results_data = []   # 用于生成报告
    json_data = []      # 用于保存 JSON 数据

    for img_name in all_files:
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        h_img, w_img = img.shape[:2]

        # 构造输出文件名：原文件名（不含扩展名）-detect.jpg
        base_name = os.path.splitext(img_name)[0]
        out_name = base_name + "-detect.jpg"
        out_path = os.path.join(output_dir, out_name)

        results = model(img_path, conf=conf_thres)
        boxes = results[0].boxes

        # 初始化当前图片的 line 列表（JSON 用）
        lines = []

        # 若无检测框，直接保存原图并记录空数据
        if boxes is None or len(boxes) == 0:
            cv2.imwrite(out_path, img)
            results_data.append({
                'img_name': img_name,
                'orig_count': 0,
                'filtered_count': 0,
                'img_path': out_path,
                'abnormal': []
            })
            json_data.append({'name': img_name, 'lines': lines})
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)

        class_mask = (cls_ids == target_id)
        if not np.any(class_mask):
            # 无目标类，保存原图
            cv2.imwrite(out_path, img)
            results_data.append({
                'img_name': img_name,
                'orig_count': len(boxes),
                'filtered_count': 0,
                'img_path': out_path,
                'abnormal': []
            })
            json_data.append({'name': img_name, 'lines': lines})
            continue

        class_boxes = xyxy[class_mask]
        class_count = len(class_boxes)

        # ---------- 第一步：合理宽高过滤 ----------
        widths = class_boxes[:, 2] - class_boxes[:, 0]
        heights = class_boxes[:, 3] - class_boxes[:, 1]
        avg_width = np.mean(widths)
        min_height = h_img - min_height_margin

        width_ok = width_valid(widths, avg_width, width_ratio_min, width_ratio_max)
        height_ok = height_valid(heights, min_height)
        keep_mask = width_ok & height_ok

        if not np.any(keep_mask):
            cv2.imwrite(out_path, img)
            results_data.append({
                'img_name': img_name,
                'orig_count': class_count,
                'filtered_count': 0,
                'img_path': out_path,
                'abnormal': []
            })
            json_data.append({'name': img_name, 'lines': lines})
            continue

        filtered_boxes = class_boxes[keep_mask]

        # ---------- 第二步：大小包含过滤 ----------
        keep_indices = filter_contained_boxes(filtered_boxes)
        if len(keep_indices) == 0:
            cv2.imwrite(out_path, img)
            results_data.append({
                'img_name': img_name,
                'orig_count': class_count,
                'filtered_count': 0,
                'img_path': out_path,
                'abnormal': []
            })
            json_data.append({'name': img_name, 'lines': lines})
            continue

        final_boxes = filtered_boxes[keep_indices]

        # ---------- 第三步：相互包含中线过滤 ----------
        adj_keep = filter_adjacent_by_center(final_boxes, h_img,
                                             width_ratio_min, width_ratio_max, min_height_margin)
        if len(adj_keep) == 0:
            cv2.imwrite(out_path, img)
            results_data.append({
                'img_name': img_name,
                'orig_count': class_count,
                'filtered_count': 0,
                'img_path': out_path,
                'abnormal': []
            })
            json_data.append({'name': img_name, 'lines': lines})
            continue

        final_boxes = final_boxes[adj_keep]

        # 按 x1 降序（从右到左）排列，便于后续编号
        sorted_indices = np.argsort(final_boxes[:, 0])[::-1]
        sorted_boxes = final_boxes[sorted_indices].copy()

        # ---------- 特殊修正（针对文件名后缀为 -l.jpg 的图片剔除最左侧误检） ----------
        if img_name.lower().endswith("-l.jpg"):
            current_widths = sorted_boxes[:, 2] - sorted_boxes[:, 0]
            if len(current_widths) > 0:
                avg_width_all = np.mean(current_widths)
                last_box = sorted_boxes[-1]
                last_width = last_box[2] - last_box[0]
                last_x1 = last_box[0]
                if last_width < avg_width_all and last_x1 < edge_threshold:
                    sorted_boxes = sorted_boxes[:-1]
                    print(f"🔧 {img_name}: 剔除最左侧误检框 (宽度={last_width:.1f}, 平均={avg_width_all:.1f}, x1={last_x1})")

        if len(sorted_boxes) == 0:
            cv2.imwrite(out_path, img)
            results_data.append({
                'img_name': img_name,
                'orig_count': class_count,
                'filtered_count': 0,
                'img_path': out_path,
                'abnormal': []
            })
            json_data.append({'name': img_name, 'lines': lines})
            continue

        # ========== 构建当前图片的 lines 数据（用于 JSON） ==========
        for box in sorted_boxes:
            x1, y1, x2, y2 = box
            left = float(x1)
            right = float(x2)
            width = float(x2 - x1)
            lines.append({'left': left, 'right': right, 'width': width})

        # ========== 异常检测与标注（使用 PIL 绘制中文） ==========
        final_avg_width = np.mean(sorted_boxes[:, 2] - sorted_boxes[:, 0])
        abnormal_list = detect_abnormalities(sorted_boxes, final_avg_width)

        if abnormal_list:
            # 构建多行异常信息文本
            lines_text = ["⚠ 检测到异常："]
            for typ, detail in abnormal_list:
                lines_text.append(detail)

            # 使用 Pillow 绘制中文（因 OpenCV 不支持中文）
            try:
                font_size = max(24, int(h_img / 25))
                font = ImageFont.truetype(font_path, font_size)
            except Exception as e:
                print(f"⚠️ 字体加载失败，使用默认字体（中文可能乱码）: {e}")
                font = ImageFont.load_default()

            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)

            # 计算文本区域大小并居中
            text_widths = []
            total_height = 0
            line_spacing = 10
            for line in lines_text:
                bbox = draw.textbbox((0, 0), line, font=font)
                text_widths.append(bbox[2] - bbox[0])
                total_height += (bbox[3] - bbox[1]) + line_spacing
            max_text_width = max(text_widths)

            x_text = (w_img - max_text_width) // 2
            y_text = (h_img - total_height) // 2
            padding = 15
            # 半透明黑色背景
            draw.rectangle([x_text-padding, y_text-padding,
                            x_text+max_text_width+padding, y_text+total_height+padding],
                           fill=(0, 0, 0, 180))
            current_y = y_text
            for line in lines_text:
                draw.text((x_text, current_y), line, font=font, fill=(255, 0, 0))
                bbox = draw.textbbox((0, 0), line, font=font)
                current_y += (bbox[3] - bbox[1]) + line_spacing

            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # ---------- 绘制编号 ----------
        for idx, box in enumerate(sorted_boxes, start=1):
            x1, y1, x2, y2 = box.astype(int)
            color = colors[(idx - 1) % 20]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            label = str(idx)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            base_y = y1 + 20
            offset = (idx - 1) * 20
            text_y = base_y + offset
            if text_y - text_h < 0:
                text_y = text_h + 5 + offset

            text_x = x1
            cv2.rectangle(img,
                          (text_x, text_y - text_h - baseline),
                          (text_x + text_w, text_y + baseline),
                          color, -1)
            cv2.putText(img, label, (text_x, text_y), font, font_scale,
                        (255, 255, 255), thickness)

        cv2.imwrite(out_path, img)

        final_count = len(sorted_boxes)
        print(f"✅ {img_name}: 类别框 {class_count} → 宽度/高度过滤后 {len(filtered_boxes)} → 包含过滤后 {len(keep_indices)} → 最终 {final_count}")

        # 保存结果（报告用）
        results_data.append({
            'img_name': img_name,
            'orig_count': class_count,
            'filtered_count': final_count,
            'img_path': out_path,
            'abnormal': abnormal_list
        })

        # 将该图片的 lines 数据加入 json_data
        json_data.append({'name': img_name, 'lines': lines})

    return results_data, json_data


def main():
    """主函数：加载模型、处理图片、生成报告和 JSON 列数据"""
    MODEL_PATH = "weights/gujiline.pt"
    INPUT_DIR = r"D:\xz\xqc\crop3"
    OUTPUT_DIR = INPUT_DIR   # 输出目录与输入目录相同
    CONF_THRES = 0.25
    TARGET_CLASS = "gujiline"
    WIDTH_RATIO_MIN = 0.6
    WIDTH_RATIO_MAX = 1.46
    MIN_HEIGHT_MARGIN = 160
    EDGE_THRESHOLD = 20
    FONT_PATH = "simhei.ttf"   # 请修改为实际存在的中文字体路径

    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型 {MODEL_PATH} 不存在")
        return

    model = YOLO(MODEL_PATH)
    results_data, json_data = process_images(
        model, INPUT_DIR, OUTPUT_DIR, TARGET_CLASS, CONF_THRES,
        WIDTH_RATIO_MIN, WIDTH_RATIO_MAX, MIN_HEIGHT_MARGIN,
        edge_threshold=EDGE_THRESHOLD,
        font_path=FONT_PATH
    )

    if results_data:
        # ---------- 生成 HTML 报告 ----------
        report_path = generate_report(results_data, OUTPUT_DIR)
        print(f"📊 报告生成地址：{report_path}")

        # ---------- 生成 JSON 列数据 ----------
        json_path = os.path.join(OUTPUT_DIR, "results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"📄 JSON 数据保存至：{json_path}")

        print("💡 提示：若报告背景为黑色，请修改 report.py 中的背景色样式。")
    else:
        print("⚠️ 没有图片数据，无法生成报告和 JSON")


if __name__ == "__main__":
    main()
