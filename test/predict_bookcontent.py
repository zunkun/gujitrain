# -*- coding: utf-8 -*-
"""
bookcontent 批量切图（抽样模式）：从 data/bookcontent/images 取图，
先按自然顺序排序，每隔 5 张取 1 张，用 weights/bookcontent.pt 检测文本区域并左右页切分。

用法:
    python test/predict_bookcontent.py
    python test/predict_bookcontent.py --every 10
"""

import argparse
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from ultralytics import YOLO

# 让相对路径（data/output/weights）统一基于项目根目录
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# ===================== 【本脚本固定使用 bookcontent 模型】 =====================
MODEL_PATH = "weights/bookcontent.pt"
TARGET_TASK = "bookcontent"

# ===================== 【抽样与目录配置】 =====================
BASE_IMAGES = "./data/bookcontent/images"  # 图片来源：训练集目录
BASE_OUTPUT = "./output/bookcontent"  # 切图输出目录（平铺）
EVERY_N = 1  # 每隔 N 张取 1 张（1 = 全部处理）
# =========================================================================

# 单本书内部图片并行线程数
MAX_WORKERS = os.cpu_count()


def natural_sort_key(s):
    """自然排序：book1_page_2 排在 book1_page_10 前面"""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]


def parse_args():
    parser = argparse.ArgumentParser(description="bookcontent 训练集抽样切图")
    parser.add_argument(
        "--every", type=int, default=EVERY_N, help="每隔几张取 1 张（默认 5）"
    )
    args = parser.parse_args()
    return args


args = parse_args()

# 全局仅加载一次模型，避免多线程重复打印日志
print("=" * 60)
print(f"📌 加载模型: {MODEL_PATH} (任务: {TARGET_TASK})")
model = YOLO(MODEL_PATH)
print("✅ 模型加载成功！")
print("=" * 60)


def process_single_image(fname, img_dir, out_dir, threshold):
    """单张图片处理逻辑：取两个最大框自动分左右，抗页面偏移。
    保存位置：双页 → out_dir；单页 → out_dir/single；未检出 → out_dir/nobookcontent
    """
    try:
        img_path = os.path.join(img_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            return f"❌ {fname} | 图片读取失败"

        H, W = img.shape[:2]
        is_double = W >= threshold

        # 推理，关闭冗余输出
        results = model(img, verbose=False)
        boxes = results[0].boxes
        if len(boxes) == 0:
            # 未检出：原图存入 nobookcontent/ 便于人工排查
            cv2.imwrite(os.path.join(out_dir, "nobookcontent", fname), img)
            return f"❌ {fname} | 未检测到文本框 | 已存 nobookcontent/"

        # 收集所有框：x1, y1, x2, y2, cx中心点, area面积
        box_list = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            area = (x2 - x1) * (y2 - y1)
            cx = (x1 + x2) / 2
            box_list.append([x1, y1, x2, y2, cx, area])

        # ===================== 【抗偏移左右分组】 =====================
        box_list_sorted = sorted(box_list, key=lambda x: x[5], reverse=True)
        best_left = None
        best_right = None

        if is_double:
            # 双页：取前两大框，按中心点自动分左右
            if len(box_list_sorted) >= 2:
                box1, box2 = box_list_sorted[0], box_list_sorted[1]
                if box1[4] < box2[4]:
                    best_left, best_right = box1, box2
                else:
                    best_left, best_right = box2, box1
            elif len(box_list_sorted) == 1:
                single_box = box_list_sorted[0]
                if single_box[4] < W * 0.6:
                    best_left = single_box
                else:
                    best_right = single_box
        else:
            # 单页：只取最大框
            best_right = box_list_sorted[0] if box_list_sorted else None
        # ============================================================

        name, ext = os.path.splitext(fname)
        marked_tags = []

        # ===================== 【标注方框：左右不同颜色，不切割】 =====================
        # 左页：绿色；右页：红色
        COLOR_LEFT = (0, 200, 0)  # BGR 绿
        COLOR_RIGHT = (0, 0, 255)  # BGR 红
        thickness = max(2, round(min(H, W) / 500))

        if best_left:
            x1, y1, x2, y2, _, _ = map(int, best_left)
            cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_LEFT, thickness)
            cv2.putText(
                img,
                "L",
                (x1 + 10, y1 + 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                COLOR_LEFT,
                max(2, thickness),
            )
            marked_tags.append("L")
        if best_right:
            x1, y1, x2, y2, _, _ = map(int, best_right)
            cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_RIGHT, thickness)
            cv2.putText(
                img,
                "R",
                (x1 + 10, y1 + 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                COLOR_RIGHT,
                max(2, thickness),
            )
            marked_tags.append("R")

        # 保存整张标注图（文件名不变）：单页 → single/，双页 → 根目录
        save_dir = os.path.join(out_dir, "single") if not is_double else out_dir
        cv2.imwrite(os.path.join(save_dir, f"{name}{ext}"), img)

        return f"✅ {fname} | {'双页' if is_double else '单页'} | 已标注: {','.join(marked_tags)}" + (
            " | 已存 single/" if not is_double else ""
        )

    except Exception as e:
        return f"❌ {fname} | 运行异常: {str(e)}"


# ===================== 主流程 =====================
img_dir = BASE_IMAGES
out_dir = BASE_OUTPUT

if not os.path.isdir(img_dir):
    print(f"❌ 图片目录不存在：{img_dir}")
    raise SystemExit(1)

# 1. 清空旧输出
if os.path.exists(out_dir):
    print(f"🗑️  清空历史文件：{out_dir}")
    shutil.rmtree(out_dir)
os.makedirs(out_dir, exist_ok=True)
os.makedirs(os.path.join(out_dir, "single"), exist_ok=True)         # 单页标注图
os.makedirs(os.path.join(out_dir, "nobookcontent"), exist_ok=True)  # 未检出原图

# 2. 扫描图片：先自然排序，再每隔 N 张取 1 张
suffix_list = ("jpg", "png", "jpeg", "tif")
all_files = [f for f in os.listdir(img_dir) if f.lower().endswith(suffix_list)]
all_files.sort(key=natural_sort_key)

img_files = all_files[:: args.every]
total_img = len(img_files)
if total_img == 0:
    print(f"❌ {img_dir} 目录下无有效图片")
    raise SystemExit(1)

print(
    f"✅ 图片总数: {len(all_files)} | 自然排序后每隔 {args.every} 张取 1 张 → 抽取 {total_img} 张"
)
print(f"📂 输入: {img_dir}")
print(f"📂 输出: {out_dir}")

# 3. 计算单双页阈值（按抽样图片的平均宽度）
width_list = []
for f in img_files:
    w = cv2.imread(os.path.join(img_dir, f)).shape[1]
    width_list.append(w)

avg_width = np.mean(width_list)
threshold = avg_width * 0.8
print(f"✅ 平均宽度: {avg_width:.0f} px | 单页阈值: {threshold:.0f} px")

# 4. 多线程处理
print(f"\n🚀 启动多线程处理（线程数: {MAX_WORKERS}）\n")
success_count = 0

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    task_futures = [
        executor.submit(process_single_image, f, img_dir, out_dir, threshold)
        for f in img_files
    ]

    for idx, future in enumerate(as_completed(task_futures), 1):
        log_info = future.result()
        print(f"[{idx}/{total_img}] {log_info}")
        if "✅" in log_info:
            success_count += 1

print(f"\n📊 处理完成 | 成功: {success_count}/{total_img}")
print("=" * 60)
print("🎉 抽样切图任务执行完毕！")
print("=" * 60)
