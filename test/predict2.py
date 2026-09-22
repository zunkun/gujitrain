# -*- coding: utf-8 -*-
"""
调试版切图：输出所有检测框不筛选（由原根目录 predict2.py 迁入并参数化）。

用法:
    python test/predict2.py bookcontent
    python test/predict2.py line --books book2
"""
import argparse
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from ultralytics import YOLO

# 让相对路径（images/output/weights）统一基于项目根目录
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# ===================== 【任务 -> 模型映射】 =====================
TASK_WEIGHTS = {
    "bookcontent": "weights/bookcontent.pt",
    "gujiline": "weights/gujiline.pt",
}

# ===================== 【可配置】 =====================
PROCESS_BOOKS = ["book2"]  # 默认处理哪本书，可用 --books 覆盖
MAX_WORKERS = 2
# ======================================================

BASE_IMAGES = "./images"
BASE_OUTPUT = "./output"


def parse_args():
    parser = argparse.ArgumentParser(description="古籍切图调试模式（输出全部检测框）")
    parser.add_argument("task", choices=list(TASK_WEIGHTS.keys()), help="使用哪个模型")
    parser.add_argument("--books", nargs="*", default=None, help="覆盖待处理书籍列表")
    args = parser.parse_args()
    return args


args = parse_args()
MODEL_PATH = TASK_WEIGHTS[args.task]

print("=" * 60)
print(f"📌 加载模型: {MODEL_PATH} (任务: {args.task})")
model = YOLO(MODEL_PATH)
print("✅ 模型加载成功！")
print("=" * 60)


# 调试：所有框都输出，不筛选
def process_image(fname, img_dir, out_dir, threshold):
    try:
        img_path = os.path.join(img_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            return f"❌ {fname} 读取失败"

        H, W = img.shape[:2]
        is_double = W >= threshold
        results = model(img, verbose=False)
        boxes = results[0].boxes

        if len(boxes) == 0:
            return f"❌ {fname} 无框"

        name, ext = os.path.splitext(fname)
        left_idx = 0
        right_idx = 0

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx = (x1 + x2) / 2

            # 裁剪
            crop = img[int(y1) : int(y2), int(x1) : int(x2)]

            if cx < W / 2:
                out_file = f"{out_dir}/{name}-l{left_idx}{ext}"
                left_idx += 1
            else:
                out_file = f"{out_dir}/{name}-r{right_idx}{ext}"
                right_idx += 1

            cv2.imwrite(out_file, crop)

        total = left_idx + right_idx
        return f"✅ {fname} | 输出 {total} 个框"

    except Exception as e:
        return f"❌ {fname} | 错误：{str(e)}"


# 一本一本处理
books = args.books if args.books else PROCESS_BOOKS
for book in books:
    print(f"\n📚 处理 {book}")
    img_dir = os.path.join(BASE_IMAGES, book, "img0")
    out_dir = os.path.join(BASE_OUTPUT, book, "img0")

    # 清空
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # 扫描图片
    img_files = []
    widths = []
    for f in os.listdir(img_dir):
        if f.endswith(("jpg", "png", "jpeg")):
            img_files.append(f)
            widths.append(cv2.imread(os.path.join(img_dir, f)).shape[1])

    if not img_files:
        print(f"❌ {book} 目录下无有效图片，跳过")
        continue

    avg_w = np.mean(widths)
    threshold = avg_w * 0.8

    # 多线程处理
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_image, f, img_dir, out_dir, threshold) for f in img_files]
        for i, future in enumerate(as_completed(futures), 1):
            print(f"[{i}/{len(img_files)}]", future.result())

print("\n🎉 调试完成！所有框都已输出！")
