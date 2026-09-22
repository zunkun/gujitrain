# -*- coding: utf-8 -*-
"""
bookcontent 批量切图：用 weights/bookcontent.pt 检测文本区域，左右页自动切分。

用法:
    python test/predict_bookcontent.py
    python test/predict_bookcontent.py --books book1 book2
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

# ===================== 【本脚本固定使用 bookcontent 模型】 =====================
MODEL_PATH = "weights/bookcontent.pt"
TARGET_TASK = "bookcontent"

# ===================== 【可配置区域 - 按你的方案设置】 =====================
# 需要处理的书籍列表，顺序执行（一本处理完再下一本）；可用 --books 覆盖
PROCESS_BOOKS = [
    "book1",
    "book2",
    "book3",
    "book4",
    "book5",
    "book6",
    "book7",
    "book8",
    "book9",
    "book10",
    "book15",
    "book16",
    "book17",
    "book18",
    # "疣贅録1",
    # "疣贅録2",
    # "疣贅録3",
    # "疣贅録4",
]
# 单本书内部图片并行线程数
MAX_WORKERS = os.cpu_count()
# =========================================================================

BASE_IMAGES = "./images"
BASE_OUTPUT = "./output"


def parse_args():
    parser = argparse.ArgumentParser(description="bookcontent 批量切图（文本区域检测）")
    parser.add_argument("--books", nargs="*", default=None, help="覆盖待处理书籍列表")
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
    """单张图片处理逻辑：取两个最大框自动分左右，抗页面偏移"""
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
            return f"❌ {fname} | 未检测到文本框"

        # 收集所有框：x1, y1, x2, y2, cx中心点, area面积
        box_list = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            area = (x2 - x1) * (y2 - y1)
            cx = (x1 + x2) / 2
            box_list.append([x1, y1, x2, y2, cx, area])

        # ===================== 【核心修复：抗偏移左右分组】 =====================
        # 按面积从大到小排序
        box_list_sorted = sorted(box_list, key=lambda x: x[5], reverse=True)
        best_left = None
        best_right = None

        if is_double:
            # 双页：取前两大框，按中心点自动分左右
            if len(box_list_sorted) >= 2:
                box1, box2 = box_list_sorted[0], box_list_sorted[1]
                # 中心点小 = 左页，中心点大 = 右页
                if box1[4] < box2[4]:
                    best_left, best_right = box1, box2
                else:
                    best_left, best_right = box2, box1
            # 只有一个框的容错处理
            elif len(box_list_sorted) == 1:
                single_box = box_list_sorted[0]
                if single_box[4] < W * 0.6:
                    best_left = single_box
                else:
                    best_right = single_box
        else:
            # 单页：只取最大框
            best_right = box_list_sorted[0] if box_list_sorted else None
        # ======================================================================

        name, ext = os.path.splitext(fname)
        saved_tags = []

        if is_double:
            # 保存左页
            if best_left:
                x1, y1, x2, y2, _, _ = map(int, best_left)
                cv2.imwrite(os.path.join(out_dir, f"{name}-l{ext}"), img[y1:y2, x1:x2])
                saved_tags.append("l")
            # 保存右页
            if best_right:
                x1, y1, x2, y2, _, _ = map(int, best_right)
                cv2.imwrite(os.path.join(out_dir, f"{name}-r{ext}"), img[y1:y2, x1:x2])
                saved_tags.append("r")
        else:
            # 单页统一保存为 -r
            if best_right:
                x1, y1, x2, y2, _, _ = map(int, best_right)
                cv2.imwrite(os.path.join(out_dir, f"{name}-r{ext}"), img[y1:y2, x1:x2])
                saved_tags.append("r")

        return f"✅ {fname} | {'双页' if is_double else '单页'} | 已保存: {','.join(saved_tags)}"

    except Exception as e:
        return f"❌ {fname} | 运行异常: {str(e)}"


# 主流程：书籍串行，单本书图片多线程并行
books = args.books if args.books else PROCESS_BOOKS
for book in books:
    print(f"\n{'=' * 60}")
    print(f"📚 开始处理：{book}")
    print(f"{'=' * 60}")

    img_dir = os.path.join(BASE_IMAGES, book, "img0")
    out_dir = os.path.join(BASE_OUTPUT, book)

    # 1. 清空当前书籍旧输出文件
    if os.path.exists(out_dir):
        print(f"🗑️  清空历史文件：{out_dir}")
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # 校验图片目录
    if not os.path.exists(img_dir):
        print(f"❌ {book} 图片目录不存在，跳过该书籍")
        continue

    # 2. 扫描当前书籍所有图片，计算平均宽度 & 单双页阈值
    print(f"\n🔍 扫描图片尺寸...")
    img_files = []
    width_list = []
    suffix_list = ("jpg", "png", "jpeg", "tif")

    for f in os.listdir(img_dir):
        if f.lower().endswith(suffix_list):
            img_files.append(f)
            w = cv2.imread(os.path.join(img_dir, f)).shape[1]
            width_list.append(w)

    total_img = len(img_files)
    if total_img == 0:
        print(f"❌ {book} 目录下无有效图片，跳过")
        continue

    avg_width = np.mean(width_list)
    threshold = avg_width * 0.8
    print(f"✅ 总计 {total_img} 张图片 | 平均宽度: {avg_width:.0f} px | 单页阈值: {threshold:.0f} px")

    # 3. 多线程处理当前书籍内所有图片
    print(f"\n🚀 启动多线程处理（线程数: {MAX_WORKERS}）\n")
    success_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        task_futures = [executor.submit(process_single_image, f, img_dir, out_dir, threshold) for f in img_files]

        for idx, future in enumerate(as_completed(task_futures), 1):
            log_info = future.result()
            print(f"[{idx}/{total_img}] {log_info}")
            if "✅" in log_info:
                success_count += 1

    # 单本书处理完成总结
    print(f"\n📊 {book} 处理完成 | 成功: {success_count}/{total_img}")

print("\n" + "=" * 60)
print("🎉 全部书籍处理任务执行完毕！")
print("=" * 60)
