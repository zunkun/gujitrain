# -*- coding: utf-8 -*-
"""
调试版检测：输出全部检测框不筛选，按类别配色，可指定任务与文件名前缀。

用法:
    python test/debug_boxes.py bookcontent                     # 位置参数写法
    python test/debug_boxes.py --task bookcontent --books b1   # 只看 b1 开头的图
    python test/debug_boxes.py --task bookcontent --books b1 --limit 3
    python test/debug_boxes.py --task gujiline --books book2

输出: output/debug/<task>/<原图名>.jpg
"""
import argparse
import os
import re
import sys

import cv2
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

TASK_WEIGHTS = {
    "bookcontent": "weights/bookcontent.pt",
    "gujiline": "weights/gujiline.pt",
}

# 每个类别一种颜色（BGR），顺序对应类别编号
PALETTE_BGR = [
    (255, 100, 0),   # 0 蓝
    (0, 0, 255),     # 1 红
    (0, 170, 0),     # 2 绿
    (0, 150, 255),   # 3 橙
    (180, 0, 180),   # 4 紫
    (180, 180, 0),   # 5 青
]

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def natural_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]


def parse_args():
    ap = argparse.ArgumentParser(description="古籍检测调试模式（输出全部框，按类配色）")
    ap.add_argument("task", nargs="?", choices=list(TASK_WEIGHTS.keys()),
                    help="任务名（位置参数写法）")
    ap.add_argument("--task", dest="task_flag", choices=list(TASK_WEIGHTS.keys()),
                    help="任务名（--task 写法）")
    ap.add_argument("--books", nargs="*", default=None,
                    help="只看文件名以这些前缀开头的图，如 --books b1（默认全部）")
    ap.add_argument("--limit", type=int, default=0, help="每个前缀最多处理几张（0=不限）")
    ap.add_argument("--conf", type=float, default=0.25, help="置信度阈值（默认 0.25）")
    ap.add_argument("--clean", action="store_true", help="先清空输出目录（默认不清）")
    args = ap.parse_args()
    if not (args.task_flag or args.task):
        ap.error("必须指定任务名，如：python test/debug_boxes.py --task bookcontent")
    args.task = args.task_flag or args.task
    return args


args = parse_args()

from ultralytics import YOLO  # noqa: E402  放在参数解析之后，--help 时不必加载 torch

MODEL_PATH = TASK_WEIGHTS[args.task]
yaml_file = os.path.join("data", args.task, f"{args.task}.yaml")
names = []
img_dir = os.path.join("data", args.task, "images")
if os.path.isfile(yaml_file):
    with open(yaml_file, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    ns = cfg.get("names")
    names = [ns[k] for k in sorted(ns)] if isinstance(ns, dict) else (ns or [])
    root = cfg.get("path", os.path.dirname(yaml_file))
    if not os.path.isabs(root):
        root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(yaml_file)), root))
    img_dir = os.path.join(root, "images")

out_dir = os.path.join("output", "debug", args.task)

print("=" * 60)
print(f"📌 模型: {MODEL_PATH} | 任务: {args.task}")
print(f"🏷️  类别: {dict(enumerate(names)) if names else '(未知，用模型自带)'}")
model = YOLO(MODEL_PATH)
if not names:
    m = model.names
    names = [m[k] for k in sorted(m)] if isinstance(m, dict) else list(m)
print("✅ 模型加载成功！")
print("=" * 60)

if not os.path.isdir(img_dir):
    sys.exit(f"❌ 图片目录不存在: {img_dir}")
if args.clean and os.path.isdir(out_dir):
    import shutil
    shutil.rmtree(out_dir)
os.makedirs(out_dir, exist_ok=True)

files = sorted((f for f in os.listdir(img_dir) if f.lower().endswith(IMAGE_EXTS)), key=natural_key)
if args.books:
    files = [f for f in files if any(f.startswith(p) for p in args.books)]
    if not files:
        sys.exit(f"❌ 没有文件名以 {args.books} 开头的图")
if args.limit:
    kept, seen = [], {}
    for f in files:
        pref = next((p for p in (args.books or [""]) if f.startswith(p)), "")
        seen[pref] = seen.get(pref, 0) + 1
        if seen[pref] <= args.limit:
            kept.append(f)
    files = kept

print(f"🖼️  {img_dir} → {out_dir} | 处理 {len(files)} 张\n")

class_total = {i: 0 for i in range(len(names))}
no_box = []

for i, f in enumerate(files, 1):
    img = cv2.imread(os.path.join(img_dir, f))
    if img is None:
        print(f"[{i}/{len(files)}] ❌ {f} 读取失败")
        continue
    H, W = img.shape[:2]
    res = model(img, conf=args.conf, verbose=False)[0]
    boxes = res.boxes

    th = max(2, round(min(H, W) / 500))
    n = 0
    for box in boxes:
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].cpu().numpy())
        cid = int(box.cls[0].cpu().numpy())
        conf = float(box.conf[0].cpu().numpy())
        color = PALETTE_BGR[cid % len(PALETTE_BGR)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, th)

        name = names[cid] if 0 <= cid < len(names) else str(cid)
        text = f"{cid}:{name} {conf:.2f}"
        fs = max(0.8, min(H, W) / 900)
        (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, max(2, th))
        ty = max(y1, tht + 8)
        cv2.rectangle(img, (x1, ty - tht - 8), (x1 + tw + 8, ty + 4), color, -1)
        cv2.putText(img, text, (x1 + 4, ty - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, (255, 255, 255), max(2, th), cv2.LINE_AA)

        if 0 <= cid < len(class_total):
            class_total[cid] += 1
        n += 1

    cv2.imwrite(os.path.join(out_dir, f), img)
    if n == 0:
        no_box.append(f)
    print(f"[{i}/{len(files)}] {'✅' if n else '❌'} {f} | {n} 框")

print("\n" + "-" * 60)
print("📊 类别统计:")
for cid, c in class_total.items():
    print(f"   {cid} {names[cid] if cid < len(names) else '?'}: {c} 框")
print(f"   合计 {sum(class_total.values())} 框 / {len(files)} 张图")
if no_box:
    print(f"\n⚠️ 无检测框 {len(no_box)} 张: {no_box[:10]}{' ...' if len(no_box) > 10 else ''}")
print(f"\n✅ 调试图已输出到: {out_dir}")
