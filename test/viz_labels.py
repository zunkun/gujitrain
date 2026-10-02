# -*- coding: utf-8 -*-
"""
标注自检：把 YOLO 标签画回原图，按类别配色，用来核对标注是否正确。

特点：不加载模型、不需要 GPU，纯读标注，秒级出结果。
顺带做数据集体检：类别号越界、坐标越界、有图无标、空标签都会报出来。

用法:
    python test/viz_labels.py --task bookcontent                 # 全部图
    python test/viz_labels.py --task bookcontent --books b1      # 只看 b1 开头
    python test/viz_labels.py --task bookcontent --books b1 --limit 3
    python test/viz_labels.py --task bookcontent --zoom 0.4      # 缩小输出省空间

输出: output/labelcheck/<task>/<原图名>.jpg
"""
import argparse
import os
import re
import sys

import cv2
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# 每个类别一种颜色（BGR）。顺序对应类别编号 0,1,2,...
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


def load_dataset(yaml_file):
    """从 yaml 解析出 names / images 目录 / labels 目录。"""
    with open(yaml_file, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    root = cfg.get("path", ".")
    if not os.path.isabs(root):
        root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(yaml_file)), root))
    names = cfg.get("names")
    if isinstance(names, dict):  # 兼容 {0: a, 1: b} 写法
        names = [names[k] for k in sorted(names)]
    return names or [], os.path.join(root, "images"), os.path.join(root, "labels")


def parse_args():
    ap = argparse.ArgumentParser(description="把 YOLO 标签画回原图做自检")
    ap.add_argument("--task", required=True, help="任务名，决定用哪个 yaml，如 bookcontent")
    ap.add_argument("--books", nargs="*", default=None,
                    help="只看文件名以这些前缀开头的图，如 --books b1（默认全部）")
    ap.add_argument("--limit", type=int, default=0, help="每个前缀最多处理几张（0=不限）")
    ap.add_argument("--zoom", type=float, default=0.0, help="输出缩放比例，如 0.4（0=保持原尺寸）")
    ap.add_argument("--no-image", action="store_true", help="只做数据体检，不输出自检图")
    return ap.parse_args()


def main():
    args = parse_args()
    yaml_file = os.path.join("data", args.task, f"{args.task}.yaml")
    if not os.path.isfile(yaml_file):
        sys.exit(f"❌ 找不到配置: {yaml_file}")

    names, img_dir, lab_dir = load_dataset(yaml_file)
    if not os.path.isdir(img_dir):
        sys.exit(f"❌ 图片目录不存在: {img_dir}")

    out_dir = os.path.join("output", "labelcheck", args.task)
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(
        (f for f in os.listdir(img_dir) if f.lower().endswith(IMAGE_EXTS)),
        key=natural_key,
    )
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

    print(f"📂 数据配置: {yaml_file}")
    print(f"🏷️  类别: {dict(enumerate(names))}")
    print(f"🖼️  图片: {img_dir}  →  输出: {out_dir}")
    print(f"📊 待处理 {len(files)} 张" + (f"（前缀 {args.books}，每个最多 {args.limit} 张）" if args.books else ""))
    print("-" * 60)

    class_total = {i: 0 for i in range(len(names))}
    no_label, empty_label, warn = [], [], []

    for f in files:
        img = cv2.imread(os.path.join(img_dir, f))
        if img is None:
            warn.append(f"{f}: 图片读取失败")
            continue
        H, W = img.shape[:2]

        lab_file = os.path.join(lab_dir, os.path.splitext(f)[0] + ".txt")
        if not os.path.isfile(lab_file):
            no_label.append(f)
            lines = []
        else:
            with open(lab_file, encoding="utf-8") as fh:
                lines = [ln.split() for ln in fh.read().splitlines() if ln.strip()]
            if not lines:
                empty_label.append(f)

        n_this = 0
        for parts in lines:
            if len(parts) < 5:
                warn.append(f"{f}: 字段数 {len(parts)} != 5 → {parts}")
                continue
            cid = int(float(parts[0]))
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            if cid < 0 or cid >= len(names):
                warn.append(f"{f}: 类别号 {cid} 越界（nc={len(names)}）")
                continue
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                warn.append(f"{f}: 坐标越界 cx={cx} cy={cy} w={bw} h={bh}")

            x1 = int((cx - bw / 2) * W)
            y1 = int((cy - bh / 2) * H)
            x2 = int((cx + bw / 2) * W)
            y2 = int((cy + bh / 2) * H)
            color = PALETTE_BGR[cid % len(PALETTE_BGR)]
            th = max(2, round(min(H, W) / 500))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, th)

            text = f"{cid}:{names[cid]}"
            fs = max(0.8, min(H, W) / 900)
            (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, max(2, th))
            ty = max(y1, tht + 8)
            cv2.rectangle(img, (x1, ty - tht - 8), (x1 + tw + 8, ty + 4), color, -1)
            cv2.putText(img, text, (x1 + 4, ty - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        fs, (255, 255, 255), max(2, th), cv2.LINE_AA)

            class_total[cid] += 1
            n_this += 1

        if args.zoom and args.zoom != 1.0:
            img = cv2.resize(img, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_AREA)
        if not args.no_image:
            cv2.imwrite(os.path.join(out_dir, f), img)

        tag = "（无标签文件）" if not os.path.isfile(lab_file) else ("（空标签/负样本）" if not lines else "")
        print(f"  {f} | {n_this} 框 {tag}")

    print("-" * 60)
    print("📊 类别统计:")
    for i, n in enumerate(names):
        print(f"   {i} {n}: {class_total[i]} 框")
    print(f"   合计: {sum(class_total.values())} 框 / {len(files)} 张图")
    if no_label:
        print(f"\n⚠️ 有图无标签 {len(no_label)} 张: {no_label[:10]}{' ...' if len(no_label) > 10 else ''}")
    if empty_label:
        print(f"ℹ️ 空标签（负样本）{len(empty_label)} 张: {empty_label[:10]}{' ...' if len(empty_label) > 10 else ''}")
    if warn:
        print(f"\n❗ 数据问题 {len(warn)} 条:")
        for w in warn[:20]:
            print(f"   {w}")
        if len(warn) > 20:
            print(f"   ... 另有 {len(warn) - 20} 条")
    print(f"\n✅ 自检图已输出到: {out_dir}")


if __name__ == "__main__":
    main()
