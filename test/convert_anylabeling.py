# -*- coding: utf-8 -*-
"""
把 X-AnyLabeling / LabelMe 导出的 .json 转成 YOLO 训练用的 .txt。

关键点：YOLO 的 txt 里**不写类名**，第一列是**类别编号**；
编号顺序由数据集 yaml 的 names 列表决定（names 的第 0 个 = 编号 0）。
所以本脚本直接从 yaml 读 names，把 json 里的 "label" 名映射成编号，避免手改出错。

用法:
    # 先干跑，只看会写出什么，不动任何文件
    python test/convert_anylabeling.py --src data/bookcontent/b1 --task bookcontent --dry-run

    # 实际转换（写入 data/bookcontent/labels/）
    python test/convert_anylabeling.py --src data/bookcontent/b1 --task bookcontent

    # 顺便把 json 旁边的图同步进 data/bookcontent/images/（已存在则不动）
    python test/convert_anylabeling.py --src data/bookcontent/b1 --task bookcontent --copy-images

支持矩形与多边形（多边形取外接矩形）。未知类名会直接报错终止，不会静默丢弃。
"""
import argparse
import glob
import os
import shutil
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def load_dataset(task, yaml_file=None):
    yaml_file = yaml_file or os.path.join("data", task, f"{task}.yaml")
    if not os.path.isfile(yaml_file):
        sys.exit(f"❌ 找不到配置: {yaml_file}")
    with open(yaml_file, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    root = cfg.get("path", ".")
    if not os.path.isabs(root):
        root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(yaml_file)), root))
    names = cfg.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return names or [], os.path.join(root, "images"), os.path.join(root, "labels"), yaml_file


def parse_args():
    ap = argparse.ArgumentParser(description="X-AnyLabeling/LabelMe json → YOLO txt")
    ap.add_argument("--src", required=True, help="放 .json 的目录，如 data/bookcontent/b1")
    ap.add_argument("--task", required=True, help="任务名，决定用哪个 yaml 取类别顺序")
    ap.add_argument("--yaml", default=None, help="直接指定 yaml（默认 data/<task>/<task>.yaml）")
    ap.add_argument("--copy-images", action="store_true", help="把同名图片同步进 images/（不覆盖已有）")
    ap.add_argument("--dry-run", action="store_true", help="只报告，不写文件")
    return ap.parse_args()


def shape_to_box(shape, W, H):
    """矩形/多边形 → YOLO 归一化 (cx, cy, w, h)。"""
    pts = shape.get("points") or []
    if not pts:
        return None
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    # 夹到图内，避免标到边缘外
    x1, x2 = max(0.0, x1), min(float(W), x2)
    y1, y2 = max(0.0, y1), min(float(H), y2)
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None
    return ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H)


def main():
    args = parse_args()
    names, img_dir, lab_dir, yaml_file = load_dataset(args.task, args.yaml)
    label2id = {n: i for i, n in enumerate(names)}

    jsons = sorted(glob.glob(os.path.join(args.src, "*.json")))
    if not jsons:
        sys.exit(f"❌ {args.src} 下没有 .json")

    print(f"📂 配置: {yaml_file}  →  类别: {dict(enumerate(names))}")
    print(f"📥 来源: {args.src}（{len(jsons)} 个 json）")
    print(f"📤 标注输出: {lab_dir}" + ("（干跑，不写入）" if args.dry_run else ""))
    if args.copy_images:
        print(f"🖼️  图片同步: {img_dir}" + ("（干跑，不写入）" if args.dry_run else ""))
    print("-" * 60)

    if not args.dry_run:
        os.makedirs(lab_dir, exist_ok=True)

    stat = {i: 0 for i in range(len(names))}
    unlabeled = []
    missing_img = []
    written = copied = 0

    for jp in jsons:
        stem = os.path.splitext(os.path.basename(jp))[0]
        with open(jp, encoding="utf-8") as fh:
            data = __import__("json").load(fh)

        W, H = data.get("imageWidth"), data.get("imageHeight")
        if not W or not H:
            sys.exit(f"❌ {os.path.basename(jp)} 缺少 imageWidth/imageHeight，无法归一化")

        rows, labels_here = [], []
        for shape in data.get("shapes") or []:
            label = shape.get("label")
            if label not in label2id:
                sys.exit(f"❌ {os.path.basename(jp)} 出现未知类名 {label!r}，"
                         f"yaml 里只有 {names}；请先把该类别加进 yaml 的 names 再转")
            box = shape_to_box(shape, W, H)
            if box is None:
                print(f"   ⚠️ {os.path.basename(jp)}: 类 {label} 的框无效（面积<=0），已跳过")
                continue
            rows.append((label2id[label], box))
            labels_here.append(label)

        if not rows:
            unlabeled.append(stem)

        # 有标注才写 txt；无标注不写空文件（会产生"负样本"语义，需人工确认后再建）
        content = "".join(
            f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for cid, (cx, cy, w, h) in rows
        )
        if rows and not args.dry_run:
            with open(os.path.join(lab_dir, stem + ".txt"), "w", encoding="utf-8", newline="\n") as fh:
                fh.write(content)
        if rows:
            for cid, _ in rows:
                stat[cid] += 1
            written += 1

        # 同步图片
        img_src = None
        for ext in IMAGE_EXTS:
            cand = os.path.join(args.src, stem + ext)
            if os.path.isfile(cand):
                img_src = cand
                break
        if args.copy_images and img_src:
            if not args.dry_run:
                os.makedirs(img_dir, exist_ok=True)
            dst = os.path.join(img_dir, os.path.basename(img_src))
            if os.path.exists(dst):
                pass
            else:
                if not args.dry_run:
                    shutil.copy2(img_src, dst)
                copied += 1
        if img_src is None:
            missing_img.append(stem)

        brief = ", ".join(f"{label2id[l]}:{l}" for l in labels_here) or "（无标注）"
        print(f"  {stem} | {len(rows)} 框 | {brief}")

    print("-" * 60)
    print("📊 类别统计:")
    for i, n in enumerate(names):
        print(f"   {i} {n}: {stat[i]} 框")
    print(f"   合计 {sum(stat.values())} 框 | 写出 txt {written} 个 | 同步图片 {copied} 张")
    if unlabeled:
        print(f"\n⚠️ 这些 json 没有任何 shape，未写 txt: {unlabeled}")
    if missing_img:
        print(f"\n⚠️ 这些 stem 没找到同名图片: {missing_img}")
    if not args.dry_run:
        print(f"\n✅ 完成。改完标注记得删掉 data/{args.task}/labels.cache（改 nc 时也必须删）")


if __name__ == "__main__":
    main()
