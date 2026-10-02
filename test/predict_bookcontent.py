# -*- coding: utf-8 -*-
"""
bookcontent 批量预测（抽样模式）：取图 → 自然排序 → 每隔 N 张取 1 张，
用 weights/bookcontent.pt 检测并把检测框标记回原图。

模型为两类，**两类互斥**（一张图要么出 harfcontent，要么出 fullcontent）：
    0 = harfcontent（半幅内容区域）
    1 = fullcontent（整幅内容区域）

检测结果先按互斥规则**后处理消解**（见 resolve_boxes），再标记、分桶：
    规则1  fullcontent 的框宽必须 > 图像宽度的 70%，否则直接判为失败检测剔除
    规则2  harfcontent ≥2 个 → fullcontent 一律不存在（全删）
    规则3  harfcontent 恰好 1 个 → fullcontent 与它比置信度，高的留、低的删；
           没有 harfcontent 时 fullcontent 原样保留
    （因此后处理之后 both 不可能出现，互斥被强制成立）

本脚本只做「识别 + 标记」，**不做任何切割**：
  - 只画**保留下来**的框，按类配色，框上写「类别号:类名 置信度」
  - **两类的标签文字画在框的不同位置**（harfcontent 贴上沿 / fullcontent 贴下沿）：
    fullcontent 是大框、harfcontent 落在它内部，若都贴左上角，
    文字与彩色底色会互相覆盖、难以区分
  - 被规则剔除的框会在日志与汇总里报出来（便于回查模型误检）

按识别结果分桶输出：
    harfcontent/ ：harfcontent ≥2 个
    fullcontent/ ：只剩 fullcontent（含本来就没有 harfcontent 的情况）
    single/      ：恰好 1 个 harfcontent（**单独的 fullcontent 不算 single**）
    nocontent/   ：无有效框（存原图便于排查）

用法:
    python test/predict_bookcontent.py
    python test/predict_bookcontent.py --every 10
    python test/predict_bookcontent.py --conf 0.4
"""

import argparse
import os
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

# 让相对路径（output/weights）统一基于项目根目录
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# ===================== 【本脚本固定使用 bookcontent 模型】 =====================
MODEL_PATH = "weights/bookcontent.pt"
TARGET_TASK = "bookcontent"
YAML_PATH = "data/bookcontent/bookcontent.yaml"

# 类别编号 = data/bookcontent/bookcontent.yaml 里 names 列表的位置，顺序不可随意调整。
# 两类互斥：一张图只应出现其中一类。
HARFCONTENT_CLASS_ID = 0  # 半幅内容区域
FULLCONTENT_CLASS_ID = 1  # 整幅内容区域

# 每类一种颜色（BGR），与 test/debug_boxes.py 的调色板保持一致
CLASS_COLORS = {
    HARFCONTENT_CLASS_ID: (255, 100, 0),  # 0 蓝
    FULLCONTENT_CLASS_ID: (0, 0, 255),  # 1 红
}
FALLBACK_COLOR = (0, 170, 0)  # 兜底色（yaml 里出现更多类别时用）

# 标签文字相对检测框的位置：两类**必须错开**。
# fullcontent 是大框（整幅），harfcontent 是小框（半幅，落在大框内部）；
# 若都贴左上角，同图命中两类时文字与底色会互相覆盖、无法区分。
#   top    = 贴上沿（画在框的外上方；顶到图像上边界时自动落进框内）
#   bottom = 贴下沿（画在框的内底部）
LABEL_POS = {
    HARFCONTENT_CLASS_ID: "top",
    FULLCONTENT_CLASS_ID: "bottom",
}

# 规则1：fullcontent 的框宽必须 > 图像宽度 × 该比例，否则视为失败检测直接剔除。
# 依据：fullcontent 语义是「整幅内容区」，宽度不到七成的框不可能是整幅。
FULL_MIN_WIDTH_RATIO = 0.70

# 输出分桶（目录名 → 说明），顺序即汇总打印顺序
# 注意：互斥规则是后处理强制的，消解完不可能再出现「两类同时命中」，故没有 both 桶
BUCKETS = (
    ("harfcontent", "harfcontent ≥2 个"),
    ("fullcontent", "只剩 fullcontent"),
    ("single", "恰好 1 个 harfcontent"),
    ("nocontent", "无有效框"),
)

# 互斥消解的规则名 → 说明（用于汇总打印「哪条规则剔了多少张」）
RULE_DESC = {
    "narrow_full": "fullcontent 宽 ≤70%，判为失败检测剔除",
    "multi_half": "harfcontent ≥2 个 → fullcontent 全部剔除",
    "conf_full": "单 harfcontent 置信度低于 fullcontent → 剔除 harfcontent",
    "conf_half": "单 harfcontent 置信度高于 fullcontent → 剔除 fullcontent",
}
# =========================================================================

# ===================== 【抽样与目录配置】 =====================
# BASE_IMAGES = (
#     r"C:\Users\liuzu\Documents\guji\tasks\0016\stages\extract"  # 图片来源：guji 流水线 extract 阶段产物
# )
BASE_IMAGES = "./data/bookcontent/images"
BASE_OUTPUT = "./output/bookcontent"  # 输出目录（下分 4 个结果桶）
EVERY_N = 1  # 每隔 N 张取 1 张（1 = 全部处理）
CONF_THRES = 0.25  # 置信度阈值
# ===========================================================

# 单本书内部图片并行线程数
MAX_WORKERS = os.cpu_count()

# ⚠️ 推理必须串行：ultralytics 的 YOLO 实例**不是线程安全**的。
# 多线程同时进 model() 时，若 self.predictor 尚未建立，各线程会各自 setup_model →
# Weight.fuse()，而 fuse 是对同一个共享模型做原地修改（delattr(m, "bn")），
# 后到的线程就会撞上 AttributeError: bn（表现为随机几张图「运行异常: bn」）。
# 图片读取/写盘仍可并行，只有推理这一步加锁。
INFER_LOCK = threading.Lock()


def natural_sort_key(s):
    """自然排序：2.jpg 排在 10.jpg 前面"""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]


def load_class_names():
    """从数据 yaml 读类别名，读不到则退回模型自带的 names"""
    names = []
    if os.path.isfile(YAML_PATH):
        with open(YAML_PATH, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        ns = cfg.get("names")
        if isinstance(ns, dict):
            names = [ns[k] for k in sorted(ns)]
        elif ns:
            names = list(ns)
    if not names:
        m = model.names
        names = [m[k] for k in sorted(m)] if isinstance(m, dict) else list(m)
    return names


def cls_name(cid):
    """类别号 → 类名（越界时显示为 ?）"""
    return NAMES[cid] if 0 <= cid < len(NAMES) else "?"


def draw_box(img, x1, y1, x2, y2, cid, conf, thickness):
    """画框 + 「类别号:类名 置信度」标签（彩色底 + 白字）。

    标签位置按类错开（见 LABEL_POS）：harfcontent 贴框的上沿、fullcontent 贴框的下沿，
    避免两类同图时文字与底色互相覆盖；文字块越界时自动收进图像内。
    """
    H, W = img.shape[:2]
    color = CLASS_COLORS.get(cid, FALLBACK_COLOR)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    text = f"{cid}:{cls_name(cid)} {conf:.2f}"
    stroke = max(2, thickness)
    fs = max(0.8, min(H, W) / 900)
    (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, stroke)

    if LABEL_POS.get(cid) == "bottom":
        # 贴框内下沿；框抵着图像底边时上移，保证整块底色都在图内
        ty = min(max(y2, tht + 8), H - 4)
    else:
        # 贴框上方（画出框外）；顶到图像上边界时落进框内
        ty = max(y1, tht + 8)
    tx = min(max(x1, 0), max(W - tw - 8, 0))  # 左右不越界

    cv2.rectangle(img, (tx, ty - tht - 8), (tx + tw + 8, ty + 4), color, -1)
    cv2.putText(
        img,
        text,
        (tx + 4, ty - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        fs,
        (255, 255, 255),
        stroke,
        cv2.LINE_AA,
    )


def resolve_boxes(raw_boxes, img_w):
    """按互斥规则消解 harfcontent / fullcontent 冲突，返回 (保留的框, 剔除说明列表)。

    raw_boxes: [(cid, conf, x1, y1, x2, y2), ...]（模型原始输出，未过滤）
    三条规则（顺序即优先级）：
      规则1 fullcontent 框宽必须 > img_w × FULL_MIN_WIDTH_RATIO，否则判为失败检测剔除
      规则2 harfcontent ≥2 个 → fullcontent 一律不存在（全删）
      规则3 harfcontent 恰好 1 个 → fullcontent（取有效框里置信度最高的）与它比置信度，
            高的留、低的删
    """
    half, full_valid, others = [], [], []
    narrow_full = 0
    for b in raw_boxes:
        cid, _conf, x1, _y1, x2, _y2 = b
        if cid == HARFCONTENT_CLASS_ID:
            half.append(b)
        elif cid == FULLCONTENT_CLASS_ID:
            if (x2 - x1) / img_w > FULL_MIN_WIDTH_RATIO:
                full_valid.append(b)
            else:
                narrow_full += 1
        else:
            others.append(b)

    notes = []
    if narrow_full:
        notes.append(("narrow_full", f"剔除{narrow_full}个fullcontent(宽≤70%)"))

    kept_half, kept_full = half, full_valid
    if len(half) >= 2:
        # 规则2：两半幅都出来了，fullcontent 一定是错的
        kept_full = []
        if full_valid:
            notes.append(
                (
                    "multi_half",
                    f"剔除{len(full_valid)}个fullcontent(与{len(half)}个harfcontent冲突)",
                )
            )
    elif len(half) == 1:
        # 规则3：只剩一个 harfcontent 时，和 fullcontent 比置信度定胜负
        if full_valid:
            best_full = max(full_valid, key=lambda b: b[1])
            if best_full[1] > half[0][1]:
                kept_half = []
                notes.append(
                    (
                        "conf_full",
                        f"剔除harfcontent({half[0][1]:.2f}<fullcontent {best_full[1]:.2f})",
                    )
                )
            else:
                kept_full = []
                notes.append(
                    (
                        "conf_half",
                        f"剔除{len(full_valid)}个fullcontent(最高{best_full[1]:.2f}<harfcontent {half[0][1]:.2f})",
                    )
                )
    return kept_half + kept_full + others, notes


def classify(counts):
    """按「保留下来」的框定桶：harfcontent / fullcontent / single / nocontent

    互斥已被 resolve_boxes 强制消解，两类不会同时留下。
    """
    n_half = counts.get(HARFCONTENT_CLASS_ID, 0)
    n_full = counts.get(FULLCONTENT_CLASS_ID, 0)
    if n_full:
        return "fullcontent"  # 单独的 fullcontent 不算 single
    if n_half >= 2:
        return "harfcontent"
    if n_half == 1:
        return "single"
    return "nocontent"


def parse_args():
    parser = argparse.ArgumentParser(description="bookcontent 批量预测标记（不切割）")
    parser.add_argument(
        "--every", type=int, default=EVERY_N, help="每隔几张取 1 张（默认 1 = 全部）"
    )
    parser.add_argument(
        "--conf", type=float, default=CONF_THRES, help="置信度阈值（默认 0.25）"
    )
    args = parser.parse_args()
    return args


args = parse_args()

# 全局仅加载一次模型，避免多线程重复打印日志
print("=" * 60)
print(f"📌 加载模型: {MODEL_PATH} (任务: {TARGET_TASK})")
model = YOLO(MODEL_PATH)
NAMES = load_class_names()
print(f"🏷️  类别: {dict(enumerate(NAMES))}")
if NAMES[:2] != ["harfcontent", "fullcontent"]:
    print(
        f"⚠️  类别名与预期 (0=harfcontent, 1=fullcontent) 不一致，"
        f"请核对 {YAML_PATH} 与模型权重！"
    )
print("✅ 模型加载成功！")
print("=" * 60)

# 预热一次：把 predictor setup + BN fuse 提前在单线程里做完，
# 否则线程池里第一批线程会同时触发 setup，直接踩上面那个竞态。
with INFER_LOCK:
    model(np.zeros((64, 64, 3), dtype=np.uint8), conf=args.conf, verbose=False)
print("🔥 预热完成（predictor 已就绪）")


def process_single_image(fname, img_dir, out_dir, conf):
    """单张图片处理：推理 → 互斥消解 → 只画保留的框 → 按桶存图，不做切割。
    按最终结果存入 out_dir/<桶>/：harfcontent / fullcontent / single / nocontent
    """
    try:
        img_path = os.path.join(img_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            return f"❌ {fname} | 图片读取失败", {}, "", []

        # 推理（两类都要，互斥所以谁都可能是"命中"的那类）
        # 加锁串行：YOLO 实例非线程安全，其余线程只在读图/写盘阶段并行
        with INFER_LOCK:
            results = model(img, conf=conf, verbose=False)
        boxes = results[0].boxes

        H, W = img.shape[:2]
        raw_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            raw_boxes.append(
                (
                    int(box.cls[0].cpu().numpy()),
                    float(box.conf[0].cpu().numpy()),
                    x1,
                    y1,
                    x2,
                    y2,
                )
            )

        # 互斥后处理：fullcontent 先过 70% 宽度筛，再按 harfcontent 的个数/置信度定去留
        kept, notes = resolve_boxes(raw_boxes, W)

        # 只画保留下来的框（两类标签位置错开，不会互相覆盖）
        thickness = max(2, round(min(H, W) / 500))
        counts = {}
        for cid, confv, x1, y1, x2, y2 in kept:
            draw_box(img, int(x1), int(y1), int(x2), int(y2), cid, confv, thickness)
            counts[cid] = counts.get(cid, 0) + 1

        bucket = classify(counts)
        cv2.imwrite(os.path.join(out_dir, bucket, fname), img)

        note_txt = (" | 消解: " + "; ".join(t for _, t in notes)) if notes else ""
        if bucket == "nocontent":
            return f"❌ {fname} | nocontent | 无有效框{note_txt}", counts, bucket, notes

        detail = " ".join(f"{cls_name(c)}×{n}" for c, n in sorted(counts.items()))
        return f"✅ {fname} | {bucket} | {detail}{note_txt}", counts, bucket, notes

    except Exception as e:
        return f"❌ {fname} | 运行异常: {str(e)}", {}, "", []


# ===================== 主流程 =====================
img_dir = BASE_IMAGES
out_dir = BASE_OUTPUT

if not os.path.isdir(img_dir):
    print(f"❌ 图片目录不存在：{img_dir}")
    raise SystemExit(1)

# 1. 清空旧输出，重建 5 个结果桶
if os.path.exists(out_dir):
    print(f"🗑️  清空历史文件：{out_dir}")
    shutil.rmtree(out_dir)
for bucket, _ in BUCKETS:
    os.makedirs(os.path.join(out_dir, bucket), exist_ok=True)

# 2. 扫描图片：先自然排序，再每隔 N 张取 1 张
suffix_list = (".jpg", ".png", ".jpeg", ".tif", ".tiff", ".bmp")
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
print(f"📂 输出: {out_dir}/<harfcontent|fullcontent|single|nocontent>/")
print(f"🔎 置信度阈值: {args.conf}")

# 3. 多线程处理
print(f"\n🚀 启动多线程处理（线程数: {MAX_WORKERS}）\n")
bucket_count = {b: 0 for b, _ in BUCKETS}
bucket_files = {b: [] for b, _ in BUCKETS}
class_total = {}
rule_count = {}

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    task_futures = [
        executor.submit(process_single_image, f, img_dir, out_dir, args.conf)
        for f in img_files
    ]

    for idx, future in enumerate(as_completed(task_futures), 1):
        log_info, counts, bucket, notes = future.result()
        print(f"[{idx}/{total_img}] {log_info}")
        if bucket:
            bucket_count[bucket] += 1
            bucket_files[bucket].append(log_info.split(" | ")[0].split(" ")[-1])
        for cid, n in counts.items():
            class_total[cid] = class_total.get(cid, 0) + n
        for key, _txt in notes:
            rule_count[key] = rule_count.get(key, 0) + 1

# 4. 汇总
print("\n" + "-" * 60)
print("📊 分类统计（按识别结果分桶）:")
for bucket, desc in BUCKETS:
    print(f"   {bucket + '/':<14} {bucket_count[bucket]:>4} 张   ({desc})")
print(f"   {'合计':<13} {sum(bucket_count.values()):>4} 张")

print("\n📊 类别统计（保留的检测框数）:")
for cid in sorted(set(list(class_total.keys()) + [0, 1])):
    print(f"   {cid} {cls_name(cid)}: {class_total.get(cid, 0)} 框")
print(f"   合计 {sum(class_total.values())} 框")

if rule_count:
    print("\n🔧 互斥消解（后处理剔除的图数）:")
    for key, desc in RULE_DESC.items():
        if rule_count.get(key):
            print(f"   {rule_count[key]:>4} 张   {key} — {desc}")
if bucket_files["nocontent"]:
    print(f"\n❓ 无有效框: {len(bucket_files['nocontent'])} 张")
    print(
        f"   {bucket_files['nocontent'][:10]}{' ...' if len(bucket_files['nocontent']) > 10 else ''}"
    )
print(f"\n🎉 任务执行完毕！标记图输出到: {out_dir}")
print("=" * 60)
