# ------------- 打包修复：解决 torch bn 异常问题 -------------
import torch
import torch.nn

torch.nn.modules.batchnorm._BatchNorm
torch.nn.BatchNorm2d
# -----------------------------------------------------------

import argparse
import datetime
import os
import platform
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from ultralytics import YOLO


# ===================== 模型路径（打包专用） =====================
def get_resource_path(rel_path):
    if hasattr(sys, "_MEIPASS"):
        base = sys._MEIPASS
    else:
        base = os.path.abspath(".")
    return os.path.join(base, rel_path)


# ===================== 日志写入 =====================
def write_log(log_path, msg):
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass


# ===================== 自然数排序 =====================
def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]


# ===================== 任务 -> 模型映射 =====================
TASK_WEIGHTS = {
    "bookcontent": "weights/bookcontent.pt",
    "gujiline": "weights/gujiline.pt",
}

model = None  # 在 main() 中按任务加载


# ===================== 单张图片处理 =====================
def process_single_image(fname, img_dir, out_dir, threshold):
    start_time = time.time()
    try:
        img_path = os.path.join(img_dir, fname) if img_dir else fname
        img = cv2.imread(img_path)
        if img is None:
            return {"status": "error", "file": fname, "reason": "图片读取失败", "time": time.time() - start_time}

        H, W = img.shape[:2]
        mid_x = W / 2
        results = model(img, verbose=False)
        boxes = results[0].boxes

        if len(boxes) == 0:
            out_path = os.path.join(out_dir, fname)
            cv2.imwrite(out_path, img)
            return {"status": "no_detect", "file": fname, "size": f"{W}x{H}", "time": time.time() - start_time}

        box_list = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            w = x2 - x1
            h = y2 - y1
            area = w * h
            cx = (x1 + x2) / 2
            box_list.append((area, cx, x1, y1, x2, y2))

        box_list.sort(reverse=True, key=lambda x: x[0])

        has_left = False
        has_right = False
        outputs = []
        name, ext = os.path.splitext(fname)

        for area, cx, x1, y1, x2, y2 in box_list:
            if cx < mid_x and not has_left:
                out_file = os.path.join(out_dir, f"{name}-l{ext}")
                cv2.imwrite(out_file, img[int(y1) : int(y2), int(x1) : int(x2)])
                outputs.append(f"{name}-l{ext}")
                has_left = True
            elif cx >= mid_x and not has_right:
                out_file = os.path.join(out_dir, f"{name}-r{ext}")
                cv2.imwrite(out_file, img[int(y1) : int(y2), int(x1) : int(x2)])
                outputs.append(f"{name}-r{ext}")
                has_right = True

            if has_left and has_right:
                break

        if not outputs:
            out_path = os.path.join(out_dir, fname)
            cv2.imwrite(out_path, img)
            return {"status": "no_detect", "file": fname, "size": f"{W}x{H}", "time": time.time() - start_time}

        return {
            "status": "success",
            "file": fname,
            "size": f"{W}x{H}",
            "outputs": outputs,
            "time": time.time() - start_time,
        }

    except Exception as e:
        return {"status": "error", "file": fname, "reason": str(e), "time": time.time() - start_time}


# ===================== 主程序 =====================
def main():
    global model
    parser = argparse.ArgumentParser(description="YOLO 古籍自动裁剪工具")
    parser.add_argument("--task", choices=list(TASK_WEIGHTS.keys()), default="bookcontent",
                        help="使用哪个模型：bookcontent（默认）或 line")
    parser.add_argument("--input", default=".", help="输入目录，默认当前目录")
    parser.add_argument("--out", default=None, help="输出目录，默认等于 input")
    parser.add_argument("--file", help="单张图片处理")
    args = parser.parse_args()

    # 按任务加载模型
    model_path = get_resource_path(TASK_WEIGHTS[args.task])
    print("=" * 60)
    print(f"📌 加载 YOLO 模型: {TASK_WEIGHTS[args.task]} (任务: {args.task})")
    model = YOLO(model_path)
    print("✅ 模型加载成功！")
    print("=" * 60)

    if args.file is not None:
        out_root = os.path.abspath("yoloout")
    else:
        input_dir = os.path.abspath(args.input)
        out_parent = input_dir if args.out is None else os.path.abspath(args.out)
        out_root = os.path.join(out_parent, "yoloout")

    os.makedirs(out_root, exist_ok=True)
    log_file = os.path.join(out_root, "process.log")

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cmd_line = " ".join(sys.argv)
    header = [
        "=" * 80,
        f"处理时间：{now}",
        f"计算机：{platform.node()}",
        f"命令：{cmd_line}",
        f"输出目录：{out_root}",
        "=" * 80,
        "",
    ]
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("\n".join(header))

    print(f"📂 输出：{out_root}")
    print(f"📄 日志：{log_file}\n")

    if args.file is not None:
        res = process_single_image(args.file, None, out_root, 0)
        print(f"✅ {args.file} | {res['status']}")
        return

    input_dir = os.path.abspath(args.input)
    exts = (".jpg", ".jpeg", ".png", ".tif", ".bmp")
    img_files = [f for f in os.listdir(input_dir) if f.lower().endswith(exts)]

    try:
        img_files.sort(key=natural_sort_key)
    except:
        img_files.sort()

    total = len(img_files)
    if total == 0:
        print("❌ 无图片")
        return

    widths = [cv2.imread(os.path.join(input_dir, f)).shape[1] for f in img_files]
    threshold = np.mean(widths) * 0.8
    print(f"📁 扫描：{input_dir}")
    print(f"✅ 共 {total} 张 | 开始处理...\n")

    results = []
    with ThreadPoolExecutor(os.cpu_count()) as executor:
        task_map = {executor.submit(process_single_image, f, input_dir, out_root, threshold): f for f in img_files}
        for i, future in enumerate(as_completed(task_map), 1):
            res = future.result()
            results.append(res)
            status_icon = "✅" if res["status"] == "success" else "⚠️" if res["status"] == "no_detect" else "❌"
            print(f"[{i}/{total}] {status_icon} {res['file']} | {res['status']}")

    success_list = [r for r in results if r["status"] == "success"]
    no_detect_list = [r for r in results if r["status"] == "no_detect"]
    error_list = [r for r in results if r["status"] == "error"]

    # ===================== 日志顺序：失败 → 未识别 → 成功 =====================
    log_blocks = [
        "\n" + "=" * 30 + " a. 异常失败文件列表 " + "=" * 30,
        *[f"❌ {r['file']} | 原因：{r['reason']}" for r in error_list],
        "\n" + "=" * 30 + " b. 未识别文件列表 " + "=" * 30,
        *[f"⚠ {r['file']} | 原图输出" for r in no_detect_list],
        "\n" + "=" * 30 + " c. 成功识别文件列表 " + "=" * 30,
        *[f"✔ {r['file']}" for r in success_list],
        "\n" + "=" * 30 + " d. 详细处理日志 " + "=" * 30,
    ]

    for r in results:
        if r["status"] == "success":
            log_blocks.append(
                f"[成功] {r['file']} | {r['size']} | 输出：{','.join(r['outputs'])} | 耗时：{r['time']:.2f}s"
            )
        elif r["status"] == "no_detect":
            log_blocks.append(f"[未识别] {r['file']} | {r['size']} | 已保存原图")
        else:
            log_blocks.append(f"[失败] {r['file']} | {r['reason']}")

    write_log(log_file, "\n".join(log_blocks))
    print(f"\n🎉 完成！失败：{len(error_list)} | 未识别：{len(no_detect_list)} | 成功：{len(success_list)}")


if __name__ == "__main__":
    main()
