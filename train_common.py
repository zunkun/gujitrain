# -*- coding: utf-8 -*-
"""
训练公共逻辑：train_bookcontent.py / train_gujiline.py 共用。
两个任务的数据、权重、训练产物完全隔离，互不融合。
"""

import os
import shutil

# 全局设置 Ultralytics 日志级别，大幅减少刷屏输出
os.environ["ULTRALYTICS_LOG_LEVEL"] = "WARNING"

import argparse

import torch
from ultralytics import YOLO

IMGSZ = 640
WORKERS = 2
AMP = False

# 项目根目录：data/project 等相对路径一律锚定到这里，转成绝对路径传给 ultralytics
# （ultralytics 8.4 对相对路径的解析受运行目录影响，绝对路径最稳）
ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_common_args(task_name):
    parser = argparse.ArgumentParser(description=f"古籍 YOLO 训练入口：{task_name}")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="忽略已有权重，强制从 weights/yolov8n.pt 从头重新训练（默认自动续训）",
    )
    return parser.parse_args()


def run_training(cfg):
    """cfg 字段见 train_bookcontent.py / train_gujiline.py 中的 TASK 配置。"""
    # 自动判断设备
    if torch.cuda.is_available():
        device = 0
        batch = cfg["batch"]
        workers = WORKERS
        print("✅ GPU 已就绪")
    else:
        device = "cpu"
        batch = 2
        workers = 0
        print("⚠️ CUDA不可用，切换为CPU模式")

    use_init = False
    # 路径统一锚定到项目根目录
    final_w = (
        cfg["final_weights"]
        if os.path.isabs(cfg["final_weights"])
        else os.path.join(ROOT, cfg["final_weights"])
    )
    backup_w = (
        cfg["backup_weights"]
        if os.path.isabs(cfg["backup_weights"])
        else os.path.join(ROOT, cfg["backup_weights"])
    )
    init_w = (
        cfg["init_weight"]
        if os.path.isabs(cfg["init_weight"])
        else os.path.join(ROOT, cfg["init_weight"])
    )

    # 自动判断：已有最终权重 -> 增量续训；没有 -> 从 yolov8n.pt 从头训练
    # 只有显式 --fresh 才忽略已有权重强制从头训练（旧权重仍会先备份）
    if cfg["fresh"]:
        use_init = True
        print("🔄 --fresh 模式：忽略已有权重，从头重新训练")
    elif os.path.exists(final_w):
        print(
            f"🔁 检测到已有权重 {cfg['final_weights']}，自动增量续训（从头重训请加 --fresh）"
        )
    else:
        use_init = True
        print(f"🆕 未发现已有权重，从 {cfg['init_weight']} 开始训练")

    # 训练前备份原有最优权重（无论续训还是 --fresh 重训，都先备份）
    if os.path.exists(final_w):
        os.makedirs(os.path.dirname(backup_w), exist_ok=True)
        shutil.copy(final_w, backup_w)
        print(f"📦 旧权重已备份至: {cfg['backup_weights']}")

    # 权重加载分支
    if use_init:
        load_weight = init_w
        print(f"🚀 加载预训练权重: {load_weight}")
    else:
        load_weight = final_w
        print(f"🔁 增量续训加载: {load_weight}")

    model = YOLO(load_weight)

    # data 与 project 转绝对路径，避免 ultralytics 8.4 相对路径解析歧义
    data_path = (
        cfg["data"] if os.path.isabs(cfg["data"]) else os.path.join(ROOT, cfg["data"])
    )
    project_path = (
        cfg["project"]
        if os.path.isabs(cfg["project"])
        else os.path.join(ROOT, cfg["project"])
    )

    try:
        results = model.train(
            data=data_path,
            epochs=cfg["epochs"],
            imgsz=IMGSZ,
            batch=batch,
            device=device,
            workers=workers,
            amp=AMP,
            patience=cfg["patience"],
            project=project_path,  # 训练产物按任务隔离（绝对路径，防止嵌套到 runs/detect 下）
            plots=False,
            verbose=False,
            save_json=False,
            save_txt=False,
            show_labels=False,
            show_conf=False,
            show_boxes=False,
            cache="disk",  # 磁盘缓存预处理图片，提升 IO 效率
            optimizer="AdamW",
            lr0=cfg["lr0"],
            lrf=0.01,
            **cfg["train_kwargs"],
        )
    except KeyboardInterrupt:
        # Ctrl+C 优雅中断：进度已在 last.pt，不复制 best、不更新最终权重
        print("\n" + "=" * 60)
        print("⚠️  已收到 Ctrl+C，训练中断")
        trainer = getattr(model, "trainer", None)
        save_dir = getattr(trainer, "save_dir", None)
        if save_dir:
            last_pt = os.path.join(str(save_dir), "weights", "last.pt")
            if os.path.exists(last_pt):
                print(f"💾 断点进度已保存: {last_pt}")
                print("✅ 精确续训（从断点的 epoch 继续）请执行:")
                print(f'   yolo resume model="{last_pt}"')
        print("ℹ️  weights/ 下的已有权重不受影响；直接重跑训练脚本则从现有权重增量续训")
        print("=" * 60)
        raise SystemExit(130)

    # 训练结束复制最新 best.pt 覆盖目标权重
    best_src = os.path.join(results.save_dir, "weights", "best.pt")
    if os.path.exists(best_src):
        os.makedirs(os.path.dirname(cfg["final_weights"]), exist_ok=True)
        # 先写临时文件再原子替换，中断/断电也不会写坏目标权重
        tmp_dst = cfg["final_weights"] + ".tmp"
        shutil.copy(best_src, tmp_dst)
        os.replace(tmp_dst, cfg["final_weights"])
        print(f"\n✅ 权重更新完成：\n{best_src}\n→ {cfg['final_weights']}")
    else:
        print("\n❌ 未找到本轮训练生成的best.pt")
