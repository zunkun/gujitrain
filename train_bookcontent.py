# -*- coding: utf-8 -*-
"""
bookcontent 任务训练入口（文本区域检测，1 类：bookcontent）。

用法:
    python train_bookcontent.py           # 自动：有权重就续训，没有就从 yolov8n.pt 训练
    python train_bookcontent.py --fresh   # 强制忽略已有权重，从头重新训练
"""
from train_common import parse_common_args, run_training

TASK = dict(
    task_name="bookcontent",
    data="data/bookcontent/bookcontent.yaml",   # nc=1, names=["bookcontent"]
    final_weights="weights/bookcontent.pt",
    backup_weights="weights/bookcontent_old_backup.pt",
    project="runs/bookcontent",                 # 训练产物输出目录
    epochs=50,
    batch=4,
    patience=10,
    lr0=0.0003,
    init_weight="weights/yolov8n.pt",
    fresh=False,
    # bookcontent 原有增强设置
    train_kwargs=dict(mosaic=0, augment=False),
)

if __name__ == "__main__":
    args = parse_common_args(TASK["task_name"])
    TASK["fresh"] = args.fresh
    run_training(TASK)
