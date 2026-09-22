# -*- coding: utf-8 -*-
"""
gujiline 任务训练入口（栏线检测，2 类：gujiline / small），权重文件名为 gujiline.pt。

用法:
    python train_gujiline.py           # 自动：有权重就续训，没有就从 yolov8n.pt 训练
    python train_gujiline.py --fresh   # 强制忽略已有权重，从头重新训练
"""
from train_common import parse_common_args, run_training

TASK = dict(
    task_name="gujiline",
    data="data/gujiline/gujiline.yaml",         # nc=2, names=["gujiline", "small"]
    final_weights="weights/gujiline.pt",
    backup_weights="weights/gujiline_old_backup.pt",
    project="runs/gujiline",
    epochs=200,
    batch=8,
    patience=30,
    lr0=0.001,
    init_weight="weights/yolov8n.pt",
    fresh=False,
    # gujiline 原有增强设置：关闭破坏空间位置的增强
    train_kwargs=dict(
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        scale=0.5,
        weight_decay=0.0005,
        warmup_epochs=3,
    ),
)

if __name__ == "__main__":
    args = parse_common_args(TASK["task_name"])
    TASK["fresh"] = args.fresh
    run_training(TASK)
