# -*- coding: utf-8 -*-
"""
bookcontent 任务训练入口（2 类：bookcontent=0 文字主体区域 / fullcontent=1 整页内容区域）。

用法:
    python train_bookcontent.py           # 自动：有权重就续训，没有就从 yolov8n.pt 训练
    python train_bookcontent.py --fresh   # 强制忽略已有权重，从头重新训练
"""

from train_common import parse_common_args, run_training

TASK = dict(
    task_name="bookcontent",
    data="data/bookcontent/bookcontent.yaml",  # nc=2, names=["bookcontent", "fullcontent"]
    final_weights="weights/bookcontent.pt",
    backup_weights="weights/bookcontent_old_backup.pt",
    project="runs/bookcontent",  # 训练产物输出目录
    # ---- 从头训练配置（追求精度）----
    epochs=80,
    batch=12,
    patience=30,  # 从头训练收敛慢，加大耐心防误停
    lr0=0.001,  # 从头训练用正常学习率（0.0003 是微调用的，太小学不动）
    init_weight="weights/yolov8n.pt",
    fresh=False,
    # 从头训练需要数据增强扩样本，close_mosaic 默认最后 10 轮自动关 mosaic 收敛
    train_kwargs=dict(
        mosaic=0.5,
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
