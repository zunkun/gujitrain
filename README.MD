# guji-train（古籍双模型检测）

古籍 OCR 前处理项目：一个工程内包含两个相互隔离的 YOLOv8 检测任务，数据、标注、权重、训练产物互不融合。

| 任务        | 目录/权重                                      | 类别                     | 用途                           |
| ----------- | ---------------------------------------------- | ------------------------ | ------------------------------ |
| bookcontent | `data/bookcontent/` → `weights/bookcontent.pt` | 1 类 `bookcontent`       | 文本主体区域检测（左右页切图） |
| gujiline    | `data/gujiline/` → `weights/gujiline.pt`       | 2 类 `gujiline`、`small` | 栏线 / 小字检测                |

数据集来源：

- bookcontent：`Harvard drs 54114467 容齋隨筆 v.1.pdf`（makesense.ai 标注）
- gujiline：原 gujilinetrain 项目迁移（82 张图 / 820 框）

## 目录结构

```
gujitrain/
├─ data/
│  ├─ bookcontent/        # bookcontent 训练集（images / labels / bookcontent.yaml）
│  └─ gujiline/           # gujiline 训练集（images / labels / gujiline.yaml）
├─ weights/
│  └─ yolov8n.pt          # 预训练权重（从头训练的起点）
│  # 训练后生成 bookcontent.pt / gujiline.pt（不入 git）
├─ runs/
│  ├─ bookcontent/        # bookcontent 训练产物（不入 git）
│  └─ gujiline/           # gujiline 训练产物（不入 git）
├─ test/                  # 所有测试 / 推理脚本（按任务命名）
│  ├─ predict_bookcontent.py   # bookcontent 批量切图（文本区域，左右页切分）
│  ├─ predict_gujiline.py      # gujiline 批量切图（栏线/小字）
│  ├─ debug_boxes.py           # 调试切图（--task 选模型，输出全部检测框）
│  └─ detector.py / detector2.py / test.py / report.py / utils.py   # gujiline 检测与报告
├─ train_bookcontent.py   # bookcontent 训练入口
├─ train_gujiline.py      # gujiline 训练入口
├─ train_common.py        # 两个入口共用的训练逻辑
└─ 说明.md                # 完整使用指南
```

## 常用命令

```shell
# 训练（自动判断：有权重则增量续训，无权重从 yolov8n.pt 训练；--fresh 强制从头）
python train_bookcontent.py
python train_gujiline.py

# 批量切图（脚本与任务一一对应）
python test/predict_bookcontent.py
python test/predict_gujiline.py --books book1 book2

# 调试模式（输出全部检测框，--task 选模型）
python test/debug_boxes.py gujiline --books book2
```

## 环境

conda `py310`：torch 2.5.1+cu121（GPU）、ultralytics、opencv-python、numpy、pillow、tqdm、matplotlib（见 requirements.txt）。
