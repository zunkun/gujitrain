# guji-train（古籍双模型检测）

古籍 OCR 前处理项目：一个工程内包含两个相互隔离的 YOLOv8 检测任务，**数据、标注、权重、训练产物完全隔离，互不融合**，只共用一套训练/推理代码。

| 任务        | 数据 / 权重                                    | 类别                     | 用途                           |
| ----------- | ---------------------------------------------- | ------------------------ | ------------------------------ |
| bookcontent | `data/bookcontent/` → `weights/bookcontent.pt` | 2 类 `bookcontent`、`fullcontent` | 文本主体区域 / 整页内容区域检测（左右页切图） |
| gujiline    | `data/gujiline/` → `weights/gujiline.pt`       | 2 类 `gujiline`、`small` | 栏线 / 小字检测                |

数据来源：bookcontent 为《容齋隨筆》（Harvard drs 54114467，makesense.ai 标注）；gujiline 为原 gujilinetrain 项目迁移（82 图 / 820 框）。

## 目录结构

```
gujitrain/
├─ data/
│  ├─ bookcontent/              # 训练集：images/ labels/ + bookcontent.yaml (nc=1)
│  └─ gujiline/                 # 训练集：images/ labels/ + gujiline.yaml (nc=2)
├─ weights/
│  ├─ yolov8n.pt                # 预训练权重（训练起点）
│  └─ <task>.pt / *_old_backup.pt   # 训练后生成；训练前旧权重自动备份（均不入 git）
├─ runs/
│  ├─ bookcontent/  gujiline/   # 各任务训练产物（不入 git）
├─ test/
│  ├─ predict_bookcontent.py    # bookcontent 批量切图（文本区域，左右页切分）
│  ├─ predict_gujiline.py       # gujiline 批量切图（栏线/小字）
│  ├─ debug_boxes.py            # 调试检测（--task 选模型，输出全部框、按类配色）
│  ├─ viz_labels.py             # 标注自检：把标签画回原图 + 数据集体检（不加载模型）
│  ├─ convert_anylabeling.py    # X-AnyLabeling/LabelMe json → YOLO txt
│  └─ detector*.py / test.py / report.py / utils.py   # gujiline 检测与报告
├─ train_bookcontent.py         # bookcontent 训练入口
├─ train_gujiline.py            # gujiline 训练入口
├─ train_common.py              # 公共训练逻辑
└─ requirements.txt
```

## DATA 数据

1. 百度网盘

通过网盘分享的文件：古籍训练数据
链接: https://pan.baidu.com/s/1UMAb6CbRwkgnwjNCNjXAkQ?pwd=2gm8 提取码: 2gm8

2. 阿里云盘

## 快速开始

环境：conda `py310`（torch 2.5.1+cu121 GPU、ultralytics、opencv-python 等，见 requirements.txt）。

```bash
conda activate py310

# 训练：自动判断——已有该任务权重则增量续训，没有则从 yolov8n.pt 从头训练
python train_bookcontent.py
python train_gujiline.py
python train_bookcontent.py --fresh    # 强制从头重训（旧权重先自动备份）

# 批量切图（脚本与任务一一对应，模型固定写在脚本头部）
python test/predict_bookcontent.py
python test/predict_gujiline.py --books book1 book2

# 标注自检：把标签画回原图 + 数据集体检（不加载模型，不占 GPU，秒级）
python test/viz_labels.py --task bookcontent --books b1 --limit 3
python test/viz_labels.py --task bookcontent --no-image   # 只体检，不输出图

# 调试检测（输出全部检测框、按类配色，--task 选模型）
python test/debug_boxes.py --task bookcontent --books b1 --limit 3
python test/debug_boxes.py gujiline --books book2          # 位置参数写法也可

# 新标注：X-AnyLabeling 导出的 json → YOLO txt（类别顺序自动取自 yaml）
python test/convert_anylabeling.py --src data/bookcontent/b1 --task bookcontent --dry-run
python test/convert_anylabeling.py --src data/bookcontent/b1 --task bookcontent --copy-images
```

## 数据与标注

- 类别：bookcontent 两类 —— `bookcontent`（编号 0）框**文字主体区域**（不框边框/印章/批注/空白/污渍），双页左右各一框、单页一框；`fullcontent`（编号 1）框**整页内容区域**，跨页只给一个大框。gujiline 标注**栏线**与**小字**两类
- **txt 里不写类名**：每行 `类别编号 cx cy w h`，编号顺序由 yaml 的 `names` 决定（`names` 第 0 个 = 编号 0）。改类别顺序会让已有标注全部错位，禁止随意调整
- 导出 **YOLO 格式 .txt**，标签与图片同名；文件名必须唯一（如 `book1_001.jpg`）
- 图片平铺放入 `data/<任务>/images/`，标签放入 `labels/`
- 每个任务一个独立 yaml，**path 必须写绝对路径**（ultralytics 8.4 对相对路径的解析受运行目录影响）：

```yaml
# data/bookcontent/bookcontent.yaml（gujiline 同理：path 指向 data/gujiline，nc=2）
path: D:/workspace/gujitrain/data/bookcontent
train: images
val: images

# 类别编号 = names 列表的位置，顺序不可随意调整
nc: 2
names: ["bookcontent", "fullcontent"]
```

## 训练说明

- **无需手动选续训/重训**：脚本按权重文件自动判断，不存在漏加参数的问题；`--fresh` 是唯一的重训开关
- **中断安全**：训练期间只写 `runs/<任务>/trainN/weights/`；结束后才以「临时文件 + 原子替换」更新 `weights/<任务>.pt`，中途断掉不会写坏最终权重
- **产物隔离**：各任务产物只进各自的 `runs/<任务>/`

| 参数                      | bookcontent                       | gujiline                                  |
| ------------------------- | --------------------------------- | ----------------------------------------- |
| epochs / batch / patience | 50 / 4 / 10                       | 200 / 8 / 30                              |
| lr0                       | 0.0003                            | 0.001                                     |
| imgsz / optimizer         | 640 / AdamW（lrf=0.01，两者相同） | 同左                                      |
| 增强                      | mosaic=0, augment=False           | mosaic/mixup/copy_paste=0, hsv/scale 微调 |
| 设备                      | GPU 自动判断（无 GPU 回退 CPU）   | 同左                                      |

公共参数（amp=False、workers、weight_decay、warmup、cache=disk 等）在 `train_common.py` 统一维护。

## 其他说明

- 切图脚本：输入 `images/<书名>/img0/`，输出 `output/<书名>/`；单双页按平均宽度 ×0.8 自动判断，左右页按检测框中心分组
- gujiline 检测报告脚本（detector\*.py / test.py）加载 `weights/gujiline.pt`，目标类别 `TARGET_CLASS = "gujiline"`
- git 约定：训练集图片/标注、runs/、\*.pt 均不入库（见 .gitignore），只跟踪代码与 yaml 配置
