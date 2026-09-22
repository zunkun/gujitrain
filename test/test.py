import os
import re
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime

# ========== 配置区 ==========
MODEL_PATH = "weights/gujiline.pt"
# INPUT_DIR = "train/images"            # 输入图片文件夹
# OUTPUT_DIR = "imagestest"             # 推理结果输出目录（报告也在此目录）
INPUT_DIR = r"D:\xz\xqc\crop"
OUTPUT_DIR = r"D:\xz\xqc\detect"
CONF_THRES = 0.25

# 目标类别名称（只显示这个类别的框）
TARGET_CLASS = "gujiline"

# 宽度过滤阈值
WIDTH_RATIO_MIN = 0.6
WIDTH_RATIO_MAX = 1.6
MIN_HEIGHT_MARGIN = 160

# 报告每页显示图片数量
IMAGES_PER_PAGE = 40
# ============================

def natural_sort_key(s):
    """
    自然排序算法：将字符串中的数字部分转为整数进行排序
    例如：img1, img2, img10 -> img1, img2, img10 (而不是 img1, img10, img2)
    """
    return [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', s)]

def generate_colors(n=20):
    """生成 n 种高对比颜色 (BGR)"""
    colors = []
    for i in range(n):
        hue = i / n
        rgb = cv2.cvtColor(np.uint8([[[hue*180, 230, 230]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(map(int, rgb)))
    return colors

def paginate(items, per_page=IMAGES_PER_PAGE):
    """将列表分页，返回生成器"""
    for i in range(0, len(items), per_page):
        yield items[i:i+per_page]

def generate_report(results_data, output_dir, report_title="检测报告"):
    """
    生成带分页、灯箱预览、上一张/下一张切换的 HTML 报告
    results_data: 列表，每个元素为 dict { 'img_name': str, 'orig_count': int, 'filtered_count': int, 'img_path': str }
    """
    # 按过滤后数量归类
    groups = {}
    for item in results_data:
        count = item['filtered_count']
        groups.setdefault(count, []).append(item)

    # 对每个组内的图片按自然数排序
    for count in groups:
        groups[count].sort(key=lambda x: natural_sort_key(x['img_name']))

    # 总体数据也按自然数排序
    results_data.sort(key=lambda x: natural_sort_key(x['img_name']))

    # 图片引用路径（相对于 HTML 所在的 output_dir）
    for item in results_data:
        item['rel_path'] = f"images/{os.path.basename(item['img_path'])}"

    # 构建 HTML
    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report_title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            display: flex;
            height: 100vh;
            background: #f5f5f5;
        }}
        /* 左侧菜单 */
        .menu {{
            width: 250px;
            background: #2c3e50;
            color: white;
            padding: 20px 0;
            overflow-y: auto;
            flex-shrink: 0;
            height: 100vh;
            position: sticky;
            top: 0;
        }}
        .menu h2 {{
            padding: 0 20px 15px;
            border-bottom: 1px solid #34495e;
            font-weight: 300;
            font-size: 1.3em;
        }}
        .menu-item {{
            cursor: pointer;
            padding: 10px 20px;
            border-bottom: 1px solid #34495e;
            transition: background 0.2s;
        }}
        .menu-item:hover {{
            background: #34495e;
        }}
        .menu-item.active {{
            background: #1abc9c;
        }}
        .menu-item .count-badge {{
            float: right;
            background: #1abc9c;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 0.8em;
        }}
        .menu-item.folder .count-badge {{
            background: #e67e22;
        }}
        /* 右侧内容 */
        .content {{
            flex: 1;
            padding: 20px 30px;
            overflow-y: auto;
            height: 100vh;
        }}
        .section {{
            display: none;
        }}
        .section.active {{
            display: block;
        }}
        .section h1 {{
            margin-bottom: 20px;
            color: #2c3e50;
            border-bottom: 2px solid #1abc9c;
            padding-bottom: 10px;
        }}
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .page {{
            display: none;
        }}
        .page.active {{
            display: grid;
        }}
        .image-card {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow: hidden;
            transition: transform 0.2s;
            cursor: pointer;
        }}
        .image-card:hover {{
            transform: scale(1.02);
            box-shadow: 0 4px 16px rgba(0,0,0,0.2);
        }}
        .image-card img {{
            width: 100%;
            height: 200px;
            object-fit: contain;
            background: #ecf0f1;
            display: block;
        }}
        .image-card .info {{
            padding: 10px 12px;
            font-size: 0.9em;
            color: #333;
        }}
        .image-card .info .filename {{
            font-weight: bold;
            margin-bottom: 4px;
            word-break: break-all;
        }}
        .image-card .info .counts {{
            color: #7f8c8d;
        }}
        .summary-stats {{
            display: flex;
            gap: 30px;
            flex-wrap: wrap;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .summary-stats .stat {{
            display: flex;
            flex-direction: column;
        }}
        .summary-stats .stat .number {{
            font-size: 2em;
            font-weight: bold;
            color: #1abc9c;
        }}
        .summary-stats .stat .label {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        /* 分页导航 */
        .pagination {{
            display: flex;
            gap: 6px;
            flex-wrap: wrap;
            margin: 15px 0 20px;
            padding: 10px 0;
            align-items: center;
        }}
        .pagination button {{
            padding: 6px 14px;
            border: 1px solid #ddd;
            background: white;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9em;
            transition: all 0.2s;
        }}
        .pagination button:hover {{
            background: #1abc9c;
            color: white;
            border-color: #1abc9c;
        }}
        .pagination button.active {{
            background: #1abc9c;
            color: white;
            border-color: #1abc9c;
        }}
        .pagination button:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
        }}
        /* ===== 灯箱（模态框）增强版 ===== */
        #lightbox {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.85);
            z-index: 9999;
            justify-content: center;
            align-items: center;
            cursor: pointer;
            flex-direction: column;
        }}
        #lightbox img {{
            max-width: 80%;
            max-height: 80%;
            object-fit: contain;
            border-radius: 8px;
            box-shadow: 0 0 40px rgba(0,0,0,0.6);
            cursor: default;
            transition: opacity 0.2s;
        }}
        #lightbox .close-btn {{
            position: absolute;
            top: 30px;
            right: 40px;
            font-size: 40px;
            color: white;
            font-weight: bold;
            cursor: pointer;
            transition: 0.3s;
            background: rgba(0,0,0,0.3);
            width: 60px;
            height: 60px;
            border-radius: 50%;
            display: flex;
            justify-content: center;
            align-items: center;
            line-height: 1;
        }}
        #lightbox .close-btn:hover {{
            background: rgba(255,0,0,0.6);
            transform: rotate(90deg);
        }}
        /* 导航按钮 */
        .lightbox-nav {{
            position: absolute;
            top: 50%;
            transform: translateY(-50%);
            color: white;
            font-size: 50px;
            background: rgba(0,0,0,0.4);
            padding: 10px 20px;
            border-radius: 50%;
            cursor: pointer;
            user-select: none;
            transition: 0.2s;
            line-height: 1;
        }}
        .lightbox-nav:hover {{
            background: rgba(255,255,255,0.3);
        }}
        .lightbox-nav.prev {{
            left: 30px;
        }}
        .lightbox-nav.next {{
            right: 30px;
        }}
        /* 图片计数 */
        .lightbox-counter {{
            position: absolute;
            bottom: 40px;
            left: 50%;
            transform: translateX(-50%);
            color: white;
            background: rgba(0,0,0,0.6);
            padding: 8px 20px;
            border-radius: 30px;
            font-size: 16px;
            letter-spacing: 1px;
        }}
    </style>
</head>
<body>
    <div class="menu">
        <h2>📊 检测报告</h2>
        <div class="menu-item folder active" data-target="summary">
            📋 总报告 <span class="count-badge">{len(results_data)}</span>
        </div>
"""

    # 生成分组菜单
    for count, items in sorted(groups.items(), reverse=True):
        html_content += f"""
        <div class="menu-item folder" data-target="group_{count}">
            🏷️ {count} 个目标 <span class="count-badge">{len(items)}</span>
        </div>
        """

    html_content += """
    </div>

    <div class="content">
        <!-- 灯箱（增强版） -->
        <div id="lightbox" onclick="closeLightbox()">
            <span class="close-btn" onclick="closeLightbox(); event.stopPropagation();">&times;</span>
            <div class="lightbox-nav prev" onclick="changeImage(-1); event.stopPropagation();">‹</div>
            <img id="lightbox-img" src="" alt="预览">
            <div class="lightbox-nav next" onclick="changeImage(1); event.stopPropagation();">›</div>
            <div class="lightbox-counter" id="lightbox-counter">1 / 1</div>
        </div>

        <!-- 总报告 -->
        <div id="section-summary" class="section active">
            <h1>📈 总体统计</h1>
            <div class="summary-stats">
                <div class="stat"><span class="number">{}</span><span class="label">总图片数</span></div>
                <div class="stat"><span class="number">{}</span><span class="label">总检测框数（过滤后）</span></div>
                <div class="stat"><span class="number">{}</span><span class="label">不同目标数量分类</span></div>
            </div>
            <div style="background:white; padding:15px; border-radius:8px; margin-bottom:20px;">
                <h3>📂 分类详情</h3>
                <ul>
        """.format(len(results_data), sum(len(items) for items in groups.values()), len(groups))

    for count, items in sorted(groups.items(), reverse=True):
        html_content += f"<li><strong>{count} 个目标：</strong>{len(items)} 张图片</li>"

    html_content += """
                </ul>
            </div>
            <h2 style="margin-top:10px;">📷 全部图片预览</h2>
            <div class="pagination" id="pagination-summary"></div>
    """

    # ---- 总报告分页 ----
    summary_pages = list(paginate(results_data))
    for page_idx, page_items in enumerate(summary_pages):
        active_class = "active" if page_idx == 0 else ""
        html_content += f'<div class="page {active_class}" data-section="summary" data-page="{page_idx}"><div class="image-grid">'
        for item in page_items:
            html_content += f"""
            <div class="image-card" data-src="{item['rel_path']}">
                <img src="{item['rel_path']}" alt="{item['img_name']}" loading="lazy">
                <div class="info">
                    <div class="filename">{item['img_name']}</div>
                    <div class="counts">原始框 {item['orig_count']} → 过滤后 {item['filtered_count']}</div>
                </div>
            </div>
            """
        html_content += '</div></div>'

    html_content += '</div>'  # 结束 section-summary

    # ---- 为每个分组生成内容区域 ----
    for count, items in sorted(groups.items(), reverse=True):
        html_content += f"""
        <div id="section-group_{count}" class="section">
            <h1>🏷️ {count} 个目标（共 {len(items)} 张）</h1>
            <div class="pagination" id="pagination-group_{count}"></div>
        """
        group_pages = list(paginate(items))
        for page_idx, page_items in enumerate(group_pages):
            active_class = "active" if page_idx == 0 else ""
            html_content += f'<div class="page {active_class}" data-section="group_{count}" data-page="{page_idx}"><div class="image-grid">'
            for item in page_items:
                html_content += f"""
                <div class="image-card" data-src="{item['rel_path']}">
                    <img src="{item['rel_path']}" alt="{item['img_name']}" loading="lazy">
                    <div class="info">
                        <div class="filename">{item['img_name']}</div>
                        <div class="counts">原始框 {item['orig_count']} → 过滤后 {item['filtered_count']}</div>
                    </div>
                </div>
                """
            html_content += '</div></div>'
        html_content += '</div>'

    # ---- JavaScript ----
    html_content += """
    </div>  <!-- 结束 content -->

    <script>
        // ========== 灯箱增强功能 ==========
        let lightboxImages = [];      // 当前打开的图片列表 (src数组)
        let currentIndex = 0;
        let lightboxOpen = false;

        // 打开灯箱，传入图片地址数组和当前索引
        function openLightbox(images, index) {
            if (!images || images.length === 0) return;
            lightboxImages = images;
            currentIndex = index;
            lightboxOpen = true;
            const lightbox = document.getElementById('lightbox');
            const img = document.getElementById('lightbox-img');
            img.src = lightboxImages[currentIndex];
            updateCounter();
            lightbox.style.display = 'flex';
            document.body.style.overflow = 'hidden';
            // 监听键盘事件
            document.addEventListener('keydown', handleKeydown);
        }

        function closeLightbox() {
            document.getElementById('lightbox').style.display = 'none';
            document.body.style.overflow = 'auto';
            lightboxOpen = false;
            document.removeEventListener('keydown', handleKeydown);
        }

        function changeImage(delta) {
            if (!lightboxImages || lightboxImages.length === 0) return;
            let newIndex = currentIndex + delta;
            if (newIndex < 0) newIndex = lightboxImages.length - 1;
            if (newIndex >= lightboxImages.length) newIndex = 0;
            currentIndex = newIndex;
            const img = document.getElementById('lightbox-img');
            img.src = lightboxImages[currentIndex];
            updateCounter();
        }

        function updateCounter() {
            const counter = document.getElementById('lightbox-counter');
            if (counter) {
                counter.textContent = (currentIndex + 1) + ' / ' + lightboxImages.length;
            }
        }

        function handleKeydown(e) {
            if (!lightboxOpen) return;
            if (e.key === 'ArrowLeft') {
                e.preventDefault();
                changeImage(-1);
            } else if (e.key === 'ArrowRight') {
                e.preventDefault();
                changeImage(1);
            } else if (e.key === 'Escape') {
                closeLightbox();
            }
        }

        // ========== 图片卡片点击事件（委托） ==========
        document.addEventListener('click', function(e) {
            const card = e.target.closest('.image-card');
            if (!card) return;
            // 获取该卡片所属的 section
            const section = card.closest('.section');
            if (!section) return;
            // 收集该 section 下所有卡片的数据源（data-src）
            const allCards = section.querySelectorAll('.image-card');
            const srcList = [];
            allCards.forEach(c => {
                const src = c.dataset.src;
                if (src) srcList.push(src);
            });
            // 找到当前卡片的索引
            const currentSrc = card.dataset.src;
            const idx = srcList.indexOf(currentSrc);
            if (idx === -1) return;
            // 打开灯箱
            openLightbox(srcList, idx);
            // 阻止点击冒泡，避免触发其他事件
            e.stopPropagation();
        });

        // 点击灯箱背景关闭（已写在#lightbox的onclick中，但需阻止冒泡）
        // 但 #lightbox 本身有 onclick="closeLightbox()"，内部元素已阻止冒泡，所以没问题。

        // ========== 菜单切换 ==========
        const menuItems = document.querySelectorAll('.menu-item');
        const sections = document.querySelectorAll('.section');

        menuItems.forEach(item => {
            item.addEventListener('click', function(e) {
                const target = this.dataset.target;
                if (!target) return;
                // 切换内容
                sections.forEach(sec => sec.classList.remove('active'));
                const targetSection = document.getElementById('section-' + target);
                if (targetSection) targetSection.classList.add('active');

                // 高亮菜单项
                menuItems.forEach(it => it.classList.remove('active'));
                this.classList.add('active');

                // 切换后，确保当前激活的部分显示第一页，并刷新分页按钮状态
                const sectionId = target;
                const container = document.getElementById('section-' + sectionId);
                if (container) {
                    const pages = container.querySelectorAll('.page');
                    pages.forEach((p, i) => {
                        p.classList.toggle('active', i === 0);
                    });
                    const btns = container.querySelectorAll('.pagination button');
                    btns.forEach((b, i) => {
                        b.classList.toggle('active', i === 0);
                    });
                }
            });
        });

        // ========== 分页功能 ==========
        function initPagination() {
            const sections = document.querySelectorAll('.section');
            sections.forEach(section => {
                const sectionId = section.id.replace('section-', '');
                const pages = section.querySelectorAll('.page');
                const paginationDiv = section.querySelector('.pagination');
                if (!paginationDiv || pages.length <= 1) {
                    if (paginationDiv && pages.length <= 1) paginationDiv.style.display = 'none';
                    return;
                }
                paginationDiv.innerHTML = '';
                // 上一页
                const prevBtn = document.createElement('button');
                prevBtn.textContent = '‹ 上一页';
                prevBtn.disabled = true;
                prevBtn.dataset.section = sectionId;
                prevBtn.dataset.dir = 'prev';
                paginationDiv.appendChild(prevBtn);

                // 页码按钮
                for (let i = 0; i < pages.length; i++) {
                    const btn = document.createElement('button');
                    btn.textContent = i + 1;
                    btn.dataset.page = i;
                    btn.dataset.section = sectionId;
                    if (i === 0) btn.classList.add('active');
                    paginationDiv.appendChild(btn);
                }

                // 下一页
                const nextBtn = document.createElement('button');
                nextBtn.textContent = '下一页 ›';
                nextBtn.dataset.section = sectionId;
                nextBtn.dataset.dir = 'next';
                if (pages.length <= 1) nextBtn.disabled = true;
                paginationDiv.appendChild(nextBtn);

                // 绑定事件
                paginationDiv.addEventListener('click', function(e) {
                    const target = e.target.closest('button');
                    if (!target) return;
                    if (target.disabled) return;

                    const sectionId = target.dataset.section;
                    const container = document.getElementById('section-' + sectionId);
                    if (!container) return;
                    const pages = container.querySelectorAll('.page');
                    const currentPage = Array.from(pages).findIndex(p => p.classList.contains('active'));
                    let newPage = currentPage;

                    if (target.dataset.page !== undefined) {
                        newPage = parseInt(target.dataset.page);
                    } else if (target.dataset.dir === 'prev') {
                        newPage = currentPage - 1;
                    } else if (target.dataset.dir === 'next') {
                        newPage = currentPage + 1;
                    }
                    if (newPage < 0 || newPage >= pages.length) return;

                    // 切换页面
                    pages.forEach((p, i) => p.classList.toggle('active', i === newPage));
                    const btns = container.querySelectorAll('.pagination button');
                    btns.forEach((b, i) => {
                        if (i > 0 && i <= pages.length) { // 跳过前后导航按钮
                            b.classList.toggle('active', (i-1) === newPage);
                        }
                    });
                    // 更新前后按钮状态
                    const prevBtn = container.querySelector('.pagination button[data-dir="prev"]');
                    const nextBtn = container.querySelector('.pagination button[data-dir="next"]');
                    if (prevBtn) prevBtn.disabled = (newPage === 0);
                    if (nextBtn) nextBtn.disabled = (newPage === pages.length - 1);
                });
            });
        }

        // 初始化分页
        document.addEventListener('DOMContentLoaded', function() {
            initPagination();
            // 默认激活总报告第一页的前后按钮状态
            const summarySection = document.getElementById('section-summary');
            if (summarySection) {
                const prevBtn = summarySection.querySelector('.pagination button[data-dir="prev"]');
                const nextBtn = summarySection.querySelector('.pagination button[data-dir="next"]');
                const pages = summarySection.querySelectorAll('.page');
                if (prevBtn) prevBtn.disabled = true;
                if (nextBtn) nextBtn.disabled = (pages.length <= 1);
            }
        });
    </script>
</body>
</html>
    """

    # 写入 HTML 文件到 output_dir
    html_path = os.path.join(output_dir, "report.html")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✅ 报告已生成：{html_path}")
    return str(html_path)


def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # 创建图片子目录
    images_dir = os.path.join(OUTPUT_DIR, "images")
    os.makedirs(images_dir, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型 {MODEL_PATH} 不存在")
        return
    model = YOLO(MODEL_PATH)

    # 获取类别映射
    class_names = model.names
    target_id = None
    for idx, name in class_names.items():
        if name == TARGET_CLASS:
            target_id = idx
            break
    if target_id is None:
        print(f"❌ 类别 '{TARGET_CLASS}' 不存在，可用类别：{list(class_names.values())}")
        return

    img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
    # 自然排序读取文件列表
    all_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(img_exts)]
    all_files.sort(key=natural_sort_key)
    if not all_files:
        print(f"❌ 在 {INPUT_DIR} 中没有图片")
        return

    colors = generate_colors(20)

    # 存储结果数据
    results_data = []

    for img_name in all_files:
        img_path = os.path.join(INPUT_DIR, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        h_img, w_img = img.shape[:2]

        # 推理
        results = model(img_path, conf=CONF_THRES)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            out_path = os.path.join(images_dir, img_name)
            cv2.imwrite(out_path, img)
            print(f"⚠️ {img_name}: 无目标，保存原图")
            results_data.append({
                'img_name': img_name,
                'orig_count': 0,
                'filtered_count': 0,
                'img_path': out_path
            })
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)

        # 按目标类别过滤
        class_mask = (cls_ids == target_id)
        if not np.any(class_mask):
            out_path = os.path.join(images_dir, img_name)
            cv2.imwrite(out_path, img)
            print(f"⚠️ {img_name}: 未检测到类别 '{TARGET_CLASS}'，保存原图")
            results_data.append({
                'img_name': img_name,
                'orig_count': len(boxes),
                'filtered_count': 0,
                'img_path': out_path
            })
            continue

        xyxy = xyxy[class_mask]
        widths = xyxy[:, 2] - xyxy[:, 0]
        heights = xyxy[:, 3] - xyxy[:, 1]

        # 宽度过滤
        avg_width = np.mean(widths)
        min_width = avg_width * WIDTH_RATIO_MIN
        max_width = avg_width * WIDTH_RATIO_MAX
        width_ok = (widths >= min_width) & (widths <= max_width)

        # 高度过滤
        min_height = h_img - MIN_HEIGHT_MARGIN
        height_ok = heights >= min_height

        keep_mask = width_ok & height_ok
        if not np.any(keep_mask):
            out_path = os.path.join(images_dir, img_name)
            cv2.imwrite(out_path, img)
            print(f"⚠️ {img_name}: 所有框均被过滤，保存原图")
            results_data.append({
                'img_name': img_name,
                'orig_count': len(xyxy),
                'filtered_count': 0,
                'img_path': out_path
            })
            continue

        filtered_boxes = xyxy[keep_mask]
        sorted_indices = np.argsort(filtered_boxes[:, 0])[::-1]
        sorted_boxes = filtered_boxes[sorted_indices]

        # 绘制
        for idx, box in enumerate(sorted_boxes, start=1):
            x1, y1, x2, y2 = box.astype(int)
            color = colors[(idx - 1) % 20]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            label = str(idx)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            base_y = y1 - 5
            offset = (idx - 1) * 20
            text_y = base_y + offset
            if text_y - text_h < 0:
                text_y = text_h + 5 + offset

            text_x = x1
            cv2.rectangle(img,
                          (text_x, text_y - text_h - baseline),
                          (text_x + text_w, text_y + baseline),
                          color, -1)
            cv2.putText(img, label, (text_x, text_y), font, font_scale,
                        (255, 255, 255), thickness)

        out_path = os.path.join(images_dir, img_name)
        cv2.imwrite(out_path, img)

        orig_count = len(xyxy)
        filtered_count = len(sorted_boxes)
        print(f"✅ {img_name}: 目标类别 '{TARGET_CLASS}' 框 {orig_count}，过滤后保留 {filtered_count} 个")

        results_data.append({
            'img_name': img_name,
            'orig_count': orig_count,
            'filtered_count': filtered_count,
            'img_path': out_path
        })

    print(f"\n🎉 推理完成！结果保存在 {OUTPUT_DIR}")

    # 生成报告
    if results_data:
        report_path = generate_report(results_data, OUTPUT_DIR)
        print(f"📊 报告生成地址：{report_path}")
    else:
        print("⚠️ 没有图片数据，无法生成报告")


if __name__ == "__main__":
    main()
