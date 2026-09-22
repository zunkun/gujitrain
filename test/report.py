import os
from utils import natural_sort_key, paginate

IMAGES_PER_PAGE = 40

def generate_report(results_data, output_dir, report_title="检测报告"):
    # 按过滤后数量归类
    groups = {}
    abnormal_items = []
    for item in results_data:
        count = item['filtered_count']
        groups.setdefault(count, []).append(item)
        if item.get('abnormal') and len(item['abnormal']) > 0:
            abnormal_items.append(item)

    for count in groups:
        groups[count].sort(key=lambda x: natural_sort_key(x['img_name']))

    results_data.sort(key=lambda x: natural_sort_key(x['img_name']))

    for item in results_data:
        item['rel_path'] = f"images/{os.path.basename(item['img_path'])}"

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
        .menu-item:hover {{ background: #34495e; }}
        .menu-item.active {{ background: #1abc9c; }}
        .menu-item .count-badge {{
            float: right;
            background: #1abc9c;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 0.8em;
        }}
        .menu-item.folder .count-badge {{ background: #e67e22; }}
        .menu-item.abnormal .count-badge {{ background: #e74c3c; }}
        .content {{
            flex: 1;
            padding: 20px 30px;
            overflow-y: auto;
            height: 100vh;
        }}
        .section {{ display: none; }}
        .section.active {{ display: block; }}
        .section h1 {{
            margin-bottom: 10px;
            color: #2c3e50;
            border-bottom: 2px solid #1abc9c;
            padding-bottom: 10px;
        }}
        /* 搜索框样式 */
        .search-container {{
            margin: 10px 0 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .search-container input {{
            padding: 8px 16px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 1em;
            width: 300px;
            transition: border 0.2s;
        }}
        .search-container input:focus {{
            outline: none;
            border-color: #1abc9c;
        }}
        .search-container .search-count {{
            color: #7f8c8d;
            font-size: 0.95em;
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
            border: 2px solid transparent;
        }}
        .image-card.abnormal-card {{
            border-color: #e74c3c;
            box-shadow: 0 0 12px rgba(231, 76, 60, 0.5);
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
            cursor: pointer;               /* 新增：鼠标指针改为手型 */
        }}
        .image-card .info .counts {{ color: #7f8c8d; }}
        .image-card .info .abnormal-tag {{
            display: inline-block;
            background: #e74c3c;
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.75em;
            margin-top: 4px;
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
        .summary-stats .stat .label {{ color: #7f8c8d; font-size: 0.9em; }}
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
        .pagination button:disabled {{ opacity: 0.5; cursor: not-allowed; }}
        .section ul {{
            list-style: none;
            padding: 0;
            margin: 10px 0;
        }}
        .section ul li {{
            padding: 8px 16px;
            border-bottom: 1px solid #e9ecef;
        }}
        .section ul li:nth-child(odd) {{ background-color: #f1f3f5; }}
        .section ul li:nth-child(even) {{ background-color: #ffffff; }}
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
        #lightbox .img-name {{
          font-size: 30px;
          padding: 20px 30px;
          color: white;
          z-index: 100000;
          cursor: pointer;           /* 新增：指示可点击复制 */
        }}
        #lightbox .img-name:hover {{
          text-decoration: underline;  /* 新增：悬停提示 */
        }}

        #lightbox img {{
            max-width: 80%;
            max-height: 80%;
            object-fit: contain;
            border-radius: 8px;
            box-shadow: 0 0 40px rgba(0,0,0,0.6);
            cursor: grab;
            transition: opacity 0.2s;
            transform-origin: center center;
            user-select: none;
            -webkit-user-drag: none;
        }}
        #lightbox img:active {{ cursor: grabbing; }}
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
        .lightbox-nav:hover {{ background: rgba(255,255,255,0.3); }}
        .lightbox-nav.prev {{ left: 30px; }}
        .lightbox-nav.next {{ right: 30px; }}
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
        /* 搜索时隐藏未匹配卡片 */
        .image-card.hidden-by-search {{
            display: none !important;
        }}
    </style>
</head>
<body>
    <div class="menu">
        <h2>📊 检测报告</h2>
"""

    # 如果有异常图片，添加异常菜单项（默认激活）
    if abnormal_items:
        html_content += f"""
        <div class="menu-item abnormal active" data-target="abnormal">
            ⚠️ 异常图片 <span class="count-badge">{len(abnormal_items)}</span>
        </div>
        """
        html_content += f"""
        <div class="menu-item folder" data-target="summary">
            📋 总报告 <span class="count-badge">{len(results_data)}</span>
        </div>
        """
    else:
        html_content += f"""
        <div class="menu-item folder active" data-target="summary">
            📋 总报告 <span class="count-badge">{len(results_data)}</span>
        </div>
        """

    for count, items in sorted(groups.items(), reverse=True):
        html_content += f"""
        <div class="menu-item folder" data-target="group_{count}">
            🏷️ {count} 个目标 <span class="count-badge">{len(items)}</span>
        </div>
        """

    html_content += """
    </div>

    <div class="content">
        <div id="lightbox">
            <div class="img-name" onclick="onCopyName()">图片名称</div>
            <span class="close-btn" onclick="closeLightbox(); event.stopPropagation();">&times;</span>
            <div class="lightbox-nav prev" onclick="changeImage(-1); event.stopPropagation();">‹</div>
            <img id="lightbox-img" src="" alt="预览">
            <div class="lightbox-nav next" onclick="changeImage(1); event.stopPropagation();">›</div>
            <div class="lightbox-counter" id="lightbox-counter">1 / 1</div>
        </div>
"""

    # ---------- 异常图片 Section ----------
    if abnormal_items:
        html_content += f"""
        <div id="section-abnormal" class="section active">
            <h1>⚠️ 异常图片（共 {len(abnormal_items)} 张）</h1>
            <p style="color:#e74c3c; margin-bottom:15px;">以下图片存在相邻列间距过大或列中心被包含等问题，已在图像中央用红色文字标注。</p>
            <div class="pagination" id="pagination-abnormal"></div>
        """
        abnormal_pages = list(paginate(abnormal_items, IMAGES_PER_PAGE))
        for page_idx, page_items in enumerate(abnormal_pages):
            active_class = "active" if page_idx == 0 else ""
            html_content += f'<div class="page {active_class}" data-section="abnormal" data-page="{page_idx}"><div class="image-grid">'
            for item in page_items:
                tags = []
                for typ, detail in item['abnormal']:
                    if typ == 'spacing':
                        tags.append('间距大')
                    elif typ == 'contain':
                        tags.append('中心被包含')
                tag_str = ' | '.join(tags)
                html_content += f"""
                <div class="image-card abnormal-card" data-src="{item['rel_path']}" data-name="{item['img_name']}">
                    <img src="{item['rel_path']}" alt="{item['img_name']}" loading="lazy">
                    <div class="info">
                        <div class="filename">{item['img_name']}</div>
                        <div class="counts">原始框 {item['orig_count']} → 过滤后 {item['filtered_count']}</div>
                        <div class="abnormal-tag">⚠ {tag_str}</div>
                    </div>
                </div>
                """
            html_content += '</div></div>'
        html_content += '</div>'

    # ---------- 总报告（新增搜索框） ----------
    summary_active = "" if abnormal_items else "active"
    html_content += f"""
    <div id="section-summary" class="section {summary_active}">
        <h1>📈 总体统计</h1>
        <div class="summary-stats">
            <div class="stat"><span class="number">{len(results_data)}</span><span class="label">总图片数</span></div>
            <div class="stat"><span class="number">{sum(len(items) for items in groups.values())}</span><span class="label">总检测框数（过滤后）</span></div>
            <div class="stat"><span class="number">{len(groups)}</span><span class="label">不同目标数量分类</span></div>
        </div>
        <div style="background:white; padding:15px; border-radius:8px; margin-bottom:20px;">
            <h3>📂 分类详情</h3>
            <ul>
    """
    for count, items in sorted(groups.items(), reverse=True):
        html_content += f"<li><strong>{count} 个目标：</strong>{len(items)} 张图片</li>"
    html_content += """
            </ul>
        </div>
        <!-- 搜索框 -->
        <div class="search-container">
            <input type="text" id="searchInput" placeholder="🔍 搜索图片名称..." autocomplete="off">
            <span class="search-count" id="searchCount"></span>
        </div>
        <h2 style="margin-top:10px;">📷 全部图片预览</h2>
        <div class="pagination" id="pagination-summary"></div>
    """
    summary_pages = list(paginate(results_data, IMAGES_PER_PAGE))
    for page_idx, page_items in enumerate(summary_pages):
        active_class = "active" if page_idx == 0 else ""
        html_content += f'<div class="page {active_class}" data-section="summary" data-page="{page_idx}"><div class="image-grid">'
        for item in page_items:
            # 修改：添加 data-name 属性
            html_content += f"""
            <div class="image-card" data-src="{item['rel_path']}" data-name="{item['img_name']}">
                <img src="{item['rel_path']}" alt="{item['img_name']}" loading="lazy">
                <div class="info">
                    <div class="filename">{item['img_name']}</div>
                    <div class="counts">原始框 {item['orig_count']} → 过滤后 {item['filtered_count']}</div>
                </div>
            </div>
            """
        html_content += '</div></div>'
    html_content += '</div>'

    # ---------- 分组 Sections ----------
    for count, items in sorted(groups.items(), reverse=True):
        html_content += f"""
        <div id="section-group_{count}" class="section">
            <h1>🏷️ {count} 个目标（共 {len(items)} 张）</h1>
            <div class="pagination" id="pagination-group_{count}"></div>
        """
        group_pages = list(paginate(items, IMAGES_PER_PAGE))
        for page_idx, page_items in enumerate(group_pages):
            active_class = "active" if page_idx == 0 else ""
            html_content += f'<div class="page {active_class}" data-section="group_{count}" data-page="{page_idx}"><div class="image-grid">'
            for item in page_items:
                # 修改：添加 data-name 属性
                html_content += f"""
                <div class="image-card" data-src="{item['rel_path']}" data-name="{item['img_name']}">
                    <img src="{item['rel_path']}" alt="{item['img_name']}" loading="lazy">
                    <div class="info">
                        <div class="filename">{item['img_name']}</div>
                        <div class="counts">原始框 {item['orig_count']} → 过滤后 {item['filtered_count']}</div>
                    </div>
                </div>
                """
            html_content += '</div></div>'
        html_content += '</div>'

    # ---------- JavaScript（含搜索和复制功能） ----------
    html_content += """
    </div>
    <script>
        // ========== 灯箱核心（修改：支持名称数组） ==========
        let lightboxImages = [];
        let lightboxNames = [];      // 新增：存储对应的图片名称
        let currentIndex = 0;
        let lightboxOpen = false;
        let currentScale = 1;
        let offsetX = 0, offsetY = 0;
        const MIN_SCALE = 0.5;
        const MAX_SCALE = 3;
        let isDragging = false;
        let startX = 0, startY = 0;
        let startOffsetX = 0, startOffsetY = 0;

        // ---------- 复制功能 ----------
        function copyFileName(name) {
            if (!name) return;
            if (navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(name).then(() => {
                    alert('已复制: ' + name);
                }).catch(() => {
                    fallbackCopy(name);
                });
            } else {
                fallbackCopy(name);
            }
        }

        function fallbackCopy(text) {
            const textarea = document.createElement('textarea');
            textarea.value = text;
            document.body.appendChild(textarea);
            textarea.select();
            try {
                document.execCommand('copy');
                alert('已复制: ' + text);
            } catch (err) {
                alert('复制失败，请手动复制');
            }
            document.body.removeChild(textarea);
        }

        // 灯箱中的复制（由 onclick 调用）
        function onCopyName() {
            const nameEl = document.querySelector('#lightbox .img-name');
            if (nameEl) {
                const name = nameEl.textContent.trim();
                if (name && name !== '图片名称') {
                    copyFileName(name);
                }
            }
        }
        // ---------------------------------

        function updateImageTransform() {
            const img = document.getElementById('lightbox-img');
            img.style.transform = `translate(${offsetX}px, ${offsetY}px) scale(${currentScale})`;
        }

        // 新增：更新灯箱中显示的名称
        function updateLightboxName() {
            const nameEl = document.querySelector('#lightbox .img-name');
            if (nameEl) {
                let name = '';
                if (lightboxNames && lightboxNames.length > currentIndex) {
                    name = lightboxNames[currentIndex];
                } else {
                    // fallback: 从图片路径提取文件名
                    const src = lightboxImages[currentIndex];
                    if (src) {
                        const parts = src.split('/');
                        name = parts[parts.length - 1];
                    }
                }
                nameEl.textContent = name || '未知图片';
            }
        }

        function openLightbox(images, index, names) {
            if (!images || images.length === 0) return;
            lightboxImages = images;
            lightboxNames = names || [];   // 接收名称数组
            currentIndex = index;
            lightboxOpen = true;
            currentScale = 1;
            offsetX = 0;
            offsetY = 0;
            const img = document.getElementById('lightbox-img');
            img.src = lightboxImages[currentIndex];
            updateImageTransform();
            updateLightboxName();   // 更新名称
            updateCounter();
            document.getElementById('lightbox').style.display = 'flex';
            document.body.style.overflow = 'hidden';
            document.addEventListener('keydown', handleKeydown);
        }

        function closeLightbox() {
            document.getElementById('lightbox').style.display = 'none';
            document.body.style.overflow = 'auto';
            lightboxOpen = false;
            document.removeEventListener('keydown', handleKeydown);
            currentScale = 1;
            offsetX = 0;
            offsetY = 0;
            updateImageTransform();
        }

        function changeImage(delta) {
            if (!lightboxImages || lightboxImages.length === 0) return;
            let newIndex = currentIndex + delta;
            if (newIndex < 0) newIndex = lightboxImages.length - 1;
            if (newIndex >= lightboxImages.length) newIndex = 0;
            currentIndex = newIndex;
            currentScale = 1;
            offsetX = 0;
            offsetY = 0;
            const img = document.getElementById('lightbox-img');
            img.src = lightboxImages[currentIndex];
            updateImageTransform();
            updateLightboxName();   // 更新名称
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
            if (e.key === 'ArrowLeft') { e.preventDefault(); changeImage(-1); }
            else if (e.key === 'ArrowRight') { e.preventDefault(); changeImage(1); }
            else if (e.key === 'Escape') { closeLightbox(); }
        }

        document.addEventListener('DOMContentLoaded', function() {
            const lightbox = document.getElementById('lightbox');
            lightbox.addEventListener('click', function(e) {
                if (e.target === this) closeLightbox();
            });
        });

        // 滚轮缩放
        document.addEventListener('DOMContentLoaded', function() {
            const lightbox = document.getElementById('lightbox');
            lightbox.addEventListener('wheel', function(e) {
                if (!lightboxOpen) return;
                e.preventDefault();
                const delta = e.deltaY > 0 ? -0.1 : 0.1;
                currentScale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, currentScale + delta));
                updateImageTransform();
            }, { passive: false });
        });

        // 拖拽平移
        function initDragEvents() {
            const img = document.getElementById('lightbox-img');
            img.addEventListener('mousedown', function(e) {
                if (!lightboxOpen) return;
                isDragging = true;
                startX = e.clientX;
                startY = e.clientY;
                startOffsetX = offsetX;
                startOffsetY = offsetY;
                img.style.cursor = 'grabbing';
                e.preventDefault();
            });
            document.addEventListener('mousemove', function(e) {
                if (!isDragging || !lightboxOpen) return;
                const dx = e.clientX - startX;
                const dy = e.clientY - startY;
                offsetX = startOffsetX + dx;
                offsetY = startOffsetY + dy;
                updateImageTransform();
            });
            document.addEventListener('mouseup', function() {
                if (isDragging) {
                    isDragging = false;
                    document.getElementById('lightbox-img').style.cursor = 'grab';
                }
            });
            // 触摸
            let touchStartX, touchStartY, touchStartOffX, touchStartOffY;
            img.addEventListener('touchstart', function(e) {
                if (!lightboxOpen) return;
                const touch = e.touches[0];
                touchStartX = touch.clientX;
                touchStartY = touch.clientY;
                touchStartOffX = offsetX;
                touchStartOffY = offsetY;
                isDragging = true;
                e.preventDefault();
            }, { passive: false });
            document.addEventListener('touchmove', function(e) {
                if (!isDragging || !lightboxOpen) return;
                const touch = e.touches[0];
                const dx = touch.clientX - touchStartX;
                const dy = touch.clientY - touchStartY;
                offsetX = touchStartOffX + dx;
                offsetY = touchStartOffY + dy;
                updateImageTransform();
                e.preventDefault();
            }, { passive: false });
            document.addEventListener('touchend', function() { isDragging = false; });
        }

        // ========== 图片卡片点击（打开灯箱，收集名称） ==========
        document.addEventListener('click', function(e) {
            const target = e.target;
            if(target.tagName.toLowerCase() !== 'img' || !target.closest('.image-card')) {
              return
            }
            const card = e.target.closest('.image-card');
            const section = card.closest('.section');
            if (!section) return;
            const allCards = section.querySelectorAll('.image-card');
            const srcList = [];
            const nameList = [];
            allCards.forEach(c => {
                const src = c.dataset.src;
                if (src) {
                    srcList.push(src);
                    const name = c.dataset.name || '';
                    nameList.push(name);
                }
            });
            const currentSrc = card.dataset.src;
            const idx = srcList.indexOf(currentSrc);
            if (idx === -1) return;
            openLightbox(srcList, idx, nameList);
            e.stopPropagation();
        });

        // ========== 点击图片名称复制 ==========
        document.addEventListener('click', function(e) {
            const target = e.target.closest('.filename');
            if (target) {
                e.stopPropagation();   // 防止冒泡到卡片（虽然卡片点击只对img生效，但保险）
                const name = target.textContent.trim();
                if (name) {
                    copyFileName(name);
                }
            }
        });

        // ========== 菜单切换 ==========
        const menuItems = document.querySelectorAll('.menu-item');
        const sections = document.querySelectorAll('.section');
        menuItems.forEach(item => {
            item.addEventListener('click', function(e) {
                const target = this.dataset.target;
                if (!target) return;
                sections.forEach(sec => sec.classList.remove('active'));
                const targetSection = document.getElementById('section-' + target);
                if (targetSection) targetSection.classList.add('active');
                menuItems.forEach(it => it.classList.remove('active'));
                this.classList.add('active');
                // 重置分页到第一页
                const container = document.getElementById('section-' + target);
                if (container) {
                    const pages = container.querySelectorAll('.page');
                    pages.forEach((p, i) => p.classList.toggle('active', i === 0));
                    const btns = container.querySelectorAll('.pagination button');
                    btns.forEach((b, i) => {
                        if (i > 0 && i <= pages.length) {
                            b.classList.toggle('active', (i-1) === 0);
                        }
                    });
                    const prevBtn = container.querySelector('.pagination button[data-dir="prev"]');
                    const nextBtn = container.querySelector('.pagination button[data-dir="next"]');
                    if (prevBtn) prevBtn.disabled = true;
                    if (nextBtn) nextBtn.disabled = (pages.length <= 1);
                }
                // 如果切换到总报告，清空搜索框并恢复分页
                if (target === 'summary') {
                    const searchInput = document.getElementById('searchInput');
                    if (searchInput) {
                        searchInput.value = '';
                        triggerSearch();
                    }
                }
            });
        });

        // ---------- 搜索功能 ----------
        function triggerSearch() {
            const input = document.getElementById('searchInput');
            if (!input) return;
            const keyword = input.value.trim().toLowerCase();
            const section = document.getElementById('section-summary');
            if (!section) return;
            const cards = section.querySelectorAll('.image-card');
            const pagination = section.querySelector('.pagination');
            const pages = section.querySelectorAll('.page');
            const countSpan = document.getElementById('searchCount');

            let matchCount = 0;

            cards.forEach(card => {
                const filenameEl = card.querySelector('.filename');
                if (!filenameEl) return;
                const filename = filenameEl.textContent.toLowerCase();
                if (keyword === '' || filename.includes(keyword)) {
                    card.classList.remove('hidden-by-search');
                    matchCount++;
                } else {
                    card.classList.add('hidden-by-search');
                }
            });

            // 更新匹配数量
            if (countSpan) {
                countSpan.textContent = keyword ? `匹配 ${matchCount} 张` : '';
            }

            // 搜索时：显示所有页，隐藏分页；清空时：恢复分页并只显示第一页
            if (keyword) {
                pages.forEach(p => p.classList.add('active'));
                if (pagination) pagination.style.display = 'none';
            } else {
                pages.forEach((p, i) => p.classList.toggle('active', i === 0));
                if (pagination) pagination.style.display = 'flex';
                // 恢复分页按钮状态
                const btns = pagination ? pagination.querySelectorAll('button') : [];
                btns.forEach((b, i) => {
                    if (i > 0 && i <= pages.length) {
                        b.classList.toggle('active', (i-1) === 0);
                    }
                });
                const prevBtn = pagination ? pagination.querySelector('button[data-dir="prev"]') : null;
                const nextBtn = pagination ? pagination.querySelector('button[data-dir="next"]') : null;
                if (prevBtn) prevBtn.disabled = true;
                if (nextBtn) nextBtn.disabled = (pages.length <= 1);
            }
        }

        // 绑定搜索框事件
        document.addEventListener('DOMContentLoaded', function() {
            const searchInput = document.getElementById('searchInput');
            if (searchInput) {
                searchInput.addEventListener('input', triggerSearch);
                searchInput.addEventListener('focus', function() {
                    const summarySection = document.getElementById('section-summary');
                    if (summarySection && !summarySection.classList.contains('active')) {
                        const menuItem = document.querySelector('.menu-item[data-target="summary"]');
                        if (menuItem) menuItem.click();
                    }
                });
            }
            initDragEvents();
            // 初始禁用状态
            document.querySelectorAll('.section').forEach(section => {
                const prevBtn = section.querySelector('.pagination button[data-dir="prev"]');
                const nextBtn = section.querySelector('.pagination button[data-dir="next"]');
                const pages = section.querySelectorAll('.page');
                if (prevBtn) prevBtn.disabled = true;
                if (nextBtn) nextBtn.disabled = (pages.length <= 1);
            });
        });

        // 分页初始化（与原有相同）
        function initPagination() {
            const sections = document.querySelectorAll('.section');
            sections.forEach(section => {
                const sectionId = section.id.replace('section-', '');
                const pages = section.querySelectorAll('.page');
                const paginationDiv = section.querySelector('.pagination');
                if (!paginationDiv || pages.length <= 1) {
                    if (paginationDiv) paginationDiv.style.display = 'none';
                    return;
                }
                paginationDiv.innerHTML = '';
                const prevBtn = document.createElement('button');
                prevBtn.textContent = '‹ 上一页';
                prevBtn.disabled = true;
                prevBtn.dataset.section = sectionId;
                prevBtn.dataset.dir = 'prev';
                paginationDiv.appendChild(prevBtn);

                for (let i = 0; i < pages.length; i++) {
                    const btn = document.createElement('button');
                    btn.textContent = i + 1;
                    btn.dataset.page = i;
                    btn.dataset.section = sectionId;
                    if (i === 0) btn.classList.add('active');
                    paginationDiv.appendChild(btn);
                }

                const nextBtn = document.createElement('button');
                nextBtn.textContent = '下一页 ›';
                nextBtn.dataset.section = sectionId;
                nextBtn.dataset.dir = 'next';
                if (pages.length <= 1) nextBtn.disabled = true;
                paginationDiv.appendChild(nextBtn);

                paginationDiv.addEventListener('click', function(e) {
                    const target = e.target.closest('button');
                    if (!target || target.disabled) return;
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
                    pages.forEach((p, i) => p.classList.toggle('active', i === newPage));
                    const btns = container.querySelectorAll('.pagination button');
                    btns.forEach((b, i) => {
                        if (i > 0 && i <= pages.length) {
                            b.classList.toggle('active', (i-1) === newPage);
                        }
                    });
                    const prevBtn = container.querySelector('.pagination button[data-dir="prev"]');
                    const nextBtn = container.querySelector('.pagination button[data-dir="next"]');
                    if (prevBtn) prevBtn.disabled = (newPage === 0);
                    if (nextBtn) nextBtn.disabled = (newPage === pages.length - 1);
                });
            });
        }

        document.addEventListener('DOMContentLoaded', function() {
            initPagination();
        });
    </script>
</body>
</html>
    """

    html_path = os.path.join(output_dir, "report.html")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✅ 报告已生成：{html_path}")
    return str(html_path)
