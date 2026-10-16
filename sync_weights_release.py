# -*- coding: utf-8 -*-
"""
把 weights/ 下各任务的最新「版本化」权重同步到 GitHub 固定 release（默认名称 database）。

用法:
    python sync_weights_release.py                 # 同步到 release "database"
    python sync_weights_release.py --dry-run       # 只生成 data.json 并打印计划，不改动 GitHub
    python sync_weights_release.py --repo zunkun/gujitrain   # 显式指定仓库（默认取当前仓库）

流程:
    1. 扫描 weights/，对每个任务取版本号最大的 <task>-v<版本>.pt
    2. 计算 sha256 / 大小 / 日期，生成本地 data.json（清单）
    3. 读取远端 release 里的 data.json，按「版本 + 内容 hash」比较
    4. 本地更新时：删除被取代的旧权重资产 → 上传新权重 → 上传最新 data.json
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(ROOT, "weights")
MANIFEST_NAME = "data.json"
MANIFEST_PATH = os.path.join(ROOT, MANIFEST_NAME)

# 需要同步的任务（与 *train.py 中的 task_name 对应）
TASKS = ("bookcontent", "gujiline")

RELEASE_TAG = "database"


def rel(path):
    """把项目内绝对路径转成相对项目根的短路径，便于打印。"""
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:
        return path


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_latest_versioned(task):
    """在 weights/ 下找 <task>-v<数字>.pt，返回 (版本号, 文件名, 绝对路径)；找不到返回 None。"""
    prefix = f"{task}-v"
    best = None  # (版本号, 文件名, 路径)
    if not os.path.isdir(WEIGHTS_DIR):
        return None
    for fn in os.listdir(WEIGHTS_DIR):
        if not (fn.startswith(prefix) and fn.endswith(".pt")):
            continue
        mid = fn[len(prefix):-len(".pt")]
        if not mid.isdigit():
            continue
        ver = int(mid)
        if best is None or ver > best[0]:
            best = (ver, fn, os.path.join(WEIGHTS_DIR, fn))
    return best


def build_entries():
    """收集各任务最新版本化权重，返回 {task: {name,version,sha256,size,date,path}}。"""
    entries = {}
    for task in TASKS:
        found = find_latest_versioned(task)
        if not found:
            print(f"⚠️  weights/ 下未找到 {task} 的版本化权重（<{task}-v<N>.pt>），跳过")
            continue
        ver, fn, path = found
        st = os.stat(path)
        entries[task] = {
            "name": fn,
            "version": ver,
            "sha256": sha256_of(path),
            "size": st.st_size,
            "date": datetime.fromtimestamp(st.st_mtime).astimezone().isoformat(
                timespec="seconds"
            ),
            "path": path,
        }
        print(f"📦 本地 {task}: {fn}  v{ver}  {st.st_size} bytes  {entries[task]['sha256'][:12]}…")
    return entries


def write_manifest(entries):
    """生成本地 data.json（去掉内部用的 path 字段）。"""
    manifest = {
        "generated_at": now_iso(),
        "release": RELEASE_TAG,
        "files": {
            task: {k: v for k, v in e.items() if k != "path"} for task, e in entries.items()
        },
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return manifest


def run_gh(args):
    """执行 gh 子命令。"""
    return subprocess.run(
        ["gh"] + args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def fetch_release_assets(repo):
    """返回 release 的 assets 列表；release 不存在时返回 None。"""
    args = ["release", "view", RELEASE_TAG, "--json", "assets"]
    if repo:
        args += ["--repo", repo]
    p = run_gh(args)
    if p.returncode != 0:
        return None
    try:
        return json.loads(p.stdout or "{}").get("assets", [])
    except json.JSONDecodeError:
        return None


def download_remote_manifest(repo, assets, tmpdir):
    """下载远端 release 里的 data.json 并解析；没有或失败返回 None。"""
    if not any(a.get("name") == MANIFEST_NAME for a in assets or []):
        return None
    args = ["release", "download", RELEASE_TAG, "--pattern", MANIFEST_NAME, "--dir", tmpdir, "--clobber"]
    if repo:
        args += ["--repo", repo]
    p = run_gh(args)
    if p.returncode != 0:
        print(f"⚠️  下载远端 {MANIFEST_NAME} 失败：{(p.stderr or '').strip()}")
        return None
    path = os.path.join(tmpdir, MANIFEST_NAME)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"⚠️  远端 {MANIFEST_NAME} 解析失败，按需全量上传：{exc}")
        return None


def plan_update(entries, remote_manifest, assets):
    """比较本地与远端，返回 (需上传的任务列表, 需删除的旧资产名列表)。"""
    remote_files = (remote_manifest or {}).get("files", {})
    upload_tasks = []
    stale_assets = []
    for task, entry in entries.items():
        remote = remote_files.get(task)
        reason = None
        if remote is None:
            reason = "远端无此文件"
        else:
            remote_ver = int(remote.get("version", -1))
            if entry["version"] > remote_ver:
                reason = f"本地更新(v{entry['version']} > v{remote_ver})"
            elif entry["version"] == remote_ver and entry["sha256"] != remote.get("sha256"):
                reason = f"同版本内容已变化(v{entry['version']})"
            elif entry["version"] < remote_ver:
                print(f"⚠️  {task}: 远端版本 v{remote_ver} 高于本地 v{entry['version']}，跳过")

        if not reason:
            print(f"✅ {task}: 已是最新(v{entry['version']})")
            continue

        print(f"⬆️  {task}: 需要上传（{reason}）")
        upload_tasks.append(task)

        # 收集该任务被取代的旧权重资产（release 里只保留最新一份）
        prefix = f"{task}-v"
        for a in assets or []:
            name = a.get("name", "")
            if name.startswith(prefix) and name.endswith(".pt") and name != entry["name"]:
                stale_assets.append(name)
    return upload_tasks, stale_assets


def delete_asset(name, repo):
    args = ["release", "delete-asset", RELEASE_TAG, name, "--yes"]
    if repo:
        args += ["--repo", repo]
    p = run_gh(args)
    if p.returncode == 0:
        print(f"  🗑️  已删除旧资产 {name}")
        return True
    print(f"  ❌ 删除 {name} 失败：{(p.stderr or '').strip()}")
    return False


def upload_asset(path, repo):
    args = ["release", "upload", RELEASE_TAG, path, "--clobber"]
    if repo:
        args += ["--repo", repo]
    p = run_gh(args)
    if p.returncode == 0:
        print(f"  ⬆️  已上传 {os.path.basename(path)}")
        return True
    print(f"  ❌ 上传 {os.path.basename(path)} 失败：{(p.stderr or '').strip()}")
    return False


def create_release(paths, repo, generated_at):
    args = [
        "release", "create", RELEASE_TAG,
        "--title", RELEASE_TAG,
        "--notes", f"weights database，自动同步于 {generated_at}",
    ] + paths
    if repo:
        args += ["--repo", repo]
    p = run_gh(args)
    if p.returncode == 0:
        print(f"🆕 已创建 release {RELEASE_TAG}: {(p.stdout or '').strip()}")
        return True
    print(f"❌ 创建 release {RELEASE_TAG} 失败：{(p.stderr or '').strip()}")
    return False


def main():
    global RELEASE_TAG
    parser = argparse.ArgumentParser(description="同步 weights/ 最新版本化权重到 GitHub release")
    parser.add_argument("--release", default=RELEASE_TAG, help="release 名称/标签（默认 database）")
    parser.add_argument("--repo", default=None, help="owner/name，默认取当前仓库")
    parser.add_argument("--dry-run", action="store_true", help="只生成 data.json 并打印计划，不改动 GitHub")
    args = parser.parse_args()
    RELEASE_TAG = args.release

    if not shutil.which("gh"):
        print("❌ 未找到 gh 命令，请先安装 GitHub CLI：https://cli.github.com/")
        return 1

    entries = build_entries()
    if not entries:
        print("❌ weights/ 下没有可同步的版本化权重，退出")
        return 1

    manifest = write_manifest(entries)
    print(f"🧾 已生成本地清单: {rel(MANIFEST_PATH)}")

    assets = fetch_release_assets(args.repo)
    remote_manifest = None
    if assets is None:
        print(f"ℹ️  远端 release {RELEASE_TAG} 不存在，将新建")
    else:
        print(f"ℹ️  远端 release {RELEASE_TAG} 存在，{len(assets)} 个资产")
        with tempfile.TemporaryDirectory() as tmp:
            remote_manifest = download_remote_manifest(args.repo, assets, tmp)
        if remote_manifest:
            print(f"ℹ️  远端清单 generated_at={remote_manifest.get('generated_at')}")

    upload_tasks, stale_assets = plan_update(entries, remote_manifest, assets)

    if not upload_tasks:
        print("🎉 远端已是最新，无需更新")
        return 0

    to_upload = [entries[t]["path"] for t in upload_tasks] + [MANIFEST_PATH]
    if args.dry_run:
        print("\n[dry-run] 计划：")
        if stale_assets:
            print("  删除旧资产:", ", ".join(stale_assets))
        print("  上传文件:", ", ".join(os.path.basename(p) for p in to_upload))
        return 0

    print("\n开始同步…")
    ok = True
    if assets is None:
        ok = create_release(to_upload, args.repo, manifest["generated_at"])
    else:
        for name in stale_assets:
            ok = delete_asset(name, args.repo) and ok
        for path in to_upload:
            ok = upload_asset(path, args.repo) and ok

    print("✅ 同步完成" if ok else "⚠️  同步结束，但有失败项，请检查上面的错误")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
