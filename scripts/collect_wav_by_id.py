# -*- coding: utf-8 -*-
"""
collect_wav_by_id.py
====================
根据 Excel 文件中的 id 列，从 data/derived/resample_16k 目录（或指定目录）中
找到对应的 WAV 文件，复制到目标目录（默认 test_wav/）。

派生文件命名规则（见 src/audio_engine/core/artifacts.py derived_audio_path）：
    <output_dir>/<subdir>/<sha256前2位>/<sha256前16位>_<sample.id>.wav

因此文件名后缀固定为 "_<sample.id>.wav"，可在不知道 sha256 的情况下直接匹配。

快速路径（推荐）：若提供已处理好的 manifest parquet，直接从 audio 字段读出绝对路径，
跳过文件系统遍历，速度极快（适合 10～20 万条规模）。

慢速兜底路径：仅扫 256 个哈希子目录下的文件名，比起暴力 rglob 效率相当，
但对 20 万文件仍需几十秒，适合没有 manifest 时使用。

Usage
-----
# 仅提供 xlsx，扫描 derived 目录（兜底）
python scripts/collect_wav_by_id.py ids.xlsx

# 提供 manifest，优先走快速路径
python scripts/collect_wav_by_id.py ids.xlsx --manifest datasets/manifests/cleaned_source_A.parquet

# 完整参数
python scripts/collect_wav_by_id.py ids.xlsx \\
    --manifest datasets/manifests/cleaned_source_A.parquet \\
    --derived-dir data/derived/resample_16k \\
    --output-dir test_wav \\
    --id-column id \\
    --audio-key resampled_16k \\
    --copy          # 默认复制；加 --symlink 改为符号链接（Linux/Mac）
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


# ──────────────────────────────────────────────
# 读取 xlsx id 列
# ──────────────────────────────────────────────

def load_ids(xlsx_path: Path, id_column: str) -> list[str]:
    """读取 Excel 文件，返回去重后的 id 列表（保持顺序）。"""
    try:
        import pandas as pd
    except ImportError:
        print("[ERROR] 缺少 pandas，请先安装：pip install pandas openpyxl", file=sys.stderr)
        sys.exit(1)

    df = pd.read_excel(xlsx_path, dtype=str)

    # 自动识别列名（大小写不敏感）
    col_map = {c.strip().lower(): c for c in df.columns}
    key = id_column.strip().lower()
    if key not in col_map:
        available = list(df.columns)
        print(
            f"[ERROR] xlsx 中未找到列 '{id_column}'，可用列：{available}",
            file=sys.stderr,
        )
        sys.exit(1)

    ids = df[col_map[key]].dropna().str.strip().unique().tolist()
    return ids


# ──────────────────────────────────────────────
# 快速路径：从 manifest parquet 读取路径
# ──────────────────────────────────────────────

def load_from_manifest(
    manifest_path: Path,
    target_ids: set[str],
    audio_key: str,
) -> dict[str, Path]:
    """
    从 manifest parquet 的 audio 字段读取派生文件路径。
    返回 {sample_id: Path} 映射，仅包含在 target_ids 中且路径存在的条目。
    """
    try:
        import pandas as pd
    except ImportError:
        print("[ERROR] 缺少 pandas，请先安装：pip install pandas pyarrow", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] 读取 manifest: {manifest_path}")
    df = pd.read_parquet(manifest_path)

    if "id" not in df.columns or "audio" not in df.columns:
        print(
            "[WARN] manifest 中缺少 'id' 或 'audio' 列，跳过快速路径",
            file=sys.stderr,
        )
        return {}

    found: dict[str, Path] = {}
    missing_key: int = 0
    not_exist: int = 0

    for _, row in df.iterrows():
        sid = str(row["id"]).strip()
        if sid not in target_ids:
            continue

        audio_raw = row.get("audio")
        if audio_raw is None:
            missing_key += 1
            continue

        # audio 列在 parquet 中可能是 JSON 字符串或 dict
        if isinstance(audio_raw, str):
            try:
                audio_dict = json.loads(audio_raw)
            except json.JSONDecodeError:
                missing_key += 1
                continue
        elif isinstance(audio_raw, dict):
            audio_dict = audio_raw
        else:
            missing_key += 1
            continue

        wav_path_str = audio_dict.get(audio_key)
        if not wav_path_str:
            missing_key += 1
            continue

        wav_path = Path(wav_path_str)
        if not wav_path.exists():
            not_exist += 1
            continue

        found[sid] = wav_path

    print(
        f"[INFO] manifest 命中 {len(found)} 条"
        + (f"，{missing_key} 条缺少 audio['{audio_key}'] 字段" if missing_key else "")
        + (f"，{not_exist} 条路径在磁盘上不存在" if not_exist else "")
    )
    return found


# ──────────────────────────────────────────────
# 兜底路径：扫描 derived 目录文件系统
# ──────────────────────────────────────────────

def build_fs_index(derived_dir: Path, target_ids: set[str]) -> dict[str, Path]:
    """
    扫描 derived_dir 下的两级目录（<hash2>/<filename>.wav），
    利用文件名后缀 "_<sample.id>.wav" 快速匹配目标 id。

    只需遍历一次文件系统，对 20 万文件约需 10～60 秒（取决于磁盘速度）。
    """
    if not derived_dir.is_dir():
        print(f"[ERROR] derived 目录不存在: {derived_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] 扫描派生目录: {derived_dir}  (可能需要数十秒...)")

    found: dict[str, Path] = {}
    scanned = 0

    # 目录结构：<derived_dir>/<hash2>/<hash16>_<sample_id>.wav
    # hash2 子目录是 00..ff，共最多 256 个，直接枚举
    for sub in sorted(derived_dir.iterdir()):
        if not sub.is_dir():
            continue  # 跳过非目录（如 .gitkeep 等）
        for wav_file in sub.iterdir():
            if not wav_file.is_file() or wav_file.suffix.lower() != ".wav":
                continue
            scanned += 1
            # 文件名格式：<sha256前16位>_<sample.id>.wav
            stem = wav_file.stem  # 去掉 .wav 后缀
            underscore_pos = stem.find("_")
            if underscore_pos < 0:
                continue  # 不符合命名规则，跳过
            sample_id = stem[underscore_pos + 1:]
            if sample_id in target_ids:
                # 同一个 id 理论上只有一个派生文件；若有多个（异常情况）取最新
                if sample_id not in found or wav_file.stat().st_mtime > found[sample_id].stat().st_mtime:
                    found[sample_id] = wav_file

    print(f"[INFO] 扫描完成：共扫描 {scanned} 个文件，命中 {len(found)} 条")
    return found


# ──────────────────────────────────────────────
# 复制 / 链接文件
# ──────────────────────────────────────────────

def collect_files(
    id_to_path: dict[str, Path],
    output_dir: Path,
    use_symlink: bool,
) -> tuple[int, int]:
    """将找到的文件复制（或符号链接）到 output_dir，返回 (成功数, 失败数)。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    fail = 0
    for sid, src in sorted(id_to_path.items()):
        dst = output_dir / src.name  # 保留原始文件名（含 sha256 前缀）
        try:
            if use_symlink:
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                dst.symlink_to(src.resolve())
            else:
                shutil.copy2(src, dst)
            ok += 1
        except Exception as exc:
            print(f"[WARN] 处理 {sid} 失败: {exc}", file=sys.stderr)
            fail += 1
    return ok, fail


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="根据 xlsx id 列收集 resample_16k WAV 文件到目标目录"
    )
    parser.add_argument("xlsx", help="包含 id 列的 Excel 文件路径")
    parser.add_argument(
        "--manifest",
        default=None,
        help="manifest parquet 路径（快速路径，推荐）；不提供则扫描文件系统",
    )
    parser.add_argument(
        "--derived-dir",
        default="data/derived/resample_16k",
        help="派生音频根目录（默认: data/derived/resample_16k）",
    )
    parser.add_argument(
        "--output-dir",
        default="test_wav",
        help="输出目录（默认: test_wav，相对于项目根路径）",
    )
    parser.add_argument(
        "--id-column",
        default="id",
        help="xlsx 中的 id 列名（默认: id）",
    )
    parser.add_argument(
        "--audio-key",
        default="resampled_16k",
        help="manifest audio 字段的 key（默认: resampled_16k）",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="使用符号链接代替复制（Linux/Mac 推荐，Windows 需管理员权限）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要操作的文件，不实际复制",
    )
    args = parser.parse_args()

    # 统一使用脚本所在目录的上级作为项目根（scripts/ 在项目根下一级）
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    xlsx_path = Path(args.xlsx)
    if not xlsx_path.is_absolute():
        xlsx_path = project_root / xlsx_path
    if not xlsx_path.exists():
        print(f"[ERROR] xlsx 文件不存在: {xlsx_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    derived_dir = Path(args.derived_dir)
    if not derived_dir.is_absolute():
        derived_dir = project_root / derived_dir

    # Step 1: 读取目标 id
    print(f"[INFO] 读取 id 列表: {xlsx_path}")
    ids = load_ids(xlsx_path, args.id_column)
    target_ids = set(ids)
    print(f"[INFO] 共 {len(target_ids)} 个唯一 id")

    if not target_ids:
        print("[WARN] id 列表为空，退出", file=sys.stderr)
        sys.exit(0)

    # Step 2: 查找文件路径
    id_to_path: dict[str, Path] = {}

    # 优先使用 manifest
    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.is_absolute():
            manifest_path = project_root / manifest_path
        if manifest_path.exists():
            id_to_path = load_from_manifest(manifest_path, target_ids, args.audio_key)
        else:
            print(f"[WARN] manifest 不存在: {manifest_path}，改用文件系统扫描", file=sys.stderr)

    # 若 manifest 未能命中全部 id，补充扫描文件系统
    remaining = target_ids - set(id_to_path.keys())
    if remaining:
        if id_to_path:
            print(f"[INFO] manifest 未命中 {len(remaining)} 条，补充扫描文件系统...")
        fs_found = build_fs_index(derived_dir, remaining)
        id_to_path.update(fs_found)

    # Step 3: 汇报缺失
    not_found = target_ids - set(id_to_path.keys())
    if not_found:
        print(f"\n[WARN] 以下 {len(not_found)} 个 id 在 derived 目录中未找到对应 WAV 文件：")
        for nf in sorted(not_found):
            print(f"  - {nf}")

    print(f"\n[INFO] 最终找到 {len(id_to_path)} / {len(target_ids)} 个文件")

    if not id_to_path:
        print("[WARN] 没有任何文件可收集，退出")
        sys.exit(0)

    # Step 4: 复制 / 链接
    if args.dry_run:
        print(f"\n[DRY-RUN] 以下文件将被{'链接' if args.symlink else '复制'}到 {output_dir}：")
        for sid, src in sorted(id_to_path.items()):
            print(f"  {sid:40s}  {src}")
        print(f"\n[DRY-RUN] 共 {len(id_to_path)} 个文件（未实际操作）")
        return

    print(f"\n[INFO] 开始{'创建符号链接' if args.symlink else '复制文件'}到: {output_dir}")
    ok, fail = collect_files(id_to_path, output_dir, use_symlink=args.symlink)

    print(f"\n{'='*50}")
    print(f"完成  成功: {ok}  失败: {fail}  未找到: {len(not_found)}")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
