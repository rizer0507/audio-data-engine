# -*- coding: utf-8 -*-
"""
collect_wav_from_parquet.py
===========================
根据一个或多个 parquet 中的样本，从指定源目录抽取对应 WAV，汇总到输出目录。

匹配规则
--------
1. 优先读 parquet 的 ``audio`` 字段（默认依次试 ``pcm_wav`` / ``pcm_to_wav`` /
   ``resampled_16k``）；路径存在则直接用。
2. 若 ``source_path`` 已是 wav（或其它容器格式），在 ``--source-dir`` 中按
   basename / stem / ``{id}.wav`` 查找。
3. 若 ``source_path`` 是 ``.pcm`` / ``.raw``（或源目录未命中），到
   ``--pcm-to-wav-dir``（默认 ``data/derived/pcm_to_wav``）按
   ``<sha16>_<id>.wav`` 后缀匹配。
4. 跨 parquet 按 ``id`` 去重；同一物理文件只复制一次。

Usage
-----
python scripts/collect_wav_from_parquet.py \\
  --parquet datasets/stage1/derived/classified_five_class_v1_0914-mixed-30000.parquet \\
  --source-dir /data/wav_a /data/wav_b \\
  --output-dir data/exports/collected_wav

# 仅 PCM 批次：可不传 --source-dir，只扫 pcm_to_wav
python scripts/collect_wav_from_parquet.py \\
  -p a.parquet -p b.parquet \\
  --pcm-to-wav-dir data/derived/pcm_to_wav \\
  -o data/exports/pcm_wav_subset --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from glob import glob
from pathlib import Path
from typing import Any, Iterable

_PCM_EXTS = {".pcm", ".raw"}
_WAV_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}
_AUDIO_KEY_FALLBACKS = ("pcm_wav", "pcm_to_wav", "resampled_16k")


class CollectError(ValueError):
    """User-facing argument / IO error."""


# ──────────────────────────────────────────────
# 路径展开
# ──────────────────────────────────────────────

def expand_file_patterns(patterns: Iterable[str], *, label: str) -> list[Path]:
    resolved: list[Path] = []
    seen: set[Path] = set()
    for raw in patterns:
        text = str(raw).strip()
        if not text:
            continue
        if any(ch in text for ch in "*?["):
            matches = [Path(item) for item in sorted(glob(text, recursive=True))]
            files = [path for path in matches if path.is_file()]
            if not files:
                raise CollectError(f"{label} glob 没有匹配到文件: {text}")
        else:
            path = Path(text)
            if not path.is_file():
                raise CollectError(f"{label} 文件不存在: {path}")
            files = [path]
        for path in files:
            key = path.resolve()
            if key in seen:
                continue
            seen.add(key)
            resolved.append(path)
    return resolved


def resolve_dirs(raw_dirs: Iterable[str] | None, *, project_root: Path) -> list[Path]:
    if not raw_dirs:
        return []
    out: list[Path] = []
    seen: set[Path] = set()
    for raw in raw_dirs:
        path = Path(raw)
        if not path.is_absolute():
            path = project_root / path
        if not path.is_dir():
            raise CollectError(f"目录不存在: {path}")
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def resolve_path(raw: str | Path, *, project_root: Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = project_root / path
    return path


# ──────────────────────────────────────────────
# parquet 读取
# ──────────────────────────────────────────────

def _parse_audio(raw: Any) -> dict[str, str]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items() if v}
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {}
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items() if v}
    return {}


def load_samples(parquet_paths: list[Path]) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise CollectError("缺少 pandas / pyarrow，请先安装") from exc

    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for path in parquet_paths:
        print(f"[INFO] 读取 parquet: {path}")
        df = pd.read_parquet(path)
        if "id" not in df.columns:
            raise CollectError(f"parquet 缺少 id 列: {path}")
        has_source = "source_path" in df.columns
        has_audio = "audio" in df.columns
        added = 0
        skipped = 0
        for _, row in df.iterrows():
            sid = str(row["id"]).strip()
            if not sid or sid in seen_ids:
                skipped += 1
                continue
            seen_ids.add(sid)
            source_path = str(row["source_path"]).strip() if has_source and row.get("source_path") is not None else ""
            audio = _parse_audio(row["audio"]) if has_audio else {}
            rows.append({"id": sid, "source_path": source_path, "audio": audio})
            added += 1
        print(f"[INFO]   +{added} 条唯一 id（跳过重复/空 id {skipped}）")
    print(f"[INFO] 合计唯一样本: {len(rows)}")
    return rows


# ──────────────────────────────────────────────
# 目录索引
# ──────────────────────────────────────────────

def build_source_wav_index(source_dirs: list[Path], *, recursive: bool) -> dict[str, list[Path]]:
    """按 lowercase 文件名 / stem 建索引，同名可能对应多个路径。"""
    index: dict[str, list[Path]] = {}
    scanned = 0
    for root in source_dirs:
        it = root.rglob("*") if recursive else root.iterdir()
        for path in it:
            if not path.is_file() or path.suffix.lower() not in _WAV_EXTS:
                continue
            scanned += 1
            resolved = path.resolve()
            for key in (path.name.lower(), path.stem.lower()):
                bucket = index.setdefault(key, [])
                if resolved not in bucket:
                    bucket.append(resolved)
    print(f"[INFO] 源目录索引: 扫描 {scanned} 个音频文件，索引键 {len(index)}")
    return index


def build_pcm_to_wav_index(pcm_to_wav_dir: Path, target_ids: set[str]) -> dict[str, Path]:
    """扫描 ``<hash2>/<hash16>_<id>.wav``，返回 {id: path}。"""
    if not pcm_to_wav_dir.is_dir():
        print(f"[WARN] pcm_to_wav 目录不存在: {pcm_to_wav_dir}")
        return {}

    print(f"[INFO] 扫描 pcm_to_wav: {pcm_to_wav_dir}")
    found: dict[str, Path] = {}
    scanned = 0
    for sub in sorted(pcm_to_wav_dir.iterdir()):
        if not sub.is_dir():
            continue
        for wav_file in sub.iterdir():
            if not wav_file.is_file() or wav_file.suffix.lower() != ".wav":
                continue
            scanned += 1
            stem = wav_file.stem
            pos = stem.find("_")
            if pos < 0:
                continue
            sample_id = stem[pos + 1 :]
            if sample_id not in target_ids:
                continue
            prev = found.get(sample_id)
            if prev is None or wav_file.stat().st_mtime > prev.stat().st_mtime:
                found[sample_id] = wav_file.resolve()
    print(f"[INFO] pcm_to_wav 命中 {len(found)} / 目标 {len(target_ids)}（扫描 {scanned} 个 wav）")
    return found


# ──────────────────────────────────────────────
# 解析单条样本 → wav 路径
# ──────────────────────────────────────────────

def _first_existing(candidates: Iterable[Path | None]) -> Path | None:
    for cand in candidates:
        if cand is None:
            continue
        try:
            if cand.is_file():
                return cand.resolve()
        except OSError:
            continue
    return None


def _lookup_source_index(index: dict[str, list[Path]], *names: str) -> Path | None:
    for name in names:
        text = (name or "").strip()
        if not text:
            continue
        for key in (text.lower(), Path(text).name.lower(), Path(text).stem.lower()):
            hits = index.get(key)
            if hits:
                return hits[0]
        if not text.lower().endswith(".wav"):
            hits = index.get(f"{text.lower()}.wav")
            if hits:
                return hits[0]
    return None


def resolve_wav_for_sample(
    sample: dict[str, Any],
    *,
    source_index: dict[str, list[Path]],
    pcm_index: dict[str, Path],
    audio_keys: tuple[str, ...],
) -> tuple[Path | None, str]:
    """返回 (path, reason)。reason 便于统计。"""
    sid = sample["id"]
    source_path = sample.get("source_path") or ""
    audio: dict[str, str] = sample.get("audio") or {}
    suffix = Path(source_path).suffix.lower() if source_path else ""

    # 1) parquet audio 字段里已有绝对路径
    for key in audio_keys:
        raw = audio.get(key)
        if not raw:
            continue
        hit = _first_existing([Path(raw)])
        if hit is not None:
            return hit, f"audio[{key}]"

    # 2) 源是容器格式 → 在 --source-dir 里找
    is_pcm = suffix in _PCM_EXTS
    is_wav_like = suffix in _WAV_EXTS or (not suffix and not is_pcm)

    if is_wav_like or not is_pcm:
        hit = _lookup_source_index(
            source_index,
            Path(source_path).name if source_path else "",
            Path(source_path).stem if source_path else "",
            sid,
            f"{sid}.wav",
        )
        if hit is not None:
            return hit, "source_dir"

        # source_path 本身若仍存在（本机路径）
        if source_path and suffix in _WAV_EXTS:
            hit = _first_existing([Path(source_path)])
            if hit is not None:
                return hit, "source_path"

    # 3) PCM / 源目录未命中 → pcm_to_wav 产物（audio 路径可能在别的机器）
    for key in audio_keys:
        raw = audio.get(key)
        if not raw:
            continue
        name = Path(raw).name
        hit = _lookup_source_index(source_index, name)
        if hit is not None:
            return hit, f"source_dir(via audio[{key}])"

    if sid in pcm_index:
        return pcm_index[sid], "pcm_to_wav(id)"

    return None, "missing"


def unique_dest_name(src: Path, sample_id: str, used: set[str]) -> str:
    base = src.name
    if base not in used:
        return base
    # 冲突时带上 id，避免覆盖
    alt = f"{sample_id}__{src.name}"
    if alt not in used:
        return alt
    i = 2
    while True:
        cand = f"{sample_id}__{src.stem}_{i}{src.suffix}"
        if cand not in used:
            return cand
        i += 1


def collect_files(
    resolved: list[tuple[str, Path, str]],
    output_dir: Path,
    *,
    use_symlink: bool,
    dry_run: bool,
) -> tuple[int, int, int]:
    """返回 (copied_or_linked, skipped_dup_file, failed)。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    used_names: set[str] = set()
    seen_files: set[Path] = set()
    ok = 0
    skip_dup = 0
    fail = 0

    for sid, src, reason in resolved:
        key = src.resolve()
        if key in seen_files:
            skip_dup += 1
            continue
        seen_files.add(key)
        dest_name = unique_dest_name(src, sid, used_names)
        used_names.add(dest_name)
        dst = output_dir / dest_name
        if dry_run:
            print(f"  [{reason}] {sid}  ->  {dst}  <=  {src}")
            ok += 1
            continue
        try:
            if use_symlink:
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                dst.symlink_to(src.resolve())
            else:
                shutil.copy2(src, dst)
            ok += 1
        except OSError as exc:
            print(f"[WARN] {sid} 失败: {exc}", file=sys.stderr)
            fail += 1
    return ok, skip_dup, fail


# ──────────────────────────────────────────────
# Entry
# ──────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="根据 parquet 从源目录 / pcm_to_wav 抽取 WAV 到指定文件夹"
    )
    parser.add_argument(
        "-p",
        "--parquet",
        nargs="+",
        required=True,
        metavar="PATH",
        help="输入 parquet（可多个，支持 glob）",
    )
    parser.add_argument(
        "-s",
        "--source-dir",
        nargs="*",
        default=None,
        metavar="DIR",
        help="原始 WAV 所在目录（可多个）；PCM 样本主要走 --pcm-to-wav-dir",
    )
    parser.add_argument(
        "--pcm-to-wav-dir",
        default="data/derived/pcm_to_wav",
        help="pcm_to_wav 产物根目录（默认: data/derived/pcm_to_wav）",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="汇总输出目录",
    )
    parser.add_argument(
        "--audio-key",
        action="append",
        default=None,
        metavar="KEY",
        help="优先使用的 audio 字段 key，可重复；默认 pcm_wav/pcm_to_wav/resampled_16k",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归扫描 --source-dir",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="用符号链接代替复制（Windows 通常需管理员权限）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印匹配结果，不写文件",
    )
    parser.add_argument(
        "--missing-list",
        default=None,
        help="把未找到的 id 写入该文本文件（一行一个）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[1]

    try:
        parquet_paths = expand_file_patterns(args.parquet, label="parquet")
        source_dirs = resolve_dirs(args.source_dir, project_root=project_root)
        pcm_dir = resolve_path(args.pcm_to_wav_dir, project_root=project_root)
        output_dir = resolve_path(args.output_dir, project_root=project_root)
    except CollectError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    if not source_dirs and not pcm_dir.exists():
        print(
            "[ERROR] 未提供可用的 --source-dir，且 --pcm-to-wav-dir 也不存在",
            file=sys.stderr,
        )
        return 1

    audio_keys = tuple(args.audio_key) if args.audio_key else _AUDIO_KEY_FALLBACKS

    try:
        samples = load_samples(parquet_paths)
    except CollectError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    if not samples:
        print("[WARN] parquet 中没有可用样本")
        return 0

    source_index = (
        build_source_wav_index(source_dirs, recursive=args.recursive) if source_dirs else {}
    )
    target_ids = {s["id"] for s in samples}
    # 仅当存在 PCM 或 audio 字段指向 pcm_to_wav 时才需要扫；简单起见总是扫（若目录存在）
    pcm_index = build_pcm_to_wav_index(pcm_dir, target_ids) if pcm_dir.is_dir() else {}

    resolved: list[tuple[str, Path, str]] = []
    missing: list[str] = []
    reason_counts: dict[str, int] = {}

    for sample in samples:
        path, reason = resolve_wav_for_sample(
            sample,
            source_index=source_index,
            pcm_index=pcm_index,
            audio_keys=audio_keys,
        )
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if path is None:
            missing.append(sample["id"])
        else:
            resolved.append((sample["id"], path, reason))

    print("\n[INFO] 匹配统计:")
    for reason, count in sorted(reason_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {reason:28s}  {count}")
    print(f"  {'命中':28s}  {len(resolved)}")
    print(f"  {'未找到':28s}  {len(missing)}")

    if missing and args.missing_list:
        miss_path = resolve_path(args.missing_list, project_root=project_root)
        miss_path.parent.mkdir(parents=True, exist_ok=True)
        miss_path.write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")
        print(f"[INFO] 未找到列表已写入: {miss_path}")

    if not resolved:
        print("[WARN] 没有任何可汇总的 WAV")
        return 0

    action = "链接" if args.symlink else "复制"
    if args.dry_run:
        print(f"\n[DRY-RUN] 将{action}到 {output_dir}:")
    else:
        print(f"\n[INFO] 开始{action}到: {output_dir}")

    ok, skip_dup, fail = collect_files(
        resolved,
        output_dir,
        use_symlink=args.symlink,
        dry_run=args.dry_run,
    )

    print(f"\n{'=' * 50}")
    print(
        f"完成  成功: {ok}  同文件去重跳过: {skip_dup}  失败: {fail}  未找到: {len(missing)}"
    )
    print(f"输出目录: {output_dir}")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
