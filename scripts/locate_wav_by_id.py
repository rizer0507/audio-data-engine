#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""根据 sample id（如 fenshen-276cc46a-bf3387be）迅速定位 WAV 路径。

查找顺序（由快到慢）
------------------
1. 派生目录按命名规则 glob：``data/derived/<subdir>/<sha2>/<sha16>_<id>.wav``
2. Manifest parquet 按 ``id`` 过滤，读 ``audio.*`` / ``source_path``
3. （可选）``--scan-manifests`` 扫 ``datasets/stage1/{cleaned,asr,derived}``

Usage
-----
# 最快：只靠派生目录命名
python scripts/locate_wav_by_id.py fenshen-276cc46a-bf3387be

# 指定批次底表（推荐有 cleaned 时）
python scripts/locate_wav_by_id.py fenshen-276cc46a-bf3387be --batch 20260813-1

# 多个 id + 显式 manifest
python scripts/locate_wav_by_id.py fenshen-aaa-bbbb fenshen-cccc-dddd \\
  --manifest datasets/stage1/cleaned/cleaned_20260813-1.parquet

# 只打印「首选存在路径」（便于管道）
python scripts/locate_wav_by_id.py fenshen-276cc46a-bf3387be -q
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DERIVED = (
    "data/derived/pcm_to_wav",
    "data/derived/resample_16k",
    "data/derived/vad_segments",
)
AUDIO_KEY_ORDER = (
    "resampled_16k",
    "pcm_wav",
    "pcm_to_wav",
    "normalized_16k",
    "raw",
)
MANIFEST_SCAN_DIRS = (
    "datasets/stage1/cleaned",
    "datasets/stage1/asr",
    "datasets/stage1/derived",
)


def _info(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr, flush=True)


def normalize_id(raw: str) -> str:
    text = str(raw or "").strip()
    # 允许粘贴时带 .wav / 路径尾
    text = Path(text).name
    if text.lower().endswith(".wav"):
        text = text[: -len(".wav")]
    # 派生文件名：<sha16>_<id>
    if "_" in text and not text.startswith("fenshen-") and not text.startswith("seg_"):
        # 仅当前缀像 sha 短 hex 时剥掉
        prefix, rest = text.split("_", 1)
        if len(prefix) in {8, 16, 32, 64} and all(c in "0123456789abcdefABCDEF" for c in prefix):
            text = rest
    return text


def resolve_under_root(raw: str | Path, *, root: Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def rewrite_path(path: str, path_maps: list[tuple[str, str]]) -> str:
    text = str(path or "")
    for old, new in path_maps:
        if text.startswith(old):
            return new + text[len(old) :]
    return text


def parse_path_maps(raw: list[str] | None) -> list[tuple[str, str]]:
    maps: list[tuple[str, str]] = []
    for item in raw or []:
        if "=" not in item:
            raise ValueError(f"--path-map 格式应为 OLD=NEW，收到: {item!r}")
        old, new = item.split("=", 1)
        if not old:
            raise ValueError(f"--path-map OLD 为空: {item!r}")
        maps.append((old, new))
    return maps


def glob_derived(sid: str, derived_dirs: list[Path]) -> list[dict[str, Any]]:
    """``<dir>/<hh>/<sha16>_<id>.wav``；也兼容更深一层。"""
    hits: list[dict[str, Any]] = []
    pattern_names = (f"*_{sid}.wav", f"{sid}.wav")
    for root in derived_dirs:
        if not root.is_dir():
            continue
        found: list[Path] = []
        # 常见两级：hash2 / file
        for name_pat in pattern_names:
            found.extend(root.glob(f"*/{name_pat}"))
            found.extend(root.glob(f"*/*/{name_pat}"))
        # 去重保序
        seen: set[Path] = set()
        for path in found:
            key = path.resolve()
            if key in seen or not path.is_file():
                continue
            seen.add(key)
            hits.append(
                {
                    "kind": "derived_fs",
                    "key": root.name,
                    "path": str(key),
                    "exists": True,
                    "source": str(root),
                }
            )
    return hits


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


def lookup_manifest(
    manifest: Path,
    target_ids: set[str],
    *,
    path_maps: list[tuple[str, str]],
) -> dict[str, list[dict[str, Any]]]:
    """按 id 过滤 parquet，返回 {id: [candidate, ...]}。"""
    out: dict[str, list[dict[str, Any]]] = {sid: [] for sid in target_ids}
    if not manifest.is_file():
        return out

    try:
        import pyarrow.dataset as ds
    except ImportError:
        return _lookup_manifest_pandas(manifest, target_ids, path_maps=path_maps)

    try:
        dataset = ds.dataset(str(manifest), format="parquet")
        names = set(dataset.schema.names)
        if "id" not in names:
            _warn(f"manifest 无 id 列，跳过: {manifest}")
            return out
        cols = [c for c in ("id", "audio", "source_path", "sha256") if c in names]
        # 单 id 用等值；多 id 用 isin
        id_list = sorted(target_ids)
        if len(id_list) == 1:
            filt = ds.field("id") == id_list[0]
        else:
            filt = ds.field("id").isin(id_list)
        table = dataset.to_table(columns=cols, filter=filt)
    except Exception as exc:  # noqa: BLE001
        _warn(f"pyarrow 读取失败，回退 pandas: {manifest} ({exc})")
        return _lookup_manifest_pandas(manifest, target_ids, path_maps=path_maps)

    for batch in table.to_batches():
        cols_map = {name: batch.column(name) for name in batch.schema.names}
        for i in range(batch.num_rows):
            sid = str(cols_map["id"][i].as_py()).strip()
            if sid not in target_ids:
                continue
            audio = _parse_audio(cols_map["audio"][i].as_py() if "audio" in cols_map else None)
            source_path = ""
            if "source_path" in cols_map and cols_map["source_path"][i].as_py() is not None:
                source_path = str(cols_map["source_path"][i].as_py()).strip()
            sha = ""
            if "sha256" in cols_map and cols_map["sha256"][i].as_py() is not None:
                sha = str(cols_map["sha256"][i].as_py()).strip()

            for key in AUDIO_KEY_ORDER:
                raw = audio.get(key)
                if not raw:
                    continue
                rewritten = rewrite_path(raw, path_maps)
                p = Path(rewritten)
                out[sid].append(
                    {
                        "kind": "manifest_audio",
                        "key": key,
                        "path": str(p),
                        "exists": p.is_file(),
                        "source": str(manifest),
                        "sha256": sha,
                    }
                )
            # 其余 audio key
            for key, raw in audio.items():
                if key in AUDIO_KEY_ORDER or not raw:
                    continue
                rewritten = rewrite_path(str(raw), path_maps)
                p = Path(rewritten)
                out[sid].append(
                    {
                        "kind": "manifest_audio",
                        "key": key,
                        "path": str(p),
                        "exists": p.is_file(),
                        "source": str(manifest),
                        "sha256": sha,
                    }
                )
            if source_path:
                rewritten = rewrite_path(source_path, path_maps)
                p = Path(rewritten)
                out[sid].append(
                    {
                        "kind": "manifest_source_path",
                        "key": "source_path",
                        "path": str(p),
                        "exists": p.is_file(),
                        "source": str(manifest),
                        "sha256": sha,
                    }
                )
    return out


def _lookup_manifest_pandas(
    manifest: Path,
    target_ids: set[str],
    *,
    path_maps: list[tuple[str, str]],
) -> dict[str, list[dict[str, Any]]]:
    import pandas as pd

    out: dict[str, list[dict[str, Any]]] = {sid: [] for sid in target_ids}
    df = pd.read_parquet(manifest, columns=None)
    if "id" not in df.columns:
        return out
    sub = df[df["id"].astype(str).str.strip().isin(target_ids)]
    for _, row in sub.iterrows():
        sid = str(row["id"]).strip()
        audio = _parse_audio(row["audio"] if "audio" in row else None)
        source_path = str(row["source_path"]).strip() if "source_path" in row and row.get("source_path") is not None else ""
        sha = str(row["sha256"]).strip() if "sha256" in row and row.get("sha256") is not None else ""
        for key in list(AUDIO_KEY_ORDER) + [k for k in audio if k not in AUDIO_KEY_ORDER]:
            raw = audio.get(key)
            if not raw:
                continue
            rewritten = rewrite_path(str(raw), path_maps)
            p = Path(rewritten)
            out[sid].append(
                {
                    "kind": "manifest_audio",
                    "key": key,
                    "path": str(p),
                    "exists": p.is_file(),
                    "source": str(manifest),
                    "sha256": sha,
                }
            )
        if source_path:
            rewritten = rewrite_path(source_path, path_maps)
            p = Path(rewritten)
            out[sid].append(
                {
                    "kind": "manifest_source_path",
                    "key": "source_path",
                    "path": str(p),
                    "exists": p.is_file(),
                    "source": str(manifest),
                    "sha256": sha,
                }
            )
    return out


def pick_best(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    # 优先：存在的 wav；再按 AUDIO_KEY_ORDER；派生 fs 次之；source_path 最后
    key_rank = {k: i for i, k in enumerate(AUDIO_KEY_ORDER)}

    def score(c: dict[str, Any]) -> tuple:
        exists = 0 if c.get("exists") else 1
        is_wav = 0 if str(c.get("path", "")).lower().endswith(".wav") else 1
        kind = str(c.get("kind") or "")
        kind_rank = 0 if kind == "manifest_audio" else (1 if kind == "derived_fs" else 2)
        kr = key_rank.get(str(c.get("key") or ""), 50)
        return (exists, is_wav, kind_rank, kr, str(c.get("path") or ""))

    return sorted(candidates, key=score)[0]


def discover_manifests(root: Path, *, batch: str | None) -> list[Path]:
    paths: list[Path] = []
    if batch:
        for rel in (
            f"datasets/stage1/cleaned/cleaned_{batch}.parquet",
            f"datasets/stage1/derived/classified_five_class_v3_{batch}.parquet",
        ):
            p = root / rel
            if p.is_file():
                paths.append(p.resolve())
        return paths

    for rel in MANIFEST_SCAN_DIRS:
        d = root / rel
        if not d.is_dir():
            continue
        paths.extend(sorted(d.glob("*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True))
    return paths


def dedupe_candidates(cands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for c in cands:
        key = (str(c.get("key") or ""), str(Path(c["path"]).resolve()) if c.get("exists") else str(c.get("path")))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="根据 fenshen-xxxx-xxxx（或其它 sample id）迅速定位 WAV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "ids",
        nargs="+",
        help="样本 id，如 fenshen-276cc46a-bf3387be（可多个）",
    )
    p.add_argument(
        "--batch",
        default=None,
        help="批次名，自动读 cleaned_{batch}.parquet 等",
    )
    p.add_argument(
        "--manifest",
        nargs="+",
        default=None,
        help="显式 manifest parquet（可多个）",
    )
    p.add_argument(
        "--scan-manifests",
        action="store_true",
        help="未指定 --manifest/--batch 时，扫描 datasets/stage1 下 parquet（较慢）",
    )
    p.add_argument(
        "--derived-dir",
        action="append",
        default=None,
        help="派生音频根目录，可重复；默认 pcm_to_wav / resample_16k / vad_segments",
    )
    p.add_argument(
        "--path-map",
        action="append",
        default=None,
        help="路径前缀重写 OLD=NEW，可重复（本机/服务器前缀不一致时用）",
    )
    p.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="每个 id 只打印一行首选存在路径（未找到打印空行到 stderr 并非 0 退出）",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="输出完整 JSON",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="打印全部候选，不只首选",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        path_maps = parse_path_maps(args.path_map)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    ids = [normalize_id(x) for x in args.ids]
    ids = [x for x in ids if x]
    if not ids:
        print("[ERROR] 未提供有效 id", file=sys.stderr)
        return 1
    target = set(ids)

    if args.derived_dir:
        derived_dirs = [resolve_under_root(d, root=_ROOT) for d in args.derived_dir]
    else:
        derived_dirs = [resolve_under_root(d, root=_ROOT) for d in DEFAULT_DERIVED]

    # 1) 派生目录 glob（通常最快）
    by_id: dict[str, list[dict[str, Any]]] = {sid: [] for sid in ids}
    for sid in ids:
        by_id[sid].extend(glob_derived(sid, derived_dirs))

    # 2) Manifest
    manifests: list[Path] = []
    if args.manifest:
        for raw in args.manifest:
            manifests.append(resolve_under_root(raw, root=_ROOT))
    elif args.batch:
        manifests.extend(discover_manifests(_ROOT, batch=args.batch))
    elif args.scan_manifests:
        manifests.extend(discover_manifests(_ROOT, batch=None))

    # 去重保序
    seen_m: set[Path] = set()
    uniq_manifests: list[Path] = []
    for m in manifests:
        key = m.resolve() if m.exists() else m
        if key in seen_m:
            continue
        seen_m.add(key)
        uniq_manifests.append(m)

    remaining = set(target)
    for manifest in uniq_manifests:
        if not remaining and not args.all:
            # 仍可继续补全候选；若只要定位且已有存在路径可提前结束
            if all(any(c.get("exists") for c in by_id[s]) for s in ids):
                break
        if not manifest.is_file():
            _warn(f"manifest 不存在: {manifest}")
            continue
        if not args.quiet:
            _info(f"查 manifest: {manifest}")
        found = lookup_manifest(manifest, remaining if not args.all else target, path_maps=path_maps)
        for sid, cands in found.items():
            if not cands:
                continue
            by_id[sid].extend(cands)
            if any(c.get("exists") and str(c.get("path", "")).lower().endswith(".wav") for c in cands):
                remaining.discard(sid)

    # 汇总
    report: dict[str, Any] = {"ids": [], "missing": []}
    exit_missing = False
    for sid in ids:
        cands = dedupe_candidates(by_id[sid])
        best = pick_best(cands)
        row = {
            "id": sid,
            "best": best,
            "candidates": cands if args.all or args.json else ([best] if best else []),
        }
        report["ids"].append(row)
        if not best or not best.get("exists"):
            report["missing"].append(sid)
            exit_missing = True

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1 if exit_missing else 0

    if args.quiet:
        for row in report["ids"]:
            best = row["best"]
            if best and best.get("exists"):
                print(best["path"])
            else:
                print(f"[MISS] {row['id']}", file=sys.stderr)
                print("")
        return 1 if exit_missing else 0

    for row in report["ids"]:
        sid = row["id"]
        best = row["best"]
        print("=" * 60)
        print(f"id: {sid}")
        if best and best.get("exists"):
            print(f"WAV: {best['path']}")
            print(f"     key={best.get('key')}  via={best.get('kind')}  from={best.get('source')}")
        elif best:
            print(f"记录到路径但文件不存在: {best['path']}")
            print(f"     key={best.get('key')}  from={best.get('source')}")
            print("     提示: 用 --path-map OLD=NEW 改前缀，或检查挂载")
        else:
            print("未找到任何路径")
            print("     提示: 加 --batch <name> / --manifest <parquet> / --scan-manifests")

        show = row["candidates"] if args.all else []
        if args.all and len(show) > 1:
            print("全部候选:")
            for c in show:
                mark = "OK" if c.get("exists") else "MISS"
                print(f"  [{mark}] {c.get('key'):16s}  {c['path']}")

    if report["missing"]:
        print("=" * 60)
        _warn(f"未定位到可读 WAV: {', '.join(report['missing'])}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
