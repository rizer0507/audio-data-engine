#!/usr/bin/env python3
"""把历史 fenshen_1015 产物名改成主题 slug fenshen。

配合新评测入口::

  bash scripts/run_asr_eval.sh fenshen <mode> <v0xx>

默认只扫::

  datasets/stage1/derived
  datasets/stage3/asr

规则：路径名中的 ``fenshen_1015`` 替换为 ``fenshen``（文件与目录）。
默认 dry-run；加 ``--apply`` 才真正改名。目标已存在则记入冲突并跳过。

服务器示例::

  cd /path/to/audio-data-engine
  python scripts/rename_fenshen_1015_theme_artifacts.py
  python scripts/rename_fenshen_1015_theme_artifacts.py --apply
  python scripts/rename_fenshen_1015_theme_artifacts.py --apply --include-runs
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

OLD = "fenshen_1015"
NEW = "fenshen"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _plan_renames(roots: list[Path]) -> list[tuple[Path, Path]]:
    """深度优先：先文件后目录，深层路径优先，避免父目录先改导致子路径失效。"""
    candidates: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if OLD in path.name:
                candidates.append(path)

    files = [p for p in candidates if p.is_file()]
    dirs = [p for p in candidates if p.is_dir()]
    files.sort(key=lambda p: len(p.parts), reverse=True)
    dirs.sort(key=lambda p: len(p.parts), reverse=True)

    planned: list[tuple[Path, Path]] = []
    seen_src: set[Path] = set()
    for path in files + dirs:
        if path in seen_src:
            continue
        seen_src.add(path)
        target = path.with_name(path.name.replace(OLD, NEW))
        if target == path:
            continue
        planned.append((path, target))
    return planned


def _resolve_roots(repo: Path, include_runs: bool) -> list[Path]:
    roots = [
        repo / "datasets" / "stage1" / "derived",
        repo / "datasets" / "stage3" / "asr",
    ]
    if include_runs:
        roots.append(repo / "runs")
    return roots


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="批量将产物名中的 fenshen_1015 改为 fenshen"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="audio-data-engine 仓库根（默认脚本上级）",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="真正执行改名；默认只预览",
    )
    parser.add_argument(
        "--include-runs",
        action="store_true",
        help="同时处理 runs/ 下含 fenshen_1015 的目录与文件",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="写出 JSON 报告路径；默认 tmp/rename_fenshen_1015_theme_<stamp>.json",
    )
    args = parser.parse_args(argv)

    repo = args.root.resolve()
    if not repo.is_dir():
        print(f"[ERROR] 根目录不存在: {repo}", file=sys.stderr)
        return 2

    roots = _resolve_roots(repo, args.include_runs)
    missing_roots = [str(p) for p in roots if not p.is_dir()]
    planned = _plan_renames([p for p in roots if p.is_dir()])

    conflicts: list[dict[str, str]] = []
    ready: list[tuple[Path, Path]] = []
    for src, dst in planned:
        if dst.exists():
            conflicts.append({"src": str(src), "dst": str(dst), "reason": "target_exists"})
            continue
        ready.append((src, dst))

    mode = "apply" if args.apply else "dry-run"
    print(f"[INFO] root={repo}")
    print(f"[INFO] mode={mode}")
    print(f"[INFO] roots={[str(p) for p in roots]}")
    if missing_roots:
        print(f"[WARN] 缺少目录（已跳过）: {missing_roots}")
    print(f"[INFO] planned={len(planned)} ready={len(ready)} conflicts={len(conflicts)}")

    for src, dst in ready:
        rel_src = src.relative_to(repo) if src.is_relative_to(repo) else src
        rel_dst = dst.relative_to(repo) if dst.is_relative_to(repo) else dst
        print(f"{'RENAME' if args.apply else 'DRY'}  {rel_src.as_posix()}  ->  {rel_dst.as_posix()}")

    for item in conflicts:
        print(f"SKIP  {item['src']}  ->  {item['dst']}  ({item['reason']})")

    done: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    if args.apply:
        for src, dst in ready:
            try:
                src.rename(dst)
                done.append({"src": str(src), "dst": str(dst)})
            except OSError as exc:
                errors.append({"src": str(src), "dst": str(dst), "error": str(exc)})
                print(f"[ERROR] {src} -> {dst}: {exc}", file=sys.stderr)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = args.report
    if report_path is None:
        report_dir = repo / "tmp"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"rename_fenshen_1015_theme_{stamp}.json"
    else:
        report_path = report_path.resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "created_at": _now_iso(),
        "root": str(repo),
        "mode": mode,
        "old": OLD,
        "new": NEW,
        "include_runs": bool(args.include_runs),
        "roots": [str(p) for p in roots],
        "missing_roots": missing_roots,
        "planned_count": len(planned),
        "ready_count": len(ready),
        "conflict_count": len(conflicts),
        "done_count": len(done),
        "error_count": len(errors),
        "ready": [{"src": str(s), "dst": str(d)} for s, d in ready],
        "conflicts": conflicts,
        "done": done,
        "errors": errors,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] report {report_path}")

    if conflicts and not args.apply:
        print("[WARN] 存在目标冲突；解决后再 --apply")
    if errors:
        return 1
    if args.apply and conflicts and not done:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
