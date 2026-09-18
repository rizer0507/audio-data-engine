"""Detached worker entry: python -m audio_engine.core.stage1.worker <job_root> [...]."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="audio_engine.core.stage1.worker")
    parser.add_argument("job_root", type=Path)
    parser.add_argument(
        "--mode",
        choices=("run", "resume", "retry"),
        default="run",
        help="run=首次贯通；resume=从检查点续跑；retry=精准补跑",
    )
    parser.add_argument("--family", default=None, help="retry 限定家族")
    parser.add_argument("--run", default=None, help="retry 限定路次编号，如 1/2")
    parser.add_argument(
        "--failed-only",
        dest="failed_only",
        action="store_true",
        default=True,
        help="只重置失败/缺失阶段（默认）",
    )
    parser.add_argument(
        "--all",
        dest="failed_only",
        action="store_false",
        help="重置目标范围内全部阶段",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="仅恢复 export/reconcile，绝不重跑 ASR",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    from audio_engine.core.stage1.orchestrator import Stage1Orchestrator

    orchestrator = Stage1Orchestrator(args.job_root)
    if args.mode == "resume":
        state = orchestrator.resume()
    elif args.mode == "retry":
        state = orchestrator.retry(
            family=args.family,
            run=args.run,
            failed_only=bool(args.failed_only),
            export_only=bool(args.export_only),
        )
    else:
        state = orchestrator.run(mode="run")

    if state.status == "succeeded":
        return 0
    if state.status == "needs_attention":
        return 3
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
