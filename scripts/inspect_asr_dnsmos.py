#!/usr/bin/env python3
"""只读检查 Linux 上已有的 pipeline 进程及其落盘进度，不启动/停止任务。"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path


def option(argv: list[str], name: str) -> str | None:
    for i, token in enumerate(argv):
        if token == name and i + 1 < len(argv):
            return argv[i + 1]
        if token.startswith(name + "="):
            return token.split("=", 1)[1]
    return None


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def tail(path: Path, size: int = 65536) -> list[str]:
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            length = stream.tell()
            stream.seek(max(0, length - size))
            data = stream.read().decode("utf-8", errors="replace")
        return data.splitlines()
    except OSError:
        return []


def run_root(path: Path) -> Path:
    # shard 子进程的 --resume 指向 <root>/shard-000，归并到父 run。
    return path.parent if re.fullmatch(r"shard-\d+", path.name) else path


def discover(project: Path, batches: list[str], proc_root: Path = Path("/proc")) -> list[dict]:
    processes = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = [
                s.decode("utf-8", errors="replace")
                for s in (entry / "cmdline").read_bytes().split(b"\0")
                if s
            ]
            if not any(argv[i : i + 2] == ["pipeline", "run"] for i in range(len(argv) - 1)):
                continue
            batch = option(argv, "--source-name")
            if batches and batch not in batches:
                continue
            cwd = (entry / "cwd").resolve(strict=True)
            if cwd != project and not cwd.is_relative_to(project):
                continue
            roots = set()
            resume = option(argv, "--resume")
            if resume:
                path = Path(resume)
                if not path.is_absolute():
                    path = cwd / path
                    if not path.exists():
                        path = cwd / "runs" / resume
                if path.is_dir():
                    roots.add(run_root(path.resolve()))
            for fd in (entry / "fd").iterdir():
                try:
                    target = Path(os.readlink(fd))
                except OSError:
                    continue
                if target.name == "run.log" or re.fullmatch(r"shard-\d+\.log", target.name):
                    roots.add(run_root(target.parent))
            gpu = None
            try:
                for item in (entry / "environ").read_bytes().split(b"\0"):
                    if item.startswith(b"CUDA_VISIBLE_DEVICES="):
                        gpu = item.split(b"=", 1)[1].decode(errors="replace")
            except OSError:
                pass
            pipeline = next((arg for arg in argv if arg.endswith(".yaml")), None)
            processes.append(
                {
                    "pid": int(entry.name),
                    "batch": batch,
                    "asr_run": option(argv, "--asr-run"),
                    "pipeline": pipeline,
                    "cuda_visible_devices": gpu,
                    "run_dirs": sorted(str(root) for root in roots),
                }
            )
        except (OSError, ValueError):
            # 任务退出或无 /proc 读取权限；不推断为失败或完成。
            continue
    return sorted(processes, key=lambda item: item["pid"])


def inspect_run(root: Path) -> dict:
    result = {"run_dir": str(root), "exists": root.is_dir()}
    if not root.is_dir():
        return result
    progress_path = root / "progress.json"
    progress = read_json(progress_path)
    if isinstance(progress, dict):
        updated = progress.get("updated_at")
        result["sharded_progress"] = {
            **progress,
            "source": str(progress_path),
            "age_seconds": round(max(0, time.time() - updated), 1)
            if isinstance(updated, (int, float))
            else None,
            "meaning": "分片聚合落盘进度；done 包含已处理的失败样本，不等于成功条数",
        }
    logs = [root / "run.log", root / "PROGRESS.log"]
    lines = []
    for log in logs:
        recent = tail(log)
        found = [line for line in recent if "[PROGRESS]" in line]
        if found:
            lines.append(
                {
                    "source": str(log),
                    "last_progress_lines": found[-3:],
                    "log_age_seconds": round(max(0, time.time() - log.stat().st_mtime), 1),
                }
            )
    result["progress_logs"] = lines
    checkpoints = []
    paths = list((root / "checkpoints").glob("*/_state.json"))
    paths += list(root.glob("shard-*/checkpoints/*/_state.json"))
    for path in sorted(paths):
        state = read_json(path)
        if not isinstance(state, dict):
            continue
        recorded = state.get("parts") or []
        parts = []
        for part in recorded:
            if not (path.parent / part.get("file", "missing")).is_file():
                break
            parts.append(part)
        checkpoints.append(
            {
                "source": str(path),
                "complete": bool(state.get("complete")) and len(parts) == len(recorded),
                "committed_inputs": sum(int(part.get("count_in", 0)) for part in parts),
                "failed": sum(int((part.get("metrics") or {}).get("failed", 0)) for part in parts),
                "missing_or_unusable_parts": len(recorded) - len(parts),
                "age_seconds": round(max(0, time.time() - path.stat().st_mtime), 1),
            }
        )
    result["checkpoints"] = checkpoints
    metrics = read_json(root / "metrics.json")
    if isinstance(metrics, dict):
        result["final_metrics"] = metrics
    result["note"] = (
        "各 checkpoint 是各步骤的已提交数量，不可跨步骤相加；尚未提交的当前块不计入。"
        "记录存在或百分比 100% 都不能单独证明整个任务成功。"
    )
    return result


def snapshot(project: Path, batches: list[str], explicit_runs: list[Path]) -> dict:
    processes = discover(project.resolve(), batches)
    roots = {Path(path) for proc in processes for path in proc["run_dirs"]}
    roots.update(path.resolve() for path in explicit_runs)
    return {
        "observed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "project_root": str(project.resolve()),
        "requested_batches": batches,
        "processes": processes,
        "runs": [inspect_run(path) for path in sorted(roots)],
        "missing_batches": [
            batch for batch in batches if not any(proc["batch"] == batch for proc in processes)
        ],
        "note": "只读快照。未发现进程可能是已退出、无权限或批次参数不匹配，不能据此认定成功。"
        "没有可靠的进程/run 对应证据时，不以历史目录猜测当前进度；可用 --run-dir 指定。",
    }


def format_snapshot(data: dict) -> str:
    lines = [f"检查时间: {data['observed_at']}（服务器时区）"]
    for item in data.get("completed_tasks", []):
        warning = (
            f"（含 {item['failed_operations']} 次样本失败）"
            if item.get("failed_operations")
            else ""
        )
        lines.append(f"已核验完成: {item['task']}{warning}；run={item['run_dir']}")
    for item in data.get("unresolved_tasks", []):
        lines.append(f"尚未核验: {item['task']}；{item['reason']}")
    for proc in data["processes"]:
        lines.append(
            f"PID={proc['pid']} batch={proc['batch']} asr_run={proc['asr_run'] or '-'} "
            f"CUDA_VISIBLE_DEVICES={proc['cuda_visible_devices']} pipeline={proc['pipeline']}"
        )
        if not proc["run_dirs"]:
            lines.append("  找到进程，但尚未确定 run 目录；可用 --run-dir 显式指定。")
    for run in data["runs"]:
        lines.append(f"\nRun: {run['run_dir']}")
        progress = run.get("sharded_progress")
        if progress:
            lines.append(
                f"  分片已处理: {progress.get('done')}/{progress.get('total')} "
                f"({progress.get('pct')}%)；最近吞吐: {progress.get('rate_window')} 条/秒；"
                f"预计剩余: {progress.get('eta_s')} 秒；记录距今: {progress['age_seconds']} 秒"
            )
        for log in run.get("progress_logs", []):
            lines.append(f"  日志（文件距今 {log['log_age_seconds']} 秒）: {log['source']}")
            lines.append("    " + log["last_progress_lines"][-1])
        for checkpoint in run.get("checkpoints", []):
            name = str(Path(checkpoint["source"]).relative_to(Path(run["run_dir"])))
            lines.append(
                f"  {name}: 已提交={checkpoint['committed_inputs']} "
                f"失败={checkpoint['failed']} 步骤完成={checkpoint['complete']} "
                f"不可用块={checkpoint['missing_or_unusable_parts']}"
            )
        if not progress and not run.get("progress_logs") and not run.get("checkpoints"):
            lines.append("  暂无可读取的进度记录，不能据此判断是否卡住或完成。")
        if run.get("final_metrics"):
            metrics = run["final_metrics"]
            lines.append(f"  最终统计: total={metrics.get('total')} failed={metrics.get('failed')}")
    if data["missing_batches"]:
        lines.append(
            "\n未匹配到运行进程的批次（已核验完成的任务见上方）: "
            + ", ".join(data["missing_batches"])
        )
    lines.append("\n" + data["note"])
    lines.append("进度以落盘记录为准；已处理数包含失败，当前未提交块会滞后，各步骤不可相加。")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--batch", action="append", default=[])
    parser.add_argument("--run-dir", type=Path, action="append", default=[])
    parser.add_argument("--json", action="store_true", help="输出完整 JSON，方便保存或程序读取")
    args = parser.parse_args(argv)
    if not sys.platform.startswith("linux"):
        parser.error("请在实际运行任务的 Linux 服务器执行 inspect")
    data = snapshot(args.project_root, args.batch, args.run_dir)
    print(json.dumps(data, ensure_ascii=False, indent=2) if args.json else format_snapshot(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
