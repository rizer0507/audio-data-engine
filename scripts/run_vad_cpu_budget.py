#!/usr/bin/env python3
"""Linux: launch VAD batches in one CPU set, or limit existing process trees.

    python scripts/run_vad_cpu_budget.py BATCH_A BATCH_B
    python scripts/run_vad_cpu_budget.py --attach-pids 1234 5678

The default 30% budget selects 38 shared logical CPUs on a 128-CPU host.
This bounds CPU capacity for these jobs, not unrelated host processes.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import signal
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_WAIT_POLICY": "PASSIVE",
    "KMP_BLOCKTIME": "0",
}


def cpu_set(total: int, allowed: set[int], percent: float) -> set[int]:
    if not math.isfinite(percent) or not 0 < percent <= 100:
        raise ValueError("cpu-percent must be greater than 0 and at most 100")
    count = min(len(allowed), math.floor(total * percent / 100))
    if count < 1:
        raise ValueError("Budget is below one logical CPU; use a cgroup CPU quota instead")
    return set(sorted(allowed)[:count])


def process_tree(roots: list[int], proc: Path = Path("/proc")) -> list[int]:
    children: dict[int, list[int]] = {}
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            # comm can contain spaces and parentheses: split after its final ')'.
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            children.setdefault(int(fields[1]), []).append(int(entry.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
    ordered: list[int] = []
    seen: set[int] = set()
    queue = list(roots)
    while queue:
        pid = queue.pop(0)
        if pid in seen:
            continue
        seen.add(pid)
        ordered.append(pid)
        queue.extend(children.get(pid, []))
    return ordered


def attach(roots: list[int], cpus: set[int]) -> None:
    for pid in roots:
        if pid <= 1 or pid == os.getpid():
            raise ValueError("Specify VAD pipeline parent PIDs greater than 1")
        os.kill(pid, 0)  # Validate every root before changing any affinity.
    changed: set[int] = set()
    # Restrict parents first, then rescan for children forked during the first pass.
    # New threads/processes inherit the creator's affinity.
    for _ in range(2):
        for pid in process_tree(roots):
            try:
                for task in Path(f"/proc/{pid}/task").iterdir():
                    try:
                        os.sched_setaffinity(int(task.name), cpus)
                        changed.add(pid)
                    except ProcessLookupError:
                        pass
            except (FileNotFoundError, ProcessLookupError):
                pass
    print(f"Limited {len(changed)} processes, including all their existing threads.", flush=True)
    print("Existing model thread counts are unchanged; ncpu=1 requires a new model load.")


def launch(batches: list[str], resumes: list[str], cpus: set[int]) -> int:
    executable = shutil.which("audio-data")
    if not executable:
        raise RuntimeError("audio-data is missing; activate the project's Python environment")
    env = os.environ.copy()
    env.update(THREAD_ENV)
    os.sched_setaffinity(0, cpus)
    processes: list[subprocess.Popen] = []
    try:
        for index, batch in enumerate(batches):
            command = [executable, "pipeline", "run", "pipelines/vad_segment_batch.yaml",
                       "--source-name", batch]
            if resumes:
                command.extend(["--resume", resumes[index]])
            processes.append(subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True))
            print(f"Started batch={batch} parent_pid={processes[-1].pid}", flush=True)
        codes = [process.wait() for process in processes]
        return 1 if any(codes) else 0
    finally:
        # Each batch has its own session; cleanup includes its shard descendants.
        for process in processes:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batches", nargs="*")
    parser.add_argument("--cpu-percent", type=float, default=30.0)
    parser.add_argument("--attach-pids", type=int, nargs="+")
    parser.add_argument("--resume", nargs="+", default=[], help="One existing run per batch, in order")
    args = parser.parse_args()
    if not hasattr(os, "sched_setaffinity"):
        parser.error("CPU affinity control requires Linux")
    if bool(args.batches) == bool(args.attach_pids):
        parser.error("Specify batch names OR --attach-pids")
    if len(set(args.batches)) != len(args.batches):
        parser.error("Batch names must be distinct")
    if args.resume and (args.attach_pids or len(args.resume) != len(args.batches)):
        parser.error("--resume needs one run directory per batch")
    total = os.cpu_count() or 1
    try:
        cpus = cpu_set(total, os.sched_getaffinity(0), args.cpu_percent)
        print(f"Shared CPU budget: {len(cpus)}/{total} = {100 * len(cpus) / total:.2f}% "
              f"of host capacity; CPUs={','.join(map(str, sorted(cpus)))}", flush=True)
        if args.attach_pids:
            attach(args.attach_pids, cpus)
            return 0
        return launch(args.batches, args.resume, cpus)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"CPU budget failed: {exc}\n")
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
