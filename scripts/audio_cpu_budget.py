#!/usr/bin/env python3
"""Linux cgroup v2: one CPU quota shared by all VAD/DNSMOS batches.

sudo python scripts/audio_cpu_budget.py configure
sudo python scripts/audio_cpu_budget.py attach PID_A PID_B
sudo -E python scripts/audio_cpu_budget.py run -- /venv/bin/audio-data pipeline run ...

Run mode drops sudo privileges before executing the actual pipeline.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

from run_vad_cpu_budget import THREAD_ENV, process_tree

CGROUP_ROOT = Path("/sys/fs/cgroup")
GROUP = "audio-engine-cpu"
PERIOD_US = 100_000


def quota_for_host(total: int, percent: float = 30) -> tuple[int, int]:
    if not math.isfinite(percent) or not 0 < percent <= 30:
        raise ValueError("CPU budget must be greater than 0 and at most 30 percent")
    cores = math.floor(total * percent / 100)
    if cores / total >= percent / 100:
        cores -= 1
    if cores < 1:
        raise ValueError("This host needs a fractional CPU quota; at least four logical CPUs required")
    return cores * PERIOD_US, PERIOD_US


def read_quota(group: Path, total: int) -> tuple[int, int]:
    try:
        quota, period = (group / "cpu.max").read_text().split()
    except FileNotFoundError as exc:
        raise RuntimeError("Shared cgroup is missing; run configure with sudo first") from exc
    if quota == "max":
        raise RuntimeError("Shared cgroup has no CPU limit")
    quota_i, period_i = int(quota), int(period)
    if quota_i <= 0 or period_i <= 0 or quota_i / period_i / total >= 0.30:
        raise RuntimeError("Shared CPU quota is not below 30% of this host")
    burst = group / "cpu.max.burst"
    if burst.is_file() and int(burst.read_text()) != 0:
        raise RuntimeError("Shared CPU cgroup has a nonzero burst allowance; re-run configure")
    return quota_i, period_i


def configure(total: int, percent: float) -> Path:
    quota, period = quota_for_host(total, percent)
    controllers = CGROUP_ROOT / "cgroup.controllers"
    if not controllers.is_file() or "cpu" not in controllers.read_text().split():
        raise RuntimeError("cgroup v2 CPU controller is unavailable; no unenforced fallback is used")
    subtree = CGROUP_ROOT / "cgroup.subtree_control"
    if "cpu" not in subtree.read_text().split():
        subtree.write_text("+cpu")
    group = CGROUP_ROOT / GROUP
    group.mkdir(exist_ok=True)
    if not (group / "cpu.max").is_file():
        raise RuntimeError("cpu.max was not created; check CPU controller delegation")
    (group / "cpu.max").write_text(f"{quota} {period}")
    # New groups default to zero; also reject stale configured burst allowance.
    burst = group / "cpu.max.burst"
    if burst.is_file():
        burst.write_text("0")
    read_quota(group, total)
    print(f"Shared quota: {quota / period:g}/{total} CPU equivalents "
          f"= {100 * quota / period / total:.4f}% at {group}", flush=True)
    return group


def attach(group: Path, roots: list[int]) -> int:
    for pid in roots:
        if pid <= 1 or pid == os.getpid():
            raise ValueError("Specify only audio pipeline parent PIDs greater than 1")
        os.kill(pid, 0)
    moved: set[int] = set()
    for _ in range(2):
        for pid in process_tree(roots):
            try:
                # One PID per write; the kernel moves every thread in the process.
                (group / "cgroup.procs").write_text(str(pid))
                moved.add(pid)
            except ProcessLookupError:
                pass
    return len(moved)


def run(group: Path, command: list[str]) -> None:
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("run requires a command after --")
    # Avoid accidentally running the scoring job as root when sudo was omitted.
    uid = int(os.environ.get("SUDO_UID", "0"))
    if os.geteuid() == 0 and uid <= 0:
        raise ValueError("Use sudo from the normal audio-engine account; root-only launches are refused")
    (group / "cgroup.procs").write_text(str(os.getpid()))
    env = os.environ.copy()
    env.update(THREAD_ENV)
    if os.geteuid() == 0:
        import pwd

        account = pwd.getpwuid(uid)
        os.initgroups(account.pw_name, account.pw_gid)
        os.setgid(account.pw_gid)
        os.setuid(uid)
        env.update(HOME=account.pw_dir, USER=account.pw_name, LOGNAME=account.pw_name)
        for key in ("SUDO_UID", "SUDO_GID", "SUDO_USER", "SUDO_COMMAND"):
            env.pop(key, None)
    os.execvpe(command[0], command, env)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cpu-percent", type=float, default=30)
    commands = parser.add_subparsers(dest="mode", required=True)
    commands.add_parser("configure")
    attach_parser = commands.add_parser("attach")
    attach_parser.add_argument("pids", nargs="+", type=int)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("This command requires Linux cgroup v2")
    try:
        total = os.cpu_count() or 1
        if args.mode == "configure":
            configure(total, args.cpu_percent)
        else:
            group = CGROUP_ROOT / GROUP
            quota, period = read_quota(group, total)
            print(f"Using shared CPU budget {100 * quota / period / total:.4f}%", flush=True)
            if args.mode == "attach":
                print(f"Moved {attach(group, args.pids)} processes into shared CPU budget")
            else:
                run(group, args.command)
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"CPU quota failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
