#!/usr/bin/env python3
"""Linux GPU 5 串行 ASR、CPU 并行 DNSMOS；新服务/任务使用独立 tmux。

在 conda activate asr 后运行，start 脱离 SSH，run 前台调试。
python scripts/run_asr_dnsmos.py --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("qwen", "sensevoice", "glm", "dnsmos")
LOG = logging.getLogger("asr-dnsmos")


def save_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    batches = cfg["batches"]
    names = set()
    if not batches:
        raise ValueError("batches 不能为空")
    for batch in batches:
        name = batch["name"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) or name in names:
            raise ValueError(f"批次名非法或重复: {name}")
        names.add(name)
        if not re.fullmatch(r"[0-9]+", str(batch["gpu"])):
            raise ValueError("gpu 必须是物理 GPU 数字编号")
        if str(batch["gpu"]) != "5":
            raise ValueError("当前服务器仅允许物理 GPU 5；所有 batch 的 gpu 必须为 5")
    cfg.setdefault("dnsmos_parallel", True)
    cfg.setdefault("existing_tasks", {})
    for key, entry in cfg["existing_tasks"].items():
        batch, separator, stage = key.partition("/")
        if not separator or batch not in names or stage not in STAGES or not entry.get("run_dir"):
            raise ValueError(f"existing_tasks 配置非法: {key}")
    for key, default in (("ready_timeout", 900), ("cleanup_timeout", 120), ("heartbeat", 30)):
        cfg.setdefault(key, default)
        if not isinstance(cfg[key], (int, float)) or cfg[key] <= 0:
            raise ValueError(f"{key} 必须大于 0")
    cfg.setdefault("gpu_idle_memory_mb", 1024)
    if cfg["gpu_idle_memory_mb"] < 0:
        raise ValueError("gpu_idle_memory_mb 不能小于 0")
    for key in ("qwen_port", "glm_port"):
        if not isinstance(cfg[key], int) or not 1 <= cfg[key] <= 65535:
            raise ValueError(f"{key} 不是有效端口")
    if cfg["qwen_port"] == cfg["glm_port"]:
        raise ValueError("Qwen 和 GLM 端口不能相同")
    return cfg


def commands(cfg: dict, batch: dict, stage: str) -> tuple[list[str], list[str] | None, dict]:
    env = dict(os.environ)
    env.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
        CUDA_VISIBLE_DEVICES=str(batch["gpu"]),
        BATCH=batch["name"],
        SOURCE_DIR=batch.get("source_dir", ""),
        QWEN_ASR_API_BASE=f"http://127.0.0.1:{cfg['qwen_port']}",
        QWEN_ASR_API_KEY="dummy",
        QWEN_ASR_MODEL="qwen3-asr",
        GLM_ASR_API_BASE=f"http://127.0.0.1:{cfg['glm_port']}",
        GLM_ASR_MODEL="glm-asr",
        GLM_ASR_API_KEY="dummy",
        SENSEVOICE_MODEL_PATH=cfg["sensevoice_model"],
    )
    # 多副本变量优先级更高，避免继承旧终端设置而请求错误的服务。
    env.pop("GLM_ASR_API_BASES", None)
    pipeline = (
        f"pipelines/{stage}_asr_batch.yaml"
        if stage != "dnsmos"
        else ("pipelines/dnsmos_score_batch.yaml")
    )
    client = ["audio-data", "pipeline", "run", pipeline, "--source-name", batch["name"]]
    service = None
    if stage != "dnsmos":
        client += ["--asr-run", f"{stage}1", "--no-sharding"]
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""  # DNSMOS 只走 CPU，保留 YAML 的分片配置。
    if stage == "qwen":
        service = [
            "vllm",
            "serve",
            cfg["qwen_model"],
            "--gpu-memory-utilization",
            "0.5",
            "--host",
            "127.0.0.1",
            "--port",
            str(cfg["qwen_port"]),
            "--tensor-parallel-size",
            "1",
            "--served-model-name",
            "qwen3-asr",
            "--chat-template",
            cfg["qwen_template"],
        ]
    if stage == "glm":
        home = PurePosixPath(cfg["glm_env"])
        service = [
            str(home / "bin/python"),
            str(home / "bin/vllm"),
            "serve",
            cfg["glm_model"],
            "--host",
            "0.0.0.0",
            "--port",
            str(cfg["glm_port"]),
            "--served-model-name",
            "glm-asr",
            "--tensor-parallel-size",
            "1",
            "--trust-remote-code",
            "--dtype",
            "bfloat16",
            "--max-model-len",
            "4096",
            "--max-num-seqs",
            "8",
            "--gpu-memory-utilization",
            "0.90",
            "--limit-mm-per-prompt",
            '{"audio":1}',
            "--no-enable-flashinfer-autotune",
            "--kernel-config",
            '{"enable_jit_warmup":false,"enable_cutedsl_warmup":false}',
        ]
    return client, service, env


def service_env(env: dict, stage: str) -> dict:
    result = dict(env)
    if stage == "glm":
        result.update(
            PYTHONNOUSERSITE="1",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            FLASHINFER_DISABLE_VERSION_CHECK="1",
            VLLM_USE_FLASHINFER_SAMPLER="0",
        )
        for key in (
            "VLLM_ATTENTION_BACKEND",
            "VLLM_USE_FLASHINFER_SAMPLE",
            "VLLM_ATTENTIOIN_BACKEND",
        ):
            result.pop(key, None)
    return result


def process_token(pid: int) -> str | None:
    """Linux starttime 区分 PID 复用；僵尸进程视为已结束。"""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (OSError, IndexError):
        return None


def is_running(state: dict) -> bool:
    return bool(state.get("process_token")) and (
        process_token(state["pid"]) == state["process_token"]
    )


def live_tasks() -> list[dict]:
    from inspect_asr_dnsmos import discover

    return discover(ROOT, [])


def matching_tasks(processes: list[dict], batch: str, stage: str) -> list[dict]:
    filename = f"{stage}_asr_batch.yaml" if stage != "dnsmos" else "dnsmos_score_batch.yaml"
    return [
        proc
        for proc in processes
        if proc["batch"] == batch
        and Path(proc.get("pipeline") or "").name == filename
        and (stage == "dnsmos" or proc.get("asr_run") == f"{stage}1")
    ]


def proc_identity(pid: int) -> dict:
    token = process_token(pid)
    if token is None:
        raise RuntimeError(f"进程 {pid} 已退出，无法绑定；请显式指定对应 run_dir 后重试")
    return {"pid": pid, "process_token": token}


def find_existing_service(
    cfg: dict, stage: str, gpu: int, *, allow_missing: bool = False
) -> dict | None:
    """只绑定 GPU、模型、端口和 served-model-name 均匹配的手工 vLLM。"""
    from inspect_asr_dnsmos import option

    found = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            argv = [
                s.decode(errors="replace")
                for s in (path / "cmdline").read_bytes().split(b"\0")
                if s
            ]
            env = (path / "environ").read_bytes().split(b"\0")
            if (
                "serve" in argv
                and cfg[f"{stage}_model"] in argv
                and option(argv, "--port") == str(cfg[f"{stage}_port"])
                and option(argv, "--served-model-name")
                == ("qwen3-asr" if stage == "qwen" else "glm-asr")
                and f"CUDA_VISIBLE_DEVICES={gpu}".encode() in env
            ):
                found.append(proc_identity(int(path.name)))
        except (OSError, RuntimeError):
            continue
    if not found and allow_missing:
        return None
    if len(found) != 1:
        raise RuntimeError(
            f"无法唯一绑定 GPU {gpu} 的 {stage} vLLM（找到 {len(found)} 个），拒绝猜测或误杀"
        )
    return found[0]


def task_output(batch: str, stage: str) -> tuple[str, Path]:
    if stage == "dnsmos":
        return (
            f"dnsmos_score_batch_{batch}",
            ROOT / f"datasets/stage1/quality/dnsmos_{batch}.parquet",
        )
    return (
        f"{stage}_asr_batch_{stage}1_{batch}",
        ROOT / f"datasets/stage1/asr/{stage}1_asr_{batch}.parquet",
    )


def verify_completed_run(root: Path, batch: str, stage: str) -> None:
    """已退出任务必须有正确身份和当前正式产物，不能仅凭目录名判成功。"""
    pipeline, expected = task_output(batch, stage)
    entry = {"run_dir": str(root), "processes": [], "artifact_before": None}
    existing_finished(entry)
    artifact = json.loads((root / "artifact.json").read_text(encoding="utf-8"))
    if artifact.get("producer", {}).get("pipeline") != pipeline:
        raise RuntimeError(f"run 的 batch/模型/ASR 别名不匹配: {root}")
    path = Path(artifact["uri"])
    if not path.is_absolute():
        path = ROOT / path
    if path.resolve() != expected.resolve():
        raise RuntimeError(f"run 未发布目标正式产物 {expected}: {root}")
    if artifact.get("size_bytes") != path.stat().st_size or not artifact.get("sha256"):
        raise RuntimeError(f"正式产物大小或哈希元数据不完整: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    if digest.hexdigest() != artifact["sha256"]:
        raise RuntimeError(f"正式产物已被其他运行覆盖或改变，哈希不匹配: {path}")


def discover_completed_run(cfg: dict, batch: str, stage: str) -> Path:
    pipeline, expected = task_output(batch, stage)
    candidates = set()
    # 目录名仅用来发现最新尝试（包括失败/中断）；成功仍须核验 artifact 和实际文件。
    pattern = re.compile(rf"\d{{8}}_\d{{6}}_{re.escape(pipeline)}(?:_shards)?$")
    for location in cfg.get("existing_runs_roots", ["runs", "var/asr_dnsmos"]):
        base = Path(location)
        if not base.is_absolute():
            base = ROOT / base
        if base.is_dir():
            for path in base.glob(f"**/*_{pipeline}*"):
                if path.is_dir() and pattern.fullmatch(path.name):
                    candidates.add(path.resolve())
    # Catalog 提供自定义 --runs-dir 的明确关联，不需要扫描整台服务器。
    catalog = ROOT / "data/catalog/artifacts"
    for record_path in catalog.glob("*.json"):
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            producer = record.get("producer") or {}
            if producer.get("pipeline") == pipeline and producer.get("run_dir"):
                path = Path(producer["run_dir"])
                if not path.is_absolute():
                    path = ROOT / path
                if path.is_dir():
                    candidates.add(path.resolve())
        except (OSError, ValueError):
            continue
    if not candidates:
        raise RuntimeError(
            f"{batch}/{stage} 没有运行进程，也未找到对应 run/正式产物 {expected}；不会重复启动"
        )

    def last_activity(root: Path) -> int:
        stamps = []
        match = re.match(r"\d{8}_\d{6}", root.name)
        if match:
            stamps.append(
                int(datetime.strptime(match[0], "%Y%m%d_%H%M%S").astimezone().timestamp() * 1e9)
            )
        for name in ("run.log", "metrics.json", "artifact.json", "config.yaml", "progress.json"):
            path = root / name
            if path.exists():
                stamps.append(path.stat().st_mtime_ns)
        return max(stamps, default=root.stat().st_mtime_ns)

    ranked = sorted(((last_activity(root), str(root)) for root in candidates), reverse=True)
    if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
        raise RuntimeError(
            f"{batch}/{stage} 最新 run 无法唯一确定，请检查 {ranked[0][1]} 和 {ranked[1][1]}"
        )
    root = Path(ranked[0][1])
    try:
        verify_completed_run(root, batch, stage)
    except (OSError, ValueError, RuntimeError) as exc:
        raise RuntimeError(
            f"{batch}/{stage} 最近一次 run 未通过完成核验，不回退到旧成功记录: {root}；{exc}"
        ) from exc
    return root


def process_started_ns(identities: list[dict]) -> int | None:
    try:
        uptime = float(Path("/proc/uptime").read_text().split()[0])
        boot_ns = time.time_ns() - int(uptime * 1e9)
        ticks = os.sysconf("SC_CLK_TCK")
        return boot_ns + int(min(int(p["process_token"]) for p in identities) / ticks * 1e9)
    except (OSError, ValueError, AttributeError):
        return None


def bind_existing(cfg: dict) -> None:
    """新任务启动前冻结已有任务的 run/PID 身份，续跑使用快照。"""
    if not cfg["existing_tasks"]:
        return
    processes = live_tasks()
    for key, entry in cfg["existing_tasks"].items():
        batch, stage = key.split("/")
        matches = matching_tasks(processes, batch, stage)
        auto = entry["run_dir"] == "auto"
        completion_verified = False
        roots = {path for proc in matches for path in proc["run_dirs"]}
        if matches and not roots:
            raise RuntimeError(f"{key} 已找到进程，但还不能关联 run_dir；请稍后 inspect 再重试")
        if auto:
            if not matches:
                entry["run_dir"] = str(discover_completed_run(cfg, batch, stage))
                completion_verified = True
                LOG.info("自动识别已完成任务 %s，run=%s", key, entry["run_dir"])
            elif len(roots) != 1:
                raise RuntimeError(
                    f"{key} 无法自动确定已有任务/run，请先 inspect，再显式填写 run_dir；不会重复启动"
                )
            else:
                entry["run_dir"] = next(iter(roots))
        root = Path(entry["run_dir"]).resolve()
        if not root.is_dir():
            raise RuntimeError(f"已有 run_dir 不存在: {root}")
        if roots and any(Path(path).resolve() != root for path in roots):
            raise RuntimeError(f"{key} 的运行进程与指定 run_dir 不一致")
        entry["run_dir"] = str(root)
        entry["processes"] = []
        for proc in matches:
            # 任务可能恰好在 discover 和绑定之间退出，转为核验已完成产物。
            if token := process_token(proc["pid"]):
                entry["processes"].append({"pid": proc["pid"], "process_token": token})
        if not entry["processes"] and not completion_verified:
            verify_completed_run(root, batch, stage)
        artifact = root / "artifact.json"
        started = process_started_ns(entry["processes"]) if entry["processes"] else None
        # 运行中的恢复任务不能复用上一次运行留下的成功标记。
        entry["artifact_before"] = (
            artifact.stat().st_mtime_ns
            if entry["processes"]
            and artifact.exists()
            and (started is None or artifact.stat().st_mtime_ns < started + 20_000_000)
            else None
        )
        if stage in ("qwen", "glm"):
            entry["service"] = find_existing_service(
                cfg, stage, 5, allow_missing=not entry["processes"]
            )


def existing_finished(entry: dict) -> bool:
    if any(is_running(proc) for proc in entry["processes"]):
        return False
    root = Path(entry["run_dir"])
    check_metrics(root, warn=False)
    artifact_path = root / "artifact.json"
    if not artifact_path.is_file():
        raise RuntimeError(f"已有任务已退出，但未发布本次成功产物: {root}")
    if artifact_path.stat().st_mtime_ns == entry.get("artifact_before"):
        # 兼容旧快照：绑定时任务可能已经发布成功产物、但仍在进程收尾。
        started = process_started_ns(entry["processes"]) if entry["processes"] else None
        if started is None or artifact_path.stat().st_mtime_ns < started + 20_000_000:
            raise RuntimeError(f"已有任务已退出，但未发布本次成功产物: {root}")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    producer_dir = artifact.get("producer", {}).get("run_dir")
    if not producer_dir or Path(producer_dir).resolve() != root.resolve():
        raise RuntimeError(f"成功产物未对应当前 run: {artifact_path}")
    output = Path(artifact["uri"])
    if not output.is_absolute():
        output = ROOT / output
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"已有任务最终产物缺失: {output}")
    # 分片任务必须覆盖所有输入 shard，不能仅凭个别 shard 成功。
    for shard in (root / "shards").glob("shard-*.parquet"):
        if not (root / shard.name).is_file() or not (root / shard.stem / "artifact.json").is_file():
            raise RuntimeError(f"已有任务有未发布分片: {shard.stem}")
    return True


def inspect_existing_tasks(cfg: dict, data: dict) -> None:
    from inspect_asr_dnsmos import inspect_run

    data["completed_tasks"], data["unresolved_tasks"] = [], []
    present = {item["run_dir"] for item in data["runs"]}
    requested = data.get("requested_batches") or []
    for key, entry in cfg["existing_tasks"].items():
        batch, stage = key.split("/")
        if (requested and batch not in requested) or matching_tasks(
            data["processes"], batch, stage
        ):
            continue
        try:
            if entry["run_dir"] == "auto":
                root = discover_completed_run(cfg, batch, stage)
            else:
                root = Path(entry["run_dir"]).resolve()
                verify_completed_run(root, batch, stage)
            summary = check_metrics(root, warn=False)
            data["completed_tasks"].append(
                {
                    "task": key,
                    "run_dir": str(root),
                    "failed_operations": summary["failed_operations"],
                }
            )
            if str(root) not in present:
                data["runs"].append(inspect_run(root))
                present.add(str(root))
        except (OSError, ValueError, RuntimeError) as exc:
            data["unresolved_tasks"].append({"task": key, "reason": str(exc)})


def refresh_existing(cfg: dict, completed: list[str]) -> None:
    """修复后手工重启的已有任务，仍须匹配原 run_dir 才能重新绑定。"""
    pending = {
        key: entry
        for key, entry in cfg["existing_tasks"].items()
        if key not in completed and not any(is_running(p) for p in entry["processes"])
    }
    if not pending:
        return
    processes = live_tasks()
    rebound = {
        key: {"run_dir": entry["run_dir"]}
        for key, entry in pending.items()
        if matching_tasks(processes, *key.split("/"))
    }
    if rebound:
        subset = {**cfg, "existing_tasks": rebound}
        bind_existing(subset)
        cfg["existing_tasks"].update(rebound)


def service_tree(identity: dict) -> list[dict]:
    if not is_running(identity):
        return []
    parents = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
            parents[int(path.name)] = int(fields[1])
        except (OSError, ValueError, IndexError):
            continue
    pids = {identity["pid"]}
    while additions := {pid for pid, parent in parents.items() if parent in pids} - pids:
        pids.update(additions)
    return [
        {"pid": pid, "process_token": token}
        for pid in sorted(pids)
        if (token := process_token(pid)) is not None
    ]


def stop_existing_service(identity: dict, timeout: float, tick) -> None:
    owned = service_tree(identity)
    if not owned:
        return
    LOG.info("已有转写成功，停止已核对的 GPU 5 vLLM 及子进程: %s", [p["pid"] for p in owned])
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for proc in owned:
            if is_running(proc):
                try:
                    os.kill(proc["pid"], sig)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + (timeout if sig == signal.SIGTERM else 10)
        while any(is_running(proc) for proc in owned) and time.monotonic() < deadline:
            tick()
            time.sleep(0.5)
    if any(is_running(proc) for proc in owned):
        raise RuntimeError("已有 GPU 5 vLLM 未能退出，停止调度")


def stop_group(proc: subprocess.Popen, timeout: float) -> None:
    """即使组长已退出，也清理本脚本创建的同组子进程。"""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        if sig == signal.SIGTERM:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                proc.poll()  # 回收组长，避免僵尸影响判断。
                try:
                    os.killpg(proc.pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.5)
            else:
                continue
            break
    proc.wait(timeout=10)


def session_name(batch: str, stage: str, service: bool = False) -> str:
    task = f"{stage}-vllm" if service else ("dnsmos" if stage == "dnsmos" else f"asr-run-{stage}1")
    return f"{batch}-{task}"


def tmux(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["tmux", *args], capture_output=True, text=True, timeout=15, check=check)


def tmux_worker(payload_path: Path) -> int:
    """在 tmux 窗格内监督任务，并把输出同时写到窗格与日志。"""
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload_path.unlink()  # 环境快照可能含凭据，仅用于启动，不留在日志中。
    status_path, gate = Path(payload["status_path"]), Path(payload["gate"])
    state = {"pid": os.getpid(), "process_token": process_token(os.getpid()), "returncode": None}
    proc = None
    pump = None

    def stop(_signum, _frame):
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        raise KeyboardInterrupt("收到 tmux 任务停止信号")

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, stop)
    save_json(status_path, state)
    code = 1
    cleanup_done = False
    try:
        # 调度器先设置 remain-on-exit，再放行，避免快速失败时丢失窗格。
        deadline = time.monotonic() + 30
        while not gate.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("等待 tmux 启动配置超时")
            time.sleep(0.1)
        gate.unlink()
        with Path(payload["log_path"]).open("ab") as output:
            try:
                proc = subprocess.Popen(
                    payload["argv"],
                    cwd=payload["cwd"],
                    env=payload["env"],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                state.update(child_pid=proc.pid, child_token=process_token(proc.pid))
                save_json(status_path, state)

                def copy_output():
                    while data := proc.stdout.read1(65536):
                        output.write(data)
                        output.flush()
                        try:
                            sys.stdout.buffer.write(data)
                            sys.stdout.buffer.flush()
                        except (OSError, ValueError):
                            pass  # 窗格关闭后仍继续落盘。

                pump = threading.Thread(target=copy_output, daemon=True)
                pump.start()
                code = proc.wait()
            finally:
                try:
                    if proc is not None:
                        stop_group(proc, payload["cleanup_timeout"])
                    cleanup_done = True
                finally:
                    if pump is not None:
                        pump.join(timeout=10)
    except KeyboardInterrupt:
        code = 130
    except Exception as exc:  # noqa: BLE001 — worker 必须发布失败退出码。
        state["error"] = str(exc)
        print(f"任务失败: {exc}", file=sys.stderr, flush=True)
    finally:
        if proc is None:
            cleanup_done = True
        state.update(returncode=code, cleanup_done=cleanup_done)
        save_json(status_path, state)
        print(f"\n任务结束，退出码={code}；日志: {payload['log_path']}", flush=True)
    return code


class TmuxTask:
    """Popen 风格句柄；监控 pane 内 worker，而非立即退出的 tmux 客户端。"""

    def __init__(
        self, name: str, job: Path, argv: list[str], env: dict, log_path: Path, timeout: float
    ):
        self.name, self.job, self.timeout = name, job, timeout
        self.returncode = None
        self.target = "=" + name  # exact-match，避免 tmux 的前缀匹配。
        self.reclaim_finished_session()
        token = uuid.uuid4().hex
        self.status_path = log_path.parent / f".tmux-{token}.status.json"
        gate = log_path.parent / f".tmux-{token}.go"
        payload_path = log_path.parent / f".tmux-{token}.payload.json"
        payload = {
            "argv": argv,
            "env": env,
            "cwd": str(ROOT),
            "log_path": str(log_path),
            "status_path": str(self.status_path),
            "gate": str(gate),
            "cleanup_timeout": timeout,
        }
        with os.fdopen(
            os.open(payload_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600),
            "w",
            encoding="utf-8",
        ) as stream:
            json.dump(payload, stream, ensure_ascii=False)
        # 使用当前 conda 的绝对 Python 路径；任务环境从快照恢复，避免旧 tmux server 环境污染。
        command = shlex.join(
            [
                "env",
                "-u",
                "PYTHONHOME",
                "-u",
                "PYTHONPATH",
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "_tmux-worker",
                str(payload_path),
            ]
        )
        created = False
        try:
            tmux("new-session", "-d", "-s", name, "-c", str(ROOT), command)
            created = True
            tmux("set-window-option", "-t", self.target, "remain-on-exit", "on")
            tmux("set-option", "-t", self.target, "@audio_engine_job", str(job))
            tmux("set-option", "-t", self.target, "@audio_engine_status", str(self.status_path))
            gate.touch()
            deadline = time.monotonic() + 10
            while not self.status_path.exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"tmux worker 未启动: {name}")
                time.sleep(0.1)
            self.pid = self.read_state()["pid"]
        except BaseException:
            if created:
                tmux("kill-session", "-t", self.target, check=False)
            payload_path.unlink(missing_ok=True)
            gate.unlink(missing_ok=True)
            raise

    def read_state(self) -> dict:
        return json.loads(self.status_path.read_text(encoding="utf-8"))

    def reclaim_finished_session(self):
        if tmux("has-session", "-t", self.target, check=False).returncode:
            return
        owner = tmux("show-options", "-v", "-t", self.target, "@audio_engine_job", check=False)
        status = tmux("show-options", "-v", "-t", self.target, "@audio_engine_status", check=False)
        dead = tmux("list-panes", "-t", self.target, "-F", "#{pane_dead}", check=False)
        previous = Path(status.stdout.strip()) if status.returncode == 0 else None
        if owner.stdout.strip() == str(self.job) and previous and previous.is_file():
            state = json.loads(previous.read_text(encoding="utf-8"))
            if (
                dead.returncode == 0
                and dead.stdout.split() == ["1"]
                and state.get("returncode") is not None
                and state.get("cleanup_done")
            ):
                tmux("kill-session", "-t", self.target)
                return
        raise RuntimeError(f"同名 tmux 会话存在，拒绝重复启动或覆盖: {self.name}")

    def poll(self):
        state = self.read_state()
        self.returncode = state.get("returncode")
        if self.returncode is None and not is_running(state):
            self.returncode = 255
        return self.returncode

    def stop(self):
        state = self.read_state()
        if is_running(state):
            os.kill(state["pid"], signal.SIGTERM)
            deadline = time.monotonic() + self.timeout + 30
            while is_running(self.read_state()) and time.monotonic() < deadline:
                time.sleep(0.2)
        state = self.read_state()
        if not state.get("cleanup_done"):
            raise RuntimeError(f"tmux 任务未确认清理完成: {self.name}；请检查会话和 GPU")
        # 保留 remain-on-exit 窗格，供 tmux attach 查看成功/失败输出。


def check_metrics(root: Path, *, warn: bool = True) -> dict:
    paths = list(root.rglob("metrics.json"))
    if not paths:
        raise RuntimeError(f"流水线没有生成 metrics.json，拒绝标记成功: {root}")
    failures = []
    for path in sorted(paths):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        if "failed" not in metrics:
            raise RuntimeError(f"缺失失败统计: {path}")
        count = metrics["failed"]
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise RuntimeError(f"失败统计值非法: {path}")
        if count:
            failures.append(
                {
                    "metrics_path": str(path),
                    "failed_operations": count,
                    "by_step": metrics.get("by_step", {}),
                }
            )
            if warn:
                LOG.warning("样本失败 %s 次，记录后继续后续任务；失败详情来源: %s", count, path)
    return {
        "failed_operations": sum(item["failed_operations"] for item in failures),
        "files": failures,
    }


class Runner:
    def __init__(self, cfg: dict, job: Path, resume: bool):
        self.cfg, self.job = cfg, job
        state_path = job / "status.json"
        self.state = json.loads(state_path.read_text(encoding="utf-8")) if resume else {}
        self.state.update(
            pid=os.getpid(), process_token=process_token(os.getpid()), status="running", error=None
        )
        self.state.setdefault("completed", [])
        self.state.setdefault("tasks", {})
        self.children: list[TmuxTask] = []
        self.cpu_tasks: dict[str, dict] = {}
        self.last_heartbeat = 0.0
        self.write_state()

    def write_state(self):
        self.state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        save_json(self.job / "status.json", self.state)

    def tick(self):
        self.poll_cpu()
        if time.monotonic() - self.last_heartbeat >= self.cfg["heartbeat"]:
            self.last_heartbeat = time.monotonic()
            for key, task in self.state["tasks"].items():
                if task["status"] == "running":
                    self.report_progress(key, Path(task["run_dir"]))
            self.write_state()
            LOG.info(
                "心跳 batch=%s stage=%s phase=%s",
                self.state.get("batch"),
                self.state.get("stage"),
                self.state.get("phase"),
            )

    def spawn(self, argv: list[str], env: dict, path: Path) -> TmuxTask:
        name = session_name(self.state["batch"], self.state["stage"], path.name == "service.log")
        LOG.info("启动 tmux=%s: %s；日志: %s", name, shlex.join(argv), path)
        proc = TmuxTask(name, self.job, argv, env, path, self.cfg["cleanup_timeout"])
        self.children.append(proc)
        self.state.setdefault("tmux_sessions", {})[name] = str(proc.status_path)
        self.write_state()
        LOG.info("tmux 已启动 %s worker PID=%s", name, proc.pid)
        return proc

    def cleanup(self, exclude=()):
        errors = []
        kept = []
        for proc in reversed(self.children):
            if proc in exclude:
                kept.append(proc)
                continue
            try:
                proc.stop()
            except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
                errors.append(str(exc))
        self.children = list(reversed(kept))
        if errors:
            raise RuntimeError("进程清理失败: " + "; ".join(errors))

    def wait_gpu(self, gpu: int):
        self.state["phase"] = "waiting_gpu"
        deadline = time.monotonic() + self.cfg["cleanup_timeout"]
        while True:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    str(gpu),
                    "--query-compute-apps=pid",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=15,
                check=True,
            )
            pids = result.stdout.strip()
            memory = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    str(gpu),
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=15,
                check=True,
            )
            used_mb = int(memory.stdout.strip())
            if not pids and used_mb <= self.cfg["gpu_idle_memory_mb"]:
                LOG.info("GPU %s 无计算进程，显存 %s MiB，可以继续", gpu, used_mb)
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"GPU {gpu} 未释放，停止调度；PIDs: {pids or '无'}；显存: {used_mb} MiB"
                )
            self.tick()
            time.sleep(2)

    def ready(self, proc: TmuxTask, stage: str):
        self.state["phase"] = "waiting_service"
        port = self.cfg[f"{stage}_port"]
        model = "qwen3-asr" if stage == "qwen" else "glm-asr"
        deadline = time.monotonic() + self.cfg["ready_timeout"]
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"{stage} 服务提前退出，退出码 {proc.returncode}")
            try:
                with opener.open(f"http://127.0.0.1:{port}/v1/models", timeout=5) as response:
                    data = json.load(response)
                if model in [item.get("id") for item in data.get("data", [])]:
                    LOG.info("%s 服务就绪，model=%s", stage, model)
                    return
            except (OSError, ValueError, urllib.error.URLError):
                pass
            self.tick()
            time.sleep(2)
        raise TimeoutError(f"{stage} 服务就绪超时（{self.cfg['ready_timeout']}s）")

    def stage(self, batch: dict, stage: str):
        self.poll_cpu()  # 发现并行评分失败后，不再启动下一项 GPU 工作。
        key = f"{batch['name']}/{stage}"
        if key in self.state["completed"]:
            LOG.info("跳过已完成阶段 %s", key)
            return
        self.state.update(batch=batch["name"], stage=stage, phase="starting")
        self.write_state()
        entry = self.cfg["existing_tasks"].get(key)
        if entry:
            self.state["tasks"][key] = {
                "status": "running",
                "mode": "existing",
                "run_dir": entry["run_dir"],
            }
            self.state["phase"] = "following_existing"
            LOG.info("跟随已有任务 %s，run=%s；不会重复启动", key, entry["run_dir"])
            while not existing_finished(entry):
                self.tick()
                time.sleep(1)
            LOG.info("已有任务 %s 已核验完成，复用正式结果并继续后续队列", key)
            if stage in ("qwen", "glm"):
                self.state["phase"] = "stopping_existing_service"
                if entry.get("service"):
                    stop_existing_service(entry["service"], self.cfg["cleanup_timeout"], self.tick)
                else:
                    LOG.info("已有 %s 服务已退出，直接检查 GPU 5 释放", stage)
            if stage != "dnsmos":
                self.wait_gpu(batch["gpu"])
            self.mark_completed(key)
            return
        client, service, env, runs, folder = self.prepare_pipeline(batch, stage)
        self.state["tasks"][key] = {"status": "running", "mode": "owned", "run_dir": str(runs)}
        previous = list(self.children)  # DNSMOS 继续运行，当前 GPU 阶段只清理自己的服务和客户端。
        try:
            if stage != "dnsmos":
                self.wait_gpu(batch["gpu"])
            server = None
            if service:
                port = self.cfg[f"{stage}_port"]
                with socket.socket() as probe:
                    probe.bind(("127.0.0.1", port))
                server = self.spawn(service, service_env(env, stage), folder / "service.log")
                self.ready(server, stage)
            self.state["phase"] = "pipeline"
            worker = self.spawn(client, env, folder / "pipeline.log")
            while worker.poll() is None:
                if server and server.poll() is not None:
                    raise RuntimeError(f"{stage} 服务在批处理中退出，退出码 {server.returncode}")
                self.tick()
                time.sleep(1)
            if worker.returncode:
                raise RuntimeError(f"{key} 流水线退出码 {worker.returncode}")
            summary = check_metrics(runs)
        finally:
            self.state["phase"] = "cleanup"
            try:
                self.write_state()
            finally:
                self.cleanup(exclude=previous)
        if stage != "dnsmos":
            self.wait_gpu(batch["gpu"])
        self.mark_completed(key, summary)

    def prepare_pipeline(self, batch: dict, stage: str):
        # 防止未绑定的手工任务被重复执行；existing_tasks 必须先冻结其身份。
        if matching_tasks(live_tasks(), batch["name"], stage):
            raise RuntimeError(
                f"{batch['name']}/{stage} 已有运行进程，请绑定 existing_tasks，拒绝重复启动"
            )
        folder = self.job / batch["name"] / stage
        folder.mkdir(parents=True, exist_ok=True)
        client, service, env = commands(self.cfg, batch, stage)
        runs = folder / "pipeline_runs"
        runs.mkdir(exist_ok=True)
        existing = list(runs.iterdir())
        client += ["--runs-dir", str(runs)]
        if existing:
            if len(existing) != 1 or not existing[0].is_dir():
                raise RuntimeError(f"无法唯一确定续跑目录: {runs}")
            client += ["--resume", str(existing[0])]
        return client, service, env, runs, folder

    def mark_completed(self, key: str, summary: dict | None = None):
        if summary is None:
            summary = check_metrics(Path(self.state["tasks"][key]["run_dir"]))
        self.state["completed"].append(key)
        self.state["tasks"][key]["failure_summary"] = summary
        self.state["tasks"][key]["status"] = (
            "completed_with_errors" if summary["failed_operations"] else "completed"
        )
        warnings = {
            name: task["failure_summary"]
            for name, task in self.state["tasks"].items()
            if task.get("failure_summary", {}).get("failed_operations")
        }
        save_json(
            self.job / "sample_failures.json",
            {
                "tasks": warnings,
                "note": "failed_operations 为流水线/步骤失败统计次数，不代表去重后的失败样本数。"
                "样本 ID、错误和重跑依据保留在对应 run 的日志、checkpoint 与 manifest 中。",
            },
        )
        self.write_state()
        LOG.info("阶段完成 %s", key)

    def start_cpu(self, batch: dict):
        key = f"{batch['name']}/dnsmos"
        if key in self.state["completed"]:
            return
        entry = self.cfg["existing_tasks"].get(key)
        if entry:
            self.cpu_tasks[key] = {"existing": entry}
            self.state["tasks"][key] = {
                "status": "running",
                "mode": "existing",
                "run_dir": entry["run_dir"],
            }
            LOG.info("并行跟随已有 DNSMOS %s，run=%s", key, entry["run_dir"])
        else:
            client, _, env, runs, folder = self.prepare_pipeline(batch, "dnsmos")
            self.state.update(batch=batch["name"], stage="dnsmos", phase="pipeline")
            worker = self.spawn(client, env, folder / "pipeline.log")
            self.cpu_tasks[key] = {"worker": worker, "runs": runs}
            self.state["tasks"][key] = {"status": "running", "mode": "owned", "run_dir": str(runs)}
        self.write_state()

    def poll_cpu(self):
        for key, task in list(self.cpu_tasks.items()):
            summary = None
            if "existing" in task:
                if not existing_finished(task["existing"]):
                    continue
            else:
                worker = task["worker"]
                code = worker.poll()
                if code is None:
                    continue
                if code:
                    raise RuntimeError(f"并行 DNSMOS {key} 退出码 {code}")
                summary = check_metrics(task["runs"])
                worker.stop()
                self.children.remove(worker)
            self.mark_completed(key, summary)
            del self.cpu_tasks[key]

    def report_progress(self, key: str, root: Path):
        from inspect_asr_dnsmos import inspect_run

        # owned 任务的 run 根位于 pipeline_runs 的唯一子目录。
        if root.name == "pipeline_runs" and root.is_dir():
            dirs = [p for p in root.iterdir() if p.is_dir()]
            if len(dirs) == 1:
                root = dirs[0]
        try:
            progress = inspect_run(root)
        except (OSError, ValueError) as exc:
            LOG.warning("%s 本次进度记录暂不可读，下次继续检查: %s", key, exc)
            return
        if item := progress.get("sharded_progress"):
            LOG.info(
                "实际进度 %s: %s/%s (%.1f%%) rate=%s/s ETA=%ss 记录距今=%ss",
                key,
                item.get("done"),
                item.get("total"),
                item.get("pct", 0),
                item.get("rate_window"),
                item.get("eta_s"),
                item.get("age_seconds"),
            )
            self.state["tasks"][key]["progress"] = item
        elif logs := progress.get("progress_logs"):
            LOG.info("实际进度 %s: %s", key, logs[-1]["last_progress_lines"][-1])
            self.state["tasks"][key]["progress"] = logs[-1]
        else:
            LOG.info("实际进度 %s: 暂无新进度行，run=%s", key, root)
        for ckpt in progress.get("checkpoints", []):
            if ckpt["failed"]:
                LOG.warning("%s 已提交失败=%s，checkpoint=%s", key, ckpt["failed"], ckpt["source"])

    def run(self) -> int:
        try:
            if self.cfg["dnsmos_parallel"]:
                for batch in self.cfg["batches"]:
                    self.start_cpu(batch)
            for batch in self.cfg["batches"]:
                for stage in STAGES:
                    if stage == "dnsmos" and self.cfg["dnsmos_parallel"]:
                        continue
                    self.stage(batch, stage)
            self.state["phase"] = "waiting_dnsmos"
            while self.cpu_tasks:
                self.tick()
                time.sleep(1)
            has_errors = any(
                task.get("failure_summary", {}).get("failed_operations")
                for task in self.state["tasks"].values()
            )
            self.state.update(
                status="completed_with_errors" if has_errors else "completed", phase="done"
            )
            if has_errors:
                LOG.warning(
                    "全部批次执行完成，包含样本失败；汇总: %s", self.job / "sample_failures.json"
                )
            else:
                LOG.info("全部批次执行完成")
            return 0
        except (Exception, KeyboardInterrupt) as exc:
            self.state.update(
                status="stopped" if isinstance(exc, KeyboardInterrupt) else "failed",
                error=str(exc) or "收到停止信号",
            )
            LOG.exception("执行停止，修复后使用 --resume；已完成阶段将跳过")
            return 1
        finally:
            try:
                self.cleanup()
            except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
                self.state.update(status="failed", error=str(exc))
                LOG.exception("任务收尾清理失败")
            self.write_state()


def execute(job: Path, resume: bool) -> int:
    import fcntl  # Linux only；plan 可在 Windows 查看。

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(job / "runner.log", encoding="utf-8"),
        ],
    )
    # 本仓库所有批次共用端口，整个队列持锁，避免重复启动。
    lock_path = ROOT / "var/asr_dnsmos.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            LOG.error("已有调度任务运行: %s", lock_path)
            return 1
        signal.signal(signal.SIGHUP, signal.SIG_IGN)

        def stop(_signum, _frame):
            # 一次中断进入 finally；清理期间忽略后续 TERM/INT。
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            raise KeyboardInterrupt("收到停止信号")

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        return Runner(load_config(job / "config.json"), job, resume).run()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "run", "plan", "status", "stop", "inspect"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/asr_dnsmos_server.json")
    parser.add_argument("--job-dir", type=Path, help="任务日志及状态目录；续跑/查看/停止时必填")
    parser.add_argument("--resume", action="store_true", help="使用任务快照续跑未完成阶段")
    parser.add_argument("--batch", action="append", default=[], help="inspect 精确筛选批次，可重复")
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=[],
        help="inspect 显式指定已有 pipeline run 目录，可重复",
    )
    parser.add_argument("--json", action="store_true", help="inspect 输出完整 JSON")
    args = parser.parse_args(argv)
    if args.action == "inspect":
        from inspect_asr_dnsmos import format_snapshot, snapshot

        if not sys.platform.startswith("linux"):
            parser.error("inspect 需要在实际运行任务的 Linux 服务器执行")
        data = snapshot(ROOT, args.batch, args.run_dir)
        inspect_existing_tasks(load_config(args.config), data)
        print(
            json.dumps(data, ensure_ascii=False, indent=2) if args.json else format_snapshot(data)
        )
        return 0
    if args.action == "plan":
        cfg = load_config(args.config)
        print("所有 GPU 任务严格串行使用物理卡 5；DNSMOS 独立并行。")
        for batch in cfg["batches"]:
            for stage in STAGES:
                if entry := cfg["existing_tasks"].get(f"{batch['name']}/{stage}"):
                    print(f"\n跟随现有任务 {batch['name']}/{stage}，run={entry['run_dir']}")
                    continue
                client, service, _ = commands(cfg, batch, stage)
                print(f"\n{batch['name']} GPU={batch['gpu']} {stage}")
                if service:
                    print(f"tmux: {session_name(batch['name'], stage, True)}")
                    print(shlex.join(service))
                print(f"tmux: {session_name(batch['name'], stage)}")
                print(shlex.join(client))
        return 0
    if not sys.platform.startswith("linux"):
        parser.error("实际执行需要 Linux 服务器；本地可以使用 plan 查看命令")
    if (args.resume or args.action in ("status", "stop")) and args.job_dir is None:
        parser.error("此操作必须传 --job-dir")
    job = (
        args.job_dir.resolve()
        if args.job_dir
        else (ROOT / "var/asr_dnsmos" / time.strftime("%Y%m%d_%H%M%S"))
    )
    state_path = job / "status.json"
    if args.action in ("status", "stop"):
        state = json.loads(state_path.read_text(encoding="utf-8"))
        alive = is_running(state)
        if args.action == "stop" and alive:
            os.kill(state["pid"], signal.SIGTERM)
            print("已发送停止信号；等待 status 显示 stopped 后再续跑。")
        else:
            print(json.dumps({**state, "process_alive": alive}, ensure_ascii=False, indent=2))
        return 0
    if args.resume:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if is_running(state):
            parser.error("任务仍在运行，不能续跑")
        cfg = load_config(job / "config.json")
        refresh_existing(cfg, state.get("completed", []))
        save_json(job / "config.json", cfg)
    else:
        cfg = load_config(args.config)
        for command in ("audio-data", "vllm", "nvidia-smi", "tmux"):
            if shutil.which(command) is None:
                parser.error(f"缺少 {command}；请先 conda activate asr")
        if job.exists():
            parser.error("任务目录已存在；续跑请加 --resume，新任务请换目录")
        bind_existing(cfg)
        job.mkdir(parents=True)
        save_json(job / "config.json", cfg)
    if args.action == "run":
        return execute(job, args.resume)
    with (job / "launcher.log").open("ab") as output:
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "run",
            "--job-dir",
            str(job),
        ]
        # run 读取已保存配置；内部启动也走 --resume，因此预置初始状态。
        if not state_path.exists():
            save_json(state_path, {"completed": [], "status": "starting"})
        command.append("--resume")
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(f"已启动 PID={proc.pid}\n任务目录: {job}\n总日志: {job / 'runner.log'}")
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            print(f"后台进程已退出（{proc.returncode}），请查看 {job / 'launcher.log'}")
            return proc.returncode
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("pid") == proc.pid:
            return 0
        time.sleep(0.1)
    return 0


if __name__ == "__main__":
    try:
        if len(sys.argv) == 3 and sys.argv[1] == "_tmux-worker":
            raise SystemExit(tmux_worker(Path(sys.argv[2])))
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"错误: {error}", file=sys.stderr)
        raise SystemExit(1) from error
