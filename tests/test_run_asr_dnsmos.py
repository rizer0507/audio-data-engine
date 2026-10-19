"""服务器调度的命令、失败阻断、清理和阶段续跑契约（不依赖模型/GPU）。"""

import hashlib
import importlib.util
import json
import shlex
import subprocess
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "run_asr_dnsmos", Path(__file__).resolve().parents[1] / "scripts/run_asr_dnsmos.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture
def cfg():
    config = runner.load_config(runner.ROOT / "configs/asr_dnsmos_server.json")
    # 旧调度测试独立验证自启动任务；当前三路手工任务由专门测试覆盖。
    config["existing_tasks"] = {}
    config["dnsmos_parallel"] = False
    return config


def test_commands_keep_batch_and_gpu_contract(cfg, monkeypatch):
    monkeypatch.setenv("GLM_ASR_API_BASES", "http://wrong:9999")
    batch = cfg["batches"][0]
    for stage in runner.STAGES:
        client, service, env = runner.commands(cfg, batch, stage)
        assert client[client.index("--source-name") + 1] == "20260813-1-n"
        assert "GLM_ASR_API_BASES" not in env
        assert env["OMP_NUM_THREADS"] == "1"
        if stage == "dnsmos":
            assert "--no-sharding" not in client
            assert env["CUDA_VISIBLE_DEVICES"] == ""
        else:
            assert "--no-sharding" in client
            assert client[client.index("--asr-run") + 1] == f"{stage}1"
            assert env["CUDA_VISIBLE_DEVICES"] == "5"
        if stage == "qwen":
            assert service[service.index("--port") + 1] == "5555"
        if stage == "glm":
            assert service[0] == cfg["glm_env"] + "/bin/python"


def test_glm_service_environment_does_not_leak_to_client(cfg, monkeypatch):
    monkeypatch.setenv("VLLM_ATTENTION_BACKEND", "old")
    _, _, client_env = runner.commands(cfg, cfg["batches"][0], "glm")
    server_env = runner.service_env(client_env, "glm")
    assert "VLLM_ATTENTION_BACKEND" not in server_env
    assert server_env["HF_HUB_OFFLINE"] == "1"
    assert client_env["VLLM_ATTENTION_BACKEND"] == "old"


@pytest.mark.parametrize("metrics", [None, {}, {"failed": -1}, {"failed": "1"}])
def test_missing_or_invalid_metrics_block_success(tmp_path, metrics):
    if metrics is not None:
        (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    with pytest.raises(RuntimeError):
        runner.check_metrics(tmp_path)


def test_nested_shard_sample_failure_warns_and_continues(tmp_path, caplog):
    for i, failed in enumerate((0, 1)):
        folder = tmp_path / str(i)
        folder.mkdir()
        (folder / "metrics.json").write_text(json.dumps({"failed": failed}))
    summary = runner.check_metrics(tmp_path)
    assert summary["failed_operations"] == 1
    assert len(summary["files"]) == 1
    assert "记录后继续" in caplog.text


def prepare_runner(cfg, tmp_path, monkeypatch, exit_code=0):
    monkeypatch.setattr(runner, "live_tasks", list)
    job = runner.Runner(cfg, tmp_path, False)
    calls = []

    def spawn(argv, env, path):
        calls.append(argv)
        if argv[0] == "audio-data":
            runs = Path(argv[argv.index("--runs-dir") + 1])
            output = runs / "one_run"
            output.mkdir(exist_ok=True)
            (output / "metrics.json").write_text(json.dumps({"failed": 0}))
        return subprocess.CompletedProcess(argv, exit_code)

    # CompletedProcess 以固定 poll 模拟已完成子进程。
    monkeypatch.setattr(
        subprocess.CompletedProcess, "poll", lambda self: self.returncode, raising=False
    )
    monkeypatch.setattr(job, "spawn", spawn)
    monkeypatch.setattr(job, "wait_gpu", lambda gpu: None)
    monkeypatch.setattr(job, "ready", lambda proc, stage: None)
    cleanups = []
    monkeypatch.setattr(job, "cleanup", lambda **kwargs: cleanups.append(True))
    return job, calls, cleanups


def test_failure_stops_queue_and_cleans_up(cfg, tmp_path, monkeypatch):
    job, calls, cleanups = prepare_runner(cfg, tmp_path, monkeypatch, exit_code=7)
    assert job.run() == 1
    assert job.state["status"] == "failed"
    assert job.state["completed"] == []
    assert len(calls) == 2  # 只有 Qwen 服务 + Qwen 客户端。
    assert cleanups == [True, True]  # 阶段清理 + 整体失败收尾。
    assert "7" in job.state["error"]


def test_success_skips_completed_and_resumes_incomplete(cfg, tmp_path, monkeypatch):
    job, calls, cleanups = prepare_runner(cfg, tmp_path, monkeypatch)
    batch = cfg["batches"][0]
    job.stage(batch, "sensevoice")
    assert job.state["completed"] == [f"{batch['name']}/sensevoice"]
    job.stage(batch, "sensevoice")
    assert len(calls) == 1
    # 模拟阶段收尾前中断：pipeline run 目录存在，但无阶段完成标记。
    job.state["completed"].clear()
    job.write_state()
    resumed = runner.Runner(cfg, tmp_path, True)
    monkeypatch.setattr(resumed, "spawn", job.spawn)
    monkeypatch.setattr(resumed, "wait_gpu", job.wait_gpu)
    monkeypatch.setattr(resumed, "cleanup", job.cleanup)
    resumed.stage(batch, "sensevoice")
    assert "--resume" in calls[-1]
    assert len(cleanups) == 2


def test_cleanup_failure_never_marks_stage_completed(cfg, tmp_path, monkeypatch):
    job, _, _ = prepare_runner(cfg, tmp_path, monkeypatch)

    def fail_cleanup(**kwargs):
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(job, "cleanup", fail_cleanup)
    assert job.run() == 1
    assert not job.state["completed"]


def test_stop_group_kills_descendants_even_if_leader_exited(monkeypatch):
    class Process:
        pid = 999999

        def poll(self):
            return 0

        def wait(self, timeout):
            pass

    signals = []
    monkeypatch.setattr(
        runner.os, "killpg", lambda pid, sig: signals.append((pid, sig)), raising=False
    )
    # Windows 上无 SIGKILL，但该逻辑要验证 Linux 的强制清理分支。
    monkeypatch.setattr(runner.signal, "SIGKILL", 9, raising=False)
    runner.stop_group(Process(), 0)
    assert signals == [(999999, runner.signal.SIGTERM), (999999, 9)]


def test_gpu_release_requires_no_processes_and_low_memory(cfg, tmp_path, monkeypatch):
    job = runner.Runner(cfg, tmp_path, False)
    cfg["cleanup_timeout"] = 0.001
    responses = iter(["", "4096"])
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=next(responses)),
    )
    ticks = iter([0, 1])
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(ticks))
    with pytest.raises(RuntimeError, match="4096"):
        job.wait_gpu(5)


def test_pid_reuse_does_not_stop_unrelated_process(monkeypatch):
    monkeypatch.setattr(runner, "process_token", lambda pid: "new-process")
    assert not runner.is_running({"pid": 123, "process_token": "old-process"})


def test_start_detaches_and_keeps_config_snapshot(tmp_path, monkeypatch):
    job = tmp_path / "job"
    launched = []

    class Process:
        pid = 987654

        def poll(self):
            return None

    def start(argv, **kwargs):
        launched.append((argv, kwargs))
        runner.save_json(job / "status.json", {"pid": Process.pid, "completed": []})
        return Process()

    monkeypatch.setattr(runner.sys, "platform", "linux")
    monkeypatch.setattr(runner.shutil, "which", lambda cmd: "/bin/" + cmd)
    monkeypatch.setattr(runner.subprocess, "Popen", start)
    monkeypatch.setattr(runner, "bind_existing", lambda config: None)
    assert runner.main(["start", "--job-dir", str(job)]) == 0
    command, options = launched[0]
    assert "--resume" in command
    assert options["start_new_session"] is True
    assert options["stdin"] == subprocess.DEVNULL
    assert options["stderr"] == subprocess.STDOUT
    assert options["stdout"].name == str(job / "launcher.log")
    assert (job / "config.json").is_file()


@pytest.mark.parametrize(
    "stage,service,expected",
    [
        ("qwen", True, "20260813-1-n-qwen-vllm"),
        ("qwen", False, "20260813-1-n-asr-run-qwen1"),
        ("sensevoice", False, "20260813-1-n-asr-run-sensevoice1"),
        ("glm", True, "20260813-1-n-glm-vllm"),
        ("glm", False, "20260813-1-n-asr-run-glm1"),
        ("dnsmos", False, "20260813-1-n-dnsmos"),
    ],
)
def test_tmux_names(stage, service, expected):
    assert runner.session_name("20260813-1-n", stage, service) == expected


def test_tmux_launch_uses_worker_status_and_environment_snapshot(tmp_path, monkeypatch):
    calls, payloads = [], []

    def tmux(*args, check=True):
        calls.append(args)
        if args[0] == "has-session":
            return subprocess.CompletedProcess(args, 1, stdout="")
        if args[0] == "new-session":
            payload_path = Path(shlex.split(args[-1])[-1])
            data = json.loads(payload_path.read_text(encoding="utf-8"))
            payloads.append(data)
            runner.save_json(
                Path(data["status_path"]),
                {
                    "pid": 123,
                    "process_token": "worker",
                    "returncode": 7,
                    "cleanup_done": True,
                },
            )
        return subprocess.CompletedProcess(args, 0, stdout="")

    monkeypatch.setattr(runner, "tmux", tmux)
    task = runner.TmuxTask(
        "batch-asr-run-qwen1",
        tmp_path,
        ["audio-data", "pipeline", "run"],
        {"PATH": "/conda/asr/bin", "PRIVATE_KEY": "not-on-command-line"},
        tmp_path / "pipeline.log",
        5,
    )
    assert task.poll() == 7  # tmux new-session 成功不能被当成任务成功。
    assert payloads[0]["env"]["PATH"] == "/conda/asr/bin"
    assert "PRIVATE_KEY" not in str(calls)
    assert ("set-window-option", "-t", "=batch-asr-run-qwen1", "remain-on-exit", "on") in calls
    task.stop()
    assert not any(call[0] == "kill-session" for call in calls)


def test_existing_foreign_tmux_session_is_never_overwritten(tmp_path, monkeypatch):
    calls = []

    def tmux(*args, check=True):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="someone-else\n")

    monkeypatch.setattr(runner, "tmux", tmux)
    with pytest.raises(RuntimeError, match="同名 tmux"):
        runner.TmuxTask("batch-dnsmos", tmp_path, ["audio-data"], {}, tmp_path / "pipeline.log", 5)
    assert not any(call[0] in ("kill-session", "new-session") for call in calls)


def test_resume_reclaims_only_own_finished_tmux_pane(tmp_path, monkeypatch):
    status = tmp_path / "previous.json"
    runner.save_json(status, {"returncode": 1, "cleanup_done": True})
    calls = []

    def tmux(*args, check=True):
        calls.append(args)
        output = ""
        if args[-1] == "@audio_engine_job":
            output = str(tmp_path)
        elif args[-1] == "@audio_engine_status":
            output = str(status)
        elif args[0] == "list-panes":
            output = "1\n"
        return subprocess.CompletedProcess(args, 0, stdout=output)

    monkeypatch.setattr(runner, "tmux", tmux)
    task = runner.TmuxTask.__new__(runner.TmuxTask)
    task.name, task.target, task.job = "batch-dnsmos", "=batch-dnsmos", tmp_path
    task.reclaim_finished_session()
    assert calls[-1] == ("kill-session", "-t", "=batch-dnsmos")


def test_tmux_worker_tees_output_and_records_real_exit_code(tmp_path, monkeypatch, capsys):
    # Worker 的真实短命令可在 Windows 验证；仅替换 Linux 信号与进程组清理。
    monkeypatch.setattr(runner.signal, "SIGHUP", 1, raising=False)
    monkeypatch.setattr(runner.signal, "signal", lambda *args: None)
    monkeypatch.setattr(runner, "stop_group", lambda proc, timeout: proc.wait(timeout=5))
    gate = tmp_path / "go"
    gate.touch()
    payload = tmp_path / "payload.json"
    status = tmp_path / "status.json"
    log = tmp_path / "pipeline.log"
    runner.save_json(
        payload,
        {
            "argv": [runner.sys.executable, "-c", "print('real-task-output', flush=True); exit(7)"],
            "cwd": str(tmp_path),
            "env": dict(runner.os.environ),
            "log_path": str(log),
            "status_path": str(status),
            "gate": str(gate),
            "cleanup_timeout": 5,
        },
    )
    assert runner.tmux_worker(payload) == 7
    result = json.loads(status.read_text(encoding="utf-8"))
    assert result["returncode"] == 7
    assert result["cleanup_done"]
    assert "real-task-output" in log.read_text(encoding="utf-8")
    assert "real-task-output" in capsys.readouterr().out
    assert not payload.exists()


def test_all_batches_use_only_gpu_5_and_reject_old_gpu_0(tmp_path):
    config = runner.load_config(runner.ROOT / "configs/asr_dnsmos_server.json")
    for batch in config["batches"]:
        for stage in ("qwen", "sensevoice", "glm"):
            assert runner.commands(config, batch, stage)[2]["CUDA_VISIBLE_DEVICES"] == "5"
    config["batches"][1]["gpu"] = 0
    path = tmp_path / "old.json"
    runner.save_json(path, config)
    with pytest.raises(ValueError, match="GPU 5"):
        runner.load_config(path)


def test_bind_current_qwen_and_two_dnsmos_without_launching(tmp_path, cfg, monkeypatch):
    tasks = {}
    processes = []
    for i, (batch, stage) in enumerate(
        [
            ("20260813-1-n", "qwen"),
            ("20260813-1-n", "dnsmos"),
            ("20260814-1-n", "dnsmos"),
        ]
    ):
        root = tmp_path / f"run{i}"
        root.mkdir()
        tasks[f"{batch}/{stage}"] = {"run_dir": "auto"}
        processes.append(
            {
                "pid": i + 10,
                "batch": batch,
                "asr_run": "qwen1" if stage == "qwen" else None,
                "pipeline": f"pipelines/{stage}_asr_batch.yaml"
                if stage == "qwen"
                else "pipelines/dnsmos_score_batch.yaml",
                "run_dirs": [str(root)],
            }
        )
    cfg["existing_tasks"] = tasks
    monkeypatch.setattr(runner, "live_tasks", lambda: processes)
    monkeypatch.setattr(runner, "process_token", lambda pid: str(pid))
    services = []
    monkeypatch.setattr(
        runner,
        "find_existing_service",
        lambda config, stage, gpu, **kwargs: (
            services.append((stage, gpu)) or {"pid": 99, "process_token": "99"}
        ),
    )
    runner.bind_existing(cfg)
    assert services == [("qwen", 5)]
    assert all(entry["run_dir"] != "auto" for entry in tasks.values())
    assert tasks["20260813-1-n/qwen"]["processes"] == [{"pid": 10, "process_token": "10"}]


def test_binding_missing_manual_task_fails_instead_of_restarting(cfg, monkeypatch):
    cfg["existing_tasks"] = {"20260813-1-n/qwen": {"run_dir": "auto"}}
    monkeypatch.setattr(runner, "live_tasks", list)
    with pytest.raises(RuntimeError, match="不会重复启动"):
        runner.bind_existing(cfg)


def existing_output(tmp_path):
    root = tmp_path / "external"
    root.mkdir()
    output = tmp_path / "output.parquet"
    output.write_bytes(b"published")
    runner.save_json(root / "metrics.json", {"failed": 0})
    runner.save_json(
        root / "artifact.json", {"producer": {"run_dir": str(root)}, "uri": str(output)}
    )
    return {"run_dir": str(root), "processes": [], "artifact_before": None}


def test_existing_task_needs_published_success_not_just_dead_pid(tmp_path, monkeypatch):
    entry = existing_output(tmp_path)
    assert runner.existing_finished(entry)
    entry["artifact_before"] = (Path(entry["run_dir"]) / "artifact.json").stat().st_mtime_ns
    with pytest.raises(RuntimeError, match="未发布本次"):
        runner.existing_finished(entry)
    entry["processes"] = [{"pid": 3, "process_token": "active"}]
    monkeypatch.setattr(runner, "process_token", lambda pid: "active")
    assert not runner.existing_finished(entry)


def test_existing_sample_failures_do_not_block_completed_task(tmp_path):
    entry = existing_output(tmp_path)
    runner.save_json(Path(entry["run_dir"]) / "metrics.json", {"failed": 1})
    assert runner.existing_finished(entry)


def test_existing_dnsmos_requires_all_shards_published(tmp_path):
    entry = existing_output(tmp_path)
    root = Path(entry["run_dir"])
    (root / "shards").mkdir()
    (root / "shards/shard-000.parquet").write_bytes(b"input")
    with pytest.raises(RuntimeError, match="未发布分片"):
        runner.existing_finished(entry)


def test_existing_qwen_finishes_before_service_stop_and_gpu_release(cfg, tmp_path, monkeypatch):
    runner.save_json(tmp_path / "metrics.json", {"failed": 0})
    entry = {"run_dir": str(tmp_path), "service": {"pid": 99}}
    cfg["existing_tasks"] = {"20260813-1-n/qwen": entry}
    job = runner.Runner(cfg, tmp_path, False)
    events = []
    results = iter([False, True])

    def finished(_entry):
        result = next(results)
        events.append("finished" if result else "running")
        return result

    monkeypatch.setattr(runner, "existing_finished", finished)
    monkeypatch.setattr(job, "tick", lambda: events.append("progress"))
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(runner, "stop_existing_service", lambda *args: events.append("stop-vllm"))
    monkeypatch.setattr(job, "wait_gpu", lambda gpu: events.append(f"gpu-{gpu}-released"))
    monkeypatch.setattr(job, "spawn", lambda *args: pytest.fail("已有 qwen1 不应重复启动"))
    job.stage(cfg["batches"][0], "qwen")
    assert events == ["running", "progress", "finished", "stop-vllm", "gpu-5-released"]
    assert job.state["completed"] == ["20260813-1-n/qwen"]


def test_gpu_stage_cleanup_keeps_parallel_dnsmos(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "live_tasks", list)
    job = runner.Runner(cfg, tmp_path, False)
    stopped = []

    class Task:
        returncode = 0

        def __init__(self, name):
            self.name = name

        def poll(self):
            return 0

        def stop(self):
            stopped.append(self.name)

    cpu = Task("dnsmos")
    job.children.append(cpu)

    def spawn(argv, env, path):
        runs = Path(argv[argv.index("--runs-dir") + 1])
        runner.save_json(runs / "metrics.json", {"failed": 0})
        gpu = Task("sensevoice")
        job.children.append(gpu)
        return gpu

    monkeypatch.setattr(job, "spawn", spawn)
    monkeypatch.setattr(job, "wait_gpu", lambda gpu: None)
    job.stage(cfg["batches"][0], "sensevoice")
    assert stopped == ["sensevoice"]
    assert job.children == [cpu]


def test_parallel_dnsmos_starts_before_serial_gpu_queue(cfg, tmp_path, monkeypatch):
    cfg["dnsmos_parallel"] = True
    job = runner.Runner(cfg, tmp_path, False)
    events = []
    monkeypatch.setattr(job, "start_cpu", lambda batch: events.append((batch["name"], "dnsmos")))
    monkeypatch.setattr(job, "stage", lambda batch, stage: events.append((batch["name"], stage)))
    assert job.run() == 0
    assert events[:2] == [("20260813-1-n", "dnsmos"), ("20260814-1-n", "dnsmos")]
    assert [stage for batch, stage in events[2:]] == ["qwen", "sensevoice", "glm"] * 2


def test_parallel_dnsmos_failure_prevents_next_gpu_start(cfg, tmp_path, monkeypatch):
    job = runner.Runner(cfg, tmp_path, False)

    class Task:
        def poll(self):
            return 7

    job.cpu_tasks["batch/dnsmos"] = {"worker": Task()}
    monkeypatch.setattr(
        job, "spawn", lambda *args: pytest.fail("DNSMOS 失败后不能启动下一 GPU 任务")
    )
    with pytest.raises(RuntimeError, match="DNSMOS.*7"):
        job.stage(cfg["batches"][0], "qwen")


def test_external_service_cleanup_checks_pid_identity_before_kill(monkeypatch):
    owned = [{"pid": 10, "process_token": "original"}, {"pid": 11, "process_token": "old"}]
    tokens = {10: "original", 11: "reused"}
    calls = []
    monkeypatch.setattr(runner, "service_tree", lambda identity: owned)
    monkeypatch.setattr(runner, "process_token", lambda pid: tokens.get(pid))

    def kill(pid, sig):
        calls.append(pid)
        tokens[pid] = None

    monkeypatch.setattr(runner.os, "kill", kill)
    monkeypatch.setattr(runner.signal, "SIGKILL", 9, raising=False)
    runner.stop_existing_service(owned[0], 1, lambda: None)
    assert calls == [10]


def test_resume_rebinds_restarted_manual_task_only_to_original_run(cfg, monkeypatch):
    key = "20260813-1-n/qwen"
    cfg["existing_tasks"] = {
        key: {
            "run_dir": "/runs/original",
            "processes": [
                {"pid": 10, "process_token": "old"},
            ],
        }
    }
    monkeypatch.setattr(runner, "process_token", lambda pid: None)
    monkeypatch.setattr(
        runner,
        "live_tasks",
        lambda: [
            {
                "batch": "20260813-1-n",
                "pipeline": "pipelines/qwen_asr_batch.yaml",
                "asr_run": "qwen1",
            }
        ],
    )
    observed = []

    def bind(subset):
        entry = subset["existing_tasks"][key]
        observed.append(entry["run_dir"])
        entry["processes"] = [{"pid": 20, "process_token": "new"}]

    monkeypatch.setattr(runner, "bind_existing", bind)
    runner.refresh_existing(cfg, [])
    assert observed == ["/runs/original"]
    assert cfg["existing_tasks"][key]["processes"][0]["pid"] == 20
    runner.refresh_existing(cfg, [key])
    assert observed == ["/runs/original"]


def published_run(project, batch="20260813-1-n", stage="qwen", stamp="20010101_010101"):
    pipeline, output = runner.task_output(batch, stage)
    root = project / "runs" / f"{stamp}_{pipeline}"
    root.mkdir(parents=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"published-output")
    runner.save_json(root / "metrics.json", {"failed": 0})
    runner.save_json(
        root / "artifact.json",
        {
            "producer": {"run_dir": str(root), "pipeline": pipeline},
            "uri": str(output),
            "size_bytes": output.stat().st_size,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        },
    )
    return root, output


def test_qwen_finished_before_script_start_is_auto_detected_and_service_stopped(
    cfg, tmp_path, monkeypatch
):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    root, _ = published_run(tmp_path)
    cfg["existing_tasks"] = {"20260813-1-n/qwen": {"run_dir": "auto"}}
    monkeypatch.setattr(runner, "live_tasks", list)
    calls = []

    def service(config, stage, gpu, **kwargs):
        assert kwargs["allow_missing"]
        return {"pid": 99, "process_token": "qwen-vllm"}

    monkeypatch.setattr(runner, "find_existing_service", service)
    runner.bind_existing(cfg)
    assert cfg["existing_tasks"]["20260813-1-n/qwen"]["run_dir"] == str(root)
    assert cfg["existing_tasks"]["20260813-1-n/qwen"]["processes"] == []
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    job = runner.Runner(cfg, job_dir, False)
    monkeypatch.setattr(
        runner, "stop_existing_service", lambda identity, *args: calls.append(identity["pid"])
    )
    monkeypatch.setattr(job, "wait_gpu", lambda gpu: calls.append(f"released-{gpu}"))
    monkeypatch.setattr(job, "spawn", lambda *args: pytest.fail("已完成 qwen1 不得重跑"))
    job.stage(cfg["batches"][0], "qwen")
    assert calls == [99, "released-5"]
    assert job.state["completed"] == ["20260813-1-n/qwen"]


def test_completed_qwen_with_service_already_stopped_can_continue(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    published_run(tmp_path)
    cfg["existing_tasks"] = {"20260813-1-n/qwen": {"run_dir": "auto"}}
    monkeypatch.setattr(runner, "live_tasks", list)
    monkeypatch.setattr(runner, "find_existing_service", lambda *args, **kwargs: None)
    runner.bind_existing(cfg)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    job = runner.Runner(cfg, job_dir, False)
    monkeypatch.setattr(
        runner, "stop_existing_service", lambda *args: pytest.fail("服务已停，无需杀进程")
    )
    released = []
    monkeypatch.setattr(job, "wait_gpu", released.append)
    job.stage(cfg["batches"][0], "qwen")
    assert released == [5]


def test_completed_discovery_never_falls_back_after_latest_failed_attempt(
    cfg, tmp_path, monkeypatch
):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    root, _ = published_run(tmp_path)
    for path in root.iterdir():
        runner.os.utime(path, (1000, 1000))
    pipeline, _ = runner.task_output("20260813-1-n", "qwen")
    failed = tmp_path / "runs" / f"20010102_010101_{pipeline}"
    failed.mkdir()
    runner.save_json(failed / "metrics.json", {"failed": 3})
    runner.os.utime(failed / "metrics.json", (2000, 2000))
    with pytest.raises(RuntimeError, match="不回退到旧成功"):
        runner.discover_completed_run(cfg, "20260813-1-n", "qwen")


def test_completed_discovery_rejects_changed_formal_output(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    _, output = published_run(tmp_path)
    output.write_bytes(b"changed---output")  # 与原文件相同大小，须通过内容哈希发现变化。
    with pytest.raises(RuntimeError, match="哈希不匹配"):
        runner.discover_completed_run(cfg, "20260813-1-n", "qwen")


def test_completed_dnsmos_can_be_discovered_without_process(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    root, _ = published_run(tmp_path, stage="dnsmos")
    assert runner.discover_completed_run(cfg, "20260813-1-n", "dnsmos") == root


def test_completed_discovery_rejects_other_asr_alias(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    root, _ = published_run(tmp_path)
    root.rename(root.with_name(root.name.replace("qwen1", "qwen2")))
    with pytest.raises(RuntimeError, match="未找到对应 run"):
        runner.discover_completed_run(cfg, "20260813-1-n", "qwen")


def test_completion_published_before_process_exit_is_not_treated_as_old_artifact(
    cfg, tmp_path, monkeypatch
):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    root, _ = published_run(tmp_path)
    cfg["existing_tasks"] = {"20260813-1-n/qwen": {"run_dir": "auto"}}
    monkeypatch.setattr(
        runner,
        "live_tasks",
        lambda: [
            {
                "batch": "20260813-1-n",
                "pipeline": "pipelines/qwen_asr_batch.yaml",
                "asr_run": "qwen1",
                "run_dirs": [str(root)],
                "pid": 10,
            }
        ],
    )
    monkeypatch.setattr(runner, "process_token", lambda pid: "10")
    monkeypatch.setattr(
        runner,
        "process_started_ns",
        lambda identities: (root / "artifact.json").stat().st_mtime_ns - 1_000_000_000,
    )
    monkeypatch.setattr(runner, "find_existing_service", lambda *args, **kwargs: {"pid": 99})
    runner.bind_existing(cfg)
    entry = cfg["existing_tasks"]["20260813-1-n/qwen"]
    assert entry["artifact_before"] is None
    monkeypatch.setattr(runner, "process_token", lambda pid: None)
    assert runner.existing_finished(entry)


def test_old_snapshot_with_current_generation_artifact_can_resume(tmp_path, monkeypatch):
    entry = existing_output(tmp_path)
    artifact = Path(entry["run_dir"]) / "artifact.json"
    entry["artifact_before"] = artifact.stat().st_mtime_ns
    entry["processes"] = [{"pid": 10, "process_token": "100"}]
    monkeypatch.setattr(runner, "process_token", lambda pid: None)
    monkeypatch.setattr(
        runner, "process_started_ns", lambda identities: artifact.stat().st_mtime_ns - 1_000_000_000
    )
    assert runner.existing_finished(entry)


def test_inspect_includes_completed_qwen_without_running_process(cfg, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from inspect_asr_dnsmos import format_snapshot

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    root, _ = published_run(tmp_path)
    cfg["existing_tasks"] = {"20260813-1-n/qwen": {"run_dir": "auto"}}
    data = {
        "processes": [],
        "runs": [],
        "requested_batches": ["20260813-1-n"],
        "observed_at": "test",
        "missing_batches": ["20260813-1-n"],
        "note": "只读",
    }
    runner.inspect_existing_tasks(cfg, data)
    assert data["completed_tasks"] == [
        {"task": "20260813-1-n/qwen", "run_dir": str(root), "failed_operations": 0}
    ]
    assert data["unresolved_tasks"] == []
    assert "已核验完成: 20260813-1-n/qwen" in format_snapshot(data)


def test_one_failed_dnsmos_sample_does_not_abort_gpu_queue_and_is_reported(
    cfg, tmp_path, monkeypatch
):
    entry = existing_output(tmp_path)
    root = Path(entry["run_dir"])
    (root / "metrics.json").unlink()
    (root / "shards").mkdir()
    for index, failed in enumerate((0, 1)):
        name = f"shard-{index:03d}"
        (root / "shards" / f"{name}.parquet").write_bytes(b"input")
        (root / f"{name}.parquet").write_bytes(b"output")
        (root / name).mkdir()
        runner.save_json(root / name / "metrics.json", {"failed": failed})
        runner.save_json(root / name / "artifact.json", {})
    key = "20260813-1-n/dnsmos"
    cfg["existing_tasks"] = {key: entry}
    job, calls, _ = prepare_runner(cfg, tmp_path, monkeypatch)
    job.start_cpu(cfg["batches"][0])
    job.stage(cfg["batches"][0], "sensevoice")
    assert job.state["tasks"][key]["status"] == "completed_with_errors"
    assert job.state["tasks"][key]["failure_summary"]["failed_operations"] == 1
    assert job.state["tasks"]["20260813-1-n/sensevoice"]["status"] == "completed"
    assert job.run() == 0
    assert job.state["status"] == "completed_with_errors"
    assert not any(
        "pipelines/dnsmos_score_batch.yaml" in argv
        and argv[argv.index("--source-name") + 1] == "20260813-1-n"
        for argv in calls
    )
    report = json.loads((tmp_path / "sample_failures.json").read_text(encoding="utf-8"))
    assert report["tasks"][key]["failed_operations"] == 1
    assert Path(report["tasks"][key]["files"][0]["metrics_path"]).parent.name == "shard-001"


def test_completed_qwen_with_one_sample_failure_still_releases_gpu(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    root, _ = published_run(tmp_path)
    runner.save_json(root / "metrics.json", {"failed": 1})
    cfg["existing_tasks"] = {"20260813-1-n/qwen": {"run_dir": "auto"}}
    monkeypatch.setattr(runner, "live_tasks", list)
    monkeypatch.setattr(runner, "find_existing_service", lambda *args, **kwargs: {"pid": 99})
    runner.bind_existing(cfg)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    job = runner.Runner(cfg, job_dir, False)
    events = []
    monkeypatch.setattr(runner, "stop_existing_service", lambda *args: events.append("stop-qwen"))
    monkeypatch.setattr(job, "wait_gpu", lambda gpu: events.append(f"released-{gpu}"))
    job.stage(cfg["batches"][0], "qwen")
    assert events == ["stop-qwen", "released-5"]
    assert job.state["tasks"]["20260813-1-n/qwen"]["status"] == "completed_with_errors"
