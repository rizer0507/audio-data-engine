"""已有任务检测必须关联真实进程，不拿历史目录猜进度。"""

import importlib.util
import json
import time
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "inspect_asr_dnsmos", Path(__file__).resolve().parents[1] / "scripts/inspect_asr_dnsmos.py"
)
inspect = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inspect)


def test_option_uses_exact_batch():
    assert inspect.option(["--source-name=20260814-1-n"], "--source-name") == "20260814-1-n"
    assert inspect.option(["--source-name", "20260814-n-1"], "--source-name") == "20260814-n-1"
    assert inspect.option(["--other-source-name", "wrong"], "--source-name") is None


def test_checkpoint_failed_counts_and_missing_parts(tmp_path):
    checkpoint = tmp_path / "checkpoints/00_qwen"
    checkpoint.mkdir(parents=True)
    (checkpoint / "part-000000.parquet").write_bytes(b"fake")
    (checkpoint / "_state.json").write_text(
        json.dumps(
            {
                "parts": [
                    {"file": "part-000000.parquet", "count_in": 500, "metrics": {"failed": 2}},
                    {"file": "part-000001.parquet", "count_in": 500, "metrics": {"failed": 4}},
                ],
                "complete": True,
            }
        )
    )
    result = inspect.inspect_run(tmp_path)["checkpoints"][0]
    assert result["committed_inputs"] == 500
    assert result["failed"] == 2
    assert result["missing_or_unusable_parts"] == 1
    assert not result["complete"]


def test_progress_snapshot_and_last_log_line(tmp_path):
    (tmp_path / "progress.json").write_text(
        json.dumps(
            {
                "done": 500,
                "total": 2000,
                "pct": 25,
                "updated_at": time.time() - 20,
            }
        )
    )
    (tmp_path / "run.log").write_text(
        "[PROGRESS] step=qwen done=500/2000 (25%)\nother message\n", encoding="utf-8"
    )
    result = inspect.inspect_run(tmp_path)
    assert result["sharded_progress"]["done"] == 500
    assert result["sharded_progress"]["age_seconds"] >= 20
    assert "500/2000" in result["progress_logs"][0]["last_progress_lines"][0]


def test_discover_reads_exact_process_batch_and_open_log(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    root = project / "runs/current_qwen"
    root.mkdir(parents=True)
    proc_root = tmp_path / "proc"
    for pid, batch in ((123, "20260813-1-n"), (124, "20260813-1-n-old")):
        proc = proc_root / str(pid)
        (proc / "fd").mkdir(parents=True)
        (proc / "fd/3").write_text("fake")
        (proc / "cwd").mkdir()
        argv = [
            "audio-data",
            "pipeline",
            "run",
            "pipelines/qwen_asr_batch.yaml",
            "--source-name",
            batch,
            "--asr-run",
            "qwen1",
            "--no-sharding",
        ]
        (proc / "cmdline").write_bytes("\0".join(argv).encode())
        (proc / "environ").write_bytes(b"SECRET=do-not-read\0CUDA_VISIBLE_DEVICES=5\0")
    original_resolve = Path.resolve

    def resolve(path, *args, **kwargs):
        if path.name == "cwd":
            return project
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    monkeypatch.setattr(inspect.os, "readlink", lambda fd: str(root / "run.log"))
    result = inspect.discover(project, ["20260813-1-n"], proc_root)
    assert len(result) == 1
    assert result[0]["pid"] == 123
    assert result[0]["run_dirs"] == [str(root)]
    assert result[0]["cuda_visible_devices"] == "5"
    assert "SECRET" not in json.dumps(result)


def test_no_live_process_does_not_pick_historical_run(tmp_path, monkeypatch):
    old = tmp_path / "runs/old_qwen_20260813-1-n"
    old.mkdir(parents=True)
    (old / "progress.json").write_text('{"done":100,"total":100}')
    monkeypatch.setattr(inspect, "discover", lambda *args: [])
    data = inspect.snapshot(tmp_path, ["20260813-1-n"], [])
    assert data["runs"] == []
    assert data["missing_batches"] == ["20260813-1-n"]
    assert "未匹配到" in inspect.format_snapshot(data)


def test_explicit_run_can_be_inspected_without_process(tmp_path, monkeypatch):
    monkeypatch.setattr(inspect, "discover", lambda *args: [])
    data = inspect.snapshot(tmp_path, [], [tmp_path])
    assert len(data["runs"]) == 1
    assert "暂无可读取" in inspect.format_snapshot(data)
