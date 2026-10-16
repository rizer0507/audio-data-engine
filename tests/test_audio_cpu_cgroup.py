"""Shared cgroup quota logic; no host resource settings are modified by these tests."""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
for name in ("run_vad_cpu_budget", "audio_cpu_budget"):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
budget = sys.modules["audio_cpu_budget"]


def test_128_cpu_shared_quota_is_strictly_under_thirty_percent():
    quota, period = budget.quota_for_host(128)
    assert (quota, period) == (3_800_000, 100_000)
    assert quota / period / 128 < 0.3


@pytest.mark.parametrize("total", [4, 10, 16, 100, 128, 192])
def test_rounding_never_reaches_thirty_percent(total):
    quota, period = budget.quota_for_host(total)
    assert 0 < quota / period / total < 0.3


@pytest.mark.parametrize("value", [0, -1, 31, float("inf"), float("nan")])
def test_budget_cannot_be_widened_above_thirty_percent(value):
    with pytest.raises(ValueError):
        budget.quota_for_host(128, value)


def test_missing_or_unlimited_quota_fails_closed(tmp_path):
    with pytest.raises(RuntimeError, match="configure"):
        budget.read_quota(tmp_path, 128)
    (tmp_path / "cpu.max").write_text("max 100000")
    with pytest.raises(RuntimeError, match="no CPU limit"):
        budget.read_quota(tmp_path, 128)
    (tmp_path / "cpu.max").write_text("4000000 100000")
    with pytest.raises(RuntimeError, match="not below"):
        budget.read_quota(tmp_path, 128)


def test_burst_allowance_cannot_bypass_budget(tmp_path):
    (tmp_path / "cpu.max").write_text("3800000 100000")
    (tmp_path / "cpu.max.burst").write_text("100000")
    with pytest.raises(RuntimeError, match="burst"):
        budget.read_quota(tmp_path, 128)


def test_setup_keeps_other_controllers_and_clears_burst(tmp_path, monkeypatch):
    (tmp_path / "cgroup.controllers").write_text("cpu memory io")
    (tmp_path / "cgroup.subtree_control").write_text("cpu memory")
    group = tmp_path / budget.GROUP
    group.mkdir()
    (group / "cpu.max").write_text("max 100000")
    (group / "cpu.max.burst").write_text("5000")
    monkeypatch.setattr(budget, "CGROUP_ROOT", tmp_path)
    assert budget.configure(128, 30) == group
    assert (tmp_path / "cgroup.subtree_control").read_text() == "cpu memory"
    assert (group / "cpu.max").read_text() == "3800000 100000"
    assert (group / "cpu.max.burst").read_text() == "0"


def test_run_enters_shared_group_before_exec_and_sets_thread_environment(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(budget.os, "geteuid", lambda: 1000, raising=False)
    monkeypatch.setattr(budget.os, "getpid", lambda: 12345)

    def execute(command, argv, env):
        assert (tmp_path / "cgroup.procs").read_text() == "12345"
        events.append((command, argv, env))

    monkeypatch.setattr(budget.os, "execvpe", execute)
    budget.run(tmp_path, ["--", "/venv/bin/audio-data", "pipeline", "run", "test.yaml"])
    assert events[0][0] == "/venv/bin/audio-data"
    assert events[0][2]["OPENBLAS_NUM_THREADS"] == "1"
    assert events[0][2]["OMP_WAIT_POLICY"] == "PASSIVE"


def test_attach_moves_all_descendants_into_the_same_group(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(budget.os, "kill", lambda *a: None)
    monkeypatch.setattr(budget.os, "getpid", lambda: 999)
    monkeypatch.setattr(budget, "process_tree", lambda roots: [100, 200, 300])
    original = Path.write_text

    def write(path, value, *args, **kwargs):
        calls.append(value)
        return original(path, value, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", write)
    assert budget.attach(tmp_path, [100, 200]) == 3
    assert calls == ["100", "200", "300"] * 2


def test_root_execution_drops_to_sudo_user_before_pipeline(tmp_path, monkeypatch):
    from types import SimpleNamespace

    calls = []
    monkeypatch.setenv("SUDO_UID", "1001")
    monkeypatch.setenv("SUDO_USER", "engine")
    monkeypatch.setattr(budget.os, "geteuid", lambda: 0, raising=False)
    monkeypatch.setattr(budget.os, "getpid", lambda: 12345)
    monkeypatch.setitem(sys.modules, "pwd", SimpleNamespace(getpwuid=lambda uid: SimpleNamespace(
        pw_name="engine", pw_gid=1002, pw_dir="/home/engine",
    )))
    for name in ("initgroups", "setgid", "setuid"):
        monkeypatch.setattr(budget.os, name,
                            lambda *args, name=name: calls.append((name, args)), raising=False)

    def execute(command, argv, env):
        assert (tmp_path / "cgroup.procs").read_text() == "12345"
        assert calls == [("initgroups", ("engine", 1002)), ("setgid", (1002,)), ("setuid", (1001,))]
        assert "SUDO_UID" not in env
        assert env["USER"] == "engine"

    monkeypatch.setattr(budget.os, "execvpe", execute)
    budget.run(tmp_path, ["/venv/bin/audio-data", "pipeline", "run", "test.yaml"])


def test_direct_root_launch_is_rejected_before_migration(tmp_path, monkeypatch):
    monkeypatch.delenv("SUDO_UID", raising=False)
    monkeypatch.setattr(budget.os, "geteuid", lambda: 0, raising=False)
    with pytest.raises(ValueError, match="root-only launches"):
        budget.run(tmp_path, ["audio-data", "pipeline", "run", "test.yaml"])
    assert not (tmp_path / "cgroup.procs").exists()
