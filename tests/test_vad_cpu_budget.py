"""CPU budgeting logic is portable; Linux affinity syscalls are mocked here."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "vad_cpu_budget", Path(__file__).resolve().parents[1] / "scripts/run_vad_cpu_budget.py",
)
budget = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(budget)


def test_two_batches_share_under_thirty_percent_of_128_cpus():
    selected = budget.cpu_set(128, set(range(128)), 30)
    assert len(selected) == 38
    assert len(selected) / 128 < 0.30


def test_budget_respects_restricted_noncontiguous_affinity():
    assert budget.cpu_set(128, {40, 42, 99}, 30) == {40, 42, 99}


@pytest.mark.parametrize("percent", [0, -1, 101, float("nan"), float("inf")])
def test_invalid_percentage_is_rejected(percent):
    with pytest.raises(ValueError):
        budget.cpu_set(128, set(range(128)), percent)


def test_sub_core_budget_requires_a_quota():
    with pytest.raises(ValueError, match="quota"):
        budget.cpu_set(2, {0, 1}, 30)


def test_process_tree_handles_spaces_and_parentheses(tmp_path):
    for pid, ppid in [(100, 1), (200, 100), (300, 200), (400, 1)]:
        directory = tmp_path / str(pid)
        directory.mkdir()
        (directory / "stat").write_text(f"{pid} (python (worker)) S {ppid} 0 0")
    assert budget.process_tree([100], tmp_path) == [100, 200, 300]


def test_launch_reuses_one_affinity_and_overrides_inherited_thread_settings(monkeypatch):
    affinity = []
    calls = []
    monkeypatch.setattr(budget.os, "sched_setaffinity", lambda pid, cpus: affinity.append((pid, cpus)),
                        raising=False)
    monkeypatch.setattr(budget.os, "killpg", lambda *a: None, raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "128")
    monkeypatch.setattr(budget.shutil, "which", lambda name: "/venv/bin/audio-data")

    def spawn(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(pid=100 + len(calls), wait=lambda **k: 0)

    monkeypatch.setattr(budget.subprocess, "Popen", spawn)
    selected = set(range(38))
    assert budget.launch(["batch-a", "batch-b"], ["runs/a", "runs/b"], selected) == 0
    assert affinity == [(0, selected)]
    assert len(calls) == 2
    for index, (command, kwargs) in enumerate(calls):
        assert command[-2:] == ["--resume", ["runs/a", "runs/b"][index]]
        assert kwargs["start_new_session"] is True
        assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
        assert kwargs["env"]["OPENBLAS_NUM_THREADS"] == "1"
        assert kwargs["env"]["OMP_WAIT_POLICY"] == "PASSIVE"


def test_attach_limits_existing_threads_and_descendants(tmp_path, monkeypatch):
    for pid, tids in [(100, [100, 101]), (200, [200])]:
        for tid in tids:
            (tmp_path / str(pid) / "task" / str(tid)).mkdir(parents=True)
    monkeypatch.setattr(budget, "Path", lambda path: tmp_path / path.removeprefix("/proc/"))
    monkeypatch.setattr(budget, "process_tree", lambda roots: [100, 200])
    monkeypatch.setattr(budget.os, "getpid", lambda: 999)
    monkeypatch.setattr(budget.os, "kill", lambda *a: None)
    calls = []
    monkeypatch.setattr(budget.os, "sched_setaffinity", lambda tid, cpus: calls.append((tid, cpus)),
                        raising=False)
    selected = set(range(38))
    budget.attach([100], selected)
    assert len(calls) == 6  # Two passes include three existing threads each.
    assert {tid for tid, _ in calls} == {100, 101, 200}
    assert all(cpus == selected for _, cpus in calls)
