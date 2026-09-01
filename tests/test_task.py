from __future__ import annotations

import sys
from pathlib import Path

import pytest

from audio_engine.core.task import TaskDefinition, TaskRunner


def test_task_dag_runs_in_dependency_order_and_resumes(tmp_path: Path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    definition = TaskDefinition.model_validate(
        {
            "name": "workflow",
            "nodes": [
                {
                    "id": "first",
                    "command": [
                        sys.executable,
                        "-c",
                        f"from pathlib import Path; Path(r'{first}').write_text('1')",
                    ],
                    "outputs": [str(first)],
                },
                {
                    "id": "second",
                    "depends_on": ["first"],
                    "command": [
                        sys.executable,
                        "-c",
                        f"from pathlib import Path; Path(r'{second}').write_text('2')",
                    ],
                    "outputs": [str(second)],
                },
            ],
        }
    )
    runner = TaskRunner(definition, runs_dir=tmp_path / "runs")
    first_state = runner.run()
    assert first_state.status == "succeeded"
    first.write_text("do-not-overwrite", encoding="utf-8")
    second_state = runner.run()
    assert second_state.status == "succeeded"
    assert first.read_text() == "do-not-overwrite"
    assert "node_skipped" in runner.events_path.read_text()


def test_task_rejects_cycles():
    with pytest.raises(ValueError, match="cycle"):
        TaskDefinition.model_validate(
            {
                "name": "cycle",
                "nodes": [
                    {"id": "a", "command": ["true"], "depends_on": ["b"]},
                    {"id": "b", "command": ["true"], "depends_on": ["a"]},
                ],
            }
        )


def test_task_records_failed_node_and_stops(tmp_path: Path):
    definition = TaskDefinition.model_validate(
        {
            "name": "failure",
            "nodes": [
                {"id": "bad", "command": [sys.executable, "-c", "raise SystemExit(7)"]},
                {
                    "id": "never",
                    "command": [sys.executable, "-c", "raise SystemExit(0)"],
                    "depends_on": ["bad"],
                },
            ],
        }
    )
    runner = TaskRunner(definition, runs_dir=tmp_path / "runs")
    with pytest.raises(RuntimeError, match="bad"):
        runner.run()
    state = runner.load_or_create()
    assert state.status == "failed"
    assert state.nodes["bad"].return_code == 7
    assert state.nodes["never"].status == "pending"


def test_task_rejects_path_traversal_names():
    with pytest.raises(ValueError, match="task name"):
        TaskDefinition.model_validate({"name": "../bad", "nodes": []})
