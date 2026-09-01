from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from audio_engine.core.artifacts import atomic_write_json
from audio_engine.core.catalog import utc_now


class TaskNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    command: list[str]
    depends_on: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    outputs: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not value or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for char in value):
            raise ValueError("task node id may contain only lowercase letters, digits, '_' and '-'")
        return value


class TaskDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    nodes: list[TaskNode]

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for char in value):
            raise ValueError("task name may contain only lowercase letters, digits, '_' and '-'")
        return value

    @model_validator(mode="after")
    def validate_graph(self) -> TaskDefinition:
        ids = [node.id for node in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("task node ids must be unique")
        known = set(ids)
        for node in self.nodes:
            missing = set(node.depends_on) - known
            if missing:
                raise ValueError(f"node {node.id} has unknown dependencies: {sorted(missing)}")
            if not node.command:
                raise ValueError(f"node {node.id} command must not be empty")
        self.topological_order()
        return self

    def topological_order(self) -> list[TaskNode]:
        remaining = {node.id: node for node in self.nodes}
        done: set[str] = set()
        ordered: list[TaskNode] = []
        while remaining:
            ready = [
                node for node in self.nodes if node.id in remaining and set(node.depends_on) <= done
            ]
            if not ready:
                raise ValueError(f"task dependency cycle: {sorted(remaining)}")
            for node in ready:
                ordered.append(node)
                done.add(node.id)
                remaining.pop(node.id)
        return ordered


class NodeState(BaseModel):
    status: str = "pending"
    started_at: str | None = None
    finished_at: str | None = None
    return_code: int | None = None
    error: str | None = None


class TaskState(BaseModel):
    schema_version: str = "1.0"
    task_id: str
    name: str
    config_digest: str
    status: str
    created_at: str
    updated_at: str
    nodes: dict[str, NodeState]


def load_task(path: Path) -> TaskDefinition:
    return TaskDefinition.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


class TaskRunner:
    """Small local DAG runner with atomic state and node-level resume semantics."""

    def __init__(self, definition: TaskDefinition, *, runs_dir: Path = Path("runs/tasks")):
        self.definition = definition
        raw = json.dumps(definition.model_dump(), ensure_ascii=False, sort_keys=True)
        self.digest = hashlib.sha256(raw.encode()).hexdigest()
        self.task_id = f"task_{definition.name}_{self.digest[:12]}"
        self.run_dir = runs_dir / self.task_id
        self.state_path = self.run_dir / "state.json"
        self.events_path = self.run_dir / "events.jsonl"

    def load_or_create(self) -> TaskState:
        if self.state_path.exists():
            state = TaskState.model_validate_json(self.state_path.read_text(encoding="utf-8"))
            if state.config_digest != self.digest:
                raise ValueError(f"task state config mismatch: {self.task_id}")
            return state
        now = utc_now()
        state = TaskState(
            task_id=self.task_id,
            name=self.definition.name,
            config_digest=self.digest,
            status="pending",
            created_at=now,
            updated_at=now,
            nodes={node.id: NodeState() for node in self.definition.nodes},
        )
        self._save(state)
        atomic_write_json(self.run_dir / "task.json", self.definition.model_dump(mode="json"))
        return state

    def run(self) -> TaskState:
        state = self.load_or_create()
        state.status = "running"
        self._save(state)
        for node in self.definition.topological_order():
            node_state = state.nodes[node.id]
            if node_state.status == "succeeded" and self._outputs_exist(node):
                self._event("node_skipped", node.id, {"reason": "already_succeeded"})
                continue
            dependency_states = {dep: state.nodes[dep].status for dep in node.depends_on}
            if any(value != "succeeded" for value in dependency_states.values()):
                raise RuntimeError(
                    f"node {node.id} dependencies not successful: {dependency_states}"
                )
            node_dir = self.run_dir / "nodes" / node.id
            node_dir.mkdir(parents=True, exist_ok=True)
            node_state.status = "running"
            node_state.started_at = utc_now()
            node_state.error = None
            self._save(state)
            self._event("node_started", node.id, {"command": node.command})
            with (
                (node_dir / "stdout.log").open("w", encoding="utf-8") as stdout,
                (node_dir / "stderr.log").open("w", encoding="utf-8") as stderr,
            ):
                result = subprocess.run(
                    node.command,
                    env={**os.environ, **node.env, "AUDIO_DATA_TASK_ID": self.task_id},
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                )
            node_state.return_code = result.returncode
            node_state.finished_at = utc_now()
            if result.returncode != 0:
                node_state.status = "failed"
                node_state.error = f"command exited with code {result.returncode}"
                state.status = "failed"
                self._save(state)
                self._event("node_failed", node.id, {"return_code": result.returncode})
                raise RuntimeError(f"task node failed: {node.id}; logs={node_dir}")
            missing = [output for output in node.outputs if not Path(output).exists()]
            if missing:
                node_state.status = "failed"
                node_state.error = f"declared outputs missing: {missing}"
                state.status = "failed"
                self._save(state)
                self._event("node_failed", node.id, {"missing_outputs": missing})
                raise RuntimeError(node_state.error)
            node_state.status = "succeeded"
            self._save(state)
            self._event("node_succeeded", node.id, {"outputs": node.outputs})
        state.status = "succeeded"
        self._save(state)
        atomic_write_json(
            self.run_dir / "outputs.json",
            {node.id: node.outputs for node in self.definition.nodes if node.outputs},
        )
        return state

    @staticmethod
    def _outputs_exist(node: TaskNode) -> bool:
        return not node.outputs or all(Path(output).exists() for output in node.outputs)

    def _save(self, state: TaskState) -> None:
        state.updated_at = utc_now()
        atomic_write_json(self.state_path, state.model_dump(mode="json"))

    def _event(self, event: str, node_id: str, details: dict[str, Any]) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {"timestamp": utc_now(), "event": event, "node_id": node_id, **details},
                    ensure_ascii=False,
                )
                + "\n"
            )
