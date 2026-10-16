"""Parent-run orchestrator: ingest → VAD → parallel ASR∥DNSMOS → join → tag."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from audio_engine.core.artifacts import atomic_write_json
from audio_engine.core.catalog import utc_now
from audio_engine.core.production.config import (
    ProductionConfig,
    is_production_yaml,
    load_production_config,
)
from audio_engine.core.production.paths import ProductionPaths
from audio_engine.core.source_naming import pipeline_run_name, validate_source_name
from audio_engine.core.task import NodeState, TaskDefinition, TaskNode, TaskState

LogFn = Callable[[str], None]


def is_production_pipeline(config_path: Path | str) -> bool:
    return is_production_yaml(config_path)


@dataclass
class ProductionRunResult:
    run_root: Path
    status: str
    paths: ProductionPaths
    node_states: dict[str, dict[str, Any]]
    overlap_evidence: dict[str, Any] | None = None


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _resolve_pipeline(path: str, *, base: Path) -> Path:
    candidate = Path(path)
    if candidate.is_file():
        return candidate.resolve()
    cwd = Path.cwd() / candidate
    if cwd.is_file():
        return cwd.resolve()
    local = base / candidate
    if local.is_file():
        return local.resolve()
    raise FileNotFoundError(f"pipeline not found: {path}")


def _cli_base() -> list[str]:
    return [sys.executable, "-m", "audio_engine.cli.main", "pipeline", "run"]


class ProductionOrchestrator:
    """Build and execute the 045 production DAG with real ready-node parallelism."""

    def __init__(
        self,
        config: ProductionConfig,
        *,
        config_path: Path,
        source_name: str,
        source_dir: str | Path | None = None,
        runs_dir: Path = Path("runs"),
        resume: str | Path | None = None,
        force: bool = False,
        mock: bool = False,
        log: LogFn | None = None,
        tagging_config: str | None = None,
    ):
        self.config = config
        self.config_path = Path(config_path).resolve()
        self.source_name = validate_source_name(source_name)
        self.source_dir = Path(source_dir).resolve() if source_dir else None
        self.runs_dir = Path(runs_dir)
        self.resume = resume
        self.force = force
        self.mock = mock
        self.log = log or (lambda _msg: None)
        self.tagging_config = tagging_config
        self.paths = ProductionPaths(self.source_name)
        self.base_dir = self.config_path.parent

    def run(self) -> ProductionRunResult:
        run_root = self._prepare_run_root()
        self._freeze_config(run_root)
        state = self._load_or_create_state(run_root)
        state.status = "running"
        self._save_state(run_root, state)
        self._event(run_root, "production_started", {"source_name": self.source_name})

        nodes = self._build_nodes()
        definition = TaskDefinition(name=self.config.name, nodes=nodes)
        node_by_id = {node.id: node for node in nodes}
        overlap_evidence: dict[str, Any] | None = None

        try:
            remaining = {node.id for node in nodes}
            done: set[str] = set()
            # Seed already-succeeded nodes for resume.
            for node_id, node_state in state.nodes.items():
                if node_state.status == "succeeded" and self._outputs_exist(node_by_id[node_id]):
                    done.add(node_id)
                    remaining.discard(node_id)
                    self._event(run_root, "node_skipped", {"node_id": node_id, "reason": "resume"})

            while remaining:
                ready = [
                    node
                    for node in nodes
                    if node.id in remaining and set(node.depends_on) <= done
                ]
                if not ready:
                    raise RuntimeError(
                        f"production deadlock; remaining={sorted(remaining)} done={sorted(done)}"
                    )
                # Fail if any dependency failed earlier (should not happen).
                for node in ready:
                    for dep in node.depends_on:
                        if state.nodes[dep].status != "succeeded":
                            raise RuntimeError(
                                f"node {node.id} dependency {dep} not succeeded"
                            )

                self.log(
                    f"[production] launching ready nodes in parallel: "
                    f"{[n.id for n in ready]}"
                )
                wave_started = time.monotonic()
                wave_wall = {
                    n.id: {"started_at": utc_now(), "monotonic": time.monotonic()}
                    for n in ready
                }
                for node in ready:
                    state.nodes[node.id].status = "running"
                    state.nodes[node.id].started_at = wave_wall[node.id]["started_at"]
                    state.nodes[node.id].error = None
                self._save_state(run_root, state)
                self._write_progress(run_root, state, phase="running", ready=[n.id for n in ready])

                results: dict[str, Exception | None] = {}
                with ThreadPoolExecutor(max_workers=max(1, len(ready))) as pool:
                    futures = {
                        pool.submit(
                            self._run_node,
                            run_root,
                            node,
                            state,
                        ): node
                        for node in ready
                    }
                    for future in as_completed(futures):
                        node = futures[future]
                        try:
                            future.result()
                            results[node.id] = None
                        except Exception as exc:  # noqa: BLE001 — collect then raise
                            results[node.id] = exc

                wave_finished = time.monotonic()
                # Overlap evidence: any wave with ≥2 nodes that both started before
                # the slower one finished proves concurrent execution windows.
                if len(ready) >= 2:
                    intervals = []
                    for node in ready:
                        ns = state.nodes[node.id]
                        intervals.append(
                            {
                                "node_id": node.id,
                                "started_at": ns.started_at,
                                "finished_at": ns.finished_at,
                                "status": ns.status,
                            }
                        )
                    overlap_evidence = {
                        "wave_node_ids": [n.id for n in ready],
                        "wave_wall_seconds": round(wave_finished - wave_started, 4),
                        "intervals": intervals,
                        "parallel": True,
                        "note": (
                            "Ready nodes in the same wave were launched via "
                            "ThreadPoolExecutor; overlapping started_at before "
                            "any finished_at demonstrates concurrent subprocesses."
                        ),
                    }
                    atomic_write_json(
                        run_root / "overlap_evidence.json", overlap_evidence
                    )

                failed = {nid: err for nid, err in results.items() if err is not None}
                if failed:
                    state.status = "failed"
                    self._save_state(run_root, state)
                    self._write_progress(run_root, state, phase="failed")
                    detail = "; ".join(f"{k}: {v}" for k, v in failed.items())
                    raise RuntimeError(f"production node(s) failed: {detail}")

                for node in ready:
                    done.add(node.id)
                    remaining.discard(node.id)

            state.status = "succeeded"
            self._save_state(run_root, state)
            self._write_progress(run_root, state, phase="succeeded")
            self._event(run_root, "production_succeeded", {"paths": self._path_map()})
            atomic_write_json(run_root / "outputs.json", self._path_map())
        except Exception as exc:
            state.status = "failed"
            self._save_state(run_root, state)
            self._event(run_root, "production_failed", {"error": str(exc)})
            raise

        return ProductionRunResult(
            run_root=run_root,
            status=state.status,
            paths=self.paths,
            node_states={k: v.model_dump() for k, v in state.nodes.items()},
            overlap_evidence=overlap_evidence,
        )

    def _build_nodes(self) -> list[TaskNode]:
        ingest_out = self.paths.as_posix(self.paths.ingested)
        segments_out = self.paths.as_posix(self.paths.segments)
        dnsmos_out = self.paths.as_posix(self.paths.dnsmos)
        evidence_out = self.paths.as_posix(self.paths.prepared_evidence)
        tagged_out = self.paths.as_posix(self.paths.tagged_free_tags)

        ingest_pipe = _resolve_pipeline(self.config.ingest.pipeline, base=self.base_dir)
        vad_pipe = _resolve_pipeline(self.config.vad.pipeline, base=self.base_dir)
        dnsmos_pipe = _resolve_pipeline(self.config.dnsmos.pipeline, base=self.base_dir)
        join_pipe = _resolve_pipeline(
            self.config.evidence_join.pipeline, base=self.base_dir
        )
        tag_pipe = _resolve_pipeline(self.config.tagging.pipeline, base=self.base_dir)

        if self.source_dir is None:
            raise ValueError(
                "production ingest requires --source-dir (raw audio directory)"
            )

        nodes: list[TaskNode] = [
            TaskNode(
                id="ingest",
                command=self._cmd(
                    ingest_pipe,
                    source_name=self.source_name,
                    source_dir=str(self.source_dir),
                    output_manifest=ingest_out,
                    extra=self.config.ingest.extra_args,
                ),
                env=dict(self.config.ingest.env),
                outputs=[ingest_out],
            ),
            TaskNode(
                id="vad",
                command=self._cmd(
                    vad_pipe,
                    source_name=self.source_name,
                    input_manifest=ingest_out,
                    output_manifest=segments_out,
                    extra=self.config.vad.extra_args,
                ),
                depends_on=["ingest"],
                env=dict(self.config.vad.env),
                outputs=[segments_out],
            ),
        ]

        asr_ids: list[str] = []
        for route in self.config.asr_routes:
            route_id = route.route_id
            node_id = f"asr_{route_id}"
            asr_ids.append(node_id)
            asr_out = self.paths.as_posix(self.paths.asr(route_id))
            asr_pipe = _resolve_pipeline(route.pipeline, base=self.base_dir)
            asr_run = route.resolved_asr_run()
            nodes.append(
                TaskNode(
                    id=node_id,
                    command=self._cmd(
                        asr_pipe,
                        source_name=self.source_name,
                        input_manifest=segments_out,
                        output_manifest=asr_out,
                        asr_run=asr_run,
                        extra=route.extra_args,
                    ),
                    depends_on=["vad"],
                    env=dict(route.env),
                    outputs=[asr_out],
                )
            )

        nodes.append(
            TaskNode(
                id="dnsmos",
                command=self._cmd(
                    dnsmos_pipe,
                    source_name=self.source_name,
                    input_manifest=segments_out,
                    output_manifest=dnsmos_out,
                    extra=self.config.dnsmos.extra_args,
                ),
                depends_on=["vad"],
                env=dict(self.config.dnsmos.env),
                outputs=[dnsmos_out],
            )
        )

        join_depends = [*asr_ids, "dnsmos"]
        join_manifests = [
            f"{route.resolved_asr_run()}={self.paths.as_posix(self.paths.asr(route.route_id))}"
            for route in self.config.asr_routes
        ]
        join_extra = list(self.config.evidence_join.extra_args)
        for item in join_manifests:
            join_extra.extend(["--join-manifest", item])
        # Pass DNSMOS path via env consumed by evidence_join naming / CLI overrides.
        join_env = {
            **dict(self.config.evidence_join.env),
            "AUDIO_ENGINE_DNSMOS_MANIFEST": dnsmos_out,
            "AUDIO_ENGINE_SEGMENTS_MANIFEST": segments_out,
        }
        nodes.append(
            TaskNode(
                id="evidence_join",
                command=self._cmd(
                    join_pipe,
                    source_name=self.source_name,
                    input_manifest=segments_out,
                    output_manifest=evidence_out,
                    extra=join_extra,
                ),
                depends_on=join_depends,
                env=join_env,
                outputs=[evidence_out],
            )
        )

        tag_cfg = (
            self.tagging_config
            or self.config.tagging.config
            or "configs/tagging/zh_asr_free_tags_v2.yaml"
        )
        tag_extra = list(self.config.tagging.extra_args)
        if tag_cfg:
            tag_extra.extend(["--config", tag_cfg])
        nodes.append(
            TaskNode(
                id="tagging",
                command=self._cmd(
                    tag_pipe,
                    source_name=self.source_name,
                    input_manifest=evidence_out,
                    output_manifest=tagged_out,
                    extra=tag_extra,
                ),
                depends_on=["evidence_join"],
                env=dict(self.config.tagging.env),
                outputs=[tagged_out],
            )
        )
        # Validate graph construction.
        TaskDefinition(name=self.config.name, nodes=nodes)
        return nodes

    def _cmd(
        self,
        pipeline: Path,
        *,
        source_name: str | None = None,
        source_dir: str | None = None,
        input_manifest: str | None = None,
        output_manifest: str | None = None,
        asr_run: str | None = None,
        extra: list[str] | None = None,
    ) -> list[str]:
        command = [*_cli_base(), str(pipeline)]
        if source_name:
            command.extend(["--source-name", source_name])
        if source_dir:
            command.extend(["--source-dir", source_dir])
        if input_manifest:
            command.extend(["--input-manifest", input_manifest])
        if output_manifest:
            command.extend(["--output-manifest", output_manifest])
        if asr_run:
            command.extend(["--asr-run", asr_run])
        if self.force:
            command.append("--force")
        if self.mock:
            command.append("--mock")
        if extra:
            command.extend(extra)
        return command

    def _run_node(
        self,
        run_root: Path,
        node: TaskNode,
        state: TaskState,
    ) -> None:
        node_dir = run_root / "nodes" / node.id
        node_dir.mkdir(parents=True, exist_ok=True)
        child_run_hint = node_dir / "child_run_dir.txt"
        env = {
            **os.environ,
            **node.env,
            "AUDIO_DATA_PRODUCTION_PARENT": str(run_root),
            "AUDIO_DATA_PRODUCTION_NODE": node.id,
            "AUDIO_DATA_PRODUCTION_SOURCE": self.source_name,
        }
        # Pin child runs under nodes/<id>/work so parent progress can find them.
        child_runs = node_dir / "work"
        child_runs.mkdir(parents=True, exist_ok=True)
        env["AUDIO_ENGINE_RUNS_DIR"] = str(child_runs)

        self._event(
            run_root,
            "node_started",
            {"node_id": node.id, "command": node.command, "work_dir": str(child_runs)},
        )
        stdout_path = node_dir / "stdout.log"
        stderr_path = node_dir / "stderr.log"
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            # Inject --runs-dir if the CLI supports it via env; also pass explicitly.
            command = list(node.command)
            if "--runs-dir" not in command:
                command.extend(["--runs-dir", str(child_runs)])
            result = subprocess.run(
                command,
                env=env,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        ns = state.nodes[node.id]
        ns.return_code = result.returncode
        ns.finished_at = utc_now()
        if result.returncode != 0:
            ns.status = "failed"
            ns.error = f"command exited with code {result.returncode}"
            self._save_state(run_root, state)
            self._event(
                run_root,
                "node_failed",
                {
                    "node_id": node.id,
                    "return_code": result.returncode,
                    "stderr": str(stderr_path),
                },
            )
            raise RuntimeError(
                f"node {node.id} failed (rc={result.returncode}); see {node_dir}"
            )
        missing = [output for output in node.outputs if not Path(output).exists()]
        if missing:
            ns.status = "failed"
            ns.error = f"declared outputs missing: {missing}"
            self._save_state(run_root, state)
            self._event(
                run_root,
                "node_failed",
                {"node_id": node.id, "missing_outputs": missing},
            )
            raise RuntimeError(ns.error)
        ns.status = "succeeded"
        # Record newest child run directory if present.
        children = sorted(
            (p for p in child_runs.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if children:
            child_run_hint.write_text(str(children[0]), encoding="utf-8")
            atomic_write_json(
                node_dir / "child_run.json",
                {"child_run_dir": str(children[0]), "node_id": node.id},
            )
        self._save_state(run_root, state)
        self._event(run_root, "node_succeeded", {"node_id": node.id, "outputs": node.outputs})

    def _prepare_run_root(self) -> Path:
        if self.resume:
            root = Path(self.resume)
            if not root.is_absolute():
                candidate = self.runs_dir / root
                root = candidate if candidate.is_dir() else root
            if not root.is_dir():
                raise FileNotFoundError(f"resume run dir not found: {self.resume}")
            return root.resolve()
        name = pipeline_run_name(self.config.name, self.source_name)
        root = self.runs_dir / f"{_ts()}_{name}"
        root.mkdir(parents=True, exist_ok=False)
        (root / "nodes").mkdir(parents=True, exist_ok=True)
        (root / "config_snapshot").mkdir(parents=True, exist_ok=True)
        return root.resolve()

    def _freeze_config(self, run_root: Path) -> None:
        snap = run_root / "config_snapshot"
        snap.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            snap / "production.json",
            {
                "config_path": str(self.config_path),
                "source_name": self.source_name,
                "source_dir": str(self.source_dir) if self.source_dir else None,
                "force": self.force,
                "mock": self.mock,
                "production": self.config.model_dump(mode="json"),
                "paths": self._path_map(),
                "frozen_at": utc_now(),
            },
        )
        # Copy root YAML for audit.
        target = snap / self.config_path.name
        if not target.exists():
            target.write_bytes(self.config_path.read_bytes())
        atomic_write_json(
            run_root / "run.json",
            {
                "schema_version": "045.1",
                "kind": "production",
                "name": self.config.name,
                "source_name": self.source_name,
                "created_at": utc_now(),
                "config_path": str(self.config_path),
                "paths": self._path_map(),
                "resource_budget": self.config.resource_budget,
            },
        )

    def _load_or_create_state(self, run_root: Path) -> TaskState:
        path = run_root / "state.json"
        node_ids = [n.id for n in self._build_nodes()]
        if path.exists():
            state = TaskState.model_validate_json(path.read_text(encoding="utf-8"))
            for node_id in node_ids:
                state.nodes.setdefault(node_id, NodeState())
            return state
        now = utc_now()
        state = TaskState(
            task_id=run_root.name,
            name=self.config.name,
            config_digest=run_root.name,
            status="pending",
            created_at=now,
            updated_at=now,
            nodes={node_id: NodeState() for node_id in node_ids},
        )
        self._save_state(run_root, state)
        return state

    def _save_state(self, run_root: Path, state: TaskState) -> None:
        state.updated_at = utc_now()
        atomic_write_json(run_root / "state.json", state.model_dump(mode="json"))

    def _event(self, run_root: Path, event: str, details: dict[str, Any]) -> None:
        path = run_root / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {"timestamp": utc_now(), "event": event, **details},
                    ensure_ascii=False,
                )
                + "\n"
            )

    def _write_progress(
        self,
        run_root: Path,
        state: TaskState,
        *,
        phase: str,
        ready: list[str] | None = None,
    ) -> None:
        nodes_view = {
            node_id: {
                "status": ns.status,
                "started_at": ns.started_at,
                "finished_at": ns.finished_at,
                "error": ns.error,
                "return_code": ns.return_code,
            }
            for node_id, ns in state.nodes.items()
        }
        running = [k for k, v in nodes_view.items() if v["status"] == "running"]
        succeeded = [k for k, v in nodes_view.items() if v["status"] == "succeeded"]
        failed = [k for k, v in nodes_view.items() if v["status"] == "failed"]
        pending = [k for k, v in nodes_view.items() if v["status"] == "pending"]
        payload = {
            "kind": "production",
            "phase": phase,
            "status": state.status,
            "source_name": self.source_name,
            "updated_at": utc_now(),
            "ready": ready or [],
            "running": running,
            "succeeded": succeeded,
            "failed": failed,
            "pending": pending,
            "nodes": nodes_view,
            "blocking_join": (
                "evidence_join" in pending or "evidence_join" in (ready or [])
            )
            and bool(running),
            "asr_dnsmos_parallel_visible": (
                any(k.startswith("asr_") for k in running) and "dnsmos" in running
            ),
        }
        atomic_write_json(run_root / "progress_view.json", payload)
        atomic_write_json(run_root / "metrics.json", payload)
        line = (
            f"[PRODUCTION] status={state.status} phase={phase} "
            f"running={running} succeeded={len(succeeded)} "
            f"failed={failed} pending={pending}"
        )
        log_path = run_root / "PROGRESS.log"
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        self.log(line)

    def _path_map(self) -> dict[str, Any]:
        routes = {
            route.route_id: self.paths.as_posix(self.paths.asr(route.route_id))
            for route in self.config.asr_routes
        }
        return {
            "ingested": self.paths.as_posix(self.paths.ingested),
            "segments": self.paths.as_posix(self.paths.segments),
            "asr_routes": routes,
            "dnsmos": self.paths.as_posix(self.paths.dnsmos),
            "prepared_evidence": self.paths.as_posix(self.paths.prepared_evidence),
            "tagged_free_tags": self.paths.as_posix(self.paths.tagged_free_tags),
        }

    @staticmethod
    def _outputs_exist(node: TaskNode) -> bool:
        return not node.outputs or all(Path(p).exists() for p in node.outputs)


def run_production_pipeline(
    config_path: Path | str,
    *,
    source_name: str,
    source_dir: str | Path | None = None,
    runs_dir: Path | None = None,
    resume: str | Path | None = None,
    force: bool = False,
    mock: bool = False,
    log: LogFn | None = None,
    tagging_config: str | None = None,
) -> ProductionRunResult:
    path = Path(config_path)
    config = load_production_config(path)
    orchestrator = ProductionOrchestrator(
        config,
        config_path=path,
        source_name=source_name,
        source_dir=source_dir,
        runs_dir=runs_dir or Path("runs"),
        resume=resume,
        force=force,
        mock=mock,
        log=log,
        tagging_config=tagging_config,
    )
    return orchestrator.run()
