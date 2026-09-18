"""Stage-1 job request, state persistence, and isolation directories."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from audio_engine.core.artifacts import atomic_write_json
from audio_engine.core.catalog import current_git_commit, utc_now
from audio_engine.core.dataset_v3.audit_plan import digest_payload
from audio_engine.core.stage1.cache_policy import (
    FAMILY_RUN_ALIASES,
    REQUIRED_FAMILIES,
    all_run_aliases,
)
from audio_engine.core.stage1.digests import write_json

_BATCH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
SELECTION_RULE = "selection_five_class_v2_2_auto_noise"
CLASSIFY_PIPELINE = "pipelines/classify_dataset_five_class_v2_2_auto_noise.yaml"
SELECTION_TEMPLATE = "configs/selection/zh_asr_five_class_v2_2_auto_noise.yaml"
DATASET_TEMPLATE = "configs/datasets/zh_asr_v3_three_family.yaml"
CLEAN_PIPELINE = "pipelines/data_cleaning_source_A.yaml"
PREPARE_PIPELINE = "pipelines/prepare_dataset_v3.yaml"
ATTACH_PIPELINE = "pipelines/attach_asr_v3.yaml"


@dataclass
class Stage1JobRequest:
    batch: str
    source: str
    source_kind: str  # directory | registered
    models: dict[str, str]  # family -> path
    runtime_config: str
    gpus: list[str]
    jobs_root: str = "runs/stage1/jobs"
    catalog_dir: str = "data/catalog"
    max_xlsx_rows: int = 20000
    attach_existing_services: bool = False

    def validate(self) -> None:
        if not _BATCH_RE.fullmatch(self.batch):
            raise ValueError(f"非法 batch 名: {self.batch}")
        missing = [name for name in REQUIRED_FAMILIES if name not in self.models]
        if missing:
            raise ValueError(f"--model 缺少家族: {missing}（须同时指定 qwen/glm/sensevoice）")
        extra = sorted(set(self.models) - set(REQUIRED_FAMILIES))
        if extra:
            raise ValueError(f"不支持的 --model 家族: {extra}")
        if not self.gpus:
            raise ValueError("--gpus 不能为空；须显式传入授权卡，禁止默认猜测")
        if not str(self.source).strip():
            raise ValueError("--source 不能为空")

    def config_digest(self) -> str:
        payload = {
            "batch": self.batch,
            "source": self.source,
            "source_kind": self.source_kind,
            "models": {k: self.models[k] for k in sorted(self.models)},
            "runtime_config": self.runtime_config,
            "gpus": list(self.gpus),
            "selection_rule": SELECTION_RULE,
            "classify_pipeline": CLASSIFY_PIPELINE,
            "max_xlsx_rows": self.max_xlsx_rows,
        }
        return digest_payload(payload)

    def job_id(self) -> str:
        return f"stage1_{self.batch}_{self.config_digest()[:12]}"


@dataclass
class StageState:
    status: str = "pending"  # pending|running|succeeded|failed|skipped|blocked|needs_attention
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    outputs: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)
    attempt_count: int = 0
    max_attempts: int = 3
    attempts: list[dict[str, Any]] = field(default_factory=list)
    circuit_open: bool = False
    error_kind: str | None = None
    checkpoint: dict[str, Any] = field(default_factory=dict)


@dataclass
class Stage1JobState:
    schema_version: str
    job_id: str
    batch: str
    status: str
    created_at: str
    updated_at: str
    config_digest: str
    request: dict[str, Any]
    git_commit: str | None
    stages: dict[str, StageState]
    pid: int | None = None
    worker_token: str | None = None
    error: str | None = None
    outputs: dict[str, Any] = field(default_factory=dict)
    reconcile: dict[str, Any] = field(default_factory=dict)
    needs_attention: list[dict[str, Any]] = field(default_factory=list)
    runtime_faults: list[dict[str, Any]] = field(default_factory=list)
    hardcase_review: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Stage1JobState:
        stages: dict[str, StageState] = {}
        for key, value in (data.get("stages") or {}).items():
            if isinstance(value, dict):
                known = {
                    field_name
                    for field_name in StageState.__dataclass_fields__  # type: ignore[attr-defined]
                }
                filtered = {k: v for k, v in value.items() if k in known}
                stages[key] = StageState(**filtered)
            else:
                stages[key] = value
        return cls(
            schema_version=str(data.get("schema_version") or "1.0"),
            job_id=str(data["job_id"]),
            batch=str(data["batch"]),
            status=str(data["status"]),
            created_at=str(data["created_at"]),
            updated_at=str(data["updated_at"]),
            config_digest=str(data["config_digest"]),
            request=dict(data.get("request") or {}),
            git_commit=data.get("git_commit"),
            stages=stages,
            pid=data.get("pid"),
            worker_token=data.get("worker_token"),
            error=data.get("error"),
            outputs=dict(data.get("outputs") or {}),
            reconcile=dict(data.get("reconcile") or {}),
            needs_attention=list(data.get("needs_attention") or []),
            runtime_faults=list(data.get("runtime_faults") or []),
            hardcase_review=list(data.get("hardcase_review") or []),
        )


def default_stage_names() -> list[str]:
    stages = ["precheck", "freeze_snapshot", "clean"]
    for family in REQUIRED_FAMILIES:
        stages.append(f"serve_start_{family}")
        for alias in FAMILY_RUN_ALIASES[family]:
            stages.append(f"asr_{alias}")
            stages.append(f"register_{alias}")
        stages.append(f"serve_stop_{family}")
    stages.extend(
        [
            "write_dataset_config",
            "prepare",
            "attach",
            "classify",
            "export",
            "reconcile",
        ]
    )
    return stages


def job_dir(jobs_root: Path | str, job_id: str) -> Path:
    return Path(jobs_root) / job_id


def create_job(request: Stage1JobRequest) -> tuple[Path, Stage1JobState]:
    request.validate()
    digest = request.config_digest()
    job_id = request.job_id()
    root = job_dir(request.jobs_root, job_id)
    state_path = root / "state.json"
    if state_path.is_file():
        existing = Stage1JobState.from_dict(
            json.loads(state_path.read_text(encoding="utf-8"))
        )
        if existing.config_digest != digest:
            raise ValueError(
                f"job_id 冲突但配置不同: {job_id}；请更换 batch 或模型路径以隔离"
            )
        if existing.status in {"running", "pending"}:
            raise ValueError(
                f"相同配置的任务已存在且未结束: {job_id} status={existing.status}；"
                "拒绝重复提交占卡。可用 stage1 status 查询。"
            )
        if existing.status == "succeeded":
            raise ValueError(
                f"相同配置已成功完成: {job_id}；拒绝覆盖。如需重跑请换配置或新 batch。"
            )
        if existing.status in {"failed", "needs_attention", "cancelled"}:
            raise ValueError(
                f"任务已存在 status={existing.status}: {job_id}；"
                "请使用 stage1 resume 或 stage1 retry，不要重复 run 覆盖。"
            )
    root.mkdir(parents=True, exist_ok=True)
    now = utc_now()
    stages = {name: StageState() for name in default_stage_names()}
    state = Stage1JobState(
        schema_version="1.1",
        job_id=job_id,
        batch=request.batch,
        status="pending",
        created_at=now,
        updated_at=now,
        config_digest=digest,
        request=asdict(request),
        git_commit=current_git_commit(),
        stages=stages,
    )
    save_job_state(root, state)
    write_json(root / "request.json", asdict(request))
    write_json(
        root / "cache_boundary.json",
        {
            "independent_dual_run": (
                "不同 --asr-run / transcript_key / 产物路径 / execution_id"
            ),
            "same_run_resume": "同 stage succeeded 且产物+identity 仍在则跳过",
            "forbidden": "禁止复制 parquet 或复用 execution_id 冒充双跑",
            "aliases": all_run_aliases(),
        },
    )
    return root, state


def save_job_state(root: Path, state: Stage1JobState) -> None:
    state.updated_at = utc_now()
    atomic_write_json(root / "state.json", state.to_dict())


def load_job_state(root: Path) -> Stage1JobState:
    path = root / "state.json"
    if not path.is_file():
        raise FileNotFoundError(f"job state 不存在: {path}")
    return Stage1JobState.from_dict(json.loads(path.read_text(encoding="utf-8")))


def append_event(root: Path, event: str, **payload: Any) -> None:
    line = json.dumps(
        {"at": utc_now(), "event": event, **payload},
        ensure_ascii=False,
    )
    with (root / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def parse_model_option(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise ValueError(f"--model 格式应为 family=/path/to/weights，收到: {raw}")
    family, path = raw.split("=", 1)
    family = family.strip().lower()
    path = path.strip()
    if family not in REQUIRED_FAMILIES:
        raise ValueError(f"未知家族 {family}；支持 {REQUIRED_FAMILIES}")
    if not path:
        raise ValueError(f"--model {family}= 路径为空")
    return family, path


def parse_gpus_option(raw: str) -> list[str]:
    items = [part.strip() for part in str(raw).replace(";", ",").split(",") if part.strip()]
    if not items:
        raise ValueError("--gpus 解析结果为空")
    return items


def resolve_source_arg(source: str) -> tuple[str, str]:
    """Return (source_kind, normalized_source)."""
    path = Path(source)
    if path.is_dir():
        return "directory", str(path.resolve())
    try:
        from audio_engine.core.source import lookup_source

        entry = lookup_source(source)
        registered_path = entry.get("path")
        if registered_path and Path(str(registered_path)).is_dir():
            return "directory", str(Path(str(registered_path)).resolve())
        return "registered", source
    except KeyError as exc:
        raise ValueError(
            f"--source 既不是本地目录，也不是 resources/manifest.yaml 中的来源: {source}"
        ) from exc
