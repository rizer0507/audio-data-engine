"""Web service settings loaded from configs/web.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConnectionConfig:
    connection_id: str
    name: str
    provider: str
    base_url: str
    model: str
    model_revision: str
    api_key_env: str
    timeout_seconds: int = 120
    concurrency: int = 4

    def public_dict(self) -> dict[str, Any]:
        import os as _os

        return {
            "connection_id": self.connection_id,
            "name": self.name,
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "model_revision": self.model_revision,
            "api_key_env": self.api_key_env,
            "api_key_configured": bool(_os.environ.get(self.api_key_env)),
            "timeout_seconds": self.timeout_seconds,
            "concurrency": self.concurrency,
        }


@dataclass
class WorkerSettings:
    poll_interval_seconds: float = 2.0
    heartbeat_interval_seconds: float = 5.0
    lease_seconds: int = 30
    max_concurrent_jobs: int = 1
    cancel_grace_seconds: int = 15
    log_chunk_max_bytes: int = 65536


@dataclass
class WebSettings:
    project_root: Path
    config_path: Path
    bind_host: str = "127.0.0.1"
    bind_port: int = 8080
    sqlite_path: Path = field(default_factory=lambda: Path("var/web.sqlite3"))
    runs_dir: Path = field(default_factory=lambda: Path("runs/web"))
    plugins_config: Path = field(default_factory=lambda: Path("configs/plugins.yaml"))
    sources_manifest: Path = field(default_factory=lambda: Path("resources/manifest.yaml"))
    static_dir: Path = field(default_factory=lambda: Path("frontend/dist"))
    allowed_source_roots: list[Path] = field(default_factory=list)
    plan_ttl_seconds: int = 900
    worker: WorkerSettings = field(default_factory=WorkerSettings)
    models: list[ModelConnectionConfig] = field(default_factory=list)
    operator_id: str = "local_operator"
    revision: int = 1

    def resolve(self, path: str | Path) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p.resolve()
        return (self.project_root / p).resolve()

    def model_map(self) -> dict[str, ModelConnectionConfig]:
        return {m.connection_id: m for m in self.models}


def _as_path(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def load_web_settings(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> WebSettings:
    cfg_path = Path(config_path).resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"web config not found: {cfg_path}")
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("web.yaml root must be a mapping")

    env_root = os.environ.get("AUDIO_ENGINE_PROJECT_ROOT")
    root_value = project_root or raw.get("project_root") or env_root
    if not root_value:
        # Fall back to repo root inferred from config location: <root>/configs/web.yaml
        root_value = cfg_path.parent.parent
    root = Path(str(root_value)).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"project_root is not a directory: {root}")

    worker_raw = raw.get("worker") or {}
    worker = WorkerSettings(
        poll_interval_seconds=float(worker_raw.get("poll_interval_seconds", 2)),
        heartbeat_interval_seconds=float(worker_raw.get("heartbeat_interval_seconds", 5)),
        lease_seconds=int(worker_raw.get("lease_seconds", 30)),
        max_concurrent_jobs=int(worker_raw.get("max_concurrent_jobs", 1)),
        cancel_grace_seconds=int(worker_raw.get("cancel_grace_seconds", 15)),
        log_chunk_max_bytes=int(worker_raw.get("log_chunk_max_bytes", 65536)),
    )

    models: list[ModelConnectionConfig] = []
    for item in raw.get("models") or []:
        if not isinstance(item, dict):
            raise ValueError("models items must be mappings")
        models.append(
            ModelConnectionConfig(
                connection_id=str(item["connection_id"]),
                name=str(item.get("name") or item["connection_id"]),
                provider=str(item.get("provider") or "").strip().lower(),
                base_url=str(item.get("base_url") or ""),
                model=str(item.get("model") or ""),
                model_revision=str(item.get("model_revision") or ""),
                api_key_env=str(item.get("api_key_env") or ""),
                timeout_seconds=int(item.get("timeout_seconds") or 120),
                concurrency=int(item.get("concurrency") or 4),
            )
        )

    allowed = [_as_path(root, p) for p in (raw.get("allowed_source_roots") or [])]

    return WebSettings(
        project_root=root,
        config_path=cfg_path,
        bind_host=str(raw.get("bind_host") or "127.0.0.1"),
        bind_port=int(raw.get("bind_port") or 8080),
        sqlite_path=_as_path(root, raw.get("sqlite_path") or "var/web.sqlite3"),
        runs_dir=_as_path(root, raw.get("runs_dir") or "runs/web"),
        plugins_config=_as_path(root, raw.get("plugins_config") or "configs/plugins.yaml"),
        sources_manifest=_as_path(
            root, raw.get("sources_manifest") or "resources/manifest.yaml"
        ),
        static_dir=_as_path(root, raw.get("static_dir") or "frontend/dist"),
        allowed_source_roots=allowed,
        plan_ttl_seconds=int(raw.get("plan_ttl_seconds") or 900),
        worker=worker,
        models=models,
        operator_id=str(raw.get("operator_id") or "local_operator"),
    )


def save_models_to_web_yaml(settings: WebSettings, models: list[ModelConnectionConfig]) -> None:
    """Persist model connection list back to web.yaml (atomic replace)."""

    raw = yaml.safe_load(settings.config_path.read_text(encoding="utf-8")) or {}
    raw["models"] = [
        {
            "connection_id": m.connection_id,
            "name": m.name,
            "provider": m.provider,
            "base_url": m.base_url,
            "model": m.model,
            "model_revision": m.model_revision,
            "api_key_env": m.api_key_env,
            "timeout_seconds": m.timeout_seconds,
            "concurrency": m.concurrency,
        }
        for m in models
    ]
    text = yaml.safe_dump(raw, allow_unicode=True, sort_keys=False)
    tmp = settings.config_path.with_suffix(".yaml.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(settings.config_path)
    settings.models = models
    settings.revision += 1
