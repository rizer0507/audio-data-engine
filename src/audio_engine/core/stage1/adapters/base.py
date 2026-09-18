"""Family launch adapter protocol and shared helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from audio_engine.core.stage1.process import ServiceSession
from audio_engine.core.stage1.probe import ProbeResult
from audio_engine.core.stage1.runtime_config import Stage1RuntimeConfig


@dataclass(frozen=True)
class LaunchPlan:
    family: str
    argv: list[str]
    env: dict[str, str]
    client_env: dict[str, str]
    api_base: str | None
    port: int | None
    gpu: int | str
    served_model_name: str | None
    notes: list[str] = field(default_factory=list)
    kind: str = "vllm"

    def argv_shell(self) -> str:
        return " ".join(_shell_quote(part) for part in self.argv)


def _shell_quote(value: str) -> str:
    if not value:
        return "''"
    if all(ch.isalnum() or ch in "._@%+=:,/-" for ch in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


class FamilyAdapter(Protocol):
    family: str

    def plan(
        self,
        runtime: Stage1RuntimeConfig,
        *,
        gpu: int | str,
        port: int | None = None,
    ) -> LaunchPlan: ...

    def start(
        self,
        runtime: Stage1RuntimeConfig,
        *,
        gpu: int | str,
        port: int | None = None,
        session_dir: Path,
        attach_existing: bool = False,
        dry_run: bool = False,
    ) -> ServiceSession: ...

    def probe(
        self,
        runtime: Stage1RuntimeConfig,
        *,
        session: ServiceSession | None = None,
        api_base: str | None = None,
        audio_path: str | Path,
        gpu: int | str | None = None,
    ) -> ProbeResult: ...

    def stop(self, session: ServiceSession) -> None: ...

    def check_paths(self, runtime: Stage1RuntimeConfig) -> list[str]: ...


def ensure_path_exists(path: Path, label: str, errors: list[str]) -> None:
    if not path or not str(path):
        errors.append(f"{label}: 路径为空")
        return
    if not path.exists():
        errors.append(f"{label} 不存在: {path}")


def client_api_base(host: str, port: int) -> str:
    # Bind host may be 0.0.0.0; clients always use loopback or explicit client_host.
    connect_host = "127.0.0.1" if host in {"0.0.0.0", "::", "[::]"} else host
    return f"http://{connect_host}:{port}"


def dump_plan(plan: LaunchPlan) -> dict[str, Any]:
    return {
        "family": plan.family,
        "kind": plan.kind,
        "gpu": plan.gpu,
        "port": plan.port,
        "api_base": plan.api_base,
        "served_model_name": plan.served_model_name,
        "argv": plan.argv,
        "argv_shell": plan.argv_shell(),
        "env": plan.env,
        "client_env": plan.client_env,
        "notes": plan.notes,
    }
