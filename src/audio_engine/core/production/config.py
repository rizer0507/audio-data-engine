"""Load and validate production DAG configuration (N ASR routes + independent DNSMOS)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from audio_engine.core.source_naming import validate_asr_run, validate_source_name


class AsrRouteConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route_id: str
    family: str
    pipeline: str
    asr_run: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    extra_args: list[str] = Field(default_factory=list)

    @field_validator("route_id")
    @classmethod
    def _route_id(cls, value: str) -> str:
        return validate_asr_run(value)

    @field_validator("family")
    @classmethod
    def _family(cls, value: str) -> str:
        text = str(value or "").strip().lower()
        if not text:
            raise ValueError("asr route family must be non-empty")
        return text

    @field_validator("pipeline")
    @classmethod
    def _pipeline(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("asr route pipeline must be non-empty")
        return text

    def resolved_asr_run(self) -> str:
        return validate_asr_run(self.asr_run or self.route_id)


class NodePipelineRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pipeline: str
    config: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    extra_args: list[str] = Field(default_factory=list)

    @field_validator("pipeline")
    @classmethod
    def _pipeline(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("pipeline path must be non-empty")
        return text


class ProductionConfig(BaseModel):
    """Config-driven production DAG; routes are not hard-coded to three families."""

    model_config = ConfigDict(extra="forbid")

    name: str = "audio_ingest_vad_asr_dnsmos_tags"
    ingest: NodePipelineRef
    vad: NodePipelineRef
    asr_routes: list[AsrRouteConfig]
    dnsmos: NodePipelineRef
    evidence_join: NodePipelineRef
    tagging: NodePipelineRef
    # Optional resource budget notes frozen into run.json (not enforced here).
    resource_budget: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("production name must be non-empty")
        return text

    @model_validator(mode="after")
    def _unique_routes(self) -> ProductionConfig:
        if not self.asr_routes:
            raise ValueError("production.asr_routes must contain at least one route")
        ids = [route.route_id for route in self.asr_routes]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate asr route_id values: {ids}")
        runs = [route.resolved_asr_run() for route in self.asr_routes]
        if len(runs) != len(set(runs)):
            raise ValueError(f"duplicate asr_run aliases: {runs}")
        return self


def load_production_config(path: Path | str) -> ProductionConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"production config must be a mapping: {path}")
    body = data.get("production")
    if body is None:
        raise ValueError(f"missing production: block in {path}")
    if not isinstance(body, dict):
        raise ValueError(f"production: must be a mapping in {path}")
    name = str(data.get("name") or body.get("name") or Path(path).stem)
    payload = {**body, "name": name}
    return ProductionConfig.model_validate(payload)


def is_production_yaml(path: Path | str) -> bool:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return isinstance(data, dict) and isinstance(data.get("production"), dict)


def validate_batch_name(source_name: str) -> str:
    return validate_source_name(source_name)
