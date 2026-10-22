"""HTTP request/response models (thin wrappers; plugin contracts remain authoritative)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ErrorBody(APIModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    request_id: str | None = None


class ErrorResponse(APIModel):
    error: ErrorBody


class SourceCreate(APIModel):
    """Create payload: business fields only; source_id is assigned by the server."""

    batch: str
    path: str
    format: str = "wav"
    name: str | None = None
    sample_rate: int | None = None
    bit_depth: int | None = None
    channels: int | None = None
    note: str = ""
    origin: str = ""


class SourcePatch(APIModel):
    batch: str | None = None
    name: str | None = None
    path: str | None = None
    format: str | None = None
    sample_rate: int | None = None
    bit_depth: int | None = None
    channels: int | None = None
    note: str | None = None
    origin: str | None = None
    # Legacy alias accepted on patch for old clients
    source_name: str | None = None


class ModelCreate(APIModel):
    """Create payload: connection fields only; connection_id is assigned by the server."""

    provider: str
    base_url: str
    model: str
    name: str | None = None
    model_revision: str = "deployment_001"
    api_key_env: str = ""
    timeout_seconds: int = 120
    concurrency: int = 4


class ModelPatch(APIModel):
    name: str | None = None
    provider: str | None = None
    base_url: str | None = None
    model: str | None = None
    model_revision: str | None = None
    api_key_env: str | None = None
    timeout_seconds: int | None = None
    concurrency: int | None = None


class ModelCheckRequest(APIModel):
    level: str = "models"  # models | transcription
    sample_ref: dict[str, str] | None = None


class PluginStatusPatch(APIModel):
    status: Literal["enabled", "archived"]
    expected_revision: int | None = None


class PluginDefinitionDraft(APIModel):
    """Editable definition draft; adapters ignore unknown keys via schema validation."""

    model_config = ConfigDict(extra="allow")

    expected_revision: int | None = None
    expected_definition_digest: str | None = None
    draft: dict[str, Any] = Field(default_factory=dict)


class PluginValidateRequest(APIModel):
    model_config = ConfigDict(extra="allow")


class JobSubmitRequest(APIModel):
    plan_id: str
    plan_digest: str


class JobRetryRequest(APIModel):
    mode: str  # resume | rerun


class TagActionRequest(APIModel):
    action: str
    tag_id: str
    reason: str = ""
    operator_id: str | None = None
    expected_revision: str | None = None
    idempotency_id: str | None = None
    work_root: str | None = None
    batch_id: str | None = None
    tagging_config: str | None = None
    audio_sha256: str | None = None


class ReviewRequest(APIModel):
    model_config = ConfigDict(extra="allow")
    operator_id: str | None = None
    reason: str = ""
    expected_revision: str | None = None
    idempotency_id: str | None = None
    review_pass: str = "first"
    queue_id: str | None = None
    decision: str = "accept"
    gold_kind: str | None = None
    gold_text: str | None = None
    speech_scope: str | None = None
    human_semantic: str | None = None
    human_noise: str | None = None
    human_crosstalk: str | None = None
    verified_error_tags: list[str] = Field(default_factory=list)
    audio_event_tags: list[str] = Field(default_factory=list)
    annotation_config: str | None = None


class ExportRequest(APIModel):
    model_config = ConfigDict(extra="allow")
    export_plugin_id: str = "warehouse_export_annotation"
    plugin_version: str = "1"
    filters: dict[str, Any] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str | None = None


class GoldGateRequest(APIModel):
    require_formal_gold: bool = True
    min_gold_ratio: float = 0.0
