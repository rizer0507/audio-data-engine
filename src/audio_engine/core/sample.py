from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LineageEntry(BaseModel):
    """Records how an artifact was produced for reproducibility."""

    operator: str
    version: str
    params: dict[str, Any] = Field(default_factory=dict)
    input_key: str | None = None
    output_key: str | None = None
    output_path: str | None = None
    cache_key: str | None = None


class TranscriptResult(BaseModel):
    text: str = ""
    model: str = ""
    version: str = ""
    confidence: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Sample(BaseModel):
    """One logical audio sample — the central data unit of the engine."""

    id: str
    source_path: str
    sha256: str = ""
    sample_rate: int | None = None
    channels: int | None = None
    duration: float | None = None

    audio: dict[str, str] = Field(default_factory=dict)
    transcripts: dict[str, dict[str, Any]] = Field(default_factory=dict)
    quality: dict[str, Any] = Field(default_factory=dict)
    labels: dict[str, Any] = Field(default_factory=dict)
    lineage: list[LineageEntry] = Field(default_factory=list)
    status: dict[str, str] = Field(default_factory=dict)
    errors: dict[str, str] = Field(default_factory=dict)

    def audio_path(self, key: str = "raw") -> str:
        if key in self.audio:
            return self.audio[key]
        if key == "raw":
            return self.source_path
        raise KeyError(f"Audio key '{key}' not found for sample {self.id}")

    def get_transcript_text(self, model: str) -> str:
        entry = self.transcripts.get(model, {})
        if isinstance(entry, dict):
            return entry.get("text", "")
        return str(entry)

    def mark_completed(self, operator_name: str) -> None:
        self.status[operator_name] = "completed"
        self.errors.pop(operator_name, None)

    def mark_failed(self, operator_name: str, error: str) -> None:
        self.status[operator_name] = "failed"
        self.errors[operator_name] = error

    def is_completed(self, operator_name: str) -> bool:
        return self.status.get(operator_name) == "completed"

    def add_lineage(
        self,
        operator: str,
        version: str,
        params: dict[str, Any],
        *,
        input_key: str | None = None,
        output_key: str | None = None,
        output_path: str | None = None,
        cache_key: str | None = None,
    ) -> None:
        self.lineage.append(
            LineageEntry(
                operator=operator,
                version=version,
                params=params,
                input_key=input_key,
                output_key=output_key,
                output_path=output_path,
                cache_key=cache_key,
            )
        )

    def to_flat_dict(self) -> dict[str, Any]:
        """Flatten sample for DataFrame / Parquet export."""
        row: dict[str, Any] = {
            "id": self.id,
            "source_path": self.source_path,
            "sha256": self.sha256,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "duration": self.duration,
            "audio": dict(self.audio),
            "transcripts": dict(self.transcripts),
            "quality": dict(self.quality),
            "labels": dict(self.labels),
            "lineage": [entry.model_dump() for entry in self.lineage],
            "status": dict(self.status),
            "errors": dict(self.errors),
        }
        for model, data in self.transcripts.items():
            if isinstance(data, dict):
                row[f"{model}_text"] = data.get("text", "")
        for key, value in self.quality.items():
            row[f"quality_{key}"] = value
        for key, value in self.labels.items():
            row[f"label_{key}"] = value
        return row

    @classmethod
    def from_flat_dict(cls, data: dict[str, Any]) -> Sample:
        return cls(
            id=data["id"],
            source_path=data["source_path"],
            sha256=data.get("sha256") or "",
            sample_rate=_optional_number(data.get("sample_rate"), as_int=True),
            channels=_optional_number(data.get("channels"), as_int=True),
            duration=_optional_number(data.get("duration")),
            audio=data.get("audio") or {},
            transcripts=data.get("transcripts") or {},
            quality=data.get("quality") or {},
            labels=data.get("labels") or {},
            lineage=[LineageEntry(**e) for e in (data.get("lineage") or [])],
            status=data.get("status") or {},
            errors=data.get("errors") or {},
        )


def _optional_number(value: Any, *, as_int: bool = False) -> int | float | None:
    """Coerce parquet/pandas missing values (NaN) to None for Pydantic optional fields."""
    if value is None:
        return None
    try:
        if value != value:  # NaN
            return None
    except (TypeError, ValueError):
        return None
    if as_int:
        return int(value)
    return float(value)
