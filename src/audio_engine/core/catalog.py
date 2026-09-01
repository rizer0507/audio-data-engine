from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from audio_engine.core.artifacts import atomic_write_json
from audio_engine.core.manifest import file_sha256

CATALOG_SCHEMA_VERSION = "1.0"
ARTIFACT_KINDS = {
    "manifest",
    "dataset_release",
    "model",
    "report",
    "training_input",
    "other",
}
ArtifactKind = Literal["manifest", "dataset_release", "model", "report", "training_input", "other"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def current_git_commit(cwd: Path | None = None) -> str | None:
    """Return the source revision without making catalog writes depend on Git."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


class ProducerRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pipeline: str | None = None
    run_id: str | None = None
    run_dir: str | None = None
    config_digest: str | None = None
    git_commit: str | None = None


class ArtifactRecord(BaseModel):
    """Immutable description of one materialized pipeline artifact."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = CATALOG_SCHEMA_VERSION
    artifact_id: str
    kind: ArtifactKind
    uri: str
    sha256: str
    size_bytes: int = Field(ge=0)
    created_at: str
    producer: ProducerRecord = Field(default_factory=ProducerRecord)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("artifact_id")
    @classmethod
    def validate_artifact_id(cls, value: str) -> str:
        if not value or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for char in value):
            raise ValueError("artifact_id may contain only lowercase letters, digits, '_' and '-'")
        return value

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
        return value


class DatasetRelease(BaseModel):
    """Frozen train/dev/test selection and the policy that produced it."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = CATALOG_SCHEMA_VERSION
    release_id: str
    source_artifact_id: str
    outputs: dict[Literal["train", "dev", "test", "holdout"], str]
    policy_version: str
    normalization_version: str
    gold_revision: str
    split_seed: int
    group_key: str
    counts: dict[str, int] = Field(default_factory=dict)
    parent_release_id: str | None = None
    created_at: str = Field(default_factory=utc_now)
    git_commit: str | None = None

    @model_validator(mode="after")
    def validate_outputs(self) -> DatasetRelease:
        required = {"train", "dev", "test"}
        missing = required - self.outputs.keys()
        if missing:
            raise ValueError(f"dataset release outputs missing: {sorted(missing)}")
        if len(set(self.outputs.values())) != len(self.outputs):
            raise ValueError("dataset release outputs must reference distinct artifacts")
        unknown_counts = set(self.counts) - set(self.outputs)
        if unknown_counts:
            raise ValueError(f"counts contain unknown splits: {sorted(unknown_counts)}")
        if any(value < 0 for value in self.counts.values()):
            raise ValueError("dataset release counts must be non-negative")
        return self


class ModelVersion(BaseModel):
    """Minimal contract between the data engine and an external trainer."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = CATALOG_SCHEMA_VERSION
    model_id: str
    base_model: str
    training_release_id: str
    training_recipe: str
    checkpoint_uri: str
    status: Literal["pending", "training", "ready", "failed"]
    created_at: str = Field(default_factory=utc_now)
    git_commit: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactCatalog:
    """Filesystem catalog using one atomic, immutable JSON record per artifact.

    Directory scanning is intentional for the first local implementation: it avoids
    a mutable shared index, so parallel shard processes cannot corrupt catalog state.
    """

    def __init__(self, root: str | Path = "data/catalog"):
        self.root = Path(root)
        self.records_dir = self.root / "artifacts"
        self.releases_dir = self.root / "releases"
        self.models_dir = self.root / "models"

    def register_file(
        self,
        path: str | Path,
        *,
        kind: ArtifactKind,
        producer: ProducerRecord | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        if kind not in ARTIFACT_KINDS:
            raise ValueError(
                f"Unknown artifact kind: {kind}; expected one of {sorted(ARTIFACT_KINDS)}"
            )
        artifact_path = Path(path)
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Artifact file not found: {artifact_path}")
        resolved = artifact_path.resolve()
        digest = file_sha256(resolved)
        identity = {
            "kind": kind,
            "uri": str(resolved),
            "sha256": digest,
            "producer": (producer or ProducerRecord()).model_dump(),
            "metadata": metadata or {},
        }
        record_digest = hashlib.sha256(
            json.dumps(identity, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        artifact_id = f"{kind}_{digest[:12]}_{record_digest[:8]}"
        record = ArtifactRecord(
            artifact_id=artifact_id,
            kind=kind,
            uri=str(resolved),
            sha256=digest,
            size_bytes=resolved.stat().st_size,
            created_at=utc_now(),
            producer=producer or ProducerRecord(),
            metadata=metadata or {},
        )
        return self.put(record)

    def put(self, record: ArtifactRecord) -> ArtifactRecord:
        destination = self.records_dir / f"{record.artifact_id}.json"
        if destination.exists():
            existing = ArtifactRecord.model_validate_json(destination.read_text(encoding="utf-8"))
            # created_at is observational and excluded from idempotency comparison.
            left = existing.model_dump(exclude={"created_at"})
            right = record.model_dump(exclude={"created_at"})
            if left != right:
                raise ValueError(f"artifact id collision: {record.artifact_id}")
            return existing
        atomic_write_json(destination, record.model_dump(mode="json"))
        return record

    def get(self, artifact_id: str, *, verify: bool = False) -> ArtifactRecord:
        path = self.records_dir / f"{artifact_id}.json"
        if not path.is_file():
            raise KeyError(f"Unknown artifact: {artifact_id}")
        record = ArtifactRecord.model_validate_json(path.read_text(encoding="utf-8"))
        if verify:
            artifact_path = Path(record.uri)
            if not artifact_path.is_file():
                raise FileNotFoundError(f"Artifact payload missing: {artifact_path}")
            actual = file_sha256(artifact_path)
            if actual != record.sha256:
                raise ValueError(
                    f"Artifact payload changed: {record.artifact_id} "
                    f"expected={record.sha256}, actual={actual}"
                )
        return record

    def list(self, *, kind: ArtifactKind | None = None) -> list[ArtifactRecord]:
        if kind is not None and kind not in ARTIFACT_KINDS:
            raise ValueError(
                f"Unknown artifact kind: {kind}; expected one of {sorted(ARTIFACT_KINDS)}"
            )
        if not self.records_dir.exists():
            return []
        records = [
            ArtifactRecord.model_validate_json(path.read_text(encoding="utf-8"))
            for path in self.records_dir.glob("*.json")
        ]
        if kind is not None:
            records = [record for record in records if record.kind == kind]
        return sorted(
            records, key=lambda record: (record.created_at, record.artifact_id), reverse=True
        )

    def put_release(self, release: DatasetRelease) -> DatasetRelease:
        for artifact_id in [release.source_artifact_id, *release.outputs.values()]:
            record = self.get(artifact_id, verify=True)
            if record.kind != "manifest":
                raise ValueError(f"dataset release requires manifest artifact: {artifact_id}")
        self._put_named(self.releases_dir, release.release_id, release.model_dump(mode="json"))
        return release

    def get_release(self, release_id: str) -> DatasetRelease:
        return DatasetRelease.model_validate(self._get_named(self.releases_dir, release_id))

    def list_releases(self) -> list[DatasetRelease]:
        return (
            [
                DatasetRelease.model_validate_json(path.read_text(encoding="utf-8"))
                for path in sorted(self.releases_dir.glob("*.json"), reverse=True)
            ]
            if self.releases_dir.exists()
            else []
        )

    def put_model(self, model: ModelVersion) -> ModelVersion:
        self.get_release(model.training_release_id)
        self._put_named(self.models_dir, model.model_id, model.model_dump(mode="json"))
        return model

    def get_model(self, model_id: str) -> ModelVersion:
        return ModelVersion.model_validate(self._get_named(self.models_dir, model_id))

    def list_models(self) -> list[ModelVersion]:
        return (
            [
                ModelVersion.model_validate_json(path.read_text(encoding="utf-8"))
                for path in sorted(self.models_dir.glob("*.json"), reverse=True)
            ]
            if self.models_dir.exists()
            else []
        )

    @staticmethod
    def _validate_name(value: str) -> None:
        if not value or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for char in value):
            raise ValueError("registry id may contain only lowercase letters, digits, '_' and '-'")

    def _put_named(self, directory: Path, identifier: str, payload: dict[str, Any]) -> None:
        self._validate_name(identifier)
        destination = directory / f"{identifier}.json"
        if destination.exists():
            existing = json.loads(destination.read_text(encoding="utf-8"))
            if existing != payload:
                raise ValueError(f"immutable registry record already exists: {identifier}")
            return
        atomic_write_json(destination, payload)

    @staticmethod
    def _get_named(directory: Path, identifier: str) -> dict[str, Any]:
        path = directory / f"{identifier}.json"
        if not path.is_file():
            raise KeyError(f"Unknown registry record: {identifier}")
        return json.loads(path.read_text(encoding="utf-8"))


def register_manifest_output(
    path: str | Path,
    *,
    catalog_dir: str | Path,
    pipeline: str,
    run_dir: str | Path,
    config_digest: str | None = None,
    sample_count: int | None = None,
) -> ArtifactRecord:
    run_path = Path(run_dir).resolve()
    producer = ProducerRecord(
        pipeline=pipeline,
        run_id=run_path.name,
        run_dir=str(run_path),
        config_digest=config_digest,
        git_commit=current_git_commit(),
    )
    record = ArtifactCatalog(catalog_dir).register_file(
        path,
        kind="manifest",
        producer=producer,
        metadata={"sample_count": sample_count} if sample_count is not None else {},
    )
    atomic_write_json(run_path / "artifact.json", record.model_dump(mode="json"))
    return record
