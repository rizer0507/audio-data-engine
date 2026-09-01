from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path
from pydantic import BaseModel, ConfigDict

from audio_engine.core.artifacts import atomic_write_json
from audio_engine.core.catalog import ArtifactCatalog, ModelVersion, current_git_commit, utc_now


class TrainingJob(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    release_id: str
    recipe: str
    command: list[str]
    checkpoint_uri: str
    model_id: str
    base_model: str
    status: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    return_code: int | None = None
    error: str | None = None


def run_training_job(
    *,
    catalog: ArtifactCatalog,
    jobs_dir: Path,
    release_id: str,
    recipe: Path,
    command: str | list[str],
    checkpoint: Path,
    model_id: str,
    base_model: str,
    cwd: Path | None = None,
) -> tuple[TrainingJob, ModelVersion]:
    """Run an external trainer synchronously through a narrow, auditable contract."""
    release = catalog.get_release(release_id)
    if not recipe.is_file():
        raise FileNotFoundError(f"training recipe not found: {recipe}")
    argv = shlex.split(command) if isinstance(command, str) else list(command)
    if not argv:
        raise ValueError("training command must not be empty")
    identity = {
        "release_id": release_id,
        "recipe_sha256": hashlib.sha256(recipe.read_bytes()).hexdigest(),
        "command": argv,
        "checkpoint": str(checkpoint.resolve()),
        "model_id": model_id,
        "base_model": base_model,
    }
    digest = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    job_id = f"train_{digest[:16]}"
    job_dir = jobs_dir / job_id
    state_path = job_dir / "state.json"
    if state_path.exists():
        existing = TrainingJob.model_validate_json(state_path.read_text(encoding="utf-8"))
        if existing.status == "succeeded":
            return existing, catalog.get_model(existing.model_id)
        if existing.status == "running":
            raise RuntimeError(f"training job is already running: {job_id}")

    job = TrainingJob(
        job_id=job_id,
        release_id=release_id,
        recipe=str(recipe.resolve()),
        command=argv,
        checkpoint_uri=str(checkpoint.resolve()),
        model_id=model_id,
        base_model=base_model,
        status="running",
        created_at=utc_now(),
        started_at=utc_now(),
    )
    atomic_write_json(state_path, job.model_dump(mode="json"))
    atomic_write_json(
        job_dir / "input.json", {"release": release.model_dump(mode="json"), **identity}
    )
    env = {
        **os.environ,
        "AUDIO_DATA_RELEASE_ID": release_id,
        "AUDIO_DATA_TRAIN_MANIFEST": catalog.get(release.outputs["train"], verify=True).uri,
        "AUDIO_DATA_DEV_MANIFEST": catalog.get(release.outputs["dev"], verify=True).uri,
        "AUDIO_DATA_RECIPE": str(recipe.resolve()),
        "AUDIO_DATA_CHECKPOINT": str(checkpoint.resolve()),
    }
    with (
        (job_dir / "stdout.log").open("w", encoding="utf-8") as stdout,
        (job_dir / "stderr.log").open("w", encoding="utf-8") as stderr,
    ):
        result = subprocess.run(argv, cwd=cwd, env=env, stdout=stdout, stderr=stderr, check=False)
    job.return_code = result.returncode
    job.finished_at = utc_now()
    if result.returncode != 0:
        job.status = "failed"
        job.error = f"trainer exited with code {result.returncode}"
        atomic_write_json(state_path, job.model_dump(mode="json"))
        raise RuntimeError(job.error)
    if not checkpoint.exists():
        job.status = "failed"
        job.error = f"trainer succeeded but checkpoint is missing: {checkpoint}"
        atomic_write_json(state_path, job.model_dump(mode="json"))
        raise RuntimeError(job.error)
    model = catalog.put_model(
        ModelVersion(
            model_id=model_id,
            base_model=base_model,
            training_release_id=release_id,
            training_recipe=str(recipe.resolve()),
            checkpoint_uri=str(checkpoint.resolve()),
            status="ready",
            git_commit=current_git_commit(),
            metadata={"training_job_id": job_id},
        )
    )
    job.status = "succeeded"
    atomic_write_json(state_path, job.model_dump(mode="json"))
    return job, model
