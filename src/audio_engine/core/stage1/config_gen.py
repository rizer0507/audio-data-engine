"""Generate per-job dataset / selection / identity configs from templates."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml

from audio_engine.core.catalog import utc_now
from audio_engine.core.stage1.cache_policy import FAMILY_RUN_ALIASES, REQUIRED_FAMILIES
from audio_engine.core.stage1.digests import (
    decode_config_digest,
    path_content_digest,
    prompt_digest,
    sha256_text,
)
from audio_engine.core.stage1.job import (
    DATASET_TEMPLATE,
    SELECTION_RULE,
    SELECTION_TEMPLATE,
    Stage1JobRequest,
)
from audio_engine.core.stage1.runtime_config import Stage1RuntimeConfig


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML 必须是映射: {path}")
    return raw


def family_decode_payload(
    family: str,
    runtime: Stage1RuntimeConfig,
    *,
    model_path: str,
    api_base: str | None,
) -> dict[str, Any]:
    if family == "qwen":
        cfg = runtime.qwen
        return {
            "family": "qwen",
            "served_model_name": cfg.served_model_name,
            "api_base": api_base,
            "gpu_memory_utilization": cfg.gpu_memory_utilization,
            "tensor_parallel_size": cfg.tensor_parallel_size,
            "temperature": 0,
            "language": "zh",
            "chat_template": str(cfg.chat_template),
            "model_path": model_path,
            "pipeline": cfg.pipeline,
        }
    if family == "glm":
        cfg = runtime.glm
        return {
            "family": "glm",
            "served_model_name": cfg.served_model_name,
            "api_base": api_base,
            "gpu_memory_utilization": cfg.gpu_memory_utilization,
            "tensor_parallel_size": cfg.tensor_parallel_size,
            "dtype": cfg.dtype,
            "max_model_len": cfg.max_model_len,
            "max_num_seqs": cfg.max_num_seqs,
            "limit_mm_per_prompt": cfg.limit_mm_per_prompt,
            "kernel_config": cfg.kernel_config,
            "no_enable_flashinfer_autotune": cfg.no_enable_flashinfer_autotune,
            "model_path": model_path,
            "pipeline": cfg.pipeline,
        }
    cfg = runtime.sensevoice
    return {
        "family": "sensevoice",
        "device": cfg.device,
        "language": cfg.language,
        "use_itn": cfg.use_itn,
        "disable_update": cfg.disable_update,
        "model_path": model_path,
        "pipeline": cfg.pipeline,
    }


def family_prompt_payload(family: str, runtime: Stage1RuntimeConfig) -> dict[str, Any]:
    if family == "qwen":
        # Keep Qwen context/hotwords from shared ASR yaml when present.
        asr_cfg = Path("configs/asr/qwen_asr.yaml")
        context = ""
        if asr_cfg.is_file():
            context = str((_load_yaml(asr_cfg) or {}).get("context") or "")
        return {
            "family": "qwen",
            "chat_template": str(runtime.qwen.chat_template),
            "context": context,
        }
    if family == "glm":
        return {"family": "glm", "prompt": None, "note": "no_qwen_chat_template"}
    return {
        "family": "sensevoice",
        "language": runtime.sensevoice.language,
        "use_itn": runtime.sensevoice.use_itn,
    }


def build_run_identity(
    *,
    family: str,
    alias: str,
    batch: str,
    model_path: str,
    runtime: Stage1RuntimeConfig,
    api_base: str | None,
    execution_salt: str,
) -> dict[str, Any]:
    checkpoint = path_content_digest(Path(model_path))
    decode = decode_config_digest(
        family_decode_payload(family, runtime, model_path=model_path, api_base=api_base)
    )
    prompt = prompt_digest(family_prompt_payload(family, runtime))
    execution_id = sha256_text(
        f"{batch}|{alias}|{checkpoint}|{decode}|{prompt}|{execution_salt}"
    )[:24]
    return {
        "run_id": alias,
        "family": family,
        "transcript_key": alias,
        "execution_id": f"{alias}_{execution_id}",
        "model_checkpoint_digest": checkpoint,
        "decode_config_digest": decode,
        "prompt_digest": prompt,
        "input_audio_digest": "",
        "input_audio_key": "resampled_16k",
        "created_at": utc_now(),
    }


def write_sensevoice_pipeline_override(
    job_root: Path,
    *,
    gpus: list[str],
    template: Path = Path("pipelines/sensevoice_asr_batch.yaml"),
) -> Path:
    """Copy SenseVoice pipeline and bind sharding.gpus to authorized cards."""
    raw = _load_yaml(template)
    sharding = dict(raw.get("sharding") or {})
    sharding["gpus"] = [int(g) if str(g).isdigit() else g for g in gpus]
    # Keep shard count within capacity; do not invent extra cards.
    capacity = max(1, len(gpus) * int(sharding.get("instances_per_gpu") or 1))
    for key in ("shards", "parallel_shards"):
        if key in sharding:
            sharding[key] = min(int(sharding[key]), capacity)
    raw["sharding"] = sharding
    out = job_root / "pipelines" / "sensevoice_asr_batch.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return out


def freeze_job_configs(
    job_root: Path,
    request: Stage1JobRequest,
    runtime: Stage1RuntimeConfig,
    *,
    api_bases: dict[str, str | None] | None = None,
) -> dict[str, Path]:
    """Write snapshot configs under job_root/config_snapshot and batch identity drafts."""
    snap = job_root / "config_snapshot"
    snap.mkdir(parents=True, exist_ok=True)
    # Copy immutable rule templates into the job for audit.
    dataset_src = Path(DATASET_TEMPLATE)
    selection_src = Path(SELECTION_TEMPLATE)
    shutil.copy2(dataset_src, snap / dataset_src.name)
    shutil.copy2(selection_src, snap / selection_src.name)
    shutil.copy2(runtime.path, snap / "server.yaml")

    id_dir = job_root / "run_identities"
    id_dir.mkdir(parents=True, exist_ok=True)
    api_bases = api_bases or {}
    identities: dict[str, Path] = {}
    salt = request.config_digest()
    for family in REQUIRED_FAMILIES:
        model_path = request.models[family]
        for alias in FAMILY_RUN_ALIASES[family]:
            identity = build_run_identity(
                family=family,
                alias=alias,
                batch=request.batch,
                model_path=model_path,
                runtime=runtime,
                api_base=api_bases.get(family),
                execution_salt=f"{salt}|{alias}",
            )
            path = id_dir / f"{alias}_identity.yaml"
            path.write_text(
                yaml.safe_dump(identity, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            identities[alias] = path

    selection = _load_yaml(selection_src)
    selection["rule_version"] = SELECTION_RULE
    selection["policy_version"] = SELECTION_RULE
    selection_out = job_root / "selection.yaml"
    selection_out.write_text(
        yaml.safe_dump(selection, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    # Dataset YAML without runs yet; write_dataset_config fills runs after register.
    dataset = _load_yaml(dataset_src)
    dataset_out = job_root / "dataset.yaml"
    dataset_out.write_text(
        yaml.safe_dump(dataset, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    sv_pipeline = write_sensevoice_pipeline_override(job_root, gpus=request.gpus)
    return {
        "dataset": dataset_out,
        "selection": selection_out,
        "sensevoice_pipeline": sv_pipeline,
        **{f"identity_{alias}": path for alias, path in identities.items()},
    }


def write_dataset_with_runs(job_root: Path, registered_paths: list[Path]) -> Path:
    """Merge registered run identities into dataset.yaml runs: list."""
    dataset_path = job_root / "dataset.yaml"
    dataset = _load_yaml(dataset_path)
    runs: list[dict[str, Any]] = []
    for path in registered_paths:
        runs.append(_load_yaml(path))
    if len(runs) != 6:
        raise ValueError(f"dataset runs 须恰好 6 条，当前 {len(runs)}")
    execution_ids = [str(item.get("execution_id") or "") for item in runs]
    if len(set(execution_ids)) != 6 or any(not item for item in execution_ids):
        raise ValueError("六路 execution_id 必须存在且互异")
    artifact_ids = [str(item.get("artifact_id") or "") for item in runs]
    if len(set(artifact_ids)) != 6 or any(not item for item in artifact_ids):
        raise ValueError("六路 artifact_id 必须存在且互异；禁止复制登记结果")
    dataset["runs"] = runs
    dataset_path.write_text(
        yaml.safe_dump(dataset, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    # Also materialize under configs for operators that resolve relative paths.
    public = Path("configs/datasets") / f"zh_asr_v3_stage1_{job_root.name}.yaml"
    public.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dataset_path, public)
    return dataset_path
