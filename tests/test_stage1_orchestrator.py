"""Tests for stage-1 job orchestration, config gen, and reconcile gates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from audio_engine.core.manifest import Manifest
from audio_engine.core.sample import Sample
from audio_engine.core.stage1.cache_policy import CACHE_BOUNDARY, all_run_aliases
from audio_engine.core.stage1.config_gen import freeze_job_configs, write_dataset_with_runs
from audio_engine.core.stage1.job import (
    SELECTION_RULE,
    Stage1JobRequest,
    create_job,
    default_stage_names,
    parse_gpus_option,
    parse_model_option,
)
from audio_engine.core.stage1.orchestrator import Stage1Orchestrator, plan_job_summary
from audio_engine.core.stage1.reconcile import discover_xlsx_parts, reconcile_delivery
from audio_engine.core.stage1.runtime_config import load_runtime_config


def _filled_runtime(tmp_path: Path) -> Path:
    engine = tmp_path / "engine.py"
    engine.write_text("#!/bin/sh\n", encoding="utf-8")
    vllm = tmp_path / "vllm"
    vllm.write_text("#!/bin/sh\n", encoding="utf-8")
    glm_env = tmp_path / "glm_env" / "bin"
    glm_env.mkdir(parents=True)
    (glm_env / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    (glm_env / "vllm").write_text("#!/bin/sh\n", encoding="utf-8")
    for name in ("qwen", "glm", "sensevoice"):
        (tmp_path / "models" / name).mkdir(parents=True)
        (tmp_path / "models" / name / "config.json").write_text("{}", encoding="utf-8")
    template = tmp_path / "chat.jinja"
    template.write_text("{{x}}", encoding="utf-8")
    dnsmos = tmp_path / "dnsmos.onnx"
    dnsmos.write_bytes(b"onnx")
    data = {
        "authorized_gpus": [4, 5],
        "engine_python": str(engine),
        "dnsmos": {"onnx_path": str(dnsmos)},
        "probe": {"ready_timeout_s": 1, "ready_poll_s": 0.1},
        "session": {"root": str(tmp_path / "serve")},
        "families": {
            "qwen": {
                "model_path": str(tmp_path / "models" / "qwen"),
                "chat_template": str(template),
                "vllm_bin": str(vllm),
                "served_model_name": "qwen3-asr",
                "host": "127.0.0.1",
                "port": 5555,
                "gpu_memory_utilization": 0.5,
                "tensor_parallel_size": 1,
                "api_key": "dummy",
                "pipeline": "pipelines/qwen_asr_batch.yaml",
            },
            "glm": {
                "model_path": str(tmp_path / "models" / "glm"),
                "env_root": str(tmp_path / "glm_env"),
                "served_model_name": "glm-asr",
                "host": "0.0.0.0",
                "client_host": "127.0.0.1",
                "port": 5570,
                "tensor_parallel_size": 1,
                "dtype": "bfloat16",
                "max_model_len": 4096,
                "max_num_seqs": 8,
                "gpu_memory_utilization": 0.9,
                "trust_remote_code": True,
                "limit_mm_per_prompt": '{"audio":1}',
                "no_enable_flashinfer_autotune": True,
                "kernel_config": '{"enable_jit_warmup":false,"enable_cutedsl_warmup":false}',
                "env": {
                    "PYTHONNOUSERSITE": "1",
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "FLASHINFER_DISABLE_VERSION_CHECK": "1",
                    "VLLM_USE_FLASHINFER_SAMPLER": "0",
                },
                "unset_env": [
                    "VLLM_ATTENTION_BACKEND",
                    "VLLM_USE_FLASHINFER_SAMPLE",
                    "VLLM_ATTENTIOIN_BACKEND",
                ],
                "api_key": "dummy",
                "pipeline": "pipelines/glm_asr_batch.yaml",
            },
            "sensevoice": {
                "model_path": str(tmp_path / "models" / "sensevoice"),
                "device": "cuda:0",
                "language": "auto",
                "use_itn": True,
                "disable_update": True,
                "pipeline": "pipelines/sensevoice_asr_batch.yaml",
            },
        },
    }
    path = tmp_path / "server.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def _request(tmp_path: Path, runtime: Path, source_dir: Path) -> Stage1JobRequest:
    return Stage1JobRequest(
        batch="demo-batch",
        source=str(source_dir),
        source_kind="directory",
        models={
            "qwen": str(tmp_path / "models" / "qwen"),
            "glm": str(tmp_path / "models" / "glm"),
            "sensevoice": str(tmp_path / "models" / "sensevoice"),
        },
        runtime_config=str(runtime),
        gpus=["4", "5"],
        jobs_root=str(tmp_path / "jobs"),
        catalog_dir=str(tmp_path / "catalog"),
    )


def test_parse_model_and_gpus():
    assert parse_model_option("qwen=/data/a") == ("qwen", "/data/a")
    with pytest.raises(ValueError):
        parse_model_option("kimi=/x")
    assert parse_gpus_option("4,5") == ["4", "5"]


def test_plan_uses_v2_2_rule_and_six_aliases(tmp_path: Path):
    runtime = _filled_runtime(tmp_path)
    source = tmp_path / "wavs"
    source.mkdir()
    req = _request(tmp_path, runtime, source)
    summary = plan_job_summary(req)
    assert summary["selection_rule"] == SELECTION_RULE
    assert "classify_dataset_five_class_v2_2_auto_noise" in summary["classify_pipeline"]
    assert len(all_run_aliases()) == 6
    assert "asr_qwen_1" in default_stage_names()
    assert "asr_qwen_2" in default_stage_names()
    assert CACHE_BOUNDARY.forbidden


def test_duplicate_submit_rejected(tmp_path: Path):
    runtime = _filled_runtime(tmp_path)
    source = tmp_path / "wavs"
    source.mkdir()
    req = _request(tmp_path, runtime, source)
    create_job(req)
    with pytest.raises(ValueError, match="已存在|已成功|拒绝"):
        create_job(req)


def test_freeze_generates_unique_execution_ids(tmp_path: Path):
    runtime_path = _filled_runtime(tmp_path)
    runtime = load_runtime_config(runtime_path)
    source = tmp_path / "wavs"
    source.mkdir()
    req = _request(tmp_path, runtime_path, source)
    root, _ = create_job(req)
    paths = freeze_job_configs(root, req, runtime)
    assert paths["selection"].is_file()
    ids = []
    for alias in all_run_aliases():
        identity = yaml.safe_load(paths[f"identity_{alias}"].read_text(encoding="utf-8"))
        ids.append(identity["execution_id"])
        assert identity["transcript_key"] == alias
        assert identity["model_checkpoint_digest"]
        assert identity["decode_config_digest"]
        assert identity["prompt_digest"]
    assert len(set(ids)) == 6


def test_write_dataset_rejects_duplicate_artifact(tmp_path: Path):
    runtime_path = _filled_runtime(tmp_path)
    runtime = load_runtime_config(runtime_path)
    source = tmp_path / "wavs"
    source.mkdir()
    req = _request(tmp_path, runtime_path, source)
    root, _ = create_job(req)
    freeze_job_configs(root, req, runtime)
    registered = []
    for alias in all_run_aliases():
        raw = yaml.safe_load(
            (root / "run_identities" / f"{alias}_identity.yaml").read_text(encoding="utf-8")
        )
        raw["artifact_id"] = "same_artifact"
        path = root / "run_identities" / f"{alias}_registered.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        registered.append(path)
    with pytest.raises(ValueError, match="artifact_id"):
        write_dataset_with_runs(root, registered)


def _sample(sid: str, sha: str, **labels) -> Sample:
    return Sample(
        id=sid,
        source_path=f"/a/{sid}.wav",
        sha256=sha,
        labels={"original_audio_sha256": sha, **labels},
        transcripts={},
    )


def test_reconcile_blocks_missing_route_and_hash_error(tmp_path: Path):
    batch = "demo"
    cleaned = tmp_path / "cleaned.parquet"
    Manifest([_sample("s1", "h1"), _sample("s2", "h2")]).save(cleaned)

    asr_paths = {}
    registered = {}
    for alias in all_run_aliases():
        path = tmp_path / f"{alias}.parquet"
        # Wrong hash on purpose for qwen_1
        sha = "bad" if alias == "qwen_1" else ("h1" if True else "h2")
        Manifest(
            [
                _sample("s1", "bad" if alias == "qwen_1" else "h1"),
                _sample("s2", "h2"),
            ]
        ).save(path)
        asr_paths[alias] = path
        identity = {
            "run_id": alias,
            "family": alias.split("_")[0] if not alias.startswith("sense") else "sensevoice",
            "transcript_key": alias,
            "execution_id": f"exec_{alias}",
            "artifact_id": f"art_{alias}",
            "model_checkpoint_digest": "a",
            "decode_config_digest": "b",
            "prompt_digest": "c",
            "created_at": "2026-01-01T00:00:00+00:00",
        }
        if alias.startswith("sensevoice"):
            identity["family"] = "sensevoice"
        elif alias.startswith("glm"):
            identity["family"] = "glm"
        else:
            identity["family"] = "qwen"
        id_path = tmp_path / f"{alias}_reg.yaml"
        id_path.write_text(yaml.safe_dump(identity), encoding="utf-8")
        registered[alias] = id_path

    classified = tmp_path / "classified.parquet"
    Manifest(
        [
            _sample(
                "s1",
                "h1",
                classification_bucket="consensus_gold",
                category="clear_semantic",
                outcome="classified",
                rule_version=SELECTION_RULE,
            ),
            _sample(
                "s2",
                "h2",
                classification_bucket="excluded_empty",
                outcome="excluded",
                classification_reason_codes=["empty"],
                rule_version=SELECTION_RULE,
            ),
        ]
    ).save(classified)

    export = tmp_path / "out.xlsx"
    report = reconcile_delivery(
        batch=batch,
        cleaned=cleaned,
        asr_paths=asr_paths,
        registered_identities=registered,
        classified=classified,
        export_xlsx=export,
    )
    assert report.ok is False
    assert any("对齐失败" in e or "hash" in e.lower() or "xlsx" in e.lower() for e in report.errors)


def test_reconcile_blocks_missing_asr_route(tmp_path: Path):
    cleaned = tmp_path / "cleaned.parquet"
    Manifest([_sample("s1", "h1")]).save(cleaned)
    classified = tmp_path / "classified.parquet"
    Manifest(
        [
            _sample(
                "s1",
                "h1",
                classification_bucket="consensus_gold",
                category="clear_semantic",
                rule_version=SELECTION_RULE,
            )
        ]
    ).save(classified)
    report = reconcile_delivery(
        batch="x",
        cleaned=cleaned,
        asr_paths={},
        registered_identities={},
        classified=classified,
        export_xlsx=tmp_path / "missing.xlsx",
    )
    assert report.ok is False
    assert any("ASR 路次缺失" in e for e in report.errors)
    assert report.stats["accounting"]["asr_routes_missing"]


def test_reconcile_xlsx_parts_and_pass(tmp_path: Path):
    pytest.importorskip("pandas")
    pytest.importorskip("openpyxl")
    cleaned = tmp_path / "cleaned.parquet"
    samples = [_sample(f"s{i}", f"h{i}") for i in range(3)]
    Manifest(samples).save(cleaned)

    asr_paths = {}
    registered = {}
    for alias in all_run_aliases():
        Manifest([_sample(f"s{i}", f"h{i}") for i in range(3)]).save(tmp_path / f"{alias}.parquet")
        asr_paths[alias] = tmp_path / f"{alias}.parquet"
        family = (
            "sensevoice"
            if alias.startswith("sensevoice")
            else ("glm" if alias.startswith("glm") else "qwen")
        )
        identity = {
            "run_id": alias,
            "family": family,
            "transcript_key": alias,
            "execution_id": f"exec_{alias}",
            "artifact_id": f"art_{alias}",
            "model_checkpoint_digest": "a",
            "decode_config_digest": "b",
            "prompt_digest": "c",
            "created_at": "2026-01-01T00:00:00+00:00",
        }
        path = tmp_path / f"{alias}_reg.yaml"
        path.write_text(yaml.safe_dump(identity), encoding="utf-8")
        registered[alias] = path

    classified_samples = [
        _sample(
            f"s{i}",
            f"h{i}",
            classification_bucket="consensus_gold",
            category="clear_semantic",
            outcome="classified",
            rule_version=SELECTION_RULE,
        )
        for i in range(3)
    ]
    classified = tmp_path / "classified.parquet"
    Manifest(classified_samples).save(classified)

    import pandas as pd

    export = tmp_path / "summary.xlsx"
    # Force multi-part with max_rows=2
    rows = [
        {
            "sample_id": f"s{i}",
            "category": "clear_semantic",
            "classification_bucket": "consensus_gold",
        }
        for i in range(3)
    ]
    stem = export.with_suffix("")
    pd.DataFrame(rows[:2]).to_excel(Path(f"{stem}-part-001.xlsx"), index=False)
    pd.DataFrame(rows[2:]).to_excel(Path(f"{stem}-part-002.xlsx"), index=False)
    parts = discover_xlsx_parts(export)
    assert len(parts) == 2

    report = reconcile_delivery(
        batch="x",
        cleaned=cleaned,
        asr_paths=asr_paths,
        registered_identities=registered,
        classified=classified,
        export_xlsx=export,
        max_xlsx_rows=2,
    )
    assert report.ok, report.errors
    assert report.stats["xlsx_rows"] == 3


def test_orchestrator_dry_run_records_commands(tmp_path: Path, monkeypatch):
    runtime = _filled_runtime(tmp_path)
    source = tmp_path / "wavs"
    source.mkdir()
    req = _request(tmp_path, runtime, source)
    root, _ = create_job(req)

    # Avoid hashing/template copy issues if selection/dataset templates missing in weird cwd —
    # they exist in repo.
    orch = Stage1Orchestrator(root, dry_run=True)
    state = orch.run()
    assert state.status in {"failed", "needs_attention"}  # dry-run cannot formally succeed
    assert state.reconcile.get("ok") is False
    assert any("dry-run" in e for e in state.reconcile.get("errors", []))
    assert (root / "config_snapshot" / "meta.json").is_file()
    assert (root / "selection.yaml").is_file()
    # dual-run aliases planned
    assert orch.state.stages["asr_qwen_1"].status in {"succeeded", "skipped"}
    assert orch.state.stages["asr_qwen_2"].status in {"succeeded", "skipped"}
    cmd_files = list((root / "stages" / "asr_qwen_1").rglob("command.json"))
    cmd_files2 = list((root / "stages" / "asr_qwen_2").rglob("command.json"))
    assert cmd_files and cmd_files2
    cmd1 = json.loads(cmd_files[0].read_text(encoding="utf-8"))
    cmd2 = json.loads(cmd_files2[0].read_text(encoding="utf-8"))
    assert "--asr-run" in cmd1["argv"] and "qwen_1" in cmd1["argv"]
    assert "--asr-run" in cmd2["argv"] and "qwen_2" in cmd2["argv"]
    assert cmd1["argv"] != cmd2["argv"]
