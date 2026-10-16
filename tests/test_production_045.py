"""Minimal acceptance tests for 045 production chain."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from audio_engine.core.manifest import Manifest
from audio_engine.core.operator import OperatorConfig
from audio_engine.core.production.config import ProductionConfig, load_production_config
from audio_engine.core.production.orchestrator import ProductionOrchestrator
from audio_engine.core.production.paths import ProductionPaths
from audio_engine.core.sample import Sample
from audio_engine.core.source_naming import (
    apply_source_name_to_single_pipeline,
    manifest_dir_for_stem,
    manifest_path,
)
from audio_engine.operators.quality.evidence_join import EvidenceJoinOperator


def test_production_paths_and_stems() -> None:
    paths = ProductionPaths("demo_batch")
    assert paths.ingested.as_posix().endswith("ingested/ingested_demo_batch.parquet")
    assert paths.segments.as_posix().endswith("segments/segments_demo_batch.parquet")
    assert paths.dnsmos.as_posix().endswith("quality/dnsmos_demo_batch.parquet")
    assert "quality_sidecar" not in paths.dnsmos.as_posix()
    assert paths.prepared_evidence.as_posix().endswith(
        "derived/prepared_evidence_demo_batch.parquet"
    )
    assert manifest_dir_for_stem("dnsmos_demo_batch").as_posix().endswith("stage1/quality")
    assert manifest_dir_for_stem("segments_demo_batch").as_posix().endswith("stage1/segments")
    assert manifest_dir_for_stem("ingested_demo_batch").as_posix().endswith("stage1/ingested")


def test_load_production_yaml_routes_not_hardcoded() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_production_config(root / "pipelines/audio_ingest_vad_asr_dnsmos_tags.yaml")
    assert cfg.name == "audio_ingest_vad_asr_dnsmos_tags"
    assert len(cfg.asr_routes) >= 1
    ids = [r.route_id for r in cfg.asr_routes]
    assert len(ids) == len(set(ids))
    assert cfg.dnsmos.pipeline.endswith("dnsmos_score_batch.yaml")
    assert "quality_sidecar" not in cfg.dnsmos.pipeline


def test_source_naming_dnsmos_and_free_tags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    batch = "b045"
    (tmp_path / "datasets/stage1/segments").mkdir(parents=True)
    (tmp_path / "datasets/stage1/derived").mkdir(parents=True)
    seg = tmp_path / f"datasets/stage1/segments/segments_{batch}.parquet"
    Manifest(samples=[]).save(seg)
    evidence = tmp_path / f"datasets/stage1/derived/prepared_evidence_{batch}.parquet"
    Manifest(samples=[]).save(evidence)

    dnsmos = apply_source_name_to_single_pipeline(
        pipeline_name="dnsmos_score_batch",
        steps=[],
        source_name=batch,
    )
    assert "dnsmos_" in dnsmos["output_manifest"]
    assert "quality_sidecar" not in dnsmos["output_manifest"]

    tags = apply_source_name_to_single_pipeline(
        pipeline_name="tag_dataset_free_tags_v2",
        steps=[],
        source_name=batch,
    )
    assert "prepared_evidence_" in tags["input_manifest"]
    assert "classified_" not in tags["input_manifest"]


def test_evidence_join_identity_and_coverage(tmp_path: Path) -> None:
    segments = [
        Sample(
            id="seg-a",
            source_path="a.wav",
            sha256="hash-a",
            labels={"original_audio_sha256": "hash-a", "is_vad_segment": True},
            audio={"resampled_16k": "a.wav"},
        ),
        Sample(
            id="seg-b",
            source_path="b.wav",
            sha256="hash-b",
            labels={"original_audio_sha256": "hash-b", "is_vad_segment": True},
            audio={"resampled_16k": "b.wav"},
        ),
    ]
    asr_path = tmp_path / "qwen1.parquet"
    Manifest(
        samples=[
            Sample(
                id="seg-a",
                source_path="a.wav",
                sha256="hash-a",
                labels={"original_audio_sha256": "hash-a"},
                transcripts={"qwen1": {"text": "你好", "status": "success_text"}},
            ),
            Sample(
                id="seg-b",
                source_path="b.wav",
                sha256="hash-b",
                labels={"original_audio_sha256": "hash-b"},
                transcripts={"qwen1": {"text": "", "status": "success_empty"}},
            ),
        ]
    ).save(asr_path)
    dnsmos_path = tmp_path / "dnsmos.parquet"
    Manifest(
        samples=[
            Sample(
                id="seg-a",
                source_path="a.wav",
                sha256="hash-a",
                labels={"original_audio_sha256": "hash-a"},
                quality={"dnsmos_status": "success", "dnsmos_ovrl": 3.1},
            ),
            Sample(
                id="seg-b",
                source_path="b.wav",
                sha256="hash-b",
                labels={"original_audio_sha256": "hash-b"},
                quality={"dnsmos_status": "failed", "dnsmos_error": "boom"},
            ),
        ]
    ).save(dnsmos_path)

    op = EvidenceJoinOperator()
    cfg = OperatorConfig(
        params={
            "manifests": [{"model": "qwen1", "path": str(asr_path)}],
            "dnsmos_manifest": str(dnsmos_path),
            "id_policy": "exact",
        },
        run_dir=tmp_path / "run",
        step_name="evidence_join",
    )
    (tmp_path / "run").mkdir()
    out = op.run(segments, cfg)
    assert len(out) == 2
    by_id = {s.id: s for s in out}
    assert by_id["seg-a"].transcripts["qwen1"]["text"] == "你好"
    assert by_id["seg-a"].quality["dnsmos_ovrl"] == 3.1
    assert by_id["seg-b"].quality["dnsmos_status"] == "failed"
    # FSMN must not invent Silero facts.
    assert "vad_speech_present" not in by_id["seg-a"].quality
    assert by_id["seg-a"].labels.get("evidence_dnsmos") is True

    # Hash mismatch rejected.
    bad = tmp_path / "bad_dnsmos.parquet"
    Manifest(
        samples=[
            Sample(
                id="seg-a",
                source_path="a.wav",
                sha256="OTHER",
                labels={"original_audio_sha256": "OTHER"},
                quality={"dnsmos_status": "success"},
            ),
            Sample(
                id="seg-b",
                source_path="b.wav",
                sha256="hash-b",
                labels={"original_audio_sha256": "hash-b"},
                quality={"dnsmos_status": "success"},
            ),
        ]
    ).save(bad)
    with pytest.raises(ValueError, match="hash mismatch"):
        op.run(
            segments,
            OperatorConfig(
                params={
                    "manifests": [{"model": "qwen1", "path": str(asr_path)}],
                    "dnsmos_manifest": str(bad),
                },
                run_dir=tmp_path / "run2",
                step_name="evidence_join",
            ),
        )

    # Legacy sidecar naming refused.
    sidecar = tmp_path / "quality_sidecar_x.parquet"
    Manifest(samples=[]).save(sidecar)
    with pytest.raises(ValueError, match="quality_sidecar"):
        op.run(
            segments,
            OperatorConfig(
                params={
                    "manifests": [{"model": "qwen1", "path": str(asr_path)}],
                    "dnsmos_manifest": str(sidecar),
                },
                run_dir=tmp_path / "run3",
                step_name="evidence_join",
            ),
        )


def test_production_parallel_wave_overlap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Prove ready ASR + DNSMOS nodes overlap via barrier + ThreadPoolExecutor."""
    monkeypatch.chdir(tmp_path)
    barrier = threading.Barrier(2, timeout=5)
    started: dict[str, float] = {}
    finished: dict[str, float] = {}
    lock = threading.Lock()

    def fake_run_node(self, run_root, node, state):  # noqa: ANN001
        with lock:
            started[node.id] = time.monotonic()
        state.nodes[node.id].started_at = "t0"
        if node.id.startswith("asr_") or node.id == "dnsmos":
            barrier.wait()
            time.sleep(0.05)
        # Write declared outputs.
        for out in node.outputs:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_text("ok", encoding="utf-8")
        with lock:
            finished[node.id] = time.monotonic()
        state.nodes[node.id].status = "succeeded"
        state.nodes[node.id].finished_at = "t1"
        state.nodes[node.id].return_code = 0

    cfg = ProductionConfig.model_validate(
        {
            "name": "prod_test",
            "ingest": {"pipeline": "pipelines/audio_ingest.yaml"},
            "vad": {"pipeline": "pipelines/vad_segment_batch.yaml"},
            "asr_routes": [
                {
                    "route_id": "qwen1",
                    "family": "qwen",
                    "pipeline": "pipelines/qwen_asr_batch.yaml",
                }
            ],
            "dnsmos": {"pipeline": "pipelines/dnsmos_score_batch.yaml"},
            "evidence_join": {"pipeline": "pipelines/evidence_join.yaml"},
            "tagging": {"pipeline": "pipelines/tag_dataset_free_tags_v2.yaml"},
        }
    )
    # Create dummy pipeline files so resolve succeeds.
    for rel in (
        "pipelines/audio_ingest.yaml",
        "pipelines/vad_segment_batch.yaml",
        "pipelines/qwen_asr_batch.yaml",
        "pipelines/dnsmos_score_batch.yaml",
        "pipelines/evidence_join.yaml",
        "pipelines/tag_dataset_free_tags_v2.yaml",
    ):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("name: stub\n", encoding="utf-8")

    orch = ProductionOrchestrator(
        cfg,
        config_path=tmp_path / "pipelines/audio_ingest_vad_asr_dnsmos_tags.yaml",
        source_name="t045",
        source_dir=tmp_path / "raw",
        runs_dir=tmp_path / "runs",
    )
    (tmp_path / "raw").mkdir()
    (tmp_path / "pipelines/audio_ingest_vad_asr_dnsmos_tags.yaml").write_text(
        "name: prod_test\nproduction: {}\n", encoding="utf-8"
    )

    # Pre-create successful ingest/vad outputs path parents via patched node runner
    # that also creates early-stage outputs.
    original_build = orch._build_nodes

    def build_and_touch():
        nodes = original_build()
        return nodes

    orch._build_nodes = build_and_touch  # type: ignore[method-assign]
    monkeypatch.setattr(ProductionOrchestrator, "_run_node", fake_run_node)

    result = orch.run()
    assert result.status == "succeeded"
    assert "asr_qwen1" in started and "dnsmos" in started
    # Overlap: each started before the other finished.
    assert started["asr_qwen1"] < finished["dnsmos"]
    assert started["dnsmos"] < finished["asr_qwen1"]
    assert result.overlap_evidence is not None
    assert result.overlap_evidence.get("parallel") is True
    wave = set(result.overlap_evidence.get("wave_node_ids") or [])
    assert "asr_qwen1" in wave and "dnsmos" in wave
    assert (result.run_root / "overlap_evidence.json").is_file()
    assert (result.run_root / "progress_view.json").is_file()


def test_master_yaml_has_no_quality_sidecar_references() -> None:
    root = Path(__file__).resolve().parents[1]
    data = yaml.safe_load(
        (root / "pipelines/audio_ingest_vad_asr_dnsmos_tags.yaml").read_text(
            encoding="utf-8"
        )
    )
    body = json.dumps(data.get("production") or {}, ensure_ascii=False)
    assert "quality_sidecar" not in body
    assert "dnsmos_score_batch" in body
    assert "production" in data
    assert data.get("stages") is None
