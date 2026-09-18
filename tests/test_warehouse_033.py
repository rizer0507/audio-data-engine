"""033: classification Manifest → annotation pack → freeze batch-unique warehouse."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from audio_engine.core.annotation_v3 import AnnotationConfig
from audio_engine.core.annotation_v3.types import (
    STATE_ANNOTATED,
    STATE_PENDING,
    STATE_REJECTED,
)
from audio_engine.core.catalog import ArtifactCatalog
from audio_engine.core.manifest import Manifest
from audio_engine.core.sample import Sample
from audio_engine.core.warehouse.categories import load_allowed_categories
from audio_engine.core.warehouse.completion import (
    WarehouseGateError,
    assert_batch_complete,
    final_category,
)
from audio_engine.core.warehouse.export import (
    export_warehouse_annotation_pack,
    select_warehouse_export_samples,
)
from audio_engine.core.warehouse.freeze import publish_warehouse, warehouse_id_for_batch
from audio_engine.core.warehouse.import_support import (
    apply_reviewed_category_from_rows,
    validate_warehouse_binding,
)
from audio_engine.core.annotation_v3.import_workflow import ImportResult


def _sha(sid: str) -> str:
    return f"{sid:0>64}"[:64]


def _write_tiny_wav(path: Path) -> None:
    # Minimal RIFF/WAVE header + a few PCM bytes (enough for size>0 existence checks).
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"RIFF$\x00\x00\x00WAVEfmt "
        b"\x10\x00\x00\x00\x01\x00\x01\x00\x40\x1f\x00\x00\x80\x3e\x00\x00"
        b"\x02\x00\x10\x00data\x00\x00\x00\x00"
    )


def _classified_sample(
    sid: str,
    *,
    category: str = "environment_noise",
    wav: Path | None = None,
    **labels,
) -> Sample:
    sha = _sha(sid)
    audio = {}
    source = f"{sid}.wav"
    if wav is not None:
        source = str(wav)
        audio = {"resampled_16k": str(wav), "raw": str(wav)}
    base = {
        "category": category,
        "type": labels.pop("type", category),
        "outcome": labels.pop("outcome", "selected"),
        "review_priority": labels.pop("review_priority", "P2"),
        "review_queue": labels.pop("review_queue", ""),
        "candidate_text": labels.pop("candidate_text", "环境音"),
        "original_audio_sha256": sha,
        "rule_version": labels.pop("rule_version", "selection_five_class_v2_2_auto_noise"),
        "annotation_state": labels.pop("annotation_state", "manual_review"),
    }
    base.update(labels)
    return Sample(
        id=sid,
        source_path=source,
        sha256=sha,
        audio=audio,
        labels=base,
        transcripts={"qwen_1": {"text": base["candidate_text"]}},
    )


def _complete(sample: Sample, *, gold_text: str = "确认文本", excluded: bool = False) -> Sample:
    s = sample.model_copy(deep=True)
    if excluded:
        s.labels["annotation_state"] = STATE_REJECTED
        s.labels["gold_kind"] = "invalid"
        s.labels["annotator_decision"] = "reject"
        s.labels["warehouse_keep_reason"] = "explicit_reject"
    else:
        s.labels["annotation_state"] = STATE_ANNOTATED
        s.labels["gold_kind"] = "speech"
        s.labels["gold_text"] = gold_text
        s.labels["label_source"] = "human"
        s.labels["is_human_verified"] = True
    return s


@pytest.fixture
def ann_cfg() -> AnnotationConfig:
    return AnnotationConfig.load(Path("configs/annotation/zh_asr_v3.yaml"))


def test_full_batch_export_covers_queue_none(tmp_path: Path, ann_cfg: AnnotationConfig):
    """Default review queues skip environment_noise; warehouse export must include them."""
    wav = tmp_path / "a.wav"
    _write_tiny_wav(wav)
    samples = [
        _classified_sample("s1", category="environment_noise", wav=wav, review_queue=""),
        _classified_sample("s2", category="voicemail", wav=wav, review_queue="voicemail_isolation"),
        _classified_sample("s3", category="gold_candidate", wav=wav, review_queue="pseudo_audit"),
    ]
    selected = select_warehouse_export_samples(samples)
    assert {s.id for s in selected} == {"s1", "s2", "s3"}

    pack = tmp_path / "pack"
    result = export_warehouse_annotation_pack(
        samples,
        config=ann_cfg,
        dataset_path=tmp_path / "classified.parquet",
        output=pack,
        batch="demo_batch",
        revision="r1",
        fmt="jsonl",
    )
    assert result["sample_count"] == 3
    meta = json.loads(pack.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["warehouse_binding"]["batch"] == "demo_batch"
    assert meta["warehouse_binding"]["protocol"] == "warehouse_batch_v1"
    assert meta["sample_count"] == 3
    rows = [
        json.loads(line)
        for line in pack.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert {r["sample_id"] for r in rows} == {"s1", "s2", "s3"}
    assert all("reviewed_category" in r for r in rows)
    assert all(r.get("batch") == "demo_batch" for r in rows)


def test_export_idempotent_preserves_manual_edits(tmp_path: Path, ann_cfg: AnnotationConfig):
    wav = tmp_path / "a.wav"
    _write_tiny_wav(wav)
    samples = [_classified_sample("s1", wav=wav)]
    pack = tmp_path / "pack"
    export_warehouse_annotation_pack(
        samples,
        config=ann_cfg,
        dataset_path="classified.parquet",
        output=pack,
        batch="demo_batch",
        revision="r1",
        fmt="jsonl",
    )
    jsonl = pack.with_suffix(".jsonl")
    payload = json.loads(jsonl.read_text(encoding="utf-8").splitlines()[0])
    payload["gold_text"] = "人工已填"
    jsonl.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    again = export_warehouse_annotation_pack(
        samples,
        config=ann_cfg,
        dataset_path="classified.parquet",
        output=pack,
        batch="demo_batch",
        revision="r1",
        fmt="jsonl",
    )
    kept = json.loads(jsonl.read_text(encoding="utf-8").splitlines()[0])
    assert kept["gold_text"] == "人工已填"
    assert again["sample_count"] == 1


def test_reviewed_category_override_and_config_extensible(tmp_path: Path):
    allowed = load_allowed_categories(Path("configs/warehouse/categories_five_class_v2_2.yaml"))
    assert "environment_noise" in allowed

    extra_cfg = tmp_path / "cats.yaml"
    extra_cfg.write_text(
        "allowed_categories:\n  - gold_candidate\n  - custom_new_class\n",
        encoding="utf-8",
    )
    extended = load_allowed_categories(extra_cfg)
    assert "custom_new_class" in extended
    assert "environment_noise" not in extended

    sample = _classified_sample("s1", category="gold_candidate")
    result = ImportResult(samples=[sample], issues=[])
    rows = [
        {
            "sample_id": "s1",
            "category": "gold_candidate",
            "reviewed_category": "custom_new_class",
        }
    ]
    apply_reviewed_category_from_rows(result, rows, allowed_categories=extended)
    assert result.samples[0].labels["reviewed_category"] == "custom_new_class"
    assert final_category(result.samples[0]) == "custom_new_class"
    assert result.samples[0].labels["category"] == "gold_candidate"

    bad = ImportResult(samples=[_classified_sample("s2", category="gold_candidate")], issues=[])
    apply_reviewed_category_from_rows(
        bad,
        [{"sample_id": "s2", "category": "gold_candidate", "reviewed_category": "nope"}],
        allowed_categories=allowed,
    )
    assert any(i.code == "invalid_reviewed_category" for i in bad.issues)


def test_freeze_happy_path_idempotent_and_content_change_refused(
    tmp_path: Path, ann_cfg: AnnotationConfig
):
    wav_a = tmp_path / "a.wav"
    wav_b = tmp_path / "b.wav"
    _write_tiny_wav(wav_a)
    _write_tiny_wav(wav_b)
    classified = [
        _classified_sample("s1", category="environment_noise", wav=wav_a),
        _classified_sample("s2", category="voicemail", wav=wav_b),
    ]
    reviewed = [
        _complete(classified[0], gold_text="噪声确认"),
        _complete(classified[1], excluded=True),
    ]
    classified_path = tmp_path / "classified.parquet"
    Manifest(classified).save(classified_path)
    catalog_dir = tmp_path / "catalog"
    out_dir = tmp_path / "warehouses"
    evidence = tmp_path / "annotation_first.json"
    evidence.write_text('{"ok": true}', encoding="utf-8")

    first = publish_warehouse(
        reviewed,
        batch="wh_demo",
        classified=classified,
        classified_path=classified_path,
        config=ann_cfg,
        allowed_categories=load_allowed_categories(),
        catalog_dir=catalog_dir,
        output_dir=out_dir,
        run_dir=tmp_path / "run1",
        review_evidence=[evidence],
    )
    assert first.warehouse_id == warehouse_id_for_batch("wh_demo")
    assert first.idempotent_hit is False
    assert (out_dir / "wh_demo" / "manifest.parquet").is_file()
    assert (out_dir / "wh_demo" / "warehouse.json").is_file()
    meta = json.loads((out_dir / "wh_demo" / "warehouse.json").read_text(encoding="utf-8"))
    assert meta["counts"]["excluded_kept"] == 1
    assert meta["counts"]["total"] == 2

    frozen = Manifest.load(out_dir / "wh_demo" / "manifest.parquet")
    by_id = {s.id: s for s in frozen}
    assert by_id["s1"].labels.get("final_category") == "environment_noise"
    assert by_id["s2"].labels.get("warehouse_excluded") is True
    # Traceability fields
    assert by_id["s1"].labels.get("warehouse_audio_path")
    assert by_id["s1"].labels.get("category") == "environment_noise"

    catalog = ArtifactCatalog(catalog_dir)
    bound = catalog.get_warehouse_by_batch("wh_demo")
    assert bound.warehouse_id == first.warehouse_id

    second = publish_warehouse(
        reviewed,
        batch="wh_demo",
        classified=classified,
        classified_path=classified_path,
        config=ann_cfg,
        allowed_categories=load_allowed_categories(),
        catalog_dir=catalog_dir,
        output_dir=out_dir,
        run_dir=tmp_path / "run2",
        review_evidence=[evidence],
    )
    assert second.idempotent_hit is True
    assert second.warehouse_id == first.warehouse_id
    assert second.content_fingerprint == first.content_fingerprint

    changed = [
        _complete(classified[0], gold_text="改了金标"),
        reviewed[1],
    ]
    with pytest.raises(WarehouseGateError, match="different content"):
        publish_warehouse(
            changed,
            batch="wh_demo",
            classified=classified,
            classified_path=classified_path,
            config=ann_cfg,
            allowed_categories=load_allowed_categories(),
            catalog_dir=catalog_dir,
            output_dir=out_dir,
            run_dir=tmp_path / "run3",
            review_evidence=[evidence],
        )


def test_freeze_refuses_incomplete_and_subset_loss(tmp_path: Path, ann_cfg: AnnotationConfig):
    wav = tmp_path / "a.wav"
    _write_tiny_wav(wav)
    classified = [
        _classified_sample("s1", wav=wav),
        _classified_sample("s2", wav=wav, category="voicemail"),
    ]
    # Sub-pack import simulated: only s1 completed; s2 still pending → cannot freeze.
    reviewed_partial = [
        _complete(classified[0]),
        classified[1].model_copy(deep=True),  # still pending-ish
    ]
    reviewed_partial[1].labels["annotation_state"] = STATE_PENDING

    report = assert_batch_complete(
        reviewed_partial,
        config=ann_cfg,
        allowed_categories=load_allowed_categories(),
    )
    assert report.ok is False
    assert any(i.code == "pending_annotation" for i in report.issues)

    with pytest.raises(WarehouseGateError, match="missing_in_reviewed|count_mismatch"):
        publish_warehouse(
            [reviewed_partial[0]],
            batch="partial_batch",
            classified=classified,
            classified_path=tmp_path / "c.parquet",
            config=ann_cfg,
            catalog_dir=tmp_path / "catalog",
            output_dir=tmp_path / "wh",
            run_dir=tmp_path / "run",
        )


def test_binding_batch_mismatch():
    meta = {
        "warehouse_binding": {
            "batch": "batch_a",
            "classified_digest": "abc",
            "classified_manifest": "x.parquet",
        }
    }
    errs = validate_warehouse_binding(meta, batch="batch_b")
    assert any("batch" in e for e in errs)


def test_concurrent_freeze_only_one_succeeds(tmp_path: Path, ann_cfg: AnnotationConfig):
    wav = tmp_path / "a.wav"
    _write_tiny_wav(wav)
    classified = [_classified_sample("s1", wav=wav)]
    reviewed = [_complete(classified[0])]
    classified_path = tmp_path / "classified.parquet"
    Manifest(classified).save(classified_path)
    catalog_dir = tmp_path / "catalog"
    out_dir = tmp_path / "warehouses"
    results: list[object] = []
    errors: list[BaseException] = []

    def _worker(idx: int) -> None:
        try:
            r = publish_warehouse(
                reviewed,
                batch="concurrent_batch",
                classified=classified,
                classified_path=classified_path,
                config=ann_cfg,
                allowed_categories=load_allowed_categories(),
                catalog_dir=catalog_dir,
                output_dir=out_dir,
                run_dir=tmp_path / f"run_{idx}",
            )
            results.append(r)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    ok = [r for r in results if getattr(r, "warehouse_id", None)]
    # One success (fresh or idempotent); the other may hit lock or content race.
    assert len(ok) >= 1
    assert len(ok) + len(errors) == 2
    assert len(list((out_dir / "concurrent_batch").glob("warehouse.json"))) == 1
    catalog = ArtifactCatalog(catalog_dir)
    assert catalog.get_warehouse_by_batch("concurrent_batch").warehouse_id == warehouse_id_for_batch(
        "concurrent_batch"
    )


def test_operators_registered():
    import audio_engine.operators  # noqa: F401
    from audio_engine.core.registry import OperatorRegistry

    assert OperatorRegistry.get("quality.warehouse_export_annotation") is not None
    assert OperatorRegistry.get("quality.warehouse_freeze") is not None
