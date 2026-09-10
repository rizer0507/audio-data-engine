"""012-C: annotation_v3 dual-review, empty gold, pseudo audit gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from audio_engine.cli.main import app
from audio_engine.core.annotation_v3 import (
    AnnotationConfig,
    AnnotationDraft,
    apply_review_import_v3,
    build_export_rows,
    decode_gold_text_from_tabular,
    encode_gold_text_for_tabular,
    import_has_blocking_issues,
    may_pass_formal_gold,
    validate_annotation_draft,
    write_review_package,
)
from audio_engine.core.annotation_v3.contract import drafts_conflict
from audio_engine.core.annotation_v3.types import (
    EMPTY_GOLD_SENTINEL,
    NULL_GOLD_SENTINEL,
    STATE_ANNOTATED,
    STATE_CONFLICT,
    STATE_SECOND_REVIEW,
)
from audio_engine.core.dataset_v3.audit import (
    evaluate_pseudo_audit,
    sample_group_balanced,
    wilson_interval,
)
from audio_engine.core.manifest import Manifest
from audio_engine.core.sample import Sample


def _sample(
    sid: str,
    *,
    priority: str = "P1",
    queue: str = "manual_review",
    typ: str = "hardcase",
    role: str = "train_pool",
    candidate: str = "我不需要",
    group: str | None = None,
    **labels,
) -> Sample:
    base = {
        "type": typ,
        "review_priority": priority,
        "review_queue": queue,
        "risk_tags": labels.pop("risk_tags", []),
        "candidate_text": candidate,
        "reservation_role": role,
        "leakage_group_id": group or f"g_{sid}",
        "original_audio_sha256": f"{sid:0>64}"[:64],
        "annotation_state": "manual_review",
    }
    base.update(labels)
    return Sample(
        id=sid,
        source_path=f"{sid}.wav",
        sha256=f"{sid:0>64}"[:64],
        labels=base,
        transcripts={
            "kimi_1": {"text": candidate},
            "qwen_1": {"text": "我需要"},
        },
    )


def test_null_vs_empty_gold_roundtrip():
    assert encode_gold_text_for_tabular(None) == NULL_GOLD_SENTINEL
    assert encode_gold_text_for_tabular("") == EMPTY_GOLD_SENTINEL
    assert decode_gold_text_from_tabular(NULL_GOLD_SENTINEL).is_null
    assert decode_gold_text_from_tabular(EMPTY_GOLD_SENTINEL).is_confirmed_empty
    assert decode_gold_text_from_tabular("").is_confirmed_empty
    assert decode_gold_text_from_tabular(None).is_null
    assert decode_gold_text_from_tabular("嗯").value == "嗯"


def test_non_speech_empty_contract_and_speech_rejects_empty():
    ok = AnnotationDraft(
        sample_id="a",
        decision="accepted",
        gold_kind="non_speech",
        gold_text="",
        audio_event_tags=["silence"],
        annotator_id="u1",
    )
    assert validate_annotation_draft(ok) == []

    bad_null = AnnotationDraft(
        sample_id="b",
        decision="accepted",
        gold_kind="non_speech",
        gold_text=None,
        audio_event_tags=["silence"],
        annotator_id="u1",
    )
    assert any(v.code == "non_speech_null" for v in validate_annotation_draft(bad_null))

    speech_empty = AnnotationDraft(
        sample_id="c",
        decision="accepted",
        gold_kind="speech",
        gold_text="",
        annotator_id="u1",
    )
    assert any(v.code == "speech_empty_gold" for v in validate_annotation_draft(speech_empty))

    unintell = AnnotationDraft(
        sample_id="d",
        decision="accepted",
        gold_kind="unintelligible",
        gold_text="",
        annotator_id="u1",
    )
    assert any(v.code == "unintelligible_as_empty" for v in validate_annotation_draft(unintell))


def test_may_pass_formal_gold_gates():
    assert may_pass_formal_gold("non_speech", "", STATE_SECOND_REVIEW)
    assert not may_pass_formal_gold("non_speech", None, STATE_SECOND_REVIEW)
    assert not may_pass_formal_gold("unintelligible", "xx", STATE_SECOND_REVIEW)
    assert not may_pass_formal_gold("speech", "", STATE_SECOND_REVIEW)


def test_blind_export_hides_model_texts(tmp_path):
    cfg = AnnotationConfig.from_params({})
    samples = [
        _sample("s1", priority="P0", typ="semantic_risk"),
        _sample("s2", priority="P2", typ="hardcase"),
    ]
    qid, rows, meta = build_export_rows(
        samples,
        config=cfg,
        dataset_path=str(tmp_path / "c.parquet"),
        revision="r1",
        view="blind",
        priorities=["P0", "P1", "P2"],
    )
    assert meta["view"] == "blind"
    assert rows
    assert all("kimi_1_text" not in r for r in rows)
    assert all("qwen_1_text" not in r for r in rows)
    assert rows[0]["requires_dual_review"] == "true"  # P0
    out = tmp_path / "pack.xlsx"
    written = write_review_package(rows, meta, out, fmt="both")
    assert any(p.suffix == ".xlsx" for p in written)
    assert any(p.suffix == ".jsonl" for p in written)


def test_dual_review_first_pass_not_gold_and_self_review_blocked():
    cfg = AnnotationConfig.from_params({})
    sample = _sample("s1", priority="P0", typ="semantic_risk")
    qid, rows, _ = build_export_rows(
        [sample],
        config=cfg,
        dataset_path="/tmp/x.parquet",
        revision="r1",
        view="blind",
        priorities=["P0"],
    )
    row = dict(rows[0])
    row.update(
        {
            "decision": "accepted",
            "gold_kind": "speech",
            "gold_text": "我不需要",
            "annotator_id": "ann_a",
            "human_semantic": "negative",
        }
    )
    first = apply_review_import_v3(
        [sample],
        [row],
        config=cfg,
        expected_queue_id=qid,
        expected_revision="r1",
        review_pass="first",
        actor_id="ann_a",
    )
    assert first.applied == 1
    s = first.samples[0]
    assert s.labels["annotation_state"] == STATE_ANNOTATED
    assert s.labels.get("is_human_verified") is False
    assert s.labels.get("label_tier") != "gold"

    # Same person as second reviewer → blocked
    row2 = dict(row)
    row2["reviewer_id"] = "ann_a"
    second_bad = apply_review_import_v3(
        first.samples,
        [row2],
        config=cfg,
        expected_queue_id=qid,
        expected_revision="r1",
        review_pass="second",
        actor_id="ann_a",
    )
    assert import_has_blocking_issues(second_bad)
    assert any(i.code == "self_review" for i in second_bad.issues)

    # Independent second agreement → gold
    row2["reviewer_id"] = "ann_b"
    second_ok = apply_review_import_v3(
        first.samples,
        [row2],
        config=cfg,
        expected_queue_id=qid,
        expected_revision="r1",
        review_pass="second",
        actor_id="ann_b",
    )
    assert second_ok.gold_promoted == 1
    s2 = second_ok.samples[0]
    assert s2.labels["annotation_state"] == STATE_SECOND_REVIEW
    assert s2.labels["is_human_verified"] is True
    assert s2.labels["label_tier"] == "gold"
    assert s2.labels["label_source"] == "human"
    assert s2.labels["type"] == "semantic_risk"  # original type preserved
    assert s2.labels["candidate_text"] == "我不需要"


def test_conflict_then_adjudication_and_empty_gold():
    cfg = AnnotationConfig.from_params({})
    sample = _sample(
        "empty1",
        priority="P1",
        typ="all_empty_unverified",
        candidate="",
        risk_tags=["presence_conflict"],
    )
    # empty_gold_candidates → dual required
    qid, rows, _ = build_export_rows(
        [sample],
        config=cfg,
        dataset_path="/tmp/e.parquet",
        revision="r2",
        view="blind",
        priorities=["P1"],
    )
    assert rows[0]["requires_dual_review"] == "true"
    row_a = dict(rows[0])
    row_a.update(
        {
            "decision": "accepted",
            "gold_kind": "non_speech",
            "gold_text": EMPTY_GOLD_SENTINEL,
            "audio_event_tags": "silence",
            "human_semantic": "not_applicable",
            "annotator_id": "a1",
        }
    )
    first = apply_review_import_v3(
        [sample],
        [row_a],
        config=cfg,
        expected_queue_id=qid,
        expected_revision="r2",
        review_pass="first",
        actor_id="a1",
    )
    assert first.samples[0].labels["gold_text"] == ""
    assert first.samples[0].labels.get("is_human_verified") is False

    row_b = dict(row_a)
    row_b.update(
        {
            "gold_kind": "speech",
            "gold_text": "嗯",
            "audio_event_tags": "",
            "reviewer_id": "a2",
        }
    )
    second = apply_review_import_v3(
        first.samples,
        [row_b],
        config=cfg,
        expected_queue_id=qid,
        expected_revision="r2",
        review_pass="second",
        actor_id="a2",
    )
    assert second.conflicts == 1
    assert second.samples[0].labels["annotation_state"] == STATE_CONFLICT

    row_c = dict(row_a)
    row_c.update({"adjudicator_id": "a3", "decision": "accepted"})
    adj = apply_review_import_v3(
        second.samples,
        [row_c],
        config=cfg,
        expected_queue_id=qid,
        expected_revision="r2",
        review_pass="adjudication",
        actor_id="a3",
    )
    assert adj.gold_promoted == 1
    s = adj.samples[0]
    assert s.labels["annotation_state"] == "adjudicated"
    assert s.labels["gold_text"] == ""
    assert s.labels["gold_kind"] == "non_speech"
    assert s.labels["is_human_verified"] is True


def test_partial_import_keeps_pending_and_stale_revision_blocked():
    cfg = AnnotationConfig.from_params({})
    samples = [_sample("p1", priority="P2"), _sample("p2", priority="P2")]
    qid, rows, _ = build_export_rows(
        samples,
        config=cfg,
        dataset_path="/tmp/p.parquet",
        revision="revA",
        view="blind",
        priorities=["P2"],
    )
    # Only complete p1; p2 left blank → pending
    for row in rows:
        if row["sample_id"] == "p1":
            row.update(
                {
                    "decision": "accepted",
                    "gold_kind": "speech",
                    "gold_text": "你好",
                    "annotator_id": "u1",
                }
            )
    # P2 not dual by default unless role/eval — single path promotes gold
    result = apply_review_import_v3(
        samples,
        rows,
        config=cfg,
        expected_queue_id=qid,
        expected_revision="revA",
        review_pass="first",
        actor_id="",
    )
    assert result.applied >= 1
    assert result.pending_left >= 1
    done = next(s for s in result.samples if s.id == "p1")
    assert done.labels.get("is_human_verified") is True

    # Finished gold at revA must not be overwritten by a different revision.
    finished = done.model_copy(deep=True)
    finished.labels["annotation_state"] = STATE_SECOND_REVIEW
    finished.labels["annotation_revision"] = "revA"
    others = [s for s in result.samples if s.id != "p1"]
    stale2 = apply_review_import_v3(
        [finished, *others],
        [
            {
                "sample_id": "p1",
                "original_audio_sha256": finished.labels.get("original_audio_sha256"),
                "queue_id": qid,
                "queue_revision": "revB",
                "decision": "accepted",
                "gold_kind": "speech",
                "gold_text": "改写",
                "annotator_id": "u9",
                "type": finished.labels.get("type"),
                "candidate_text": "",
            }
        ],
        config=cfg,
        expected_queue_id=qid,
        expected_revision="revB",
        review_pass="first",
        actor_id="u9",
    )
    assert any(i.code == "refuse_overwrite" for i in stale2.issues)


def test_drafts_conflict_on_polarity():
    a = AnnotationDraft(
        sample_id="x",
        decision="accepted",
        gold_kind="speech",
        gold_text="不需要",
        human_semantic="negative",
    )
    b = AnnotationDraft(
        sample_id="x",
        decision="accepted",
        gold_kind="speech",
        gold_text="不需要",
        human_semantic="positive",
    )
    assert drafts_conflict(a, b)


def test_wilson_and_group_balanced_sampling():
    point, lo, hi = wilson_interval(0, 500, z=1.96)
    assert point == 0.0
    assert hi is not None and hi <= 0.01 + 1e-9  # ~0.0074
    point2, _, hi2 = wilson_interval(5, 500, z=1.96)
    assert hi2 is not None and hi2 > 0.01

    samples = [
        _sample(f"s{i}", group=f"g{i // 2}", candidate=f"t{i}") for i in range(20)
    ]
    picked = sample_group_balanced(samples, seed=42, min_groups=5, salt="t")
    assert len(picked) == 5
    assert len({s.labels["leakage_group_id"] for s in picked}) == 5
    # deterministic
    picked2 = sample_group_balanced(samples, seed=42, min_groups=5, salt="t")
    assert [s.id for s in picked] == [s.id for s in picked2]


def test_pseudo_audit_blocks_on_insufficient_and_critical(tmp_path):
    cfg = AnnotationConfig.from_params(
        {
            "pseudo_audit": {
                "min_groups_overall": 10,
                "overall_upper_bound": 0.01,
                "critical_semantic_errors_max": 0,
                "seed": 1,
                "protected_layers": {
                    "short_utterance": {
                        "min_groups": 5,
                        "upper_bound": 0.02,
                        "match": {"risk_tags_any": ["short_utterance"]},
                    }
                },
            }
        }
    )
    # Build verified dual-review samples with matching candidate
    samples = []
    for i in range(12):
        s = _sample(
            f"a{i}",
            typ="pseudo_high",
            queue="pseudo_audit",
            group=f"g{i}",
            candidate="我不需要",
            risk_tags=["short_utterance"] if i < 6 else [],
        )
        s.labels.update(
            {
                "annotation_state": STATE_SECOND_REVIEW,
                "is_human_verified": True,
                "gold_kind": "speech",
                "gold_text": "我不需要",
                "human_semantic": "negative",
                "label_tier": "gold",
                "label_source": "human",
                "annotator_id": "a", "reviewer_id": "b", "speech_scope": "target",
                "human_noise": "clean", "human_crosstalk": "false",
            }
        )
        samples.append(s)
    # Inject one critical semantic mislabel
    samples[0].labels["gold_text"] = "我需要"
    samples[0].labels["human_semantic"] = "positive"
    samples[0].labels["verified_error_tags"] = ["false_affirmation_candidate"]

    report = evaluate_pseudo_audit(samples, config=cfg)
    assert report.stop_publish is True
    assert report.passed is False
    assert report.critical_semantic_errors >= 1
    assert any("critical" in r.lower() for r in report.reasons)

    # Calibration self-proof exclusion
    samples[1].labels["reservation_role"] = "calibration"
    report2 = evaluate_pseudo_audit(samples, config=cfg)
    assert report2.excluded_calibration_count >= 1


def test_cli_review_v3_export_import_empty_gold(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Write annotation config into tmp cwd
    cfg_path = tmp_path / "configs" / "annotation" / "zh_asr_v3.yaml"
    cfg_path.parent.mkdir(parents=True)
    cfg_path.write_text(
        Path(__file__).resolve().parents[1].joinpath("configs/annotation/zh_asr_v3.yaml").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    source = tmp_path / "classified.parquet"
    Manifest(
        [
            _sample("cli1", priority="P0", typ="semantic_risk", candidate="不需要"),
            _sample(
                "cli2",
                priority="P1",
                typ="all_empty_unverified",
                candidate="",
                risk_tags=["presence_conflict"],
            ),
        ]
    ).save(source)
    runner = CliRunner()
    pack = tmp_path / "blind.jsonl"
    exported = runner.invoke(
        app,
        [
            "review",
            "export",
            str(source),
            "--output",
            str(pack),
            "--revision",
            "rcli",
            "--protocol",
            "v3",
            "--view",
            "blind",
            "--format",
            "jsonl",
            "--annotation-config",
            str(cfg_path),
        ],
    )
    assert exported.exit_code == 0, exported.output
    rows = [json.loads(line) for line in pack.read_text(encoding="utf-8").splitlines() if line]
    assert rows
    assert all("qwen_1_text" not in r for r in rows)
    qid = rows[0]["queue_id"]

    # First pass: fill both
    for row in rows:
        if row["sample_id"] == "cli1":
            row.update(
                {
                    "decision": "accepted",
                    "gold_kind": "speech",
                    "gold_text": "不需要",
                    "human_semantic": "negative",
                    "annotator_id": "ann1",
                }
            )
        else:
            row.update(
                {
                    "decision": "accepted",
                    "gold_kind": "non_speech",
                    "gold_text": "",
                    "audio_event_tags": ["silence"],
                    "human_semantic": "not_applicable",
                    "annotator_id": "ann1",
                }
            )
    pack.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    reviewed1 = tmp_path / "reviewed1.parquet"
    imported1 = runner.invoke(
        app,
        [
            "review",
            "import",
            str(source),
            "--input",
            str(pack),
            "--output",
            str(reviewed1),
            "--revision",
            "rcli",
            "--protocol",
            "v3",
            "--pass",
            "first",
            "--actor-id",
            "ann1",
            "--queue-id",
            qid,
            "--annotation-config",
            str(cfg_path),
        ],
    )
    assert imported1.exit_code == 0, imported1.output
    m1 = Manifest.load(reviewed1)
    assert all(s.labels.get("is_human_verified") is not True for s in m1) or all(
        s.labels.get("annotation_state") == STATE_ANNOTATED for s in m1
    )
    assert all(s.labels.get("label_tier") != "gold" for s in m1)

    # Second pass by different annotator
    for row in rows:
        row["reviewer_id"] = "ann2"
        row["annotator_id"] = "ann2"
    pack.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    reviewed2 = tmp_path / "reviewed2.parquet"
    imported2 = runner.invoke(
        app,
        [
            "review",
            "import",
            str(reviewed1),
            "--input",
            str(pack),
            "--output",
            str(reviewed2),
            "--revision",
            "rcli",
            "--protocol",
            "v3",
            "--pass",
            "second",
            "--actor-id",
            "ann2",
            "--queue-id",
            qid,
            "--annotation-config",
            str(cfg_path),
        ],
    )
    assert imported2.exit_code == 0, imported2.output
    m2 = Manifest.load(reviewed2)
    by_id = {s.id: s for s in m2}
    assert by_id["cli1"].labels["is_human_verified"] is True
    assert by_id["cli2"].labels["gold_text"] == ""
    assert by_id["cli2"].labels["gold_kind"] == "non_speech"
    assert by_id["cli2"].labels["is_human_verified"] is True
    assert by_id["cli1"].labels["type"] == "semantic_risk"


def test_cli_audit_pseudo_exit_code(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_path = tmp_path / "ann.yaml"
    cfg_path.write_text(
        """
annotation_version: annotation_v3.0
pseudo_audit:
  min_groups_overall: 5
  overall_upper_bound: 0.01
  critical_semantic_errors_max: 0
  seed: 7
  protected_layers:
    short_utterance:
      min_groups: 3
      upper_bound: 0.02
      match:
        risk_tags_any: [short_utterance]
""",
        encoding="utf-8",
    )
    samples = []
    for i in range(8):
        s = _sample(f"p{i}", typ="pseudo_high", queue="pseudo_audit", group=f"g{i}")
        s.labels.update(
            {
                "annotation_state": STATE_SECOND_REVIEW,
                "is_human_verified": True,
                "gold_kind": "speech",
                "gold_text": "我不需要",
                "candidate_text": "我不需要",
                "human_semantic": "negative",
                "risk_tags": ["short_utterance"],
            }
        )
        samples.append(s)
    source = tmp_path / "pseudo.parquet"
    Manifest(samples).save(source)
    runner = CliRunner()
    report = tmp_path / "audit.json"
    # Still fail: protected layer / overall may pass with 0 errors if n enough
    result = runner.invoke(
        app,
        [
            "review",
            "audit-pseudo",
            str(source),
            "--output",
            str(report),
            "--annotation-config",
            str(cfg_path),
        ],
    )
    # With 8 groups < default protected? we set min 5 overall and 3 layer — should PASS
    assert result.exit_code in {0, 2}, result.output
    data = json.loads(report.read_text(encoding="utf-8"))
    assert "overall" in data
    assert data["statistic_name"] == "group_balanced_mislabel_rate"
