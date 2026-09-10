"""Adversarial checks of the actual v3 admission boundaries (013)."""
import copy
import json

import numpy as np
import pytest

from audio_engine.core.sample import Sample
from audio_engine.core.annotation_v3 import AnnotationConfig, apply_review_import_v3, build_export_rows
from audio_engine.core.annotation_v3.contract import AnnotationDraft, drafts_conflict, may_pass_formal_gold
from audio_engine.core.dataset_v3.grouping import GroupingConfig, build_leakage_groups
from audio_engine.core.dataset_v3.reservation import ReservationConfig, build_reservation
from audio_engine.core.dataset_v3.audit import evaluate_pseudo_audit, mark_pseudo_audit_outcome
from audio_engine.core.dataset_v3.audit_plan import freeze_audit_plan, validate_publish_audit
from audio_engine.core.dataset_v3.sampling import is_audited_pseudo_high, assign_eval_core_stratum
from audio_engine.core.selection_v3.input_contract import classify_run_status
from audio_engine.core.quality.dnsmos_p835 import DnsmosP835Session
from audio_engine.metrics.gate import GateConfig, evaluate_release_gate


def sample(sid, **labels):
    return Sample(id=sid, source_path=f"{sid}.wav", sha256=f"hash_{sid}", duration=3,
                  labels={"call_id": sid, "leakage_group_id": sid, **labels})


def test_random_draw_keeps_original_occurrences_and_locks_group():
    samples = [sample(str(i), call_id="one_call") for i in range(5)]
    grouping = build_leakage_groups(samples)
    reservation = build_reservation(samples, grouping, ReservationConfig(
        eval_random_target=4, calibration_target=0))
    assert len(reservation.eval_random_ids) == 4
    assert set(reservation.sample_role.values()) == {"eval_random"}
    assert build_reservation(list(reversed(samples)), grouping, ReservationConfig(
        eval_random_target=4, calibration_target=0)).content_digest == reservation.content_digest


def test_uncertain_duplicate_and_snapshot_only_are_quarantined():
    uncertain = sample("u", near_duplicate_uncertain_id="near")
    no_mapping = Sample(id="s", source_path="s.wav", sha256="h", labels={"source_snapshot_id": "snapshot"})
    samples = [uncertain, no_mapping]
    reservation = build_reservation(samples, build_leakage_groups(samples), ReservationConfig())
    assert set(reservation.sample_role.values()) == {"governance_hold"}


@pytest.mark.parametrize("entry", [{}, {"text": None}, {"status": "missing", "text": ""}])
def test_null_or_missing_prediction_is_not_success_empty(entry):
    s = sample("s")
    s.transcripts["qwen_1"] = entry
    assert classify_run_status(s, "qwen_1") == "missing"


def test_second_review_hides_model_candidate_and_missing_hash_rejected():
    s = sample("s", review_priority="P0", review_queue="manual_review", candidate_text="需要",
               annotator_id="a", annotation_state="annotated")
    cfg = AnnotationConfig()
    qid, rows, _ = build_export_rows([s], config=cfg, dataset_path="input", revision="r",
                                    view="second_review", priorities=["P0"])
    assert rows[0]["candidate_text"] == ""
    row = {**rows[0], "original_audio_sha256": "", "decision": "accepted",
           "gold_kind": "speech", "gold_text": "不需要"}
    result = apply_review_import_v3([s], [row], config=cfg, expected_queue_id=qid,
                                    expected_revision="r", review_pass="first", actor_id="a")
    assert any(i.code == "hash_mismatch" for i in result.issues)
    assert result.gold_promoted == 0


def test_adjudication_cannot_replace_missing_second_review():
    s = sample("s", annotation_state="annotated", annotator_id="a", gold_kind="speech", gold_text="需要")
    result = apply_review_import_v3([s], [{"sample_id": "s", "queue_id": "q", "queue_revision": "r",
        "original_audio_sha256": s.sha256, "decision": "accepted", "gold_kind": "speech", "gold_text": "需要"}],
        config=AnnotationConfig(), expected_queue_id="q", expected_revision="r", review_pass="adjudication", actor_id="c")
    assert any(i.code == "not_in_conflict" for i in result.issues)
    assert result.gold_promoted == 0


def test_material_audio_labels_require_adjudication():
    a = AnnotationDraft(sample_id="s", gold_kind="non_speech", gold_text="", audio_event_tags=["silence"])
    b = copy.deepcopy(a)
    b.audio_event_tags = ["music"]
    assert drafts_conflict(a, b)
    assert not may_pass_formal_gold("invented", "text", "second_review")
    assert assign_eval_core_stratum(sample("s", gold_kind="non_speech", gold_text="", human_semantic="neutral")) == "confirmed_non_speech"


def test_auto_accept_alone_never_proves_audit():
    s = sample("s", type="pseudo_high", candidate_text="你好", decision="auto_accept", pseudo_audit_passed=True)
    assert not is_audited_pseudo_high(s, require_audit=True)


def audit_fixture():
    cfg = AnnotationConfig()
    pool = [sample(str(i), type="pseudo_high", candidate_text="普通陈述", dataset_role="train_pool",
                   run_identities_digest="fixture_eight_runs", run_identities_verified=True) for i in range(501)]
    for s in pool:
        s.quality.update(dnsmos_model_digest="fixture_model", dnsmos_preprocess_version="fixture_preprocess",
                         quality_policy_version="fixture_calibration", noise_band="clean", noise_risk=False)
    plan = freeze_audit_plan(pool, cfg)
    reviewed = copy.deepcopy(pool)
    for s in reviewed:
        s.labels.update(annotation_state="second_review", is_human_verified=True,
                        gold_kind="speech", gold_text="普通陈述", annotator_id="a", reviewer_id="b",
                        speech_scope="target", human_noise="clean", human_crosstalk="false", human_semantic="neutral")
    return cfg, pool, plan, reviewed


def test_frozen_audit_passes_and_binds_scope_and_content():
    cfg, pool, plan, reviewed = audit_fixture()
    report = evaluate_pseudo_audit(reviewed, config=cfg, plan=plan)
    assert report.passed
    stamped = mark_pseudo_audit_outcome(pool, report)
    assert is_audited_pseudo_high(stamped[0], require_audit=True)
    validate_publish_audit(stamped, report.to_dict())
    stamped[0].labels["candidate_text"] = "修改后的文本"
    with pytest.raises(ValueError, match="scope mismatch"):
        validate_publish_audit(stamped, report.to_dict())
    outsider = sample("outsider", type="pseudo_high", candidate_text="普通陈述")
    assert not mark_pseudo_audit_outcome([outsider], report)[0].labels.get("pseudo_audit_passed")


def test_audit_cannot_redraw_after_skipping_unfinished_or_omit_plan():
    cfg, pool, plan, reviewed = audit_fixture()
    assert not evaluate_pseudo_audit(reviewed, config=cfg).passed
    missing = plan["draws"]["overall"][0]
    remaining = [s for s in reviewed if s.id != missing]
    report = evaluate_pseudo_audit(remaining, config=cfg, plan=plan)
    assert not report.passed
    assert any("incomplete_frozen_draw" in r for r in report.reasons)
    with pytest.raises(ValueError, match="before annotation"):
        freeze_audit_plan(reviewed, cfg)


def test_dnsmos_window_hop_and_polynomial():
    class FakeSession:
        def __init__(self):
            self.windows = []
        def run(self, _, feed):
            self.windows.append(next(iter(feed.values())))
            return [np.array([[3., 4., 2.]])]
    scorer = object.__new__(DnsmosP835Session)
    scorer._session = FakeSession()
    scorer.input_name = "input_1"
    scorer.model_digest = "test"
    result = scorer.score_array(np.arange(20 * 16000, dtype=np.float32), 16000)
    assert result.status == "success"
    assert len(scorer._session.windows) == 11
    assert scorer._session.windows[1][0, 0] == 16000
    assert result.sig == pytest.approx(np.polyval([-0.08397278, 1.22083953, 0.0052439], 3))
    assert result.bak == pytest.approx(np.polyval([-0.13166888, 1.60915514, -0.39604546], 4))


def test_gate_can_pass_only_with_complete_strata_and_intervals():
    samples = []
    for i in range(30):
        for kind, gold, semantic, base in [("speech", "不需要", "negative", "需要"),
                                          ("speech", "需要", "positive", "需要"),
                                          ("non_speech", "", "not_applicable", "")]:
            s = sample(f"{i}_{semantic}", gold_kind=kind, gold_text=gold, human_semantic=semantic)
            s.transcripts = {"base": {"text": base}, "cand": {"text": gold}}
            if kind == "speech":
                for model in ("base", "cand"):
                    s.quality.update({f"{model}_reference_length": len(gold),
                                      f"{model}_deletions": int(model == "base" and semantic == "negative"),
                                      f"{model}_substitutions": 0, f"{model}_insertions": 0})
            samples.append(s)
    cfg = GateConfig(min_improvement=.1, cer_non_inferiority=.01, positive_retention_non_inferiority=.01,
                     nshr_non_inferiority=.01, max_semantic_unk_rate=1., min_denominator=20, bootstrap_iterations=30)
    result = evaluate_release_gate(samples, baseline="base", candidate="cand", gate=cfg)
    assert result.status == "pass", result.to_dict()
    assert all(c["passed"] for c in result.checks if c["name"].startswith("ci_"))
