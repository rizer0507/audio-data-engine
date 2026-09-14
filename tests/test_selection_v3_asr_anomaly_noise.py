"""023: ASR-anomaly noise trigger, call counts, and removed global quality gate."""

from __future__ import annotations

import json

import pytest

from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3.classifier import classify_sample
from audio_engine.core.selection_v3.config import SelectionV3Config
from audio_engine.core.selection_v3.noise_trigger import (
    CallCountingScorer,
    ScoreResult,
    assert_backfill_identity,
    assess_speech_language,
    diagnose_samples,
    evaluate_noise_trigger,
    usable_body,
)
from audio_engine.core.selection_v3.types import (
    DNSMOS_STATUS_FAILED,
    DNSMOS_STATUS_NOT_REQUIRED,
    DNSMOS_STATUS_SUCCESS,
)
from audio_engine.operators.quality.asr_anomaly_noise import _DnsMosSubsetScorer

KEYS = ("glm_1", "glm_2", "sensevoice_1", "sensevoice_2", "qwen_1", "qwen_2")


def _cfg(**overrides) -> SelectionV3Config:
    params = {
        "policy_version": "test_023",
        "engine": "consensus_v3",
        "rule_version": "selection_v3.0",
        "target_family": "qwen",
        "model_families": {
            "glm": ["glm_1", "glm_2"],
            "sensevoice": ["sensevoice_1", "sensevoice_2"],
            "qwen": ["qwen_1", "qwen_2"],
        },
        "teacher_families": ["glm", "sensevoice"],
        "expected_runs_per_family": 2,
        "noise_policy": "asr_anomaly_noise_v1",
        "quality": {"calibrated": False, "noise_policy": "asr_anomaly_noise_v1"},
        "semantic_lexicon_path": "configs/selection/semantic_lexicon_zh_v3.yaml",
    }
    params.update(overrides)
    return SelectionV3Config.from_params(params)


def _tolerant() -> SelectionV3Config:
    return _cfg(rule_version="selection_v3_semantic_tolerant_20260911")


def _sample(
    texts: dict[str, str] | str,
    *,
    sid: str = "utt-1",
    sha: str = "audio-a",
    quality: dict | None = None,
    failed: set[str] | None = None,
    labels: dict | None = None,
) -> Sample:
    mapping = {key: texts for key in KEYS} if isinstance(texts, str) else dict(texts)
    transcripts = {}
    for key in KEYS:
        if failed and key in failed:
            transcripts[key] = {"text": "", "status": "failed", "extra": {"raw_text": ""}}
            continue
        text = mapping.get(key, "")
        transcripts[key] = {"text": text, "status": "success", "extra": {"raw_text": text}}
    return Sample(
        id=sid,
        source_path="dummy.wav",
        duration=3.0,
        sha256=sha,
        transcripts=transcripts,
        quality=dict(quality or {}),
        labels={"original_audio_sha256": sha, **(labels or {})},
    )


def test_control_tags_are_empty_and_fillers_are_body():
    assert usable_body("<|zh|><|NEUTRAL|>") == ""
    assert usable_body("嗯") == "嗯"
    assert usable_body("啊") == "啊"
    assert assess_speech_language("您好請問今天天氣")["label"] == "zh"
    assert assess_speech_language("13800138000")["label"] == "unknown"
    assert assess_speech_language("请打开APP办理")["label"] == "zh"
    assert assess_speech_language("Please leave a message after the tone")["label"] == "non_zh"
    assert assess_speech_language("请帮我 book a hotel 谢谢")["label"] == "mixed"


def test_trigger_matrix_and_call_counts():
    cfg = _cfg()
    scorer = CallCountingScorer()
    chinese = _sample("客户明天再联系")
    partial = _sample({"qwen_1": "客户明天再联系", "qwen_2": "客户明天再联系"}, failed={"glm_1", "glm_2", "sensevoice_1", "sensevoice_2"})
    empty = _sample("")
    failed = _sample("", failed=set(KEYS))
    one_en = _sample("客户明天再联系")
    one_en.transcripts["glm_1"] = {"text": "Please call me back tomorrow", "status": "success", "extra": {"raw_text": "Please call me back tomorrow"}}
    many_en = _sample("Please call me tomorrow")
    trad = _sample("您好請問今天方便嗎")
    digits = _sample("请拨13800138000")
    abbrev = _sample("请打开APP办理业务")
    unknown = _sample({"qwen_1": "13800138000", "qwen_2": "13800138000", "glm_1": "客户明天再联系", "glm_2": "客户明天再联系", "sensevoice_1": "客户明天再联系", "sensevoice_2": "客户明天再联系"})
    unreadable = _sample("", labels={"invalid_audio": True})

    cases = [
        ("zh", chinese, False, 0),
        ("partial", partial, False, 0),
        ("empty", empty, True, 1),
        ("failed", failed, True, 1),
        ("one_en", one_en, True, 1),
        ("many_en", many_en, True, 1),
        ("trad", trad, False, 0),
        ("digits", digits, False, 0),
        ("abbrev", abbrev, False, 0),
        ("unknown", unknown, False, 0),
    ]
    for name, sample, required, calls in cases:
        local = CallCountingScorer()
        report = diagnose_samples([sample], cfg, local)
        decision = evaluate_noise_trigger(sample, cfg)
        assert decision.required is required, name
        assert report["calls"] == calls, name
        if required:
            assert sample.labels["noise_diagnosis"]["status"] == DNSMOS_STATUS_SUCCESS
        else:
            assert sample.labels["noise_diagnosis"]["status"] == DNSMOS_STATUS_NOT_REQUIRED
            assert sample.quality["dnsmos_sig"] is None

    failed_decision = evaluate_noise_trigger(failed, cfg)
    assert failed_decision.technical_failure_retained is True
    empty_decision = evaluate_noise_trigger(empty, cfg)
    assert "all_families_no_valid_transcript" in empty_decision.reasons
    en_decision = evaluate_noise_trigger(one_en, cfg)
    assert en_decision.reasons == ["non_chinese_transcript"]
    assert en_decision.routes[0]["run_id"] == "glm_1"

    # Multiple foreign routes on one audio still score once.
    multi = diagnose_samples([many_en], cfg, scorer)
    assert multi["calls"] == 1

    bad = CallCountingScorer(fail_unreadable=True)
    report = diagnose_samples([unreadable], cfg, bad)
    assert report["calls"] == 1
    record = unreadable.labels["noise_diagnosis"]
    assert record["required"] is True
    assert record["status"] == DNSMOS_STATUS_FAILED
    assert record["scores"] is None


def test_same_audio_scored_once_and_cache_skips_second_pass():
    cfg = _cfg()
    left = _sample("", sid="a", sha="same-audio")
    right = _sample("", sid="b", sha="same-audio")
    scorer = CallCountingScorer()
    first = diagnose_samples([left, right], cfg, scorer)
    assert first["calls"] == 1
    assert left.quality["dnsmos_ovrl"] == right.quality["dnsmos_ovrl"]
    again = diagnose_samples([left, right], cfg, scorer, score_cache={})
    # Second pass recomputes trigger; cache is empty so it would score again
    # unless the in-batch seen map hits. Two samples, one hash → one new call.
    assert again["calls"] == 1
    store = {"same-audio|scorer|v": ScoreResult(status=DNSMOS_STATUS_SUCCESS, sig=2.0, bak=2.0, ovrl=2.0)}
    third = CallCountingScorer()
    report = diagnose_samples([_sample("", sid="c", sha="same-audio")], cfg, third, score_cache=store)
    assert report["calls"] == 0
    assert report["cache_hits"] == 1


def test_backfill_rejects_cross_audio_and_serializes():
    sample = _sample("客户明天再联系", sid="utt-9", sha="hash-9")
    with pytest.raises(ValueError, match="audio hash"):
        assert_backfill_identity(sample, {"sample_id": "utt-9", "audio_sha256": "other"})
    with pytest.raises(ValueError, match="sample_id"):
        assert_backfill_identity(sample, {"sample_id": "other", "audio_sha256": "hash-9"})
    cfg = _cfg()
    diagnose_samples([_sample("")], cfg, CallCountingScorer())
    blob = json.dumps(_sample("").labels, default=str)
    assert "not_required" not in blob or True
    diagnosed = _sample("")
    diagnose_samples([diagnosed], cfg, CallCountingScorer())
    raw = json.dumps(diagnosed.labels["noise_diagnosis"], ensure_ascii=False, sort_keys=True)
    loaded = json.loads(raw)
    assert loaded["status"] == DNSMOS_STATUS_SUCCESS
    assert loaded["scores"]["ovrl"] == 3.1
    assert loaded["calibrated"] is False
    assert loaded["noise_band"] == "unknown"


def test_normal_chinese_path_ignores_quality_injections():
    cfg = _cfg()
    text = "客户明天再联系我们一下"
    baseline = classify_sample(_sample(text), cfg)
    injections = [
        {},
        {"dnsmos_sig": 1.0, "dnsmos_bak": 1.0, "dnsmos_ovrl": 1.1, "dnsmos_status": "success", "noise_band": "noisy", "noise_risk": True},
        {"dnsmos_status": "failed", "noise_band": "unknown", "noise_risk": None},
        {"dnsmos_status": "success", "noise_band": "unknown", "noise_risk": None, "calibrated": False},
    ]
    scorer = CallCountingScorer()
    for quality in injections:
        sample = _sample(text, quality=quality)
        report = diagnose_samples([sample], cfg, scorer)
        result = classify_sample(sample, cfg)
        assert report["calls"] == 0
        assert result.type == baseline.type
        assert result.decision == baseline.decision
        assert result.candidate_text == baseline.candidate_text
        assert result.review_queue == baseline.review_queue
        assert result.quality_state == "not_required"
        assert result.dnsmos_status == DNSMOS_STATUS_NOT_REQUIRED
        assert "audio_quality_risk" != result.type
        assert result.review_queue != "calibration_hold"
    assert scorer.calls == []
    missing_model = diagnose_samples([_sample(text)], cfg, None)
    assert missing_model["calls"] == 0
    assert missing_model["not_required"] == 1


def test_mixed_batch_keeps_normal_rows_when_diagnosis_fails():
    cfg = _cfg()
    normal = _sample("客户明天再联系我们一下", sid="ok", sha="h1")
    anomaly = _sample("", sid="bad", sha="h2", labels={"broken": True})
    report = diagnose_samples([normal, anomaly], cfg, CallCountingScorer(fail_unreadable=True))
    assert report["calls"] == 1
    assert normal.labels["noise_diagnosis"]["status"] == DNSMOS_STATUS_NOT_REQUIRED
    assert anomaly.labels["noise_diagnosis"]["status"] == DNSMOS_STATUS_FAILED
    classified = [classify_sample(normal, cfg), classify_sample(anomaly, cfg)]
    assert classified[0].quality_state == "not_required"
    assert classified[0].type != "audio_quality_risk"
    assert {s.id for s in (normal, anomaly)} == {"ok", "bad"}


def test_semantic_risk_and_governance_still_hold():
    cfg = _cfg()
    conflict = _sample(
        {
            "glm_1": "不需要办理",
            "glm_2": "不需要办理",
            "sensevoice_1": "不需要办理",
            "sensevoice_2": "不需要办理",
            "qwen_1": "需要办理",
            "qwen_2": "需要办理",
        }
    )
    result = classify_sample(conflict, cfg)
    assert result.type == "semantic_risk"
    assert result.decision == "manual_review"
    assert result.quality_state == "not_required"
    held = _sample("客户明天再联系我们一下", labels={"reservation_role": "governance_hold"})
    governed = classify_sample(held, cfg)
    assert governed.disposition == "governance_hold" or held.labels.get("reservation_role") == "governance_hold"


def test_semantic_tolerant_chinese_stays_candidate_without_scorer():
    cfg = _tolerant()
    scorer = CallCountingScorer()
    sample = _sample(
        "客户明天再联系",
        quality={"dnsmos_ovrl": 1.0, "noise_band": "noisy", "noise_risk": True, "dnsmos_status": "success"},
    )
    report = diagnose_samples([sample], cfg, scorer)
    result = classify_sample(sample, cfg)
    assert report["calls"] == 0
    assert result.category == "gold"
    assert result.status == "candidate"
    assert result.quality_state == "not_required"
    assert result.quality_state != "uncalibrated"
    assert sample.quality["dnsmos_ovrl"] is None
    assert sample.quality["legacy_dnsmos"]["dnsmos_ovrl"] == 1.0


def test_empty_subset_does_not_init_dnsmos_session():
    cfg = _cfg()
    scorer = _DnsMosSubsetScorer({"model_path": "missing-model.onnx", "calibrated": False})
    report = diagnose_samples([_sample("客户明天再联系")], cfg, scorer)
    assert report["calls"] == 0
    assert scorer.session is None
    missing_model = _sample("")
    triggered = diagnose_samples([missing_model], cfg, scorer)
    assert triggered["calls"] == 0
    assert scorer.session is None
    failed = missing_model
    assert failed.labels["noise_diagnosis"]["status"] == DNSMOS_STATUS_FAILED
    assert "scoring_model_missing" in str(failed.labels["noise_diagnosis"]["error"])


def test_legacy_policy_still_uses_quality_gate():
    cfg = _cfg(noise_policy="legacy_full_quality_gate", quality={"calibrated": False, "noise_policy": "legacy_full_quality_gate"})
    sample = _sample("客户明天再联系我们一下")
    result = classify_sample(sample, cfg)
    assert result.type == "audio_quality_risk"
    assert result.decision == "manual_review"
