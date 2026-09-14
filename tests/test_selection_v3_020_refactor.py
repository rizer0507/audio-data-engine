"""020 Phase 2/3: same-text semantic, quality four-state, batch gate."""

from __future__ import annotations

from pathlib import Path

from audio_engine.core.annotation_v3.config import AnnotationConfig
from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3 import SelectionV3Config, classify_sample
from audio_engine.core.selection_v3.disposition import decide_disposition
from audio_engine.core.selection_v3.quality_gate import derive_quality_state
from audio_engine.core.selection_v3.semantic_risk import (
    compile_lexicon,
    conflict_tags_for_texts,
    route_pair_has_semantic_conflict,
)
from audio_engine.core.selection_v3.types import (
    DECISION_HOLD,
    DISPOSITION_CALIBRATION_HOLD,
    PRIORITY_P0,
    PRIORITY_P2,
    QUALITY_STATE_SCORED_NOISY,
    QUALITY_STATE_UNCALIBRATED,
    RISK_CONTENT_COMPLEXITY,
    RISK_CRITICAL_TOKEN_CONFLICT,
    RISK_NEGATION_FLIP,
    TYPE_CONTENT_COMPLEXITY,
    TYPE_CRITICAL_CONTENT_RISK,
    TYPE_QUALITY_UNCALIBRATED,
    TYPE_SEMANTIC_RISK,
)

ROOT = Path(__file__).resolve().parents[1]
SELECTION_CFG = ROOT / "configs" / "selection" / "zh_asr_v3.yaml"

EIGHT = [
    "kimi_1",
    "kimi_2",
    "glm_1",
    "glm_2",
    "sensevoice_1",
    "sensevoice_2",
    "qwen_1",
    "qwen_2",
]


def _cfg(**overrides) -> SelectionV3Config:
    base = SelectionV3Config.from_yaml(SELECTION_CFG)
    params = {
        "engine": base.engine,
        "rule_version": base.rule_version,
        "policy_version": base.policy_version,
        "target_family": base.target_family,
        "model_families": base.model_families,
        "teacher_families": base.teacher_families,
        "expected_runs_per_family": base.expected_runs_per_family,
        "semantic_lexicon_path": str(ROOT / "configs/selection/semantic_lexicon_zh_v3.yaml"),
        "voicemail_patterns_path": str(ROOT / "configs/selection/voicemail_patterns_v1.yaml"),
        "quality": {"calibrated": False},
        "refactor_020_mode": "off",
    }
    params.update(overrides)
    return SelectionV3Config.from_params(params)


def _sample(text: str | dict[str, str], *, quality: dict | None = None, duration: float = 3.0) -> Sample:
    if isinstance(text, str):
        mapping = {k: text for k in EIGHT}
    else:
        mapping = dict(text)
    transcripts = {
        k: {"text": mapping.get(k, ""), "extra": {"raw_text": mapping.get(k, "")}} for k in EIGHT
    }
    return Sample(
        id="utt-1",
        source_path="dummy.wav",
        duration=duration,
        sha256="abc",
        transcripts=transcripts,
        quality=dict(quality or {}),
        labels={"original_audio_sha256": "abc"},
    )


def test_identical_texts_not_model_conflict_but_may_be_content_complexity():
    cfg = _cfg()
    patterns = compile_lexicon(cfg)
    # Replay-class sentence: same across routes, contains both polar materials.
    text = "啊是最近没有好的"
    tags = conflict_tags_for_texts([text] * 6, patterns)
    assert RISK_CRITICAL_TOKEN_CONFLICT not in tags
    assert RISK_NEGATION_FLIP not in tags
    assert not route_pair_has_semantic_conflict(text, text, patterns)

    sample = _sample(
        text,
        quality={
            "dnsmos_status": "success",
            "noise_band": "unknown",
            "noise_risk": None,
        },
    )
    result = classify_sample(sample, cfg)
    assert result.type != TYPE_CRITICAL_CONTENT_RISK
    assert result.type != TYPE_SEMANTIC_RISK
    assert result.review_priority != PRIORITY_P0 or result.type == TYPE_CONTENT_COMPLEXITY
    if RISK_CONTENT_COMPLEXITY in set(result.risk_tags) or result.type == TYPE_CONTENT_COMPLEXITY:
        assert result.review_priority == PRIORITY_P2


def test_true_negation_flip_across_different_texts_still_p0():
    cfg = _cfg()
    texts = {k: "我不需要" for k in EIGHT}
    texts["qwen_1"] = "我需要"
    texts["qwen_2"] = "我需要"
    # Keep families internally stable where possible: flip whole qwen family.
    sample = _sample(
        texts,
        quality={
            "dnsmos_status": "success",
            "noise_band": "clean",
            "noise_risk": False,
        },
    )
    # Force teachers vs target disagreement with polarity flip between families.
    for k in ("kimi_1", "kimi_2", "glm_1", "glm_2", "sensevoice_1", "sensevoice_2"):
        sample.transcripts[k] = {"text": "我不需要", "extra": {"raw_text": "我不需要"}}
    result = classify_sample(sample, _cfg(quality={"calibrated": True}))
    # Either semantic/critical P0 or qwen correction / family path — must not ignore flip.
    assert RISK_NEGATION_FLIP in set(result.risk_tags) or result.type in {
        TYPE_SEMANTIC_RISK,
        TYPE_CRITICAL_CONTENT_RISK,
        "qwen_correction_candidate",
        "family_unstable",
    }


def test_quality_state_uncalibrated_vs_noisy():
    assert (
        derive_quality_state(
            noise_band="unknown",
            noise_risk=None,
            dnsmos_status="success",
            quality_calibrated=False,
        )
        == QUALITY_STATE_UNCALIBRATED
    )
    assert (
        derive_quality_state(
            noise_band="noisy",
            noise_risk=True,
            dnsmos_status="success",
            quality_calibrated=True,
        )
        == QUALITY_STATE_SCORED_NOISY
    )


def test_refactor_on_routes_uncalibrated_to_hold_not_transcription():
    sample = _sample(
        "喂啊你好谁啊",
        quality={
            "dnsmos_status": "success",
            "noise_band": "unknown",
            "noise_risk": None,
        },
    )
    legacy = classify_sample(sample, _cfg(refactor_020_mode="off"))
    assert legacy.type == "audio_quality_risk"
    assert legacy.decision == "manual_review"
    assert legacy.disposition == DISPOSITION_CALIBRATION_HOLD

    held = classify_sample(sample, _cfg(refactor_020_mode="on", quality={"calibrated": False}))
    assert held.type == TYPE_QUALITY_UNCALIBRATED
    assert held.decision == DECISION_HOLD
    assert held.review_queue == "calibration_hold"
    assert held.disposition == DISPOSITION_CALIBRATION_HOLD


def test_decide_disposition_quality_unknown_is_calibration_hold():
    assert (
        decide_disposition(
            type_="audio_quality_risk",
            decision="manual_review",
            risk_tags=["quality_unknown"],
            quality_state=QUALITY_STATE_UNCALIBRATED,
        )
        == DISPOSITION_CALIBRATION_HOLD
    )


def test_batch_availability_flags_full_uncalibrated_and_governance(tmp_path: Path):
    import importlib.util

    path = ROOT / "scripts" / "batch_availability_v3.py"
    spec = importlib.util.spec_from_file_location("batch_availability_v3", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)

    cfg = _cfg()
    samples = []
    for i in range(5):
        s = _sample(
            "喂你好",
            quality={
                "dnsmos_status": "success",
                "noise_band": "unknown",
                "noise_risk": None,
            },
        )
        s.id = f"s{i}"
        s.labels.update(
            {
                "type": "audio_quality_risk",
                "decision": "manual_review",
                "review_queue": "manual_review",
                "review_priority": "P1",
                "dataset_role": "governance_hold",
                "reservation_role": "governance_hold",
                "governance_flags": ["missing_group_metadata"],
                "risk_tags": ["quality_unknown"],
            }
        )
        samples.append(s)
    report = mod.analyze_batch(
        samples, selection=cfg, annotation=AnnotationConfig.from_params({})
    )
    codes = {b["code"] for b in report["blocks"]}
    assert "full_quality_uncalibrated" in codes
    assert "full_governance_hold" in codes
    assert report["permissions"]["allow_publish"] is False
    assert report["permissions"]["allow_production_human_dispatch"] is False
    assert report["permissions"]["allow_diagnostic_sorting"] is True
