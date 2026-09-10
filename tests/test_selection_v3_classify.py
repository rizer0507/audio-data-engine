"""012-B: DNSMOS risk derivation + selection_v3 classification acceptance cases."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import audio_engine.operators  # noqa: F401
from audio_engine.core.operator import OperatorConfig
from audio_engine.core.quality.dnsmos_p835 import (
    derive_noise_band_explicit,
    prepare_audio_for_dnsmos,
    scores_to_quality_dict,
    DnsmosScores,
)
from audio_engine.core.registry import OperatorRegistry
from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3 import SelectionV3Config, classify_sample
from audio_engine.core.selection_v3.types import (
    DECISION_AUDIT_PENDING,
    DECISION_MANUAL_REVIEW,
    DECISION_RETRY,
    NOISE_BAND_CLEAN,
    NOISE_BAND_NOISY,
    NOISE_BAND_UNKNOWN,
    PRIORITY_P0,
    RISK_FILLER_AFFIRMATION,
    RISK_NEGATION_FLIP,
    RISK_PRESENCE_CONFLICT,
    RISK_REJECTION_SANITIZATION,
    TYPE_ALL_EMPTY_UNVERIFIED,
    TYPE_AUDIO_QUALITY_RISK,
    TYPE_HARDCASE,
    TYPE_INFERENCE_INCOMPLETE,
    TYPE_PSEUDO_HIGH,
    TYPE_QWEN_CORRECTION_CANDIDATE,
    TYPE_SEMANTIC_RISK,
    TYPE_SPEECH_PRESENCE_DISAGREEMENT,
)

ROOT = Path(__file__).resolve().parents[1]
SELECTION_CFG = ROOT / "configs" / "selection" / "zh_asr_v3.yaml"
DNSMOS_CFG = ROOT / "configs" / "quality" / "dnsmos_p835.yaml"
MODEL_PATH = ROOT / "third_party" / "dnsmos" / "sig_bak_ovr.onnx"

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
TEACHERS = ["kimi_1", "kimi_2", "glm_1", "glm_2", "sensevoice_1", "sensevoice_2"]


def _cfg(**overrides) -> SelectionV3Config:
    base = SelectionV3Config.from_yaml(SELECTION_CFG)
    # Force calibrated quality consumption off at config level is irrelevant —
    # quality fields are injected on the sample.
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
        "similarity": {
            "family_threshold": base.family_threshold,
            "teacher_consensus_threshold": base.teacher_consensus_threshold,
            "pseudo_high_min_similarity": base.pseudo_high_min_similarity,
            "pseudo_medium_min_similarity": base.pseudo_medium_min_similarity,
        },
        "short_utterance": {
            "max_audio_sec": base.short_audio_sec,
            "max_text_chars": base.short_text_chars,
        },
    }
    params.update(overrides)
    return SelectionV3Config.from_params(params)


def _clean_quality() -> dict:
    return {
        "dnsmos_sig": 4.0,
        "dnsmos_bak": 4.0,
        "dnsmos_ovrl": 3.8,
        "dnsmos_status": "success",
        "noise_band": NOISE_BAND_CLEAN,
        "noise_risk": False,
        "quality_policy_version": "quality_policy_v3.0",
    }


def _sample(
    texts: dict[str, str] | str,
    *,
    quality: dict | None = None,
    duration: float = 3.0,
    failed: set[str] | None = None,
) -> Sample:
    if isinstance(texts, str):
        mapping = {k: texts for k in EIGHT}
    else:
        mapping = dict(texts)
    transcripts = {}
    for key in EIGHT:
        if failed and key in failed:
            transcripts[key] = {"text": "", "status": "failed"}
            continue
        text = mapping.get(key, "")
        transcripts[key] = {
            "text": text,
            "extra": {"raw_text": text},
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


# ---------------------------------------------------------------------------
# DNSMOS risk derivation (no model required)
# ---------------------------------------------------------------------------


def test_noise_risk_or_rule_and_unknown_when_uncalibrated():
    risk = derive_noise_band_explicit(
        bak=2.5,
        ovrl=3.5,
        clean_bak=3.5,
        clean_ovrl=3.2,
        moderate_bak=3.0,
        moderate_ovrl=2.8,
        calibrated=True,
        status="success",
    )
    assert risk.noise_risk is True
    assert risk.noise_band == NOISE_BAND_NOISY

    uncal = derive_noise_band_explicit(
        bak=4.0,
        ovrl=4.0,
        clean_bak=3.5,
        clean_ovrl=3.2,
        moderate_bak=3.0,
        moderate_ovrl=2.8,
        calibrated=False,
        status="success",
    )
    assert uncal.noise_risk is None
    assert uncal.noise_band == NOISE_BAND_UNKNOWN


def test_missing_score_never_defaults_clean():
    risk = derive_noise_band_explicit(
        bak=None,
        ovrl=3.0,
        clean_bak=3.5,
        clean_ovrl=3.2,
        moderate_bak=3.0,
        moderate_ovrl=2.8,
        calibrated=True,
        status="success",
    )
    assert risk.noise_risk is None
    assert risk.noise_band == NOISE_BAND_UNKNOWN
    payload = scores_to_quality_dict(
        DnsmosScores(sig=None, bak=None, ovrl=None, status="failed", error="boom"),
        risk,
    )
    assert payload["dnsmos_sig"] is None
    assert payload["noise_band"] == NOISE_BAND_UNKNOWN


def test_short_audio_repeat_pad_strategy_does_not_claim_source_mutation():
    audio = np.zeros(8000, dtype=np.float32)  # 0.5s @ 16k
    prepared, strategy = prepare_audio_for_dnsmos(audio, 16000)
    assert strategy == "repeat_pad_to_window"
    assert len(prepared) == 8000 * 32  # official repeated doubling, then 1-second windows
    assert len(audio) == 8000  # original buffer length unchanged by contract


def test_dnsmos_operator_registered():
    assert OperatorRegistry.get("quality.dnsmos") is not None


def test_dnsmos_startup_fails_without_model(tmp_path: Path):
    op = OperatorRegistry.get("quality.dnsmos")
    sample = Sample(id="x", source_path="x.wav", audio={"resampled_16k": str(tmp_path / "a.wav")})
    # write tiny wav
    import soundfile as sf

    sf.write(str(tmp_path / "a.wav"), np.zeros(1600, dtype=np.float32), 16000)
    cfg = OperatorConfig(
        params={
            "model_path": str(tmp_path / "missing.onnx"),
            "calibrated": False,
            "input_audio_key": "resampled_16k",
        },
        cache_dir=tmp_path / "cache",
        force=True,
    )
    with pytest.raises((FileNotFoundError, RuntimeError)):
        op.process(sample, cfg)


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="official DNSMOS ONNX not vendored yet")
def test_dnsmos_fixed_wav_against_official_model(tmp_path: Path):
    """Official score acceptance — requires third_party model; never pass via mock."""
    import soundfile as sf

    wav = tmp_path / "ref.wav"
    # 1s of low-amplitude noise
    rng = np.random.default_rng(0)
    sf.write(str(wav), (rng.normal(0, 0.01, 16000)).astype(np.float32), 16000)
    op = OperatorRegistry.get("quality.dnsmos")
    sample = Sample(
        id="ref",
        source_path=str(wav),
        audio={"resampled_16k": str(wav)},
        sha256="ref",
    )
    cfg = OperatorConfig(
        params={"config_path": str(DNSMOS_CFG), "calibrated": False},
        cache_dir=tmp_path / "cache",
        force=True,
    )
    result = op.process(sample, cfg)
    q = result.sample.quality
    assert q["dnsmos_status"] == "success"
    assert q["dnsmos_sig"] is not None
    assert q["noise_band"] == NOISE_BAND_UNKNOWN  # uncalibrated
    assert q["noise_risk"] is None


# ---------------------------------------------------------------------------
# Classification acceptance (§15.1)
# ---------------------------------------------------------------------------


def test_four_family_我不需要_pseudo_high():
    sample = _sample("我不需要", quality=_clean_quality(), duration=1.5)
    result = classify_sample(sample, _cfg())
    assert result.type == TYPE_PSEUDO_HIGH
    assert result.decision == DECISION_AUDIT_PENDING
    assert result.is_human_verified is False
    assert result.label_tier == "pseudo_high"
    assert "不需要" in (result.candidate_text or "")
    assert result.configured_family_count == 4
    assert result.support_ratio_of_4 == 1.0
    labels = result.to_labels("selection_zh_asr_v3_0")
    assert labels["support_ratio"] == 1.0
    assert labels["configured_family_count"] == 4


def test_three_family_我不需要_pseudo_high():
    three_cfg = _cfg(
        model_families={
            "glm": ["glm_1", "glm_2"],
            "sensevoice": ["sensevoice_1", "sensevoice_2"],
            "qwen": ["qwen_1", "qwen_2"],
        },
        teacher_families=["glm", "sensevoice"],
    )
    six = [
        "glm_1",
        "glm_2",
        "sensevoice_1",
        "sensevoice_2",
        "qwen_1",
        "qwen_2",
    ]
    transcripts = {
        k: {"text": "我不需要", "extra": {"raw_text": "我不需要"}} for k in six
    }
    sample = Sample(
        id="s3",
        source_path="s3.wav",
        sha256="h",
        duration=1.5,
        transcripts=transcripts,
        labels={},
        quality=_clean_quality(),
    )
    result = classify_sample(sample, three_cfg)
    assert result.type == TYPE_PSEUDO_HIGH
    assert result.decision == DECISION_AUDIT_PENDING
    assert result.configured_family_count == 3
    assert result.support_family_count == 3
    assert result.support_ratio_of_4 == 1.0
    assert result.reason == "configured_family_strict_consensus"


def test_teachers_neg_qwen_pos_semantic_and_correction():
    texts = {k: "我不需要" for k in TEACHERS}
    texts["qwen_1"] = "我需要"
    texts["qwen_2"] = "我需要"
    result = classify_sample(_sample(texts, quality=_clean_quality()), _cfg())
    assert result.type == TYPE_SEMANTIC_RISK
    assert result.decision == DECISION_MANUAL_REVIEW
    assert result.review_priority == PRIORITY_P0
    assert RISK_NEGATION_FLIP in result.risk_tags
    assert result.qwen_correction_candidate is True


def test_qwen_internal_negation_flip_p0():
    texts = {k: "需要" for k in EIGHT}
    texts["qwen_1"] = "需要"
    texts["qwen_2"] = "不需要"
    result = classify_sample(_sample(texts, quality=_clean_quality()), _cfg())
    assert result.type == TYPE_SEMANTIC_RISK
    assert result.review_priority == PRIORITY_P0


def test_seven_to_one_negation_not_majority_accept():
    texts = {k: "需要" for k in EIGHT}
    texts["kimi_2"] = "不需要"
    result = classify_sample(_sample(texts, quality=_clean_quality()), _cfg())
    assert result.type == TYPE_SEMANTIC_RISK
    assert result.decision == DECISION_MANUAL_REVIEW
    assert result.review_priority == PRIORITY_P0


def test_filler_affirmation_p0():
    texts = {k: "嗯" for k in EIGHT}
    texts["qwen_1"] = "需要"
    texts["qwen_2"] = "需要"
    result = classify_sample(_sample(texts, quality=_clean_quality()), _cfg())
    assert RISK_FILLER_AFFIRMATION in result.risk_tags or result.type == TYPE_SEMANTIC_RISK
    assert result.review_priority == PRIORITY_P0


def test_rejection_sanitization_p0():
    texts = {k: "好的" for k in EIGHT}
    texts["kimi_1"] = "别再打电话"
    texts["kimi_2"] = "别再打电话"
    result = classify_sample(_sample(texts, quality=_clean_quality()), _cfg())
    assert result.type == TYPE_SEMANTIC_RISK
    assert RISK_REJECTION_SANITIZATION in result.risk_tags
    assert result.review_priority == PRIORITY_P0


def test_all_empty_unverified_regardless_of_dnsmos():
    sample = _sample("", quality={**_clean_quality(), "dnsmos_bak": 1.0, "noise_risk": True, "noise_band": NOISE_BAND_NOISY})
    # empty string → success_empty for all
    for key in EIGHT:
        sample.transcripts[key] = {"text": "", "extra": {"raw_text": ""}}
    result = classify_sample(sample, _cfg())
    assert result.type == TYPE_ALL_EMPTY_UNVERIFIED
    assert result.decision == DECISION_MANUAL_REVIEW


def test_presence_conflict_p0_with_affirmation():
    texts = {k: "" for k in EIGHT}
    texts["qwen_1"] = "好的"
    sample = _sample(texts, quality=_clean_quality())
    for key, text in texts.items():
        sample.transcripts[key] = {"text": text, "extra": {"raw_text": text}}
    result = classify_sample(sample, _cfg())
    assert result.type == TYPE_SPEECH_PRESENCE_DISAGREEMENT
    assert RISK_PRESENCE_CONFLICT in result.risk_tags
    assert result.review_priority == PRIORITY_P0


def test_one_failed_seven_empty_is_incomplete_not_all_empty():
    sample = _sample("", quality=_clean_quality(), failed={"kimi_1"})
    for key in EIGHT:
        if key == "kimi_1":
            continue
        sample.transcripts[key] = {"text": "", "extra": {"raw_text": ""}}
    result = classify_sample(sample, _cfg())
    assert result.type == TYPE_INFERENCE_INCOMPLETE
    assert result.decision == DECISION_RETRY


def test_teacher_typo_qwen_correction_without_semantic():
    # Ordinary character difference, no polarity conflict
    texts = {k: "今天天气不错" for k in TEACHERS}
    texts["qwen_1"] = "今天天气不措"
    texts["qwen_2"] = "今天天气不措"
    result = classify_sample(_sample(texts, quality=_clean_quality(), duration=4.0), _cfg())
    assert result.type == TYPE_QWEN_CORRECTION_CANDIDATE
    assert result.qwen_correction_candidate is True
    assert result.decision == DECISION_MANUAL_REVIEW


def test_family_unstable_presence_no_text_vote():
    texts = {k: "需要" for k in EIGHT}
    texts["kimi_1"] = "需要"
    texts["kimi_2"] = ""
    result = classify_sample(_sample(texts, quality=_clean_quality()), _cfg())
    # May hit presence_conflict first (empty + nonempty across all routes)
    assert result.type in {
        TYPE_SPEECH_PRESENCE_DISAGREEMENT,
        "family_unstable",
        TYPE_SEMANTIC_RISK,
    }
    assert result.family_status.get("kimi") in {"unstable_presence", "incomplete"}


def test_ambiguous_or_split_vote_not_pseudo():
    texts = {
        "kimi_1": "我需要贷款",
        "kimi_2": "我需要贷款",
        "glm_1": "我需要贷款",
        "glm_2": "我需要贷款",
        "sensevoice_1": "我不需要贷款",
        "sensevoice_2": "我不需要贷款",
        "qwen_1": "我不需要贷款",
        "qwen_2": "我不需要贷款",
    }
    result = classify_sample(_sample(texts, quality=_clean_quality(), duration=4.0), _cfg())
    assert result.type != TYPE_PSEUDO_HIGH
    assert result.decision == DECISION_MANUAL_REVIEW


def test_high_agreement_but_noisy_dnsmos_not_auto():
    sample = _sample("我不需要", quality={
        **_clean_quality(),
        "noise_band": NOISE_BAND_NOISY,
        "noise_risk": True,
    }, duration=1.5)
    result = classify_sample(sample, _cfg())
    assert result.type == TYPE_AUDIO_QUALITY_RISK
    assert result.decision == DECISION_MANUAL_REVIEW


def test_noisy_switch_changes_auto_receive_decision():
    clean = classify_sample(_sample("我不需要", quality=_clean_quality(), duration=1.5), _cfg())
    noisy = classify_sample(
        _sample(
            "我不需要",
            quality={**_clean_quality(), "noise_band": NOISE_BAND_NOISY, "noise_risk": True},
            duration=1.5,
        ),
        _cfg(),
    )
    assert clean.type == TYPE_PSEUDO_HIGH
    assert noisy.type == TYPE_AUDIO_QUALITY_RISK
    assert clean.decision == DECISION_AUDIT_PENDING
    assert noisy.decision == DECISION_MANUAL_REVIEW


def test_p0_not_overridden_by_high_similarity():
    texts = {k: "需要" for k in EIGHT}
    texts["glm_1"] = "不需要"
    texts["glm_2"] = "不需要"
    result = classify_sample(_sample(texts, quality=_clean_quality()), _cfg())
    assert result.review_priority == PRIORITY_P0
    assert result.type == TYPE_SEMANTIC_RISK


def test_classify_operator_consensus_v3(tmp_path: Path):
    from audio_engine.core.manifest import Manifest

    sample = _sample("我不需要", quality=_clean_quality(), duration=1.5)
    op = OperatorRegistry.get("quality.classify")
    out = op.run(
        [sample],
        OperatorConfig(
            params={
                "config_path": str(SELECTION_CFG),
                "policy_version": "selection_zh_asr_v3_0",
            },
            cache_dir=tmp_path / "cache",
            force=True,
        ),
    )
    assert out[0].labels["type"] == TYPE_PSEUDO_HIGH
    assert out[0].labels["decision"] == DECISION_AUDIT_PENDING
    assert out[0].labels["is_human_verified"] is False
    assert out[0].labels["gold_text"] is None


def test_hardcase_for_two_two_without_semantic():
    texts = {
        "kimi_1": "今天开会",
        "kimi_2": "今天开会",
        "glm_1": "今天开会",
        "glm_2": "今天开会",
        "sensevoice_1": "明天出差",
        "sensevoice_2": "明天出差",
        "qwen_1": "明天出差",
        "qwen_2": "明天出差",
    }
    result = classify_sample(_sample(texts, quality=_clean_quality(), duration=4.0), _cfg())
    assert result.type in {TYPE_HARDCASE, "pseudo_medium"}
    assert result.type != TYPE_PSEUDO_HIGH
