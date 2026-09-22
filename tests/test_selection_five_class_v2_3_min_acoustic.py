"""034 selection_five_class_v2_3_min_acoustic acceptance / contract tests."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3 import SelectionV3Config, classify_sample
from audio_engine.core.selection_v3.types import (
    CATEGORY_ENVIRONMENT_NOISE,
    CATEGORY_GOLD_CANDIDATE,
    CATEGORY_HARDCASE,
    CATEGORY_SEMANTIC_RISK,
    ENERGY_STATE_AUDIBLE,
    ENERGY_STATE_INAUDIBLE,
    ENERGY_STATE_TOO_SHORT,
    NOISE_KIND_AUDIO_TOO_SHORT,
    NOISE_KIND_BACKGROUND,
    NOISE_KIND_HUMAN_NOISE,
    NOISE_KIND_SILENCE,
    RULE_VERSION_FIVE_CLASS_V2,
    RULE_VERSION_FIVE_CLASS_V2_2,
    RULE_VERSION_FIVE_CLASS_V2_3,
    is_five_class_v2_2_rule,
    is_five_class_v2_3_rule,
    is_five_class_v2_rule,
)
from audio_engine.core.source_naming import apply_source_name_to_single_pipeline
from audio_engine.operators.quality.efficientat_candidates import needs_efficientat_v2_3

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "selection" / "zh_asr_five_class_v2_3_min_acoustic.yaml"
VM_PATH = ROOT / "configs" / "selection" / "voicemail_patterns_v1.yaml"
PIPELINE = ROOT / "pipelines" / "classify_dataset_five_class_v2_3_min_acoustic.yaml"

KEYS = [
    "glm_1",
    "glm_2",
    "sensevoice_1",
    "sensevoice_2",
    "qwen_1",
    "qwen_2",
]


def _vm() -> re.Pattern[str]:
    raw = yaml.safe_load(VM_PATH.read_text(encoding="utf-8")) or {}
    patterns = [f"(?:{p})" for p in (raw.get("patterns") or []) if str(p).strip()]
    return re.compile("|".join(patterns), re.IGNORECASE)


def _cfg(**overrides) -> SelectionV3Config:
    params = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8")) or {}
    params.update(overrides)
    ct = dict(params.get("classify_text") or {})
    ct.setdefault("echo_missing", "echo_list_missing")
    echo = {
        "qwen": {
            "extra_exact": ["这是一段测试提示词正文"],
            "asr_config": str(ROOT / "configs/asr/qwen_asr.yaml"),
            "blank_exact": str(ROOT / "configs/normalization/blank_exact_qwen_v1.yaml"),
        },
        "glm": {"extra_exact": [], "asr_config": str(ROOT / "configs/asr/glm.yaml")},
        "sensevoice": {
            "extra_exact": [],
            "asr_config": str(ROOT / "configs/asr/sensevoice.yaml"),
        },
    }
    ct["echo"] = echo
    params["classify_text"] = ct
    return SelectionV3Config.from_params(params)


def _energy(
    *,
    state: str = ENERGY_STATE_AUDIBLE,
    duration_ms: float = 3000.0,
    rms_dbfs: float = -20.0,
    peak_dbfs: float = -10.0,
    non_silent_ratio: float = 0.5,
) -> dict:
    return {
        "duration_ms": duration_ms,
        "rms_dbfs": rms_dbfs,
        "peak_dbfs": peak_dbfs,
        "non_silent_ratio": non_silent_ratio,
        "energy_state": state,
        "energy_policy_version": "audio_energy_v1",
    }


def _acoustic(
    *,
    vad: str = "absent",
    event: str = "non_speech_environment",
    acoustic: str | None = None,
    eat_status: str = "success",
    speech_prob: float | None = 0.05,
    env_prob: float | None = 0.8,
    babble_prob: float | None = 0.05,
) -> dict:
    payload = {
        "vad_status": "failed" if vad == "failed" else "success",
        "vad_speech_state": vad,
        "speech_ratio": 0.5 if vad == "present" else 0.01,
        "speech_duration_ms": 500 if vad == "present" else 20,
        "longest_speech_ms": 500 if vad == "present" else 20,
        "speech_segment_count": 1 if vad == "present" else 0,
        "vad_config_version": "silero_vad_v1",
        "efficientat_status": eat_status,
        "speech_prob": speech_prob,
        "conversation_prob": 0.05,
        "speech_babble_prob": babble_prob,
        "environment_noise_prob": env_prob,
        "music_prob": 0.1,
        "vehicle_prob": 0.05,
        "mechanical_prob": 0.05,
        "audio_event_state": event,
        "efficientat_config_version": "efficientat_v1",
    }
    if acoustic is not None:
        payload["acoustic_state"] = acoustic
    return payload


def _sample(
    texts: dict[str, str] | str,
    *,
    sample_id: str = "utt-1",
    quality: dict | None = None,
    labels: dict | None = None,
    failed: set[str] | None = None,
    duration: float = 3.0,
    sha256: str = "abc123",
) -> Sample:
    if isinstance(texts, str):
        mapping = {k: texts for k in KEYS}
    else:
        mapping = {k: texts.get(k, "") for k in KEYS}
    transcripts = {}
    for key in KEYS:
        if failed and key in failed:
            transcripts[key] = {"text": "", "status": "failed"}
        else:
            transcripts[key] = {"text": mapping[key], "status": "success"}
    return Sample(
        id=sample_id,
        source_path=f"/data/{sample_id}.wav",
        duration=duration,
        sha256=sha256,
        transcripts=transcripts,
        quality=dict(quality or {**_energy(), **_acoustic()}),
        labels={**(labels or {}), "original_audio_sha256": sha256},
        audio={"resampled_16k": f"/data/{sample_id}.wav"},
    )


def _all_empty() -> dict[str, str]:
    return {k: "" for k in KEYS}


def _family_text(family_map: dict[str, str]) -> dict[str, str]:
    out = {k: "" for k in KEYS}
    if "sensevoice" in family_map:
        out["sensevoice_1"] = family_map["sensevoice"]
        out["sensevoice_2"] = family_map["sensevoice"]
    if "glm" in family_map:
        out["glm_1"] = family_map["glm"]
        out["glm_2"] = family_map["glm"]
    if "qwen" in family_map:
        out["qwen_1"] = family_map["qwen"]
        out["qwen_2"] = family_map["qwen"]
    return out


def _classify(texts, *, quality=None, **kwargs):
    sample_kwargs = {
        k: kwargs.pop(k)
        for k in ("sample_id", "labels", "failed", "duration", "sha256")
        if k in kwargs
    }
    sample = kwargs.pop("sample", None) or _sample(
        texts, quality=quality, **sample_kwargs
    )
    cfg = kwargs.pop("cfg", None) or _cfg()
    assert not kwargs
    return classify_sample(sample, cfg, voicemail_pattern=_vm())


def test_rule_version_predicates_do_not_collide():
    assert is_five_class_v2_3_rule(RULE_VERSION_FIVE_CLASS_V2_3)
    assert not is_five_class_v2_2_rule(RULE_VERSION_FIVE_CLASS_V2_3)
    assert not is_five_class_v2_rule(RULE_VERSION_FIVE_CLASS_V2_3)
    assert is_five_class_v2_2_rule(RULE_VERSION_FIVE_CLASS_V2_2)
    assert is_five_class_v2_rule(RULE_VERSION_FIVE_CLASS_V2)


def test_config_parses_v2_3():
    cfg = _cfg()
    assert cfg.rule_version == RULE_VERSION_FIVE_CLASS_V2_3
    assert cfg.dnsmos_decision_enabled is False
    assert cfg.silero_vad_config_path.endswith("silero_vad_v1.yaml")
    assert cfg.efficientat_config_path.endswith("efficientat_v1.yaml")
    assert cfg.acoustic_evidence_config_path.endswith("acoustic_evidence_v1.yaml")


def test_pipeline_output_isolated_from_v2_2():
    raw = yaml.safe_load(PIPELINE.read_text(encoding="utf-8"))
    out = str(raw["output"]["manifest"])
    assert "v2_3_min_acoustic" in out
    assert "v2_2_auto_noise" not in out

    class _Step:
        def __init__(self, operator: str):
            self.operator = operator
            self.params: dict = {}

    v23 = apply_source_name_to_single_pipeline(
        pipeline_name="classify_dataset_five_class_v2_3_min_acoustic",
        steps=[_Step("quality.classify")],
        source_name="0914-mixed-30000",
    )
    v22 = apply_source_name_to_single_pipeline(
        pipeline_name="classify_dataset_five_class_v2_2_auto_noise",
        steps=[_Step("quality.classify")],
        source_name="0914-mixed-30000",
    )
    assert "classified_five_class_v2_3_min_acoustic_" in str(v23["output_manifest"])
    assert "classified_five_class_v2_2_auto_noise_" in str(v22["output_manifest"])
    assert v23["output_manifest"] != v22["output_manifest"]


def test_matrix_all_empty_silence():
    q = {
        **_energy(state=ENERGY_STATE_INAUDIBLE),
        **_acoustic(vad="absent", event="unknown", acoustic="silence", eat_status="skipped"),
    }
    result = _classify(_all_empty(), quality=q)
    assert result.category == CATEGORY_ENVIRONMENT_NOISE
    assert result.noise_kind == NOISE_KIND_SILENCE
    assert result.decision_evidence["text_evidence_state"] == "all_empty"
    assert result.decision_evidence["acoustic_state"] == "silence"


def test_matrix_all_empty_non_speech_background():
    q = {
        **_energy(state=ENERGY_STATE_AUDIBLE),
        **_acoustic(vad="absent", event="non_speech_environment", acoustic="non_speech_noise"),
    }
    result = _classify(_all_empty(), quality=q)
    assert result.category == CATEGORY_ENVIRONMENT_NOISE
    assert result.noise_kind == NOISE_KIND_BACKGROUND
    assert "dnsmos" not in " ".join(result.evidence_sources or []).lower()


def test_matrix_all_empty_speech_supported_hardcase():
    q = {
        **_energy(state=ENERGY_STATE_AUDIBLE),
        **_acoustic(
            vad="present",
            event="speech_like",
            acoustic="speech_supported",
            speech_prob=0.8,
            env_prob=0.1,
        ),
    }
    result = _classify(_all_empty(), quality=q)
    assert result.category == CATEGORY_HARDCASE
    assert result.hardcase_reason == "empty_asr_but_speech_present"


def test_matrix_consensus_strong_veto_hallucination():
    texts = _family_text({"qwen": "你好世界", "glm": "你好世界", "sensevoice": ""})
    q = {
        **_energy(state=ENERGY_STATE_AUDIBLE),
        **_acoustic(vad="absent", event="non_speech_environment", acoustic="non_speech_noise"),
    }
    result = _classify(texts, quality=q)
    assert result.category == CATEGORY_ENVIRONMENT_NOISE
    assert result.hallucination_state == "multi_family_consensus"
    assert result.decision_evidence["decision"] == CATEGORY_ENVIRONMENT_NOISE


def test_matrix_consensus_speech_supported_gold():
    texts = _family_text({"qwen": "你好世界", "glm": "你好世界", "sensevoice": ""})
    q = {
        **_energy(state=ENERGY_STATE_AUDIBLE),
        **_acoustic(
            vad="present",
            event="speech_like",
            acoustic="speech_supported",
            speech_prob=0.9,
            env_prob=0.05,
        ),
    }
    result = _classify(texts, quality=q)
    assert result.category == CATEGORY_GOLD_CANDIDATE
    assert result.classification_confidence == "high"
    assert result.decision_evidence["text_evidence_state"] == "consensus_text"


def test_matrix_single_text_human_noise_critical_short_protected():
    texts = _family_text({"qwen": "不要", "glm": "", "sensevoice": ""})
    q = {
        **_energy(state=ENERGY_STATE_AUDIBLE),
        **_acoustic(
            vad="present",
            event="human_background",
            acoustic="human_noise_suspected",
            babble_prob=0.7,
            speech_prob=0.4,
        ),
    }
    result = _classify(texts, quality=q)
    assert result.category == CATEGORY_HARDCASE
    assert result.hardcase_reason == "critical_short_response_unsupported"
    assert result.noise_kind in {None, ""}


def test_matrix_single_text_human_noise_non_critical():
    texts = _family_text({"qwen": "今天天气不错啊", "glm": "", "sensevoice": ""})
    q = {
        **_energy(state=ENERGY_STATE_AUDIBLE),
        **_acoustic(
            vad="present",
            event="human_background",
            acoustic="human_noise_suspected",
            babble_prob=0.7,
            speech_prob=0.4,
        ),
    }
    result = _classify(texts, quality=q)
    assert result.category == CATEGORY_ENVIRONMENT_NOISE
    assert result.noise_kind == NOISE_KIND_HUMAN_NOISE
    assert result.needs_review is True


def test_too_short_no_longer_audio_too_short_noise():
    texts = _family_text({"qwen": "嗯", "glm": "", "sensevoice": ""})
    q = {
        **_energy(state=ENERGY_STATE_TOO_SHORT, duration_ms=200.0),
        **_acoustic(
            vad="present",
            event="speech_like",
            acoustic="speech_supported",
            speech_prob=0.8,
            env_prob=0.05,
        ),
    }
    result = _classify(texts, quality=q, duration=0.2)
    assert result.noise_kind != NOISE_KIND_AUDIO_TOO_SHORT
    assert "short_audio" in (result.quality_tags or [])
    # single_text + speech_supported → hardcase, not env/audio_too_short
    assert result.category in {CATEGORY_HARDCASE, CATEGORY_ENVIRONMENT_NOISE}
    if result.category == CATEGORY_ENVIRONMENT_NOISE:
        assert result.noise_kind != NOISE_KIND_AUDIO_TOO_SHORT


def test_vad_failed_does_not_forge_absent():
    q = {
        **_energy(state=ENERGY_STATE_AUDIBLE),
        **_acoustic(vad="failed", event="failed", acoustic="failed", eat_status="skipped"),
    }
    result = _classify(_all_empty(), quality=q)
    assert result.category == CATEGORY_HARDCASE
    assert result.hardcase_reason in {"vad_failed", "energy_failed", "efficientat_failed"}
    assert result.noise_kind in {None, ""}


def test_efficientat_failed_on_candidate_uncertain_or_hardcase():
    # Candidate-like all_empty + audible with eat failed → do not forge background.
    q = {
        **_energy(state=ENERGY_STATE_AUDIBLE),
        **_acoustic(
            vad="absent",
            event="failed",
            acoustic="uncertain",
            eat_status="failed",
            speech_prob=None,
            env_prob=None,
        ),
    }
    result = _classify(_all_empty(), quality=q)
    assert result.category == CATEGORY_HARDCASE
    assert result.noise_kind != NOISE_KIND_BACKGROUND


def test_candidate_trigger_all_empty_not_inaudible():
    sample = _sample(
        _all_empty(),
        quality={
            **_energy(state=ENERGY_STATE_AUDIBLE),
            **_acoustic(vad="absent", eat_status="skipped"),
        },
    )
    need = needs_efficientat_v2_3(sample, _cfg())
    assert need["required"] is True
    assert "all_empty_not_inaudible" in need["reasons"]


def test_candidate_not_triggered_all_empty_inaudible():
    sample = _sample(
        _all_empty(),
        quality={
            **_energy(state=ENERGY_STATE_INAUDIBLE),
            **_acoustic(vad="absent", eat_status="skipped"),
        },
    )
    need = needs_efficientat_v2_3(sample, _cfg())
    assert need["required"] is False


def test_candidate_trigger_consensus_vad_not_present():
    texts = _family_text({"qwen": "你好", "glm": "你好", "sensevoice": ""})
    sample = _sample(
        texts,
        quality={
            **_energy(state=ENERGY_STATE_AUDIBLE),
            **_acoustic(vad="absent", eat_status="skipped"),
        },
    )
    need = needs_efficientat_v2_3(sample, _cfg())
    assert need["required"] is True
    assert "consensus_text_vad_not_present" in need["reasons"]


def test_dnsmos_not_used_as_category_evidence():
    texts = _family_text({"qwen": "你好世界", "glm": "你好世界", "sensevoice": ""})
    q = {
        **_energy(state=ENERGY_STATE_AUDIBLE),
        **_acoustic(
            vad="present",
            event="speech_like",
            acoustic="speech_supported",
            speech_prob=0.9,
            env_prob=0.05,
        ),
        "dnsmos_status": "success",
        "dnsmos_sig": 1.0,
        "dnsmos_bak": 1.0,
        "dnsmos_ovrl": 1.0,
    }
    result = _classify(texts, quality=q)
    assert result.category == CATEGORY_GOLD_CANDIDATE
    # May attach perceptual quality tags, but not change class via DNSMOS.
    assert result.category != CATEGORY_ENVIRONMENT_NOISE
    assert "dnsmos_decision_v2_2" not in (result.evidence_sources or [])


def test_strict_conflict_with_speech_is_semantic_risk():
    texts = _family_text({"qwen": "要", "glm": "不要", "sensevoice": ""})
    q = {
        **_energy(state=ENERGY_STATE_AUDIBLE),
        **_acoustic(
            vad="present",
            event="speech_like",
            acoustic="speech_supported",
            speech_prob=0.9,
            env_prob=0.05,
        ),
    }
    result = _classify(texts, quality=q)
    assert result.category == CATEGORY_SEMANTIC_RISK


def test_labels_contain_decision_evidence():
    q = {
        **_energy(state=ENERGY_STATE_INAUDIBLE),
        **_acoustic(vad="absent", event="unknown", acoustic="silence", eat_status="skipped"),
    }
    result = _classify(_all_empty(), quality=q)
    labels = result.to_labels("selection_five_class_v2_3_min_acoustic")
    assert labels["rule_version"] == RULE_VERSION_FIVE_CLASS_V2_3
    assert isinstance(labels["decision_evidence"], dict)
    assert labels["decision_evidence"]["acoustic_state"] == "silence"
    assert labels["text_evidence_state"] == "all_empty"
