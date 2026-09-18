"""029 selection_five_class_v2_2_auto_noise acceptance cases (§11)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from audio_engine.core.manifest import Manifest
from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3 import SelectionV3Config, classify_sample
from audio_engine.core.selection_v3.dnsmos_decision import (
    DNSMOS_NOISE_CLEAN,
    DNSMOS_NOISE_MODERATE,
    DNSMOS_NOISE_NOISY,
    DNSMOS_NOISE_UNAVAILABLE,
    derive_dnsmos_decision,
    load_dnsmos_decision_config,
)
from audio_engine.core.selection_v3.types import (
    CATEGORY_ENVIRONMENT_NOISE,
    CATEGORY_GOLD_CANDIDATE,
    CATEGORY_HARDCASE,
    CATEGORY_SEMANTIC_RISK,
    CATEGORY_VOICEMAIL,
    ENERGY_STATE_AUDIBLE,
    ENERGY_STATE_BORDERLINE,
    ENERGY_STATE_INAUDIBLE,
    ENERGY_STATE_TOO_SHORT,
    NOISE_KIND_AUDIO_TOO_SHORT,
    NOISE_KIND_BACKGROUND,
    NOISE_KIND_HUMAN_NOISE,
    NOISE_KIND_SILENCE,
    RULE_VERSION_FIVE_CLASS,
    RULE_VERSION_FIVE_CLASS_V2,
    RULE_VERSION_FIVE_CLASS_V2_2,
    is_five_class_rule,
    is_five_class_v2_2_rule,
    is_five_class_v2_rule,
)
from audio_engine.core.source_naming import apply_source_name_to_single_pipeline
from audio_engine.operators.quality.dnsmos_v2_2_candidates import (
    merge_dnsmos_sidecar_by_hash,
    needs_dnsmos_v2_2,
)

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "selection" / "zh_asr_five_class_v2_2_auto_noise.yaml"
V2_CFG_PATH = ROOT / "configs" / "selection" / "zh_asr_five_class_v2_auto_noise.yaml"
DECISION_PATH = ROOT / "configs" / "quality" / "dnsmos_decision_v2_2.yaml"
VM_PATH = ROOT / "configs" / "selection" / "voicemail_patterns_v1.yaml"

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


def _cfg(path: Path = CFG_PATH, **overrides) -> SelectionV3Config:
    params = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
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


def _dnsmos(
    *,
    status: str = "success",
    sig: float | None = 3.0,
    bak: float | None = 2.5,
    ovrl: float | None = 2.5,
) -> dict:
    return {
        "dnsmos_status": status,
        "dnsmos_sig": sig,
        "dnsmos_bak": bak,
        "dnsmos_ovrl": ovrl,
        "dnsmos_model_digest": "test-digest",
        "dnsmos_preprocess_version": "p835_official_nonpersonalized_v2",
    }


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
        quality=dict(quality or _energy()),
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


def test_config_parses_dnsmos_decision():
    cfg = _cfg()
    assert cfg.rule_version == RULE_VERSION_FIVE_CLASS_V2_2
    assert cfg.dnsmos_decision_enabled is True
    assert cfg.dnsmos_decision_policy_version == "dnsmos_decision_v2_2"
    assert cfg.dnsmos_clean_bak >= cfg.dnsmos_noisy_bak
    assert cfg.dnsmos_strong_sig > cfg.dnsmos_weak_sig
    decision = load_dnsmos_decision_config(DECISION_PATH)
    assert decision.policy_version == "dnsmos_decision_v2_2"


def test_01_all_empty_audible_dnsmos_noisy_background_high():
    q = {**_energy(state=ENERGY_STATE_AUDIBLE), **_dnsmos(bak=2.5, ovrl=2.5, sig=3.0)}
    result = _classify(_all_empty(), quality=q)
    assert result.category == CATEGORY_ENVIRONMENT_NOISE
    assert result.noise_kind == NOISE_KIND_BACKGROUND
    assert result.classification_confidence == "high"
    assert result.classification_source == "auto_family_energy_dnsmos"
    assert result.dnsmos_noise_state == DNSMOS_NOISE_NOISY
    assert "dnsmos_decision_v2_2" in (result.evidence_sources or [])


def test_02_all_empty_audible_dnsmos_unavailable_v2_fallback():
    q = {
        **_energy(state=ENERGY_STATE_AUDIBLE),
        **_dnsmos(status="failed", sig=None, bak=None, ovrl=None),
    }
    result = _classify(_all_empty(), quality=q)
    assert result.category == CATEGORY_ENVIRONMENT_NOISE
    assert result.noise_kind == NOISE_KIND_BACKGROUND
    assert result.classification_confidence == "medium"
    assert result.classification_source == "auto_family_energy"
    assert result.v2_fallback is True
    assert result.dnsmos_noise_state == DNSMOS_NOISE_UNAVAILABLE


def test_03_all_empty_audible_clean_strong_hardcase():
    q = {**_energy(state=ENERGY_STATE_AUDIBLE), **_dnsmos(bak=3.6, ovrl=3.4, sig=3.8)}
    result = _classify(_all_empty(), quality=q)
    assert result.category == CATEGORY_HARDCASE
    assert result.hardcase_reason == "empty_asr_but_clean_strong_speech"
    assert result.dnsmos_noise_state == DNSMOS_NOISE_CLEAN


def test_04_all_empty_audible_moderate_background_medium():
    q = {**_energy(state=ENERGY_STATE_AUDIBLE), **_dnsmos(bak=3.2, ovrl=3.0, sig=3.0)}
    result = _classify(_all_empty(), quality=q)
    assert result.category == CATEGORY_ENVIRONMENT_NOISE
    assert result.noise_kind == NOISE_KIND_BACKGROUND
    assert result.classification_confidence == "medium"
    assert result.dnsmos_noise_state == DNSMOS_NOISE_MODERATE


def test_05_borderline_dnsmos_noisy_background_medium():
    q = {
        **_energy(state=ENERGY_STATE_BORDERLINE, rms_dbfs=-50.0, peak_dbfs=-40.0),
        **_dnsmos(bak=2.5, ovrl=2.5),
    }
    result = _classify(_all_empty(), quality=q)
    assert result.category == CATEGORY_ENVIRONMENT_NOISE
    assert result.noise_kind == NOISE_KIND_BACKGROUND
    assert result.classification_confidence == "medium"
    assert result.borderline_resolved_by_dnsmos is True


def test_06_borderline_dnsmos_unavailable_hardcase():
    q = {
        **_energy(state=ENERGY_STATE_BORDERLINE),
        **_dnsmos(status="failed", sig=None, bak=None, ovrl=None),
    }
    result = _classify(_all_empty(), quality=q)
    assert result.category == CATEGORY_HARDCASE
    assert "borderline" in (result.hardcase_reason or result.reason or "")


def test_07_human_noise_audible_dnsmos_noisy_high():
    texts = _family_text(
        {"glm": "", "sensevoice": "", "qwen": "今天天气不错我们出去走走吧"}
    )
    q = {**_energy(state=ENERGY_STATE_AUDIBLE), **_dnsmos(bak=2.5, ovrl=2.5)}
    result = _classify(texts, quality=q)
    assert result.category == CATEGORY_ENVIRONMENT_NOISE
    assert result.noise_kind == NOISE_KIND_HUMAN_NOISE
    assert result.classification_confidence == "high"


def test_08_human_noise_audible_dnsmos_clean_medium():
    texts = _family_text(
        {"glm": "", "sensevoice": "", "qwen": "今天天气不错我们出去走走吧"}
    )
    q = {**_energy(state=ENERGY_STATE_AUDIBLE), **_dnsmos(bak=3.6, ovrl=3.4, sig=3.0)}
    result = _classify(texts, quality=q)
    assert result.category == CATEGORY_ENVIRONMENT_NOISE
    assert result.noise_kind == NOISE_KIND_HUMAN_NOISE
    assert result.classification_confidence == "medium"


def test_09_human_noise_borderline_dnsmos_noisy_medium():
    texts = _family_text(
        {"glm": "", "sensevoice": "", "qwen": "今天天气不错我们出去走走吧"}
    )
    q = {
        **_energy(state=ENERGY_STATE_BORDERLINE),
        **_dnsmos(bak=2.5, ovrl=2.5),
    }
    result = _classify(texts, quality=q)
    assert result.category == CATEGORY_ENVIRONMENT_NOISE
    assert result.noise_kind == NOISE_KIND_HUMAN_NOISE
    assert result.classification_confidence == "medium"
    assert result.borderline_resolved_by_dnsmos is True


def test_10_critical_short_still_hardcase():
    texts = _family_text({"glm": "", "sensevoice": "", "qwen": "不需要"})
    q = {**_energy(state=ENERGY_STATE_AUDIBLE), **_dnsmos(bak=2.5, ovrl=2.5)}
    result = _classify(texts, quality=q)
    assert result.category == CATEGORY_HARDCASE
    assert "critical_short" in (result.hardcase_reason or result.reason)


def test_11_gold_keeps_category_with_background_noisy_tag():
    texts = _family_text(
        {
            "glm": "不需要贷款服务",
            "sensevoice": "不需要贷款服务",
            "qwen": "",
        }
    )
    q = {**_energy(state=ENERGY_STATE_AUDIBLE), **_dnsmos(bak=2.5, ovrl=2.5)}
    result = _classify(texts, quality=q)
    assert result.category == CATEGORY_GOLD_CANDIDATE
    assert result.quality_tag == "background_noisy"
    assert result.background_quality_risk is True
    assert result.classification_confidence == "high"


def test_12_voicemail_and_semantic_not_overridden_by_dnsmos():
    texts = _family_text(
        {
            "glm": "您好，您拨打的电话正在通话中",
            "sensevoice": "不需要",
            "qwen": "需要这个服务",
        }
    )
    q = {**_energy(state=ENERGY_STATE_AUDIBLE), **_dnsmos(bak=1.0, ovrl=1.0)}
    result = _classify(texts, quality=q)
    assert result.category == CATEGORY_VOICEMAIL

    texts2 = _family_text(
        {
            "glm": "我需要这个服务",
            "sensevoice": "我不需要这个服务",
            "qwen": "",
        }
    )
    result2 = _classify(texts2, quality=q)
    assert result2.category == CATEGORY_SEMANTIC_RISK


def test_13_too_short_and_silence_ignore_dnsmos():
    q_short = {
        **_energy(state=ENERGY_STATE_TOO_SHORT, duration_ms=100.0),
        **_dnsmos(bak=1.0, ovrl=1.0, sig=4.0),
    }
    r1 = _classify(_all_empty(), quality=q_short, duration=0.1)
    assert r1.category == CATEGORY_ENVIRONMENT_NOISE
    assert r1.noise_kind == NOISE_KIND_AUDIO_TOO_SHORT
    assert r1.classification_source == "auto_family_energy"

    q_sil = {
        **_energy(
            state=ENERGY_STATE_INAUDIBLE,
            rms_dbfs=-70.0,
            peak_dbfs=-65.0,
            non_silent_ratio=0.0,
        ),
        **_dnsmos(bak=1.0, ovrl=1.0, sig=4.0),
    }
    r2 = _classify(_all_empty(), quality=q_sil)
    assert r2.category == CATEGORY_ENVIRONMENT_NOISE
    assert r2.noise_kind == NOISE_KIND_SILENCE


def test_14_failed_dnsmos_not_forged_noisy():
    decision = load_dnsmos_decision_config(DECISION_PATH)
    evidence = derive_dnsmos_decision(
        {"dnsmos_status": "failed", "dnsmos_sig": None, "dnsmos_bak": None, "dnsmos_ovrl": None},
        decision,
    )
    assert evidence.noise_state == DNSMOS_NOISE_UNAVAILABLE
    evidence2 = derive_dnsmos_decision(
        {"dnsmos_status": "unsupported", "dnsmos_sig": 1.0, "dnsmos_bak": 1.0, "dnsmos_ovrl": 1.0},
        decision,
    )
    assert evidence2.noise_state == DNSMOS_NOISE_UNAVAILABLE


def test_15_sidecar_hash_mismatch_rejected(tmp_path: Path):
    base = [_sample(_all_empty(), sample_id="a", sha256="hash-a")]
    sidecar_sample = _sample(
        _all_empty(),
        sample_id="a",
        sha256="hash-wrong",
        quality={**_energy(), **_dnsmos()},
    )
    path = tmp_path / "sidecar.parquet"
    Manifest([sidecar_sample]).save(path)
    with pytest.raises(ValueError, match="hash mismatch"):
        merge_dnsmos_sidecar_by_hash(base, path)


def test_16_deterministic_rerun():
    texts = _family_text(
        {"glm": "", "sensevoice": "", "qwen": "今天天气不错我们出去走走吧"}
    )
    q = {**_energy(state=ENERGY_STATE_AUDIBLE), **_dnsmos(bak=2.5, ovrl=2.5)}
    a = _classify(texts, sample_id="det-1", quality=q)
    b = _classify(texts, sample_id="det-1", quality=q)
    assert a.category == b.category
    assert a.noise_kind == b.noise_kind
    assert a.classification_confidence == b.classification_confidence
    assert a.decision_trace == b.decision_trace


def test_17_v1_v2_v22_product_paths_isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    manifests = tmp_path / "datasets" / "manifests"
    manifests.mkdir(parents=True)
    Manifest(
        [Sample(id="a", source_path="/tmp/a.wav", duration=1.0)]
    ).save(manifests / "prepared_asr_v3_0914-mixed-30000.parquet")

    class _Step:
        def __init__(self, operator: str):
            self.operator = operator
            self.params: dict = {}

    v1 = apply_source_name_to_single_pipeline(
        pipeline_name="classify_dataset_five_class_v1",
        steps=[_Step("quality.classify")],
        source_name="0914-mixed-30000",
    )
    v2 = apply_source_name_to_single_pipeline(
        pipeline_name="classify_dataset_five_class_v2_auto_noise",
        steps=[_Step("quality.classify")],
        source_name="0914-mixed-30000",
    )
    v22 = apply_source_name_to_single_pipeline(
        pipeline_name="classify_dataset_five_class_v2_2_auto_noise",
        steps=[_Step("quality.classify")],
        source_name="0914-mixed-30000",
    )
    assert "classified_five_class_v1_" in str(v1["output_manifest"])
    assert "classified_five_class_v2_auto_noise_" in str(v2["output_manifest"])
    assert "classified_five_class_v2_2_auto_noise_" in str(v22["output_manifest"])
    assert len({v1["output_manifest"], v2["output_manifest"], v22["output_manifest"]}) == 3
    assert is_five_class_rule(RULE_VERSION_FIVE_CLASS)
    assert is_five_class_v2_rule(RULE_VERSION_FIVE_CLASS_V2)
    assert is_five_class_v2_2_rule(RULE_VERSION_FIVE_CLASS_V2_2)
    assert not is_five_class_v2_rule(RULE_VERSION_FIVE_CLASS_V2_2)
    assert not is_five_class_v2_2_rule(RULE_VERSION_FIVE_CLASS_V2)


def test_18_v2_auto_noise_unchanged_on_all_empty_audible():
    """v2 must still auto-classify background without DNSMOS joint semantics."""
    params = yaml.safe_load(V2_CFG_PATH.read_text(encoding="utf-8")) or {}
    ct = dict(params.get("classify_text") or {})
    ct["echo_missing"] = "echo_list_missing"
    ct["echo"] = {
        "qwen": {"extra_exact": [], "asr_config": None, "blank_exact": None},
        "glm": {"extra_exact": [], "asr_config": None},
        "sensevoice": {"extra_exact": [], "asr_config": None},
    }
    params["classify_text"] = ct
    cfg = SelectionV3Config.from_params(params)
    sample = _sample(_all_empty(), quality=_energy(state=ENERGY_STATE_AUDIBLE))
    result = classify_sample(sample, cfg, voicemail_pattern=_vm())
    assert result.category == CATEGORY_ENVIRONMENT_NOISE
    assert result.noise_kind == NOISE_KIND_BACKGROUND
    assert result.classification_confidence == "high"
    assert result.classification_source == "auto_family_energy"
    assert result.rule_version == RULE_VERSION_FIVE_CLASS_V2


def test_19_export_summary_fields_present():
    q = {**_energy(state=ENERGY_STATE_AUDIBLE), **_dnsmos(bak=2.5, ovrl=2.5)}
    result = _classify(_all_empty(), quality=q)
    labels = result.to_labels(RULE_VERSION_FIVE_CLASS_V2_2)
    for key in (
        "dnsmos_noise_state",
        "dnsmos_speech_state",
        "dnsmos_decision_policy_version",
        "evidence_sources",
        "decision_trace",
        "v2_fallback",
    ):
        assert key in labels


def test_20_disabled_dnsmos_falls_back_to_v2_auditable():
    cfg = _cfg()
    cfg.dnsmos_decision_enabled = False
    sample = _sample(_all_empty(), quality=_energy(state=ENERGY_STATE_AUDIBLE))
    result = classify_sample(sample, cfg, voicemail_pattern=_vm())
    assert result.category == CATEGORY_ENVIRONMENT_NOISE
    assert result.noise_kind == NOISE_KIND_BACKGROUND
    assert result.decision_trace.get("dnsmos_decision") == "disabled"
    assert result.decision_trace.get("fallback_policy") == RULE_VERSION_FIVE_CLASS_V2
    assert result.v2_fallback is True


def test_candidate_trigger_covers_empty_and_human_shapes():
    cfg = _cfg()
    empty = _sample(_all_empty(), quality=_energy(state=ENERGY_STATE_AUDIBLE))
    assert needs_dnsmos_v2_2(empty, cfg)["required"] is True
    human = _sample(
        _family_text({"glm": "", "sensevoice": "", "qwen": "外面雨下得很大"}),
        quality=_energy(state=ENERGY_STATE_BORDERLINE),
    )
    assert needs_dnsmos_v2_2(human, cfg)["required"] is True
    gold = _sample(
        _family_text(
            {
                "glm": "不需要贷款服务",
                "sensevoice": "不需要贷款服务",
                "qwen": "不需要贷款服务",
            }
        ),
        quality=_energy(state=ENERGY_STATE_AUDIBLE),
    )
    assert needs_dnsmos_v2_2(gold, cfg)["required"] is False

