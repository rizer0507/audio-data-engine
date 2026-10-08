"""037 selection_five_class_v2_4_on_demand_acoustic acceptance tests."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import yaml

from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3 import SelectionV3Config, classify_sample
from audio_engine.core.selection_v3.audio_energy import compute_strict_channel_zero
from audio_engine.core.selection_v3.family_evidence import collect_route_views
from audio_engine.core.selection_v3.five_class_v2 import build_family_states
from audio_engine.core.selection_v3.five_class_v2_4_on_demand_acoustic import (
    QUEUE_A,
    QUEUE_B,
    QUEUE_E,
    QUEUE_NONE,
    needs_dnsmos_v2_4,
    needs_energy_v2_4,
    needs_silero_v2_4,
    route_asr_queue,
)
from audio_engine.core.selection_v3.types import (
    CATEGORY_ENVIRONMENT_NOISE,
    CATEGORY_GOLD_CANDIDATE,
    CATEGORY_HARDCASE,
    CATEGORY_SEMANTIC_RISK,
    CATEGORY_VOICEMAIL,
    NOISE_KIND_BACKGROUND_SUSPECTED,
    NOISE_KIND_HUMAN_NOISE,
    NOISE_KIND_SILENCE,
    RISK_TAG_QWEN_HALLUCINATION_SUSPECTED,
    RULE_VERSION_FIVE_CLASS_V2_3,
    RULE_VERSION_FIVE_CLASS_V2_4,
    RULE_VERSION_FIVE_CLASS_V3,
    is_five_class_v2_3_rule,
    is_five_class_v2_4_rule,
    is_five_class_v3_rule,
)

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "selection" / "zh_asr_five_class_v2_4_on_demand_acoustic.yaml"
VM_PATH = ROOT / "configs" / "selection" / "voicemail_patterns_v1.yaml"
PIPELINE = ROOT / "pipelines" / "classify_dataset_five_class_v2_4_on_demand_acoustic.yaml"

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


def _tx(text: str) -> dict:
    return {"text": text, "status": "success", "raw_text": text}


def _sample(
    texts: dict[str, str],
    *,
    quality: dict | None = None,
    sid: str = "s1",
) -> Sample:
    transcripts = {key: _tx(texts.get(key, "")) for key in KEYS}
    return Sample(
        id=sid,
        source_path=f"{sid}.wav",
        sha256="a" * 64,
        duration=3.0,
        transcripts=transcripts,
        quality=dict(quality or {}),
        audio={"resampled_16k": f"{sid}.wav"},
    )


def _all_empty() -> dict[str, str]:
    return {k: "" for k in KEYS}


def _route_and_classify(sample: Sample, cfg: SelectionV3Config):
    return classify_sample(sample, cfg, voicemail_pattern=_vm())


def test_rule_predicates_isolated():
    assert is_five_class_v2_4_rule(RULE_VERSION_FIVE_CLASS_V2_4)
    assert is_five_class_v2_4_rule(RULE_VERSION_FIVE_CLASS_V3)
    assert is_five_class_v3_rule(RULE_VERSION_FIVE_CLASS_V3)
    assert not is_five_class_v3_rule(RULE_VERSION_FIVE_CLASS_V2_4)
    assert not is_five_class_v2_3_rule(RULE_VERSION_FIVE_CLASS_V2_4)
    assert is_five_class_v2_3_rule(RULE_VERSION_FIVE_CLASS_V2_3)
    assert not is_five_class_v2_4_rule(RULE_VERSION_FIVE_CLASS_V2_3)


def test_config_and_pipeline_exist():
    cfg = _cfg()
    assert cfg.rule_version == RULE_VERSION_FIVE_CLASS_V2_4
    assert CFG_PATH.is_file()
    assert PIPELINE.is_file()
    raw = yaml.safe_load(PIPELINE.read_text(encoding="utf-8"))
    ops = [step["operator"] for step in raw["pipeline"]]
    assert "quality.efficientat" not in ops
    assert "quality.asr_route_v2_4" in ops
    assert "quality.dnsmos_v2_4_candidates" in ops


def test_voicemail_raw_priority():
    cfg = _cfg()
    texts = _all_empty()
    texts["qwen_1"] = "您好，这里是语音信箱，请在提示音后留言"
    sample = _sample(texts)
    result = _route_and_classify(sample, cfg)
    assert result.category == CATEGORY_VOICEMAIL
    assert result.decision_evidence.get("queue_id") == QUEUE_NONE
    assert result.decision_evidence.get("silero_computed") is False


def test_queue_a_before_unstable_hardcase():
    """A: Qwen dual success with raw once + others empty — even if Qwen texts differ."""
    cfg = _cfg()
    texts = _all_empty()
    texts["qwen_1"] = "你好世界"
    texts["qwen_2"] = "完全不同的另一句"
    sample = _sample(texts)
    routes = collect_route_views(sample, cfg)
    bundles = build_family_states(routes, cfg)
    route = route_asr_queue(routes, bundles, cfg, voicemail_pattern=_vm())
    assert route.queue_id == QUEUE_A
    assert route.early_category is None


def test_unstable_non_a_is_hardcase():
    cfg = _cfg()
    texts = {
        "qwen_1": "你好",
        "qwen_2": "你好啊不同",
        "glm_1": "你好",
        "glm_2": "你好",
        "sensevoice_1": "你好",
        "sensevoice_2": "你好",
    }
    sample = _sample(texts)
    routes = collect_route_views(sample, cfg)
    bundles = build_family_states(routes, cfg)
    route = route_asr_queue(routes, bundles, cfg, voicemail_pattern=_vm())
    assert route.early_category == CATEGORY_HARDCASE
    assert route.queue_id == QUEUE_NONE


def test_queue_a_sig_low_qwen_hallucination():
    cfg = _cfg()
    texts = _all_empty()
    texts["qwen_1"] = "幻觉句子"
    texts["qwen_2"] = "幻觉句子"
    sample = _sample(
        texts,
        quality={
            "vad_status": "success",
            "vad_speech_state": "present",
            "dnsmos_status": "success",
            "dnsmos_sig": 2.0,
            "dnsmos_bak": 3.0,
            "dnsmos_ovrl": 3.0,
        },
    )
    result = _route_and_classify(sample, cfg)
    assert result.category == CATEGORY_SEMANTIC_RISK
    assert result.subtype == RISK_TAG_QWEN_HALLUCINATION_SUSPECTED
    assert result.decision_evidence.get("queue_id") == QUEUE_A


def test_queue_a_sig_missing_hardcase():
    cfg = _cfg()
    texts = _all_empty()
    texts["qwen_1"] = "幻觉句子"
    texts["qwen_2"] = ""
    sample = _sample(
        texts,
        quality={
            "vad_status": "success",
            "vad_speech_state": "present",
            "dnsmos_status": "failed",
            "dnsmos_sig": None,
        },
    )
    result = _route_and_classify(sample, cfg)
    assert result.category == CATEGORY_HARDCASE
    assert "sig" in (result.hardcase_reason or result.reason or "")


def test_queue_b_human_noise():
    cfg = _cfg()
    sample = _sample(
        _all_empty(),
        quality={
            "vad_status": "success",
            "vad_speech_state": "present",
            "dnsmos_status": "success",
            "dnsmos_sig": 2.0,
            "dnsmos_bak": 2.5,
            "dnsmos_ovrl": 2.5,
        },
    )
    result = _route_and_classify(sample, cfg)
    assert result.category == CATEGORY_ENVIRONMENT_NOISE
    assert result.noise_kind == NOISE_KIND_HUMAN_NOISE
    assert result.decision_evidence.get("queue_id") == QUEUE_B


def test_queue_b_silence_strict_zero_not_energy_proxy():
    cfg = _cfg()
    sample = _sample(
        _all_empty(),
        quality={
            "vad_status": "success",
            "vad_speech_state": "absent",
            "energy_status": "success",
            "strict_all_channels_zero": True,
            "any_channel_nonzero": False,
            "energy_state": "audible",
            "rms_dbfs": -10.0,
        },
    )
    result = _route_and_classify(sample, cfg)
    assert result.category == CATEGORY_ENVIRONMENT_NOISE
    assert result.noise_kind == NOISE_KIND_SILENCE


def test_queue_b_background_suspected_nonzero():
    cfg = _cfg()
    sample = _sample(
        _all_empty(),
        quality={
            "vad_status": "success",
            "vad_speech_state": "absent",
            "energy_status": "success",
            "strict_all_channels_zero": False,
            "any_channel_nonzero": True,
            "energy_state": "inaudible",
        },
    )
    result = _route_and_classify(sample, cfg)
    assert result.category == CATEGORY_ENVIRONMENT_NOISE
    assert result.noise_kind == NOISE_KIND_BACKGROUND_SUSPECTED


def test_strict_zero_missing_hardcase():
    cfg = _cfg()
    sample = _sample(
        _all_empty(),
        quality={
            "vad_status": "success",
            "vad_speech_state": "absent",
            "energy_status": "success",
            "energy_state": "inaudible",
        },
    )
    result = _route_and_classify(sample, cfg)
    assert result.category == CATEGORY_HARDCASE


def test_gold_critical_short_same_text():
    """名单同文「不要」可为 gold（不触发 D）。"""
    cfg = _cfg()
    texts = {k: "不要" for k in KEYS}
    sample = _sample(
        texts,
        quality={"vad_status": "success", "vad_speech_state": "present"},
    )
    routes = collect_route_views(sample, cfg)
    bundles = build_family_states(routes, cfg)
    route = route_asr_queue(routes, bundles, cfg, voicemail_pattern=_vm())
    assert route.queue_id == QUEUE_E
    result = _route_and_classify(sample, cfg)
    assert result.category == CATEGORY_GOLD_CANDIDATE


def test_on_demand_trigger_counts():
    cfg = _cfg()
    texts = _all_empty()
    texts["qwen_1"] = "有字"
    texts["qwen_2"] = "有字"
    sample = _sample(texts)
    routes = collect_route_views(sample, cfg)
    bundles = build_family_states(routes, cfg)
    route = route_asr_queue(routes, bundles, cfg, voicemail_pattern=_vm())
    assert needs_silero_v2_4(route)["required"] is True
    assert needs_energy_v2_4(route, vad_speech_state="absent")["required"] is False
    assert needs_dnsmos_v2_4(route, vad_speech_state="present")["required"] is True

    texts_vm = _all_empty()
    texts_vm["glm_1"] = "您的来电已转入语音信箱"
    sample_vm = _sample(texts_vm)
    routes_vm = collect_route_views(sample_vm, cfg)
    bundles_vm = build_family_states(routes_vm, cfg)
    route_vm = route_asr_queue(routes_vm, bundles_vm, cfg, voicemail_pattern=_vm())
    assert needs_silero_v2_4(route_vm)["required"] is False

    sample_b = _sample(_all_empty())
    routes_b = collect_route_views(sample_b, cfg)
    bundles_b = build_family_states(routes_b, cfg)
    route_b = route_asr_queue(routes_b, bundles_b, cfg, voicemail_pattern=_vm())
    assert route_b.queue_id == QUEUE_B
    assert needs_energy_v2_4(route_b, vad_speech_state="absent")["required"] is True
    assert needs_dnsmos_v2_4(route_b, vad_speech_state="absent")["required"] is False
    assert needs_dnsmos_v2_4(route_b, vad_speech_state="present")["required"] is True


def test_strict_channel_zero_helper():
    z = compute_strict_channel_zero(np.zeros(100))
    assert z["strict_all_channels_zero"] is True
    nz = compute_strict_channel_zero(np.array([0.0, 0.1, 0.0]))
    assert nz["strict_all_channels_zero"] is False
    stereo = np.zeros((50, 2))
    stereo[10, 1] = 0.01
    s = compute_strict_channel_zero(stereo)
    assert s["strict_all_channels_zero"] is False
    assert s["channel_count"] == 2


def test_five_class_conservation_and_noise_subkinds():
    cfg = _cfg()
    cases = [
        _sample({**_all_empty(), "qwen_1": "语音信箱留言请开始"}),
        _sample(
            {
                "qwen_1": "甲",
                "qwen_2": "乙",
                "glm_1": "甲",
                "glm_2": "甲",
                "sensevoice_1": "甲",
                "sensevoice_2": "甲",
            }
        ),
        _sample(
            _all_empty(),
            quality={
                "vad_status": "success",
                "vad_speech_state": "absent",
                "strict_all_channels_zero": True,
                "any_channel_nonzero": False,
                "energy_status": "success",
            },
        ),
        _sample(
            {k: "今天天气不错" for k in KEYS},
            quality={"vad_status": "success", "vad_speech_state": "present"},
        ),
        _sample(
            _all_empty(),
            quality={
                "vad_status": "success",
                "vad_speech_state": "present",
                "dnsmos_status": "success",
                "dnsmos_sig": 2.0,
                "dnsmos_bak": 3.0,
                "dnsmos_ovrl": 3.0,
            },
        ),
    ]
    cats = []
    noise_kinds = []
    for sample in cases:
        result = _route_and_classify(sample, cfg)
        assert result.category in {
            CATEGORY_VOICEMAIL,
            CATEGORY_SEMANTIC_RISK,
            CATEGORY_ENVIRONMENT_NOISE,
            CATEGORY_GOLD_CANDIDATE,
            CATEGORY_HARDCASE,
        }
        cats.append(result.category)
        if result.category == CATEGORY_ENVIRONMENT_NOISE:
            noise_kinds.append(result.noise_kind)
        assert "queue_id" in (result.decision_evidence or {})
        assert "silero_computed" in (result.decision_evidence or {})
    assert len(cats) == len(cases)
    assert NOISE_KIND_SILENCE in noise_kinds
    assert NOISE_KIND_HUMAN_NOISE in noise_kinds


def test_decision_evidence_has_family_mapping():
    cfg = _cfg()
    sample = _sample(
        {k: "共识文本" for k in KEYS},
        quality={"vad_status": "success", "vad_speech_state": "present"},
    )
    result = _route_and_classify(sample, cfg)
    ev = result.decision_evidence or {}
    assert set(ev.get("configured_families") or []) == {"glm", "sensevoice", "qwen"}
