"""028 selection_five_class_v2_auto_noise acceptance cases (§10)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import yaml

from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3 import SelectionV3Config, classify_sample
from audio_engine.core.selection_v3.audio_energy import (
    classify_energy_state,
    compute_energy_from_array,
)
from audio_engine.core.selection_v3.types import (
    BUSINESS_CATEGORIES_FIVE_CLASS,
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
    OUTCOME_CLASSIFIED,
    RULE_VERSION_FIVE_CLASS,
    RULE_VERSION_FIVE_CLASS_V2,
    is_five_class_rule,
    is_five_class_v2_rule,
)
from audio_engine.core.source_naming import apply_source_name_to_single_pipeline
from audio_engine.core.manifest import Manifest

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "selection" / "zh_asr_five_class_v2_auto_noise.yaml"
V1_CFG_PATH = ROOT / "configs" / "selection" / "zh_asr_five_class_v1.yaml"
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


def _sample(
    texts: dict[str, str] | str,
    *,
    sample_id: str = "utt-1",
    quality: dict | None = None,
    labels: dict | None = None,
    failed: set[str] | None = None,
    duration: float = 3.0,
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
        transcripts=transcripts,
        quality=dict(quality or _energy()),
        labels=dict(labels or {}),
        audio={"resampled_16k": f"/data/{sample_id}.wav"},
    )


def _classify(texts, **kwargs):
    sample_kwargs = {
        k: kwargs.pop(k)
        for k in ("sample_id", "quality", "labels", "failed", "duration")
        if k in kwargs
    }
    sample = kwargs.pop("sample", None) or _sample(texts, **sample_kwargs)
    assert not kwargs
    return classify_sample(sample, _cfg(), voicemail_pattern=_vm())


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


def test_01_voicemail_any_route_only():
    texts = _family_text(
        {
            "glm": "您好，您拨打的电话正在通话中",
            "sensevoice": "不需要",
            "qwen": "需要这个服务",
        }
    )
    result = _classify(texts)
    assert result.outcome == OUTCOME_CLASSIFIED
    assert result.category == CATEGORY_VOICEMAIL


def test_02_semantic_risk_polarity_conflict():
    texts = _family_text(
        {
            "glm": "我需要这个服务",
            "sensevoice": "我不需要这个服务",
            "qwen": "",
        }
    )
    result = _classify(texts, quality=_energy(state=ENERGY_STATE_AUDIBLE))
    assert result.category == CATEGORY_SEMANTIC_RISK
    assert result.semantic_subtype in {
        "semantic_reversal",
        "short_polarity_ambiguity",
    }


def test_03_environment_noise_background():
    result = _classify(
        _all_empty(),
        quality=_energy(state=ENERGY_STATE_AUDIBLE, rms_dbfs=-18.0, peak_dbfs=-8.0),
    )
    assert result.category == CATEGORY_ENVIRONMENT_NOISE
    assert result.noise_kind == NOISE_KIND_BACKGROUND
    assert result.classification_source == "auto_family_energy"


def test_04_environment_noise_silence():
    result = _classify(
        _all_empty(),
        quality=_energy(
            state=ENERGY_STATE_INAUDIBLE,
            rms_dbfs=-70.0,
            peak_dbfs=-65.0,
            non_silent_ratio=0.0,
        ),
    )
    assert result.category == CATEGORY_ENVIRONMENT_NOISE
    assert result.noise_kind == NOISE_KIND_SILENCE


def test_05_environment_noise_audio_too_short():
    result = _classify(
        _all_empty(),
        duration=0.1,
        quality=_energy(state=ENERGY_STATE_TOO_SHORT, duration_ms=100.0),
    )
    assert result.category == CATEGORY_ENVIRONMENT_NOISE
    assert result.noise_kind == NOISE_KIND_AUDIO_TOO_SHORT


def test_06_environment_noise_human_noise():
    texts = _family_text(
        {
            "glm": "",
            "sensevoice": "",
            "qwen": "今天天气不错我们出去走走吧",
        }
    )
    result = _classify(texts, quality=_energy(state=ENERGY_STATE_AUDIBLE))
    assert result.category == CATEGORY_ENVIRONMENT_NOISE
    assert result.noise_kind == NOISE_KIND_HUMAN_NOISE


def test_07_critical_short_response_hardcase():
    texts = _family_text(
        {
            "glm": "",
            "sensevoice": "",
            "qwen": "不需要",
        }
    )
    result = _classify(texts, quality=_energy(state=ENERGY_STATE_AUDIBLE))
    assert result.category == CATEGORY_HARDCASE
    assert "critical_short" in (result.hardcase_reason or result.reason)


def test_08_failed_route_not_stable_empty():
    texts = _all_empty()
    result = _classify(
        texts,
        failed={"glm_1", "glm_2"},
        quality=_energy(state=ENERGY_STATE_AUDIBLE),
    )
    assert result.category == CATEGORY_HARDCASE
    assert result.family_state_by_name.get("glm") in {"unavailable", "unstable"}
    assert result.stable_empty_family_count == 2


def test_09_gold_two_stable_one_empty():
    texts = _family_text(
        {
            "glm": "不需要贷款服务",
            "sensevoice": "不需要贷款服务",
            "qwen": "",
        }
    )
    result = _classify(texts, quality=_energy(state=ENERGY_STATE_AUDIBLE))
    assert result.category == CATEGORY_GOLD_CANDIDATE
    assert result.selected_family in {"glm", "sensevoice"}


def test_10_substantive_divergence_hardcase():
    texts = _family_text(
        {
            "glm": "我想办理信用卡业务",
            "sensevoice": "请帮我查询快递物流信息",
            "qwen": "",
        }
    )
    result = _classify(texts, quality=_energy(state=ENERGY_STATE_AUDIBLE))
    assert result.category == CATEGORY_HARDCASE
    assert result.category != CATEGORY_SEMANTIC_RISK


def test_11_intra_family_unstable_hardcase():
    texts = _all_empty()
    texts["qwen_1"] = "需要"
    texts["qwen_2"] = ""
    texts["glm_1"] = "需要"
    texts["glm_2"] = "需要"
    texts["sensevoice_1"] = "需要"
    texts["sensevoice_2"] = "需要"
    result = _classify(texts, quality=_energy(state=ENERGY_STATE_AUDIBLE))
    assert result.category == CATEGORY_HARDCASE
    assert result.family_state_by_name.get("qwen") == "unstable"


def test_12_energy_borderline_hardcase():
    result = _classify(
        _all_empty(),
        quality=_energy(state=ENERGY_STATE_BORDERLINE, rms_dbfs=-50.0, peak_dbfs=-40.0),
    )
    assert result.category == CATEGORY_HARDCASE
    assert "borderline" in (result.hardcase_reason or result.reason or "")
    assert result.needs_review is True


def test_13_classified_category_always_in_five():
    cases = [
        _family_text(
            {"glm": "您好您拨打的用户正在通话中", "sensevoice": "x", "qwen": "y"}
        ),
        _all_empty(),
        _family_text(
            {
                "glm": "我需要这个服务",
                "sensevoice": "我不需要这个服务",
                "qwen": "我需要这个服务",
            }
        ),
        _family_text(
            {
                "glm": "不需要贷款服务",
                "sensevoice": "不需要贷款服务",
                "qwen": "不需要贷款服务",
            }
        ),
    ]
    for texts in cases:
        result = _classify(texts, quality=_energy(state=ENERGY_STATE_AUDIBLE))
        if result.outcome == OUTCOME_CLASSIFIED:
            assert result.category in BUSINESS_CATEGORIES_FIVE_CLASS
            assert result.category is not None


def test_14_mutual_exclusion_counts():
    samples = [
        _sample(
            _all_empty(),
            sample_id="a",
            quality=_energy(state=ENERGY_STATE_AUDIBLE),
        ),
        _sample(
            _family_text(
                {
                    "glm": "不需要贷款服务",
                    "sensevoice": "不需要贷款服务",
                    "qwen": "不需要贷款服务",
                }
            ),
            sample_id="b",
            quality=_energy(state=ENERGY_STATE_AUDIBLE),
        ),
        _sample(
            _family_text(
                {
                    "glm": "我想办理信用卡业务",
                    "sensevoice": "请帮我查询快递物流信息",
                    "qwen": "今天天气不错",
                }
            ),
            sample_id="c",
            quality=_energy(state=ENERGY_STATE_AUDIBLE),
        ),
    ]
    results = [classify_sample(s, _cfg(), voicemail_pattern=_vm()) for s in samples]
    classified = [r for r in results if r.outcome == OUTCOME_CLASSIFIED]
    assert len(classified) == len(samples)
    cats = [r.category for r in classified]
    assert all(c in BUSINESS_CATEGORIES_FIVE_CLASS for c in cats)


def test_15_deterministic_rerun():
    texts = _family_text(
        {
            "glm": "不需要贷款服务",
            "sensevoice": "不需要贷款服务",
            "qwen": "",
        }
    )
    a = _classify(texts, sample_id="det-1", quality=_energy(state=ENERGY_STATE_AUDIBLE))
    b = _classify(texts, sample_id="det-1", quality=_energy(state=ENERGY_STATE_AUDIBLE))
    assert a.category == b.category
    assert a.selected_family == b.selected_family
    assert a.selected_run_id == b.selected_run_id
    assert a.noise_kind == b.noise_kind
    assert a.reason == b.reason


def test_16_v1_v2_product_paths_isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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
    assert "classified_five_class_v1_" in str(v1["output_manifest"])
    assert "classified_five_class_v2_auto_noise_" in str(v2["output_manifest"])
    assert v1["output_manifest"] != v2["output_manifest"]
    assert is_five_class_rule(RULE_VERSION_FIVE_CLASS)
    assert not is_five_class_rule(RULE_VERSION_FIVE_CLASS_V2)
    assert is_five_class_v2_rule(RULE_VERSION_FIVE_CLASS_V2)
    assert not is_five_class_v2_rule(RULE_VERSION_FIVE_CLASS)


def test_17_fixture_noise_kinds_cover_four_paths():
    kinds = {}
    kinds["background"] = _classify(
        _all_empty(), quality=_energy(state=ENERGY_STATE_AUDIBLE)
    ).noise_kind
    kinds["silence"] = _classify(
        _all_empty(), quality=_energy(state=ENERGY_STATE_INAUDIBLE)
    ).noise_kind
    kinds["audio_too_short"] = _classify(
        _all_empty(),
        quality=_energy(state=ENERGY_STATE_TOO_SHORT, duration_ms=50.0),
    ).noise_kind
    kinds["human_noise"] = _classify(
        _family_text(
            {
                "glm": "",
                "sensevoice": "",
                "qwen": "外面雨下得很大路上都积水了",
            }
        ),
        quality=_energy(state=ENERGY_STATE_AUDIBLE),
    ).noise_kind
    assert kinds["background"] == NOISE_KIND_BACKGROUND
    assert kinds["silence"] == NOISE_KIND_SILENCE
    assert kinds["audio_too_short"] == NOISE_KIND_AUDIO_TOO_SHORT
    assert kinds["human_noise"] == NOISE_KIND_HUMAN_NOISE


def test_energy_state_classifier_unit():
    assert (
        classify_energy_state(
            duration_ms=100.0,
            rms_dbfs=-10.0,
            peak_dbfs=-5.0,
            non_silent_ratio=1.0,
            min_duration_ms=300.0,
            min_rms_dbfs=-50.0,
            min_peak_dbfs=-40.0,
            min_non_silent_ratio=0.02,
            borderline_margin_db=3.0,
        )
        == ENERGY_STATE_TOO_SHORT
    )
    assert (
        classify_energy_state(
            duration_ms=3000.0,
            rms_dbfs=-20.0,
            peak_dbfs=-10.0,
            non_silent_ratio=0.5,
            min_duration_ms=300.0,
            min_rms_dbfs=-50.0,
            min_peak_dbfs=-40.0,
            min_non_silent_ratio=0.02,
            borderline_margin_db=3.0,
        )
        == ENERGY_STATE_AUDIBLE
    )
    metrics = compute_energy_from_array(np.zeros(16000), 16000)
    assert metrics["duration_ms"] == pytest.approx(1000.0)
    assert metrics["non_silent_ratio"] == 0.0


def test_v1_still_not_auto_noise_on_all_empty():
    params = yaml.safe_load(V1_CFG_PATH.read_text(encoding="utf-8")) or {}
    ct = dict(params.get("classify_text") or {})
    ct["echo_missing"] = "echo_list_missing"
    ct["echo"] = {
        "qwen": {"extra_exact": [], "asr_config": None, "blank_exact": None},
        "glm": {"extra_exact": [], "asr_config": None},
        "sensevoice": {"extra_exact": [], "asr_config": None},
    }
    params["classify_text"] = ct
    cfg = SelectionV3Config.from_params(params)
    sample = _sample(
        _all_empty(),
        quality={
            "dnsmos_ovrl": 1.0,
            "noise_band": "noisy",
            "calibrated": False,
            **_energy(state=ENERGY_STATE_AUDIBLE),
        },
    )
    result = classify_sample(sample, cfg, voicemail_pattern=_vm())
    assert result.category != CATEGORY_ENVIRONMENT_NOISE


def test_no_manual_annotation_outcome_on_v2():
    result = _classify(
        _family_text(
            {
                "glm": "今天开会讨论项目进度",
                "sensevoice": "明天天气可能会下雨",
                "qwen": "请把文件发给我一份",
            }
        ),
        quality=_energy(state=ENERGY_STATE_AUDIBLE),
    )
    assert result.outcome != "manual_annotation"
    assert result.category == CATEGORY_HARDCASE
    labels = result.to_labels(RULE_VERSION_FIVE_CLASS_V2)
    assert labels["category"] == CATEGORY_HARDCASE
    assert labels["category"] is not None
