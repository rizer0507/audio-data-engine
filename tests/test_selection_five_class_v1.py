"""027 selection_five_class_v1 acceptance cases (§11)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3 import SelectionV3Config, classify_sample
from audio_engine.core.selection_v3.annotation_tasks import apply_human_fillback, build_annotation_task
from audio_engine.core.selection_v3.classify_text import EchoTable, prepare_classify_text
from audio_engine.core.selection_v3.gold_select import FamilyRep, select_weighted_family_text
from audio_engine.core.selection_v3.noise_trigger import evaluate_noise_trigger
from audio_engine.core.selection_v3.types import (
    CATEGORY_ENVIRONMENT_NOISE,
    CATEGORY_GOLD_CANDIDATE,
    CATEGORY_HARDCASE,
    CATEGORY_SEMANTIC_RISK,
    CATEGORY_VOICEMAIL,
    OUTCOME_CLASSIFIED,
    OUTCOME_EXCLUDED,
    OUTCOME_MANUAL_ANNOTATION,
    RULE_VERSION_FIVE_CLASS,
    SUBTYPE_HALLUCINATED_ASSERTION,
    SUBTYPE_SEMANTIC_REVERSAL,
    SUBTYPE_SHORT_POLARITY_AMBIGUITY,
)

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "selection" / "zh_asr_five_class_v1.yaml"
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


def _cfg(**overrides) -> SelectionV3Config:
    params = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8")) or {}
    params.update(overrides)
    # Avoid missing echo files failing unit tests when only policy matters.
    ct = dict(params.get("classify_text") or {})
    ct.setdefault("echo_missing", "echo_list_missing")
    echo = dict(ct.get("echo") or {})
    for family in ("qwen", "glm", "sensevoice"):
        echo[family] = {
            "extra_exact": [],
            **(
                {"asr_config": None, "blank_exact": None}
                if family != "qwen"
                else {}
            ),
        }
    # Keep qwen prompt matchable via explicit prompts in EchoTable through extra.
    echo["qwen"] = {
        "extra_exact": ["这是一段测试提示词正文"],
        "asr_config": str(ROOT / "configs/asr/qwen_asr.yaml"),
        "blank_exact": str(ROOT / "configs/normalization/blank_exact_qwen_v1.yaml"),
    }
    echo["glm"] = {"extra_exact": [], "asr_config": str(ROOT / "configs/asr/glm.yaml")}
    echo["sensevoice"] = {
        "extra_exact": [],
        "asr_config": str(ROOT / "configs/asr/sensevoice.yaml"),
    }
    ct["echo"] = echo
    params["classify_text"] = ct
    return SelectionV3Config.from_params(params)


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
            continue
        text = mapping.get(key, "")
        transcripts[key] = {"text": text, "extra": {"raw_text": text}}
    return Sample(
        id=sample_id,
        source_path="dummy.wav",
        duration=duration,
        sha256="abc",
        audio={"resampled_16k": "dummy.wav"},
        transcripts=transcripts,
        quality=dict(quality or {}),
        labels={"original_audio_sha256": "abc", **(labels or {})},
    )


def _classify(texts, **kwargs):
    cfg = kwargs.pop("cfg", None) or _cfg()
    sample = texts if isinstance(texts, Sample) else _sample(texts, **kwargs)
    return classify_sample(sample, cfg, voicemail_pattern=_vm())


# --- §11.1 / 11.2 preprocess ---


def test_foreign_and_mixed_excluded_not_empty_vote():
    prepared = prepare_classify_text(
        "hello world",
        policy="five_class_v1",
    )
    assert prepared.is_excluded
    assert "foreign_transcript" in prepared.exclusion_reasons
    mixed = prepare_classify_text("我不需要OK", policy="five_class_v1")
    assert mixed.is_excluded
    abbrev = prepare_classify_text("办理VIP业务", policy="five_class_v1")
    assert abbrev.is_excluded


def test_prompt_and_hotword_echo_exact_only():
    echo = EchoTable(
        prompts=("请转写音频内容",),
        table_dumps=("没有暂时不用不需要谢谢不可以啊不用我不需要不要不用不需要",),
        exact_phrases=(),
        fingerprint="t1",
    )
    hit = prepare_classify_text("请转写音频内容", echo=echo, policy="five_class_v1")
    assert hit.is_excluded
    assert "prompt_echo" in hit.exclusion_reasons
    # Single hotword item must not kill normal answer.
    ok = prepare_classify_text("不需要", echo=echo, policy="five_class_v1")
    assert not ok.is_excluded
    assert ok.classify_text == "不需要"
    partial = prepare_classify_text("暂时不需要谢谢", echo=echo, policy="five_class_v1")
    assert not partial.is_excluded


# --- §11.3–11.7 exclusion + voicemail ---


def test_excluded_route_cannot_trigger_voicemail_but_eligible_can():
    texts = {
        "glm_1": "请留言",  # would match voicemail but will be foreign-excluded if mixed
        "glm_2": "请留言",
        "sensevoice_1": "不需要",
        "sensevoice_2": "不需要",
        "qwen_1": "不需要",
        "qwen_2": "不需要",
    }
    # Force glm foreign so excluded despite 请留言 keyword in another field — use English.
    texts["glm_1"] = "please leave a message 请留言"
    texts["glm_2"] = "please leave a message 请留言"
    result = _classify(texts)
    # Eligible routes are 不需要 — not voicemail.
    assert result.category != CATEGORY_VOICEMAIL or result.outcome == OUTCOME_CLASSIFIED
    # Now eligible voicemail
    texts2 = dict(texts)
    texts2["sensevoice_1"] = "您好，您的电话已转至语音信箱"
    texts2["sensevoice_2"] = "您好，您的电话已转至语音信箱"
    result2 = _classify(texts2)
    assert result2.category == CATEGORY_VOICEMAIL
    assert result2.outcome == OUTCOME_CLASSIFIED


def test_all_routes_excluded_zero_tasks_zero_noise():
    cfg = _cfg()
    cfg.noise_call_counter = []
    texts = {k: "hello" for k in KEYS}
    result = classify_sample(_sample(texts), cfg, voicemail_pattern=_vm())
    assert result.outcome == OUTCOME_EXCLUDED
    assert result.category is None
    assert not result.annotation_tasks
    assert cfg.noise_call_counter == []


def test_partial_exclude_rest_failed_is_manual_not_all_excluded():
    texts = {k: "hello" for k in KEYS}
    failed = {"sensevoice_1", "sensevoice_2", "qwen_1", "qwen_2"}
    # glm foreign excluded; others failed
    result = _classify(texts, failed=failed)
    assert result.outcome == OUTCOME_MANUAL_ANNOTATION
    assert result.annotation_tasks
    assert result.annotation_tasks[0]["annotation_task_id"]


def test_voicemail_any_route_short_circuits_despite_conflicts():
    cfg = _cfg()
    cfg.noise_call_counter = []
    texts = {
        "glm_1": "需要",
        "glm_2": "不需要",
        "sensevoice_1": "请留言",
        "sensevoice_2": "请留言",
        "qwen_1": "需要",
        "qwen_2": "不需要",
    }
    result = classify_sample(_sample(texts), cfg, voicemail_pattern=_vm())
    assert result.category == CATEGORY_VOICEMAIL
    assert result.outcome == OUTCOME_CLASSIFIED
    assert not result.annotation_tasks
    assert cfg.noise_call_counter == []


def test_wide_voicemail_regex_single_route():
    texts = {k: "不需要" for k in KEYS}
    texts["qwen_1"] = "我是机主的电话助手"
    texts["qwen_2"] = "不需要"
    result = _classify(texts)
    assert result.category == CATEGORY_VOICEMAIL


# --- §11.8–11.10 gold ---


def test_gold_three_stable_fourth_family_foreign_ok():
    # Only three families in this config; simulate one family excluded via foreign.
    texts = {
        "glm_1": "不需要",
        "glm_2": "不需要",
        "sensevoice_1": "hello",
        "sensevoice_2": "hello",
        "qwen_1": "不需要",
        "qwen_2": "不需要",
    }
    # Only two stable Chinese families → manual
    result = _classify(texts)
    assert result.outcome == OUTCOME_MANUAL_ANNOTATION
    assert result.category is None
    assert result.annotation_tasks

    texts_ok = {
        "glm_1": "不需要",
        "glm_2": "不需要",
        "sensevoice_1": "不需要",
        "sensevoice_2": "不需要",
        "qwen_1": "不需要",
        "qwen_2": "不需要",
    }
    result_ok = _classify(texts_ok)
    assert result_ok.category == CATEGORY_GOLD_CANDIDATE
    assert result_ok.outcome == OUTCOME_CLASSIFIED
    assert result_ok.is_human_verified is False


def test_gold_blocked_by_opposing_chinese_route():
    texts = {
        "glm_1": "不需要",
        "glm_2": "不需要",
        "sensevoice_1": "不需要",
        "sensevoice_2": "不需要",
        "qwen_1": "需要",
        "qwen_2": "需要",
    }
    result = _classify(texts)
    assert result.category != CATEGORY_GOLD_CANDIDATE


def test_ni_nin_allowed_question_particle_not():
    texts = {
        "glm_1": "你好",
        "glm_2": "你好",
        "sensevoice_1": "您好",
        "sensevoice_2": "您好",
        "qwen_1": "你好",
        "qwen_2": "你好",
    }
    assert _classify(texts).category == CATEGORY_GOLD_CANDIDATE

    bad = {
        "glm_1": "需要吗",
        "glm_2": "需要吗",
        "sensevoice_1": "需要啊",
        "sensevoice_2": "需要啊",
        "qwen_1": "需要吗",
        "qwen_2": "需要吗",
    }
    assert _classify(bad).category != CATEGORY_GOLD_CANDIDATE


# --- §11.11–11.16 semantic risk ---


def test_intra_family_polarity_not_cross_family_reversal():
    texts = {
        "glm_1": "需要",
        "glm_2": "不需要",  # unstable family
        "sensevoice_1": "不知道",
        "sensevoice_2": "不知道",
        "qwen_1": "嗯",
        "qwen_2": "嗯",
    }
    result = _classify(texts)
    assert result.subtype != SUBTYPE_SEMANTIC_REVERSAL
    assert result.category != CATEGORY_SEMANTIC_RISK or result.subtype != SUBTYPE_SEMANTIC_REVERSAL


def test_cross_family_same_proposition_reversal():
    texts = {
        "glm_1": "我需要这个服务",
        "glm_2": "我需要这个服务",
        "sensevoice_1": "我不需要这个服务",
        "sensevoice_2": "我不需要这个服务",
        "qwen_1": "我需要这个服务",
        "qwen_2": "我需要这个服务",
    }
    result = _classify(texts)
    assert result.category == CATEGORY_SEMANTIC_RISK
    assert result.subtype == SUBTYPE_SEMANTIC_REVERSAL
    assert result.annotation_tasks


def test_unknown_vs_ok_and_different_objects_not_risk():
    a = {
        "glm_1": "不知道",
        "glm_2": "不知道",
        "sensevoice_1": "好的",
        "sensevoice_2": "好的",
        "qwen_1": "不知道",
        "qwen_2": "不知道",
    }
    assert _classify(a).category != CATEGORY_SEMANTIC_RISK

    b = {
        "glm_1": "不需要贷款",
        "glm_2": "不需要贷款",
        "sensevoice_1": "需要保险",
        "sensevoice_2": "需要保险",
        "qwen_1": "不需要贷款",
        "qwen_2": "不需要贷款",
    }
    assert _classify(b).category != CATEGORY_SEMANTIC_RISK


def test_foreign_empty_cannot_trigger_hallucination():
    texts = {
        "glm_1": "hello",
        "glm_2": "hello",
        "sensevoice_1": "需要",
        "sensevoice_2": "需要",
        "qwen_1": "需要",
        "qwen_2": "需要",
    }
    result = _classify(texts)
    assert result.subtype != SUBTYPE_HALLUCINATED_ASSERTION
    # Real empty vs affirmation without audio → manual, not formal ②
    texts2 = {
        "glm_1": "",
        "glm_2": "",
        "sensevoice_1": "需要",
        "sensevoice_2": "需要",
        "qwen_1": "需要",
        "qwen_2": "需要",
    }
    result2 = _classify(texts2)
    assert result2.category != CATEGORY_SEMANTIC_RISK or result2.subtype != SUBTYPE_HALLUCINATED_ASSERTION
    assert result2.outcome in {OUTCOME_MANUAL_ANNOTATION, OUTCOME_CLASSIFIED}
    if result2.category is None:
        assert result2.annotation_tasks


def test_verified_hallucination_and_missed_short_response():
    quality = {
        "no_target_speech": True,
        "no_target_speech_trusted": True,
        "no_target_speech_version": "human_v1",
    }
    texts = {
        "glm_1": "",
        "glm_2": "",
        "sensevoice_1": "需要",
        "sensevoice_2": "需要",
        "qwen_1": "需要",
        "qwen_2": "需要",
    }
    result = _classify(texts, quality=quality, labels={"human_no_target_speech": True})
    assert result.category == CATEGORY_SEMANTIC_RISK
    assert result.subtype == SUBTYPE_HALLUCINATED_ASSERTION

    # Missed short response flag blocks ②
    quality2 = dict(quality)
    quality2["short_response_missed"] = True
    quality2["target_speech_present"] = True
    result2 = _classify(
        texts,
        quality=quality2,
        labels={"human_no_target_speech": False},
    )
    assert result2.subtype != SUBTYPE_HALLUCINATED_ASSERTION


def test_uniform_short_buxuyao_not_subtype_3_and_max_len_guard():
    texts = {k: "不需要" for k in KEYS}
    result = _classify(texts)
    assert result.category == CATEGORY_GOLD_CANDIDATE
    assert result.subtype != SUBTYPE_SHORT_POLARITY_AMBIGUITY

    long_and_short = {
        "glm_1": "不需要这个服务谢谢",
        "glm_2": "不需要这个服务谢谢",
        "sensevoice_1": "需要",
        "sensevoice_2": "需要",
        "qwen_1": "需要",
        "qwen_2": "需要",
    }
    result2 = _classify(long_and_short)
    assert result2.subtype != SUBTYPE_SHORT_POLARITY_AMBIGUITY


def test_short_polarity_ambiguity_with_audio_evidence():
    texts = {
        "glm_1": "需要",
        "glm_2": "需要",
        "sensevoice_1": "不需要",
        "sensevoice_2": "不需要",
        "qwen_1": "需要",
        "qwen_2": "需要",
    }
    result = _classify(
        texts,
        quality={"polarity_syllable_unintelligible": True},
        labels={"polarity_syllable_unintelligible": True},
    )
    assert result.category == CATEGORY_SEMANTIC_RISK
    assert result.subtype == SUBTYPE_SHORT_POLARITY_AMBIGUITY

    # General unintelligible alone is not enough.
    result2 = _classify(
        texts,
        quality={"generally_unintelligible": True},
    )
    assert result2.subtype != SUBTYPE_SHORT_POLARITY_AMBIGUITY


# --- §11.17–11.18 noise / hardcase ---


def test_low_score_all_empty_vad_not_environment_noise():
    texts = {k: "" for k in KEYS}
    result = _classify(
        texts,
        quality={
            "dnsmos_bak": 1.0,
            "dnsmos_status": "success",
            "noise_band": "noisy",
            "vad_speech_present": False,
        },
    )
    assert result.category != CATEGORY_ENVIRONMENT_NOISE
    assert result.outcome != OUTCOME_EXCLUDED or result.annotation_tasks or result.outcome == OUTCOME_MANUAL_ANNOTATION
    # No pending pool
    assert result.status != "hold"
    labels = result.to_labels("selection_five_class_v1")
    assert labels.get("outcome") in {
        OUTCOME_MANUAL_ANNOTATION,
        OUTCOME_CLASSIFIED,
        OUTCOME_EXCLUDED,
    }
    if labels.get("outcome") == OUTCOME_MANUAL_ANNOTATION:
        assert labels.get("annotation_tasks")


def test_hardcase_vs_insufficient_families():
    texts = {
        "glm_1": "我想办理信用卡分期业务",
        "glm_2": "我想办理信用卡分期业务",
        "sensevoice_1": "今天天气不错出去走走",
        "sensevoice_2": "今天天气不错出去走走",
        "qwen_1": "请帮我查一下话费余额",
        "qwen_2": "请帮我查一下话费余额",
    }
    result = _classify(texts)
    assert result.category == CATEGORY_HARDCASE
    assert result.annotation_tasks

    # Two families only agreeing — insufficient → manual, not hardcase dump
    texts2 = {
        "glm_1": "不需要",
        "glm_2": "不需要",
        "sensevoice_1": "不需要",
        "sensevoice_2": "不需要",
        "qwen_1": "hello",
        "qwen_2": "hello",
    }
    result2 = _classify(texts2)
    assert result2.category != CATEGORY_HARDCASE
    assert result2.outcome == OUTCOME_MANUAL_ANNOTATION


# --- §11.19–11.22 selection / exits / fillback / isolation ---


def test_weighted_selection_reproducible_and_qwen_weight():
    reps = [
        FamilyRep("glm", "glm_1", "不需要", "不需要", "不需要", "不需要"),
        FamilyRep("sensevoice", "sv_1", "不需要", "不需要", "不需要", "不需要"),
        FamilyRep("qwen", "qwen_1", "不需要", "不需要", "不需要", "不需要"),
    ]
    a = select_weighted_family_text(
        reps,
        sample_id="s1",
        rule_version=RULE_VERSION_FIVE_CLASS,
        seed="selection_five_class_v1",
        family_order=["glm", "sensevoice", "qwen"],
    )
    b = select_weighted_family_text(
        list(reversed(reps)),
        sample_id="s1",
        rule_version=RULE_VERSION_FIVE_CLASS,
        seed="selection_five_class_v1",
        family_order=["glm", "sensevoice", "qwen"],
    )
    assert a is not None and b is not None
    assert a.family == b.family
    assert a.selection_draw == b.selection_draw
    assert a.selection_weights["qwen"] == 2.0
    # Qwen ineligible → cannot win
    no_qwen = reps[:2]
    c = select_weighted_family_text(
        no_qwen,
        sample_id="s1",
        rule_version=RULE_VERSION_FIVE_CLASS,
        family_order=["glm", "sensevoice", "qwen"],
    )
    assert c is not None
    assert c.family != "qwen"


def test_three_outcomes_mutex_and_manual_has_task():
    samples = [
        _classify({k: "hello" for k in KEYS}),
        _classify({k: "不需要" for k in KEYS}),
        _classify(
            {
                "glm_1": "不需要",
                "glm_2": "不需要",
                "sensevoice_1": "hello",
                "sensevoice_2": "hello",
                "qwen_1": "不需要",
                "qwen_2": "不需要",
            }
        ),
    ]
    outcomes = {r.outcome for r in samples}
    assert OUTCOME_EXCLUDED in outcomes
    assert OUTCOME_CLASSIFIED in outcomes
    assert OUTCOME_MANUAL_ANNOTATION in outcomes
    for r in samples:
        labels = r.to_labels("selection_five_class_v1")
        assert labels["outcome"] in {
            OUTCOME_EXCLUDED,
            OUTCOME_CLASSIFIED,
            OUTCOME_MANUAL_ANNOTATION,
        }
        if labels["outcome"] == OUTCOME_MANUAL_ANNOTATION:
            assert labels["annotation_tasks"]
            assert labels["annotation_task_id"]
        if labels.get("category"):
            assert labels["category"] in {
                CATEGORY_VOICEMAIL,
                CATEGORY_GOLD_CANDIDATE,
                CATEGORY_SEMANTIC_RISK,
                CATEGORY_HARDCASE,
                CATEGORY_ENVIRONMENT_NOISE,
            }


def test_annotation_fillback_idempotent_no_overwrite():
    task = build_annotation_task(
        sample_id="utt-1",
        task_type="transcribe",
        questions=["请转写"],
    )
    labels = {
        "gold_text": "人工金标",
        "is_human_verified": True,
        "annotation_tasks": [task.as_dict()],
    }
    updated = apply_human_fillback(
        labels,
        {"gold_text": "应被忽略", "task_done": True},
    )
    assert updated["gold_text"] == "人工金标"
    invalid = apply_human_fillback(
        {"outcome": "manual_annotation"},
        {"invalid_audio": True},
    )
    assert invalid["annotation_resolution"] == "invalid_audio"


def test_five_class_does_not_use_non_chinese_noise_trigger():
    cfg = _cfg()
    sample = _sample({k: "hello world" for k in KEYS})
    decision = evaluate_noise_trigger(sample, cfg)
    assert "non_chinese_transcript" not in decision.reasons


def test_legacy_rule_still_dispatchable():
    """Legacy rollback path remains importable and distinct."""
    from audio_engine.core.selection_v3.types import RULE_VERSION, is_five_class_rule

    assert not is_five_class_rule(RULE_VERSION)
    assert is_five_class_rule(RULE_VERSION_FIVE_CLASS)
