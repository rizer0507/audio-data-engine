"""025 classify_text_zh_only_v1: punctuation, tags, non-Chinese, and echo emptying."""

from __future__ import annotations

from pathlib import Path

import yaml

from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3 import SelectionV3Config, classify_sample
from audio_engine.core.selection_v3.classify_text import (
    CLASSIFY_TEXT_VERSION,
    EMPTY_CONTROL_TAG_ONLY,
    EMPTY_HOTWORD_ECHO,
    EMPTY_NON_CHINESE,
    EMPTY_PROMPT_ECHO,
    EMPTY_PUNCT_OR_SYMBOL,
    EchoTable,
    load_echo_table_from_files,
    prepare_classify_text,
)
from audio_engine.core.selection_v3.input_contract import classify_run_status
from audio_engine.core.selection_v3.noise_trigger import evaluate_noise_trigger
from audio_engine.core.selection_v3.types import (
    RUN_STATUS_SUCCESS_EMPTY,
    RUN_STATUS_SUCCESS_TEXT,
    TRIGGER_NON_CHINESE,
)

ROOT = Path(__file__).resolve().parents[1]
V3_CFG = ROOT / "configs" / "selection" / "zh_asr_v3.yaml"
ZH_ONLY_CFG = ROOT / "configs" / "selection" / "zh_asr_v3_zh_only.yaml"
QWEN_DUMP = "没有暂时不用不需要谢谢不可以啊不用我不需要不要不用不需要"

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
SIX = ["glm_1", "glm_2", "sensevoice_1", "sensevoice_2", "qwen_1", "qwen_2"]


def _qwen_echo() -> EchoTable:
    return load_echo_table_from_files(
        asr_config=ROOT / "configs" / "asr" / "qwen_asr.yaml",
        blank_exact=ROOT / "configs" / "normalization" / "blank_exact_qwen_v1.yaml",
    )


def _kimi_echo() -> EchoTable:
    return load_echo_table_from_files(asr_config=ROOT / "configs" / "asr" / "kimi.yaml")


def _cfg_v3(*, chinese_only: bool = False) -> SelectionV3Config:
    raw = yaml.safe_load(V3_CFG.read_text(encoding="utf-8")) or {}
    raw["semantic_lexicon_path"] = str(ROOT / "configs/selection/semantic_lexicon_zh_v3.yaml")
    if chinese_only:
        raw["classify_text"] = {
            "policy": "chinese_only_v1",
            "keep_digits": True,
            "echo_missing": "fail",
        }
        raw["speech_rate"] = {
            "max_chars_per_sec": 25,
            "min_text_chars": 80,
            "disposition": "route_quarantine",
        }
        raw["noise_policy"] = "asr_anomaly_noise_v1"
        raw["quality"] = {"calibrated": False, "noise_policy": "asr_anomaly_noise_v1"}
    return SelectionV3Config.from_params(raw)


def _cfg_zh_only() -> SelectionV3Config:
    return SelectionV3Config.from_yaml(ZH_ONLY_CFG)


def _sample(
    texts: dict[str, str] | str,
    *,
    keys: list[str] | None = None,
    duration: float = 3.0,
) -> Sample:
    keys = list(keys or EIGHT)
    mapping = {key: texts for key in keys} if isinstance(texts, str) else dict(texts)
    transcripts = {}
    for key in keys:
        text = mapping.get(key, "")
        transcripts[key] = {"text": text, "status": "success", "extra": {"raw_text": text}}
    return Sample(
        id="utt-025",
        source_path="dummy.wav",
        duration=duration,
        sha256="abc",
        transcripts=transcripts,
        quality={"dnsmos_status": "not_required", "noise_band": "unknown", "calibrated": False},
        labels={"original_audio_sha256": "abc"},
    )


def _by_family_six(glm: str, sense: str, qwen: str) -> dict[str, str]:
    return {
        "glm_1": glm,
        "glm_2": glm,
        "sensevoice_1": sense,
        "sensevoice_2": sense,
        "qwen_1": qwen,
        "qwen_2": qwen,
    }


def test_tag_and_punct_only_are_empty_not_unknown():
    prepared = prepare_classify_text("<|zh|><|NEUTRAL|><|withitn|>。")
    assert prepared.classify_text == ""
    assert prepared.status == RUN_STATUS_SUCCESS_EMPTY
    assert EMPTY_CONTROL_TAG_ONLY in prepared.empty_reason_codes
    assert EMPTY_PUNCT_OR_SYMBOL in prepared.empty_reason_codes
    assert prepared.pre_filter_language == "empty"

    result = classify_sample(_sample(_by_family_six(prepared.raw_text, prepared.raw_text, prepared.raw_text), keys=SIX), _cfg_zh_only())
    assert result.classify_text_policy == "chinese_only_v1"
    assert result.classify_text_version == CLASSIFY_TEXT_VERSION
    assert "language_unverified" not in result.reason_codes
    assert "language_unresolved" not in result.reason_codes
    assert result.category != "non_speech"


def test_control_tag_around_chinese_is_kept_and_invisible():
    prepared = prepare_classify_text("<||>你好。")
    assert prepared.classify_text == "你好"
    assert prepared.transcript_text == "你好。"
    assert "<|" not in prepared.transcript_text
    assert prepared.status == RUN_STATUS_SUCCESS_TEXT


def test_english_and_mixed_are_whole_route_empty():
    english = prepare_classify_text("Hello, this is a mailbox.")
    assert english.classify_text == ""
    assert EMPTY_NON_CHINESE in english.empty_reason_codes

    mixed = prepare_classify_text("你好 hello")
    assert mixed.classify_text == ""
    assert EMPTY_NON_CHINESE in mixed.empty_reason_codes

    please = prepare_classify_text("Please 转人工")
    assert please.classify_text == ""
    assert EMPTY_NON_CHINESE in please.empty_reason_codes


def test_digits_and_fillers_are_kept():
    phone = prepare_classify_text("回拨13812345678。")
    assert phone.classify_text == "回拨13812345678"
    assert phone.status == RUN_STATUS_SUCCESS_TEXT

    for text in ("嗯", "好", "不需要"):
        prepared = prepare_classify_text(text, echo=_qwen_echo())
        assert prepared.classify_text == text
        assert prepared.status == RUN_STATUS_SUCCESS_TEXT
        assert EMPTY_HOTWORD_ECHO not in prepared.empty_reason_codes


def test_product_abbrev_beside_chinese_is_not_discarded():
    prepared = prepare_classify_text("转IVR")
    assert prepared.status == RUN_STATUS_SUCCESS_TEXT
    assert prepared.classify_text == "转"
    assert EMPTY_NON_CHINESE not in prepared.empty_reason_codes

    # 4.4：2～6 位全大写缩写在有汉字时不单独作为 discard；小写外语词仍整路空。
    ok_abbrev = prepare_classify_text("OK 好的")
    assert ok_abbrev.status == RUN_STATUS_SUCCESS_TEXT
    assert ok_abbrev.classify_text == "好的"
    assert EMPTY_NON_CHINESE not in ok_abbrev.empty_reason_codes

    lowercase = prepare_classify_text("ok 好的")
    assert lowercase.classify_text == ""
    assert EMPTY_NON_CHINESE in lowercase.empty_reason_codes


def test_qwen_table_dump_is_echo_and_single_hotword_is_not():
    echo = _qwen_echo()
    dumped = prepare_classify_text(
        "没有，暂时不用，不需要谢谢，不可以，啊不用，我不需要，不要，不用，不需要。",
        echo=echo,
    )
    assert dumped.classify_text == ""
    assert EMPTY_HOTWORD_ECHO in dumped.empty_reason_codes

    keep = prepare_classify_text("不需要", echo=echo)
    assert keep.classify_text == "不需要"
    assert keep.status == RUN_STATUS_SUCCESS_TEXT


def test_kimi_prompt_echo_is_empty():
    prepared = prepare_classify_text("请撰写这段语音：", echo=_kimi_echo())
    assert prepared.classify_text == ""
    assert EMPTY_PROMPT_ECHO in prepared.empty_reason_codes
    assert prepare_classify_text("请撰写这段语音", echo=_kimi_echo()).status == RUN_STATUS_SUCCESS_EMPTY

    cfg = _cfg_zh_only()
    table = cfg.echo_table_for("kimi")
    assert table is not None
    assert any("请撰写这段语音" in prompt for prompt in table.prompts)


def test_glm_english_does_not_language_hold_when_chinese_agrees():
    texts = _by_family_six("Hello, this is a mailbox.", "客户表示暂时不需要", "客户表示暂时不需要")
    result = classify_sample(_sample(texts, keys=SIX), _cfg_zh_only())
    assert result.empty_reason_by_run["glm_1"]
    assert EMPTY_NON_CHINESE in result.empty_reason_by_run["glm_1"]
    assert result.classify_text_by_run["qwen_1"] == "客户表示暂时不需要"
    assert "language_unverified" not in result.reason_codes
    assert "language_unresolved" not in result.reason_codes
    assert result.category == "business_consistent"
    raw = texts["glm_1"]
    sample = _sample(texts, keys=SIX)
    assert sample.transcripts["glm_1"]["extra"]["raw_text"] == raw


def test_023_keeps_pre_filter_english_and_chinese_is_not_required():
    cfg = _cfg_zh_only()
    english = _sample(
        _by_family_six("Please leave a message", "客户表示暂时不需要", "客户表示暂时不需要"),
        keys=SIX,
    )
    trigger = evaluate_noise_trigger(english, cfg)
    assert TRIGGER_NON_CHINESE in trigger.reasons
    assert trigger.language_by_run["glm_1"] in {"non_zh", "en", "mixed"}
    classified = classify_sample(english, cfg)
    assert TRIGGER_NON_CHINESE in (classified.noise_diagnosis or {}).get("trigger_reasons", [])
    assert classified.pre_filter_language_by_run["glm_1"] in {"non_zh", "en", "mixed"}
    assert classified.classify_text_by_run["glm_1"] == ""

    chinese = _sample(_by_family_six("客户表示暂时不需要", "客户表示暂时不需要", "客户表示暂时不需要"), keys=SIX)
    chinese_trigger = evaluate_noise_trigger(chinese, cfg)
    assert chinese_trigger.required is False
    chinese_result = classify_sample(chinese, cfg)
    assert (chinese_result.noise_diagnosis or {}).get("status") == "not_required"


def test_english_longform_does_not_speech_rate_quarantine():
    english = "hello there " * 40
    texts = {key: "客户表示暂时不需要" for key in EIGHT}
    texts["glm_1"] = english
    texts["glm_2"] = english
    result = classify_sample(_sample(texts, duration=1.0), _cfg_v3(chinese_only=True))
    assert "glm_1" not in result.implausible_routes
    assert result.empty_reason_by_run["glm_1"]
    assert result.classify_text_policy == "chinese_only_v1"


def test_legacy_v3_keeps_tag_only_as_success_text():
    tag = "<|zh|><|NEUTRAL|><|withitn|>。"
    sample = _sample({key: tag for key in EIGHT})
    legacy = _cfg_v3(chinese_only=False)
    assert classify_run_status(sample, "sensevoice_1") == RUN_STATUS_SUCCESS_TEXT
    assert classify_run_status(sample, "sensevoice_1", legacy) == RUN_STATUS_SUCCESS_TEXT
    new = _cfg_v3(chinese_only=True)
    assert classify_run_status(sample, "sensevoice_1", new) == RUN_STATUS_SUCCESS_EMPTY
    legacy_result = classify_sample(sample, legacy)
    new_result = classify_sample(sample, new)
    assert legacy_result.classify_text_policy == "legacy"
    assert new_result.classify_text_policy == "chinese_only_v1"
    assert new_result.classify_text_version == CLASSIFY_TEXT_VERSION
    assert sample.transcripts["sensevoice_1"]["extra"]["raw_text"] == tag


def test_missing_echo_file_fails_closed():
    raw = yaml.safe_load(ZH_ONLY_CFG.read_text(encoding="utf-8")) or {}
    raw["classify_text"]["echo"] = {
        "qwen": {"asr_config": "configs/asr/does-not-exist.yaml"},
        "glm": {"asr_config": "configs/asr/glm.yaml"},
        "sensevoice": {"asr_config": "configs/asr/sensevoice.yaml"},
    }
    try:
        SelectionV3Config.from_params(raw)
    except FileNotFoundError:
        return
    raise AssertionError("missing echo file must fail")
