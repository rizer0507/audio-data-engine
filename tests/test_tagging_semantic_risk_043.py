"""043 semantic_risk_text_disagreement: (①且②) 或 ③ 双分支验收。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from audio_engine.core.sample import Sample
from audio_engine.core.tagging.config import load_tagging_config
from audio_engine.core.tagging.engine import MultiTagEngine
from audio_engine.core.tagging.library import (
    TagLibraryError,
    assert_resume_compatible,
    build_definition_snapshot,
)
from audio_engine.core.tagging.service import TaggingRunRequest, TaggingService
from audio_engine.core.tagging.types import Ternary

ROOT = Path(__file__).resolve().parents[1]
LIB_CFG = ROOT / "configs" / "tagging" / "zh_asr_multi_tag_v1.yaml"
OLD_INLINE_CFG = ROOT / "configs" / "tagging" / "semantic_keywords_sg_qwen_1012.yaml"
LIBRARY_DIR = ROOT / "configs" / "tagging" / "library"
LEXICON_PATH = ROOT / "configs" / "tagging" / "lexicons" / "semantic_risk_keywords_1012.yaml"

TAG_ID = "semantic_risk_text_disagreement"
RULE_A = "rule_semantic_risk_keyword_sg_qwen_v1"
RULE_B = "rule_semantic_risk_single_char_pairwise_v1"
QWEN_EXCLUSIONS = "semantic_risk_qwen_exclusions"
OLD_TAG_ID = "semantic_keywords_sg_agree_qwen_disagree"
OLD_RULE_ID = "rule_semantic_keywords_sg_qwen_v1"

KEYS = [
    "sensevoice_1",
    "sensevoice_2",
    "glm_1",
    "glm_2",
    "qwen_1",
    "qwen_2",
]


def _tx(text: str, *, status: str = "success", raw: str | None = None) -> dict:
    """原文优先走 extra.raw_text；未指定时与 text 相同（经 text 回退读取）。"""
    entry: dict = {"text": text, "status": status}
    if raw is not None:
        entry["extra"] = {"raw_text": raw}
    else:
        entry["extra"] = {"raw_text": text}
    return entry


def _sample_families(
    sv: str,
    glm: str,
    qwen: str,
    *,
    sid: str = "s1",
    failed: set[str] | None = None,
    raw_overrides: dict[str, str] | None = None,
    status_overrides: dict[str, str] | None = None,
    labels: dict | None = None,
) -> Sample:
    """双跑同文构造；可选按路覆盖原文或状态。"""
    failed = failed or set()
    raw_overrides = raw_overrides or {}
    status_overrides = status_overrides or {}
    family_text = {
        "sensevoice_1": sv,
        "sensevoice_2": sv,
        "glm_1": glm,
        "glm_2": glm,
        "qwen_1": qwen,
        "qwen_2": qwen,
    }
    transcripts = {}
    for key in KEYS:
        if key in failed:
            # 失败路可残留原文，验证不得绕过 G
            residual = raw_overrides.get(key, family_text[key])
            transcripts[key] = _tx(residual, status="failed", raw=residual)
        else:
            text = family_text[key]
            raw = raw_overrides.get(key, text)
            st = status_overrides.get(key, "success")
            transcripts[key] = _tx(text, status=st, raw=raw)
    return Sample(
        id=sid,
        source_path=f"{sid}.wav",
        sha256="c" * 64,
        duration=2.0,
        transcripts=transcripts,
        quality={},
        labels=dict(labels or {}),
        audio={"resampled_16k": f"{sid}.wav"},
    )


def _rule_map(result) -> dict[str, str]:
    return {rr.rule_id: rr.ternary.value for rr in result.rule_results}


def _rule_ternary(result, rule_id: str) -> str:
    for rr in result.rule_results:
        if rr.rule_id == rule_id:
            return rr.ternary.value
    raise AssertionError(f"missing rule {rule_id}")


def _has_tag(result, tag_id: str = TAG_ID) -> bool:
    return tag_id in result.effective_tags or any(
        rr.tag_id == tag_id and rr.ternary == Ternary.MATCHED for rr in result.rule_results
    )


@pytest.fixture(scope="module")
def lib_engine():
    return MultiTagEngine(load_tagging_config(LIB_CFG))


@pytest.fixture(scope="module")
def old_engine():
    return MultiTagEngine(load_tagging_config(OLD_INLINE_CFG))


# ---------------------------------------------------------------------------
# §7.1 固定离线规则样例
# ---------------------------------------------------------------------------


def test_case1_long_text_branch_a_no_single_char_limit(lib_engine):
    """需要贷款/需要贷款/天气晴朗：A 命中、B 不命中；证明 A 无单字限制。"""
    sample = _sample_families("需要贷款", "需要贷款", "天气晴朗", sid="c1")
    result = lib_engine.evaluate_sample(sample, dry_run=True)
    assert _rule_ternary(result, RULE_A) == Ternary.MATCHED.value
    assert _rule_ternary(result, RULE_B) == Ternary.NOT_MATCHED.value
    assert TAG_ID in {
        rr.tag_id for rr in result.rule_results if rr.ternary == Ternary.MATCHED
    }


def test_case2_single_char_alone_does_not_support_branch_a(lib_engine):
    """好/好/要：单字不触发 A；SG 同文不满足三族两两不等价，B 也不命中。"""
    sample = _sample_families("好", "好", "要", sid="c2")
    result = lib_engine.evaluate_sample(sample, dry_run=True)
    assert _rule_ternary(result, RULE_A) == Ternary.NOT_MATCHED.value
    assert _rule_ternary(result, RULE_B) == Ternary.NOT_MATCHED.value
    assert not _has_tag(result)


@pytest.mark.parametrize(
    "sg,qwen,expected",
    [
        ("需要", "不需要", Ternary.NOT_MATCHED),
        ("不需要", "需要", Ternary.MATCHED),
        ("要用", "不用", Ternary.NOT_MATCHED),
        ("不用", "要用", Ternary.MATCHED),
        ("需要用", "不用", Ternary.NOT_MATCHED),
        ("不用", "需要用", Ternary.MATCHED),
    ],
)
def test_branch_a_excludes_qwen_refusal_but_retains_sg_refusal(
    lib_engine, old_engine, sg, qwen, expected
):
    """旧 A 接受双向分歧；新 A 仅因 Qwen 命中排除表而退出，SG 拒绝不受影响。"""
    sample = _sample_families(sg, sg, qwen)
    old_result = old_engine.evaluate_sample(sample, dry_run=True)
    assert _rule_ternary(old_result, OLD_RULE_ID) == Ternary.MATCHED.value
    result = lib_engine.evaluate_sample(sample, dry_run=True)
    assert _rule_ternary(result, RULE_A) == expected.value
    assert _rule_ternary(result, RULE_B) == Ternary.NOT_MATCHED.value
    assert _has_tag(result) == (expected == Ternary.MATCHED)


@pytest.mark.parametrize(
    "qwen",
    [
        "不需要了谢谢",
        "不用了谢谢",
        "不要了谢谢",
        "不可用了谢谢",
        "没有了谢谢",
        "沒有了謝謝",
        "不，需 要了，谢谢。",
        "不要紧，可以发给我。",
    ],
)
def test_qwen_exclusions_block_tag_and_explain_both_branches(lib_engine, old_engine, qwen):
    """按稳定代表文本做包含匹配；后缀、繁体、清洗及嵌套肯定均不能绕过排除。"""
    sample = _sample_families("需要贷款", "需要贷款", qwen)
    old_result = old_engine.evaluate_sample(sample, dry_run=True)
    assert _rule_ternary(old_result, OLD_RULE_ID) == Ternary.MATCHED.value

    result = lib_engine.evaluate_sample(sample, dry_run=True)
    representative = lib_engine.build_asr(sample).representative_text["qwen"]
    assert not _has_tag(result)
    for rule_id in (RULE_A, RULE_B):
        assert _rule_ternary(result, rule_id) == Ternary.NOT_MATCHED.value
        rule = next(rr for rr in result.rule_results if rr.rule_id == rule_id)
        exclusion_evidence = [
            ev for ev in rule.condition.evidence
            if ev.capability == "lexicon_match" and ev.details.get("lexicon_id") == QWEN_EXCLUSIONS
        ]
        assert len(exclusion_evidence) == 1
        assert exclusion_evidence[0].value["matched"]
        assert exclusion_evidence[0].value["hits"]
        assert exclusion_evidence[0].details["text"] == representative


@pytest.mark.parametrize("sg", ["不需要", "不用", "不要", "不可用", "没有"])
def test_exclusions_only_apply_to_qwen(lib_engine, sg):
    """排除词只出现在 SG 时，Qwen 明确肯定仍可命中 A。"""
    sample = _sample_families(sg, sg, "需要贷款")
    result = lib_engine.evaluate_sample(sample, dry_run=True)
    assert _rule_ternary(result, RULE_A) == Ternary.MATCHED.value
    assert _has_tag(result)


@pytest.mark.parametrize("qwen", ["需要了，谢谢。", "可以用，谢谢。", "不可以，谢谢。"])
def test_qwen_exclusion_allowlist_controls(lib_engine, qwen):
    """标点不影响正向文本；排除范围限于指定五词，不自动扩展为其他否定词。"""
    sample = _sample_families("需要贷款", "需要贷款", qwen)
    result = lib_engine.evaluate_sample(sample, dry_run=True)
    assert _rule_ternary(result, RULE_A) == Ternary.MATCHED.value
    assert _has_tag(result)


@pytest.mark.parametrize(
    "text",
    [
        "好",
        "要",
        "是",
        "有",
        "不",
        "没",
        "用",
        "对",
        "行",
        "嗯",
        "啊",
        "哦",
        "呃",
        "吗",
        "吧",
        "嗯嗯",
        "好的",
        "不是",
        "没有",
        "没用",
        "没要",
        "是不是",
        "有没有",
        "怎么使用",
        "这是一个普通句子",
        "嗯，今天怎么了",
        "你说啊",
    ],
)
def test_branch_a_rejects_generic_words_and_particles(lib_engine, old_engine, text):
    """即使 SG 同文且 Qwen 异文，泛语气、普通单字和普通句子也不能触发关键词分支。"""
    sample = _sample_families(text, text, "天气晴朗")
    for engine, rule_id in ((lib_engine, RULE_A), (old_engine, OLD_RULE_ID)):
        result = engine.evaluate_sample(sample, dry_run=True)
        assert _rule_ternary(result, rule_id) == Ternary.NOT_MATCHED.value
    result = lib_engine.evaluate_sample(sample, dry_run=True)
    assert _rule_ternary(result, RULE_B) == Ternary.NOT_MATCHED.value
    assert not _has_tag(result)


def test_fenshen_generic_particle_regression(lib_engine, old_engine):
    """真实六路只因 Qwen 原文中的「啊」入选的样本，在收紧词表后退出。"""
    sv_raw = "<|zh|><|HAPPY|><|BGM|><|withitn|>怎么了？你说。"
    glm_raw = "怎么了你说"
    qwen_raw = "怎么了？你说啊。连队，连队。"
    sample = _sample_families(
        sv_raw,
        glm_raw,
        qwen_raw,
        sid="fenshen-7b304a93-2c9a884c",
    )
    asr = lib_engine.build_asr(sample)
    assert asr.all_families_dual_success
    assert set(asr.family_state_by_name.values()) == {"stable_text"}

    for engine, rule_id in ((lib_engine, RULE_A), (old_engine, OLD_RULE_ID)):
        result = engine.evaluate_sample(sample, dry_run=True)
        assert _rule_ternary(result, rule_id) == Ternary.NOT_MATCHED.value
        rule = next(rr for rr in result.rule_results if rr.rule_id == rule_id)
        lexicon_evidence = [
            ev for ev in rule.condition.evidence
            if ev.capability == "lexicon_match"
            and ev.details.get("lexicon_id") == "semantic_risk_keywords"
        ]
        assert len(lexicon_evidence) == len(KEYS)
        assert all(not ev.value["matched"] for ev in lexicon_evidence)
        comparisons = {
            ev.capability: ev.value["equal"]
            for ev in rule.condition.evidence
            if ev.capability in {"texts_equivalent", "texts_not_equivalent"}
        }
        assert comparisons == {"texts_equivalent": True, "texts_not_equivalent": False}
    result = lib_engine.evaluate_sample(sample, dry_run=True)
    assert _rule_ternary(result, RULE_B) == Ternary.NOT_MATCHED.value
    assert not _has_tag(result)


def test_case3_keyword_only_insufficient(lib_engine):
    """三族同文有关键词：两分支均不命中。"""
    sample = _sample_families("需要贷款", "需要贷款", "需要贷款", sid="c3")
    result = lib_engine.evaluate_sample(sample, dry_run=True)
    assert _rule_ternary(result, RULE_A) == Ternary.NOT_MATCHED.value
    assert _rule_ternary(result, RULE_B) == Ternary.NOT_MATCHED.value
    assert TAG_ID not in {
        rr.tag_id for rr in result.rule_results if rr.ternary == Ternary.MATCHED
    }


def test_case4_disagreement_without_keyword_or_single_char(lib_engine):
    """甲乙/甲乙/甲丙：仅有 SG 同文 Qwen 异文但无关键词且非单字 → 不命中。"""
    sample = _sample_families("甲乙", "甲乙", "甲丙", sid="c4")
    result = lib_engine.evaluate_sample(sample, dry_run=True)
    assert _rule_ternary(result, RULE_A) == Ternary.NOT_MATCHED.value
    assert _rule_ternary(result, RULE_B) == Ternary.NOT_MATCHED.value


@pytest.mark.parametrize(
    "sv,glm,qwen",
    [
        ("甲", "乙", "丙"),
        ("好", "要", "不"),
        ("嗯", "啊", "哦"),
    ],
)
def test_case5_branch_b_independent_no_keyword(lib_engine, sv, glm, qwen):
    """Qwen 1 字且三族两两不等价即进入 B，包括从关键词表删除的单字。"""
    sample = _sample_families(sv, glm, qwen, sid="c5")
    result = lib_engine.evaluate_sample(sample, dry_run=True)
    assert _rule_ternary(result, RULE_A) == Ternary.NOT_MATCHED.value
    assert _rule_ternary(result, RULE_B) == Ternary.MATCHED.value
    assert _has_tag(result)
    # 证据中不应出现关键词命中
    a_rr = next(rr for rr in result.rule_results if rr.rule_id == RULE_A)
    lex_hits = []
    for ev in a_rr.condition.evidence or []:
        val = ev.value
        if isinstance(val, dict) and val.get("hits"):
            lex_hits.extend(val["hits"])
    assert not lex_hits


@pytest.mark.parametrize(
    "sv,glm,qwen",
    [("甲乙", "丙丁", "戊"), ("不需要", "没有", "有"), ("没有", "不要", "有")],
)
def test_branch_b_matches_without_admission_lexicon(sv, glm, qwen):
    """B 不依赖准入词表；在三族两两不等价时，SG 排除词不阻止短字命中。"""
    cfg = load_tagging_config(LIB_CFG)
    cfg.lexicons.pop("semantic_risk_keywords")
    assert QWEN_EXCLUSIONS in cfg.lexicons
    sample = _sample_families(sv, glm, qwen, sid="b_without_admission_lexicon")
    result = MultiTagEngine(cfg).evaluate_sample(sample, dry_run=True, rule_ids={RULE_B})
    assert _rule_ternary(result, RULE_B) == Ternary.MATCHED.value
    assert _has_tag(result)


@pytest.mark.parametrize(
    "sv,glm,qwen,sid,expected",
    [
        ("甲", "乙", "乙", "c6_gq", Ternary.NOT_MATCHED),
        ("甲", "乙", "甲", "c6_sq", Ternary.NOT_MATCHED),
        ("甲", "甲", "乙", "c6_sg", Ternary.NOT_MATCHED),
        ("甲乙", "甲丙", "丁", "c6_multi_sg", Ternary.MATCHED),
        ("甲乙", "甲乙", "丁", "c6_equal_multi_sg", Ternary.NOT_MATCHED),
        ("甲乙", "丙丁", "戊己", "c6_two_han", Ternary.MATCHED),
        ("甲乙丙", "丁戊己", "庚辛壬", "c6_three_han", Ternary.MATCHED),
        ("甲乙", "丙丁", "戊己庚辛", "c6_four_han", Ternary.NOT_MATCHED),
    ],
)
def test_case6_qwen_short_text_requires_pairwise_disagreement(
    lib_engine, sv, glm, qwen, sid, expected
):
    """Qwen 须为 1~3 汉字整段，且三族两两不等价；SG 不限长度。"""
    sample = _sample_families(sv, glm, qwen, sid=sid)
    result = lib_engine.evaluate_sample(sample, dry_run=True)
    assert _rule_ternary(result, RULE_B) == expected.value
    assert _rule_ternary(result, RULE_A) == Ternary.NOT_MATCHED.value
    assert _has_tag(result) == (expected == Ternary.MATCHED)


@pytest.mark.parametrize(
    "sv,glm,qwen,expected",
    [
        ("您", "你", "他", Ternary.NOT_MATCHED),
        ("對", "对", "他", Ternary.NOT_MATCHED),
        ("您", "你", "你", Ternary.NOT_MATCHED),
        ("對", "对", "对", Ternary.NOT_MATCHED),
        ("甲", "甲", "甲", Ternary.NOT_MATCHED),
        ("甲", "乙", "丙", Ternary.MATCHED),
    ],
)
def test_case7_pairwise_disagreement_and_harmless_equivalence(
    lib_engine, sv, glm, qwen, expected
):
    """三族两两不等价才命中 B；无害等价（您/你、繁简）视为同文，阻断 B。"""
    sample = _sample_families(sv, glm, qwen, sid="c7")
    result = lib_engine.evaluate_sample(sample, dry_run=True)
    assert _rule_ternary(result, RULE_B) == expected.value
    assert _rule_ternary(result, RULE_A) == Ternary.NOT_MATCHED.value
    assert _has_tag(result) == (expected == Ternary.MATCHED)


@pytest.mark.parametrize(
    "sv,glm,qwen,sid",
    [
        ("甲", "乙", "丙丁戊己", "c8_four"),
        ("甲", "乙", "1", "c8_digit"),
    ],
)
def test_case8_qwen_must_be_one_to_three_han(lib_engine, sv, glm, qwen, sid):
    """即使三族两两不等价，Qwen 超过 3 字或非汉字也不得进入 B。"""
    sample = _sample_families(sv, glm, qwen, sid=sid)
    result = lib_engine.evaluate_sample(sample, dry_run=True)
    assert _rule_ternary(result, RULE_B) == Ternary.NOT_MATCHED.value


def test_case9_cleaned_punct_space_forms_single_han(lib_engine):
    """SG 可为多字，Qwen 原文去空格标点后恰好一个汉字且三族两两不等价。"""
    sample = _sample_families(" 甲乙！ ", " 甲丙。 ", " 丁？ ", sid="c9")
    result = lib_engine.evaluate_sample(sample, dry_run=True)
    assert _rule_ternary(result, RULE_B) == Ternary.MATCHED.value


@pytest.mark.parametrize(
    "sv,glm,qwen,matched_rule",
    [
        pytest.param("需要贷款", "需要贷款", "天气晴朗", RULE_A, id="branch_a"),
        pytest.param("甲乙", "丙丁", "戊", RULE_B, id="branch_b"),
    ],
)
@pytest.mark.parametrize(
    "fault,route",
    [
        ("failed", "qwen_2"),
        ("empty", "sensevoice_1"),
        ("empty", "glm_1"),
        ("empty", "qwen_1"),
        ("unstable", "sensevoice_2"),
        ("missing", "glm_2"),
    ],
)
def test_case10_g_gate_blocks_failure_empty_unstable(
    lib_engine, sv, glm, qwen, matched_rule, fault, route
):
    """已可命中的 A/B 样本遇到失败、任一家空文、族内不稳定或缺路后均退出。"""
    sample = _sample_families(sv, glm, qwen, sid=f"c10_{fault}_{route}")
    baseline = lib_engine.evaluate_sample(sample, dry_run=True)
    assert _rule_ternary(baseline, matched_rule) == Ternary.MATCHED.value

    if fault == "failed":
        # 失败状态仍残留原文，不能靠内容满足规则。
        sample.transcripts[route]["status"] = "failed"
    elif fault == "empty":
        family = route.rsplit("_", 1)[0]
        for run in (1, 2):
            sample.transcripts[f"{family}_{run}"] = _tx("")
    elif fault == "unstable":
        sample.transcripts[route] = _tx("丁")
    else:
        del sample.transcripts[route]

    result = lib_engine.evaluate_sample(sample, dry_run=True)
    assert _rule_ternary(result, RULE_A) != Ternary.MATCHED.value
    assert _rule_ternary(result, RULE_B) != Ternary.MATCHED.value
    assert not _has_tag(result)


@pytest.mark.parametrize("hit_run", KEYS)
def test_case11_keyword_on_each_of_six_routes(lib_engine, hit_run):
    """唯一关键词命中放在六路中任一路，其余满足 A：均可命中。"""
    # SG 四路原文「闭 嘴」、Qwen「骚 扰」；仅指定一路去空格命中词表
    raw_overrides = {
        "sensevoice_1": "闭 嘴",
        "sensevoice_2": "闭 嘴",
        "glm_1": "闭 嘴",
        "glm_2": "闭 嘴",
        "qwen_1": "骚 扰",
        "qwen_2": "骚 扰",
    }
    if hit_run.startswith("qwen"):
        raw_overrides[hit_run] = "骚扰"
        # 可比文本：SG=闭嘴（空格折叠），Qwen=骚扰 → 异文；SG≈GLM
        sample = _sample_families(
            "闭 嘴",
            "闭 嘴",
            "骚 扰",
            sid=f"c11_{hit_run}",
            raw_overrides=raw_overrides,
        )
        # 命中路的 text 也改为无空格，保持与 raw 一致且家族仍稳定为「骚扰」
        sample.transcripts[hit_run] = _tx("骚扰", status="success", raw="骚扰")
        # 另一路 Qwen 保持「骚 扰」→ 清洗后同为「骚扰」，族内稳定
    else:
        raw_overrides[hit_run] = "闭嘴"
        sample = _sample_families(
            "闭 嘴",
            "闭 嘴",
            "骚 扰",
            sid=f"c11_{hit_run}",
            raw_overrides=raw_overrides,
        )
        sample.transcripts[hit_run] = _tx("闭嘴", status="success", raw="闭嘴")

    result = lib_engine.evaluate_sample(sample, dry_run=True)
    assert _rule_ternary(result, RULE_A) == Ternary.MATCHED.value, hit_run
    # 不误用演示词表 tag_1
    a_rr = next(rr for rr in result.rule_results if rr.rule_id == RULE_A)
    evidence_blob = json.dumps(
        [e.as_dict() for e in (a_rr.condition.evidence or [])],
        ensure_ascii=False,
        default=str,
    )
    assert "semantic_risk_keywords" in evidence_blob or "闭嘴" in evidence_blob or "骚扰" in evidence_blob


def test_branch_a_parity_when_qwen_is_not_excluded(lib_engine, old_engine):
    """Qwen 未命中新增排除表时，A 与旧内联规则的三态及等价证据保持一致。"""
    samples = [
        _sample_families("需要贷款", "需要贷款", "天气晴朗", sid="parity_hit"),
        _sample_families("好", "好", "要", sid="parity_hao"),
        _sample_families("需要贷款", "需要贷款", "需要贷款", sid="parity_agree"),
        _sample_families("甲乙", "甲乙", "甲丙", sid="parity_no_kw"),
        _sample_families(
            "需要贷款",
            "需要贷款",
            "天气晴朗",
            sid="parity_fail",
            failed={"glm_1"},
        ),
    ]
    for sample in samples:
        new_r = lib_engine.evaluate_sample(sample, dry_run=True)
        old_r = old_engine.evaluate_sample(sample, dry_run=True)
        new_a = _rule_ternary(new_r, RULE_A)
        old_t = _rule_ternary(old_r, OLD_RULE_ID)
        assert new_a == old_t, sample.id

        new_rr = next(rr for rr in new_r.rule_results if rr.rule_id == RULE_A)
        old_rr = next(rr for rr in old_r.rule_results if rr.rule_id == OLD_RULE_ID)
        assert new_rr.tag_id == TAG_ID
        assert old_rr.tag_id == OLD_TAG_ID
        # 关键能力证据：等价/不等价结论一致
        def _eq_vals(rr):
            out = []
            for ev in rr.condition.evidence or []:
                if ev.capability in {"texts_equivalent", "texts_not_equivalent"} and isinstance(
                    ev.value, dict
                ):
                    out.append(
                        (
                            ev.capability,
                            ev.value.get("equal"),
                            ev.value.get("left"),
                            ev.value.get("right"),
                        )
                    )
            return out

        assert _eq_vals(new_rr) == _eq_vals(old_rr), sample.id


def test_coexist_with_tag_1_and_critical_short(lib_engine):
    """新 tag 可与 tag_1 / critical_short 同时存在；有效列表去重。"""
    sample = _sample_families("好", "好", "需要", sid="coexist")
    result = lib_engine.evaluate_sample(sample, dry_run=False, snapshot_id="snap043")
    matched = {rr.tag_id for rr in result.rule_results if rr.ternary == Ternary.MATCHED}
    assert TAG_ID in matched
    assert "tag_1" in matched
    assert "critical_short" in matched
    assert result.effective_tags.count(TAG_ID) == 1


def test_branches_a_and_b_are_mutually_exclusive_on_sg(lib_engine):
    """A 要求 SV≈GLM，B 要求 SV≉GLM，故两分支不能同时命中；tag 仍只写一份。"""
    only_a = _sample_families("需要", "需要", "要", sid="only_a")
    only_b = _sample_families("甲", "乙", "丙", sid="only_b")
    ra = lib_engine.evaluate_sample(only_a, dry_run=False, snapshot_id="snap043a")
    rb = lib_engine.evaluate_sample(only_b, dry_run=False, snapshot_id="snap043b")
    assert _rule_ternary(ra, RULE_A) == Ternary.MATCHED.value
    assert _rule_ternary(ra, RULE_B) == Ternary.NOT_MATCHED.value
    assert ra.effective_tags.count(TAG_ID) == 1
    assert _rule_ternary(rb, RULE_A) == Ternary.NOT_MATCHED.value
    assert _rule_ternary(rb, RULE_B) == Ternary.MATCHED.value
    assert rb.effective_tags.count(TAG_ID) == 1


def test_rule_order_independent_for_new_tag(lib_engine):
    cfg = load_tagging_config(LIB_CFG)
    sample = _sample_families("甲", "乙", "丙", sid="order")
    r1 = MultiTagEngine(cfg).evaluate_sample(sample, dry_run=True)
    cfg.rules = list(reversed(list(cfg.rules)))
    r2 = MultiTagEngine(cfg).evaluate_sample(sample, dry_run=True)
    assert _rule_ternary(r1, RULE_A) == _rule_ternary(r2, RULE_A)
    assert _rule_ternary(r1, RULE_B) == _rule_ternary(r2, RULE_B)
    assert sorted(r1.effective_tags) == sorted(r2.effective_tags)


def test_unselected_strategy_unchanged(tmp_path: Path):
    """未选入新 tag 的策略结果不变（沿用旧内联七项 fixture）。"""
    inline = ROOT / "tests" / "fixtures" / "tagging" / "zh_asr_multi_tag_v1_inline.yaml"
    cfg = load_tagging_config(inline)
    assert TAG_ID not in cfg.tags
    sample = _sample_families("甲", "乙", "丙", sid="unselected")
    result = MultiTagEngine(cfg).evaluate_sample(sample, dry_run=True)
    assert TAG_ID not in result.effective_tags
    assert all(rr.tag_id != TAG_ID for rr in result.rule_results)


def test_manual_block_and_restore(tmp_path: Path):
    cfg = load_tagging_config(LIB_CFG)
    service = TaggingService(cfg, config_path=LIB_CFG)
    sample = _sample_families("甲", "乙", "丙", sid="manual043")
    work = tmp_path / "work"
    r1 = service.run(
        TaggingRunRequest(
            batch_id="b043",
            config_path=LIB_CFG,
            samples=[sample],
            work_root=work,
            publish=True,
            output_manifest=tmp_path / "out1.parquet",
            summary_xlsx=tmp_path / "sum1.xlsx",
        )
    )
    assert TAG_ID in r1.samples[0].labels["tags"]
    service.remove_tag(
        batch_id="b043",
        sample_id=sample.id,
        tag_id=TAG_ID,
        audio_sha256=sample.sha256,
        actor="test",
        reason="block",
        work_root=work,
    )
    r2 = service.run(
        TaggingRunRequest(
            batch_id="b043",
            config_path=LIB_CFG,
            samples=[sample],
            work_root=work,
            publish=True,
            output_manifest=tmp_path / "out2.parquet",
            summary_xlsx=tmp_path / "sum2.xlsx",
        )
    )
    assert TAG_ID not in r2.samples[0].labels["tags"]
    service.restore_tag(
        batch_id="b043",
        sample_id=sample.id,
        tag_id=TAG_ID,
        audio_sha256=sample.sha256,
        actor="test",
        reason="restore",
        work_root=work,
    )
    r3 = service.run(
        TaggingRunRequest(
            batch_id="b043",
            config_path=LIB_CFG,
            samples=[sample],
            work_root=work,
            publish=True,
            output_manifest=tmp_path / "out3.parquet",
            summary_xlsx=tmp_path / "sum3.xlsx",
        )
    )
    assert TAG_ID in r3.samples[0].labels["tags"]
    # 来源可区分（字段存在即可；具体结构沿用现有契约）
    assert "tagging" in r3.samples[0].labels


def test_offline_batch_pipeline_outputs_and_explain(tmp_path: Path):
    """正式离线小批：parquet / xlsx / 明细 / 快照 / explain 分支。"""
    cfg = load_tagging_config(LIB_CFG)
    service = TaggingService(cfg, config_path=LIB_CFG)
    samples = [
        _sample_families("需要贷款", "需要贷款", "天气晴朗", sid="batch_a"),
        _sample_families("甲", "乙", "丙", sid="batch_b"),
        _sample_families("需要贷款", "需要贷款", "需要贷款", sid="batch_miss"),
    ]
    work = tmp_path / "batch_work"
    out_pq = tmp_path / "tagged.parquet"
    out_xlsx = tmp_path / "summary.xlsx"
    result = service.run(
        TaggingRunRequest(
            batch_id="offline043",
            config_path=LIB_CFG,
            samples=samples,
            work_root=work,
            publish=True,
            output_manifest=out_pq,
            summary_xlsx=out_xlsx,
        )
    )
    assert out_pq.is_file()
    assert out_xlsx.is_file()
    snap = work / "work" / result.run_id / "definition_snapshot.json"
    assert snap.is_file()
    snap_data = json.loads(snap.read_text(encoding="utf-8"))
    assert TAG_ID in snap_data["tags"]
    assert RULE_A in {r["rule_id"] for r in snap_data["rules"]}
    assert RULE_B in {r["rule_id"] for r in snap_data["rules"]}
    assert "semantic_risk_keywords" in snap_data["lexicons"]
    assert QWEN_EXCLUSIONS in snap_data["lexicons"]
    assert snap_data["content_fingerprint"] == cfg.content_fingerprint

    # 逐规则明细
    details = list((work / "work" / result.run_id).glob("rule_details*.jsonl"))
    assert details, "expected rule_details jsonl"
    lines = details[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 3

    expl_a = service.explain(samples[0], rule_id=RULE_A)
    assert expl_a["rule_results"]
    assert expl_a["rule_results"][0]["ternary"] == Ternary.MATCHED.value
    expl_b = service.explain(samples[1], rule_id=RULE_B)
    assert expl_b["rule_results"][0]["ternary"] == Ternary.MATCHED.value

    by_id = {s.id: s for s in result.samples}
    assert TAG_ID in by_id["batch_a"].labels["tags"]
    assert TAG_ID in by_id["batch_b"].labels["tags"]
    assert TAG_ID not in by_id["batch_miss"].labels["tags"]


def test_fingerprint_drift_on_temp_copies_not_production(tmp_path: Path):
    """临时副本改规则/词表 → 摘要变化；旧进度拒绝续跑。不改真实词表。"""
    lib_copy = tmp_path / "library"
    shutil.copytree(LIBRARY_DIR, lib_copy)
    lex_copy = tmp_path / "lex.yaml"
    shutil.copy(LEXICON_PATH, lex_copy)
    run = yaml.safe_load(LIB_CFG.read_text(encoding="utf-8"))
    run["library"] = {"path": str(lib_copy.as_posix())}
    run["lexicons"] = dict(run["lexicons"])
    run["lexicons"]["semantic_risk_keywords"] = {"path": str(lex_copy.as_posix())}
    run_path = tmp_path / "run.yaml"
    run_path.write_text(yaml.safe_dump(run, allow_unicode=True), encoding="utf-8")
    cfg1 = load_tagging_config(run_path)
    fp1 = cfg1.content_fingerprint

    tag_path = lib_copy / "semantic_risk_text_disagreement.yaml"
    raw = yaml.safe_load(tag_path.read_text(encoding="utf-8"))
    raw["description"] = str(raw.get("description") or "") + "\nchanged043"
    tag_path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    fp2 = load_tagging_config(run_path).content_fingerprint
    assert fp1 != fp2

    lex_raw = yaml.safe_load(lex_copy.read_text(encoding="utf-8"))
    lex_raw["entries"].append(
        {
            "entry_id": "test_only_extra",
            "type": "literal",
            "match_mode": "search",
            "pattern": "仅测试用词条xyz",
        }
    )
    lex_copy.write_text(yaml.safe_dump(lex_raw, allow_unicode=True), encoding="utf-8")
    fp3 = load_tagging_config(run_path).content_fingerprint
    assert fp2 != fp3

    # 生产词表未被改动
    prod = yaml.safe_load(LEXICON_PATH.read_text(encoding="utf-8"))
    assert all(e.get("entry_id") != "test_only_extra" for e in prod["entries"])

    progress = {
        "completed_sample_ids": ["x"],
        "definition_fingerprint": fp1,
    }
    with pytest.raises(TagLibraryError):
        assert_resume_compatible(progress, load_tagging_config(run_path))


def test_view_and_snapshot_include_new_tag():
    cfg = load_tagging_config(LIB_CFG)
    assert TAG_ID in cfg.tags
    assert cfg.tags[TAG_ID].version == "5"
    rules = [r for r in cfg.rules if r.tag_id == TAG_ID]
    assert {r.rule_id for r in rules} == {RULE_A, RULE_B}
    assert {r.rule_id: r.version for r in rules} == {RULE_A: "2", RULE_B: "5"}
    snap = build_definition_snapshot(cfg)
    assert TAG_ID in snap["tags"]
    assert "semantic_risk_keywords" in snap["lexicons"]
    assert snap["lexicons"]["semantic_risk_keywords"].get("entries")
    assert len(snap["lexicons"][QWEN_EXCLUSIONS]["entries"]) == 5


def test_legacy_seven_semantics_unchanged_on_shared_evidence(lib_engine):
    """默认策略选入新 tag 后，旧七项在共享证据上的 rule/tag 语义字段仍成立。"""
    sample = _sample_families("好", "好", "要", sid="legacy7")
    result = lib_engine.evaluate_sample(sample, dry_run=True)
    by_rule = {rr.rule_id: rr for rr in result.rule_results}
    assert by_rule["rule_tag_1_v1"].ternary == Ternary.MATCHED
    assert by_rule["rule_tag_1_v1"].tag_id == "tag_1"
    assert by_rule["rule_critical_short_v1"].ternary == Ternary.MATCHED
    assert RULE_A in by_rule
    assert by_rule[RULE_A].tag_id == TAG_ID
