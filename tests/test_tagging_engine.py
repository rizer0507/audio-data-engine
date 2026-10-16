"""040 multi-tag engine acceptance tests (doc §9)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from audio_engine.core.sample import Sample
from audio_engine.core.tagging.adapters import CountingAcousticAdapter
from audio_engine.core.tagging.config import load_tagging_config
from audio_engine.core.tagging.engine import MultiTagEngine
from audio_engine.core.tagging.evidence.acoustic import AcousticEvidenceView
from audio_engine.core.tagging.service import TaggingRunRequest, TaggingService
from audio_engine.core.tagging.state import (
    SampleTagState,
    TagStateStore,
    apply_manual_remove,
    apply_restore_auto,
    effective_tags,
)
from audio_engine.core.tagging.ternary import ternary_and, ternary_not, ternary_or
from audio_engine.core.tagging.types import Ternary

ROOT = Path(__file__).resolve().parents[1]
TAG_CFG = ROOT / "configs" / "tagging" / "zh_asr_multi_tag_v1.yaml"
KEYS = [
    "glm_1",
    "glm_2",
    "sensevoice_1",
    "sensevoice_2",
    "qwen_1",
    "qwen_2",
]


def _tx(text: str, *, status: str = "success") -> dict:
    return {"text": text, "status": status, "raw_text": text}


def _sample(texts: dict[str, str], *, sid: str = "s1", quality: dict | None = None, failed: set[str] | None = None) -> Sample:
    failed = failed or set()
    transcripts = {}
    for key in KEYS:
        if key in failed:
            transcripts[key] = {"text": "", "status": "failed", "raw_text": ""}
        else:
            transcripts[key] = _tx(texts.get(key, ""))
    return Sample(
        id=sid,
        source_path=f"{sid}.wav",
        sha256="b" * 64,
        duration=2.0,
        transcripts=transcripts,
        quality=dict(quality or {}),
        audio={"resampled_16k": f"{sid}.wav"},
    )


def _tag_cfg():
    return load_tagging_config(TAG_CFG)


def test_config_loads_and_validates():
    cfg = _tag_cfg()
    assert cfg.policy_version == "selection_multi_tag_v1"
    assert "tag_1" in cfg.tags
    assert any(r.rule_id == "rule_tag_1_v1" for r in cfg.enabled_rules())
    assert "tag_1" in cfg.lexicons


def test_ternary_not_unknown_stays_unknown():
    assert ternary_not(Ternary.UNKNOWN) == Ternary.UNKNOWN
    assert ternary_and([Ternary.MATCHED, Ternary.UNKNOWN]) == Ternary.UNKNOWN
    assert ternary_and([Ternary.NOT_MATCHED, Ternary.UNKNOWN]) == Ternary.NOT_MATCHED
    assert ternary_or([Ternary.NOT_MATCHED, Ternary.UNKNOWN]) == Ternary.UNKNOWN
    assert ternary_or([Ternary.MATCHED, Ternary.UNKNOWN]) == Ternary.MATCHED


def test_tag_1_example_and_overlap_critical_short():
    """§9.1 / §9.2: SenseVoice/GLM 好 + Qwen 要 → tag_1；同时可命中 critical_short。"""
    sample = _sample(
        {
            "sensevoice_1": "好",
            "sensevoice_2": "好",
            "glm_1": "好",
            "glm_2": "好",
            "qwen_1": "要",
            "qwen_2": "要",
        }
    )
    engine = MultiTagEngine(_tag_cfg())
    result = engine.evaluate_sample(sample, dry_run=True)
    tags = {rr.tag_id for rr in result.rule_results if rr.ternary == Ternary.MATCHED}
    assert "tag_1" in tags
    assert "critical_short" in tags


def test_tag_1_rejects_hao_a_and_qwen_agree():
    """§9.2: 好啊 非单字；Qwen 也是好 不构成分歧。"""
    engine = MultiTagEngine(_tag_cfg())
    s1 = _sample(
        {
            "sensevoice_1": "好啊",
            "sensevoice_2": "好啊",
            "glm_1": "好啊",
            "glm_2": "好啊",
            "qwen_1": "要",
            "qwen_2": "要",
        }
    )
    r1 = engine.evaluate_sample(s1, dry_run=True)
    assert all(
        rr.ternary != Ternary.MATCHED
        for rr in r1.rule_results
        if rr.tag_id == "tag_1"
    )

    s2 = _sample(
        {
            "sensevoice_1": "好",
            "sensevoice_2": "好",
            "glm_1": "好",
            "glm_2": "好",
            "qwen_1": "好",
            "qwen_2": "好",
        }
    )
    r2 = engine.evaluate_sample(s2, dry_run=True)
    assert all(
        rr.ternary != Ternary.MATCHED
        for rr in r2.rule_results
        if rr.tag_id == "tag_1"
    )


def test_qwen_failure_or_empty_does_not_match_tag_1():
    """§9.3: 失败/稳定空不得经 not(一致) 命中 tag_1。"""
    engine = MultiTagEngine(_tag_cfg())
    base = {
        "sensevoice_1": "好",
        "sensevoice_2": "好",
        "glm_1": "好",
        "glm_2": "好",
    }
    cases = [
        _sample({**base, "qwen_1": "要", "qwen_2": "要"}, failed={"qwen_2"}),
        _sample({**base, "qwen_1": "", "qwen_2": ""}, failed={"qwen_1", "qwen_2"}),
        _sample({**base, "qwen_1": "", "qwen_2": ""}),  # stable_empty
    ]
    for sample in cases:
        result = engine.evaluate_sample(sample, dry_run=True)
        tag1 = next(rr for rr in result.rule_results if rr.tag_id == "tag_1")
        assert tag1.ternary != Ternary.MATCHED
        # stable empty can match diagnostic rule
        if sample.transcripts["qwen_1"]["status"] == "success" and not sample.transcripts["qwen_1"]["text"]:
            empty = next(rr for rr in result.rule_results if rr.tag_id == "qwen_stable_empty_diag")
            assert empty.ternary == Ternary.MATCHED


def test_rule_order_independent():
    """§9.1: 打乱规则顺序标签集合不变。"""
    cfg = _tag_cfg()
    sample = _sample(
        {
            "sensevoice_1": "好",
            "sensevoice_2": "好",
            "glm_1": "好",
            "glm_2": "好",
            "qwen_1": "要",
            "qwen_2": "要",
        }
    )
    engine = MultiTagEngine(cfg)
    r1 = engine.evaluate_sample(sample, dry_run=True)
    tags1 = {rr.tag_id for rr in r1.rule_results if rr.ternary == Ternary.MATCHED}

    cfg.rules = list(reversed(cfg.rules))
    engine2 = MultiTagEngine(cfg)
    r2 = engine2.evaluate_sample(sample, dry_run=True)
    tags2 = {rr.tag_id for rr in r2.rule_results if rr.ternary == Ternary.MATCHED}
    assert tags1 == tags2


def test_nested_at_least_and_invalid_config():
    """§9.4: at_least 可运行；非法配置启动前报错。"""
    from audio_engine.core.tagging.conditions import EvalContext, evaluate_condition
    from audio_engine.core.tagging.evidence.acoustic import acoustic_from_quality
    from audio_engine.core.tagging.evidence.asr_text import build_asr_text_evidence

    cfg = _tag_cfg()
    sample = _sample(
        {
            "sensevoice_1": "好",
            "sensevoice_2": "好",
            "glm_1": "好",
            "glm_2": "好",
            "qwen_1": "要",
            "qwen_2": "要",
        }
    )
    asr = build_asr_text_evidence(sample, cfg.evidence)
    ctx = EvalContext(asr=asr, acoustic=acoustic_from_quality({}), config=cfg)
    result = evaluate_condition(
        {
            "capability": "family_state",
            "families": ["sensevoice", "glm", "qwen"],
            "state": "stable_text",
            "quantifier": "at_least",
            "n": 2,
        },
        ctx,
    )
    assert result.ternary == Ternary.MATCHED

    # Mutate an inline fixture (library-mode production config has no top-level rules list).
    inline_cfg = ROOT / "tests" / "fixtures" / "tagging" / "zh_asr_multi_tag_v1_inline.yaml"
    bad = yaml.safe_load(inline_cfg.read_text(encoding="utf-8"))
    bad["rules"].append(
        {
            "rule_id": "bad_cycle",
            "version": "1",
            "tag_id": "tag_1",
            "condition": {"fragment": "missing_frag"},
        }
    )
    bad_path = ROOT / "configs" / "tagging" / "_tmp_bad_cycle.yaml"
    bad_path.write_text(yaml.safe_dump(bad, allow_unicode=True), encoding="utf-8")
    try:
        with pytest.raises(ValueError):
            load_tagging_config(bad_path)
    finally:
        bad_path.unlink(missing_ok=True)


def test_accuracy_semantics_reuse_family_states():
    """§9.5: 家族四态与无害等价与 selection_v3 一致。"""
    from audio_engine.core.selection_v3.family_evidence import collect_route_views
    from audio_engine.core.selection_v3.five_class_v2 import (
        _harmless_equivalent,
        build_family_states,
    )
    from audio_engine.core.tagging.evidence.asr_text import build_asr_text_evidence

    cfg = _tag_cfg()
    sample = _sample(
        {
            "sensevoice_1": "好",
            "sensevoice_2": "好",
            "glm_1": "好",
            "glm_2": "好",
            "qwen_1": "要",
            "qwen_2": "要",
        }
    )
    routes = collect_route_views(sample, cfg.evidence)
    bundles = build_family_states(routes, cfg.evidence)
    asr = build_asr_text_evidence(sample, cfg.evidence)
    for b in bundles:
        assert asr.family_state_by_name[b.family] == b.state
    assert _harmless_equivalent("好", "好")
    assert not _harmless_equivalent("好", "要")


def test_acoustic_shared_and_isolation(tmp_path: Path):
    """§9.6 / §9.7: 纯文本零调用；共享去重；DNSMOS 失败隔离。"""
    cfg = _tag_cfg()
    from dataclasses import replace

    cfg = replace(cfg, evidence_mode="compute_acoustic")
    adapter = CountingAcousticAdapter(
        results={
            "s_text": AcousticEvidenceView(),
            "s_ac": AcousticEvidenceView(
                vad_status="success",
                vad_speech_state="present",
                dnsmos_status="success",
                dnsmos_sig=1.5,
            ),
        }
    )
    engine = MultiTagEngine(cfg, acoustic_adapter=adapter)

    # Pure text sample: only text rules can match; acoustic rules unknown without compute
    # but planner may still request acoustics for rules that aren't ruled out.
    text_sample = _sample(
        {
            "sensevoice_1": "好",
            "sensevoice_2": "好",
            "glm_1": "好",
            "glm_2": "好",
            "qwen_1": "要",
            "qwen_2": "要",
        },
        sid="s_text",
    )
    # Restrict to pure-text rules only
    r_text = engine.evaluate_sample(
        text_sample,
        dry_run=True,
        rule_ids={"rule_tag_1_v1", "rule_critical_short_v1"},
        compute_acoustics=True,
    )
    assert adapter.call_count() == 0
    tags = {rr.tag_id for rr in r_text.rule_results if rr.ternary == Ternary.MATCHED}
    assert "tag_1" in tags

    # Two acoustic rules share one adapter call
    adapter.calls.clear()
    ac_sample = _sample(
        {
            "sensevoice_1": "x",
            "sensevoice_2": "x",
            "glm_1": "y",
            "glm_2": "y",
            "qwen_1": "z",
            "qwen_2": "z",
        },
        sid="s_ac",
    )
    engine.evaluate_sample(
        ac_sample,
        dry_run=True,
        rule_ids={"rule_human_noise_sig_low_v1", "rule_silence_strict_zero_v1"},
        compute_acoustics=True,
    )
    assert adapter.call_count() == 1
    assert adapter.call_count("vad") == 1
    assert adapter.call_count("dnsmos") == 1

    # DNSMOS failure isolates; text tags still work
    fail_adapter = CountingAcousticAdapter(fail_deps={"dnsmos"})
    engine_fail = MultiTagEngine(cfg, acoustic_adapter=fail_adapter)
    mixed = _sample(
        {
            "sensevoice_1": "好",
            "sensevoice_2": "好",
            "glm_1": "好",
            "glm_2": "好",
            "qwen_1": "要",
            "qwen_2": "要",
        },
        sid="s_mix",
    )
    result = engine_fail.evaluate_sample(mixed, dry_run=True, compute_acoustics=True)
    text_tags = {rr.tag_id for rr in result.rule_results if rr.ternary == Ternary.MATCHED}
    assert "tag_1" in text_tags
    hum = next(rr for rr in result.rule_results if rr.tag_id == "human_noise_sig_low")
    assert hum.ternary == Ternary.UNKNOWN


def test_manual_block_survives_rerun(tmp_path: Path):
    """§9.9 / §9.10: 人工移除持续屏蔽；恢复后重现；多来源去重。"""
    cfg = _tag_cfg()
    service = TaggingService(cfg, config_path=TAG_CFG)
    sample = _sample(
        {
            "sensevoice_1": "好",
            "sensevoice_2": "好",
            "glm_1": "好",
            "glm_2": "好",
            "qwen_1": "要",
            "qwen_2": "要",
        }
    )
    work = tmp_path / "work"
    r1 = service.run(
        TaggingRunRequest(
            batch_id="mini040",
            config_path=TAG_CFG,
            samples=[sample],
            work_root=work,
            publish=True,
            output_manifest=tmp_path / "out1.parquet",
            summary_xlsx=tmp_path / "sum1.xlsx",
        )
    )
    assert "tag_1" in r1.samples[0].labels["tags"]

    service.remove_tag(
        batch_id="mini040",
        sample_id=sample.id,
        tag_id="tag_1",
        audio_sha256=sample.sha256,
        actor="test",
        reason="reject",
        work_root=work,
    )
    r2 = service.run(
        TaggingRunRequest(
            batch_id="mini040",
            config_path=TAG_CFG,
            samples=[sample],
            work_root=work,
            publish=True,
            output_manifest=tmp_path / "out2.parquet",
            summary_xlsx=tmp_path / "sum2.xlsx",
        )
    )
    assert "tag_1" not in r2.samples[0].labels["tags"]

    service.restore_tag(
        batch_id="mini040",
        sample_id=sample.id,
        tag_id="tag_1",
        audio_sha256=sample.sha256,
        actor="test",
        reason="restore",
        work_root=work,
    )
    r3 = service.run(
        TaggingRunRequest(
            batch_id="mini040",
            config_path=TAG_CFG,
            samples=[sample],
            work_root=work,
            publish=True,
            output_manifest=tmp_path / "out3.parquet",
            summary_xlsx=tmp_path / "sum3.xlsx",
        )
    )
    assert "tag_1" in r3.samples[0].labels["tags"]
    # no duplicate tag ids
    assert r3.samples[0].labels["tags"].count("tag_1") == 1


def test_status_conservation_and_export_filter(tmp_path: Path):
    """§9.11: 状态守恒；导出去重。"""
    cfg = _tag_cfg()
    service = TaggingService(cfg, config_path=TAG_CFG)
    samples = [
        _sample(
            {
                "sensevoice_1": "好",
                "sensevoice_2": "好",
                "glm_1": "好",
                "glm_2": "好",
                "qwen_1": "要",
                "qwen_2": "要",
            },
            sid="a",
        ),
        _sample(
            {
                "sensevoice_1": "你好世界",
                "sensevoice_2": "你好世界",
                "glm_1": "你好世界",
                "glm_2": "你好世界",
                "qwen_1": "你好世界",
                "qwen_2": "你好世界",
            },
            sid="b",
        ),
    ]
    result = service.run(
        TaggingRunRequest(
            batch_id="cons040",
            config_path=TAG_CFG,
            samples=samples,
            work_root=tmp_path / "w",
            publish=True,
            output_manifest=tmp_path / "out.parquet",
            summary_xlsx=tmp_path / "sum.xlsx",
        )
    )
    statuses = [s.labels["tagging"]["eval_status"] for s in result.samples]
    assert len(statuses) == len(samples)
    assert statuses.count("tagged") + statuses.count("untagged") + statuses.count("unresolved") + statuses.count(
        "not_evaluated"
    ) == len(samples)

    exported = service.export_filtered(
        result.samples,
        output=tmp_path / "filt.parquet",
        include_all=["tag_1"],
    )
    assert exported["count"] == 1


def test_checkpoint_resume(tmp_path: Path):
    """§9.12: 中断恢复复用进度。"""
    cfg = _tag_cfg()
    service = TaggingService(cfg, config_path=TAG_CFG)
    samples = [
        _sample(
            {
                "sensevoice_1": "好",
                "sensevoice_2": "好",
                "glm_1": "好",
                "glm_2": "好",
                "qwen_1": "要",
                "qwen_2": "要",
            },
            sid="c1",
        ),
        _sample(
            {
                "sensevoice_1": "嗯",
                "sensevoice_2": "嗯",
                "glm_1": "嗯",
                "glm_2": "嗯",
                "qwen_1": "要",
                "qwen_2": "要",
            },
            sid="c2",
        ),
    ]
    work = tmp_path / "resume_w"
    r1 = service.run(
        TaggingRunRequest(
            batch_id="resume040",
            config_path=TAG_CFG,
            samples=samples[:1],
            work_root=work,
            publish=False,
        )
    )
    run_id = r1.run_id
    r2 = service.run(
        TaggingRunRequest(
            batch_id="resume040",
            config_path=TAG_CFG,
            samples=samples,
            work_root=work,
            resume_run_id=run_id,
            publish=True,
            output_manifest=tmp_path / "resume_out.parquet",
            summary_xlsx=tmp_path / "resume_sum.xlsx",
        )
    )
    assert r2.snapshot_id
    assert (tmp_path / "resume_out.parquet").exists()
    assert (tmp_path / "resume_sum.xlsx").exists()


def test_operator_registered():
    from audio_engine.core.registry import OperatorRegistry
    import audio_engine.operators  # noqa: F401

    assert "quality.multi_tag_v1" in OperatorRegistry.list_operators() or OperatorRegistry.get(
        "quality.multi_tag_v1"
    )
