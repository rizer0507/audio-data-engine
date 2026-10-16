"""Offline acceptance coverage for unclear-speech candidates.

The tag uses DNSMOS SIG-low plus cross-family ASR evidence. Silero VAD is not
a membership condition and must not be scheduled by this tag.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from audio_engine.core.sample import Sample
from audio_engine.core.tagging.adapters import CountingAcousticAdapter
from audio_engine.core.tagging.config import load_tagging_config
from audio_engine.core.tagging.engine import MultiTagEngine
from audio_engine.core.tagging.evidence.acoustic import AcousticEvidenceView
from audio_engine.core.tagging.library import build_definition_snapshot
from audio_engine.core.tagging.types import Ternary

ROOT = Path(__file__).resolve().parents[1]
TAG_ID = "qwen_unclear_speech_candidate"
RULE_ID = "rule_qwen_unclear_speech_candidate_v1"
FAMILIES = ("sensevoice", "glm", "qwen")
RUN_CONFIGS = (
    ROOT / "configs/tagging/zh_asr_multi_tag_v1.yaml",
    ROOT / "configs/tagging/zh_asr_free_tags_v2.yaml",
)


def _tx(raw: str, *, status: str = "success", text: str | None = None) -> dict:
    return {"text": raw if text is None else text, "extra": {"raw_text": raw}, "status": status}


def _sample(
    sensevoice: str = "好",
    glm: str = "好",
    qwen: str = "要",
    *,
    quality: dict | None = None,
    overrides: dict[str, dict | None] | None = None,
) -> Sample:
    transcripts = {
        f"{family}_{run}": _tx(raw)
        for family, raw in zip(FAMILIES, (sensevoice, glm, qwen))
        for run in (1, 2)
    }
    for key, entry in (overrides or {}).items():
        if entry is None:
            transcripts.pop(key, None)
        else:
            transcripts[key] = dict(entry)
    return Sample(
        id="unclear_candidate",
        source_path="not_a_real_audio_file.wav",
        sha256="b" * 64,
        duration=2.0,
        transcripts=transcripts,
        quality=(
            {"dnsmos_status": "success", "dnsmos_sig": 1.5}
            if quality is None
            else dict(quality)
        ),
    )


@pytest.fixture(scope="module", params=RUN_CONFIGS, ids=lambda path: path.stem)
def config(request):
    return load_tagging_config(request.param)


def _rule_result(config, sample: Sample):
    result = MultiTagEngine(config).evaluate_sample(
        sample, dry_run=True, rule_ids={RULE_ID}
    )
    assert len(result.rule_results) == 1
    return result.rule_results[0]


@pytest.mark.parametrize(
    ("texts", "expected"),
    [
        (("好", "好", "要"), Ternary.MATCHED),
        (("好", "要", "是"), Ternary.MATCHED),
        (("好", "好", "好"), Ternary.NOT_MATCHED),
        (("好！", "好。", "好，"), Ternary.NOT_MATCHED),
        (("", "", "要"), Ternary.MATCHED),
    ],
    ids=["two_agree_one_differs", "all_differ", "all_agree", "punctuation", "success_empty"],
)
def test_cross_family_disagreement_includes_successful_empty(config, texts, expected):
    assert _rule_result(config, _sample(*texts)).ternary == expected


@pytest.mark.parametrize("family", FAMILIES)
def test_every_successful_route_counts_even_if_family_dual_run_is_unstable(config, family):
    # The only difference is on run 2; choosing only run 1/a stable representative
    # would miss it, whether the unstable family is Qwen or either other model.
    sample = _sample("好", "好", "好", overrides={f"{family}_2": _tx("要")})
    assert MultiTagEngine(config).build_asr(sample).family_state_by_name[family] == "unstable"
    assert _rule_result(config, sample).ternary == Ternary.MATCHED


def test_one_qwen_success_with_text_is_enough_despite_other_run_failure(config):
    sample = _sample(overrides={"qwen_2": _tx("失败残留", status="failed")})
    assert _rule_result(config, sample).ternary == Ternary.MATCHED


@pytest.mark.parametrize("qwen_raw", ["", " \t\n "])
def test_successful_qwen_with_only_blank_raw_text_is_not_a_candidate(config, qwen_raw):
    assert _rule_result(config, _sample(qwen=qwen_raw)).ternary == Ternary.NOT_MATCHED


def test_all_qwen_failures_with_residual_text_remain_unknown(config):
    sample = _sample(
        overrides={f"qwen_{run}": _tx("要", status="failed") for run in (1, 2)}
    )
    assert _rule_result(config, sample).ternary == Ternary.UNKNOWN


def test_failed_qwen_residual_text_does_not_make_successful_empty_qwen_nonempty(config):
    sample = _sample(qwen="", overrides={"qwen_2": _tx("要", status="failed")})
    assert _rule_result(config, sample).ternary == Ternary.NOT_MATCHED


def test_qwen_original_output_counts_when_cleaning_removes_all_characters(config):
    sample = _sample(qwen="？！", overrides={"qwen_1": _tx("？！", text="")})
    qwen_routes = [r for r in MultiTagEngine(config).build_asr(sample).routes if r.family == "qwen"]
    assert all(r.raw_text == "？！" and r.comparison_text == "" for r in qwen_routes)
    assert _rule_result(config, sample).ternary == Ternary.MATCHED


def test_nonempty_qwen_raw_alone_cannot_create_disagreement_when_all_comparison_text_is_empty(config):
    assert _rule_result(config, _sample("", "", "？！")).ternary == Ternary.NOT_MATCHED


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("route_kind", ["failed", "missing", "excluded"])
def test_a_whole_unavailable_family_is_unknown_even_when_other_families_differ(
    config, family, route_kind
):
    if route_kind == "missing":
        entry = None
    elif route_kind == "excluded":
        entry = _tx("hello")
    else:
        entry = _tx("残留文本", status="failed")
    sample = _sample("好", "要", "是", overrides={f"{family}_{run}": entry for run in (1, 2)})
    assert _rule_result(config, sample).ternary == Ternary.UNKNOWN


@pytest.mark.parametrize(
    "bad_entry",
    [_tx("", status="failed"), None, _tx("hello")],
    ids=["failed", "missing", "excluded"],
)
def test_unavailable_routes_do_not_vote_as_empty_when_another_route_is_available(config, bad_entry):
    sample = _sample("好", "好", "好", overrides={"sensevoice_2": bad_entry})
    assert _rule_result(config, sample).ternary == Ternary.NOT_MATCHED


@pytest.mark.parametrize(
    ("quality", "expected"),
    [
        ({}, Ternary.UNKNOWN),
        ({"dnsmos_status": "failed"}, Ternary.UNKNOWN),
        ({"dnsmos_status": "success", "dnsmos_sig": 2.9}, Ternary.NOT_MATCHED),
        ({"dnsmos_status": "success", "dnsmos_sig": 2.8}, Ternary.MATCHED),
        ({"dnsmos_status": "success", "dnsmos_sig": 1.5}, Ternary.MATCHED),
    ],
    ids=["missing_sig", "failed_sig", "high_sig", "threshold_sig", "low_sig"],
)
def test_dnsmos_sig_low_is_required(config, quality, expected):
    assert _rule_result(config, _sample(quality=quality)).ternary == expected


def test_vad_is_ignored_for_membership(config):
    sample = _sample(
        quality={
            "dnsmos_status": "success",
            "dnsmos_sig": 1.5,
            "vad_status": "success",
            "vad_speech_state": "absent",
        }
    )
    assert _rule_result(config, sample).ternary == Ternary.MATCHED


def test_candidate_plans_and_computes_only_dnsmos(config):
    adapter = CountingAcousticAdapter(
        default=AcousticEvidenceView(dnsmos_status="success", dnsmos_sig=1.5)
    )
    engine = MultiTagEngine(replace(config, evidence_mode="compute_acoustic"), acoustic_adapter=adapter)
    result = engine.evaluate_sample(_sample(quality={}), dry_run=True, rule_ids={RULE_ID})
    assert result.rule_results[0].ternary == Ternary.MATCHED
    assert result.rule_results[0].acoustic_plan == ["dnsmos"]
    assert result.acoustic_plan.required == {"dnsmos"}
    assert adapter.call_count() == 1
    assert adapter.calls[0]["deps"] == ["dnsmos"]
    assert engine.stats.actual_calls == {"vad": 0, "energy": 0, "dnsmos": 1}


def test_text_that_cannot_match_does_not_schedule_dnsmos(config):
    adapter = CountingAcousticAdapter()
    engine = MultiTagEngine(replace(config, evidence_mode="compute_acoustic"), acoustic_adapter=adapter)
    result = engine.evaluate_sample(
        _sample("好", "好", "好", quality={}), dry_run=True, rule_ids={RULE_ID}
    )
    assert result.rule_results[0].ternary == Ternary.NOT_MATCHED
    assert not result.acoustic_plan.required
    assert adapter.call_count() == 0


def test_candidate_is_in_both_run_configs_and_can_overlap_existing_tags(config):
    assert config.tags[TAG_ID].version == "2"
    rules = [r for r in config.rules if r.tag_id == TAG_ID]
    assert [r.rule_id for r in rules] == [RULE_ID]
    assert rules[0].version == "2"
    assert TAG_ID in build_definition_snapshot(config)["tags"]
    sample = _sample(
        quality={
            "vad_status": "success",
            "vad_speech_state": "present",
            "dnsmos_status": "success",
            "dnsmos_sig": 1.5,
        }
    )
    result = MultiTagEngine(config).evaluate_sample(sample, snapshot_id="offline_test")
    assert {TAG_ID, "tag_1", "critical_short", "human_noise_sig_low"}.issubset(result.effective_tags)
    assert result.effective_tags.count(TAG_ID) == 1
