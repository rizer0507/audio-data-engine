"""012-E business metrics: manually accountable numerator/denominator tests."""

from __future__ import annotations

from pathlib import Path

import pytest

import audio_engine.operators  # noqa: F401
from audio_engine.core.sample import Sample
from audio_engine.metrics.business import (
    compute_business_metrics_for_model,
    compute_far,
    compute_faer,
    compute_nahr,
    compute_nfr,
    compute_nshr,
    compute_positive_retention,
    gold_view,
    prediction_view,
)
from audio_engine.metrics.gate import (
    GATE_INCOMPLETE,
    GATE_NEEDS_REVIEW,
    GATE_PASS,
    GateConfig,
    evaluate_release_gate,
)
from audio_engine.metrics.runner import MetricConfigError, MetricRunner, run_text_metrics
from audio_engine.metrics.semantic_judge import load_semantic_judge
from audio_engine.metrics.stat import MetricStat
from audio_engine.core.operator import OperatorConfig
from audio_engine.core.registry import OperatorRegistry


def _sample(
    sid: str,
    *,
    gold_text: str,
    gold_kind: str = "speech",
    human_semantic: str = "unknown",
    hyp: str = "",
    model: str = "qwen",
    audio_event_tags: list[str] | None = None,
    status: str | None = None,
    leakage_group_id: str | None = None,
    eval_role: str = "",
    cer_bits: dict | None = None,
) -> Sample:
    s = Sample(id=sid, source_path=f"{sid}.wav", sha256=f"h-{sid}")
    s.labels["gold_text"] = gold_text
    s.labels["gold_kind"] = gold_kind
    s.labels["human_semantic"] = human_semantic
    if audio_event_tags is not None:
        s.labels["audio_event_tags"] = list(audio_event_tags)
    if leakage_group_id:
        s.labels["leakage_group_id"] = leakage_group_id
    if eval_role:
        s.labels["dataset_role"] = eval_role
    entry: dict = {"text": hyp, "model": model}
    if status:
        entry["status"] = status
    s.transcripts[model] = entry
    if cer_bits:
        for k, v in cer_bits.items():
            s.quality[f"{model}_{k}"] = v
    return s


@pytest.fixture(scope="module")
def judge():
    return load_semantic_judge(
        lexicon_path="configs/selection/semantic_lexicon_zh_v3.yaml"
    )


def test_metric_stat_zero_denominator_is_null():
    stat = MetricStat.from_counts("nfr", numerator=0, denominator=0)
    assert stat.value is None
    assert stat.numerator == 0
    assert stat.denominator == 0


def test_nfr_manual_counts(judge):
    # 3 explicit negatives; 2 predicted positive → NFR = 2/3
    samples = [
        _sample("n1", gold_text="不需要", human_semantic="negative", hyp="需要"),
        _sample("n2", gold_text="不要", human_semantic="negative", hyp="不要"),
        _sample("n3", gold_text="不用", human_semantic="negative", hyp="可以"),
        _sample("p1", gold_text="需要", human_semantic="positive", hyp="需要"),
    ]
    golds = [gold_view(s) for s in samples]
    preds = [prediction_view(s, "qwen", judge) for s in samples]
    stat = compute_nfr(golds, preds, excluded=0)
    assert stat.numerator == 2
    assert stat.denominator == 3
    assert stat.value == pytest.approx(2 / 3)


def test_far_excludes_mixed_unknown(judge):
    samples = [
        _sample("a", gold_text="不需要", human_semantic="negative", hyp="需要"),
        _sample("b", gold_text="嗯", human_semantic="neutral", hyp="需要"),
        _sample(
            "c",
            gold_text="",
            gold_kind="non_speech",
            human_semantic="not_applicable",
            hyp="好的",
            audio_event_tags=["silence"],
        ),
        _sample("d", gold_text="可能吧", human_semantic="unknown", hyp="需要"),
        _sample("e", gold_text="又要又不", human_semantic="mixed", hyp="需要"),
    ]
    golds = [gold_view(s) for s in samples]
    preds = [prediction_view(s, "qwen", judge) for s in samples]
    # den = a,b,c (3); hits = a,b,c all positive → 3/3
    stat = compute_far(golds, preds, excluded=0)
    assert stat.denominator == 3
    assert stat.numerator == 3
    assert stat.value == pytest.approx(1.0)


def test_nahr_and_silence_slice(judge):
    samples = [
        _sample(
            "noise1",
            gold_text="",
            gold_kind="non_speech",
            human_semantic="not_applicable",
            hyp="需要",
            audio_event_tags=["music"],
        ),
        _sample(
            "noise2",
            gold_text="",
            gold_kind="non_speech",
            human_semantic="not_applicable",
            hyp="",
            audio_event_tags=["busy_tone"],
        ),
        _sample(
            "sil1",
            gold_text="",
            gold_kind="non_speech",
            human_semantic="not_applicable",
            hyp="嗯",
            audio_event_tags=["silence"],
        ),
    ]
    golds = [gold_view(s) for s in samples]
    preds = [prediction_view(s, "qwen", judge) for s in samples]
    nahr = compute_nahr(
        golds,
        preds,
        noise_tags=frozenset({"music", "busy_tone"}),
        excluded=0,
    )
    assert nahr.numerator == 1
    assert nahr.denominator == 2
    silence = compute_nahr(
        golds,
        preds,
        noise_tags=frozenset({"silence"}),
        excluded=0,
    )
    # 「嗯」is neutral filler, not confirmed positive
    assert silence.numerator == 0
    assert silence.denominator == 1


def test_faer_rejects_negation_with_filler(judge):
    samples = [
        _sample("f1", gold_text="嗯", human_semantic="neutral", hyp="需要"),
        _sample(
            "f2",
            gold_text="嗯，我不需要",
            human_semantic="neutral",
            hyp="需要",
        ),
        _sample("f3", gold_text="啊", human_semantic="neutral", hyp="啊"),
    ]
    golds = [gold_view(s) for s in samples]
    preds = [prediction_view(s, "qwen", judge) for s in samples]
    stat = compute_faer(golds, preds, judge, excluded=0)
    # only pure fillers f1,f3 in denominator; f1 hit
    assert stat.denominator == 2
    assert stat.numerator == 1


def test_nshr_any_nonempty(judge):
    samples = [
        _sample(
            "s1",
            gold_text="",
            gold_kind="non_speech",
            human_semantic="not_applicable",
            hyp="嘈杂",
            audio_event_tags=["noise"],
        ),
        _sample(
            "s2",
            gold_text="",
            gold_kind="non_speech",
            human_semantic="not_applicable",
            hyp="",
            audio_event_tags=["silence"],
        ),
    ]
    golds = [gold_view(s) for s in samples]
    preds = [prediction_view(s, "qwen", judge) for s in samples]
    stat = compute_nshr(golds, preds, excluded=0)
    assert stat.numerator == 1
    assert stat.denominator == 2


def test_positive_retention_blocks_all_empty_cheat(judge):
    samples = [
        _sample("p1", gold_text="需要", human_semantic="positive", hyp=""),
        _sample("p2", gold_text="可以", human_semantic="positive", hyp="不需要"),
        _sample("p3", gold_text="好的", human_semantic="positive", hyp="好的"),
    ]
    golds = [gold_view(s) for s in samples]
    preds = [prediction_view(s, "qwen", judge) for s in samples]
    stat = compute_positive_retention(golds, preds, excluded=0)
    assert stat.numerator == 1
    assert stat.denominator == 3
    assert stat.value == pytest.approx(1 / 3)


def test_unknown_polarity_not_confirmed_positive(judge):
    s = _sample("u1", gold_text="不需要", human_semantic="negative", hyp="大概吧")
    pred = prediction_view(s, "qwen", judge)
    assert pred.polarity in {"neutral", "unknown", "mixed"}
    assert pred.confirmed_positive is False


def test_inference_failure_incomplete_gate(judge, tmp_path: Path):
    samples = [
        _sample("ok", gold_text="需要", human_semantic="positive", hyp="需要"),
        _sample(
            "miss",
            gold_text="不需要",
            human_semantic="negative",
            hyp="",
            status="failed",
        ),
    ]
    # baseline complete, candidate incomplete
    for s in samples:
        s.transcripts["base"] = {"text": s.get_transcript_text("qwen")}
    samples[1].transcripts["cand"] = {"text": "", "status": "failed"}
    samples[0].transcripts["cand"] = {"text": "需要"}

    result = evaluate_release_gate(
        samples,
        baseline="base",
        candidate="cand",
        gate=GateConfig(
            min_improvement=0.01,
            cer_non_inferiority=0.01,
            positive_retention_non_inferiority=0.01,
            nshr_non_inferiority=0.01,
            require_calibrated_thresholds=True,
            min_denominator=1,
        ),
    )
    assert result.status == GATE_INCOMPLETE


def test_uncalibrated_gate_needs_review():
    samples = [
        _sample("a", gold_text="不需要", human_semantic="negative", hyp="不需要"),
        _sample("b", gold_text="需要", human_semantic="positive", hyp="需要"),
    ]
    for s in samples:
        s.transcripts["base"] = {"text": s.get_transcript_text("qwen")}
        s.transcripts["cand"] = {"text": s.get_transcript_text("qwen")}
    result = evaluate_release_gate(
        samples,
        baseline="base",
        candidate="cand",
        gate=GateConfig(require_calibrated_thresholds=True, min_denominator=1),
    )
    assert result.status == GATE_NEEDS_REVIEW


def test_incomplete_protection_evidence_cannot_pass():
    # Base has high NFR; candidate lower NFR; protection ok
    samples = [
        _sample("n1", gold_text="不需要", human_semantic="negative", hyp="需要", model="base"),
        _sample("n2", gold_text="不要", human_semantic="negative", hyp="需要", model="base"),
        _sample("n3", gold_text="不用", human_semantic="negative", hyp="不用", model="base"),
        _sample("p1", gold_text="需要", human_semantic="positive", hyp="需要", model="base"),
    ]
    for s in samples:
        base_text = s.get_transcript_text("base")
        s.transcripts["base"] = {"text": base_text}
        # candidate: fix the two false positives
        if s.id in {"n1", "n2"}:
            s.transcripts["cand"] = {"text": "不需要"}
        else:
            s.transcripts["cand"] = {"text": base_text}
        # trivial CER fields (speech)
        for m in ("base", "cand"):
            s.quality[f"{m}_substitutions"] = 0
            s.quality[f"{m}_deletions"] = 0
            s.quality[f"{m}_insertions"] = 0
            s.quality[f"{m}_reference_length"] = max(len(s.labels["gold_text"]), 1)
            s.quality[f"{m}_cer"] = 0.0

    result = evaluate_release_gate(
        samples,
        baseline="base",
        candidate="cand",
        gate=GateConfig(
            primary_metric="nfr",
            min_improvement=0.2,
            cer_non_inferiority=0.05,
            positive_retention_non_inferiority=0.05,
            nshr_non_inferiority=0.05,
            min_denominator=2,
            require_calibrated_thresholds=True,
            bootstrap_iterations=20,
        ),
    )
    assert result.status == GATE_NEEDS_REVIEW


def test_metric_runner_rejects_business_as_sample_metric():
    with pytest.raises(MetricConfigError, match="corpus metrics"):
        run_text_metrics(
            {"gold_text": "需要", "qwen_text": "需要"},
            {
                "reference": {"field": "gold_text"},
                "hypothesis": {"field": "qwen_text"},
                "metrics": ["nfr"],
                "output": {"prefix": "qwen"},
            },
            {},
        )


def test_metric_runner_corpus_and_config():
    samples = [
        _sample("n1", gold_text="不需要", human_semantic="negative", hyp="需要"),
        _sample(
            "ns1",
            gold_text="",
            gold_kind="non_speech",
            human_semantic="not_applicable",
            hyp="有",
            audio_event_tags=["music"],
            eval_role="eval_core",
        ),
    ]
    runner = MetricRunner(
        business_config_path="configs/metrics/business_risk_v1.yaml"
    )
    report = runner.score_corpus(samples, ["qwen"])
    assert "qwen" in report["models"]
    metrics = report["models"]["qwen"]["metrics"]
    for name in ("nfr", "far", "nahr", "faer", "nshr", "positive_retention", "cer"):
        assert name in metrics
        assert "numerator" in metrics[name]
        assert "denominator" in metrics[name]
        assert "value" in metrics[name]
        assert "eligible_count" in metrics[name]
        assert "excluded_count" in metrics[name]
    assert metrics["nfr"]["numerator"] == 1
    assert metrics["nfr"]["denominator"] == 1


def test_evaluation_report_writes_business_block(tmp_path: Path):
    samples = [
        _sample(
            "e1",
            gold_text="不需要",
            human_semantic="negative",
            hyp="不需要",
            cer_bits={
                "substitutions": 0,
                "deletions": 0,
                "insertions": 0,
                "reference_length": 3,
                "cer": 0.0,
            },
        ),
        _sample(
            "e2",
            gold_text="需要",
            human_semantic="positive",
            hyp="需要",
            cer_bits={
                "substitutions": 0,
                "deletions": 0,
                "insertions": 0,
                "reference_length": 2,
                "cer": 0.0,
            },
        ),
    ]
    samples[0].labels["type"] = "human_gold"
    samples[1].labels["type"] = "human_gold"
    op = OperatorRegistry.get("quality.evaluation_report")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    op.run(
        samples,
        OperatorConfig(
            step_name="evaluation_report",
            run_dir=run_dir,
            params={
                "model_prefixes": ["qwen"],
                "bucket_key": "type",
                "export_xlsx": False,
                "enable_business_metrics": True,
                "business_metrics_config": "configs/metrics/business_risk_v1.yaml",
                "baseline_prefix": "qwen",
                "candidate_prefix": "qwen",
                "fail_on_regression": False,
            },
        ),
    )
    report_path = run_dir / "reports" / "evaluation.json"
    assert report_path.is_file()
    import json

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "business_metrics" in report
    assert report["publish_status"] in {
        GATE_NEEDS_REVIEW,
        "diagnostic_only",
        GATE_PASS,
        GATE_INCOMPLETE,
        "fail",
    }
    assert "nfr" in report["business_metrics"]["models"]["qwen"]["metrics"]


def test_all_empty_outputs_fail_positive_retention(judge):
    """全输出空不能通过肯定保留率（制造虚假低风险）。"""
    samples = [
        _sample("p1", gold_text="需要", human_semantic="positive", hyp=""),
        _sample("p2", gold_text="可以", human_semantic="positive", hyp=""),
        _sample("n1", gold_text="不需要", human_semantic="negative", hyp=""),
    ]
    block = compute_business_metrics_for_model(samples, "qwen", judge=judge)
    assert block["metrics"]["positive_retention"]["value"] == 0.0
    assert block["metrics"]["nfr"]["value"] == 0.0
    # 低 NFR 但肯定保留率=0 → 不能仅凭风险指标宣称通过
    assert block["metrics"]["positive_retention"]["numerator"] == 0
    assert block["metrics"]["positive_retention"]["denominator"] == 2
