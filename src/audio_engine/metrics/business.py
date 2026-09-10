"""Business risk metrics (NFR/FAR/NAHR/FAER/NSHR/…) for MetricRunner."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3.types import (
    POLARITY_MIXED,
    POLARITY_NEGATIVE,
    POLARITY_NEUTRAL,
    POLARITY_POSITIVE,
    POLARITY_UNKNOWN,
    RUN_STATUS_FAILED,
    RUN_STATUS_MISSING,
    RUN_STATUS_SUCCESS_EMPTY,
    RUN_STATUS_SUCCESS_TEXT,
)
from audio_engine.metrics.semantic_judge import SemanticJudge, load_semantic_judge
from audio_engine.metrics.stat import MetricStat

BUSINESS_METRICS_VERSION = "business_metrics_v1.0"
GOLD_KIND_SPEECH = "speech"
GOLD_KIND_NON_SPEECH = "non_speech"

# Default audio-event tags for NAHR noise vs pure silence slices.
DEFAULT_NOISE_EVENT_TAGS = frozenset(
    {
        "noise",
        "env_noise",
        "environment_noise",
        "ambient",
        "busy",
        "busy_tone",
        "music",
        "background_music",
    }
)
DEFAULT_SILENCE_EVENT_TAGS = frozenset({"silence", "true_silence", "quiet"})


@dataclass(frozen=True)
class GoldView:
    sample_id: str
    gold_kind: str
    human_semantic: str
    gold_text: str
    audio_event_tags: frozenset[str]
    leakage_group_id: str
    eval_role: str  # eval_core | eval_random | "" | other


@dataclass(frozen=True)
class PredictionView:
    sample_id: str
    model: str
    status: str  # success_text | success_empty | failed | missing
    text: str
    polarity: str
    confirmed_positive: bool


@dataclass
class BusinessMetricConfig:
    version: str = BUSINESS_METRICS_VERSION
    lexicon_path: str = "configs/selection/semantic_lexicon_zh_v3.yaml"
    judge_version: str = "business_semantic_judge_v1.0"
    noise_event_tags: frozenset[str] = field(default_factory=lambda: DEFAULT_NOISE_EVENT_TAGS)
    silence_event_tags: frozenset[str] = field(
        default_factory=lambda: DEFAULT_SILENCE_EVENT_TAGS
    )
    require_complete_predictions: bool = True
    max_semantic_unk_rate: float | None = None
    min_denominator: int = 1


def load_business_metric_config(raw: dict[str, Any] | None) -> BusinessMetricConfig:
    data = dict(raw or {})
    noise = data.get("noise_event_tags")
    silence = data.get("silence_event_tags")
    return BusinessMetricConfig(
        version=str(data.get("version") or BUSINESS_METRICS_VERSION),
        lexicon_path=str(
            data.get("lexicon_path") or "configs/selection/semantic_lexicon_zh_v3.yaml"
        ),
        judge_version=str(
            data.get("judge_version") or "business_semantic_judge_v1.0"
        ),
        noise_event_tags=frozenset(str(x) for x in (noise or DEFAULT_NOISE_EVENT_TAGS)),
        silence_event_tags=frozenset(
            str(x) for x in (silence or DEFAULT_SILENCE_EVENT_TAGS)
        ),
        require_complete_predictions=bool(
            data.get("require_complete_predictions", True)
        ),
        max_semantic_unk_rate=(
            float(data["max_semantic_unk_rate"])
            if data.get("max_semantic_unk_rate") is not None
            else None
        ),
        min_denominator=int(data.get("min_denominator", 1)),
    )


def _as_tags(value: Any) -> frozenset[str]:
    if value is None:
        return frozenset()
    if isinstance(value, (list, tuple, set, frozenset)):
        return frozenset(str(x).strip() for x in value if str(x).strip())
    text = str(value).strip()
    if not text:
        return frozenset()
    if "," in text:
        return frozenset(part.strip() for part in text.split(",") if part.strip())
    if "|" in text:
        return frozenset(part.strip() for part in text.split("|") if part.strip())
    return frozenset({text})


def _gold_text(sample: Sample) -> str:
    text = sample.labels.get("gold_text")
    if text is None:
        text = sample.labels.get("label")
    if text is None:
        text = sample.get_transcript_text("gold")
    if text is None:
        return ""
    return str(text)


def _gold_kind(sample: Sample) -> str:
    kind = str(sample.labels.get("gold_kind") or "").strip().lower()
    if kind:
        return kind
    # Legacy empty-ref buckets → non_speech
    bucket = str(
        sample.labels.get("type")
        or sample.labels.get("classification_bucket")
        or sample.labels.get("subtype")
        or ""
    ).strip()
    if bucket in {"true_silence", "invalid_audio", "noise", "empty_gold", "non_speech"}:
        return GOLD_KIND_NON_SPEECH
    if str(_gold_text(sample)).strip() == "" and sample.labels.get("gold_text") == "":
        return GOLD_KIND_NON_SPEECH
    return GOLD_KIND_SPEECH


def _human_semantic(sample: Sample) -> str:
    value = sample.labels.get("human_semantic")
    if value is None or not str(value).strip():
        value = sample.labels.get("semantic") or sample.labels.get("polarity")
    return str(value or "").strip().lower() or POLARITY_UNKNOWN


def _leakage_group(sample: Sample) -> str:
    for key in (
        "leakage_group_id",
        "call_group_id",
        "source_group_id",
        "group_id",
    ):
        value = sample.labels.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return sample.id


def _eval_role(sample: Sample) -> str:
    for key in ("dataset_role", "eval_role", "split", "reservation_role"):
        value = str(sample.labels.get(key) or "").strip().lower()
        if value in {"eval_core", "eval_random"}:
            return value
    return ""


def gold_view(sample: Sample) -> GoldView:
    return GoldView(
        sample_id=sample.id,
        gold_kind=_gold_kind(sample),
        human_semantic=_human_semantic(sample),
        gold_text=_gold_text(sample),
        audio_event_tags=_as_tags(sample.labels.get("audio_event_tags")),
        leakage_group_id=_leakage_group(sample),
        eval_role=_eval_role(sample),
    )


def _transcript_entry(sample: Sample, model: str) -> dict[str, Any] | None:
    entry = sample.transcripts.get(model)
    if entry is None:
        return None
    if isinstance(entry, dict):
        return entry
    return {"text": str(entry)}


def prediction_status(sample: Sample, model: str) -> str:
    """Map one model prediction to success_text|success_empty|failed|missing."""
    entry = _transcript_entry(sample, model)
    if entry is None:
        # quality flag from aggregate / text_metrics
        if sample.quality.get(f"{model}_prediction_status"):
            return str(sample.quality[f"{model}_prediction_status"])
        if sample.quality.get(f"{model}_missing") is True:
            return RUN_STATUS_MISSING
        if sample.quality.get(f"{model}_failed") is True:
            return RUN_STATUS_FAILED
        return RUN_STATUS_MISSING
    status = str(entry.get("status") or "").strip().lower()
    if status in {"failed", "error", "fail"}:
        return RUN_STATUS_FAILED
    if status in {"missing", "pending", "running"}:
        return RUN_STATUS_MISSING
    extra = entry.get("extra") if isinstance(entry.get("extra"), dict) else {}
    inf = str(extra.get("inference_status") or extra.get("status") or "").strip().lower()
    if inf in {"failed", "error", "fail"}:
        return RUN_STATUS_FAILED
    if inf in {"missing"}:
        return RUN_STATUS_MISSING
    if entry.get("failed") is True:
        return RUN_STATUS_FAILED
    text = entry.get("text")
    if text is None:
        return RUN_STATUS_MISSING
    from audio_engine.core.selection_v3.text import raw_transcript_text
    if raw_transcript_text(entry).strip() == "":
        return RUN_STATUS_SUCCESS_EMPTY
    return RUN_STATUS_SUCCESS_TEXT


def prediction_view(sample: Sample, model: str, judge: SemanticJudge) -> PredictionView:
    status = prediction_status(sample, model)
    from audio_engine.core.selection_v3.text import raw_transcript_text
    text = raw_transcript_text(_transcript_entry(sample, model)) if status != RUN_STATUS_MISSING else ""
    if status in {RUN_STATUS_FAILED, RUN_STATUS_MISSING}:
        polarity = POLARITY_UNKNOWN
        confirmed = False
    elif status == RUN_STATUS_SUCCESS_EMPTY:
        polarity = POLARITY_UNKNOWN
        confirmed = False
    else:
        polarity = judge.polarity(text)
        confirmed = polarity == POLARITY_POSITIVE
    return PredictionView(
        sample_id=sample.id,
        model=model,
        status=status,
        text=str(text or ""),
        polarity=polarity,
        confirmed_positive=confirmed,
    )


def _rate(
    name: str,
    hits: Sequence[bool],
    *,
    excluded: int = 0,
    extras: dict[str, Any] | None = None,
) -> MetricStat:
    den = len(hits)
    num = sum(1 for item in hits if item)
    return MetricStat.from_counts(
        name,
        numerator=num,
        denominator=den,
        eligible_count=den,
        excluded_count=excluded,
        extras=extras,
    )


def compute_cer_speech(
    samples: Sequence[Sample],
    model: str,
    *,
    excluded_non_speech: int = 0,
) -> MetricStat:
    """CER only on gold_kind=speech scored samples; zero den → null."""
    subs = dele = ins = ref_len = 0
    eligible = 0
    for sample in samples:
        gold = gold_view(sample)
        if gold.gold_kind != GOLD_KIND_SPEECH:
            continue
        if prediction_status(sample, model) not in {RUN_STATUS_SUCCESS_TEXT, RUN_STATUS_SUCCESS_EMPTY}:
            continue
        from audio_engine.metrics.align import align_characters
        from audio_engine.metrics.normalization import normalize_text
        from audio_engine.core.selection_v3.text import raw_transcript_text
        reference = normalize_text(gold.gold_text)
        hypothesis = normalize_text(raw_transcript_text(_transcript_entry(sample, model)))
        eligible += 1
        ops = align_characters(reference, hypothesis)
        subs += sum(op["operation"] == "substitution" for op in ops)
        dele += sum(op["operation"] == "deletion" for op in ops)
        ins += sum(op["operation"] == "insertion" for op in ops)
        ref_len += len(reference)
    errors = subs + dele + ins
    return MetricStat.from_counts(
        "cer",
        numerator=errors,
        denominator=ref_len,
        eligible_count=eligible,
        excluded_count=excluded_non_speech,
        extras={
            "substitutions": subs,
            "deletions": dele,
            "insertions": ins,
            "substitution_rate": None if ref_len <= 0 else round(subs / ref_len, 6),
            "deletion_rate": None if ref_len <= 0 else round(dele / ref_len, 6),
            "insertion_rate": None if ref_len <= 0 else round(ins / ref_len, 6),
        },
    )


def compute_nfr(
    golds: Sequence[GoldView], preds: Sequence[PredictionView], *, excluded: int
) -> MetricStat:
    hits = [
        pred.confirmed_positive
        for gold, pred in zip(golds, preds)
        if gold.human_semantic == POLARITY_NEGATIVE
        and pred.status in {RUN_STATUS_SUCCESS_TEXT, RUN_STATUS_SUCCESS_EMPTY}
    ]
    return _rate("nfr", hits, excluded=excluded)


def compute_far(
    golds: Sequence[GoldView], preds: Sequence[PredictionView], *, excluded: int
) -> MetricStat:
    """FAR denominator: explicit negative / neutral / confirmed non_speech; no mixed/unknown."""
    hits: list[bool] = []
    for gold, pred in zip(golds, preds):
        if pred.status not in {RUN_STATUS_SUCCESS_TEXT, RUN_STATUS_SUCCESS_EMPTY}:
            continue
        if gold.human_semantic in {POLARITY_MIXED, POLARITY_UNKNOWN}:
            continue
        if gold.gold_kind == GOLD_KIND_NON_SPEECH:
            hits.append(pred.confirmed_positive)
            continue
        if gold.human_semantic in {POLARITY_NEGATIVE, POLARITY_NEUTRAL}:
            hits.append(pred.confirmed_positive)
    return _rate("far", hits, excluded=excluded)


def compute_nahr(
    golds: Sequence[GoldView],
    preds: Sequence[PredictionView],
    *,
    noise_tags: frozenset[str],
    excluded: int,
) -> MetricStat:
    hits = []
    for gold, pred in zip(golds, preds):
        if gold.gold_kind != GOLD_KIND_NON_SPEECH:
            continue
        if not (gold.audio_event_tags & noise_tags):
            continue
        if pred.status not in {RUN_STATUS_SUCCESS_TEXT, RUN_STATUS_SUCCESS_EMPTY}:
            continue
        hits.append(pred.confirmed_positive)
    return _rate("nahr", hits, excluded=excluded)


def compute_nahr_silence(
    golds: Sequence[GoldView],
    preds: Sequence[PredictionView],
    *,
    silence_tags: frozenset[str],
    excluded: int,
) -> MetricStat:
    hits = []
    for gold, pred in zip(golds, preds):
        if gold.gold_kind != GOLD_KIND_NON_SPEECH:
            continue
        if not (gold.audio_event_tags & silence_tags):
            continue
        if pred.status not in {RUN_STATUS_SUCCESS_TEXT, RUN_STATUS_SUCCESS_EMPTY}:
            continue
        hits.append(pred.confirmed_positive)
    return _rate("nahr_silence", hits, excluded=excluded)


def compute_faer(
    golds: Sequence[GoldView],
    preds: Sequence[PredictionView],
    judge: SemanticJudge,
    *,
    excluded: int,
) -> MetricStat:
    hits = []
    for gold, pred in zip(golds, preds):
        if pred.status not in {RUN_STATUS_SUCCESS_TEXT, RUN_STATUS_SUCCESS_EMPTY}:
            continue
        # Human-confirmed pure filler / neutral short feedback
        if gold.human_semantic != POLARITY_NEUTRAL:
            continue
        if gold.gold_kind == GOLD_KIND_NON_SPEECH:
            continue
        if not judge.is_pure_filler(gold.gold_text):
            continue
        hits.append(pred.confirmed_positive)
    return _rate("faer", hits, excluded=excluded)


def compute_nshr(
    golds: Sequence[GoldView], preds: Sequence[PredictionView], *, excluded: int
) -> MetricStat:
    hits = []
    for gold, pred in zip(golds, preds):
        if gold.gold_kind != GOLD_KIND_NON_SPEECH:
            continue
        if pred.status not in {RUN_STATUS_SUCCESS_TEXT, RUN_STATUS_SUCCESS_EMPTY}:
            continue
        hits.append(bool(str(pred.text or "").strip()))
    return _rate("nshr", hits, excluded=excluded)


def compute_positive_retention(
    golds: Sequence[GoldView], preds: Sequence[PredictionView], *, excluded: int
) -> MetricStat:
    hits = []
    for gold, pred in zip(golds, preds):
        if gold.human_semantic != POLARITY_POSITIVE:
            continue
        if pred.status not in {RUN_STATUS_SUCCESS_TEXT, RUN_STATUS_SUCCESS_EMPTY}:
            continue
        hits.append(pred.confirmed_positive)
    return _rate("positive_retention", hits, excluded=excluded)


def compute_inference_failure_rate(
    preds: Sequence[PredictionView], *, expected: int
) -> MetricStat:
    failed = sum(
        1 for pred in preds if pred.status in {RUN_STATUS_FAILED, RUN_STATUS_MISSING}
    )
    return MetricStat.from_counts(
        "inference_failure_rate",
        numerator=failed,
        denominator=expected,
        eligible_count=expected,
        excluded_count=0,
        extras={
            "failed_or_missing": failed,
            "success_empty": sum(
                1 for pred in preds if pred.status == RUN_STATUS_SUCCESS_EMPTY
            ),
            "success_text": sum(
                1 for pred in preds if pred.status == RUN_STATUS_SUCCESS_TEXT
            ),
        },
    )


def compute_semantic_unk_rate(
    golds: Sequence[GoldView], preds: Sequence[PredictionView]
) -> MetricStat:
    """Share of successful predictions whose polarity is mixed/unknown."""
    scored = [
        pred
        for gold, pred in zip(golds, preds)
        if pred.status in {RUN_STATUS_SUCCESS_TEXT, RUN_STATUS_SUCCESS_EMPTY}
    ]
    unk = sum(1 for pred in scored if pred.polarity in {POLARITY_MIXED, POLARITY_UNKNOWN})
    return MetricStat.from_counts(
        "semantic_unk_rate",
        numerator=unk,
        denominator=len(scored),
        eligible_count=len(scored),
        excluded_count=0,
    )


def compute_semantic_unk_as_risk_upper(
    golds: Sequence[GoldView], preds: Sequence[PredictionView]
) -> MetricStat:
    """Conservative upper bound: treat unknown/mixed predictions as risk hits on FAR pool."""
    hits: list[bool] = []
    for gold, pred in zip(golds, preds):
        if pred.status not in {RUN_STATUS_SUCCESS_TEXT, RUN_STATUS_SUCCESS_EMPTY}:
            continue
        if gold.human_semantic in {POLARITY_MIXED, POLARITY_UNKNOWN}:
            continue
        eligible = gold.gold_kind == GOLD_KIND_NON_SPEECH or gold.human_semantic in {
            POLARITY_NEGATIVE,
            POLARITY_NEUTRAL,
        }
        if not eligible:
            continue
        risk = pred.confirmed_positive or pred.polarity in {
            POLARITY_MIXED,
            POLARITY_UNKNOWN,
        }
        hits.append(risk)
    return _rate("far_unk_as_risk_upper", hits)


def prediction_completeness(
    samples: Sequence[Sample], model: str
) -> dict[str, Any]:
    statuses = [prediction_status(s, model) for s in samples]
    incomplete_ids = [
        s.id
        for s, st in zip(samples, statuses)
        if st in {RUN_STATUS_FAILED, RUN_STATUS_MISSING}
    ]
    return {
        "expected": len(samples),
        "complete": len(samples) - len(incomplete_ids),
        "incomplete": len(incomplete_ids),
        "incomplete_ids": incomplete_ids[:200],
        "incomplete_truncated": len(incomplete_ids) > 200,
        "is_complete": len(incomplete_ids) == 0,
    }


def compute_business_metrics_for_model(
    samples: Sequence[Sample],
    model: str,
    *,
    judge: SemanticJudge | None = None,
    config: BusinessMetricConfig | None = None,
) -> dict[str, Any]:
    cfg = config or BusinessMetricConfig()
    judge = judge or load_semantic_judge(
        lexicon_path=cfg.lexicon_path, judge_version=cfg.judge_version
    )
    golds = [gold_view(s) for s in samples]
    preds = [prediction_view(s, model, judge) for s in samples]
    completeness = prediction_completeness(samples, model)
    excluded_incomplete = int(completeness["incomplete"])
    non_speech = sum(1 for g in golds if g.gold_kind == GOLD_KIND_NON_SPEECH)

    metrics = {
        "cer": compute_cer_speech(samples, model, excluded_non_speech=non_speech),
        "nfr": compute_nfr(golds, preds, excluded=excluded_incomplete),
        "far": compute_far(golds, preds, excluded=excluded_incomplete),
        "nahr": compute_nahr(
            golds, preds, noise_tags=cfg.noise_event_tags, excluded=excluded_incomplete
        ),
        "nahr_silence": compute_nahr_silence(
            golds,
            preds,
            silence_tags=cfg.silence_event_tags,
            excluded=excluded_incomplete,
        ),
        "faer": compute_faer(golds, preds, judge, excluded=excluded_incomplete),
        "nshr": compute_nshr(golds, preds, excluded=excluded_incomplete),
        "positive_retention": compute_positive_retention(
            golds, preds, excluded=excluded_incomplete
        ),
        "inference_failure_rate": compute_inference_failure_rate(
            preds, expected=len(samples)
        ),
        "semantic_unk_rate": compute_semantic_unk_rate(golds, preds),
        "far_unk_as_risk_upper": compute_semantic_unk_as_risk_upper(golds, preds),
    }
    return {
        "model": model,
        "metrics_version": cfg.version,
        "judge": judge.to_meta(),
        "prediction_completeness": completeness,
        "metrics": {name: stat.to_dict() for name, stat in metrics.items()},
        "metric_objects": metrics,
    }


def filter_by_eval_role(
    samples: Sequence[Sample], role: str | None
) -> list[Sample]:
    if not role:
        return list(samples)
    return [s for s in samples if _eval_role(s) == role or role == "all"]


def slice_samples(
    samples: Sequence[Sample],
    *,
    human_semantic: str | None = None,
    gold_kind: str | None = None,
    short_max_chars: int | None = None,
    noise_only: bool = False,
    crosstalk: bool | None = None,
    source_key: str | None = None,
    source_value: str | None = None,
    noise_tags: frozenset[str] = DEFAULT_NOISE_EVENT_TAGS,
) -> list[Sample]:
    out: list[Sample] = []
    for sample in samples:
        gold = gold_view(sample)
        if human_semantic and gold.human_semantic != human_semantic:
            continue
        if gold_kind and gold.gold_kind != gold_kind:
            continue
        if short_max_chars is not None and len(gold.gold_text) > short_max_chars:
            continue
        if noise_only and not (gold.audio_event_tags & noise_tags):
            continue
        if crosstalk is not None:
            raw = str(sample.labels.get("human_crosstalk") or sample.labels.get("crosstalk") or "").strip().lower()
            is_ct = raw in {"true", "1", "yes", "y"}
            if is_ct != crosstalk:
                continue
        if source_key and source_value is not None:
            if str(sample.labels.get(source_key) or "").strip() != source_value:
                continue
        out.append(sample)
    return out


def group_bootstrap_metric_delta(
    samples: Sequence[Sample],
    baseline: str,
    candidate: str,
    metric_name: str,
    *,
    judge: SemanticJudge,
    config: BusinessMetricConfig,
    iterations: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Paired bootstrap by leakage_group to avoid multi-segment inflation."""
    groups: dict[str, list[Sample]] = {}
    for sample in samples:
        groups.setdefault(_leakage_group(sample), []).append(sample)
    group_ids = sorted(groups.keys())
    if not group_ids or iterations <= 0:
        return {"iterations": 0, "seed": seed, "ci": [], "confidence": confidence}

    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(iterations):
        drawn: list[Sample] = []
        for _gid in group_ids:
            pick = group_ids[rng.randrange(len(group_ids))]
            drawn.extend(groups[pick])
        base = compute_business_metrics_for_model(
            drawn, baseline, judge=judge, config=config
        )
        cand = compute_business_metrics_for_model(
            drawn, candidate, judge=judge, config=config
        )
        bv = base["metrics"].get(metric_name, {}).get("value")
        cv = cand["metrics"].get(metric_name, {}).get("value")
        if bv is None or cv is None:
            continue
        # For risk metrics (lower better) and CER: candidate - baseline
        # For positive_retention (higher better): baseline - candidate (so positive delta = regression)
        if metric_name == "positive_retention":
            deltas.append(float(bv) - float(cv))
        else:
            deltas.append(float(cv) - float(bv))
    if not deltas:
        return {
            "iterations": iterations,
            "seed": seed,
            "ci": [],
            "confidence": confidence,
            "n_valid": 0,
        }
    deltas.sort()
    alpha = 1.0 - confidence
    lo_i = int(alpha / 2 * (len(deltas) - 1))
    hi_i = int((1 - alpha / 2) * (len(deltas) - 1))
    return {
        "iterations": iterations,
        "seed": seed,
        "confidence": confidence,
        "n_valid": len(deltas),
        "mean": round(sum(deltas) / len(deltas), 6),
        "ci": [deltas[lo_i], deltas[hi_i]],
        "group_count": len(group_ids),
    }


def aggregate_business_report(
    samples: Sequence[Sample],
    models: Sequence[str],
    *,
    config: BusinessMetricConfig | None = None,
    judge: SemanticJudge | None = None,
    slices: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cfg = config or BusinessMetricConfig()
    judge = judge or load_semantic_judge(
        lexicon_path=cfg.lexicon_path, judge_version=cfg.judge_version
    )
    report: dict[str, Any] = {
        "metrics_version": cfg.version,
        "judge": judge.to_meta(),
        "models": {},
        "by_eval_role": {},
        "slices": {},
    }
    for model in models:
        report["models"][model] = compute_business_metrics_for_model(
            samples, model, judge=judge, config=cfg
        )
        # Drop non-serializable metric_objects from top-level export copy
        report["models"][model] = {
            k: v for k, v in report["models"][model].items() if k != "metric_objects"
        }

    for role in ("eval_core", "eval_random"):
        subset = [s for s in samples if _eval_role(s) == role]
        if not subset:
            continue
        role_block: dict[str, Any] = {}
        for model in models:
            payload = compute_business_metrics_for_model(
                subset, model, judge=judge, config=cfg
            )
            role_block[model] = {
                k: v for k, v in payload.items() if k != "metric_objects"
            }
        report["by_eval_role"][role] = role_block

    default_slices = list(slices or [])
    if not default_slices:
        default_slices = [
            {"name": "negative", "human_semantic": POLARITY_NEGATIVE},
            {"name": "positive", "human_semantic": POLARITY_POSITIVE},
            {"name": "filler_neutral", "human_semantic": POLARITY_NEUTRAL},
            {"name": "short_le6", "short_max_chars": 6},
            {"name": "noise_events", "noise_only": True},
            {"name": "crosstalk", "crosstalk": True},
            {"name": "non_speech", "gold_kind": GOLD_KIND_NON_SPEECH},
        ]
    for spec in default_slices:
        name = str(spec.get("name") or "slice")
        subset = slice_samples(
            samples,
            human_semantic=spec.get("human_semantic"),
            gold_kind=spec.get("gold_kind"),
            short_max_chars=spec.get("short_max_chars"),
            noise_only=bool(spec.get("noise_only", False)),
            crosstalk=spec.get("crosstalk"),
            source_key=spec.get("source_key"),
            source_value=spec.get("source_value"),
            noise_tags=cfg.noise_event_tags,
        )
        if not subset:
            report["slices"][name] = {"n": 0, "models": {}}
            continue
        block = {"n": len(subset), "models": {}}
        for model in models:
            payload = compute_business_metrics_for_model(
                subset, model, judge=judge, config=cfg
            )
            block["models"][model] = {
                k: v for k, v in payload.items() if k != "metric_objects"
            }
        report["slices"][name] = block
    return report
