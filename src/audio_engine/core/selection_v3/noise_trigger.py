"""ASR-anomaly noise trigger and diagnosis contract (023).

DNSMOS runs only when every configured family lacks a usable transcript, or
any successful route contains non-Chinese speech. Normal Chinese samples stay
``not_required``: scores stay null, and missing/low/unknown/uncalibrated scores
are not an admission gate.

Execution status is not calibration. A successful score may still have
``noise_band=unknown`` when thresholds are uncalibrated.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable

from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3.config import SelectionV3Config
from audio_engine.core.selection_v3.input_contract import original_audio_sha256
from audio_engine.core.selection_v3.text import transcript_text
from audio_engine.core.selection_v3.types import (
    DNSMOS_STATUS_FAILED,
    DNSMOS_STATUS_NOT_REQUIRED,
    DNSMOS_STATUS_PENDING,
    DNSMOS_STATUS_SUCCESS,
    DNSMOS_STATUS_UNSUPPORTED,
    NOISE_POLICY_ASR_ANOMALY,
    NOISE_POLICY_LEGACY,
    QUALITY_STATE_FAILED,
    QUALITY_STATE_NOT_REQUIRED,
    QUALITY_STATE_UNCALIBRATED,
    QUALITY_STATE_UNSUPPORTED,
    TRIGGER_ALL_FAMILIES_NO_VALID,
    TRIGGER_NON_CHINESE,
)

TRIGGER_VERSION = "asr_anomaly_noise_v1"
LANGUAGE_POLICY_VERSION = "zh_speech_v1"

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_LATIN_RE = re.compile(r"[A-Za-z]+")
_PUNCT_RE = re.compile(r"[\s\W_]+", re.UNICODE)
_FAILED_STATUSES = frozenset(
    {"failed", "error", "fail", "timeout", "missing", "pending", "running"}
)
_ABBREV = frozenset(
    {
        "AI",
        "APP",
        "ATM",
        "CPU",
        "GPS",
        "ID",
        "LCD",
        "LED",
        "NFC",
        "PDF",
        "POS",
        "QR",
        "SIM",
        "SMS",
        "TV",
        "URL",
        "USB",
        "VIP",
        "WIFI",
    }
)


def uses_asr_anomaly_noise(config: SelectionV3Config | None) -> bool:
    policy = str(getattr(config, "noise_policy", "") or "").strip()
    return policy == NOISE_POLICY_ASR_ANOMALY


def normalize_noise_policy(value: Any) -> str:
    text = str(value or "").strip()
    if text in {NOISE_POLICY_ASR_ANOMALY, NOISE_POLICY_LEGACY}:
        return text
    if not text:
        return NOISE_POLICY_LEGACY
    raise ValueError(
        "noise_policy must be 'asr_anomaly_noise_v1' or 'legacy_full_quality_gate', "
        f"got {value!r}"
    )


def usable_body(value: Any) -> str:
    """Transcript body after control-tag strip. Punctuation-only is empty.

    Short fillers such as 嗯/啊/好 remain. They are real speech, not empty.
    """
    body = transcript_text(value)
    if not body:
        return ""
    compact = _PUNCT_RE.sub("", body)
    if not compact:
        return ""
    return body.strip()


def _is_abbreviation(token: str, *, has_cjk: bool) -> bool:
    if len(token) <= 1:
        return True
    if not has_cjk:
        return False
    upper = token.upper()
    if upper in _ABBREV:
        return True
    if token.isupper() and 2 <= len(token) <= 6:
        return True
    if 2 <= len(token) <= 8 and any(ch.isupper() for ch in token) and any(ch.islower() for ch in token):
        return True
    return False


def assess_speech_language(value: Any) -> dict[str, Any]:
    """Versioned speech-language check of the actual body, not the model tag.

    Returns ``zh`` / ``non_zh`` / ``mixed`` / ``unknown`` / ``empty``.
    Unknown does not add a scoring trigger. Traditional Chinese is Chinese.
    """
    body = usable_body(value)
    if not body:
        return {
            "label": "empty",
            "evidence": "empty_after_tag_strip",
            "spans": [],
            "policy": LANGUAGE_POLICY_VERSION,
        }
    normalized = unicodedata.normalize("NFKC", body)
    cjk = _CJK_RE.findall(normalized)
    tokens = _LATIN_RE.findall(normalized)
    has_cjk = bool(cjk)
    foreign = [tok for tok in tokens if not _is_abbreviation(tok, has_cjk=has_cjk)]
    # A lone 1-2 letter token without Chinese is too short to call foreign speech.
    if not has_cjk:
        foreign = [tok for tok in foreign if len(tok) >= 3]
    if has_cjk and not foreign:
        label = "zh"
        evidence = "cjk_without_foreign_speech"
    elif has_cjk and foreign:
        label = "mixed"
        evidence = "foreign_fragment_in_chinese"
    elif foreign:
        label = "non_zh"
        evidence = "foreign_speech"
    else:
        label = "unknown"
        evidence = "no_reliable_speech_script"
    return {
        "label": label,
        "evidence": evidence,
        "spans": foreign[:8],
        "policy": LANGUAGE_POLICY_VERSION,
        "snippet": body[:80],
    }


def _entry_status(entry: Any) -> str:
    if entry is None:
        return "missing"
    if not isinstance(entry, dict):
        return "success" if usable_body(entry) or str(entry or "").strip() else "success"
    status = str(entry.get("status") or "").strip().lower()
    extra = entry.get("extra") if isinstance(entry.get("extra"), dict) else {}
    extra_status = str(extra.get("inference_status") or extra.get("status") or "").strip().lower()
    if status in _FAILED_STATUSES or extra_status in _FAILED_STATUSES:
        return "failed" if status != "missing" and extra_status != "missing" else "missing"
    if status in {"invalid", "invalid_text"} or extra.get("invalid_text") is True:
        return "invalid_text"
    if entry.get("failed") is True:
        return "failed"
    return "success"


def _route_body(entry: Any) -> str:
    if entry is None:
        return ""
    if isinstance(entry, dict):
        extra = entry.get("extra") if isinstance(entry.get("extra"), dict) else {}
        raw = extra.get("raw_text")
        if raw is None:
            raw = entry.get("text")
        return usable_body(raw)
    return usable_body(entry)


@dataclass
class NoiseTriggerDecision:
    required: bool
    reasons: list[str] = field(default_factory=list)
    routes: list[dict[str, Any]] = field(default_factory=list)
    language_by_run: dict[str, str] = field(default_factory=dict)
    asr_snapshot: dict[str, str] = field(default_factory=dict)
    audio_sha256: str = ""
    technical_failure_retained: bool = False
    family_valid: dict[str, bool] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "required": self.required,
            "trigger_reasons": list(self.reasons),
            "trigger_routes": list(self.routes),
            "language_by_run": dict(self.language_by_run),
            "pre_filter_language_by_run": dict(self.language_by_run),
            "asr_status_snapshot": dict(self.asr_snapshot),
            "audio_sha256": self.audio_sha256,
            "technical_failure_retained": self.technical_failure_retained,
            "family_has_valid_transcript": dict(self.family_valid),
            "trigger_version": TRIGGER_VERSION,
            "language_policy_version": LANGUAGE_POLICY_VERSION,
        }


def evaluate_noise_trigger(sample: Sample, config: SelectionV3Config) -> NoiseTriggerDecision:
    """Compute the current trigger from raw ASR, before route quarantine."""
    snapshot: dict[str, str] = {}
    languages: dict[str, str] = {}
    family_valid: dict[str, bool] = {}
    foreign_routes: list[dict[str, Any]] = []
    technical_families = 0

    for family in config.ordered_families():
        keys = list(config.model_families.get(family, []))
        valid = False
        family_failed = bool(keys)
        for key in keys:
            entry = sample.transcripts.get(key)
            status = _entry_status(entry)
            body = _route_body(entry) if status == "success" else ""
            if status == "success" and not body:
                refined = "success_empty"
            elif status == "success" and body:
                refined = "success_text"
            else:
                refined = status
            snapshot[str(key)] = refined
            if refined != "success_text":
                languages[str(key)] = "empty" if refined == "success_empty" else refined
                continue
            family_failed = False
            lang = assess_speech_language(body)
            languages[str(key)] = str(lang["label"])
            # 嗯/啊/好 and digit-only unknown still count as usable body.
            valid = True
            if lang["label"] in {"non_zh", "mixed"}:
                foreign_routes.append(
                    {
                        "family": family,
                        "run_id": str(key),
                        "language": lang["label"],
                        "snippet": lang.get("snippet") or body[:80],
                        "spans": list(lang.get("spans") or []),
                        "evidence": lang.get("evidence"),
                    }
                )
        if family_failed and keys and all(
            snapshot.get(str(key)) in {"failed", "missing", "invalid_text"} for key in keys
        ):
            technical_families += 1
        family_valid[family] = valid

    reasons: list[str] = []
    if family_valid and not any(family_valid.values()):
        reasons.append(TRIGGER_ALL_FAMILIES_NO_VALID)
    if foreign_routes:
        reasons.append(TRIGGER_NON_CHINESE)
    technical = bool(family_valid) and technical_families == len(family_valid) and not any(
        family_valid.values()
    )
    return NoiseTriggerDecision(
        required=bool(reasons),
        reasons=reasons,
        routes=foreign_routes,
        language_by_run=languages,
        asr_snapshot=snapshot,
        audio_sha256=original_audio_sha256(sample),
        technical_failure_retained=technical and TRIGGER_ALL_FAMILIES_NO_VALID in reasons,
        family_valid=family_valid,
    )


def diagnosis_record(sample: Sample) -> dict[str, Any]:
    labels = sample.labels if isinstance(sample.labels, dict) else {}
    quality = sample.quality if isinstance(sample.quality, dict) else {}
    record = labels.get("noise_diagnosis")
    if isinstance(record, dict):
        return dict(record)
    nested = quality.get("noise_diagnosis")
    if isinstance(nested, dict):
        return dict(nested)
    return {}


def diagnosis_status(sample: Sample) -> str | None:
    record = diagnosis_record(sample)
    status = record.get("status") or (sample.quality or {}).get("noise_diagnosis_status")
    if status is None:
        return None
    return str(status)


def ensure_trigger_record(sample: Sample, config: SelectionV3Config) -> dict[str, Any]:
    """Recompute the current trigger. Does not score and does not drop first_trigger."""
    decision = evaluate_noise_trigger(sample, config)
    existing = diagnosis_record(sample)
    status = str(existing.get("status") or "")
    if not decision.required:
        status = DNSMOS_STATUS_NOT_REQUIRED
    elif status not in {DNSMOS_STATUS_SUCCESS, DNSMOS_STATUS_FAILED, DNSMOS_STATUS_UNSUPPORTED}:
        status = DNSMOS_STATUS_PENDING
    first = existing.get("first_trigger")
    if not first and decision.required:
        first = {
            "trigger_reasons": list(decision.reasons),
            "trigger_routes": list(decision.routes),
            "asr_status_snapshot": dict(decision.asr_snapshot),
            "trigger_version": TRIGGER_VERSION,
        }
    record = {
        "policy": NOISE_POLICY_ASR_ANOMALY,
        "required": decision.required,
        "trigger_reasons": list(decision.reasons),
        "trigger_version": TRIGGER_VERSION,
        "language_policy_version": LANGUAGE_POLICY_VERSION,
        "status": status,
        "error": None if status == DNSMOS_STATUS_NOT_REQUIRED else existing.get("error"),
        "unsupported_reason": existing.get("unsupported_reason"),
        "trigger_routes": list(decision.routes),
        "language_by_run": dict(decision.language_by_run),
        "pre_filter_language_by_run": dict(decision.language_by_run),
        "asr_status_snapshot": dict(decision.asr_snapshot),
        "audio_sha256": decision.audio_sha256,
        "technical_failure_retained": decision.technical_failure_retained,
        "family_has_valid_transcript": dict(decision.family_valid),
        "first_trigger": first,
        "cache_hit": existing.get("cache_hit") if decision.required else None,
        "scores": None if status == DNSMOS_STATUS_NOT_REQUIRED else existing.get("scores"),
        "noise_band": None if status == DNSMOS_STATUS_NOT_REQUIRED else existing.get("noise_band"),
        "calibrated": False if status != DNSMOS_STATUS_SUCCESS else bool(existing.get("calibrated")),
        "model_digest": existing.get("model_digest") or "",
        "preprocess_version": existing.get("preprocess_version") or "",
        "threshold_version": existing.get("threshold_version") or "",
    }
    if status == DNSMOS_STATUS_NOT_REQUIRED:
        record["scores"] = None
        record["noise_band"] = None
        record["error"] = None
        park_legacy_scores(sample)
    sample.labels["noise_diagnosis"] = record
    sample.labels["noise_policy"] = NOISE_POLICY_ASR_ANOMALY
    sample.labels["noise_diagnosis_status"] = status
    sample.labels["noise_trigger_reasons"] = list(decision.reasons)
    _mirror_quality(sample, record)
    return record


def park_legacy_scores(sample: Sample) -> None:
    """Keep old full-batch scores auditable, but they must not gate not_required rows."""
    quality = sample.quality if isinstance(sample.quality, dict) else {}
    if not quality:
        return
    if quality.get("dnsmos_status") == DNSMOS_STATUS_NOT_REQUIRED and quality.get("dnsmos_sig") is None:
        return
    interesting = any(quality.get(key) is not None for key in ("dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "noise_band"))
    if interesting and "legacy_dnsmos" not in quality:
        quality = dict(quality)
        quality["legacy_dnsmos"] = {
            "dnsmos_sig": quality.get("dnsmos_sig"),
            "dnsmos_bak": quality.get("dnsmos_bak"),
            "dnsmos_ovrl": quality.get("dnsmos_ovrl"),
            "dnsmos_status": quality.get("dnsmos_status"),
            "noise_band": quality.get("noise_band"),
            "noise_risk": quality.get("noise_risk"),
        }
        sample.quality = quality


def _mirror_quality(sample: Sample, record: dict[str, Any]) -> None:
    quality = dict(sample.quality or {})
    status = str(record.get("status") or "")
    scores = record.get("scores") if isinstance(record.get("scores"), dict) else None
    if status == DNSMOS_STATUS_NOT_REQUIRED:
        quality.update(
            {
                "dnsmos_sig": None,
                "dnsmos_bak": None,
                "dnsmos_ovrl": None,
                "dnsmos_status": DNSMOS_STATUS_NOT_REQUIRED,
                "noise_band": None,
                "noise_risk": None,
                "noise_diagnosis_status": DNSMOS_STATUS_NOT_REQUIRED,
                "noise_diagnosis_required": False,
                "calibrated": False,
            }
        )
    else:
        quality["dnsmos_status"] = status
        quality["noise_diagnosis_status"] = status
        quality["noise_diagnosis_required"] = True
        if scores:
            quality["dnsmos_sig"] = scores.get("sig")
            quality["dnsmos_bak"] = scores.get("bak")
            quality["dnsmos_ovrl"] = scores.get("ovrl")
        elif status in {DNSMOS_STATUS_FAILED, DNSMOS_STATUS_UNSUPPORTED, DNSMOS_STATUS_PENDING}:
            quality["dnsmos_sig"] = None
            quality["dnsmos_bak"] = None
            quality["dnsmos_ovrl"] = None
        if status == DNSMOS_STATUS_SUCCESS and record.get("calibrated") is not True:
            quality["noise_band"] = "unknown"
            quality["noise_risk"] = None
            quality["calibrated"] = False
        elif status != DNSMOS_STATUS_SUCCESS:
            quality["noise_band"] = None
            quality["noise_risk"] = None
            quality["calibrated"] = False
    quality["noise_policy"] = NOISE_POLICY_ASR_ANOMALY
    quality["noise_trigger_version"] = TRIGGER_VERSION
    sample.quality = quality


def quality_state_for_diagnosis(record: dict[str, Any] | None) -> str:
    status = str((record or {}).get("status") or "")
    required = bool((record or {}).get("required"))
    if status == DNSMOS_STATUS_NOT_REQUIRED or (not required and status in {"", DNSMOS_STATUS_NOT_REQUIRED}):
        return QUALITY_STATE_NOT_REQUIRED
    if status == DNSMOS_STATUS_FAILED:
        return QUALITY_STATE_FAILED
    if status == DNSMOS_STATUS_UNSUPPORTED:
        return QUALITY_STATE_UNSUPPORTED
    if status == DNSMOS_STATUS_SUCCESS and (record or {}).get("calibrated") is not True:
        return QUALITY_STATE_UNCALIBRATED
    if status == DNSMOS_STATUS_PENDING:
        return DNSMOS_STATUS_PENDING
    return QUALITY_STATE_UNCALIBRATED


def admission_ignores_dnsmos(sample: Sample, config: SelectionV3Config | None) -> bool:
    """New policy, and not_required rows, never re-enter the old quality gate."""
    if uses_asr_anomaly_noise(config):
        return True
    return diagnosis_status(sample) == DNSMOS_STATUS_NOT_REQUIRED


@dataclass
class ScoreResult:
    status: str
    sig: float | None = None
    bak: float | None = None
    ovrl: float | None = None
    error: str | None = None
    model_digest: str = "fake"
    preprocess_version: str = "fake_preprocess_v1"
    unsupported_reason: str | None = None


class CallCountingScorer:
    """Offline scorer for tests. Does not load ONNX."""

    def __init__(self, result: ScoreResult | None = None, *, fail_unreadable: bool = False) -> None:
        self.calls: list[str] = []
        self.result = result or ScoreResult(status=DNSMOS_STATUS_SUCCESS, sig=3.2, bak=3.4, ovrl=3.1)
        self.fail_unreadable = fail_unreadable

    def score(self, *, sample: Sample, audio_path: str, audio_sha256: str) -> ScoreResult:
        self.calls.append(audio_sha256 or str(sample.id))
        if self.fail_unreadable or sample.labels.get("invalid_audio") is True or sample.labels.get("broken") is True:
            return ScoreResult(
                status=DNSMOS_STATUS_FAILED,
                error="audio_unreadable",
                model_digest=self.result.model_digest,
                preprocess_version=self.result.preprocess_version,
            )
        return self.result


def apply_score(
    sample: Sample,
    config: SelectionV3Config,
    scored: ScoreResult,
    *,
    cache_hit: bool,
    threshold_version: str,
    calibrated: bool,
) -> dict[str, Any]:
    record = ensure_trigger_record(sample, config)
    if not record.get("required"):
        return record
    record["status"] = scored.status
    record["cache_hit"] = cache_hit
    record["error"] = scored.error
    record["unsupported_reason"] = scored.unsupported_reason
    record["model_digest"] = scored.model_digest
    record["preprocess_version"] = scored.preprocess_version
    record["threshold_version"] = threshold_version
    record["calibrated"] = bool(calibrated) and scored.status == DNSMOS_STATUS_SUCCESS
    if scored.status == DNSMOS_STATUS_SUCCESS and None not in {scored.sig, scored.bak, scored.ovrl}:
        record["scores"] = {"sig": scored.sig, "bak": scored.bak, "ovrl": scored.ovrl}
        record["noise_band"] = "unknown" if not calibrated else record.get("noise_band")
    else:
        record["scores"] = None
        record["noise_band"] = None
        if scored.status == DNSMOS_STATUS_SUCCESS:
            record["status"] = DNSMOS_STATUS_FAILED
            record["error"] = scored.error or "incomplete_scores"
    sample.labels["noise_diagnosis"] = record
    sample.labels["noise_diagnosis_status"] = record["status"]
    _mirror_quality(sample, record)
    return record


def diagnose_samples(
    samples: list[Sample],
    config: SelectionV3Config,
    scorer: Any | None,
    *,
    calibrated: bool = False,
    threshold_version: str = "uncalibrated",
    score_cache: dict[str, ScoreResult] | None = None,
) -> dict[str, Any]:
    """Route, score the unique triggered audio, and backfill. Empty subset never calls scorer."""
    cache = score_cache if score_cache is not None else {}
    decisions = [evaluate_noise_trigger(sample, config) for sample in samples]
    required_indexes = [i for i, decision in enumerate(decisions) if decision.required]
    calls_before = len(getattr(scorer, "calls", [])) if scorer is not None else 0

    for sample, decision in zip(samples, decisions, strict=True):
        if not decision.required:
            ensure_trigger_record(sample, config)

    if not required_indexes:
        return _report(samples, calls=0, cache_hits=0, scored_audio=0)

    seen: dict[str, ScoreResult] = {}
    scored_audio = 0
    cache_hits = 0
    model_missing = scorer is None
    for index in required_indexes:
        sample = samples[index]
        decision = decisions[index]
        digest = decision.audio_sha256 or f"missing-hash:{sample.id}"
        if digest in seen:
            apply_score(
                sample,
                config,
                seen[digest],
                cache_hit=True,
                threshold_version=threshold_version,
                calibrated=calibrated,
            )
            cache_hits += 1
            continue
        cache_key = f"{digest}|{getattr(scorer, 'model_id', 'scorer')}|{getattr(scorer, 'preprocess_version', 'v')}"
        if cache_key in cache:
            result = cache[cache_key]
            cache_hits += 1
            hit = True
        elif model_missing:
            result = ScoreResult(status=DNSMOS_STATUS_FAILED, error="scoring_model_missing")
            hit = False
        else:
            try:
                audio_path = ""
                try:
                    audio_path = str(sample.audio_path("resampled_16k"))
                except Exception:
                    audio_path = str(getattr(sample, "source_path", "") or "")
                before = len(getattr(scorer, "calls", []))
                result = scorer.score(sample=sample, audio_path=audio_path, audio_sha256=digest)
                if len(getattr(scorer, "calls", [])) > before:
                    scored_audio += 1
                hit = False
            except Exception as exc:  # noqa: BLE001 — isolate diagnosis failures
                result = ScoreResult(status=DNSMOS_STATUS_FAILED, error=str(exc))
                hit = False
            cache[cache_key] = result
        if result.audio_sha256_mismatch if hasattr(result, "audio_sha256_mismatch") else False:
            raise ValueError(f"noise score audio hash mismatch for {sample.id}")
        seen[digest] = result
        apply_score(
            sample,
            config,
            result,
            cache_hit=hit,
            threshold_version=threshold_version,
            calibrated=calibrated,
        )

    calls = 0
    if scorer is not None and hasattr(scorer, "calls"):
        calls = len(scorer.calls) - calls_before
    return _report(samples, calls=calls, cache_hits=cache_hits, scored_audio=scored_audio)


def _report(samples: list[Sample], *, calls: int, cache_hits: int, scored_audio: int) -> dict[str, Any]:
    reasons = {"all_families_no_valid_transcript": 0, "non_chinese_transcript": 0, "both": 0}
    statuses: dict[str, int] = {}
    ids = []
    for sample in samples:
        ids.append(str(sample.id))
        record = diagnosis_record(sample)
        status = str(record.get("status") or "")
        statuses[status] = statuses.get(status, 0) + 1
        got = set(record.get("trigger_reasons") or [])
        if TRIGGER_ALL_FAMILIES_NO_VALID in got and TRIGGER_NON_CHINESE in got:
            reasons["both"] += 1
        elif TRIGGER_ALL_FAMILIES_NO_VALID in got:
            reasons["all_families_no_valid_transcript"] += 1
        elif TRIGGER_NON_CHINESE in got:
            reasons["non_chinese_transcript"] += 1
    return {
        "sample_count": len(samples),
        "sample_ids": ids,
        "calls": calls,
        "scored_audio": scored_audio,
        "cache_hits": cache_hits,
        "status_counts": statuses,
        "trigger_counts": reasons,
        "not_required": statuses.get(DNSMOS_STATUS_NOT_REQUIRED, 0),
    }


def assert_backfill_identity(sample: Sample, row: dict[str, Any]) -> None:
    """Reject a score row that belongs to a different sample or audio."""
    if str(row.get("sample_id") or "") != str(sample.id):
        raise ValueError(f"noise backfill sample_id mismatch: {sample.id}")
    expected = original_audio_sha256(sample)
    got = str(row.get("audio_sha256") or "")
    if not expected or not got or expected != got:
        raise ValueError(f"noise backfill audio hash mismatch: {sample.id}")


def diagnosis_digest(record: dict[str, Any]) -> str:
    payload = json.dumps(record, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


ScorerFn = Callable[..., ScoreResult]
