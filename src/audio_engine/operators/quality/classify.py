from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd
import yaml

from audio_engine.core.operator import ManifestOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample
from audio_engine.core.transcript_reconcile import (
    plain_transcript_text,
    rewrite_plain_transcript_entry,
)

# Flat columns that end with _text but are not ASR transcript fields.
_NON_ASR_TEXT_FIELDS = frozenset(
    {"gold_text", "baseline_text", "label_text", "ref_text", "hyp_text"}
)


def _rewrite_plain_transcripts(sample: Sample) -> None:
    """Idempotent plain-text rewrite; blanking already happened before CER."""
    for model, entry in list(sample.transcripts.items()):
        sample.transcripts[model] = rewrite_plain_transcript_entry(entry, keep_raw=True)


def _asr_text_columns(frame: pd.DataFrame) -> list[str]:
    return [
        name
        for name in frame.columns
        if name.endswith("_text")
        and name not in _NON_ASR_TEXT_FIELDS
        and not name.startswith("label_")
        and not name.startswith("quality_")
    ]


def _mark_all_transcripts_empty(frame: pd.DataFrame) -> None:
    """Set ``all_transcripts_empty``: every ASR ``*_text`` cell is blank."""
    cols = _asr_text_columns(frame)
    if not cols:
        frame["all_transcripts_empty"] = False
        return
    empty = pd.Series(True, index=frame.index)
    for column in cols:
        empty &= frame[column].fillna("").astype(str).str.strip().eq("")
    frame["all_transcripts_empty"] = empty


def _load_voicemail_patterns(path: str | Path | None) -> re.Pattern[str] | None:
    if not path:
        return None
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    patterns = raw.get("patterns") or []
    compiled: list[str] = []
    for item in patterns:
        text = str(item or "").strip()
        if text:
            compiled.append(f"(?:{text})")
    if not compiled:
        return None
    flags_name = str(raw.get("flags") or "IGNORECASE").upper()
    flags = re.IGNORECASE if "IGNORECASE" in flags_name else 0
    return re.compile("|".join(compiled), flags)


def _sample_match_texts(sample: Sample) -> list[str]:
    texts: list[str] = []
    for entry in sample.transcripts.values():
        if not isinstance(entry, dict):
            text = str(entry or "").strip()
            if text:
                texts.append(text)
            continue
        plain = str(entry.get("text") or "").strip()
        if plain:
            texts.append(plain)
        extra = entry.get("extra") if isinstance(entry.get("extra"), dict) else {}
        raw = str(extra.get("raw_text") or "").strip()
        if raw and raw != plain:
            texts.append(raw)
    return texts


def _mark_voicemail_hits(
    frame: pd.DataFrame, samples: list[Sample], pattern: re.Pattern[str] | None
) -> None:
    if pattern is None:
        frame["voicemail_hit"] = False
        return
    hits: list[bool] = []
    for sample in samples:
        texts = _sample_match_texts(sample)
        hits.append(any(pattern.search(text) is not None for text in texts))
    frame["voicemail_hit"] = hits


def _nonempty_transcripts(sample: Sample) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for model, entry in sample.transcripts.items():
        text = plain_transcript_text(sample.get_transcript_text(model))
        if text:
            pairs.append((str(model), text))
    return pairs


def _pick_gold_transcript(
    sample: Sample,
    *,
    gold_source_model: str,
    gold_pick: str,
) -> tuple[str, str]:
    """Return (model_key, gold_text)."""
    candidates = _nonempty_transcripts(sample)
    if not candidates:
        raise ValueError(f"auto_gold sample {sample.id} has no non-empty transcripts")
    mode = (gold_pick or "base").strip().lower()
    if mode in {"base", "gold_source", "fixed"}:
        for model, text in candidates:
            if model == gold_source_model:
                return model, text
        # fall through to first non-empty if configured base is empty
        return candidates[0]
    if mode in {"random", "any"}:
        digest = hashlib.sha256(str(sample.id).encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % len(candidates)
        return candidates[index]
    raise ValueError(f"unsupported gold_pick={gold_pick!r}; use base|random")


@register_operator
class ClassifyOperator(ManifestOperator):
    """Apply ordered, versioned selection rules and retain an auditable reason code."""

    name = "classify"
    version = "1.3.0"
    category = "quality"

    def run(self, samples: list[Sample], config: OperatorConfig) -> list[Sample]:
        params = dict(config.params)
        if params.get("config_path"):
            loaded = yaml.safe_load(Path(params["config_path"]).read_text(encoding="utf-8")) or {}
            params = {**loaded, **params}
        policy_version = str(params.get("policy_version") or "")
        rules = params.get("rules") or []
        default_bucket = str(params.get("default_bucket", "review_queue"))
        gold_source_model = str(params.get("gold_source_model", ""))
        gold_pick = str(params.get("gold_pick", "base"))
        if not policy_version:
            raise ValueError("quality.classify requires policy_version")
        if not isinstance(rules, list) or not rules:
            raise ValueError("quality.classify requires non-empty ordered rules")

        voicemail_pattern = _load_voicemail_patterns(params.get("voicemail_patterns_path"))

        updated = [sample.model_copy(deep=True) for sample in samples]
        for sample in updated:
            _rewrite_plain_transcripts(sample)
        frame = pd.DataFrame([sample.to_flat_dict() for sample in updated])
        for field, value in (params.get("defaults") or {}).items():
            if field not in frame:
                frame[field] = value
            else:
                frame[field] = frame[field].fillna(value)
        for column in [name for name in frame.columns if name.endswith("_text")]:
            frame[column] = frame[column].fillna("")
        _mark_all_transcripts_empty(frame)
        _mark_voicemail_hits(frame, updated, voicemail_pattern)

        decisions: list[tuple[str, list[str]]] = [(default_bucket, ["default"]) for _ in updated]
        undecided = set(range(len(updated)))
        for rule in rules:
            expr = str(rule.get("expr") or "")
            bucket = str(rule.get("bucket") or "")
            reason = str(rule.get("reason") or "")
            if not expr or not bucket or not reason:
                raise ValueError("each classify rule requires expr, bucket and reason")
            try:
                matched = set(frame.query(expr, engine="python").index) & undecided
            except Exception as exc:
                raise ValueError(f"invalid classify expression {expr!r}: {exc}") from exc
            for position in matched:
                decisions[position] = (bucket, [reason])
            undecided -= matched

        for sample, (bucket, reasons) in zip(updated, decisions, strict=True):
            sample.labels.update(
                {
                    "classification_bucket": bucket,
                    "type": bucket,
                    "classification_reason_codes": reasons,
                    "selection_policy_version": policy_version,
                }
            )
            if bucket == "auto_gold":
                if not gold_source_model and gold_pick.lower() in {"base", "gold_source", "fixed"}:
                    raise ValueError("auto_gold classification requires gold_source_model")
                model_key, gold_text = _pick_gold_transcript(
                    sample,
                    gold_source_model=gold_source_model,
                    gold_pick=gold_pick,
                )
                if not gold_text:
                    raise ValueError(
                        f"auto_gold sample {sample.id} has empty Gold transcript"
                    )
                sample.labels.update(
                    {
                        "annotation_state": "auto_accepted",
                        "gold_text": gold_text,
                        "label": gold_text,
                        "gold_source": model_key,
                        "annotation_revision": policy_version,
                    }
                )
            sample.mark_completed(self.full_name)
            sample.add_lineage(
                self.full_name,
                self.version,
                {"policy_version": policy_version, "bucket": bucket, "reason_codes": reasons},
            )
        return updated
