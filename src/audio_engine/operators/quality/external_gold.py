"""External-gold helpers for process-1 split (inject + external bucketing).

Keeps ``gold_text`` from the external xlsx; never overwrites with consensus medoid.
First-cut buckets align with the bypass script, converged to four types:
``voicemail`` / ``noise`` / ``auto_gold`` / ``hardcase``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from audio_engine.core.sample import Sample
from audio_engine.core.transcript_reconcile import (
    plain_transcript_text,
    resolve_blank_exact_hotwords,
)

RULE_VERSION = "external_gold_v1"
EMPTY_GOLD_MARKER = "噪声"

_SKIP_IDS = frozenset({"总体统计", "汇总", "total", "summary"})
_ID_CANDIDATES = ("id", "sample_id", "audio_id", "utt_id")
_LABEL_CANDIDATES = (
    "label_text_raw",
    "label",
    "gold_text",
    "金标",
    "标注",
    "label_text",
)


def _cell(value: Any) -> str:
    if value is None:
        return ""
    try:
        if value != value:  # NaN
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.lower() in {"nan", "none"}:
        return ""
    return text


def blank_exact(text: str, hotwords: frozenset[str]) -> str:
    plain = plain_transcript_text(text)
    if plain and plain in hotwords:
        return ""
    return plain


def load_hotwords(path: str | Path | None) -> frozenset[str]:
    if not path:
        return frozenset()
    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg = loaded.get("blank_exact_hotwords", loaded)
    hotwords, _ = resolve_blank_exact_hotwords(cfg)
    return hotwords


def load_voicemail_pattern(path: str | Path | None) -> re.Pattern[str] | None:
    if not path:
        return None
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    patterns = [str(item).strip() for item in (raw.get("patterns") or []) if str(item).strip()]
    if not patterns:
        return None
    flags = re.IGNORECASE if "IGNORECASE" in str(raw.get("flags") or "IGNORECASE").upper() else 0
    return re.compile("|".join(f"(?:{p})" for p in patterns), flags)


def resolve_xlsx_columns(
    columns: list[str],
    *,
    id_col: str | None,
    label_col: str | None,
) -> tuple[str, str]:
    cols = list(columns)
    if id_col:
        if id_col not in cols:
            raise ValueError(f"xlsx missing id column {id_col!r}; available: {cols}")
        resolved_id = id_col
    else:
        resolved_id = next((name for name in _ID_CANDIDATES if name in cols), "")
        if not resolved_id:
            raise ValueError(f"xlsx missing id/sample_id; available: {cols}")

    if label_col:
        if label_col not in cols:
            raise ValueError(f"xlsx missing label column {label_col!r}; available: {cols}")
        resolved_label = label_col
    else:
        resolved_label = next((name for name in _LABEL_CANDIDATES if name in cols), "")
        if not resolved_label:
            raise ValueError(
                f"xlsx missing label column (tried {list(_LABEL_CANDIDATES)}); available: {cols}"
            )
    return resolved_id, resolved_label


def load_external_gold_table(
    xlsx_path: str | Path,
    *,
    id_col: str | None = None,
    label_col: str | None = None,
    type_col: str | None = None,
) -> tuple[dict[str, dict[str, str]], dict[str, int]]:
    """Return ``{sample_id: {gold_text, label_text_raw, type?}}`` plus counters."""
    path = Path(xlsx_path)
    if not path.is_file():
        raise FileNotFoundError(f"external gold xlsx not found: {path}")
    frame = pd.read_excel(path, dtype=str).fillna("")
    resolved_id, resolved_label = resolve_xlsx_columns(
        list(frame.columns), id_col=id_col, label_col=label_col
    )
    if type_col and type_col not in frame.columns:
        raise ValueError(f"xlsx missing type column {type_col!r}; available: {list(frame.columns)}")

    table: dict[str, dict[str, str]] = {}
    counts = {
        "xlsx_rows": 0,
        "skipped_summary": 0,
        "dup_id": 0,
        "empty_id": 0,
        "loaded": 0,
    }
    for _, row in frame.iterrows():
        counts["xlsx_rows"] += 1
        sample_id = _cell(row.get(resolved_id))
        if not sample_id:
            counts["empty_id"] += 1
            continue
        if sample_id in _SKIP_IDS:
            counts["skipped_summary"] += 1
            continue
        if sample_id in table:
            counts["dup_id"] += 1
            raise ValueError(f"duplicate id in external gold xlsx: {sample_id}")
        entry = {
            "gold_text": _cell(row.get(resolved_label)),
            "label_text_raw": _cell(row.get(resolved_label)),
        }
        if type_col:
            entry["type"] = _cell(row.get(type_col))
        table[sample_id] = entry
        counts["loaded"] += 1
    return table, counts


@dataclass
class ExternalClassification:
    type: str
    decision: str
    label: str
    reason: str
    compare_model: str
    gold_plain: str
    hyp_plain: str
    rule_version: str = RULE_VERSION

    def to_labels(self, policy_version: str, *, preserve_raw: str) -> dict[str, Any]:
        return {
            "classification_bucket": self.type,
            "type": self.type,
            "decision": self.decision,
            "classification_reason_codes": [self.reason],
            "selection_policy_version": policy_version,
            "rule_version": self.rule_version,
            "gold_mode": "external",
            "gold_source": "external",
            "selected_model": "external",
            "compare_model": self.compare_model,
            "label_text_raw": preserve_raw,
            "annotation_state": (
                "auto_accepted"
                if self.decision in {"auto_accept", "auto_empty"}
                else "pending_review"
            ),
            "annotation_revision": policy_version,
            "gold_text": self.label,
            "label": self.label,
        }


def pick_compare_model(sample: Sample, preferred: str | None) -> str:
    key = str(preferred or "").strip()
    if key:
        if key not in sample.transcripts:
            raise ValueError(
                f"sample {sample.id}: compare_model {key!r} missing from transcripts "
                f"{sorted(sample.transcripts)}"
            )
        return key
    if not sample.transcripts:
        raise ValueError(f"sample {sample.id}: no transcripts for external classify")
    # Stable pick: prefer names starting with qwen, else lexicographic.
    names = sorted(sample.transcripts)
    for name in names:
        if name.lower().startswith("qwen"):
            return name
    return names[0]


def classify_external_sample(
    sample: Sample,
    *,
    compare_model: str | None,
    hotwords: frozenset[str],
    voicemail_pattern: re.Pattern[str] | None,
    preassigned_type: str | None = None,
) -> ExternalClassification:
    """Bucket one sample using external gold vs one ASR hypothesis."""
    raw_gold = str(
        sample.labels.get("label_text_raw")
        or sample.labels.get("gold_text")
        or sample.labels.get("label")
        or ""
    ).strip()
    if preassigned_type:
        bucket = str(preassigned_type).strip()
        label_out = blank_exact(raw_gold, hotwords) or raw_gold
        if bucket == "noise" and not label_out:
            label_out = EMPTY_GOLD_MARKER
        decision = "auto_accept" if bucket in {"auto_gold", "voicemail", "noise"} else "model_review"
        if bucket == "noise":
            decision = "auto_empty"
        model = pick_compare_model(sample, compare_model) if sample.transcripts else ""
        hyp_plain = blank_exact(str(sample.get_transcript_text(model) or ""), hotwords) if model else ""
        return ExternalClassification(
            type=bucket,
            decision=decision,
            label=label_out,
            reason="external_type_preassigned",
            compare_model=model,
            gold_plain=blank_exact(raw_gold, hotwords),
            hyp_plain=hyp_plain,
        )

    model = pick_compare_model(sample, compare_model)
    hyp_raw = str(sample.get_transcript_text(model) or "")
    gold_plain = blank_exact(raw_gold, hotwords)
    hyp_plain = blank_exact(hyp_raw, hotwords)

    match_texts = [raw_gold, hyp_raw, gold_plain, hyp_plain]
    voicemail_hit = bool(
        voicemail_pattern and any(voicemail_pattern.search(text) for text in match_texts if text)
    )

    if voicemail_hit:
        return ExternalClassification(
            type="voicemail",
            decision="auto_accept",
            label=gold_plain or raw_gold,
            reason="voicemail_or_phone_assistant",
            compare_model=model,
            gold_plain=gold_plain,
            hyp_plain=hyp_plain,
        )
    if not gold_plain and not hyp_plain:
        return ExternalClassification(
            type="noise",
            decision="auto_empty",
            label=EMPTY_GOLD_MARKER,
            reason="empty_both_after_clean",
            compare_model=model,
            gold_plain=gold_plain,
            hyp_plain=hyp_plain,
        )
    if not gold_plain and hyp_plain:
        # Empty gold + model speech: treat as hardcase; keep raw annotation if any.
        return ExternalClassification(
            type="hardcase",
            decision="model_review",
            label=raw_gold or EMPTY_GOLD_MARKER,
            reason="empty_label_model_hallucination",
            compare_model=model,
            gold_plain=gold_plain,
            hyp_plain=hyp_plain,
        )
    if gold_plain == hyp_plain:
        return ExternalClassification(
            type="auto_gold",
            decision="auto_accept",
            label=gold_plain,
            reason="external_label_exact_match",
            compare_model=model,
            gold_plain=gold_plain,
            hyp_plain=hyp_plain,
        )
    return ExternalClassification(
        type="hardcase",
        decision="model_review",
        label=gold_plain,
        reason="external_label_mismatch",
        compare_model=model,
        gold_plain=gold_plain,
        hyp_plain=hyp_plain,
    )
