"""Independent tag evidence / family-route config (044). No SelectionV3Config."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_PUNCT = list("，。！？、；：\"\"''（）【】《》…—·,.!?;:'\"()[]{}")

_DEFAULT_CRITICAL_SHORT = (
    "要",
    "不要",
    "需要",
    "不需要",
    "好",
    "不好",
    "可以",
    "不可以",
    "行",
    "不行",
    "是",
    "不是",
    "有",
    "没有",
    "用",
    "不用",
    "好的",
    "嗯",
    "对",
)


def _sha256_obj(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


@dataclass
class TagEvidenceConfig:
    """Neutral family/route/text/threshold contract for free-tag evaluation."""

    policy_version: str = "tag_evidence_v2"
    model_families: dict[str, list[str]] = field(default_factory=dict)
    teacher_families: list[str] = field(default_factory=lambda: ["glm", "sensevoice"])
    target_family: str = "qwen"
    expected_runs_per_family: int = 2
    run_aliases: dict[str, str] = field(default_factory=dict)

    punctuation_to_strip: list[str] = field(default_factory=lambda: list(_DEFAULT_PUNCT))
    critical_short_responses: list[str] = field(
        default_factory=lambda: list(_DEFAULT_CRITICAL_SHORT)
    )
    family_threshold: float = 0.98
    short_polarity_max_han_chars: int = 4

    classify_text_policy: str = "five_class_v1"
    classify_text_keep_digits: bool = True
    classify_text_echo_missing: str = "fail"
    classify_text_echo: dict[str, Any] = field(default_factory=dict)

    voicemail_patterns_path: str = ""
    semantic_lexicon_path: str = ""
    dnsmos_weak_sig: float | None = 2.8

    # Lexicon phrase lists used by polarity / critical-short (independent copies).
    negative_phrases: list[str] = field(default_factory=list)
    positive_phrases: list[str] = field(default_factory=list)
    critical_tokens: list[str] = field(default_factory=list)
    filler_phrases: list[str] = field(default_factory=list)
    affirmation_phrases: list[str] = field(default_factory=list)
    profanity_or_reject: list[str] = field(default_factory=list)

    silero_vad_config_path: str = ""
    audio_energy_config_path: str = ""
    dnsmos_config_path: str = ""

    source_path: str = ""
    content_digest: str = ""

    def ordered_families(self) -> list[str]:
        ordered: list[str] = []
        for family in list(self.teacher_families) + [self.target_family]:
            if family in self.model_families and family not in ordered:
                ordered.append(family)
        for family in sorted(self.model_families):
            if family not in ordered:
                ordered.append(family)
        return ordered

    def family_of_key(self, key: str) -> str | None:
        alias = self.run_aliases.get(str(key))
        if alias:
            return str(alias)
        for family, keys in self.model_families.items():
            if key in keys:
                return family
        return None

    def uses_chinese_only_text(self) -> bool:
        from audio_engine.core.evidence.classify_text import uses_chinese_only_text

        return uses_chinese_only_text(self.classify_text_policy)

    def echo_table_for(self, family: str | None):
        from audio_engine.core.evidence.classify_text import echo_for_family

        return echo_for_family(self.classify_text_echo, family)

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "model_families": {k: list(v) for k, v in self.model_families.items()},
            "teacher_families": list(self.teacher_families),
            "target_family": self.target_family,
            "expected_runs_per_family": self.expected_runs_per_family,
            "run_aliases": dict(self.run_aliases),
            "punctuation_to_strip": list(self.punctuation_to_strip),
            "critical_short_responses": list(self.critical_short_responses),
            "family_threshold": self.family_threshold,
            "short_polarity_max_han_chars": self.short_polarity_max_han_chars,
            "classify_text_policy": self.classify_text_policy,
            "classify_text_keep_digits": self.classify_text_keep_digits,
            "classify_text_echo_missing": self.classify_text_echo_missing,
            "classify_text_echo": {
                str(k): (
                    {
                        "prompt_count": len(getattr(v, "prompts", ()) or ()),
                        "table_dump_count": len(getattr(v, "table_dumps", ()) or ()),
                        "exact_phrase_count": len(getattr(v, "exact_phrases", ()) or ()),
                        "fingerprint": str(getattr(v, "fingerprint", "") or ""),
                    }
                    if v is not None and not isinstance(v, dict)
                    else v
                )
                for k, v in (self.classify_text_echo or {}).items()
            },
            "voicemail_patterns_path": self.voicemail_patterns_path,
            "semantic_lexicon_path": self.semantic_lexicon_path,
            "dnsmos_weak_sig": self.dnsmos_weak_sig,
            "silero_vad_config_path": self.silero_vad_config_path,
            "audio_energy_config_path": self.audio_energy_config_path,
            "dnsmos_config_path": self.dnsmos_config_path,
            "referenced_file_digests": self._referenced_file_digests(),
            "source_path": self.source_path,
            "content_digest": self.content_digest,
        }

    def _referenced_file_digests(self) -> dict[str, str]:
        """Content hashes of lexicon / acoustic config files this evidence points at."""
        out: dict[str, str] = {}
        for label, raw in (
            ("voicemail_patterns", self.voicemail_patterns_path),
            ("semantic_lexicon", self.semantic_lexicon_path),
            ("silero_vad_config", self.silero_vad_config_path),
            ("audio_energy_config", self.audio_energy_config_path),
            ("dnsmos_config", self.dnsmos_config_path),
        ):
            if not raw:
                continue
            path = Path(str(raw))
            if not path.is_absolute() and self.source_path and not self.source_path.startswith("migrated_from:"):
                from audio_engine.core.tagging.paths import find_repo_root

                path = find_repo_root(Path(self.source_path).parent) / path
            try:
                out[label] = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                out[label] = "missing"
        return out

    def fingerprint(self) -> str:
        return _sha256_obj(self.as_dict())


def _load_yaml(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"evidence config not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_tag_evidence_config(path: str | Path) -> TagEvidenceConfig:
    p = Path(path).resolve()
    raw = _load_yaml(p)
    if not isinstance(raw, dict):
        raise ValueError(f"evidence config must be mapping: {p}")
    # Allow nested evidence: key or flat.
    body = raw.get("evidence") if isinstance(raw.get("evidence"), dict) else raw
    families = body.get("model_families") or {}
    if not isinstance(families, dict) or not families:
        raise ValueError(f"{p}: model_families is required")
    ct = body.get("classify_text") if isinstance(body.get("classify_text"), dict) else {}
    acoustic = body.get("acoustic") if isinstance(body.get("acoustic"), dict) else {}
    echo_missing = str(
        ct.get("echo_missing") or body.get("classify_text_echo_missing") or "fail"
    )
    echo_raw = dict(ct.get("echo") or body.get("classify_text_echo") or {})
    # Materialize EchoTable objects (same as SelectionV3Config.from_yaml).
    from audio_engine.core.evidence.classify_text import load_echo_tables

    teacher_families = [str(x) for x in (body.get("teacher_families") or ["glm", "sensevoice"])]
    target_family = str(body.get("target_family") or "qwen")
    model_families = {str(k): [str(x) for x in (v or [])] for k, v in families.items()}
    # Provisional family order for echo loading (mirrors ordered_families).
    family_order: list[str] = []
    for fam in list(teacher_families) + [target_family]:
        if fam in model_families and fam not in family_order:
            family_order.append(fam)
    for fam in sorted(model_families):
        if fam not in family_order:
            family_order.append(fam)
    # Echo ASR/blank paths are repo-relative (configs/asr/...), not relative to this YAML.
    from audio_engine.core.tagging.paths import find_repo_root

    repo_root = find_repo_root(p.parent)
    try:
        echo_tables = load_echo_tables(
            families=family_order,
            echo_cfg=echo_raw,
            missing=echo_missing,
            base_dir=repo_root,
        )
    except Exception:
        if echo_missing == "fail":
            raise
        echo_tables = {}

    cfg = TagEvidenceConfig(
        policy_version=str(body.get("policy_version") or "tag_evidence_v2"),
        model_families=model_families,
        teacher_families=teacher_families,
        target_family=target_family,
        expected_runs_per_family=int(body.get("expected_runs_per_family") or 2),
        run_aliases={str(k): str(v) for k, v in (body.get("run_aliases") or {}).items()},
        punctuation_to_strip=[
            str(x) for x in (body.get("punctuation_to_strip") or list(_DEFAULT_PUNCT))
        ],
        critical_short_responses=[
            str(x)
            for x in (
                body.get("critical_short_responses")
                or (body.get("selection") or {}).get("critical_short_responses")
                or list(_DEFAULT_CRITICAL_SHORT)
            )
        ],
        family_threshold=float(
            (body.get("similarity") or {}).get("family_threshold")
            or body.get("family_threshold")
            or 0.98
        ),
        short_polarity_max_han_chars=int(
            (body.get("selection") or {}).get("short_polarity_max_han_chars")
            or body.get("short_polarity_max_han_chars")
            or 4
        ),
        classify_text_policy=str(ct.get("policy") or body.get("classify_text_policy") or "five_class_v1"),
        classify_text_keep_digits=bool(
            ct.get("keep_digits", body.get("classify_text_keep_digits", True))
        ),
        classify_text_echo_missing=echo_missing,
        classify_text_echo=echo_tables,
        voicemail_patterns_path=str(
            body.get("voicemail_patterns_path")
            or acoustic.get("voicemail_patterns_path")
            or ""
        ),
        semantic_lexicon_path=str(body.get("semantic_lexicon_path") or ""),
        dnsmos_weak_sig=float(
            body.get("dnsmos_weak_sig")
            or (body.get("dnsmos_decision") or {}).get("weak_sig")
            or 2.8
        ),
        silero_vad_config_path=str(
            acoustic.get("silero_vad_config")
            or (body.get("silero_vad") or {}).get("config_path")
            or ""
        ),
        audio_energy_config_path=str(
            acoustic.get("audio_energy_config")
            or (body.get("audio_energy") or {}).get("config_path")
            or ""
        ),
        dnsmos_config_path=str(
            acoustic.get("dnsmos_config")
            or body.get("dnsmos_config_path")
            or ""
        ),
        source_path=str(p.as_posix()),
    )
    _apply_semantic_lexicon(cfg, body, p)
    cfg.content_digest = cfg.fingerprint()
    return cfg


_LEXICON_KEYS = {
    "negative": "negative_phrases",
    "positive": "positive_phrases",
    "critical_tokens": "critical_tokens",
    "profanity_or_reject": "profanity_or_reject",
    "filler": "filler_phrases",
    "affirmation": "affirmation_phrases",
}


def _clean_list(values: Any) -> list[str]:
    return [str(x) for x in (values or []) if str(x).strip()]


def _apply_semantic_lexicon(cfg: TagEvidenceConfig, body: dict[str, Any], evidence_path: Path) -> None:
    """Fill polarity / critical / filler phrase lists (same precedence as selection_v3).

    Inline ``semantic:`` block wins over the lexicon file; the lexicon file only fills
    empty lists; defaults for filler / affirmation match the historical behaviour.
    """
    from audio_engine.core.tagging.paths import find_repo_root, resolve_tagging_path

    inline = body.get("semantic") if isinstance(body.get("semantic"), dict) else {}
    lists: dict[str, list[str]] = {
        attr: _clean_list(inline.get(key)) for key, attr in _LEXICON_KEYS.items()
    }
    punct_override = None
    if inline.get("punctuation_to_strip") is not None:
        punct_override = [str(x) for x in inline["punctuation_to_strip"] if str(x)]
    if cfg.semantic_lexicon_path:
        resolved = resolve_tagging_path(
            cfg.semantic_lexicon_path,
            base_dir=evidence_path.parent,
            repo_root=find_repo_root(evidence_path.parent),
        )
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"semantic lexicon must be mapping: {resolved}")
        for key, attr in _LEXICON_KEYS.items():
            if not lists[attr]:
                lists[attr] = _clean_list(raw.get(key))
        if punct_override is None and raw.get("punctuation_to_strip"):
            cfg.punctuation_to_strip = [str(x) for x in raw["punctuation_to_strip"] if str(x)]
    if punct_override is not None:
        cfg.punctuation_to_strip = punct_override
    if not lists["filler_phrases"]:
        lists["filler_phrases"] = ["嗯嗯", "嗯", "啊", "哦", "呃", "额", "唔"]
    if not lists["affirmation_phrases"]:
        lists["affirmation_phrases"] = list(lists["positive_phrases"]) or [
            "需要",
            "可以",
            "是",
            "有",
            "好的",
            "好",
        ]
    for attr, values in lists.items():
        setattr(cfg, attr, values)


def migrate_evidence_from_selection_yaml(selection_path: str | Path) -> TagEvidenceConfig:
    """Build TagEvidenceConfig from a selection YAML's *current effective* values.

    Import of SelectionV3Config is confined to this migration boundary.
    """
    from audio_engine.core.selection_v3.config import SelectionV3Config

    sel = SelectionV3Config.from_yaml(selection_path)
    cfg = TagEvidenceConfig(
        policy_version="tag_evidence_v2",
        model_families={k: list(v) for k, v in sel.model_families.items()},
        teacher_families=list(sel.teacher_families),
        target_family=str(sel.target_family),
        expected_runs_per_family=int(sel.expected_runs_per_family or 2),
        run_aliases=dict(sel.run_aliases or {}),
        punctuation_to_strip=list(sel.punctuation_to_strip or list(_DEFAULT_PUNCT)),
        critical_short_responses=list(
            sel.critical_short_responses or list(_DEFAULT_CRITICAL_SHORT)
        ),
        family_threshold=float(getattr(sel, "family_threshold", 0.98) or 0.98),
        short_polarity_max_han_chars=int(getattr(sel, "short_polarity_max_han_chars", 4) or 4),
        classify_text_policy=str(sel.classify_text_policy or "five_class_v1"),
        classify_text_keep_digits=bool(sel.classify_text_keep_digits),
        classify_text_echo_missing=str(sel.classify_text_echo_missing or "fail"),
        classify_text_echo=dict(sel.classify_text_echo or {}),
        voicemail_patterns_path=str(sel.voicemail_patterns_path or ""),
        semantic_lexicon_path=str(sel.semantic_lexicon_path or ""),
        dnsmos_weak_sig=float(getattr(sel, "dnsmos_weak_sig", 2.8) or 2.8),
        silero_vad_config_path=str(getattr(sel, "silero_vad_config_path", "") or ""),
        audio_energy_config_path=str(
            getattr(sel, "audio_energy_config_path", "")
            or getattr(sel, "audio_energy_policy_path", "")
            or ""
        ),
        dnsmos_config_path=str(getattr(sel, "dnsmos_config_path", "") or ""),
        negative_phrases=list(sel.negative_phrases or []),
        positive_phrases=list(sel.positive_phrases or []),
        critical_tokens=list(sel.critical_tokens or []),
        filler_phrases=list(sel.filler_phrases or []),
        affirmation_phrases=list(sel.affirmation_phrases or []),
        profanity_or_reject=list(sel.profanity_or_reject or []),
        source_path=f"migrated_from:{Path(selection_path).as_posix()}",
    )
    cfg.content_digest = cfg.fingerprint()
    return cfg


def write_evidence_yaml(cfg: TagEvidenceConfig, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "policy_version": cfg.policy_version,
        "model_families": cfg.model_families,
        "teacher_families": cfg.teacher_families,
        "target_family": cfg.target_family,
        "expected_runs_per_family": cfg.expected_runs_per_family,
        "run_aliases": cfg.run_aliases,
        "punctuation_to_strip": cfg.punctuation_to_strip,
        "critical_short_responses": cfg.critical_short_responses,
        "family_threshold": cfg.family_threshold,
        "short_polarity_max_han_chars": cfg.short_polarity_max_han_chars,
        "classify_text": {
            "policy": cfg.classify_text_policy,
            "keep_digits": cfg.classify_text_keep_digits,
            "echo_missing": cfg.classify_text_echo_missing,
            "echo": cfg.classify_text_echo,
        },
        "voicemail_patterns_path": cfg.voicemail_patterns_path,
        "semantic_lexicon_path": cfg.semantic_lexicon_path,
        "dnsmos_weak_sig": cfg.dnsmos_weak_sig,
        "acoustic": {
            "silero_vad_config": cfg.silero_vad_config_path,
            "audio_energy_config": cfg.audio_energy_config_path,
            "dnsmos_config": cfg.dnsmos_config_path,
        },
    }
    tmp = p.with_suffix(".tmp")
    tmp.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    tmp.replace(p)
    return p
