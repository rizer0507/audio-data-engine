"""Shared pre-layer: Chinese-only classify text (025 / classify_text_zh_only_v1).

Classification, consensus, semantic checks, gold selection, and speech-rate
consume ``classify_text``. Control tags, Unicode punctuation/symbols, Latin and
other writing systems do not vote. Prompt / whole-table hotword echoes are
empty routes. Digits stay as slots. This module does not rewrite ``raw_text``.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from audio_engine.core.selection_v3.text import transcript_text
from audio_engine.core.selection_v3.types import (
    RUN_STATUS_SUCCESS_EMPTY,
    RUN_STATUS_SUCCESS_TEXT,
)

CLASSIFY_TEXT_VERSION = "classify_text_zh_only_v1"
POLICY_LEGACY = "legacy"
POLICY_CHINESE_ONLY = "chinese_only_v1"

EMPTY_CONTROL_TAG_ONLY = "control_tag_only"
EMPTY_PUNCT_OR_SYMBOL = "punctuation_or_symbol_only"
EMPTY_NON_CHINESE = "non_chinese_discarded"
EMPTY_PROMPT_ECHO = "prompt_echo"
EMPTY_HOTWORD_ECHO = "hotword_echo"

_VOCAB_PREFIX_RE = re.compile(r"^\s*vocabulary\s*:\s*", re.IGNORECASE)
_HOTWORD_SPLIT_RE = re.compile(r"[/，,|]+")
_LATIN_RUN_RE = re.compile(r"[A-Za-z]+")
_HAS_CONTROL_TAG_RE = re.compile(r"<\|.*?\|>|<EMO_[A-Za-z0-9_]+>\|?|</?EMO_[A-Za-z0-9_]+>", re.IGNORECASE)

_DEFAULT_ECHO_FILES = {
    "qwen": {
        "asr_config": "configs/asr/qwen_asr.yaml",
        "blank_exact": "configs/normalization/blank_exact_qwen_v1.yaml",
    },
    "kimi": {"asr_config": "configs/asr/kimi.yaml"},
    "glm": {"asr_config": "configs/asr/glm.yaml"},
    "sensevoice": {"asr_config": "configs/asr/sensevoice.yaml"},
}


def uses_chinese_only_text(policy: str | None) -> bool:
    text = str(policy or POLICY_LEGACY).strip().lower()
    return text in {POLICY_CHINESE_ONLY, "chinese_only", CLASSIFY_TEXT_VERSION}


def normalize_classify_text_policy(value: Any) -> str:
    text = str(value or POLICY_LEGACY).strip().lower()
    if text in {"", POLICY_LEGACY, "off", "false", "0"}:
        return POLICY_LEGACY
    if text in {POLICY_CHINESE_ONLY, "chinese_only", CLASSIFY_TEXT_VERSION, "on", "true", "1"}:
        return POLICY_CHINESE_ONLY
    raise ValueError(
        "classify_text_policy must be 'legacy' or 'chinese_only_v1', "
        f"got {value!r}"
    )


def is_cjk_ideograph(ch: str) -> bool:
    code = ord(ch)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x2A6DF
        or 0x2A700 <= code <= 0x2B73F
        or 0x2B740 <= code <= 0x2B81F
        or 0x2B820 <= code <= 0x2CEAF
        or 0x2CEB0 <= code <= 0x2EBEF
        or 0x30000 <= code <= 0x3134F
    )


def is_latin_letter(ch: str) -> bool:
    if not unicodedata.category(ch).startswith("L"):
        return False
    return "LATIN" in unicodedata.name(ch, "")


def is_ascii_digit(ch: str) -> bool:
    return "0" <= ch <= "9"


def is_other_letter_script(ch: str) -> bool:
    if not unicodedata.category(ch).startswith("L"):
        return False
    if is_cjk_ideograph(ch) or is_latin_letter(ch):
        return False
    return True


def strip_punct_symbols(text: str) -> str:
    """Drop Unicode P*/S*/Z*/C*. Letters and decimal digits stay."""
    kept: list[str] = []
    for ch in text:
        major = unicodedata.category(ch)[:1]
        if major in {"P", "S", "Z", "C"}:
            continue
        kept.append(ch)
    return "".join(kept)


def echo_form(value: Any) -> str:
    """Tag-strip + NFKC + punct/whitespace collapse. Latin is kept for prompt match."""
    body = transcript_text(value)
    body = unicodedata.normalize("NFKC", body)
    body = _VOCAB_PREFIX_RE.sub("", body)
    body = strip_punct_symbols(body)
    return "".join(body.split())


def pre_filter_language(body_no_tag: str) -> str:
    """Language of the tag-stripped body, before Chinese-only emptying (023)."""
    from audio_engine.core.selection_v3.noise_trigger import assess_speech_language

    return str(assess_speech_language(body_no_tag).get("label") or "empty")


def _latin_triggers_discard(compact: str) -> bool:
    """Latin/other letters empty the route; 2–6 uppercase abbrevs beside CJK do not."""
    if any(is_other_letter_script(ch) for ch in compact):
        return True
    has_cjk = any(is_cjk_ideograph(ch) for ch in compact)
    runs = _LATIN_RUN_RE.findall(compact)
    if not runs:
        return False
    for token in runs:
        if has_cjk and token.isupper() and 2 <= len(token) <= 6:
            continue
        if len(token) >= 3:
            return True
        if not has_cjk and len(token) >= 2:
            return True
        if has_cjk and not token.isupper() and len(token) >= 2:
            return True
    return False


def classify_chars(compact: str, *, keep_digits: bool = True) -> str:
    chars: list[str] = []
    for ch in compact:
        if is_cjk_ideograph(ch):
            chars.append(ch)
        elif keep_digits and is_ascii_digit(ch):
            chars.append(ch)
    return "".join(chars)


@dataclass(frozen=True)
class EchoTable:
    """Prompt / whole-table dump / explicit whole-sentence phrases for one family."""

    prompts: tuple[str, ...] = ()
    table_dumps: tuple[str, ...] = ()
    exact_phrases: tuple[str, ...] = ()
    source_paths: tuple[str, ...] = ()
    fingerprint: str = ""

    def match(self, normalized_body: str) -> tuple[str, ...]:
        if not normalized_body:
            return ()
        reasons: list[str] = []
        for prompt in self.prompts:
            form = echo_form(prompt)
            if form and form == normalized_body:
                reasons.append(EMPTY_PROMPT_ECHO)
                break
        dump_hit = False
        for dump in self.table_dumps:
            form = echo_form(dump)
            if form and form == normalized_body:
                dump_hit = True
                break
        if not dump_hit:
            for phrase in self.exact_phrases:
                form = echo_form(phrase)
                if form and form == normalized_body:
                    dump_hit = True
                    break
        if dump_hit:
            reasons.append(EMPTY_HOTWORD_ECHO)
        return tuple(dict.fromkeys(reasons))


def echo_fingerprint(tables: Mapping[str, EchoTable]) -> str:
    payload = {
        family: {
            "prompts": list(table.prompts),
            "table_dumps": list(table.table_dumps),
            "exact_phrases": list(table.exact_phrases),
        }
        for family, table in sorted(tables.items())
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ClassifyTextResult:
    raw_text: str
    transcript_text: str
    classify_text: str
    empty_reason_codes: tuple[str, ...]
    pre_filter_language: str
    status: str
    policy: str
    version: str = CLASSIFY_TEXT_VERSION
    had_control_tags: bool = False

    @property
    def is_empty(self) -> bool:
        return self.status == RUN_STATUS_SUCCESS_EMPTY or not self.classify_text


def prepare_classify_text(
    raw: Any,
    *,
    echo: EchoTable | None = None,
    keep_digits: bool = True,
    policy: str = POLICY_CHINESE_ONLY,
) -> ClassifyTextResult:
    """Apply the 025 order to one route. Does not mutate ``raw``."""
    raw_text = "" if raw is None else str(raw)
    had_tags = bool(_HAS_CONTROL_TAG_RE.search(raw_text))
    body_no_tag = transcript_text(raw_text)
    language = pre_filter_language(body_no_tag)
    echo_norm = echo_form(body_no_tag)
    reasons: list[str] = []

    if echo is not None:
        reasons.extend(echo.match(echo_norm))

    nfkc = unicodedata.normalize("NFKC", body_no_tag)
    compact = strip_punct_symbols(nfkc)
    if not compact and not reasons:
        if had_tags or (raw_text and not body_no_tag):
            reasons.append(EMPTY_CONTROL_TAG_ONLY)
        if raw_text.strip() and (body_no_tag or had_tags):
            reasons.append(EMPTY_PUNCT_OR_SYMBOL)
        if not reasons:
            reasons.append(EMPTY_PUNCT_OR_SYMBOL)

    if compact and _latin_triggers_discard(compact):
        reasons.append(EMPTY_NON_CHINESE)

    chinese = classify_chars(compact, keep_digits=keep_digits) if not reasons else ""
    if not reasons and not chinese:
        reasons.append(EMPTY_PUNCT_OR_SYMBOL)

    unique = tuple(dict.fromkeys(reasons))
    if unique:
        return ClassifyTextResult(
            raw_text=raw_text,
            transcript_text=body_no_tag,
            classify_text="",
            empty_reason_codes=unique,
            pre_filter_language=language,
            status=RUN_STATUS_SUCCESS_EMPTY,
            policy=normalize_classify_text_policy(policy),
            had_control_tags=had_tags,
        )
    return ClassifyTextResult(
        raw_text=raw_text,
        transcript_text=body_no_tag,
        classify_text=chinese,
        empty_reason_codes=(),
        pre_filter_language=language,
        status=RUN_STATUS_SUCCESS_TEXT,
        policy=normalize_classify_text_policy(policy),
        had_control_tags=had_tags,
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"echo source must be a mapping: {path}")
    return raw


def _vocab_dump_from_context(context: str) -> str:
    text = _VOCAB_PREFIX_RE.sub("", str(context or "").strip())
    if not text:
        return ""
    parts = [echo_form(token) for token in _HOTWORD_SPLIT_RE.split(text)]
    return "".join(part for part in parts if part)


def _phrases_from_blank_exact(data: Mapping[str, Any]) -> list[str]:
    block = data.get("blank_exact_hotwords") or data
    if isinstance(block, Mapping):
        raw = block.get("hotwords") or block.get("vocabulary") or block.get("words") or []
    else:
        raw = block
    if isinstance(raw, str):
        return [raw] if raw.strip() else []
    return [str(item) for item in (raw or []) if str(item).strip()]


def load_echo_table_from_files(
    *,
    asr_config: str | Path | None = None,
    blank_exact: str | Path | None = None,
    extra_exact: Iterable[str] = (),
    missing: str = "fail",
) -> EchoTable:
    prompts: list[str] = []
    dumps: list[str] = []
    phrases: list[str] = []
    paths: list[str] = []

    def _require(path: Path) -> dict[str, Any] | None:
        if path.exists():
            paths.append(path.as_posix())
            return _read_yaml(path)
        if missing == "fail":
            raise FileNotFoundError(
                f"classify_text echo source missing: {path} "
                "(set classify_text.echo_missing=echo_list_missing to record the gap)"
            )
        return None

    if asr_config:
        data = _require(Path(asr_config))
        if data is not None:
            prompt = str(data.get("prompt") or "").strip()
            context = str(data.get("context") or "").strip()
            if prompt:
                prompts.append(prompt)
            if context:
                prompts.append(context)
                dump = _vocab_dump_from_context(context)
                if dump:
                    dumps.append(dump)
    if blank_exact:
        data = _require(Path(blank_exact))
        if data is not None:
            phrases.extend(_phrases_from_blank_exact(data))
    phrases.extend(str(item) for item in extra_exact if str(item).strip())
    table = EchoTable(
        prompts=tuple(dict.fromkeys(prompts)),
        table_dumps=tuple(dict.fromkeys(dumps)),
        exact_phrases=tuple(dict.fromkeys(phrases)),
        source_paths=tuple(paths),
    )
    return EchoTable(
        prompts=table.prompts,
        table_dumps=table.table_dumps,
        exact_phrases=table.exact_phrases,
        source_paths=table.source_paths,
        fingerprint=echo_fingerprint({"_": table}),
    )


def default_echo_spec_for_family(family: str) -> dict[str, str]:
    return dict(_DEFAULT_ECHO_FILES.get(str(family).strip().lower()) or {})


def load_echo_tables(
    *,
    families: Iterable[str],
    echo_cfg: Mapping[str, Any] | None = None,
    extra_exact: Iterable[str] = (),
    missing: str = "fail",
    base_dir: str | Path | None = None,
) -> dict[str, EchoTable]:
    """Load per-family echo tables. Missing listed files fail unless ``echo_list_missing``."""
    root = Path(base_dir) if base_dir is not None else Path.cwd()
    configured = dict(echo_cfg or {})
    tables: dict[str, EchoTable] = {}
    shared_extra = [str(item) for item in extra_exact if str(item).strip()]
    family_names = [str(name).strip() for name in families if str(name).strip()]
    for extra in configured:
        name = str(extra).strip()
        if name and name not in family_names:
            family_names.append(name)
    for family in family_names:
        spec = configured.get(family)
        if spec is None:
            spec = default_echo_spec_for_family(family)
        if not spec:
            tables[family] = EchoTable()
            continue
        if isinstance(spec, EchoTable):
            tables[family] = spec
            continue
        if not isinstance(spec, Mapping):
            raise ValueError(f"classify_text.echo.{family} must be a mapping")
        asr = spec.get("asr_config") or spec.get("asr")
        blank = spec.get("blank_exact") or spec.get("blank_exact_path")
        extra = list(spec.get("extra_exact") or spec.get("exact_phrases") or [])
        extra.extend(shared_extra)

        def _resolve(value: Any) -> str | None:
            text = str(value or "").strip()
            if not text:
                return None
            path = Path(text)
            if not path.is_absolute():
                path = root / path
            return str(path)

        tables[family] = load_echo_table_from_files(
            asr_config=_resolve(asr),
            blank_exact=_resolve(blank),
            extra_exact=extra,
            missing=missing,
        )
    return tables


def echo_for_family(tables: Mapping[str, EchoTable] | None, family: str | None) -> EchoTable | None:
    if not tables:
        return None
    if family and family in tables:
        return tables[family]
    return None


def attach_classify_text_fields(
    labels_or_result: Any,
    *,
    policy: str,
    fingerprint: str,
    empty_reason_by_run: Mapping[str, list[str]] | Mapping[str, tuple[str, ...]],
    pre_filter_language_by_run: Mapping[str, str],
    classify_text_by_run: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    payload = {
        "classify_text_policy": normalize_classify_text_policy(policy),
        "classify_text_version": CLASSIFY_TEXT_VERSION
        if uses_chinese_only_text(policy)
        else "",
        "classify_text_echo_fingerprint": fingerprint or "",
        "empty_reason_by_run": {key: list(value) for key, value in empty_reason_by_run.items()},
        "pre_filter_language_by_run": dict(pre_filter_language_by_run),
        "classify_text_by_run": dict(classify_text_by_run or {}),
    }
    if hasattr(labels_or_result, "classify_text_policy"):
        labels_or_result.classify_text_policy = payload["classify_text_policy"]
        labels_or_result.classify_text_version = payload["classify_text_version"]
        labels_or_result.classify_text_echo_fingerprint = payload["classify_text_echo_fingerprint"]
        labels_or_result.empty_reason_by_run = payload["empty_reason_by_run"]
        labels_or_result.pre_filter_language_by_run = payload["pre_filter_language_by_run"]
        labels_or_result.classify_text_by_run = payload["classify_text_by_run"]
    return payload


def audit_routes(config: Any, routes: Iterable[Any]) -> dict[str, Any]:
    empty_reason: dict[str, list[str]] = {}
    pre_lang: dict[str, str] = {}
    classify_map: dict[str, str] = {}
    for route in routes:
        run_id = str(getattr(route, "run_id", "") or "")
        if not run_id:
            continue
        layers = getattr(route, "layers", None)
        if layers is not None:
            empty_reason[run_id] = list(getattr(layers, "empty_reason_codes", ()) or ())
            pre_lang[run_id] = str(getattr(layers, "pre_filter_language", "") or "")
            classify_map[run_id] = str(getattr(layers, "classify_text", "") or "")
        else:
            empty_reason[run_id] = list(getattr(route, "empty_reason_codes", ()) or ())
            pre_lang[run_id] = str(getattr(route, "pre_filter_language", "") or "")
            classify_map[run_id] = str(getattr(route, "classify_text", "") or "")
    return attach_classify_text_fields(
        None,
        policy=str(getattr(config, "classify_text_policy", POLICY_LEGACY) or POLICY_LEGACY),
        fingerprint=str(getattr(config, "classify_text_echo_fingerprint", "") or ""),
        empty_reason_by_run=empty_reason,
        pre_filter_language_by_run=pre_lang,
        classify_text_by_run=classify_map,
    )


def apply_route_audit(result: Any, config: Any, routes: Iterable[Any]) -> Any:
    payload = audit_routes(config, routes)
    attach_classify_text_fields(
        result,
        policy=payload["classify_text_policy"],
        fingerprint=payload["classify_text_echo_fingerprint"],
        empty_reason_by_run=payload["empty_reason_by_run"],
        pre_filter_language_by_run=payload["pre_filter_language_by_run"],
        classify_text_by_run=payload["classify_text_by_run"],
    )
    return result
