"""Optional semantic verifier. Local rules are the default; no network assumed.

Verdicts are ``equivalent`` / ``conflict`` / ``unknown``. Missing service,
timeout, invalid structure, invalid citations, or thin evidence all become
``unknown``. Unknown must not be treated as a pass or as a confirmed semantic-risk.
Unknown is also not business equivalence.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from audio_engine.core.selection_v3.types import VERIFIER_VERSION_LOCAL

VERIFIER_ENV_ENDPOINT = "AUDIO_ENGINE_SEMANTIC_VERIFIER_ENDPOINT"
_ASR_ENDPOINT_MARKERS = ("/audio/transcriptions", "/v1/audio", "transcri")
_CHAT_ENDPOINT_MARKERS = ("/chat/completions", "/v1/chat", "/chat")
_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_CHAT_PROMPT_VERSION = "business_compare_chat_v1"

_FLIP_PAIRS = (
    ("不好", "好"),
    ("不需要", "需要"),
    ("不同意", "同意"),
    ("不可以", "可以"),
    ("不要", "要"),
    ("不是", "是"),
    ("没有", "有"),
    ("不行", "行"),
    ("不用", "要"),
    ("不能", "能"),
    ("不想", "想"),
    ("不要再打", "再打"),
    ("一万", "一千"),
    ("一千", "一签"),
    ("一万", "一签"),
)

_AMOUNT_TOKEN = re.compile(r"[零〇一二三四五六七八九十百千万亿0-9]+")
_EN_NEG = (
    "i do not want",
    "i don't want",
    "i dont want",
    "i disagree",
    "do not call",
    "don't call",
)
_EN_POS = (
    "i want it",
    "i want",
    "i agree",
    "i need it",
    "i need",
)
_ZH_NEG = ("我不要", "不需要", "不同意", "不要再打", "不想")
_ZH_POS = ("我要", "需要", "同意", "再打给我", "想要")


@dataclass(frozen=True)
class VerifyRequest:
    left_raw: str
    right_raw: str
    left_transcript: str
    right_transcript: str
    left_language: str
    right_language: str
    aligned_diff: str = ""
    rule_version: str = ""
    prompt_version: str = ""
    model_version: str = ""
    context_digest: str = ""
    lexicon_version: str = ""


@dataclass(frozen=True)
class VerifyResult:
    verdict: str
    conflict_type: str | None
    citations: tuple[str, ...]
    version: str
    error: str | None = None
    affects_business: bool | None = None

    @property
    def ok_structure(self) -> bool:
        return self.verdict in {"equivalent", "conflict", "unknown"}


@dataclass
class SemanticEvidence:
    kind: str
    verdict: str
    families: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    version: str = ""
    conflict_type: str | None = None


def cache_key(request: VerifyRequest, *, verifier_version: str) -> str:
    payload = {
        "left": request.left_raw,
        "right": request.right_raw,
        "lt": request.left_transcript,
        "rt": request.right_transcript,
        "ll": request.left_language,
        "rl": request.right_language,
        "rule": request.rule_version,
        "prompt": request.prompt_version,
        "model": request.model_version,
        "verifier": verifier_version,
        "context": request.context_digest,
        "lexicon": request.lexicon_version,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _aligned_diff(left: str, right: str) -> tuple[str, str]:
    """Longest shared prefix/suffix; remaining spans are the proposition diff."""
    if left == right:
        return "", ""
    prefix = 0
    limit = min(len(left), len(right))
    while prefix < limit and left[prefix] == right[prefix]:
        prefix += 1
    suffix = 0
    while (
        suffix < (len(left) - prefix)
        and suffix < (len(right) - prefix)
        and left[-1 - suffix] == right[-1 - suffix]
    ):
        suffix += 1
    end_l = len(left) - suffix if suffix else len(left)
    end_r = len(right) - suffix if suffix else len(right)
    return left[prefix:end_l], right[prefix:end_r]


def local_polarity_conflict(left: str, right: str) -> SemanticEvidence | None:
    """Clear yes/no flip on the same proposition. Identical text is never a conflict."""
    a = str(left or "")
    b = str(right or "")
    if not a or not b or a == b:
        return None
    da, db = _aligned_diff(a, b)
    spans = {da, db}
    for neg, pos in _FLIP_PAIRS:
        if spans == {neg, pos} or spans == {pos, neg}:
            return SemanticEvidence(
                kind="polarity_flip",
                verdict="conflict",
                citations=[a, b, f"{da}|{db}"],
                version=VERIFIER_VERSION_LOCAL,
                conflict_type="polarity_flip",
            )
    # Rejection wiped: one span is 不/没/别 plus the other span, rest aligned.
    if da in {"不", "没", "沒", "别", "別"} and db == "":
        return SemanticEvidence(
            kind="rejection_sanitization",
            verdict="conflict",
            citations=[a, b],
            version=VERIFIER_VERSION_LOCAL,
            conflict_type="rejection_sanitization",
        )
    if db in {"不", "没", "沒", "别", "別"} and da == "":
        return SemanticEvidence(
            kind="rejection_sanitization",
            verdict="conflict",
            citations=[a, b],
            version=VERIFIER_VERSION_LOCAL,
            conflict_type="rejection_sanitization",
        )
    if _amount_slot_conflict(da, db):
        return SemanticEvidence(
            kind="critical_slot",
            verdict="conflict",
            citations=[a, b, f"{da}|{db}"],
            version=VERIFIER_VERSION_LOCAL,
            conflict_type="critical_slot",
        )
    return None


def _amount_slot_conflict(left_span: str, right_span: str) -> bool:
    if not left_span or not right_span or left_span == right_span:
        return False
    if left_span in {"一千", "一万", "一签", "一百"} and right_span in {
        "一千",
        "一万",
        "一签",
        "一百",
    }:
        return True
    if _AMOUNT_TOKEN.fullmatch(left_span) and _AMOUNT_TOKEN.fullmatch(right_span):
        return left_span != right_span
    return False


def bilingual_polarity(zh_text: str, en_text: str) -> str:
    """Local cross-language polarity. Unmatched English is unknown, not a vote."""
    en = " ".join(str(en_text or "").lower().split())
    zh = str(zh_text or "")
    en_neg = any(p in en for p in _EN_NEG)
    en_pos = (not en_neg) and any(p in en for p in _EN_POS)
    zh_neg = any(p in zh for p in _ZH_NEG)
    zh_pos = (not zh_neg) and any(p in zh for p in _ZH_POS)
    if not (en_neg or en_pos) or not (zh_neg or zh_pos):
        return "unknown"
    if zh_neg and en_neg:
        return "equivalent"
    if zh_pos and en_pos:
        return "equivalent"
    if (zh_neg and en_pos) or (zh_pos and en_neg):
        return "conflict"
    return "unknown"


class SemanticVerifier(Protocol):
    version: str

    def verify(self, request: VerifyRequest) -> VerifyResult: ...


class LocalSemanticVerifier:
    """Deterministic local rules. Complex residuals stay unknown."""

    def __init__(self) -> None:
        self.version = VERIFIER_VERSION_LOCAL
        self._cache: dict[str, VerifyResult] = {}

    def verify(self, request: VerifyRequest) -> VerifyResult:
        key = cache_key(request, verifier_version=self.version)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        result = self._verify_uncached(request)
        self._cache[key] = result
        return result

    def _verify_uncached(self, request: VerifyRequest) -> VerifyResult:
        left = request.left_transcript
        right = request.right_transcript
        if left == right and left:
            return VerifyResult("equivalent", None, (left,), self.version)
        langs = {request.left_language, request.right_language}
        if "en" in langs and "zh" in langs:
            zh = left if request.left_language == "zh" else right
            en = right if request.right_language == "en" else left
            verdict = bilingual_polarity(zh, en)
            return VerifyResult(
                verdict,
                "cross_language_polarity" if verdict == "conflict" else None,
                (zh, en),
                self.version,
                None if verdict != "unknown" else "cross_language_unmatched",
            )
        hit = local_polarity_conflict(left, right)
        if hit is not None:
            return VerifyResult(
                "conflict",
                hit.conflict_type,
                tuple(hit.citations),
                self.version,
            )
        # Residual character differences are not auto-equivalent.
        return VerifyResult(
            "unknown",
            None,
            (left, right),
            self.version,
            "residual_unverified",
        )


class NullSemanticVerifier:
    """Used when the optional layer is disabled. Never claims a pass."""

    version = "null_verifier_v1"

    def verify(self, request: VerifyRequest) -> VerifyResult:
        return VerifyResult(
            "unknown",
            None,
            (request.left_transcript, request.right_transcript),
            self.version,
            "verifier_disabled",
        )


class UnavailableSemanticVerifier:
    """HTTP/model adapter stand-in. No endpoint means unknown, never a call."""

    def __init__(self, *, endpoint: str = "", timeout_sec: float = 5.0, max_retries: int = 1) -> None:
        self.endpoint = endpoint.strip()
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.version = "http_verifier_unconfigured_v1"
        self.calls = 0

    def verify(self, request: VerifyRequest) -> VerifyResult:
        self.calls += 1
        if not self.endpoint:
            return VerifyResult(
                "unknown",
                None,
                (request.left_transcript, request.right_transcript),
                self.version,
                "no_endpoint",
            )
        # This demand does not upload audio or call paid services.
        return VerifyResult(
            "unknown",
            None,
            (request.left_transcript, request.right_transcript),
            self.version,
            "external_call_disabled",
        )


def coerce_verifier_verdict(
    payload: Any,
    *,
    timed_out: bool = False,
    version: str = "",
    citations: tuple[str, ...] = (),
) -> VerifyResult:
    """Map a remote payload to a verdict. Timeout and bad structure are unknown."""
    if timed_out:
        return VerifyResult("unknown", None, citations, version, "timeout")
    if not isinstance(payload, dict):
        return VerifyResult("unknown", None, citations, version, "invalid_structure")
    verdict = payload.get("verdict")
    if verdict not in {"equivalent", "conflict", "unknown"}:
        return VerifyResult("unknown", None, citations, version, "invalid_structure")
    conflict_type = payload.get("conflict_type")
    if conflict_type is not None and not isinstance(conflict_type, str):
        return VerifyResult("unknown", None, citations, version, "invalid_structure")
    raw_citations = payload.get("citations", citations)
    if not isinstance(raw_citations, (list, tuple)) or not all(isinstance(item, str) for item in raw_citations):
        return VerifyResult("unknown", None, citations, version, "invalid_structure")
    return VerifyResult(
        str(verdict),
        conflict_type,
        tuple(raw_citations),
        version or str(payload.get("version") or ""),
        None if verdict != "unknown" else str(payload.get("error") or "unknown"),
    )


def citation_is_grounded(citation: str, *sources: str) -> bool:
    value = str(citation or "").strip()
    if not value:
        return False
    return any(value in str(source or "") for source in sources)


def enforce_citation_policy(result: VerifyResult, *sources: str) -> VerifyResult:
    """A non-unknown verdict must quote text that actually appears in the inputs."""
    if result.verdict == "unknown":
        return result
    if not result.citations or not all(citation_is_grounded(item, *sources) for item in result.citations):
        return VerifyResult(
            "unknown",
            None,
            result.citations,
            result.version,
            "invalid_citation",
            affects_business=None,
        )
    return result


@dataclass(frozen=True)
class BusinessFields:
    speech_act: str
    intent: str
    polarity: str
    negation_scope: str
    subject: str
    slots: tuple[tuple[str, str], ...]
    dialogue_state: str
    next_action: str
    commitment: str
    evidence: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "speech_act": self.speech_act,
            "intent": self.intent,
            "polarity": self.polarity,
            "negation_scope": self.negation_scope,
            "subject": self.subject,
            "slots": [list(item) for item in self.slots],
            "dialogue_state": self.dialogue_state,
            "next_action": self.next_action,
            "commitment": self.commitment,
            "evidence": self.evidence,
        }


_UNKNOWN_FIELDS = BusinessFields(
    "unknown", "unknown", "unknown", "unknown", "unknown", (), "unknown", "unknown", "unknown", ""
)

_AMOUNT_SLOT = re.compile(r"[0-9零〇一二三四五六七八九十百千万亿]+(?:元|块|万|千)")
_TIME_SLOT = re.compile(r"(?:今天|明天|后天|大后天|[0-9一二三四五六七八九十]+点(?:半)?)")
_CONTINUE = (
    "你说",
    "你说吧",
    "嗯你说",
    "啊你说",
    "哦你说",
    "唔你说",
    "你讲",
    "你讲吧",
    "请说",
    "请讲",
    "继续说",
    "你继续说",
    "你继续讲",
)
_REFUSAL = (
    "不需要",
    "不用了",
    "不要再打",
    "别打",
    "不想",
    "不考虑",
    "拒绝",
    "没兴趣",
    "不去了",
    "不要",
    "不用",
    "不行",
)
_ABSENCE = ("他歇了", "她歇了", "他不在", "她不在", "他出去了", "人不在", "他休息了")
_CALLBACK = (
    "稍后联系",
    "稍后再联系",
    "回头打",
    "回头再打",
    "等会再打",
    "等一下再打",
    "待会儿联系",
    "一会再打",
    "晚点再打",
    "回头联系",
)
_IDENTITY = (
    "你是谁",
    "哪位",
    "您是哪位",
    "什么事",
    "找谁",
    "请问你是",
    "请问您是",
    "哪里",
    "谁啊",
    "谁呀",
    "怎么了",
    "哪位啊",
    "你哪位",
)
_VOICEMAIL = (
    "语音留言",
    "语音信箱",
    "转至语音",
    "已转留言",
    "提示音后留言",
    "滴声后留言",
    "请你留言",
    "请您留言",
    "来电原因我会帮你确认",
    "通知机主",
    "屏幕留言",
    "录音超时",
    "通信助理",
    "通讯助理",
    "电话助理",
    "语音助理",
)
_BRIEF = frozenset(
    {
        "嗯",
        "哦",
        "好",
        "行",
        "对",
        "啊",
        "嗯嗯",
        "啊啊",
        "好好",
        "哦哦",
        "嗯对",
        "对对",
        "哎",
        "好啊",
        "行啊",
        "对啊",
        "嗯啊",
        "唔",
        "嗯哼",
    }
)
_BARE_AFFIRM = frozenset({"需要", "要", "可以", "好的", "好吧", "行吧", "是", "是的", "是啊", "有", "有啊"})
_GREET_EXACT = frozenset({"你好", "您好", "喂", "哎你好", "喂你好", "诶你好", "嗨你好", "你好对", "喂你好对"})
_REFUSAL_EXACT = frozenset(
    {
        "没有",
        "没有了",
        "没有没有",
        "没有没有没有",
        "没有谢谢",
        "没有了谢谢",
        "没有了谢谢啊",
        "啊没有",
        "暂时没有",
        "不是",
        "不是的",
        "不是不是",
        "啊没有谢谢",
        "没有嗯",
    }
)
_REFUSAL_PREFIXES = ("没有了谢谢", "没有谢谢", "没有没有", "暂时没有", "啊没有")


def _comparison_body(text: str) -> str:
    from audio_engine.core.selection_v3.text_tolerance import comparison_form

    body, _changed = comparison_form(text)
    return body.replace("您", "你")


def same_business_wording(left: str, right: str) -> bool:
    """Punctuation and one trailing particle do not change the business body.

    ``需要`` / ``不需要`` stay different: negation is never an edge particle.
    """
    if left == right and str(left or "").strip():
        return True
    a = _comparison_body(left)
    b = _comparison_body(right)
    if a == b and a:
        return True
    from audio_engine.core.selection_v3.text_tolerance import apply_tolerance_key

    ka = apply_tolerance_key(a)
    kb = apply_tolerance_key(b)
    return bool(ka and ka == kb)


def _is_short_refusal(body: str) -> bool:
    if body in _REFUSAL_EXACT:
        return True
    if "有没有" in body:
        return False
    if any(body.startswith(prefix) for prefix in _REFUSAL_PREFIXES) and len(body) <= 12:
        return True
    return False


def _is_identity(body: str) -> bool:
    if any(token in body for token in _IDENTITY):
        return True
    return len(body) <= 10 and "谁" in body


def _is_greeting(body: str) -> bool:
    if body in _GREET_EXACT:
        return True
    if "谁" in body or "哪" in body:
        return False
    return "你好" in body and len(body) <= 8 and body.startswith(("喂", "哎", "诶", "嗨", "啊", "哦"))


def _slots_of(text: str) -> tuple[tuple[str, str], ...]:
    found: list[tuple[str, str]] = []
    for kind, pattern in (("amount", _AMOUNT_SLOT), ("time", _TIME_SLOT)):
        for match in pattern.finditer(text):
            found.append((kind, match.group(0)))
    return tuple(sorted(set(found)))


def extract_business_fields(text: str) -> BusinessFields:
    """Deterministic business fields. Does not delete 不/没/别/好/行/嗯."""
    body = _comparison_body(text)
    if not body:
        return BusinessFields(
            "empty", "none", "neutral", "none", "none", (), "no_speech_text", "none", "none", ""
        )
    slots = _slots_of(body)
    subject = "third_party" if body.startswith(("他", "她")) else ("caller" if body.startswith("我") else "unspecified")
    evidence = body[:24]
    if any(token in body for token in _VOICEMAIL):
        return BusinessFields(
            "voicemail", "mailbox_prompt", "neutral", "none", "system", slots,
            "automated_prompt", "leave_message", "none", evidence,
        )
    if any(token in body for token in _ABSENCE):
        return BusinessFields(
            "absence", "third_party_unavailable", "neutral", "none", "third_party", slots,
            "party_unavailable", "no_commitment", "none", evidence,
        )
    if _is_short_refusal(body) or any(token in body for token in _REFUSAL) or body in {"不", "没", "别"}:
        intent = "decline_attendance" if "不去" in body else "decline_offer"
        return BusinessFields(
            "refusal", intent, "negative", "predicate", subject, slots,
            "closed", "stop_or_decline", "stated_refusal", evidence,
        )
    if any(token in body for token in _CALLBACK):
        return BusinessFields(
            "callback", "defer_contact", "neutral", "none", subject, slots,
            "deferred", "call_later", "undetermined", evidence,
        )
    if _is_identity(body):
        return BusinessFields(
            "identity_inquiry", "ask_identity", "neutral", "none", subject, slots,
            "opening", "identify_caller", "undetermined", evidence,
        )
    if body in _CONTINUE or any(body.startswith(token) or body.endswith(token) for token in ("你说", "你讲", "请说")):
        if "不" not in body and "没" not in body:
            return BusinessFields(
                "continue_listening", "yield_floor", "neutral", "none", "other", slots,
                "listen", "yield_floor", "undetermined", evidence,
            )
    if body in _BRIEF:
        return BusinessFields(
            "brief_response", "backchannel", "neutral", "none", "unspecified", slots,
            "listen", "undetermined", "undetermined", evidence,
        )
    if body in _BARE_AFFIRM:
        return BusinessFields(
            "affirmation_undetermined", "possible_assent", "positive", "none", subject, slots,
            "listen", "undetermined", "undetermined", evidence,
        )
    if _is_greeting(body):
        return BusinessFields(
            "greeting", "greeting", "neutral", "none", subject, slots,
            "opening", "undetermined", "undetermined", evidence,
        )
    if any(token in body for token in ("我同意", "同意办理", "可以办理", "帮我办理")):
        return BusinessFields(
            "acceptance", "stated_assent", "positive", "none", subject, slots,
            "open", "undetermined", "undetermined", evidence,
        )
    return BusinessFields(
        "unknown", "unknown", "unknown", "unknown", subject, slots,
        "unknown", "unknown", "unknown", evidence,
    )


def _fields_compatible(left: BusinessFields, right: BusinessFields) -> str:
    """equivalent / conflict / insufficient. Unknown does not confirm the other side."""
    if left.speech_act == "empty" or right.speech_act == "empty":
        if left.speech_act == "empty" and right.speech_act == "empty":
            return "insufficient"
        return "conflict"
    if left.speech_act == "unknown" or right.speech_act == "unknown":
        return "insufficient"
    if left.speech_act != right.speech_act:
        return "conflict"
    if left.polarity != right.polarity and "unknown" not in {left.polarity, right.polarity}:
        return "conflict"
    if "unknown" in {left.polarity, right.polarity} and left.polarity != right.polarity:
        return "insufficient"
    if left.intent != right.intent and "unknown" not in {left.intent, right.intent}:
        return "conflict"
    if left.subject != right.subject and left.subject not in {"unspecified", "unknown"} and right.subject not in {
        "unspecified",
        "unknown",
    }:
        return "conflict"
    if left.slots != right.slots:
        return "conflict"
    if left.next_action != right.next_action and "unknown" not in {left.next_action, right.next_action}:
        return "conflict"
    if left.commitment == "undetermined" or right.commitment == "undetermined":
        # Agreed act, but a bare acknowledgment is not confirmed authorization.
        return "equivalent"
    return "equivalent"


class BusinessLocalVerifier:
    """High-frequency business rules. Residuals stay unknown; distance does not skip them."""

    version = "business_local_verifier_v4.1"

    def __init__(self) -> None:
        self._cache: dict[str, VerifyResult] = {}
        self.calls = 0

    def verify(self, request: VerifyRequest) -> VerifyResult:
        key = cache_key(request, verifier_version=self.version)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        self.calls += 1
        result = self._verify_uncached(request)
        self._cache[key] = result
        return result

    def _verify_uncached(self, request: VerifyRequest) -> VerifyResult:
        left = request.left_transcript
        right = request.right_transcript
        if same_business_wording(left, right):
            return VerifyResult("equivalent", None, (left, right) if left != right else (left,), self.version, affects_business=False)
        langs = {request.left_language, request.right_language}
        if "en" in langs and "zh" in langs:
            zh = left if request.left_language == "zh" else right
            en = right if request.right_language == "en" else left
            verdict = bilingual_polarity(zh, en)
            return VerifyResult(
                verdict,
                "cross_language_polarity" if verdict == "conflict" else None,
                (zh, en) if zh and en else (),
                self.version,
                None if verdict != "unknown" else "cross_language_unmatched",
                affects_business=True if verdict == "conflict" else None,
            )
        hit = local_polarity_conflict(_comparison_body(left), _comparison_body(right))
        if hit is not None:
            return VerifyResult(
                "conflict",
                hit.conflict_type,
                (left, right),
                self.version,
                affects_business=True,
            )
        relation = _fields_compatible(extract_business_fields(left), extract_business_fields(right))
        if relation == "equivalent":
            return VerifyResult("equivalent", None, (left, right), self.version, affects_business=False)
        if relation == "conflict":
            return VerifyResult(
                "conflict",
                "business_field_conflict",
                (left, right),
                self.version,
                affects_business=True,
            )
        return VerifyResult("unknown", None, (left, right), self.version, "residual_unverified")


class HttpSemanticVerifier:
    """POST a text-only comparison. Timeout, bad JSON, and ungrounded citations are unknown.

    Does not upload audio. An endpoint that is set but unreachable is a failed
    call, never an automatic equivalent.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        timeout_sec: float = 5.0,
        max_retries: int = 1,
        model_version: str = "",
        prompt_version: str = "",
        transport: Any | None = None,
    ) -> None:
        self.endpoint = endpoint.strip()
        self.timeout_sec = timeout_sec
        self.max_retries = max(0, int(max_retries))
        self.model_version = model_version
        self.prompt_version = prompt_version
        self.transport = transport
        self.version = f"http_semantic_verifier_v4:{model_version or 'unspecified'}"
        self.calls = 0
        self.errors = 0

    def verify(self, request: VerifyRequest) -> VerifyResult:
        self.calls += 1
        if not self.endpoint:
            return VerifyResult(
                "unknown",
                None,
                (request.left_transcript, request.right_transcript),
                self.version,
                "no_endpoint",
            )
        payload = {
            "left": request.left_transcript,
            "right": request.right_transcript,
            "left_language": request.left_language,
            "right_language": request.right_language,
            "context_digest": request.context_digest,
            "model_version": self.model_version or request.model_version,
            "prompt_version": self.prompt_version or request.prompt_version,
            "rule_version": request.rule_version,
        }
        last_error = "external_call_failed"
        attempts = self.max_retries + 1
        for _ in range(attempts):
            try:
                body = self._post(payload)
            except TimeoutError:
                last_error = "timeout"
                self.errors += 1
                continue
            except Exception:
                last_error = "external_call_failed"
                self.errors += 1
                continue
            result = coerce_verifier_verdict(body, version=self.version)
            if result.error == "timeout":
                last_error = "timeout"
                continue
            return enforce_citation_policy(
                result,
                request.left_raw,
                request.right_raw,
                request.left_transcript,
                request.right_transcript,
            )
        return VerifyResult(
            "unknown",
            None,
            (request.left_transcript, request.right_transcript),
            self.version,
            last_error,
        )

    def _post(self, payload: dict[str, Any]) -> Any:
        if self.transport is not None:
            return self.transport(self.endpoint, payload, self.timeout_sec)
        import urllib.request

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as response:
                raw = response.read().decode("utf-8")
        except TimeoutError as exc:
            raise TimeoutError("timeout") from exc
        except Exception as exc:
            if "timed out" in str(exc).lower():
                raise TimeoutError("timeout") from exc
            raise
        return json.loads(raw)


class ChatSemanticVerifier:
    """OpenAI-compatible chat comparison. Text only; never uploads audio.

    ASR transcription endpoints are rejected before any request. Timeout,
    non-JSON, and ungrounded citations stay ``unknown``.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        timeout_sec: float = 5.0,
        max_retries: int = 1,
        model_version: str = "",
        prompt_version: str = "",
        transport: Any | None = None,
    ) -> None:
        self.endpoint = endpoint.strip()
        self.timeout_sec = timeout_sec
        self.max_retries = max(0, int(max_retries))
        self.model_version = model_version or "unspecified"
        self.prompt_version = prompt_version or _CHAT_PROMPT_VERSION
        self.transport = transport
        self.version = f"chat_semantic_verifier_v4:{self.model_version}:{self.prompt_version}"
        self.calls = 0
        self.errors = 0

    def verify(self, request: VerifyRequest) -> VerifyResult:
        self.calls += 1
        if not self.endpoint:
            return VerifyResult(
                "unknown",
                None,
                (request.left_transcript, request.right_transcript),
                self.version,
                "no_endpoint",
            )
        payload = {
            "model": self.model_version,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Compare two ASR transcripts for business action only. "
                        "Reply with JSON keys verdict, affects_business, citations, conflict_type. "
                        "verdict is equivalent, conflict, or unknown. "
                        "citations must be exact substrings of the two transcripts. "
                        "Unknown is not equivalent. Do not invent wording."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "left": request.left_transcript,
                            "right": request.right_transcript,
                            "left_language": request.left_language,
                            "right_language": request.right_language,
                            "context_digest": request.context_digest,
                            "prompt_version": self.prompt_version,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        }
        last_error = "external_call_failed"
        for _ in range(self.max_retries + 1):
            try:
                body = self._post(payload)
            except TimeoutError:
                last_error = "timeout"
                self.errors += 1
                continue
            except Exception:
                last_error = "external_call_failed"
                self.errors += 1
                continue
            parsed = _chat_payload_to_verdict(body)
            result = coerce_verifier_verdict(parsed, version=self.version)
            if result.error == "timeout":
                last_error = "timeout"
                continue
            return enforce_citation_policy(
                result,
                request.left_raw,
                request.right_raw,
                request.left_transcript,
                request.right_transcript,
            )
        return VerifyResult(
            "unknown",
            None,
            (request.left_transcript, request.right_transcript),
            self.version,
            last_error,
        )

    def _post(self, payload: dict[str, Any]) -> Any:
        if self.transport is not None:
            return self.transport(self.endpoint, payload, self.timeout_sec)
        import urllib.request

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as response:
                raw = response.read().decode("utf-8")
        except TimeoutError as exc:
            raise TimeoutError("timeout") from exc
        except Exception as exc:
            if "timed out" in str(exc).lower():
                raise TimeoutError("timeout") from exc
            raise
        return json.loads(raw)


def _chat_payload_to_verdict(body: Any) -> Any:
    if isinstance(body, dict) and body.get("verdict") in {"equivalent", "conflict", "unknown"}:
        return body
    content = ""
    if isinstance(body, dict):
        choices = body.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message") if isinstance(choices[0], dict) else {}
            if isinstance(message, dict):
                content = str(message.get("content") or "")
        elif isinstance(body.get("content"), str):
            content = body["content"]
    elif isinstance(body, str):
        content = body
    if not content.strip():
        return {"verdict": "unknown", "error": "invalid_structure", "citations": []}
    fenced = _JSON_FENCE.search(content)
    blob = fenced.group(1) if fenced else content
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError:
        start = blob.find("{")
        end = blob.rfind("}")
        if start < 0 or end <= start:
            return {"verdict": "unknown", "error": "invalid_structure", "citations": []}
        try:
            parsed = json.loads(blob[start : end + 1])
        except json.JSONDecodeError:
            return {"verdict": "unknown", "error": "invalid_structure", "citations": []}
    return parsed


def resolve_verifier_endpoint(configured: str = "") -> str:
    return (configured or "").strip() or os.environ.get(VERIFIER_ENV_ENDPOINT, "").strip()


def looks_like_asr_endpoint(endpoint: str) -> bool:
    low = endpoint.lower()
    return any(marker in low for marker in _ASR_ENDPOINT_MARKERS)


def infer_verifier_protocol(endpoint: str, protocol: str = "auto") -> str:
    proto = (protocol or "auto").strip().lower()
    if looks_like_asr_endpoint(endpoint):
        return "rejected"
    if proto in {"chat", "compare", "http", "json"}:
        return "chat" if proto == "chat" else "compare"
    if any(marker in endpoint.lower() for marker in _CHAT_ENDPOINT_MARKERS):
        return "chat"
    if "/compare" in endpoint.lower() or endpoint.lower().rstrip("/").endswith("/verify"):
        return "compare"
    return "rejected"


class CompositeSemanticVerifier:
    """Local business rules first. A configured model only sees residuals.

    The same result object is what family, clique, and dissent checks consume.
    """

    def __init__(self, local: SemanticVerifier, remote: SemanticVerifier | None = None) -> None:
        self.local = local
        self.remote = remote
        self.version = getattr(local, "version", "composite")
        self.calls = 0
        self.remote_calls = 0

    def verify(self, request: VerifyRequest) -> VerifyResult:
        local = self.local.verify(request)
        if local.verdict in {"equivalent", "conflict"}:
            return enforce_citation_policy(
                local,
                request.left_raw,
                request.right_raw,
                request.left_transcript,
                request.right_transcript,
            )
        if self.remote is None:
            return local
        self.remote_calls += 1
        remote = self.remote.verify(request)
        self.calls += 1
        if remote.verdict == "unknown":
            return remote
        return enforce_citation_policy(
            remote,
            request.left_raw,
            request.right_raw,
            request.left_transcript,
            request.right_transcript,
        )


def build_verifier(mode: str, *, endpoint: str = "") -> SemanticVerifier:
    mode = (mode or "local").strip().lower()
    if mode in {"none", "off", "disabled"}:
        return NullSemanticVerifier()
    if mode in {"http", "remote", "qwen"}:
        if endpoint.strip():
            return HttpSemanticVerifier(endpoint=endpoint)
        return UnavailableSemanticVerifier(endpoint=endpoint)
    if mode in {"business", "business_local", "v4"}:
        return BusinessLocalVerifier()
    return LocalSemanticVerifier()


def build_callable_verifier(
    mode: str,
    *,
    endpoint: str = "",
    timeout_sec: float = 5.0,
    max_retries: int = 1,
    model_version: str = "",
    prompt_version: str = "",
    protocol: str = "auto",
    transport: Any | None = None,
) -> SemanticVerifier:
    """Local business rules always run. A text model runs only on residuals.

    An endpoint is taken from config or AUDIO_ENGINE_SEMANTIC_VERIFIER_ENDPOINT.
    ASR transcription URLs are rejected and never called. Missing, rejected,
    or unreachable endpoints stay unknown.
    """
    local = BusinessLocalVerifier()
    mode = (mode or "business_local").strip().lower()
    if mode in {"none", "off", "disabled"}:
        return CompositeSemanticVerifier(local, None)
    resolved = resolve_verifier_endpoint(endpoint)
    if not resolved:
        return CompositeSemanticVerifier(local, None)
    kind = infer_verifier_protocol(resolved, protocol)
    if kind == "rejected":
        return CompositeSemanticVerifier(
            local,
            UnavailableSemanticVerifier(endpoint=resolved),
        )
    remote_cls = ChatSemanticVerifier if kind == "chat" else HttpSemanticVerifier
    remote = remote_cls(
        endpoint=resolved,
        timeout_sec=timeout_sec,
        max_retries=max_retries,
        model_version=model_version,
        prompt_version=prompt_version,
        transport=transport,
    )
    return CompositeSemanticVerifier(local, remote)
