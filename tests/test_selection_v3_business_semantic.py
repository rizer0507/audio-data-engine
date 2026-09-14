"""024 business-semantic rule. Opt-in; does not change 022 shadow behavior."""

from __future__ import annotations

import json
from pathlib import Path

from audio_engine.core.annotation_v3 import AnnotationConfig, build_export_rows
from audio_engine.core.annotation_v3.queue import requires_dual_review
from audio_engine.core.dataset_v3.sampling import blocked_speech_train_target
from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3 import SelectionV3Config, classify_sample
from audio_engine.core.selection_v3.business_semantic import (
    classify_business_semantic,
    governance_release_status,
    legacy_float_ratio_accepts,
    partition_coverage,
    support_ratio_met,
    _LIBRARY_CACHE as _V4_LIB_CACHE,
)
from audio_engine.core.selection_v3.semantic_verify import (
    HttpSemanticVerifier,
    VerifyRequest,
    VerifyResult,
)
from audio_engine.core.selection_v3.spot_audit import apply_spot_audit_flags
from audio_engine.core.selection_v3.text_tolerance import content_language, has_lexical_content
from audio_engine.core.selection_v3.types import is_business_semantic_rule

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "selection" / "zh_asr_v3_business_semantic_v4.yaml"
SIX = ["glm_1", "glm_2", "sensevoice_1", "sensevoice_2", "qwen_1", "qwen_2"]


def _cfg() -> SelectionV3Config:
    return SelectionV3Config.from_yaml(CFG)


def _sample(texts: dict[str, str], **labels) -> Sample:
    transcripts = {}
    for key in SIX:
        if key in texts and texts[key] is None:
            transcripts[key] = {"text": "", "status": "failed", "failed": True}
            continue
        text = texts.get(key, "")
        transcripts[key] = {"text": text, "extra": {"raw_text": text}, "status": "success"}
    quality = labels.pop("quality", {"dnsmos_status": "not_required", "noise_band": "unknown", "calibrated": False})
    return Sample(
        id=labels.pop("sid", "utt-024"),
        source_path="dummy.wav",
        duration=labels.pop("duration", 3.0),
        sha256="abc",
        transcripts=transcripts,
        quality=quality,
        labels={"original_audio_sha256": "abc", **labels},
    )


def _by_family(glm: str, sense: str, qwen: str) -> dict[str, str]:
    return {
        "glm_1": glm,
        "glm_2": glm,
        "sensevoice_1": sense,
        "sensevoice_2": sense,
        "qwen_1": qwen,
        "qwen_2": qwen,
    }


class _Stub:
    def __init__(self, verdict: str, error: str | None = None, citations: tuple[str, ...] | None = None) -> None:
        self.version = "stub_v4"
        self.calls = 0
        self.verdict = verdict
        self.error = error
        self.citations = citations

    def verify(self, request: VerifyRequest) -> VerifyResult:
        self.calls += 1
        cites = self.citations if self.citations is not None else (request.left_transcript, request.right_transcript)
        return VerifyResult(self.verdict, None, cites, self.version, self.error, affects_business=True)


def test_punctuation_control_tags_are_empty_digits_and_foreign_are_kept():
    assert content_language(".") == "empty"
    assert content_language("<|zh|><|NEUTRAL|>.") == "empty"
    assert has_lexical_content("10086")
    assert content_language("10086") == "unknown"
    assert content_language("hello") == "en"

    empty = classify_sample(_sample(_by_family(".", "<|zh|>.", "。")), _cfg())
    assert empty.category is None
    assert empty.category != "non_speech"
    assert empty.status == "hold"
    assert empty.coverage_bucket == "U"
    assert empty.label_grade == "none"
    assert empty.label_grade != "no_transcript"
    assert "language_unverified" not in empty.reason_codes
    assert "presence_unconfirmed" in empty.reason_codes
    assert empty.candidate_text in {None, ""}
    assert empty.is_human_verified is False

    digits = classify_sample(_sample(_by_family("10086", "10086", "10086")), _cfg())
    assert digits.language_by_run["glm_1"] == "unknown"
    assert digits.category != "non_speech"
    assert digits.status != "auto_classified"
    assert "10086" in (digits.abstain_reasons or {}) or digits.coverage_bucket == "U"


def test_continue_listening_is_equivalent_and_conflicts_are_not_distance_exempt():
    continued = classify_sample(_sample(_by_family("你说", "嗯，你说", "你说吧")), _cfg())
    assert continued.category == "business_consistent"
    assert continued.status == "auto_classified"
    assert continued.coverage_bucket == "A"
    assert continued.type == "business_consistent"
    assert continued.type != "gold"
    assert continued.decision != "accepted"
    assert continued.label_tier == "semantic_only"
    assert continued.commitment == "undetermined"
    assert "not_authorization" in continued.usage_blocks
    assert continued.is_human_verified is False
    assert continued.candidate_text in {"你说", "嗯，你说", "你说吧"}

    flipped = classify_sample(_sample(_by_family("需要", "不需要", "需要")), _cfg())
    assert flipped.category == "semantic_risk"
    assert flipped.status == "manual_review"
    assert flipped.coverage_bucket == "H"
    assert flipped.decision == "manual_review"
    assert flipped.decision != "accepted"

    subjects = classify_sample(_sample(_by_family("他歇了", "不需要了", "不去了")), _cfg())
    assert subjects.category == "semantic_risk"
    assert subjects.coverage_bucket == "H"


def test_integer_two_thirds_passes_but_unresolved_objection_does_not_auto_clear():
    assert support_ratio_met(2, 3)
    assert legacy_float_ratio_accepts(2, 3, configured=0.6666667) is False

    held = classify_sample(_sample(_by_family("你现在方便吗", "你现在方便吗", "他歇了")), _cfg())
    assert held.coverage_bucket == "U"
    assert held.category is None
    assert "support_ratio_not_sufficient_alone" in held.reason_codes
    assert held.status != "auto_classified"


def test_dual_run_is_one_vote_and_minority_conflict_stays():
    risk = classify_sample(_sample(_by_family("需要", "需要", "不需要")), _cfg())
    assert risk.category == "semantic_risk"
    assert risk.chinese_available_family_count == 3
    assert risk.chinese_available_family_count != 6
    assert risk.coverage_bucket == "H"


def test_single_family_gap_does_not_block_when_other_evidence_agrees():
    failed = _sample(_by_family(None, "你好", "你好"))
    agreed = classify_sample(failed, _cfg())
    assert agreed.category == "business_consistent"
    assert agreed.status == "auto_classified"
    assert agreed.coverage_bucket == "A"
    assert agreed.status != "retry"
    assert "glm" in agreed.abstain_reasons

    long_glm = "你" * 90
    rated = classify_sample(_sample(_by_family(long_glm, "你好", "你好"), duration=1.0), _cfg())
    assert rated.category == "business_consistent"
    assert "glm_1" in rated.implausible_routes
    assert rated.status != "retry"

    english = "hello there " * 20
    mixed = classify_sample(_sample(_by_family("你好", "你好", english), duration=1.0), _cfg())
    assert mixed.category == "business_consistent"
    assert "qwen_1" not in mixed.implausible_routes
    assert "non_chinese_rate_not_applied" in mixed.auxiliary_tags


def test_verifier_changes_family_consensus_and_dissent_timeout_does_not_pass():
    left = "客户说明天再联系"
    right = "客户说明日回电"
    same = _by_family(left, right, left)
    passed = classify_business_semantic(_sample(same), _cfg(), verifier=_Stub("equivalent"))
    assert passed.category == "business_consistent"
    denied = classify_business_semantic(_sample(same), _cfg(), verifier=_Stub("conflict"))
    assert denied.category == "semantic_risk"
    assert denied.coverage_bucket == "H"
    timed = classify_business_semantic(
        _sample(same),
        _cfg(),
        verifier=_Stub("unknown", error="timeout"),
    )
    assert timed.category is None
    assert timed.coverage_bucket == "U"
    assert timed.status != "auto_classified"


def test_invalid_citation_does_not_auto_pass():
    def transport(_url, _payload, _timeout):
        return {"verdict": "equivalent", "citations": ["这句话没有出现在输入里"], "version": "remote"}

    remote = HttpSemanticVerifier(endpoint="http://verifier.invalid/compare", transport=transport)
    request = VerifyRequest("客户说明天再联系", "客户说明日回电", "客户说明天再联系", "客户说明日回电", "zh", "zh")
    result = remote.verify(request)
    assert result.verdict == "unknown"
    assert result.error == "invalid_citation"

    def boom(_url, _payload, _timeout):
        raise TimeoutError("timeout")

    timed = HttpSemanticVerifier(endpoint="http://verifier.invalid/compare", transport=boom, max_retries=0)
    timed_result = timed.verify(request)
    assert timed_result.verdict == "unknown"
    assert timed_result.error == "timeout"


def test_presence_voicemail_and_listening_boundaries():
    empty = classify_sample(_sample(_by_family(".", ".", ".")), _cfg())
    assert empty.category != "non_speech"
    assert empty.coverage_bucket == "U"

    confirmed = classify_sample(
        _sample(
            _by_family(".", ".", "."),
            speech_presence={
                "deployed": True,
                "calibrated": True,
                "model_version": "event-test-1",
                "speech_present": False,
                "event": "silence",
                "sources": ["event", "cross_family_empty"],
            },
        ),
        _cfg(),
    )
    assert confirmed.category == "non_speech"
    assert confirmed.subtype == "silence"
    assert confirmed.label_tier == "no_transcript"
    assert confirmed.coverage_bucket == "A"
    assert confirmed.candidate_text in {None, ""}

    placeholder = classify_sample(
        _sample(
            _by_family(".", ".", "."),
            speech_presence={"deployed": False, "placeholder": True, "speech_present": False, "event": "silence"},
        ),
        _cfg(),
    )
    assert placeholder.category != "non_speech"
    assert placeholder.coverage_bucket == "U"

    vad_only = classify_sample(
        _sample(
            _by_family(".", ".", "."),
            speech_presence={
                "deployed": True,
                "calibrated": True,
                "model_version": "vad-only",
                "speech_present": False,
                "event": "silence",
                "sources": ["vad"],
            },
        ),
        _cfg(),
    )
    assert vad_only.category != "non_speech"

    heard = classify_sample(
        _sample(
            _by_family(".", ".", "."),
            speech_presence={
                "deployed": True,
                "calibrated": True,
                "model_version": "event-test-1",
                "speech_present": True,
                "event": "speech",
                "sources": ["event"],
            },
        ),
        _cfg(),
    )
    assert heard.category == "hardcase"
    assert heard.coverage_bucket == "H"

    refusal = classify_sample(_sample(_by_family("不需要", "不需要了", "不用了")), _cfg())
    assert refusal.category == "business_consistent"
    assert refusal.category != "non_speech"
    assert refusal.subtype == "refusal"

    mailbox = classify_sample(
        _sample(_by_family("请在滴声后留言", "请在嘀一声后留言", "请在滴声后留言")),
        _cfg(),
    )
    assert mailbox.category == "voicemail"
    assert mailbox.status == "auto_classified"
    assert mailbox.type != "gold"
    assert mailbox.label_tier == "semantic_only"

    quoted = classify_sample(_sample(_by_family("请在滴声后留言", "请在滴声后留言", "不需要")), _cfg())
    assert quoted.category == "semantic_risk"
    assert quoted.category != "voicemail"


def test_auto_labels_are_consumed_but_not_accepted_or_verbatim_train():
    result = classify_sample(_sample(_by_family("你说", "嗯，你说", "你说")), _cfg())
    labels = result.to_labels("p")
    assert "gold_text" not in labels
    assert labels["is_human_verified"] is False
    assert labels["status"] == "auto_classified"
    assert labels["category"] == "business_consistent"
    assert labels["coverage_bucket"] == "A"
    assert blocked_speech_train_target(_sample(_by_family("你说", "你说", "你说"), **labels)) == "semantic_layer_not_verbatim"

    risk = classify_sample(_sample(_by_family("需要", "不需要", "需要"), sid="s-risk"), _cfg())
    risk_labels = risk.to_labels("p")
    sample = _sample(_by_family("需要", "不需要", "需要"), sid="s-risk")
    sample.labels.update(risk_labels)
    assert requires_dual_review(sample, AnnotationConfig.from_params({}))
    _qid, rows, _meta = build_export_rows(
        [sample],
        config=AnnotationConfig.from_params({}),
        dataset_path="input",
        revision="v4",
        view="blind",
        priorities=["P0"],
        queues=["manual_review"],
    )
    assert rows[0]["category"] == "semantic_risk"
    assert rows[0]["status"] == "manual_review"
    assert rows[0]["coverage_bucket"] == "H"
    assert rows[0]["label_tier"] == "none"
    assert rows[0]["label_grade"] == "none"

    report = partition_coverage(
        [
            {"id": "a", "coverage_bucket": "A"},
            {"id": "h", "coverage_bucket": "H"},
            {"id": "u", "coverage_bucket": "U"},
        ]
    )
    assert report["conserved"] is True
    assert report["A"] + report["H"] + report["U"] == 3
    assert report["arithmetic_target_for_30000"]["met"] is False
    assert report["acceptance_claimed"] is False

    governance = {"missing_group_metadata": True, "dataset_role": "governance_hold"}
    gate = governance_release_status(governance)
    assert gate["classification_allowed"] is True
    assert gate["train_eval_release"] == "blocked"
    assert governance["missing_group_metadata"] is True
    assert is_business_semantic_rule("selection_v3.0") is False
    assert is_business_semantic_rule("selection_business_semantic_v4") is True


def test_punctuation_and_particle_wording_is_business_equivalent():
    where = classify_sample(_sample(_by_family("哪里", "哪里？", "哪里")), _cfg())
    assert where.category == "business_consistent"
    assert where.status == "auto_classified"
    assert where.subtype == "identity_inquiry"
    assert where.coverage_bucket == "A"

    refused = classify_sample(_sample(_by_family("没有了谢谢", "没有了谢谢啊", "没有了谢谢")), _cfg())
    assert refused.category == "business_consistent"
    assert refused.subtype == "refusal"
    assert refused.coverage_bucket == "A"

    who = classify_sample(_sample(_by_family("喂啊你好谁啊", "喂啊你好谁啊", "喂啊你好谁啊")), _cfg())
    assert who.category == "business_consistent"
    assert who.subtype == "identity_inquiry"


def test_empty_versus_text_stays_unconfirmed_without_audio_evidence():
    mixed = classify_sample(_sample(_by_family("你好", "你好", ".")), _cfg())
    assert mixed.coverage_bucket == "U"
    assert mixed.category is None
    assert "presence_unconfirmed" in mixed.reason_codes
    assert mixed.status != "auto_classified"

    heard = classify_sample(
        _sample(
            _by_family("你好", "你好", "."),
            speech_presence={
                "deployed": True,
                "calibrated": True,
                "model_version": "event-test-1",
                "speech_present": True,
                "event": "speech",
                "sources": ["event"],
            },
        ),
        _cfg(),
    )
    assert heard.category == "business_consistent"
    assert heard.coverage_bucket == "A"


def test_protocol_error_route_abstains_and_voicemail_templates_expand():
    _V4_LIB_CACHE.clear()
    protocol = classify_sample(
        _sample(_by_family("I'm sorry I cannot transcribe this audio.", "你好", "你好")),
        _cfg(),
    )
    assert protocol.category == "business_consistent"
    assert protocol.abstain_reasons.get("glm") == "protocol_error"
    assert protocol.coverage_bucket == "A"

    mailbox = classify_sample(
        _sample(
            _by_family(
                "您拨打的电话暂时不方便接听请您留言我将为您转达",
                "您拨打的电话暂时不方便接听请您留言我将为您转达",
                "您拨打的电话暂时不方便接听请您留言我将为您转答",
            )
        ),
        _cfg(),
    )
    assert mailbox.category == "voicemail"
    assert mailbox.status == "auto_classified"


def test_v4_auto_classes_are_spot_audit_eligible():
    _V4_LIB_CACHE.clear()
    result = classify_sample(_sample(_by_family("你说", "嗯，你说", "你说")), _cfg())
    sample = _sample(_by_family("你说", "你说", "你说"), sid="s-auto")
    sample.labels.update(result.to_labels("p"))
    apply_spot_audit_flags([sample], floor_n=1, rate=1.0)
    assert sample.labels["spot_audit_eligible"] is True
    assert sample.labels["spot_audit_selected"] is True
    assert sample.labels["review_priority"] == "P2"


def test_high_frequency_local_fields_and_single_vote_is_not_leftover_objection():
    who = classify_sample(_sample(_by_family("喂啊你好谁啊", "喂啊你好谁啊", "喂啊你好谁啊")), _cfg())
    assert who.category == "business_consistent"
    assert who.coverage_bucket == "A"
    assert who.subtype == "identity_inquiry"

    refused = classify_sample(_sample(_by_family("没有了谢谢", "没有了谢谢啊", "没有了谢谢")), _cfg())
    assert refused.category == "business_consistent"
    assert refused.subtype == "refusal"
    assert refused.commitment == "stated_refusal"

    brief = classify_sample(_sample(_by_family("好啊", "好啊", "好啊")), _cfg())
    assert brief.category == "business_consistent"
    assert brief.subtype == "brief_response"
    assert "not_authorization" in brief.usage_blocks

    flipped = classify_sample(_sample(_by_family("需要", "不需要", "需要")), _cfg())
    assert flipped.category == "semantic_risk"

    one = classify_sample(_sample(_by_family(".", ".", "哦")), _cfg())
    assert one.coverage_bucket == "U"
    assert one.reason != "residual_business_objection"


def test_chat_verifier_runs_on_residual_and_asr_endpoint_is_rejected(monkeypatch):
    from audio_engine.core.selection_v3.semantic_verify import (
        ChatSemanticVerifier,
        UnavailableSemanticVerifier,
        VERIFIER_ENV_ENDPOINT,
        build_callable_verifier,
        infer_verifier_protocol,
    )

    left = "客户说明天再联系"
    right = "客户说明日回电"
    request = VerifyRequest(left, right, left, right, "zh", "zh")

    def transport(_url, payload, _timeout):
        assert "messages" in payload
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "verdict": "equivalent",
                                "affects_business": False,
                                "citations": [left, right],
                                "conflict_type": None,
                            },
                            ensure_ascii=False,
                        )
                    }
                }
            ]
        }

    chat = ChatSemanticVerifier(
        endpoint="http://127.0.0.1:8080/v1/chat/completions",
        transport=transport,
        model_version="local-test",
    )
    ok = chat.verify(request)
    assert ok.verdict == "equivalent"
    assert ok.error is None

    def bad_cite(_url, _payload, _timeout):
        return {"verdict": "equivalent", "citations": ["这句话没有出现在输入里"]}

    bad = ChatSemanticVerifier(endpoint="http://127.0.0.1:8080/v1/chat/completions", transport=bad_cite)
    assert bad.verify(request).verdict == "unknown"
    assert bad.verify(request).error == "invalid_citation"

    assert infer_verifier_protocol("http://127.0.0.1:5555/v1/audio/transcriptions", "chat") == "rejected"
    calls = {"n": 0}

    def boom(_url, _payload, _timeout):
        calls["n"] += 1
        raise AssertionError("ASR endpoint must not be called")

    rejected = build_callable_verifier(
        "business_local",
        endpoint="http://127.0.0.1:5555/v1/audio/transcriptions",
        transport=boom,
    )
    assert isinstance(rejected.remote, UnavailableSemanticVerifier)
    out = rejected.verify(request)
    assert out.verdict == "unknown"
    assert calls["n"] == 0

    monkeypatch.setenv(VERIFIER_ENV_ENDPOINT, "http://127.0.0.1:8080/v1/chat/completions")
    wired = build_callable_verifier("business_local", endpoint="", transport=transport)
    residual = wired.verify(request)
    assert residual.verdict == "equivalent"
    assert wired.remote_calls >= 1
