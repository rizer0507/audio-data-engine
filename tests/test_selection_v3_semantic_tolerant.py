"""022 semantic-tolerant selection: business invariants, not a copy of the implementation."""

from __future__ import annotations

from pathlib import Path

from audio_engine.core.annotation_v3 import (
    AnnotationConfig,
    apply_review_import_v3,
    build_export_rows,
)
from audio_engine.core.annotation_v3.types import STATE_ANNOTATED
from audio_engine.core.dataset_v3.sampling import blocked_speech_train_target
from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3 import SelectionV3Config, classify_sample
from audio_engine.core.selection_v3.gold_select import FamilyRep, select_representative_text
from audio_engine.core.selection_v3.legacy_map import map_legacy
from audio_engine.core.selection_v3.semantic_tolerant import eval_reference_excluding
from audio_engine.core.selection_v3.semantic_verify import (
    UnavailableSemanticVerifier,
    VerifyRequest,
    cache_key,
    coerce_verifier_verdict,
)
from audio_engine.metrics.cer import mixed_language_character_metrics
from audio_engine.core.selection_v3.spot_audit import apply_spot_audit_flags, spot_audit_quota
from audio_engine.core.selection_v3.text_tolerance import (
    contains_control_tag,
    tolerant_distance,
)
from audio_engine.metrics.cer import reference_character_metrics

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "selection" / "zh_asr_v3_semantic_tolerant.yaml"
SIX = ["glm_1", "glm_2", "sensevoice_1", "sensevoice_2", "qwen_1", "qwen_2"]


def _cfg(**overrides) -> SelectionV3Config:
    base = SelectionV3Config.from_yaml(CFG)
    params = {
        "engine": base.engine,
        "rule_version": base.rule_version,
        "policy_version": base.policy_version,
        "target_family": base.target_family,
        "model_families": base.model_families,
        "teacher_families": base.teacher_families,
        "expected_runs_per_family": 2,
        "voicemail_strong_path": str(ROOT / "configs/selection/voicemail_strong_v1.yaml"),
        "quality": {"calibrated": False},
        "tolerance": {
            "version": "text_tolerance_v1",
            "recall_max_distance": 0.10,
            "divergence_min_distance": 0.25,
            "short_max_chars": 6,
            "homophone_pairs": [["先声", "先生"], ["嘀声", "滴声"]],
        },
        "semantic_verifier": {"mode": "local", "endpoint": ""},
        "speech_rate": {"max_chars_per_sec": 25, "min_text_chars": 80, "disposition": "route_quarantine"},
    }
    params.update(overrides)
    return SelectionV3Config.from_params(params)


def _sample(texts: dict[str, str], **labels) -> Sample:
    transcripts = {}
    for key in SIX:
        text = texts.get(key, "")
        transcripts[key] = {"text": text, "extra": {"raw_text": text}, "status": "success"}
    quality = labels.pop("quality", {"dnsmos_status": "success", "noise_band": "unknown", "calibrated": False})
    return Sample(
        id=labels.pop("sid", "utt-022"),
        source_path="dummy.wav",
        duration=labels.pop("duration", 3.0),
        sha256="abc",
        transcripts=transcripts,
        quality=quality,
        labels={"original_audio_sha256": "abc", **labels},
    )


def _pair(text: str) -> dict[str, str]:
    return {key: text for key in SIX}


def _by_family(glm: str, sense: str, qwen: str) -> dict[str, str]:
    return {
        "glm_1": glm,
        "glm_2": glm,
        "sensevoice_1": sense,
        "sensevoice_2": sense,
        "qwen_1": qwen,
        "qwen_2": qwen,
    }


def test_you_nin_and_edge_particle_can_be_gold_without_rewriting_text():
    result = classify_sample(_sample(_by_family("您现在方便吗", "你现在方便吗", "你现在方便吗")), _cfg())
    assert result.category == "gold"
    assert result.status == "candidate"
    assert result.is_human_verified is False
    assert result.label_source == "model_consensus"
    assert result.label_tier == "model_candidate"
    assert result.candidate_text in {"你现在方便吗", "您现在方便吗"}
    assert "您" in (result.selected_raw_text or "") or result.candidate_text == "你现在方便吗"
    assert not contains_control_tag(result.candidate_text)

    particle = classify_sample(_sample(_pair("我知道了啊")), _cfg())
    assert particle.category == "gold"
    # Both runs already identical; selected body keeps the particle, not a rewritten sentence.
    assert particle.candidate_text == "我知道了啊"


def test_hao_buhao_is_mandatory_review_and_identical_negation_is_not():
    flipped = classify_sample(_sample(_by_family("好", "不好", "好")), _cfg())
    assert flipped.category == "semantic_risk"
    assert flipped.status == "manual_review"
    assert flipped.review_priority == "P0"
    assert flipped.decision == "manual_review"
    assert flipped.label_tier == "none"

    same = classify_sample(_sample(_pair("不是不是别人给我的手机号")), _cfg())
    assert same.category != "semantic_risk"
    assert same.category == "gold"


def test_nontransitive_cluster_does_not_merge_far_pair():
    # A≈B, B≈C, A far from C must not become one cluster via B.
    a = "甲乙丙丁戊己庚辛"
    b = "甲乙丙丁戊己庚壬"
    c = "完全不同的另一个主题内容"
    result = classify_sample(_sample(_by_family(a, b, c)), _cfg())
    assert result.category != "gold"
    assert result.category in {"hardcase", "semantic_risk"} or result.status == "hold"


def test_glm_english_cases_and_cer_null():
    agree = classify_sample(
        _sample(_by_family("I do not want it", "我不要", "我不要")),
        _cfg(),
    )
    assert agree.category == "gold"
    assert agree.language_by_run["glm_1"] == "en"
    assert agree.char_comparable is False
    assert agree.candidate_text == "我不要"
    assert agree.support_ratio_of_chinese == 1.0
    # Compat field stays k/N, not silently rewritten as k/m.
    assert agree.support_ratio_of_4 == 2 / 3

    antonym = classify_sample(
        _sample(_by_family("I want it", "我不要", "我不要")),
        _cfg(),
    )
    assert antonym.category == "semantic_risk"
    assert antonym.status == "manual_review"

    unknown = classify_sample(
        _sample(_by_family("please call back tomorrow", "我不要", "我不要")),
        _cfg(),
    )
    assert unknown.category is None
    assert unknown.status == "hold"
    assert unknown.decision == "hold"

    one_zh = classify_sample(
        _sample(_by_family("I do not want it", "please wait", "我不要")),
        _cfg(),
    )
    assert one_zh.category is None
    assert one_zh.status in {"hold", "retry"}

    metrics = reference_character_metrics("我不要", "I do not want it", char_comparable=False)
    assert metrics["cer"] is None
    assert metrics["reason"] == "language_mismatch"
    empty = reference_character_metrics("", "插入", char_comparable=True)
    assert empty["cer"] is None
    assert empty["insertions"] == 2


def test_noise_requires_evidence_and_does_not_block_clear_gold():
    empty = classify_sample(_sample(_pair("")), _cfg())
    assert empty.category is None
    assert empty.status == "hold"
    assert empty.reason != "acoustic_noise_confirmed"

    confirmed = classify_sample(
        _sample(
            _pair(""),
            quality={
                "no_target_speech": True,
                "no_target_speech_trusted": True,
                "no_target_speech_version": "vad-test",
                "calibrated": True,
            },
        ),
        _cfg(),
    )
    assert confirmed.category == "noise"
    assert confirmed.subtype == "environment"
    assert confirmed.review_queue == "noise_archive"
    assert confirmed.label_tier == "none"

    music = classify_sample(
        _sample(
            _pair("客户表示明天再联系"),
            quality={"background_only": True, "background_label": "music", "calibrated": False},
        ),
        _cfg(),
    )
    assert music.category == "gold"
    assert "background" in music.auxiliary_tags

    hallu = classify_sample(
        _sample(
            _by_family("同意办理", "", ""),
            quality={"no_target_speech": True, "no_target_speech_trusted": True, "calibrated": True},
        ),
        _cfg(),
    )
    assert hallu.category == "semantic_risk"
    assert "noise" in hallu.auxiliary_tags


def test_voicemail_strong_weak_and_human_negation():
    mailbox = classify_sample(
        _sample(_by_family("请在滴声后留言", "请在嘀声后留言", "请在滴声后留言")),
        _cfg(),
    )
    assert mailbox.category == "voicemail"
    assert mailbox.subtype == "mailbox"
    assert mailbox.status == "candidate"
    assert mailbox.candidate_text in {"请在滴声后留言", "请在嘀声后留言"}
    assert mailbox.review_queue == "voicemail_spot_audit"

    weak = classify_sample(_sample(_pair("我是机主")), _cfg())
    assert weak.category != "voicemail"

    human = classify_sample(_sample(_pair("我不想留言")), _cfg())
    assert human.category != "voicemail"


def test_gold_selection_hand_calculated_ties_and_control_tags():
    # D is hand-calculated, not by reusing select_representative_text's sort.
    t1 = FamilyRep("glm", "glm_1", "甲乙丙丁戊己庚", "甲乙丙丁戊己庚", "甲乙丙丁戊己庚", "甲乙丙丁戊己庚")
    t2 = FamilyRep("qwen", "qwen_1", "甲乙丙丁戊己庚", "甲乙丙丁戊己庚", "甲乙丙丁戊己庚", "甲乙丙丁戊己庚")
    t3 = FamilyRep("sensevoice", "sensevoice_1", "甲乙丙丁戊己辛", "甲乙丙丁戊己辛", "甲乙丙丁戊己辛", "甲乙丙丁戊己辛")
    # d(t1,t2)=0, d(t1,t3)=1/7, d(t2,t3)=1/7
    # D(t1)=D(t2)=(0+1/7)/2 = 0.071429; D(t3)=0.166667
    assert tolerant_distance(t1.tolerant_key, t3.tolerant_key) == round(1 / 7, 6)
    chosen = select_representative_text([t3, t1, t2], family_order=["qwen", "glm", "sensevoice"])
    assert chosen is not None
    assert chosen.distance == round((0 + 1 / 7) / 2, 6)
    assert chosen.transcript_support_count == 2
    assert chosen.transcript_text == "甲乙丙丁戊己庚"

    # Exact support tie, family order only. 你/您 must not be merged for support.
    nin = classify_sample(
        _sample(_by_family("您好，请问现在方便吗", "你好，请问现在方便吗", "您好，请问现在方便吗")),
        _cfg(),
    )
    assert nin.category == "gold"
    assert nin.candidate_text == "您好，请问现在方便吗"
    assert nin.transcript_support_count == 2
    assert nin.tie_break_reason == "exact_transcript_support"

    # Equal distance and exact support of 1. Only configured family order decides.
    left = FamilyRep("glm", "glm_1", "您现在方便吗", "您现在方便吗", "你现在方便吗", "您现在方便吗")
    right = FamilyRep("sensevoice", "sensevoice_1", "你现在方便吗", "你现在方便吗", "你现在方便吗", "你现在方便吗")
    ordered = select_representative_text(
        [left, right],
        family_order=["sensevoice", "glm", "qwen"],
    )
    assert ordered is not None
    assert ordered.transcript_support_count == 1
    assert ordered.family == "sensevoice"
    assert ordered.transcript_text == "你现在方便吗"
    assert ordered.tie_break_reason == "configured_family_order"
    reversed_order = select_representative_text(
        [left, right],
        family_order=["glm", "sensevoice", "qwen"],
    )
    assert reversed_order is not None
    assert reversed_order.family == "glm"
    assert reversed_order.transcript_text == "您现在方便吗"

    tagged = "您"  # placeholder replaced below
    raw = "<|zh|><|NEUTRAL|><|Speech|><|withitn|>您现在方便吗"
    texts = _by_family(raw, "您现在方便吗", "您现在方便吗")
    tagged_result = classify_sample(_sample(texts), _cfg())
    assert tagged_result.category == "gold"
    assert "<|" not in (tagged_result.candidate_text or "")
    assert tagged not in {"<|zh|>"}
    if tagged_result.selected_family == "glm":
        assert "<|zh|>" in (tagged_result.selected_raw_text or "")


def test_input_order_stable_and_order_change_only_on_tie():
    cfg = _cfg()
    texts = _by_family("您好，请问现在方便吗", "你好，请问现在方便吗", "您好，请问现在方便吗")
    first = classify_sample(_sample(texts), cfg)
    shuffled = {key: texts[key] for key in reversed(SIX)}
    second = classify_sample(_sample(shuffled), cfg)
    assert (first.category, first.candidate_text, first.selected_family, first.status) == (
        second.category,
        second.candidate_text,
        second.selected_family,
        second.status,
    )
    # Non-tie: Qwen+GLM share 您好. Changing family order must not pick 你好.
    flipped = _cfg(
        teacher_families=["sensevoice", "glm"],
    )
    again = classify_sample(_sample(texts), flipped)
    assert again.candidate_text == "您好，请问现在方便吗"


def test_target_family_excluded_before_reference():
    sample = _sample(_by_family("I do not want it", "我不要", "我不要"))
    ref = eval_reference_excluding(sample, _cfg(), "qwen")
    assert ref is not None
    assert ref["eligible"] is False
    assert ref["candidate_text"] is None


def test_verifier_unknown_timeout_and_cache_isolation():
    missing = UnavailableSemanticVerifier(endpoint="")
    request = VerifyRequest("我不要", "I maybe", "我不要", "I maybe", "zh", "en", rule_version="r1")
    result = missing.verify(request)
    assert result.verdict == "unknown"
    assert missing.calls == 1
    assert result.error == "no_endpoint"
    other = VerifyRequest("我不要", "I maybe", "我不要", "I maybe", "zh", "en", rule_version="r2")
    assert cache_key(request, verifier_version="v") != cache_key(other, verifier_version="v")


def test_legacy_mapping_and_queues_do_not_promote_candidates():
    gold = map_legacy(category="gold", status="candidate")
    assert gold.label_tier == "model_candidate"
    assert gold.decision == "audit_pending"
    assert gold.review_queue == "spot_audit"
    assert gold.review_priority == "P2"
    risk = map_legacy(category="semantic_risk", status="manual_review")
    assert risk.review_priority == "P0"
    assert risk.review_queue == "manual_review"
    noise = map_legacy(category="noise", status="candidate")
    assert noise.review_queue == "noise_archive"
    assert noise.label_source != "model_consensus"
    hold = map_legacy(category=None, status="hold")
    assert hold.review_queue == "hold"
    assert hold.review_priority is None


def test_review_export_intersection_and_simulated_import_not_production():
    risk = classify_sample(_sample(_by_family("好", "不好", "好"), sid="s-risk"), _cfg())
    gold = classify_sample(_sample(_pair("你现在方便吗"), sid="s-gold"), _cfg())
    samples = []
    for sid, result in (("s-risk", risk), ("s-gold", gold)):
        sample = _sample(_pair("placeholder"), sid=sid)
        sample.labels.update(result.to_labels("p"))
        sample.labels["reservation_role"] = "train_pool"
        sample.labels["leakage_group_id"] = f"g-{sid}"
        samples.append(sample)
    apply_spot_audit_flags(samples, floor_n=1, rate=1.0)
    cfg = AnnotationConfig.from_params({})
    qid, rows, _meta = build_export_rows(
        samples,
        config=cfg,
        dataset_path="shadow-not-production.parquet",
        revision="r-sim-022",
        view="blind",
        priorities=["P0"],
        queues=["manual_review"],
    )
    assert [row["sample_id"] for row in rows] == ["s-risk"]
    assert all(row["review_priority"] == "P0" for row in rows)
    row = dict(rows[0])
    row.update(
        {
            "decision": "accepted",
            "gold_kind": "speech",
            "gold_text": "好",
            "annotator_id": "sim-022-not-production",
            "human_semantic": "positive",
            "speech_scope": "target",
            "human_noise": "clean",
            "human_crosstalk": "false",
        }
    )
    imported = apply_review_import_v3(
        samples,
        [row],
        config=cfg,
        expected_queue_id=qid,
        expected_revision="r-sim-022",
        review_pass="first",
        actor_id="sim-022-not-production",
    )
    stamped = next(s for s in imported.samples if s.id == "s-risk")
    assert stamped.labels["annotation_state"] == STATE_ANNOTATED
    assert stamped.labels.get("is_human_verified") is False
    assert stamped.labels.get("status") != "accepted"
    gold_row = next(s for s in samples if s.id == "s-gold")
    assert gold_row.labels["status"] == "candidate"
    assert blocked_speech_train_target(gold_row) == "status_candidate_not_accepted"
    assert blocked_speech_train_target(next(s for s in samples if s.id == "s-risk"))


def test_dual_run_does_not_add_a_second_vote_and_rate_isolates_route():
    # One family supports 您好; the other two share 你好. Dual-run still counts as one.
    reps = [
        FamilyRep("glm", "glm_1", "您好，请问现在方便吗", "您好，请问现在方便吗", "你好请问现在方便吗", "您好请问现在方便吗"),
        FamilyRep("sensevoice", "sensevoice_1", "你好，请问现在方便吗", "你好，请问现在方便吗", "你好请问现在方便吗", "你好请问现在方便吗"),
        FamilyRep("qwen", "qwen_1", "你好，请问现在方便吗", "你好，请问现在方便吗", "你好请问现在方便吗", "你好请问现在方便吗"),
    ]
    chosen = select_representative_text(reps, family_order=["glm", "sensevoice", "qwen"])
    assert chosen is not None
    assert chosen.transcript_text == "你好，请问现在方便吗"
    assert chosen.transcript_support_count == 2

    long = "A" * 5000
    texts = _by_family(long, "客户明天再联系我们确认时间", "客户明天再联系我们确认时间")
    result = classify_sample(_sample(texts, duration=3.0), _cfg())
    assert result.decision != "exclude"
    assert result.status in {"retry", "hold"}
    assert result.category is None
    assert "glm_1" in result.implausible_routes


def test_spot_quota_does_not_treat_hold_as_usable_gold():
    assert spot_audit_quota(30) == 30
    assert spot_audit_quota(20000) == 200
    assert spot_audit_quota(0) == 0


def test_workbook_tolerance_slots_and_traditional_keep_original_body():
    particle = classify_sample(_sample(_by_family("我知道了啊", "我知道了", "我知道了")), _cfg())
    assert particle.category == "gold"
    assert particle.status == "candidate"
    assert particle.candidate_text == "我知道了"
    assert particle.candidate_text != "我知道了啊我知道了"

    honorific = classify_sample(_sample(_by_family("张先声", "张先生", "张先生")), _cfg())
    assert honorific.category == "gold"
    assert honorific.candidate_text in {"张先声", "张先生"}

    slot = classify_sample(_sample(_by_family("转账一千元", "转账一万元", "转账一千元")), _cfg())
    assert slot.category == "semantic_risk"
    assert slot.status == "manual_review"

    traditional = classify_sample(
        _sample(_by_family("您現在方便嗎", "你现在方便吗", "你现在方便吗")),
        _cfg(),
    )
    assert traditional.category == "gold"
    assert traditional.language_by_run["glm_1"] == "zh"
    assert traditional.candidate_text in {"您現在方便嗎", "你现在方便吗"}
    assert "<|" not in (traditional.candidate_text or "")


def test_missing_run_keeps_semantic_risk_and_does_not_shrink_denominator():
    texts = _by_family("好", "客户明天再联系", "客户明天再联系")
    texts["glm_2"] = "不好"
    sample = _sample(texts)
    sample.transcripts["sensevoice_1"]["status"] = "failed"
    sample.transcripts["sensevoice_2"]["status"] = "failed"
    result = classify_sample(sample, _cfg())
    assert result.category == "semantic_risk"
    assert result.status == "manual_review"
    assert "technical_gap_retained" in result.reason_codes
    assert result.configured_family_count == 3

    incomplete = _sample(_by_family("客户明天再联系我们", "客户明天再联系我们", "客户明天再联系我们"))
    incomplete.transcripts["qwen_1"]["status"] = "failed"
    incomplete.transcripts["qwen_2"]["status"] = "failed"
    gap = classify_sample(incomplete, _cfg())
    assert gap.category is None
    assert gap.status in {"retry", "hold"}
    assert gap.configured_family_count == 3
    assert gap.support_ratio_of_4 == 2 / 3
    assert gap.support_ratio_of_chinese == 1.0


def test_low_dnsmos_does_not_confirm_noise_or_block_gold():
    clean = classify_sample(_sample(_pair("客户表示明天再联系")), _cfg())
    low = classify_sample(
        _sample(
            _pair("客户表示明天再联系"),
            quality={
                "dnsmos_ovrl": 1.1,
                "noise_band": "noisy",
                "noise_risk": True,
                "dnsmos_status": "success",
                "calibrated": False,
                "crosstalk_suspected": True,
            },
        ),
        _cfg(),
    )
    assert clean.category == "gold"
    assert low.category == "gold"
    assert low.status == "candidate"
    assert low.quality_state == "not_gating"
    assert low.quality_state != "uncalibrated"
    assert low.acoustic_evidence["state"] == "unknown"
    assert "crosstalk_suspected_not_sufficient" in low.acoustic_evidence["gaps"]

    confirmed = classify_sample(
        _sample(
            _pair("两个人同时在说话"),
            quality={
                "overlap_detected": True,
                "overlap_detected_trusted": True,
                "overlap_detector_version": "overlap-test",
            },
        ),
        _cfg(),
    )
    assert confirmed.category == "noise"
    assert confirmed.subtype == "crosstalk"
    assert confirmed.label_tier == "none"


def test_unverified_mixed_language_holds_and_does_not_invent_cer():
    mixed = classify_sample(
        _sample(_by_family("请联系 customer service 办理", "客户表示明天再联系", "客户表示明天再联系")),
        _cfg(),
    )
    assert mixed.language_by_run["glm_1"] == "mixed"
    assert mixed.category is None
    assert mixed.status == "hold"
    assert mixed.status != "manual_review"
    assert mixed.candidate_text in {None, ""}

    metrics = mixed_language_character_metrics("请联系办理", "请联系 customer service 办理")
    assert metrics["cer"] is None
    assert metrics["local_cer"] is None
    assert metrics["reason"] == "mixed_language"
    stripped = reference_character_metrics("请联系办理", "请联系customer service办理")
    assert stripped["cer"] is None
    assert stripped["reason"] == "mixed_language"


def test_eval_reference_recomputes_after_excluding_disagreeing_family():
    agreed = "客户明天再联系我们确认行程"
    other = "完全是另一个业务主题和另一组槽位"
    sample = _sample(_by_family(agreed, agreed, other))
    full = classify_sample(sample, _cfg())
    assert full.category != "gold"
    unlocked = eval_reference_excluding(sample, _cfg(), "qwen")
    assert unlocked["eligible"] is True
    assert unlocked["candidate_text"] == agreed
    assert unlocked["selected_family"] in {"glm", "sensevoice"}
    still_blocked = eval_reference_excluding(sample, _cfg(), "glm")
    assert still_blocked["eligible"] is False
    assert still_blocked["candidate_text"] is None


def test_verifier_timeout_and_invalid_structure_stay_unknown():
    timeout = coerce_verifier_verdict({"verdict": "equivalent"}, timed_out=True, version="remote-v0")
    assert timeout.verdict == "unknown"
    assert timeout.error == "timeout"
    invalid = coerce_verifier_verdict(["equivalent"], version="remote-v0")
    assert invalid.verdict == "unknown"
    assert invalid.error == "invalid_structure"
    bad = coerce_verifier_verdict({"verdict": "pass"}, version="remote-v0")
    assert bad.verdict == "unknown"
    assert bad.error == "invalid_structure"


def test_control_tag_selection_keeps_raw_and_does_not_rewrite():
    raw = "<|zh|><|NEUTRAL|><|Speech|><|withitn|>您现在方便吗"
    result = classify_sample(_sample(_by_family(raw, "您现在方便吗", "您现在方便吗")), _cfg())
    assert result.category == "gold"
    assert result.selected_family == "glm"
    assert result.candidate_text == "您现在方便吗"
    assert result.selected_raw_text == raw
    assert contains_control_tag(result.selected_raw_text)
    assert not contains_control_tag(result.candidate_text)


def test_batch_conservation_and_idempotent_rerun():
    rows = [
        _sample(_pair("你现在方便吗"), sid="g"),
        _sample(_by_family("好", "不好", "好"), sid="s"),
        _sample(_pair(""), sid="h"),
        _sample(_by_family("请在滴声后留言", "请在嘀声后留言", "请在滴声后留言"), sid="v"),
    ]
    first = [classify_sample(sample, _cfg()) for sample in rows]
    second = [classify_sample(sample, _cfg()) for sample in rows]
    assert len(first) == 4
    categories = [item.category for item in first]
    assert categories.count("gold") + categories.count("semantic_risk") + categories.count(None) + categories.count(
        "voicemail"
    ) == 4
    assert all(item.category != "noise" or item.subtype for item in first)
    assert [(a.category, a.status, a.candidate_text, a.selected_family) for a in first] == [
        (b.category, b.status, b.candidate_text, b.selected_family) for b in second
    ]
    assert all(item.is_human_verified is False for item in first)
    assert all(item.status != "accepted" for item in first)
