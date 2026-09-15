"""Configuration loader for selection_v3.0 (contract + classification)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from audio_engine.core.selection_v3.types import (
    DEFAULT_EXPECTED_RUNS_PER_FAMILY,
    DEFAULT_TARGET_FAMILY,
    DEFAULT_TEACHER_FAMILIES,
    MIN_MODEL_FAMILIES,
    RULE_VERSION,
)


@dataclass
class RunIdentity:
    """Explicit identity of one ASR result artifact (one of 2N configured routes)."""

    run_id: str
    family: str
    model_checkpoint_digest: str = ""
    decode_config_digest: str = ""
    prompt_digest: str = ""
    input_audio_digest: str = ""
    input_audio_key: str = "resampled_16k"
    created_at: str = ""
    transcript_key: str = ""
    artifact_id: str = ""
    execution_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, family: str | None = None) -> RunIdentity:
        run_id = str(data.get("run_id") or data.get("id") or "").strip()
        if not run_id:
            raise ValueError("run identity requires run_id")
        fam = str(data.get("family") or family or "").strip()
        if not fam:
            raise ValueError(f"run identity {run_id!r} requires family")
        transcript_key = str(
            data.get("transcript_key") or data.get("model") or run_id
        ).strip()
        return cls(
            run_id=run_id,
            family=fam,
            model_checkpoint_digest=str(
                data.get("model_checkpoint_digest") or data.get("model") or ""
            ),
            decode_config_digest=str(data.get("decode_config_digest") or ""),
            prompt_digest=str(data.get("prompt_digest") or ""),
            input_audio_digest=str(data.get("input_audio_digest") or ""),
            input_audio_key=str(data.get("input_audio_key") or "resampled_16k"),
            created_at=str(data.get("created_at") or ""),
            transcript_key=transcript_key,
            artifact_id=str(data.get("artifact_id") or ""),
            execution_id=str(data.get("execution_id") or ""),
        )


@dataclass
class SelectionV3Config:
    """Family / run contract + classification thresholds for consensus_v3."""

    engine: str = "consensus_v3"
    rule_version: str = RULE_VERSION
    policy_version: str = "selection_zh_asr_v3_0"
    target_family: str = DEFAULT_TARGET_FAMILY
    model_families: dict[str, list[str]] = field(default_factory=dict)
    teacher_families: list[str] = field(
        default_factory=lambda: list(DEFAULT_TEACHER_FAMILIES)
    )
    expected_runs_per_family: int = DEFAULT_EXPECTED_RUNS_PER_FAMILY
    runs: list[RunIdentity] = field(default_factory=list)
    # Aliases: transcript_key / file stem → family (never guessed from filename alone)
    run_aliases: dict[str, str] = field(default_factory=dict)

    # Similarity / consensus
    family_threshold: float = 0.98
    teacher_consensus_threshold: float = 0.98
    pseudo_high_min_similarity: float = 0.98
    pseudo_medium_min_similarity: float = 0.95
    pseudo_medium_min_stable_families: int = 3

    # Short utterance
    short_audio_sec: float = 2.0
    short_text_chars: int = 6

    # Implausible speech rate (018): len(comparison_text) / duration
    # Only texts with >= min_text_chars are checked (avoid sub-second false positives).
    max_chars_per_sec: float = 25.0
    speech_rate_min_text_chars: int = 80
    # exclude | manual_review — production default is exclude (drop from auto pools).
    speech_rate_disposition: str = "exclude"

    # Lexicon / patterns
    negative_phrases: list[str] = field(default_factory=list)
    positive_phrases: list[str] = field(default_factory=list)
    critical_tokens: list[str] = field(default_factory=list)
    filler_phrases: list[str] = field(default_factory=list)
    affirmation_phrases: list[str] = field(default_factory=list)
    profanity_or_reject: list[str] = field(default_factory=list)
    punctuation_to_strip: list[str] = field(
        default_factory=lambda: list("，。！？、；：""''（）【】《》…—·,.!?;:'\"()[]{}")
    )
    voicemail_patterns_path: str = ""
    semantic_lexicon_path: str = ""

    # Calibration gate: when false, noisy risk stays unknown even if scores exist
    quality_calibrated: bool = False
    # 022 semantic-tolerant options. Ignored unless rule_version selects that rule.
    recall_max_distance: float = 0.10
    divergence_min_distance: float = 0.25
    min_support_ratio: float = 2 / 3
    min_support_families: int = 2
    short_pair_max_chars: int = 6
    tolerance_version: str = "text_tolerance_v1"
    homophone_pairs: list[tuple[str, str]] = field(default_factory=list)
    voicemail_strong_path: str = ""
    semantic_verifier_mode: str = "local"
    semantic_verifier_endpoint: str = ""
    semantic_verifier_protocol: str = "auto"
    semantic_verifier_model: str = ""
    semantic_verifier_timeout_sec: float = 5.0
    max_route_retries: int = 1
    # 024: disclose when this batch was already inspected while designing the rule.
    prior_information_used: bool = False
    # 020 refactor routing: off = legacy queues (plus same-text semantic fix);
    # shadow = write disposition/quality_state without changing type/queue;
    # on = uncalibrated quality becomes calibration_hold (not transcription jobs).
    refactor_020_mode: str = "off"
    # 023: asr_anomaly_noise_v1 scores only ASR anomalies. legacy_full_quality_gate
    # is the explicit rollback that still requires a full DNSMOS pass.
    noise_policy: str = "legacy_full_quality_gate"
    # 025: chinese_only_v1 is the shared classify-text pre-layer. Default legacy
    # keeps historical v3.0 buckets; the shadow pipeline turns the new policy on.
    classify_text_policy: str = "legacy"
    classify_text_keep_digits: bool = True
    classify_text_echo_missing: str = "fail"
    classify_text_echo: dict[str, Any] = field(default_factory=dict)
    classify_text_echo_fingerprint: str = ""
    # 027 five-class: reproducible weighted gold selection (Qwen=2, others=1).
    selection_seed: str = "selection_five_class_v1"
    family_selection_weights: dict[str, float] = field(default_factory=dict)
    gold_min_stable_families: int = 3
    short_polarity_max_han_chars: int = 4
    noise_call_counter: list[int] | None = field(default=None, repr=False)

    def all_transcript_keys(self) -> list[str]:
        """Ordered unique keys expected across configured families."""
        keys: list[str] = []
        seen: set[str] = set()
        if self.runs:
            for run in self.runs:
                key = run.transcript_key or run.run_id
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
            return keys
        for family in sorted(self.model_families):
            for key in self.model_families[family]:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        return keys

    def family_of_key(self, transcript_key: str) -> str | None:
        key = str(transcript_key).strip()
        if key in self.run_aliases:
            return self.run_aliases[key]
        for run in self.runs:
            if run.transcript_key == key or run.run_id == key:
                return run.family
        for family, keys in self.model_families.items():
            if key in keys:
                return family
        return None

    def uses_chinese_only_text(self) -> bool:
        from audio_engine.core.selection_v3.classify_text import uses_chinese_only_text

        return uses_chinese_only_text(self.classify_text_policy)

    def echo_table_for(self, family: str | None):
        from audio_engine.core.selection_v3.classify_text import echo_for_family

        return echo_for_family(self.classify_text_echo, family)

    def ordered_families(self) -> list[str]:
        """Stable family order: teachers then target, then any extras."""
        ordered: list[str] = []
        for family in list(self.teacher_families) + [self.target_family]:
            if family in self.model_families and family not in ordered:
                ordered.append(family)
        for family in sorted(self.model_families):
            if family not in ordered:
                ordered.append(family)
        return ordered

    @property
    def configured_family_count(self) -> int:
        return len(self.model_families)

    @property
    def expected_total_runs(self) -> int:
        return self.configured_family_count * self.expected_runs_per_family

    def validate_family_config(self) -> None:
        """Fail-fast on duplicate family membership or wrong run counts.

        Contract (015): N ≥ 3 families, exactly ``expected_runs_per_family`` (2)
        runs each, teachers = non-target configured families (count N−1).
        Four families / eight routes remain a valid recommended shape, not a floor.
        """
        if not self.model_families:
            raise ValueError("model_families must be configured explicitly")
        seen_keys: dict[str, str] = {}
        for family, keys in self.model_families.items():
            if not keys:
                raise ValueError(f"family {family!r} has empty run list")
            if len(keys) != self.expected_runs_per_family:
                raise ValueError(
                    f"family {family!r} expects {self.expected_runs_per_family} runs, "
                    f"got {len(keys)}: {keys}"
                )
            if len(set(keys)) != len(keys):
                raise ValueError(f"family {family!r} has duplicate run aliases: {keys}")
            for key in keys:
                if key in seen_keys:
                    raise ValueError(
                        f"run alias {key!r} assigned to both "
                        f"{seen_keys[key]!r} and {family!r}"
                    )
                seen_keys[key] = family
        n = self.configured_family_count
        if n < MIN_MODEL_FAMILIES:
            raise ValueError(
                f"consensus_v3 requires at least {MIN_MODEL_FAMILIES} model families, "
                f"got {n}"
            )
        if self.expected_runs_per_family != 2:
            raise ValueError(
                "consensus_v3 requires expected_runs_per_family=2 "
                f"(got {self.expected_runs_per_family})"
            )
        teachers = list(self.teacher_families)
        if len(teachers) != n - 1 or len(set(teachers)) != n - 1:
            raise ValueError(
                f"consensus_v3 requires exactly {n - 1} distinct teacher families "
                f"(N-1 for configured_family_count={n}), got {teachers}"
            )
        if self.target_family not in self.model_families:
            raise ValueError(
                f"target_family {self.target_family!r} not in model_families"
            )
        expected_teachers = sorted(
            f for f in self.model_families if f != self.target_family
        )
        if sorted(teachers) != expected_teachers:
            raise ValueError(
                "teacher_families must be exactly the non-target configured families; "
                f"expected {expected_teachers}, got {sorted(teachers)}"
            )
        for teacher in teachers:
            if teacher == self.target_family:
                raise ValueError("target_family must not be listed as a teacher_family")
        if self.runs:
            expected_n = self.expected_total_runs
            if len(self.runs) != expected_n or {r.transcript_key for r in self.runs} != set(
                seen_keys
            ):
                raise ValueError(
                    f"runs must cover exactly the {expected_n} configured transcript keys "
                    f"(2N for N={n})"
                )
            if any(seen_keys.get(r.transcript_key) != r.family for r in self.runs):
                raise ValueError("run family differs from model_families")
            if any(not all((r.model_checkpoint_digest, r.decode_config_digest, r.prompt_digest,
                            r.input_audio_digest, r.created_at)) for r in self.runs):
                raise ValueError("every run requires checkpoint/decode/prompt/audio digests and execution timestamp")
            run_ids = [r.run_id for r in self.runs]
            if len(run_ids) != len(set(run_ids)):
                raise ValueError("runs contain duplicate run_id values")
            digests = [
                (
                    r.family,
                    r.model_checkpoint_digest,
                    r.decode_config_digest,
                    r.prompt_digest,
                    r.input_audio_digest,
                    r.created_at,
                )
                for r in self.runs
            ]
            for i, left in enumerate(digests):
                for j, right in enumerate(digests):
                    if i >= j:
                        continue
                    independent = (self.runs[i].execution_id and self.runs[j].execution_id
                                   and self.runs[i].execution_id != self.runs[j].execution_id)
                    if left == right and all(left) and not independent:
                        raise ValueError(
                            "duplicate run identity digests detected — "
                            "copying one artifact twice is not dual-run "
                            f"(run_id={self.runs[i].run_id!r} vs {self.runs[j].run_id!r})"
                        )

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> SelectionV3Config:
        families_raw = params.get("model_families") or {}
        if not isinstance(families_raw, dict) or not families_raw:
            raise ValueError("selection_v3 requires explicit model_families")
        families = {
            str(k): [str(x) for x in (v or [])]
            for k, v in families_raw.items()
        }
        teachers = params.get("teacher_families")
        target = str(params.get("target_family") or DEFAULT_TARGET_FAMILY)
        if teachers is None:
            # Prefer known default order, then any remaining non-target families.
            teachers = [
                f
                for f in DEFAULT_TEACHER_FAMILIES
                if f in families and f != target
            ]
            for family in sorted(families):
                if family != target and family not in teachers:
                    teachers.append(family)
        runs_raw = params.get("runs") or []
        runs = [
            RunIdentity.from_dict(item) if isinstance(item, dict) else item
            for item in runs_raw
        ]
        aliases_raw = params.get("run_aliases") or {}
        aliases = {str(k): str(v) for k, v in aliases_raw.items()}
        for family, keys in families.items():
            for key in keys:
                aliases.setdefault(key, family)

        similarity = params.get("similarity") or {}
        short = params.get("short_utterance") or {}
        speech_rate = params.get("speech_rate") or {}
        quality = params.get("quality") or {}
        tolerance = params.get("tolerance") or {}
        verifier = params.get("semantic_verifier") or {}
        consensus_req = params.get("consensus") or {}
        classify_text = params.get("classify_text") or {}
        selection = params.get("selection") or params.get("gold_selection") or {}
        homophone_pairs = []
        for item in tolerance.get("homophone_pairs") or []:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                homophone_pairs.append((str(item[0]), str(item[1])))

        weight_raw = (
            selection.get("family_weights")
            or params.get("family_selection_weights")
            or {}
        )
        family_weights = {
            str(k): float(v)
            for k, v in (weight_raw.items() if isinstance(weight_raw, dict) else {})
        }

        negative: list[str] = []
        positive: list[str] = []
        critical: list[str] = []
        reject: list[str] = []
        fillers: list[str] = []
        affirmations: list[str] = []
        punctuation: list[str] | None = None

        semantic = params.get("semantic") or {}
        if semantic:
            negative = [str(x) for x in (semantic.get("negative") or []) if str(x).strip()]
            positive = [str(x) for x in (semantic.get("positive") or []) if str(x).strip()]
            critical = [
                str(x) for x in (semantic.get("critical_tokens") or []) if str(x).strip()
            ]
            reject = [
                str(x)
                for x in (semantic.get("profanity_or_reject") or [])
                if str(x).strip()
            ]
            fillers = [str(x) for x in (semantic.get("filler") or []) if str(x).strip()]
            affirmations = [
                str(x) for x in (semantic.get("affirmation") or []) if str(x).strip()
            ]
            if semantic.get("punctuation_to_strip") is not None:
                punctuation = [
                    str(x) for x in semantic.get("punctuation_to_strip") if str(x)
                ]

        lexicon_path = str(params.get("semantic_lexicon_path") or "").strip()
        if lexicon_path:
            loaded = _load_lexicon(lexicon_path)
            if not negative:
                negative = loaded.get("negative") or []
            if not positive:
                positive = loaded.get("positive") or []
            if not critical:
                critical = loaded.get("critical_tokens") or []
            if not reject:
                reject = loaded.get("profanity_or_reject") or []
            if not fillers:
                fillers = loaded.get("filler") or []
            if not affirmations:
                affirmations = loaded.get("affirmation") or []
            if punctuation is None and loaded.get("punctuation_to_strip"):
                punctuation = loaded["punctuation_to_strip"]

        if not fillers:
            fillers = ["嗯嗯", "嗯", "啊", "哦", "呃", "额", "唔"]
        if not affirmations:
            affirmations = list(positive) or ["需要", "可以", "是", "有", "好的", "好"]

        cfg = cls(
            engine=str(params.get("engine") or "consensus_v3"),
            rule_version=str(params.get("rule_version") or RULE_VERSION),
            policy_version=str(params.get("policy_version") or "selection_zh_asr_v3_0"),
            target_family=target,
            model_families=families,
            teacher_families=[str(x) for x in teachers],
            expected_runs_per_family=int(
                params.get("expected_runs_per_family", DEFAULT_EXPECTED_RUNS_PER_FAMILY)
            ),
            runs=list(runs),
            run_aliases=aliases,
            family_threshold=float(
                similarity.get(
                    "family_threshold",
                    params.get("family_threshold", 0.98),
                )
            ),
            teacher_consensus_threshold=float(
                similarity.get(
                    "teacher_consensus_threshold",
                    params.get("teacher_consensus_threshold", 0.98),
                )
            ),
            pseudo_high_min_similarity=float(
                similarity.get(
                    "pseudo_high_min_similarity",
                    params.get("pseudo_high_min_similarity", 0.98),
                )
            ),
            pseudo_medium_min_similarity=float(
                similarity.get(
                    "pseudo_medium_min_similarity",
                    params.get("pseudo_medium_min_similarity", 0.95),
                )
            ),
            pseudo_medium_min_stable_families=int(
                similarity.get(
                    "pseudo_medium_min_stable_families",
                    params.get("pseudo_medium_min_stable_families", 3),
                )
            ),
            short_audio_sec=float(short.get("max_audio_sec", 2.0)),
            short_text_chars=int(short.get("max_text_chars", 6)),
            max_chars_per_sec=float(
                speech_rate.get(
                    "max_chars_per_sec",
                    params.get("max_chars_per_sec", 25.0),
                )
            ),
            speech_rate_min_text_chars=int(
                speech_rate.get(
                    "min_text_chars",
                    params.get("speech_rate_min_text_chars", 80),
                )
            ),
            speech_rate_disposition=str(
                speech_rate.get(
                    "disposition",
                    params.get("speech_rate_disposition", "exclude"),
                )
                or "exclude"
            )
            .strip()
            .lower(),
            negative_phrases=negative,
            positive_phrases=positive,
            critical_tokens=critical,
            filler_phrases=fillers,
            affirmation_phrases=affirmations,
            profanity_or_reject=reject,
            punctuation_to_strip=punctuation
            if punctuation is not None
            else list("，。！？、；：\"\"''（）【】《》…—·,.!?;:'\"()[]{}"),
            voicemail_patterns_path=str(params.get("voicemail_patterns_path") or ""),
            semantic_lexicon_path=lexicon_path,
            quality_calibrated=bool(quality.get("calibrated", params.get("quality_calibrated", False))),
            refactor_020_mode=_normalize_refactor_020_mode(
                params.get("refactor_020_mode")
                if params.get("refactor_020_mode") is not None
                else quality.get("refactor_020_mode")
            ),
            recall_max_distance=float(
                tolerance.get("recall_max_distance", params.get("recall_max_distance", 0.10))
            ),
            divergence_min_distance=float(
                tolerance.get(
                    "divergence_min_distance",
                    params.get("divergence_min_distance", 0.25),
                )
            ),
            min_support_ratio=float(
                consensus_req.get("min_support_ratio", params.get("min_support_ratio", 2 / 3))
            ),
            min_support_families=int(
                consensus_req.get("min_support_families", params.get("min_support_families", 2))
            ),
            short_pair_max_chars=int(
                tolerance.get(
                    "short_max_chars",
                    short.get("max_text_chars", params.get("short_pair_max_chars", 6)),
                )
            ),
            tolerance_version=str(
                tolerance.get("version") or params.get("tolerance_version") or "text_tolerance_v1"
            ),
            homophone_pairs=homophone_pairs,
            voicemail_strong_path=str(
                params.get("voicemail_strong_path") or params.get("voicemail_strong_patterns_path") or ""
            ),
            semantic_verifier_mode=str(verifier.get("mode") or params.get("semantic_verifier_mode") or "local"),
            semantic_verifier_endpoint=str(verifier.get("endpoint") or ""),
            semantic_verifier_protocol=str(
                verifier.get("protocol") or params.get("semantic_verifier_protocol") or "auto"
            ),
            semantic_verifier_model=str(verifier.get("model") or params.get("semantic_verifier_model") or ""),
            semantic_verifier_timeout_sec=float(
                verifier.get("timeout_sec", params.get("semantic_verifier_timeout_sec", 5.0)) or 5.0
            ),
            max_route_retries=int(params.get("max_route_retries", 1)),
            prior_information_used=bool(params.get("prior_information_used", False)),
            noise_policy=str(
                params.get("noise_policy")
                or quality.get("noise_policy")
                or "legacy_full_quality_gate"
            ),
            classify_text_policy=str(
                classify_text.get("policy")
                or params.get("classify_text_policy")
                or "legacy"
            ),
            classify_text_keep_digits=bool(
                classify_text.get(
                    "keep_digits",
                    params.get("classify_text_keep_digits", True),
                )
            ),
            classify_text_echo_missing=str(
                classify_text.get("echo_missing")
                or params.get("classify_text_echo_missing")
                or "fail"
            )
            .strip()
            .lower(),
            selection_seed=str(
                selection.get("seed")
                or params.get("selection_seed")
                or "selection_five_class_v1"
            ),
            family_selection_weights=family_weights,
            gold_min_stable_families=int(
                selection.get(
                    "min_stable_families",
                    params.get("gold_min_stable_families", 3),
                )
            ),
            short_polarity_max_han_chars=int(
                selection.get(
                    "short_polarity_max_han_chars",
                    params.get("short_polarity_max_han_chars", 4),
                )
            ),
        )
        cfg.validate_family_config()
        disposition = cfg.speech_rate_disposition
        if disposition not in {"exclude", "manual_review", "route_quarantine"}:
            raise ValueError(
                "speech_rate.disposition must be 'exclude', 'manual_review', or "
                f"'route_quarantine', got {disposition!r}"
            )
        if cfg.refactor_020_mode not in {"off", "shadow", "on"}:
            raise ValueError(
                "refactor_020_mode must be 'off', 'shadow', or 'on', "
                f"got {cfg.refactor_020_mode!r}"
            )
        from audio_engine.core.selection_v3.noise_trigger import normalize_noise_policy
        from audio_engine.core.selection_v3.classify_text import (
            echo_fingerprint,
            load_echo_tables,
            normalize_classify_text_policy,
            uses_chinese_only_text,
        )

        cfg.noise_policy = normalize_noise_policy(cfg.noise_policy)
        cfg.classify_text_policy = normalize_classify_text_policy(cfg.classify_text_policy)
        from audio_engine.core.selection_v3.types import is_five_class_rule

        if is_five_class_rule(cfg.rule_version) and cfg.classify_text_policy == "legacy":
            cfg.classify_text_policy = normalize_classify_text_policy("five_class_v1")
        if cfg.classify_text_echo_missing not in {"fail", "echo_list_missing"}:
            raise ValueError(
                "classify_text.echo_missing must be 'fail' or 'echo_list_missing', "
                f"got {cfg.classify_text_echo_missing!r}"
            )
        if uses_chinese_only_text(cfg.classify_text_policy):
            echo_cfg = classify_text.get("echo") or params.get("classify_text_echo") or {}
            extra_exact = classify_text.get("extra_exact") or params.get("classify_text_extra_exact") or []
            cfg.classify_text_echo = load_echo_tables(
                families=cfg.ordered_families(),
                echo_cfg=echo_cfg if isinstance(echo_cfg, dict) else {},
                extra_exact=extra_exact,
                missing=cfg.classify_text_echo_missing,
            )
            cfg.classify_text_echo_fingerprint = echo_fingerprint(cfg.classify_text_echo)
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> SelectionV3Config:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"config must be a mapping: {path}")
        return cls.from_params(raw)


def _normalize_refactor_020_mode(value: Any) -> str:
    """YAML ``on``/``off`` become bools; accept those and string forms."""
    if value is None:
        return "off"
    if isinstance(value, bool):
        return "on" if value else "off"
    text = str(value).strip().lower()
    if text in {"true", "yes", "1"}:
        return "on"
    if text in {"false", "no", "0", ""}:
        return "off"
    return text


def _load_lexicon(path: str | Path) -> dict[str, list[str]]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    result: dict[str, list[str]] = {}
    for key in (
        "negative",
        "positive",
        "critical_tokens",
        "profanity_or_reject",
        "filler",
        "affirmation",
        "punctuation_to_strip",
    ):
        result[key] = [str(x) for x in (raw.get(key) or []) if str(x).strip() or key == "punctuation_to_strip"]
    return result
