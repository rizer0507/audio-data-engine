"""Versioned DNSMOS decision states for selection_five_class_v2_2_auto_noise.

Derives ``dnsmos_noise_state`` / ``dnsmos_speech_state`` from SIG/BAK/OVRL.
Thresholds live only in ``configs/quality/dnsmos_decision_v2_2.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DNSMOS_NOISE_NOISY = "noisy"
DNSMOS_NOISE_MODERATE = "moderate"
DNSMOS_NOISE_CLEAN = "clean"
DNSMOS_NOISE_UNAVAILABLE = "unavailable"

DNSMOS_SPEECH_STRONG = "strong"
DNSMOS_SPEECH_WEAK = "weak"
DNSMOS_SPEECH_UNKNOWN = "unknown"
DNSMOS_SPEECH_UNAVAILABLE = "unavailable"

DNSMOS_NOISE_STATES = frozenset(
    {
        DNSMOS_NOISE_NOISY,
        DNSMOS_NOISE_MODERATE,
        DNSMOS_NOISE_CLEAN,
        DNSMOS_NOISE_UNAVAILABLE,
    }
)
DNSMOS_SPEECH_STATES = frozenset(
    {
        DNSMOS_SPEECH_STRONG,
        DNSMOS_SPEECH_WEAK,
        DNSMOS_SPEECH_UNKNOWN,
        DNSMOS_SPEECH_UNAVAILABLE,
    }
)

@dataclass(frozen=True)
class DnsmosDecisionConfig:
    policy_version: str
    clean_bak: float
    clean_ovrl: float
    noisy_bak: float
    noisy_ovrl: float
    strong_sig: float
    weak_sig: float
    noisy_operator: str = "or"
    require_status_success: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "thresholds": {
                "clean_bak": self.clean_bak,
                "clean_ovrl": self.clean_ovrl,
                "noisy_bak": self.noisy_bak,
                "noisy_ovrl": self.noisy_ovrl,
                "strong_sig": self.strong_sig,
                "weak_sig": self.weak_sig,
            },
            "decision": {
                "noisy_operator": self.noisy_operator,
                "require_status_success": self.require_status_success,
            },
        }

def _finite(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"dnsmos_decision threshold {name} must be numeric, got {value!r}") from exc
    if number != number or number in {float("inf"), float("-inf")}:
        raise ValueError(f"dnsmos_decision threshold {name} must be finite, got {value!r}")
    return number

def load_dnsmos_decision_config(path: str | Path | None = None, raw: dict[str, Any] | None = None) -> DnsmosDecisionConfig:
    data: dict[str, Any] = {}
    if path:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"dnsmos_decision config must be a mapping: {path}")
        data.update(loaded)
    if raw:
        data.update(raw)
    thresholds = data.get("thresholds") if isinstance(data.get("thresholds"), dict) else {}
    decision = data.get("decision") if isinstance(data.get("decision"), dict) else {}
    policy = str(data.get("policy_version") or data.get("dnsmos_decision_policy_version") or "").strip()
    if not policy:
        raise ValueError("dnsmos_decision.policy_version must be non-empty")
    clean_bak = _finite(thresholds.get("clean_bak", data.get("clean_bak")), name="clean_bak")
    clean_ovrl = _finite(thresholds.get("clean_ovrl", data.get("clean_ovrl")), name="clean_ovrl")
    noisy_bak = _finite(thresholds.get("noisy_bak", data.get("noisy_bak")), name="noisy_bak")
    noisy_ovrl = _finite(thresholds.get("noisy_ovrl", data.get("noisy_ovrl")), name="noisy_ovrl")
    strong_sig = _finite(thresholds.get("strong_sig", data.get("strong_sig")), name="strong_sig")
    weak_sig = _finite(thresholds.get("weak_sig", data.get("weak_sig")), name="weak_sig")
    if clean_bak < noisy_bak:
        raise ValueError(f"clean_bak ({clean_bak}) must be >= noisy_bak ({noisy_bak})")
    if clean_ovrl < noisy_ovrl:
        raise ValueError(f"clean_ovrl ({clean_ovrl}) must be >= noisy_ovrl ({noisy_ovrl})")
    if strong_sig <= weak_sig:
        raise ValueError(f"strong_sig ({strong_sig}) must be > weak_sig ({weak_sig})")
    op = str(decision.get("noisy_operator") or data.get("noisy_operator") or "or").strip().lower()
    if op not in {"or", "and"}:
        raise ValueError(f"dnsmos_decision.noisy_operator must be 'or' or 'and', got {op!r}")
    return DnsmosDecisionConfig(
        policy_version=policy,
        clean_bak=clean_bak,
        clean_ovrl=clean_ovrl,
        noisy_bak=noisy_bak,
        noisy_ovrl=noisy_ovrl,
        strong_sig=strong_sig,
        weak_sig=weak_sig,
        noisy_operator=op,
        require_status_success=bool(
            decision.get(
                "require_status_success",
                data.get("require_status_success", True),
            )
        ),
    )

def _opt_score(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number

@dataclass(frozen=True)
class DnsmosDecisionEvidence:
    noise_state: str
    speech_state: str
    policy_version: str
    status: str | None
    sig: float | None
    bak: float | None
    ovrl: float | None
    model_digest: str = ""
    preprocess_version: str = ""
    consumed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "dnsmos_noise_state": self.noise_state,
            "dnsmos_speech_state": self.speech_state,
            "dnsmos_decision_policy_version": self.policy_version,
            "dnsmos_status": self.status,
            "dnsmos_sig": self.sig,
            "dnsmos_bak": self.bak,
            "dnsmos_ovrl": self.ovrl,
            "dnsmos_model_digest": self.model_digest,
            "dnsmos_preprocess_version": self.preprocess_version,
            "dnsmos_decision_consumed": self.consumed,
        }

def derive_dnsmos_decision(
    quality: dict[str, Any] | None,
    config: DnsmosDecisionConfig,
) -> DnsmosDecisionEvidence:
    """Map raw DNSMOS fields to versioned noise/speech states.

    Failed/unsupported/missing scores become ``unavailable``. Never invent noisy.
    """
    q = quality if isinstance(quality, dict) else {}
    status = str(q.get("dnsmos_status") or "").strip().lower() or None
    sig = _opt_score(q.get("dnsmos_sig"))
    bak = _opt_score(q.get("dnsmos_bak"))
    ovrl = _opt_score(q.get("dnsmos_ovrl"))
    digest = str(q.get("dnsmos_model_digest") or "")
    preprocess = str(q.get("dnsmos_preprocess_version") or "")

    unavailable = DnsmosDecisionEvidence(
        noise_state=DNSMOS_NOISE_UNAVAILABLE,
        speech_state=DNSMOS_SPEECH_UNAVAILABLE,
        policy_version=config.policy_version,
        status=status,
        sig=sig,
        bak=bak,
        ovrl=ovrl,
        model_digest=digest,
        preprocess_version=preprocess,
        consumed=False,
    )

    if config.require_status_success and status != "success":
        return unavailable
    if None in {sig, bak, ovrl}:
        return unavailable

    assert sig is not None and bak is not None and ovrl is not None
    bak_noisy = bak < config.noisy_bak
    ovrl_noisy = ovrl < config.noisy_ovrl
    if config.noisy_operator == "and":
        is_noisy = bak_noisy and ovrl_noisy
    else:
        is_noisy = bak_noisy or ovrl_noisy

    if is_noisy:
        noise_state = DNSMOS_NOISE_NOISY
    elif bak >= config.clean_bak and ovrl >= config.clean_ovrl:
        noise_state = DNSMOS_NOISE_CLEAN
    else:
        noise_state = DNSMOS_NOISE_MODERATE

    if sig >= config.strong_sig:
        speech_state = DNSMOS_SPEECH_STRONG
    elif sig <= config.weak_sig:
        speech_state = DNSMOS_SPEECH_WEAK
    else:
        speech_state = DNSMOS_SPEECH_UNKNOWN

    return DnsmosDecisionEvidence(
        noise_state=noise_state,
        speech_state=speech_state,
        policy_version=config.policy_version,
        status=status,
        sig=sig,
        bak=bak,
        ovrl=ovrl,
        model_digest=digest,
        preprocess_version=preprocess,
        consumed=True,
    )
