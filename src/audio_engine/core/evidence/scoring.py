"""Neutral acoustic scoring interfaces (044).

``RealAcousticScorer`` executes the neutral backends (silero VAD, lightweight energy,
DNSMOS P.835) for the requested dependencies and records *planned* vs *actual* model
calls in a :class:`ScoreLedger`. Backends are injectable so scheduling can be tested
without ONNX files.

Rules enforced here:
- an empty / status-less backend result is **not** a computed fact (it is rejected and
  surfaces as ``failed`` with a locatable error, and is counted as ``rejected``);
- a backend that cannot run (missing model/config) is ``unavailable`` -> ``failed`` fact;
- an existing SUCCESS fact in the shared store is reused (``cache_hits``), never recomputed;
- nothing here imports tagging or selection_v3.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from audio_engine.core.evidence.store import (
    DEP_STATUS_KEYS,
    AcousticFact,
    SharedAcousticEvidenceStore,
)
from audio_engine.core.evidence.types import (
    ACOUSTIC_DEPS,
    ENERGY_STATE_FAILED,
    FACT_FAILED,
    FACT_SUCCESS,
    normalize_fact_status,
)

# backend(audio_path, config_path) -> quality-style fields
Backend = Callable[[str, "str | None"], "dict[str, Any]"]

# Value keys a SUCCESS result must carry (otherwise it is an empty shell).
_REQUIRED_SUCCESS_KEYS: dict[str, tuple[str, ...]] = {
    "vad": ("vad_speech_state",),
    "energy": ("energy_state",),
    "dnsmos": ("dnsmos_sig",),
}


class AcousticScorer(Protocol):
    name: str

    def score(
        self,
        dep: str,
        *,
        sample_id: str,
        audio_path: str,
        config_path: str | None = None,
    ) -> AcousticFact: ...


@dataclass
class ScoreLedger:
    """Counts of what was planned vs what really ran."""

    planned: Counter = field(default_factory=Counter)
    actual: Counter = field(default_factory=Counter)
    cache_hits: Counter = field(default_factory=Counter)
    failed: Counter = field(default_factory=Counter)
    unavailable: Counter = field(default_factory=Counter)
    rejected: Counter = field(default_factory=Counter)
    errors: list[dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "planned": dict(self.planned),
            "actual": dict(self.actual),
            "cache_hits": dict(self.cache_hits),
            "failed": dict(self.failed),
            "unavailable": dict(self.unavailable),
            "rejected": dict(self.rejected),
            "errors": list(self.errors[:50]),
        }


def result_is_computed(dep: str, fields: Mapping[str, Any] | None) -> bool:
    """A result counts as computed only with an explicit status and real values.

    ``failed`` with an error is a legitimate computed *failure*; an empty dict, a result
    with only unrelated keys, or ``success`` without the dep's value keys is rejected.
    """
    if not fields:
        return False
    status = normalize_fact_status(fields.get(DEP_STATUS_KEYS[dep]))
    raw_status = str(fields.get(DEP_STATUS_KEYS[dep]) or "").strip().lower()
    if raw_status in {"", "not_required", "pending", "not_computed"}:
        return False
    if status == FACT_FAILED:
        return True
    if status != FACT_SUCCESS:
        return False
    return all(fields.get(k) is not None for k in _REQUIRED_SUCCESS_KEYS[dep])


def fact_from_result(dep: str, fields: Mapping[str, Any], *, source: str = "computed") -> AcousticFact:
    from audio_engine.core.evidence.store import DEP_VALUE_KEYS

    status = normalize_fact_status(fields.get(DEP_STATUS_KEYS[dep]))
    values = {k: fields[k] for k in DEP_VALUE_KEYS[dep] if k in fields and fields[k] is not None}
    error = fields.get(f"{dep}_error")
    return AcousticFact(
        dep=dep,
        status=status,
        values=values,
        error=str(error) if error else None,
        source=source,
    )


# --- default backends ---------------------------------------------------------------


class _BackendUnavailable(RuntimeError):
    pass


def _read_yaml_mapping(path: str | None, *, label: str) -> dict[str, Any]:
    import yaml

    if not path:
        raise _BackendUnavailable(f"{label}_config_missing")
    p = Path(path)
    if not p.is_file():
        raise _BackendUnavailable(f"{label}_config_not_found:{p.as_posix()}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise _BackendUnavailable(f"{label}_config_not_mapping:{p.as_posix()}")
    return raw


class _SileroBackend:
    def __init__(self) -> None:
        self._sessions: dict[str, Any] = {}

    def __call__(self, audio_path: str, config_path: str | None) -> dict[str, Any]:
        if not config_path:
            raise _BackendUnavailable("vad_config_missing")
        from audio_engine.core.quality.silero_vad import SileroVadSession, load_silero_vad_config

        if not Path(config_path).is_file():
            raise _BackendUnavailable(f"vad_config_not_found:{config_path}")
        session = self._sessions.get(config_path)
        if session is None:
            session = SileroVadSession(load_silero_vad_config(config_path))
            self._sessions[config_path] = session
        out = session.analyze(audio_path)
        return dict(out) if isinstance(out, dict) else {}


def _energy_backend(audio_path: str, config_path: str | None) -> dict[str, Any]:
    from audio_engine.core.evidence.energy import classify_energy_state, read_energy_from_path

    cfg = _read_yaml_mapping(config_path, label="energy")
    metrics = read_energy_from_path(
        audio_path,
        frame_ms=int(cfg.get("frame_ms") or 20),
        silence_frame_dbfs=float(cfg.get("silence_frame_dbfs", -45.0)),
    )
    state = classify_energy_state(
        duration_ms=metrics.get("duration_ms"),
        rms_dbfs=metrics.get("rms_dbfs"),
        peak_dbfs=metrics.get("peak_dbfs"),
        non_silent_ratio=metrics.get("non_silent_ratio"),
        min_duration_ms=float(cfg.get("min_duration_ms", 300.0)),
        min_rms_dbfs=float(cfg.get("min_rms_dbfs", -50.0)),
        min_peak_dbfs=float(cfg.get("min_peak_dbfs", -40.0)),
        min_non_silent_ratio=float(cfg.get("min_non_silent_ratio", 0.02)),
        borderline_margin_db=float(cfg.get("borderline_margin_db", 3.0)),
    )
    out: dict[str, Any] = {
        "energy_state": state,
        "strict_all_channels_zero": metrics.get("strict_all_channels_zero"),
        "duration_ms": metrics.get("duration_ms"),
        "rms_dbfs": metrics.get("rms_dbfs"),
        "peak_dbfs": metrics.get("peak_dbfs"),
        "non_silent_ratio": metrics.get("non_silent_ratio"),
    }
    if state == ENERGY_STATE_FAILED:
        out["energy_status"] = "failed"
        out["energy_error"] = "audio_energy_metrics_missing"
    else:
        out["energy_status"] = "success"
    return out


class _DnsmosBackend:
    def __init__(self) -> None:
        self._sessions: dict[str, Any] = {}

    def __call__(self, audio_path: str, config_path: str | None) -> dict[str, Any]:
        import soundfile as sf

        from audio_engine.core.quality.dnsmos_p835 import (
            DnsmosP835Session,
            DnsmosRuntimeOptions,
            derive_noise_risk,
            scores_to_quality_dict,
        )

        cfg = _read_yaml_mapping(config_path, label="dnsmos")
        model_path = str(cfg.get("model_path") or "")
        if not model_path or not Path(model_path).is_file():
            raise _BackendUnavailable(f"dnsmos_model_not_found:{model_path or '<unset>'}")
        runtime_options = DnsmosRuntimeOptions.from_config(cfg.get("runtime"))
        key = json.dumps({"config": str(config_path), "model": model_path,
                          "providers": cfg.get("providers"), "runtime": runtime_options.__dict__},
                         sort_keys=True)
        session = self._sessions.get(key)
        if session is None:
            session = DnsmosP835Session(
                Path(model_path), providers=list(cfg.get("providers") or []) or None,
                runtime_options=runtime_options,
            )
            self._sessions[key] = session
        data, sr = sf.read(str(audio_path), always_2d=False)
        scores = session.score_array(data, int(sr))
        thresholds = cfg.get("thresholds") or {}
        risk = derive_noise_risk(
            bak=scores.bak,
            ovrl=scores.ovrl,
            t_bak=float(thresholds.get("t_bak", 3.0)),
            t_ovrl=float(thresholds.get("t_ovrl", 2.8)),
            calibrated=bool(cfg.get("calibrated", False)),
            status=scores.status,
        )
        return scores_to_quality_dict(scores, risk)


@dataclass
class RealAcousticScorer:
    """Execute neutral acoustic backends for requested deps (injectable)."""

    name: str = "real_acoustic_scorer"
    config_paths: dict[str, str] = field(default_factory=dict)
    backends: dict[str, Backend] = field(default_factory=dict)
    store: SharedAcousticEvidenceStore = field(default_factory=SharedAcousticEvidenceStore)
    ledger: ScoreLedger = field(default_factory=ScoreLedger)

    def __post_init__(self) -> None:
        defaults: dict[str, Backend] = {
            "vad": _SileroBackend(),
            "energy": _energy_backend,
            "dnsmos": _DnsmosBackend(),
        }
        for dep, backend in defaults.items():
            self.backends.setdefault(dep, backend)

    # -- planning ---------------------------------------------------------------
    def plan(self, sample_id: str, deps: set[str] | list[str] | tuple[str, ...]) -> None:
        for dep in sorted(set(deps)):
            if dep in DEP_STATUS_KEYS:
                self.ledger.planned[dep] += 1

    # -- execution --------------------------------------------------------------
    def score(
        self,
        dep: str,
        *,
        sample_id: str,
        audio_path: str,
        config_path: str | None = None,
    ) -> AcousticFact:
        if dep not in DEP_STATUS_KEYS:
            raise KeyError(f"unknown acoustic dep: {dep!r}")
        cached = self.store.get(sample_id, dep)
        if cached is not None and cached.is_success():
            self.ledger.cache_hits[dep] += 1
            return AcousticFact(
                dep=dep,
                status=cached.status,
                values=dict(cached.values),
                source="cached",
            )
        cfg_path = config_path or self.config_paths.get(dep)
        backend = self.backends.get(dep)
        if backend is None:
            return self._failure(dep, sample_id, "backend_missing", unavailable=True)
        try:
            self.ledger.actual[dep] += 1
            fields = backend(str(audio_path), cfg_path)
        except _BackendUnavailable as exc:
            # The backend could not start: it was not a real model call.
            self.ledger.actual[dep] -= 1
            return self._failure(dep, sample_id, str(exc), unavailable=True)
        except Exception as exc:  # noqa: BLE001 - isolate per-sample failures
            return self._failure(dep, sample_id, f"{type(exc).__name__}: {exc}", executed=True)
        if not result_is_computed(dep, fields):
            self.ledger.actual[dep] -= 1
            self.ledger.rejected[dep] += 1
            return self._failure(dep, sample_id, "empty_or_uncomputed_result", rejected=True)
        fact = fact_from_result(dep, fields)
        if fact.status == FACT_FAILED:
            self.ledger.failed[dep] += 1
            self.ledger.errors.append(
                {"sample_id": str(sample_id), "dep": dep, "error": fact.error or "backend_reported_failed"}
            )
        return self.store.put(sample_id, fact)

    def compute(
        self,
        sample_id: str,
        deps: set[str] | list[str] | tuple[str, ...],
        *,
        audio_path_for: Callable[[str], str],
        config_paths: Mapping[str, str] | None = None,
    ) -> dict[str, AcousticFact]:
        """Plan + run every dep in ``deps``; return one fact per dep."""
        self.plan(sample_id, deps)
        out: dict[str, AcousticFact] = {}
        for dep in ACOUSTIC_DEPS:
            if dep not in set(deps):
                continue
            cfg_path = (config_paths or {}).get(dep) or self.config_paths.get(dep)
            out[dep] = self.score(
                dep,
                sample_id=sample_id,
                audio_path=audio_path_for(dep),
                config_path=cfg_path,
            )
        return out

    def _failure(
        self,
        dep: str,
        sample_id: str,
        error: str,
        *,
        unavailable: bool = False,
        executed: bool = False,
        rejected: bool = False,
    ) -> AcousticFact:
        if unavailable:
            self.ledger.unavailable[dep] += 1
        if executed:
            self.ledger.failed[dep] += 1
        self.ledger.errors.append({"sample_id": str(sample_id), "dep": dep, "error": error})
        # Failed (not success) -> merge keeps any earlier SUCCESS untouched.
        return self.store.put(
            sample_id,
            AcousticFact(dep=dep, status=FACT_FAILED, values={}, error=error, source="computed"),
        )
