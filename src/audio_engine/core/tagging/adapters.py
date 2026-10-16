"""Injectable acoustic compute adapters (real ops or mocks for scheduling tests)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from audio_engine.core.sample import Sample
from audio_engine.core.tagging.evidence.acoustic import AcousticEvidenceView, acoustic_from_quality


class AcousticAdapter(Protocol):
    name: str

    def compute(self, sample: Sample, *, deps: set[str], params: dict[str, Any]) -> AcousticEvidenceView:
        ...


@dataclass
class CountingAcousticAdapter:
    """Test/double adapter that records calls and returns configurable results."""

    name: str = "counting"
    calls: list[dict[str, Any]] = field(default_factory=list)
    results: dict[str, AcousticEvidenceView] = field(default_factory=dict)
    default: AcousticEvidenceView | None = None
    fail_deps: set[str] = field(default_factory=set)

    def compute(self, sample: Sample, *, deps: set[str], params: dict[str, Any]) -> AcousticEvidenceView:
        self.calls.append({"sample_id": sample.id, "deps": sorted(deps), "params": dict(params)})
        base = self.results.get(sample.id) or self.default or AcousticEvidenceView()
        # Merge onto existing quality view.
        view = AcousticEvidenceView(
            vad_status=base.vad_status,
            vad_speech_state=base.vad_speech_state,
            energy_status=base.energy_status,
            strict_all_channels_zero=base.strict_all_channels_zero,
            energy_state=base.energy_state,
            dnsmos_status=base.dnsmos_status,
            dnsmos_sig=base.dnsmos_sig,
            dnsmos_bak=base.dnsmos_bak,
            dnsmos_ovrl=base.dnsmos_ovrl,
            details=dict(base.details),
        )
        if "vad" in deps:
            if "vad" in self.fail_deps:
                view.vad_status = "failed"
                view.vad_speech_state = None
            else:
                view.vad_status = view.vad_status or "success"
                view.vad_speech_state = view.vad_speech_state or "present"
        if "energy" in deps:
            if "energy" in self.fail_deps:
                view.energy_status = "failed"
                view.strict_all_channels_zero = None
            else:
                view.energy_status = view.energy_status or "success"
                if view.strict_all_channels_zero is None:
                    view.strict_all_channels_zero = False
        if "dnsmos" in deps:
            if "dnsmos" in self.fail_deps:
                view.dnsmos_status = "failed"
                view.dnsmos_sig = None
            else:
                view.dnsmos_status = view.dnsmos_status or "success"
                if view.dnsmos_sig is None:
                    view.dnsmos_sig = 3.5
        return view

    def call_count(self, dep: str | None = None) -> int:
        if dep is None:
            return len(self.calls)
        return sum(1 for c in self.calls if dep in c["deps"])


@dataclass
class QualityPassthroughAdapter:
    """Use already-attached sample.quality; optionally invoke real compute hooks."""

    name: str = "quality_passthrough"
    compute_hooks: dict[str, Any] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def compute(self, sample: Sample, *, deps: set[str], params: dict[str, Any]) -> AcousticEvidenceView:
        self.calls.append({"sample_id": sample.id, "deps": sorted(deps)})
        quality = dict(sample.quality or {})
        for dep in sorted(deps):
            hook = self.compute_hooks.get(dep)
            if hook is None:
                continue
            fields = hook(sample, params)
            if isinstance(fields, dict):
                quality.update(fields)
        sample.quality = quality
        return acoustic_from_quality(quality)


def apply_view_to_sample(sample: Sample, view: AcousticEvidenceView) -> None:
    quality = dict(sample.quality or {})
    if view.vad_status is not None:
        quality["vad_status"] = view.vad_status
    if view.vad_speech_state is not None:
        quality["vad_speech_state"] = view.vad_speech_state
    if view.energy_status is not None:
        quality["energy_status"] = view.energy_status
    if view.strict_all_channels_zero is not None:
        quality["strict_all_channels_zero"] = view.strict_all_channels_zero
    if view.energy_state is not None:
        quality["energy_state"] = view.energy_state
    if view.dnsmos_status is not None:
        quality["dnsmos_status"] = view.dnsmos_status
    if view.dnsmos_sig is not None:
        quality["dnsmos_sig"] = view.dnsmos_sig
    if view.dnsmos_bak is not None:
        quality["dnsmos_bak"] = view.dnsmos_bak
    if view.dnsmos_ovrl is not None:
        quality["dnsmos_ovrl"] = view.dnsmos_ovrl
    sample.quality = quality


@dataclass
class RealAcousticAdapter:
    """Production adapter backed by the neutral ``core.evidence`` scorer (044).

    - executes silero VAD / lightweight energy / DNSMOS only for the requested deps;
    - an empty passthrough or empty backend result is NOT a computed fact;
    - missing models/configs become ``failed`` evidence with a locatable error;
    - existing SUCCESS facts already on ``sample.quality`` are reused, never recomputed
      and never overwritten by ``not_required`` / ``pending`` schedule labels.
    Inject ``scorer`` (or ``backends`` on a ``RealAcousticScorer``) in tests to verify
    scheduling without real ONNX files. ``scorer_hooks`` are kept as a shorthand for
    ``dep -> callable(sample, params) -> quality fields``.
    """

    name: str = "real_acoustic"
    scorer: Any = None
    scorer_hooks: dict[str, Any] = field(default_factory=dict)
    model_calls: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.scorer is None:
            from audio_engine.core.evidence.scoring import RealAcousticScorer

            self.scorer = RealAcousticScorer()

    @property
    def ledger(self) -> Any:
        return self.scorer.ledger

    def call_count(self, dep: str | None = None) -> int:
        if dep is None:
            return len(self.model_calls)
        return sum(1 for c in self.model_calls if c.get("dep") == dep)

    @staticmethod
    def _config_paths(params: dict[str, Any]) -> dict[str, str]:
        out: dict[str, str] = {}
        for dep, keys in (
            ("vad", ("silero_vad_config", "vad_config")),
            ("energy", ("audio_energy_config", "energy_config")),
            ("dnsmos", ("dnsmos_config",)),
        ):
            for key in keys:
                if params.get(key):
                    out[dep] = str(params[key])
                    break
        return out

    def _audio_path(self, sample: Sample, input_key: str) -> str:
        try:
            return str(sample.audio_path(input_key))
        except Exception:  # noqa: BLE001
            return str((sample.audio or {}).get(input_key) or sample.source_path or "")

    @staticmethod
    def _input_key(config_path: str | None) -> str:
        if not config_path:
            return "resampled_16k"
        try:
            from pathlib import Path

            import yaml

            raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
            return str(raw.get("input_audio_key") or "resampled_16k") if isinstance(raw, dict) else "resampled_16k"
        except Exception:  # noqa: BLE001
            return "resampled_16k"

    def compute(self, sample: Sample, *, deps: set[str], params: dict[str, Any]) -> AcousticEvidenceView:
        from audio_engine.core.evidence.store import merge_quality_fields

        scorer = self.scorer
        cfg_paths = self._config_paths(params)
        quality = dict(sample.quality or {})
        # Seed the shared store with already-computed facts so they are reused.
        scorer.store.put_from_quality(sample.id, quality)
        details: dict[str, Any] = {"adapter": self.name, "deps": sorted(deps)}

        hook_deps = {d for d in deps if d in self.scorer_hooks}
        for dep in sorted(hook_deps):
            self.model_calls.append({"sample_id": sample.id, "dep": dep, "via": "hook"})
            fields = self.scorer_hooks[dep](sample, params)
            if isinstance(fields, dict):
                quality = merge_quality_fields(quality, fields)
                scorer.store.put_from_quality(sample.id, fields)

        real_deps = {d for d in deps if d not in hook_deps}
        if real_deps:
            before = dict(scorer.ledger.actual)
            facts = scorer.compute(
                sample.id,
                real_deps,
                audio_path_for=lambda dep: self._audio_path(sample, self._input_key(cfg_paths.get(dep))),
                config_paths=cfg_paths,
            )
            for dep, fact in sorted(facts.items()):
                if scorer.ledger.actual.get(dep, 0) > before.get(dep, 0):
                    self.model_calls.append({"sample_id": sample.id, "dep": dep, "via": "evidence_scorer"})
                quality = merge_quality_fields(quality, fact.to_quality_fields())
                if fact.error:
                    details[f"{dep}_error"] = fact.error
        sample.quality = quality
        view = acoustic_from_quality(quality)
        view.details = {**dict(view.details or {}), **details}
        return view
