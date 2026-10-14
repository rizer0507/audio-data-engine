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
