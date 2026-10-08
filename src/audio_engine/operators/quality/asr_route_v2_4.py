"""037 ASR pre-route for on-demand acoustic five-class (v2.4).

Writes queue_id / early_category without running VAD/energy/DNSMOS.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from audio_engine.core.operator import ManifestOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3.config import SelectionV3Config
from audio_engine.core.selection_v3.family_evidence import collect_route_views
from audio_engine.core.selection_v3.five_class_v2 import build_family_states
from audio_engine.core.selection_v3.five_class_v2_4_on_demand_acoustic import (
    route_asr_queue,
)


def _load_voicemail_pattern(config: SelectionV3Config) -> re.Pattern[str] | None:
    path = getattr(config, "voicemail_patterns_path", None)
    if not path:
        return None
    raw = yaml.safe_load(Path(str(path)).read_text(encoding="utf-8")) or {}
    patterns = [f"(?:{p})" for p in (raw.get("patterns") or []) if str(p).strip()]
    if not patterns:
        return None
    return re.compile("|".join(patterns), re.IGNORECASE)


@register_operator
class AsrRouteV24Operator(ManifestOperator):
    """ASR-only pre-route → labels.asr_route_v2_4."""

    name = "asr_route_v2_4"
    version = "1.0.0"
    category = "quality"

    def run(self, samples: list[Sample], config: OperatorConfig) -> list[Sample]:
        params = dict(config.params)
        selection_path = params.get("config_path") or params.get("selection_config")
        if not selection_path:
            raise ValueError("quality.asr_route_v2_4 requires selection_config")
        selection = SelectionV3Config.from_yaml(selection_path)
        vm = _load_voicemail_pattern(selection)

        updated = [sample.model_copy(deep=True) for sample in samples]
        counts: dict[str, int] = {}
        for sample in updated:
            routes = collect_route_views(sample, selection)
            bundles = build_family_states(routes, selection)
            decision = route_asr_queue(
                routes, bundles, selection, voicemail_pattern=vm
            )
            labels = dict(sample.labels or {})
            labels["asr_route_v2_4"] = decision.as_dict()
            sample.labels = labels
            key = decision.early_category or f"queue_{decision.queue_id}"
            counts[key] = counts.get(key, 0) + 1

        if getattr(config, "run_dir", None):
            try:
                context = getattr(config, "context", None)
                if context is not None and hasattr(context, "log"):
                    context.log("asr_route_v2_4_done", counts=counts, n=len(updated))
            except Exception:  # noqa: BLE001
                pass
        return updated
