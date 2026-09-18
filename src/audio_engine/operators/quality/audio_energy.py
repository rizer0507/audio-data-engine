"""quality.audio_energy — lightweight duration / RMS / peak / non-silent evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from audio_engine.core.operator import BaseOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3.audio_energy import compute_audio_energy_for_sample
from audio_engine.core.selection_v3.config import SelectionV3Config


def _load_energy_config(params: dict[str, Any]) -> dict[str, Any]:
    loaded: dict[str, Any] = {}
    cfg_path = params.get("config_path") or params.get("config")
    if cfg_path:
        raw = yaml.safe_load(Path(str(cfg_path)).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"audio_energy config must be a mapping: {cfg_path}")
        loaded = dict(raw)
    merged = {
        **loaded,
        **{k: v for k, v in params.items() if k not in {"config", "config_path"}},
    }
    return merged


def _selection_config_for_energy(params: dict[str, Any]) -> SelectionV3Config:
    energy = _load_energy_config(params)
    selection_path = params.get("selection_config")
    base: dict[str, Any] = {}
    if selection_path:
        raw = yaml.safe_load(Path(str(selection_path)).read_text(encoding="utf-8")) or {}
        if isinstance(raw, dict):
            base = dict(raw)
    # Minimal family stub so SelectionV3Config validates when only energy is needed.
    if "model_families" not in base:
        base["model_families"] = {
            "glm": ["glm_1", "glm_2"],
            "sensevoice": ["sensevoice_1", "sensevoice_2"],
            "qwen": ["qwen_1", "qwen_2"],
        }
        base.setdefault("teacher_families", ["glm", "sensevoice"])
        base.setdefault("target_family", "qwen")
        base.setdefault("expected_runs_per_family", 2)
        base.setdefault("run_aliases", {
            "glm_1": "glm",
            "glm_2": "glm",
            "sensevoice_1": "sensevoice",
            "sensevoice_2": "sensevoice",
            "qwen_1": "qwen",
            "qwen_2": "qwen",
        })
    base["audio_energy"] = energy
    base.setdefault("rule_version", "selection_five_class_v2_auto_noise")
    # Avoid echo-table hard fail when operator runs without full selection YAML.
    ct = dict(base.get("classify_text") or {})
    ct.setdefault("policy", "legacy")
    ct.setdefault("echo_missing", "echo_list_missing")
    base["classify_text"] = ct
    return SelectionV3Config.from_params(base)


@register_operator
class AudioEnergyOperator(BaseOperator):
    """Write duration_ms / rms_dbfs / peak_dbfs / non_silent_ratio / energy_state."""

    name = "audio_energy"
    version = "1.0.0"
    category = "quality"

    def compute_cache_key(self, sample: Sample, config: OperatorConfig) -> str:
        params = dict(config.params)
        energy_cfg = _load_energy_config(params)
        payload = {
            "audio": sample.sha256,
            "energy": energy_cfg,
            "operator": self.full_name,
            "version": self.version,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()

    def _execute(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        params = dict(config.params)
        energy_cfg = _load_energy_config(params)
        sel = _selection_config_for_energy(params)
        input_key = str(energy_cfg.get("input_audio_key") or "resampled_16k")
        audio_path = None
        try:
            audio_path = sample.audio_path(input_key)
        except Exception:  # noqa: BLE001
            audio_path = (sample.audio or {}).get(input_key) or sample.source_path

        evidence = compute_audio_energy_for_sample(
            audio_path=audio_path,
            config=sel,
            duration_sec=float(sample.duration) if sample.duration is not None else None,
            quality=sample.quality if isinstance(sample.quality, dict) else {},
        )
        quality = evidence.as_dict()
        # Drop None-only noise; keep explicit failed state.
        return {
            "quality": quality,
            "lineage_entry": {
                "operator": self.full_name,
                "version": self.version,
                "params": {
                    "policy_version": evidence.energy_policy_version,
                    "input_audio_key": input_key,
                },
                "input_key": input_key,
            },
        }
