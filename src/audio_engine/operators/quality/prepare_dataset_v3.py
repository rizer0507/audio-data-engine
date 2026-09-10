"""Prepare dataset v3: contract check → leakage grouping → immutable reservation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from audio_engine.core.artifacts import atomic_write_json
from audio_engine.core.dataset_v3.grouping import (
    GroupingConfig,
    apply_grouping_to_samples,
    build_leakage_groups,
)
from audio_engine.core.dataset_v3.reservation import (
    ReservationConfig,
    apply_reservation_to_samples,
    build_reservation,
)
from audio_engine.core.operator import ManifestOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample
from audio_engine.core.selection_v3.config import SelectionV3Config
from audio_engine.core.selection_v3.input_contract import apply_contract_to_samples
from audio_engine.core.selection_v3.types import DATASET_POLICY_VERSION, RULE_VERSION


def _load_dataset_config(path: str | Path | None, params: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if path:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"dataset config must be a mapping: {path}")
        merged.update(raw)
    # Operator params override file (except nested blocks merged shallowly)
    for key, value in params.items():
        if key in {"config_path", "reservation_output", "report_output"}:
            continue
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged


@register_operator
class PrepareDatasetV3Operator(ManifestOperator):
    """Stage-A prepare: N≥3 family contract, leakage groups, immutable reservation.

    Params:
      config_path: configs/datasets/zh_asr_v3.yaml or zh_asr_v3_three_family.yaml
      reservation_output: optional explicit path for reservation.json
      report_output: optional path for conservation/grouping report
    """

    name = "prepare_dataset_v3"
    version = "1.0.0"
    category = "quality"

    def run(self, samples: list[Sample], config: OperatorConfig) -> list[Sample]:
        params = dict(config.params or {})
        cfg_path = params.get("config_path")
        merged = _load_dataset_config(cfg_path, params)

        selection_cfg = SelectionV3Config.from_params(merged)
        grouping_cfg = GroupingConfig.from_params(merged.get("grouping"))
        reservation_cfg = ReservationConfig.from_params(merged.get("reservation"))
        if not reservation_cfg.policy_version:
            reservation_cfg.policy_version = str(
                merged.get("dataset_policy_version") or DATASET_POLICY_VERSION
            )

        # 1) Input contract / conservation
        annotated, conservation, alignment = apply_contract_to_samples(
            samples, selection_cfg
        )

        # 2) Leakage grouping
        grouping = build_leakage_groups(annotated, grouping_cfg)
        grouped = apply_grouping_to_samples(annotated, grouping)

        # 3) Immutable reservation (before model-error inspection by design)
        reservation = build_reservation(
            grouped,
            grouping,
            reservation_cfg,
            grouping_config=grouping_cfg,
        )
        stamped = apply_reservation_to_samples(grouped, reservation)

        # Persist immutable artifacts
        run_dir = Path(config.run_dir) if config.run_dir else None
        reservation_path = params.get("reservation_output")
        if reservation_path:
            out = Path(str(reservation_path))
        elif run_dir is not None:
            out = run_dir / "artifacts" / "reservation.json"
        else:
            out = Path("data/derived") / "reservation.json"
        reservation.write_json(out)

        report = {
            "operator": self.full_name,
            "version": self.version,
            "rule_version": selection_cfg.rule_version or RULE_VERSION,
            "dataset_policy_version": reservation.policy_version,
            "config_path": str(cfg_path) if cfg_path else None,
            "reservation_path": str(out.resolve()),
            "reservation_digest": reservation.content_digest,
            "alignment": alignment.to_dict(),
            "conservation": conservation.to_dict(),
            "grouping": grouping.to_dict(),
            "reservation_counts": reservation.to_dict().get("counts"),
        }
        report_path = params.get("report_output")
        if report_path:
            rpath = Path(str(report_path))
        elif run_dir is not None:
            rpath = run_dir / "reports" / f"{config.step_name or self.name}_prepare.json"
        else:
            rpath = out.with_name("prepare_dataset_v3_report.json")
        atomic_write_json(rpath, report)

        # Stamp lineage
        for sample in stamped:
            sample.mark_completed(self.full_name)
            sample.add_lineage(
                self.full_name,
                self.version,
                {
                    "config_path": str(cfg_path) if cfg_path else None,
                    "reservation_digest": reservation.content_digest,
                    "reservation_path": str(out),
                    "rule_version": selection_cfg.rule_version,
                    "dataset_policy_version": reservation.policy_version,
                },
            )
            sample.labels["prepare_report_path"] = str(rpath)
            sample.labels["reservation_path"] = str(out)

        return stamped
