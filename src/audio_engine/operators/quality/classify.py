from __future__ import annotations

import pandas as pd
import yaml
from pathlib import Path

from audio_engine.core.operator import ManifestOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample


@register_operator
class ClassifyOperator(ManifestOperator):
    """Apply ordered, versioned selection rules and retain an auditable reason code."""

    name = "classify"
    version = "1.0.0"
    category = "quality"

    def run(self, samples: list[Sample], config: OperatorConfig) -> list[Sample]:
        params = dict(config.params)
        if params.get("config_path"):
            loaded = yaml.safe_load(Path(params["config_path"]).read_text(encoding="utf-8")) or {}
            params = {**loaded, **params}
        policy_version = str(params.get("policy_version") or "")
        rules = params.get("rules") or []
        default_bucket = str(params.get("default_bucket", "review_queue"))
        gold_source_model = str(params.get("gold_source_model", ""))
        if not policy_version:
            raise ValueError("quality.classify requires policy_version")
        if not isinstance(rules, list) or not rules:
            raise ValueError("quality.classify requires non-empty ordered rules")

        updated = [sample.model_copy(deep=True) for sample in samples]
        frame = pd.DataFrame([sample.to_flat_dict() for sample in updated])
        for field, value in (params.get("defaults") or {}).items():
            if field not in frame:
                frame[field] = value
            else:
                frame[field] = frame[field].fillna(value)
        decisions: list[tuple[str, list[str]]] = [(default_bucket, ["default"]) for _ in updated]
        undecided = set(range(len(updated)))
        for rule in rules:
            expr = str(rule.get("expr") or "")
            bucket = str(rule.get("bucket") or "")
            reason = str(rule.get("reason") or "")
            if not expr or not bucket or not reason:
                raise ValueError("each classify rule requires expr, bucket and reason")
            try:
                matched = set(frame.query(expr, engine="python").index) & undecided
            except Exception as exc:
                raise ValueError(f"invalid classify expression {expr!r}: {exc}") from exc
            for position in matched:
                decisions[position] = (bucket, [reason])
            undecided -= matched

        for sample, (bucket, reasons) in zip(updated, decisions, strict=True):
            sample.labels.update(
                {
                    "classification_bucket": bucket,
                    "classification_reason_codes": reasons,
                    "selection_policy_version": policy_version,
                }
            )
            if bucket == "auto_gold":
                if not gold_source_model:
                    raise ValueError("auto_gold classification requires gold_source_model")
                gold_text = sample.get_transcript_text(gold_source_model).strip()
                if not gold_text:
                    raise ValueError(
                        f"auto_gold sample {sample.id} has empty Gold source transcript"
                    )
                sample.labels.update(
                    {
                        "annotation_state": "auto_accepted",
                        "gold_text": gold_text,
                        "gold_source": gold_source_model,
                        "annotation_revision": policy_version,
                    }
                )
            sample.mark_completed(self.full_name)
            sample.add_lineage(
                self.full_name,
                self.version,
                {"policy_version": policy_version, "bucket": bucket, "reason_codes": reasons},
            )
        return updated
