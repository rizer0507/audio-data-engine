"""Manifest-level normalize summary: aggregate 8k / non-16k / multi-channel alerts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from loguru import logger

from audio_engine.core.artifacts import atomic_write_json
from audio_engine.core.operator import ManifestOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample


@register_operator
class NormalizeReportOperator(ManifestOperator):
    """Scan samples + run_dir/normalize_alerts.jsonl; print and persist batch summary."""

    name = "normalize_report"
    version = "1.0.0"
    category = "audio"

    def run(self, samples: list[Sample], config: OperatorConfig) -> list[Sample]:
        alerts: list[dict[str, Any]] = []
        if config.run_dir:
            path = Path(config.run_dir) / "normalize_alerts.jsonl"
            if path.is_file():
                for line in path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        alerts.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        # Also harvest from sample quality (covers workers that did not share run_dir).
        from_samples: list[dict[str, Any]] = []
        for s in samples:
            q = s.quality or {}
            src_sr = q.get("source_sample_rate")
            tgt_sr = q.get("target_sample_rate") or s.sample_rate
            if src_sr is None:
                continue
            if int(src_sr) == int(tgt_sr or 16000) and q.get("channel_policy") != "take_channel_0":
                continue
            from_samples.append(
                {
                    "sample_id": s.id,
                    "path": s.source_path,
                    "source_sample_rate": int(src_sr),
                    "target_sample_rate": int(tgt_sr or 16000),
                    "source_channels": q.get("source_channels"),
                    "selected_channel": q.get("selected_channel", 0),
                    "channel_policy": q.get("channel_policy"),
                    "sample_rate_source": q.get("sample_rate_source"),
                    "resampled": q.get("normalize_resampled") or q.get("pcm_to_wav") == "resampled",
                    "duration_source_sec": q.get("duration_source_sec"),
                    "duration_target_sec": q.get("duration_target_sec") or s.duration,
                }
            )

        # Prefer alerts file when present; else sample harvest.
        rows = alerts or from_samples
        rate_counts = Counter(int(r["source_sample_rate"]) for r in rows if r.get("source_sample_rate"))
        ch0_count = sum(1 for r in rows if r.get("channel_policy") == "take_channel_0")
        eight_k = [r for r in rows if int(r.get("source_sample_rate") or 0) == 8000]

        summary = {
            "total_samples": len(samples),
            "alert_rows": len(rows),
            "source_rate_histogram": dict(sorted(rate_counts.items())),
            "eight_k_count": len(eight_k),
            "multi_channel_to_ch0_count": ch0_count,
            "eight_k_files": [
                {
                    "sample_id": r.get("sample_id"),
                    "path": r.get("path"),
                    "source_sample_rate": r.get("source_sample_rate"),
                    "target_sample_rate": r.get("target_sample_rate"),
                    "duration_source_sec": r.get("duration_source_sec"),
                    "duration_target_sec": r.get("duration_target_sec"),
                }
                for r in eight_k
            ],
            "note": (
                "source_sample_rate is the real/decoded input rate; "
                "normalized output is always target_sample_rate (default 16000). "
                "Do not treat them as the same field."
            ),
        }

        banner = (
            f"[AUDIO-NORMALIZE SUMMARY] samples={summary['total_samples']} "
            f"alerts={summary['alert_rows']} eight_k={summary['eight_k_count']} "
            f"multi_ch_to_ch0={ch0_count} rates={summary['source_rate_histogram']}"
        )
        print(banner, flush=True)
        logger.warning(banner) if summary["eight_k_count"] or ch0_count else logger.info(banner)

        out_dir = Path(config.run_dir) if config.run_dir else Path(config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(out_dir / "normalize_summary.json", summary)

        for s in samples:
            s.mark_completed(self.full_name)
            s.add_lineage(
                operator=self.full_name,
                version=self.version,
                params={"summary_path": str(out_dir / "normalize_summary.json")},
            )
        return list(samples)
