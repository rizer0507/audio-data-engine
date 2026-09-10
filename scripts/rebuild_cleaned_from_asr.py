"""Rebuild cleaned_<batch>.parquet from one dual-run ASR parquet (014 存量底表缺口).

Strips transcripts; keeps id / sha256 / source_path / audio / quality / duration.
Does not re-decode audio. Paths inside audio may still point at remote hosts
(e.g. /data2/...); DNSMOS/prepare consumers must run where those paths are readable.

Usage:
  python scripts/rebuild_cleaned_from_asr.py \\
    --asr datasets/stage1/asr/qwen3-asr-1_asr_0908-30000.parquet \\
    --output datasets/stage1/cleaned/cleaned_0908-30000.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audio_engine.core.manifest import Manifest
from audio_engine.core.sample import Sample


def _parse_maybe_json(value):
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{") or text.startswith("["):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return value
    return value


def process_sample(sample: Sample) -> Sample:
    out = sample.model_copy(deep=True)
    out.transcripts = {}
    out.audio = _parse_maybe_json(out.audio) or {}
    if not isinstance(out.audio, dict):
        raise ValueError(f"sample {out.id}: audio must be object, got {type(out.audio)}")
    out.quality = _parse_maybe_json(out.quality) or out.quality
    out.labels = _parse_maybe_json(out.labels) or out.labels
    # Drop ASR-specific labels that must not look like gold / selection state
    for key in list(out.labels.keys()):
        if key.startswith(("type", "decision", "gold", "pseudo", "noise_risk", "noise_band")):
            out.labels.pop(key, None)
    if out.sha256 and not out.labels.get("original_audio_sha256"):
        out.labels["original_audio_sha256"] = str(out.sha256)
    out.lineage = list(out.lineage or [])
    out.lineage.append(
        {
            "op": "rebuild_cleaned_from_asr",
            "note": "derived from ASR parquet; not a fresh cleaning pass",
        }
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asr", required=True, type=Path, help="One ASR parquet for the batch")
    parser.add_argument("--output", required=True, type=Path, help="cleaned_<batch>.parquet")
    args = parser.parse_args()

    samples = [process_sample(s) for s in Manifest.load(args.asr)]
    if not samples:
        raise SystemExit("ASR manifest is empty")
    missing_hash = sum(1 for s in samples if not (s.sha256 or s.labels.get("original_audio_sha256")))
    if missing_hash:
        raise SystemExit(f"{missing_hash} samples lack sha256 / original_audio_sha256")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Manifest(samples).save(args.output)
    print(f"wrote {len(samples)} samples → {args.output}")


if __name__ == "__main__":
    main()
