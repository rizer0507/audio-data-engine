#!/usr/bin/env python3
"""Prepare a dedicated FSMN VAD runtime dir with fenshen config.yaml overlay.

Production (server source weights):
  python scripts/prepare_fsmn_vad_runtime.py \\
    --source-model-dir /data2/data-cp/xijujun/models/fsmn-vad \\
    --fenshen-config configs/vad/fenshen_vad_config.yaml \\
    --runtime-dir var/models/fsmn-vad-fenshen-v1

Local example:
  python scripts/prepare_fsmn_vad_runtime.py \\
    --source-model-dir D:/Work/ph_ai_model_asr_rt_cpu/fsmn-vad \\
    --fenshen-config configs/vad/fenshen_vad_config.yaml \\
    --runtime-dir var/models/fsmn-vad-fenshen-v1

Then:
  export FSMN_VAD_SOURCE_MODEL_DIR=/data2/data-cp/xijujun/models/fsmn-vad
  export FSMN_VAD_RUNTIME_DIR=$PWD/var/models/fsmn-vad-fenshen-v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running without editable install when cwd is repo root.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from audio_engine.core.fsmn_vad.runtime import prepare_runtime_model_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model-dir", required=True)
    parser.add_argument("--fenshen-config", required=True)
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-fingerprint-check",
        action="store_true",
        help="Do not enforce documented asset SHA256 (not for production baseline).",
    )
    args = parser.parse_args()
    info = prepare_runtime_model_dir(
        source_model_dir=args.source_model_dir,
        fenshen_config_path=args.fenshen_config,
        runtime_dir=args.runtime_dir,
        overwrite=args.overwrite,
        verify_fingerprints=not args.skip_fingerprint_check,
    )
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
