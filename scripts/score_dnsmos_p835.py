#!/usr/bin/env python3
"""Standalone DNSMOS P.835 quality scorer (no pipeline / Manifest).

Colleague handoff: this file + ``score_dnsmos_p835.yaml`` + ``sig_bak_ovr.onnx``.

Same scoring math as ``src/audio_engine/core/quality/dnsmos_p835.py``
(preprocess ``p835_official_nonpersonalized_v2``, per-window calibration, mean).

Deps: ``pip install onnxruntime numpy soundfile pyyaml``
Non-16 kHz input also needs ``librosa`` (optional resample).

Example::

  python scripts/score_dnsmos_p835.py \\
    --config scripts/score_dnsmos_p835.yaml \\
    /path/to/audio.wav
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import yaml

DNSMOS_SAMPLE_RATE = 16000
DNSMOS_INPUT_LENGTH_SEC = 9.01

# Microsoft DNSMOS non-personalized P.835 calibration (per window).
_SIG_POLY = [-0.08397278, 1.22083953, 0.0052439]
_BAK_POLY = [-0.13166888, 1.60915514, -0.39604546]
_OVRL_POLY = [-0.06766283, 1.11546468, 0.04602535]


def load_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return dict(raw)


def _prepare_mono(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    mono = np.asarray(audio, dtype=np.float64)
    if mono.ndim > 1:
        mono = mono.mean(axis=1)
    if sample_rate <= 0 or not np.all(np.isfinite(mono)) or len(mono) == 0:
        raise ValueError("invalid or empty audio buffer")
    if sample_rate != DNSMOS_SAMPLE_RATE:
        import librosa

        mono = librosa.resample(mono, orig_sr=sample_rate, target_sr=DNSMOS_SAMPLE_RATE)
    mono = mono.astype(np.float32)
    desired = int(DNSMOS_INPUT_LENGTH_SEC * DNSMOS_SAMPLE_RATE)
    if len(mono) < desired:
        while len(mono) < desired:
            mono = np.concatenate((mono, mono))
    return mono


class DnsmosP835Session:
    def __init__(
        self,
        model_path: Path,
        *,
        providers: list[str] | None = None,
        intra_op_num_threads: int = 2,
        inter_op_num_threads: int = 1,
        allow_spinning: bool = False,
        window_batch_size: int = 4,
    ) -> None:
        import onnxruntime as ort

        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"DNSMOS model file missing: {self.model_path}")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.intra_op_num_threads = int(intra_op_num_threads)
        opts.inter_op_num_threads = int(inter_op_num_threads)
        spinning = "1" if allow_spinning else "0"
        opts.add_session_config_entry("session.intra_op.allow_spinning", spinning)
        opts.add_session_config_entry("session.inter_op.allow_spinning", spinning)
        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=opts,
            providers=list(providers) if providers else ["CPUExecutionProvider"],
        )
        model_input = self._session.get_inputs()[0]
        self.input_name = model_input.name
        shape = getattr(model_input, "shape", [1])
        requested = max(1, int(window_batch_size))
        self.window_batch_size = requested if not isinstance(shape[0], int) else 1

    def score_array(self, audio: np.ndarray, sample_rate: int) -> dict[str, float | None | str]:
        try:
            mono = _prepare_mono(audio, sample_rate)
            desired = int(DNSMOS_INPUT_LENGTH_SEC * DNSMOS_SAMPLE_RATE)
            n_hops = int(np.floor(len(mono) / DNSMOS_SAMPLE_RATE) - DNSMOS_INPUT_LENGTH_SEC) + 1
            chunks = [
                mono[i * DNSMOS_SAMPLE_RATE : i * DNSMOS_SAMPLE_RATE + desired]
                for i in range(max(1, n_hops))
            ]

            sigs: list[float] = []
            baks: list[float] = []
            ovrls: list[float] = []
            for left in range(0, len(chunks), self.window_batch_size):
                batch = chunks[left : left + self.window_batch_size]
                inp = np.zeros((len(batch), desired), dtype=np.float32)
                for index, chunk in enumerate(batch):
                    inp[index, : len(chunk)] = chunk
                out = self._session.run(None, {self.input_name: inp})
                arr = np.asarray(out[0]).reshape(len(batch), -1)
                if arr.shape[1] < 3:
                    raise RuntimeError(f"unexpected DNSMOS output shape: {arr.shape}")
                if not np.all(np.isfinite(arr[:, :3])):
                    raise RuntimeError("non-finite DNSMOS output")
                sigs.extend(float(np.polyval(_SIG_POLY, row[0])) for row in arr)
                baks.extend(float(np.polyval(_BAK_POLY, row[1])) for row in arr)
                ovrls.extend(float(np.polyval(_OVRL_POLY, row[2])) for row in arr)
            return {
                "status": "success",
                "sig": float(np.mean(sigs)),
                "bak": float(np.mean(baks)),
                "ovrl": float(np.mean(ovrls)),
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "failed",
                "sig": None,
                "bak": None,
                "ovrl": None,
                "error": str(exc),
            }


def score_file(session: DnsmosP835Session, audio_path: Path) -> dict:
    data, sr = sf.read(str(audio_path), always_2d=False)
    result = session.score_array(data, int(sr))
    result["audio"] = str(audio_path)
    return result


def session_from_config(cfg: dict[str, Any], model_override: Path | None) -> DnsmosP835Session:
    model_path = model_override or Path(str(cfg.get("model_path") or ""))
    if not str(model_path):
        raise ValueError("model_path missing: set it in config or pass --model")
    runtime = cfg.get("runtime") or {}
    if not isinstance(runtime, dict):
        raise ValueError("runtime must be a mapping")
    providers = cfg.get("providers") or ["CPUExecutionProvider"]
    if isinstance(providers, str):
        providers = [providers]
    return DnsmosP835Session(
        model_path,
        providers=list(providers),
        intra_op_num_threads=int(runtime.get("intra_op_num_threads", 2)),
        inter_op_num_threads=int(runtime.get("inter_op_num_threads", 1)),
        allow_spinning=bool(runtime.get("allow_spinning", False)),
        window_batch_size=int(runtime.get("window_batch_size", 4)),
    )


def main(argv: list[str] | None = None) -> int:
    default_cfg = Path(__file__).with_name("score_dnsmos_p835.yaml")
    parser = argparse.ArgumentParser(description="Standalone DNSMOS P.835 scorer")
    parser.add_argument("audio", type=Path, help="WAV/audio file to score")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_cfg,
        help=f"config yaml (default: {default_cfg.name})",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="override config model_path",
    )
    args = parser.parse_args(argv)

    if not args.audio.is_file():
        print(f"audio not found: {args.audio}", file=sys.stderr)
        return 2
    if not args.config.is_file():
        print(f"config not found: {args.config}", file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    session = session_from_config(cfg, args.model)
    result = score_file(session, args.audio)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
