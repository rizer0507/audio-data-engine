"""Silero VAD speech evidence on already-cut wav (no re-segmentation).

Audio I/O uses soundfile (same as audio_energy / resample), not Silero's
``read_audio`` → torchaudio/torchcodec path, which breaks on some torch 2.9+
envs (libtorchcodec_*.so load failures).
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SILERO_ROOT = REPO_ROOT / "silero-vad-master" / "src"


@dataclass
class SileroVadConfig:
    policy_version: str = "silero_vad_v1"
    input_audio_key: str = "resampled_16k"
    sample_rate: int = 16000
    onnx: bool = True
    opset_version: int = 16
    model_path: str = ""
    min_speech_ratio_present: float = 0.10
    max_speech_ratio_absent: float = 0.02
    min_speech_duration_ms: int = 120
    get_speech_timestamps: dict[str, Any] | None = None

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> SileroVadConfig:
        data = dict(raw or {})
        gst = data.get("get_speech_timestamps")
        return cls(
            policy_version=str(data.get("policy_version") or "silero_vad_v1"),
            input_audio_key=str(data.get("input_audio_key") or "resampled_16k"),
            sample_rate=int(data.get("sample_rate") or 16000),
            onnx=bool(data.get("onnx", True)),
            opset_version=int(data.get("opset_version") or 16),
            model_path=str(data.get("model_path") or ""),
            min_speech_ratio_present=float(data.get("min_speech_ratio_present", 0.10)),
            max_speech_ratio_absent=float(data.get("max_speech_ratio_absent", 0.02)),
            min_speech_duration_ms=int(data.get("min_speech_duration_ms") or 120),
            get_speech_timestamps=dict(gst) if isinstance(gst, dict) else {},
        )


def load_silero_vad_config(path: str | Path | None) -> SileroVadConfig:
    if not path:
        return SileroVadConfig()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"silero_vad config must be a mapping: {path}")
    return SileroVadConfig.from_mapping(raw)


def _ensure_silero_on_path() -> Path:
    root = DEFAULT_SILERO_ROOT
    if root.is_dir():
        text = str(root)
        if text not in sys.path:
            sys.path.insert(0, text)
    return root


def _file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _classify_speech_state(
    *,
    speech_ratio: float,
    speech_duration_ms: int,
    cfg: SileroVadConfig,
) -> str:
    if (
        speech_ratio >= float(cfg.min_speech_ratio_present)
        and speech_duration_ms >= int(cfg.min_speech_duration_ms)
    ):
        return "present"
    if speech_ratio <= float(cfg.max_speech_ratio_absent):
        return "absent"
    return "uncertain"


def _resample_mono(mono: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    """Match operators/audio/resample.py semantics for 1-D signals."""
    if source_sr == target_sr:
        return mono
    if source_sr <= 0 or target_sr <= 0:
        raise ValueError(f"invalid sample rate: source={source_sr} target={target_sr}")
    from scipy import signal

    n_out = int(len(mono) * target_sr / source_sr)
    if n_out < 1:
        n_out = 1
    return signal.resample(mono, n_out)


def read_wav_for_silero(path: str | Path, sampling_rate: int = 16000):
    """Load wav via soundfile → mono float32 torch.Tensor at ``sampling_rate``.

    Mirrors Silero ``read_audio`` channel-mean + resample behavior without
    torchaudio/torchcodec I/O.
    """
    import torch

    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    # (frames, channels) → mono mean, same as silero utils_vad.read_audio
    mono = np.asarray(data, dtype=np.float32).mean(axis=1)
    if int(sr) != int(sampling_rate):
        mono = np.asarray(
            _resample_mono(mono, int(sr), int(sampling_rate)),
            dtype=np.float32,
        )
    else:
        mono = np.ascontiguousarray(mono, dtype=np.float32)
    return torch.from_numpy(mono)


class SileroVadSession:
    """Lazy Silero VAD session. Construction does not load the model."""

    def __init__(self, cfg: SileroVadConfig) -> None:
        self.cfg = cfg
        self._model = None
        self._get_speech_timestamps = None
        self.model_digest = ""
        self.runtime = "onnx" if cfg.onnx else "jit"

    def _load(self) -> None:
        if self._model is not None:
            return
        _ensure_silero_on_path()
        # Do not import silero_vad.read_audio — it pulls torchaudio/torchcodec I/O.
        from silero_vad import get_speech_timestamps, load_silero_vad

        model_path = str(self.cfg.model_path or "").strip()
        if model_path:
            path = Path(model_path).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"silero_vad model missing: {path}")
            # Explicit path: use OnnxWrapper / JIT via utils when possible.
            if path.suffix.lower() == ".onnx":
                from silero_vad.utils_vad import OnnxWrapper

                self._model = OnnxWrapper(str(path), force_onnx_cpu=True)
            else:
                from silero_vad.utils_vad import init_jit_model

                self._model = init_jit_model(str(path))
            self.model_digest = _file_digest(path)
        else:
            self._model = load_silero_vad(
                onnx=bool(self.cfg.onnx),
                opset_version=int(self.cfg.opset_version),
            )
            # Digests from vendored package file when available.
            data_dir = DEFAULT_SILERO_ROOT / "silero_vad" / "data"
            candidate = (
                data_dir / "silero_vad.onnx"
                if self.cfg.onnx
                else data_dir / "silero_vad.jit"
            )
            self.model_digest = _file_digest(candidate) if candidate.is_file() else "bundled"
        self._get_speech_timestamps = get_speech_timestamps

    def analyze(self, audio_path: str | Path) -> dict[str, Any]:
        """Return VAD quality fields. Never forges present/absent on failure."""
        path = Path(audio_path) if audio_path else None
        base = {
            "vad_status": "failed",
            "vad_speech_state": "failed",
            "speech_ratio": None,
            "speech_duration_ms": None,
            "longest_speech_ms": None,
            "speech_segment_count": None,
            "speech_prob_max": None,
            "speech_prob_mean": None,
            "vad_model_digest": self.model_digest,
            "vad_config_version": self.cfg.policy_version,
            "vad_runtime": self.runtime,
            "vad_error": None,
        }
        if path is None or not path.is_file():
            base["vad_error"] = f"audio_unreadable:{audio_path}"
            return base
        try:
            self._load()
            assert self._model is not None
            assert self._get_speech_timestamps is not None
            wav = read_wav_for_silero(path, sampling_rate=int(self.cfg.sample_rate))
            duration_sec = (
                float(wav.shape[0]) / float(self.cfg.sample_rate)
                if hasattr(wav, "shape")
                else 0.0
            )
            gst_kwargs = dict(self.cfg.get_speech_timestamps or {})
            gst_kwargs.setdefault("return_seconds", True)
            # Drop keys that collide with positional API.
            gst_kwargs.pop("sampling_rate", None)
            timestamps = self._get_speech_timestamps(
                wav,
                self._model,
                sampling_rate=int(self.cfg.sample_rate),
                **gst_kwargs,
            )
            segments: list[tuple[float, float]] = []
            for item in timestamps or []:
                start = float(item.get("start", 0.0))
                end = float(item.get("end", 0.0))
                if end > start:
                    segments.append((start, end))
            speech_duration_sec = sum(end - start for start, end in segments)
            longest_sec = max((end - start for start, end in segments), default=0.0)
            speech_duration_ms = int(round(speech_duration_sec * 1000.0))
            longest_speech_ms = int(round(longest_sec * 1000.0))
            speech_ratio = (
                float(speech_duration_sec / duration_sec) if duration_sec > 0 else 0.0
            )
            speech_ratio = max(0.0, min(1.0, speech_ratio))
            state = _classify_speech_state(
                speech_ratio=speech_ratio,
                speech_duration_ms=speech_duration_ms,
                cfg=self.cfg,
            )
            return {
                "vad_status": "success",
                "vad_speech_state": state,
                "speech_ratio": speech_ratio,
                "speech_duration_ms": speech_duration_ms,
                "longest_speech_ms": longest_speech_ms,
                "speech_segment_count": len(segments),
                "speech_prob_max": None,
                "speech_prob_mean": None,
                "vad_model_digest": self.model_digest,
                "vad_config_version": self.cfg.policy_version,
                "vad_runtime": self.runtime,
                "vad_error": None,
            }
        except Exception as exc:  # noqa: BLE001
            base["vad_model_digest"] = self.model_digest
            base["vad_error"] = str(exc)
            return base
