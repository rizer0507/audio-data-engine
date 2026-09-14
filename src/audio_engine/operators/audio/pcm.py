from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from audio_engine.core.artifacts import atomic_path, derived_audio_path
from audio_engine.core.manifest import probe_audio
from audio_engine.core.operator import BaseOperator, OperatorConfig
from audio_engine.core.registry import register_operator
from audio_engine.core.sample import Sample
from audio_engine.operators.audio.resample import DEFAULT_SAMPLE_RATE, resample_audio

# Already a container format — probe the header instead of assuming a rate.
_CONTAINER_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}
_PCM_EXTS = {".pcm", ".raw"}
_RIFF_MAGIC = b"RIFF"
_WAVE_MAGIC = b"WAVE"


def looks_like_wav(path: Path) -> bool:
    """True when the file has a RIFF/WAVE header, regardless of extension."""
    try:
        with path.open("rb") as f:
            header = f.read(12)
    except OSError:
        return False
    return len(header) >= 12 and header[:4] == _RIFF_MAGIC and header[8:12] == _WAVE_MAGIC


@register_operator
class PcmToWavOperator(BaseOperator):
    """Wrap PCM as WAV, probe real sample rate when a header exists, output at 16k.

    ``sample_rate`` is the *output* target (default 16 kHz). Headered files are
    probed and then resampled to that target. Headerless ``.pcm`` / ``.raw``
    use ``source_sample_rate`` (also default 16 kHz) because they have no header.
    """

    name = "pcm_to_wav"
    version = "1.2.0"
    category = "audio"

    def _execute(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        input_key = config.params.get("input_audio_key", "raw")
        output_key = config.params.get("output_audio_key", "pcm_to_wav")
        target_sr = int(config.params.get("sample_rate", DEFAULT_SAMPLE_RATE))
        channels = int(config.params.get("channels", 1))
        dtype = config.params.get("dtype", "int16")

        input_path = Path(sample.audio_path(input_key))
        suffix = input_path.suffix.lower()
        source_sr, probed, data = self._load_source(
            input_path, suffix, config, channels, dtype
        )

        if probed and source_sr == target_sr:
            return self._passthrough_result(
                sample, config, input_key, output_key, input_path, source_sr
            )

        if data is None:
            data, read_sr = sf.read(str(input_path), always_2d=False)
            source_sr = int(read_sr)
            if probed and source_sr == target_sr:
                return self._passthrough_result(
                    sample, config, input_key, output_key, input_path, source_sr
                )

        if source_sr == target_sr:
            to_write = data
            quality_status = "converted"
        else:
            to_write = resample_audio(np.asanyarray(data), source_sr, target_sr)
            quality_status = "resampled"

        output_path = derived_audio_path(config.output_dir, "pcm_to_wav", sample)
        with atomic_path(output_path) as tmp:
            sf.write(str(tmp), to_write, target_sr, subtype="PCM_16")

        n_frames = to_write.shape[0] if getattr(to_write, "ndim", 1) > 1 else len(to_write)
        duration = n_frames / target_sr if target_sr else sample.duration
        out_channels = int(to_write.shape[1]) if getattr(to_write, "ndim", 1) > 1 else 1
        return {
            "audio": {output_key: str(output_path.resolve())},
            "sample_rate": target_sr,
            "channels": out_channels,
            "duration": duration,
            "labels": {"pcm_converted": not probed},
            "quality": {
                "pcm_to_wav": quality_status,
                "source_sample_rate": source_sr,
                "target_sample_rate": target_sr,
                "probed": probed,
            },
            "lineage_entry": self._lineage(
                config, input_key, output_key, str(output_path.resolve())
            ),
        }

    def _load_source(
        self,
        input_path: Path,
        suffix: str,
        config: OperatorConfig,
        channels: int,
        dtype: str,
    ) -> tuple[int, bool, np.ndarray | None]:
        """Return (source_sr, probed_from_header, data_or_none).

        ``data`` is loaded for headerless PCM. Headered files return ``None`` so
        the caller can passthrough without decoding when already at target.
        """
        headered = suffix in _CONTAINER_EXTS or looks_like_wav(input_path)
        if headered:
            meta = probe_audio(input_path)
            if meta.get("valid") and meta.get("sample_rate"):
                return int(meta["sample_rate"]), True, None
            try:
                _data, sr = sf.read(str(input_path), always_2d=False)
                return int(sr), True, _data
            except Exception as exc:
                raise ValueError(
                    f"audio.pcm_to_wav: cannot probe sample rate for {input_path}"
                ) from exc

        if suffix not in _PCM_EXTS and not headered:
            raise ValueError(
                f"audio.pcm_to_wav: unsupported format '{suffix}' for {input_path}"
            )

        source_sr = self._headerless_source_rate(config)
        raw = np.fromfile(input_path, dtype=dtype)
        if channels > 1:
            raw = raw.reshape(-1, channels)
        return source_sr, False, raw

    @staticmethod
    def _headerless_source_rate(config: OperatorConfig) -> int:
        if "source_sample_rate" in config.params:
            return int(config.params["source_sample_rate"])
        return DEFAULT_SAMPLE_RATE

    def _passthrough_result(
        self,
        sample: Sample,
        config: OperatorConfig,
        input_key: str,
        output_key: str,
        input_path: Path,
        source_sr: int,
    ) -> dict[str, Any]:
        meta = probe_audio(input_path)
        updates: dict[str, Any] = {
            "audio": {output_key: str(input_path.resolve())},
            "sample_rate": source_sr,
            "labels": {"pcm_converted": False},
            "quality": {
                "pcm_to_wav": "passthrough",
                "source_sample_rate": source_sr,
                "target_sample_rate": source_sr,
                "probed": True,
            },
            "lineage_entry": self._lineage(
                config, input_key, output_key, str(input_path.resolve())
            ),
        }
        if meta.get("valid"):
            if meta.get("channels") is not None:
                updates["channels"] = meta.get("channels")
            if meta.get("duration") is not None:
                updates["duration"] = meta.get("duration")
        elif sample.duration is not None:
            updates["duration"] = sample.duration
        return updates

    def _lineage(
        self,
        config: OperatorConfig,
        input_key: str,
        output_key: str,
        output_path: str,
    ) -> dict[str, Any]:
        return {
            "operator": self.full_name,
            "version": self.version,
            "params": dict(config.params),
            "input_key": input_key,
            "output_key": output_key,
            "output_path": output_path,
        }
