from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from loguru import logger

from audio_engine.core.artifacts import atomic_path, derived_audio_path
from audio_engine.core.audio_normalize import (
    TARGET_SR,
    NormalizeMeta,
    format_rate_alert,
    is_non_target_rate,
    resolve_headerless_rate,
    take_channel_0,
    validate_headerless_pcm_bytes,
)
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


def _append_run_alert(config: OperatorConfig, record: dict[str, Any]) -> None:
    """Append one normalize alert into run_dir for final summary aggregation."""
    if not config.run_dir:
        return
    path = Path(config.run_dir) / "normalize_alerts.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@register_operator
class PcmToWavOperator(BaseOperator):
    """Normalize to 16 kHz mono PCM16 WAV (channel 0 only).

    ``sample_rate`` is the *output* target (default 16 kHz). Headered files are
    probed for real rate/channels. Headerless ``.pcm`` / ``.raw`` use
    ``source_sample_rate`` (default 16 kHz) and ``channels`` (default 1);
    multi-channel headerless must be explicitly declared then de-interleaved —
    never interpreted as packed mono.
    """

    name = "pcm_to_wav"
    version = "1.3.0"
    category = "audio"

    def _execute(self, sample: Sample, config: OperatorConfig) -> dict[str, Any]:
        input_key = config.params.get("input_audio_key", "raw")
        output_key = config.params.get("output_audio_key", "pcm_to_wav")
        target_sr = int(config.params.get("sample_rate", DEFAULT_SAMPLE_RATE or TARGET_SR))
        declared_channels = int(config.params.get("channels", 1))
        dtype = config.params.get("dtype", "int16")

        input_path = Path(sample.audio_path(input_key))
        suffix = input_path.suffix.lower()
        headered = suffix in _CONTAINER_EXTS or looks_like_wav(input_path)
        source_sr, rate_source, data, source_channels, channel_policy = self._load_mono(
            input_path, suffix, config, declared_channels, dtype
        )

        mono = data
        duration_src = float(len(mono) / source_sr) if source_sr else float(sample.duration or 0)

        if source_sr == target_sr and channel_policy == "mono_passthrough" and headered:
            # True mono already at target — alias path (WAV/FLAC/etc.).
            meta = NormalizeMeta(
                source_sample_rate=source_sr,
                target_sample_rate=target_sr,
                source_channels=source_channels,
                selected_channel=0,
                channel_policy=channel_policy,
                sample_rate_source=rate_source,
                resampled=False,
                duration_source_sec=duration_src,
                duration_target_sec=duration_src,
            )
            return self._result(
                sample,
                config,
                input_key,
                output_key,
                str(input_path.resolve()),
                meta,
                quality_status="passthrough",
                pcm_converted=False,
            )

        if source_sr == target_sr:
            to_write = mono
            quality_status = "converted" if channel_policy == "take_channel_0" else "converted"
            resampled = False
        else:
            to_write = resample_audio(np.asanyarray(mono), source_sr, target_sr)
            quality_status = "resampled"
            resampled = True

        output_path = derived_audio_path(config.output_dir, "pcm_to_wav", sample)
        with atomic_path(output_path) as tmp:
            sf.write(str(tmp), to_write, target_sr, subtype="PCM_16")

        n_frames = len(to_write) if getattr(to_write, "ndim", 1) == 1 else to_write.shape[0]
        duration_tgt = n_frames / target_sr if target_sr else duration_src
        meta = NormalizeMeta(
            source_sample_rate=source_sr,
            target_sample_rate=target_sr,
            source_channels=source_channels,
            selected_channel=0,
            channel_policy=channel_policy,
            sample_rate_source=rate_source,
            resampled=resampled,
            duration_source_sec=duration_src,
            duration_target_sec=duration_tgt,
        )
        return self._result(
            sample,
            config,
            input_key,
            output_key,
            str(output_path.resolve()),
            meta,
            quality_status=quality_status,
            pcm_converted=rate_source != "file_header",
        )

    def _load_mono(
        self,
        input_path: Path,
        suffix: str,
        config: OperatorConfig,
        declared_channels: int,
        dtype: str,
    ) -> tuple[int, str, np.ndarray, int, str]:
        """Return (source_sr, rate_source, mono_data, source_channels, channel_policy)."""
        headered = suffix in _CONTAINER_EXTS or looks_like_wav(input_path)
        if headered:
            data, sr = sf.read(str(input_path), always_2d=True)
            mono, source_channels, channel_policy = take_channel_0(data)
            return int(sr), "file_header", mono, source_channels, channel_policy

        if suffix not in _PCM_EXTS and not headered:
            raise ValueError(
                f"audio.pcm_to_wav: unsupported format '{suffix}' for {input_path}"
            )

        source_sr, rate_source = resolve_headerless_rate(
            config.params, default_sr=DEFAULT_SAMPLE_RATE
        )
        channels = max(1, int(declared_channels))
        raw = np.fromfile(input_path, dtype=dtype)
        validate_headerless_pcm_bytes(len(raw), channels=channels)
        if channels == 1:
            mono = raw
            source_channels = 1
            channel_policy = "mono_passthrough"
        else:
            shaped = raw.reshape(-1, channels)
            mono, source_channels, channel_policy = take_channel_0(shaped)
        return source_sr, rate_source, mono, source_channels, channel_policy

    def _result(
        self,
        sample: Sample,
        config: OperatorConfig,
        input_key: str,
        output_key: str,
        output_path: str,
        meta: NormalizeMeta,
        *,
        quality_status: str,
        pcm_converted: bool,
    ) -> dict[str, Any]:
        if is_non_target_rate(meta.source_sample_rate, meta.target_sample_rate):
            msg = format_rate_alert(
                sample_id=sample.id,
                path=str(Path(sample.audio_path(input_key))),
                source_sr=meta.source_sample_rate,
                target_sr=meta.target_sample_rate,
                source_channels=meta.source_channels,
                selected_channel=meta.selected_channel,
            )
            # Prominent console + log (user requirement: never silent for 8k / non-16k).
            print(msg, flush=True)
            logger.warning(msg)
            _append_run_alert(
                config,
                {
                    "sample_id": sample.id,
                    "path": str(Path(sample.audio_path(input_key))),
                    "source_sample_rate": meta.source_sample_rate,
                    "target_sample_rate": meta.target_sample_rate,
                    "source_channels": meta.source_channels,
                    "selected_channel": meta.selected_channel,
                    "channel_policy": meta.channel_policy,
                    "sample_rate_source": meta.sample_rate_source,
                    "resampled": meta.resampled,
                    "duration_source_sec": meta.duration_source_sec,
                    "duration_target_sec": meta.duration_target_sec,
                },
            )

        if meta.channel_policy == "take_channel_0":
            ch_msg = (
                f"[AUDIO-NORMALIZE] multi-channel→ch0: id={sample.id} "
                f"source_channels={meta.source_channels}"
            )
            print(ch_msg, flush=True)
            logger.info(ch_msg)

        quality = {
            "pcm_to_wav": quality_status,
            "probed": meta.sample_rate_source == "file_header",
            **meta.as_quality(),
        }
        labels = {
            "pcm_converted": pcm_converted,
            **meta.as_labels(),
        }
        return {
            "audio": {output_key: output_path},
            "sample_rate": meta.target_sample_rate,
            "channels": 1,
            "duration": meta.duration_target_sec,
            "labels": labels,
            "quality": quality,
            "lineage_entry": {
                "operator": self.full_name,
                "version": self.version,
                "params": dict(config.params),
                "input_key": input_key,
                "output_key": output_key,
                "output_path": output_path,
            },
        }
