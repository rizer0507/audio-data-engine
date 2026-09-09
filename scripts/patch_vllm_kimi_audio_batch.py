#!/usr/bin/env python3
"""Patch vLLM Kimi-Audio so a mixed-T encoder batch does not kill EngineCore.

vLLM 0.19.0 ``kimi_audio.py`` calls ``input_features.dim()`` and assumes a
tensor. Concurrent transcriptions of different audio lengths arrive as a Python
list and raise ``AttributeError: 'list' object has no attribute 'dim'``.

Run this **inside the same conda env that launched** ``vllm serve``, then restart
the server:

    python scripts/patch_vllm_kimi_audio_batch.py
    python scripts/patch_vllm_kimi_audio_batch.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MARKER = "# patched-by: audio-data-engine kimi mm list batch"

OLD_SNIPPET = '''    def _process_audio_input(
        self, audio_input: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        input_features = audio_input["whisper_input_features"]

        # KimiAudioWhisperEncoder expects list of tensors
        if input_features.dim() == 3:
            input_features = input_features.unbind(dim=0)

        # Run through Whisper encoder
        audio_features = self.audio_tower(input_features)

        # Reshape for 4x downsampling (Whisper outputs at 50Hz, need 12.5Hz)
        B, T, D = audio_features.shape
        if T % 4 != 0:
            pad_len = 4 - (T % 4)
            audio_features = torch.nn.functional.pad(audio_features, (0, 0, 0, pad_len))
            T = audio_features.shape[1]  # Update T after padding

        audio_features = audio_features.reshape(B, T // 4, D * 4)

        # Project to LLM dimension
        audio_embeds = self.multi_modal_projector(audio_features)
        return audio_embeds

    def embed_multimodal(self, **kwargs: object) -> list[torch.Tensor] | None:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return []

        audio_embeds = self._process_audio_input(audio_input)

        # audio_embeds shape: [batch_size, seq_len, hidden_dim]
        # Return as list of 2D tensors, one per batch item
        if audio_embeds.dim() == 3:
            # Unbind batch dimension: [B, T, D] -> list of B tensors [T, D]
            return list(audio_embeds.unbind(dim=0))
        else:
            # Single sample: [T, D] -> wrap in list
            return [audio_embeds]
'''

NEW_SNIPPET = '''    def _encode_whisper_input_features(self, input_features: torch.Tensor) -> torch.Tensor:
        ''' + MARKER + '''
        # KimiAudioWhisperEncoder expects list of tensors
        if input_features.dim() == 3:
            input_features = input_features.unbind(dim=0)

        audio_features = self.audio_tower(input_features)

        B, T, D = audio_features.shape
        if T % 4 != 0:
            pad_len = 4 - (T % 4)
            audio_features = torch.nn.functional.pad(audio_features, (0, 0, 0, pad_len))
            T = audio_features.shape[1]

        audio_features = audio_features.reshape(B, T // 4, D * 4)
        return self.multi_modal_projector(audio_features)

    def _process_audio_input(
        self, audio_input: dict[str, torch.Tensor]
    ) -> torch.Tensor | list[torch.Tensor]:
        input_features = audio_input["whisper_input_features"]

        if isinstance(input_features, (list, tuple)):
            tensors = [
                feat if isinstance(feat, torch.Tensor) else torch.as_tensor(feat)
                for feat in input_features
            ]
            if not tensors:
                raise ValueError("whisper_input_features is empty")
            if all(feat.shape == tensors[0].shape for feat in tensors):
                stacked = tensors[0] if len(tensors) == 1 else torch.stack(tensors, dim=0)
                return self._encode_whisper_input_features(stacked)
            return [self._encode_whisper_input_features(feat) for feat in tensors]

        return self._encode_whisper_input_features(input_features)

    def embed_multimodal(self, **kwargs: object) -> list[torch.Tensor] | None:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return []

        audio_embeds = self._process_audio_input(audio_input)

        if isinstance(audio_embeds, (list, tuple)):
            out: list[torch.Tensor] = []
            for item in audio_embeds:
                if item.dim() == 3:
                    out.extend(list(item.unbind(dim=0)))
                else:
                    out.append(item)
            return out
        if audio_embeds.dim() == 3:
            return list(audio_embeds.unbind(dim=0))
        return [audio_embeds]
'''


def apply_patch(source: str) -> str:
    """Return patched module text. Idempotent if MARKER is already present."""
    if MARKER in source:
        return source
    if OLD_SNIPPET not in source:
        raise ValueError(
            "未找到 vLLM 0.19 Kimi _process_audio_input 原始片段；"
            "请确认当前环境的 vllm.model_executor.models.kimi_audio 未改过"
        )
    return source.replace(OLD_SNIPPET, NEW_SNIPPET, 1)


def locate_kimi_audio_py() -> Path:
    try:
        import vllm.model_executor.models.kimi_audio as module
    except ImportError as exc:
        raise SystemExit(
            "当前 Python 找不到 vllm。请在启动 vllm serve 的同一个 conda 环境里运行本脚本。"
        ) from exc
    path = Path(module.__file__ or "")
    if not path.is_file():
        raise SystemExit(f"无法定位 kimi_audio.py: {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="给已安装的 vLLM Kimi-Audio 打混 T batch 补丁")
    parser.add_argument("--dry-run", action="store_true", help="只检查，不写文件")
    args = parser.parse_args(argv)

    path = locate_kimi_audio_py()
    original = path.read_text(encoding="utf-8")
    try:
        patched = apply_patch(original)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if patched == original:
        print(f"already patched: {path}")
        return 0
    if args.dry_run:
        print(f"would patch: {path}")
        return 0

    backup = path.with_suffix(path.suffix + ".bak-audio-engine")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
    path.write_text(patched, encoding="utf-8")
    print(f"patched: {path}")
    print(f"backup:  {backup}")
    print("请重启 vllm serve 后再跑探针。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
