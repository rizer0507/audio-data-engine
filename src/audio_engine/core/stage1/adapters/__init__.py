"""Registry for stage-1 family launch adapters."""

from __future__ import annotations

from audio_engine.core.stage1.adapters.glm import GlmAdapter
from audio_engine.core.stage1.adapters.qwen import QwenAdapter
from audio_engine.core.stage1.adapters.sensevoice import SenseVoiceAdapter

_ADAPTERS = {
    "qwen": QwenAdapter(),
    "glm": GlmAdapter(),
    "sensevoice": SenseVoiceAdapter(),
    "sv": SenseVoiceAdapter(),
}


def get_adapter(family: str):
    key = family.lower().strip()
    adapter = _ADAPTERS.get(key)
    if adapter is None:
        known = ", ".join(sorted({"qwen", "glm", "sensevoice"}))
        raise ValueError(f"未知模型家族: {family}（支持 {known}）")
    return adapter


def list_families() -> list[str]:
    return ["qwen", "glm", "sensevoice"]
