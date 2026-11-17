from __future__ import annotations

import pytest

from audio_engine.core.audio_scene import (
    SCENE_FENSHEN,
    SCENE_SHENPI_INTERVIEW,
    SCENE_SHENPI_OUTBOUND,
    SCENE_UNKNOWN,
    infer_legacy_fenshen_scene,
    requires_reingest_for_shenpi,
    resolve_audio_scene,
)


@pytest.mark.parametrize(
    "stem,scene,rate",
    [
        ("fenshen-276cc46a-bf3387be", SCENE_FENSHEN, 16000),
        ("fenshen276cc46a", SCENE_FENSHEN, 16000),
        ("shenpi_ai-c24b27s-V2610091", SCENE_SHENPI_INTERVIEW, 8000),
        ("shepi_ai-h726h9999-Ph829888", SCENE_SHENPI_INTERVIEW, 8000),
        ("shepi_ai-V726-8jP29887", SCENE_SHENPI_INTERVIEW, 8000),
        ("shepi_ai-7jsn7288-8j22j987", SCENE_SHENPI_OUTBOUND, 16000),
        ("shepi_ai-V726-8j229887", SCENE_SHENPI_OUTBOUND, 16000),
        ("other_file", SCENE_UNKNOWN, None),
    ],
)
def test_resolve_audio_scene_happy(stem, scene, rate):
    result = resolve_audio_scene(stem)
    assert result.scene == scene
    assert result.source_sample_rate == rate
    assert result.rejected is False


def test_resolve_rejects_malformed_shenpi():
    missing = resolve_audio_scene("shenpi_ai-onlymid")
    assert missing.rejected is True
    assert missing.reject_reason == "shenpi_missing_second_separator"

    empty = resolve_audio_scene("shepi_ai--V123")
    assert empty.rejected is True
    assert empty.reject_reason == "shenpi_empty_mid_or_suffix"


def test_legacy_helpers():
    assert infer_legacy_fenshen_scene("fenshen-1", {}) == SCENE_FENSHEN
    assert infer_legacy_fenshen_scene("other", {}) is None
    assert requires_reingest_for_shenpi("shenpi_ai-a-b", {}) is True
    assert requires_reingest_for_shenpi("shenpi_ai-a-b", {"audio_scene": "shenpi_outbound"}) is False
