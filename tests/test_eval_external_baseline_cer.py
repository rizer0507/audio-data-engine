"""Tests for external-baseline CER alignment helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "eval_external_baseline_cer.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("eval_external_baseline_cer", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_align_baseline_id_contains_stem():
    mod = _load_script()
    baseline = pd.DataFrame(
        {
            "baseline_id": ["batch/mt3000/utt_001.wav", "other_utt_002"],
            "baseline_text": ["不需要", "需要"],
        }
    )
    index = mod.build_baseline_index(baseline)
    ids = baseline["baseline_id"].tolist()
    assert mod.align_baseline_id("utt_001", ids, index) == "batch/mt3000/utt_001.wav"
    assert mod.align_baseline_id("utt_002", ids, index) == "other_utt_002"
    assert mod.align_baseline_id("missing", ids, index) is None


def test_score_pair_uses_metric_runner(tmp_path: Path):
    mod = _load_script()
    normalization = {
        "punctuation": {"remove": True},
        "whitespace": {"remove": True},
        "filler": {"remove": False},
    }
    out = mod.score_pair("不需要", "需要", normalization, "qwen_vs_baseline")
    assert out["qwen_vs_baseline_cer"] == 0.333333
    assert out["qwen_vs_baseline_字准率"] == round(1 - 0.333333, 6)
