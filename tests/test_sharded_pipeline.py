"""Tests for YAML sharding and single-command sharded pipeline run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml
from typer.testing import CliRunner

import audio_engine.operators  # noqa: F401
from audio_engine.cli.main import app
from audio_engine.core.manifest import Manifest
from audio_engine.core.pipeline import PipelineConfig, ShardingConfig
from audio_engine.core.sample import Sample
from audio_engine.core.sharded_run import run_sharded_pipeline


def _write_wav(path: Path, duration_s: float = 0.2, sr: int = 16000) -> None:
    n = int(sr * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)
    data = 0.2 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(path), data.astype(np.float32), sr)


def test_sharding_config_from_yaml(tmp_path: Path):
    manifest = tmp_path / "in.parquet"
    Manifest([]).save(manifest)
    yaml_path = tmp_path / "pipe.yaml"
    yaml_path.write_text(
        yaml.dump(
            {
                "name": "with_shards",
                "input": {"manifest": str(manifest)},
                "output": {"manifest": str(tmp_path / "out.parquet")},
                "sharding": {
                    "shards": 4,
                    "strategy": "duration-balanced",
                    "gpus": [0, 1],
                    "instances_per_gpu": 2,
                    "parallel_shards": 4,
                    "workers": 2,
                },
                "pipeline": [],
            }
        ),
        encoding="utf-8",
    )
    cfg = PipelineConfig.from_yaml(yaml_path)
    assert cfg.sharding is not None
    assert cfg.sharding.shards == 4
    assert cfg.sharding.strategy == "duration-balanced"
    assert cfg.sharding.gpus == ("0", "1")
    assert cfg.sharding.effective_parallel == 4
    assert cfg.config_path == str(yaml_path.resolve())


def test_sharding_requires_output_manifest(tmp_path: Path):
    manifest = tmp_path / "in.parquet"
    Manifest([]).save(manifest)
    yaml_path = tmp_path / "pipe.yaml"
    yaml_path.write_text(
        yaml.dump(
            {
                "name": "bad",
                "input": {"manifest": str(manifest)},
                "sharding": {"shards": 2},
                "pipeline": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="output.manifest"):
        PipelineConfig.from_yaml(yaml_path)


def test_sharding_gpu_slot_validation():
    with pytest.raises(ValueError, match="cannot exceed"):
        ShardingConfig(
            shards=4,
            parallel_shards=4,
            gpus=("0",),
            instances_per_gpu=1,
        ).validate_parallel()


def test_pipeline_run_sharding_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    audio_dir = tmp_path / "wavs"
    audio_dir.mkdir()
    samples = []
    for i in range(4):
        wav = audio_dir / f"clip{i:03d}.wav"
        _write_wav(wav, duration_s=0.15 + 0.05 * i)
        samples.append(
            Sample(
                id=f"clip{i:03d}",
                source_path=str(wav),
                sha256=f"{i:064x}",
                audio={"raw": str(wav), "resampled_16k": str(wav)},
                sample_rate=16000,
                duration=0.15 + 0.05 * i,
            )
        )
    input_path = tmp_path / "input.parquet"
    Manifest(samples).save(input_path)
    output_path = tmp_path / "out.parquet"

    yaml_path = tmp_path / "label_pipe.yaml"
    script_path = tmp_path / "label.py"
    script_path.write_text(
        "def process(sample, params, context):\n"
        "    context.log('ok')\n"
        "    return {'labels': {'sharded_ok': True}}\n",
        encoding="utf-8",
    )
    yaml_path.write_text(
        yaml.dump(
            {
                "name": "shard_script",
                "input": {"manifest": str(input_path)},
                "output": {"manifest": str(output_path)},
                "runs_dir": str(tmp_path / "runs"),
                "execution": {"executor": "sequential", "workers": 1, "checkpoint_every": 0},
                "sharding": {
                    "shards": 2,
                    "strategy": "hash",
                    "parallel_shards": 2,
                    "workers": 1,
                    "executor": "sequential",
                },
                "pipeline": [
                    {
                        "name": "tag",
                        "operator": "script.python",
                        "params": {"path": str(script_path)},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["pipeline", "run", str(yaml_path)])
    assert result.exit_code == 0, result.output
    assert output_path.exists()
    merged = Manifest.load(output_path)
    assert len(merged) == 4
    assert all(s.labels.get("sharded_ok") is True for s in merged.samples)


def test_run_sharded_pipeline_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    samples = [
        Sample(
            id=f"s{i}",
            source_path=f"/tmp/{i}.wav",
            sha256=f"{i:064x}",
            audio={"raw": f"/tmp/{i}.wav"},
            duration=float(i + 1),
        )
        for i in range(3)
    ]
    input_path = tmp_path / "in.parquet"
    Manifest(samples).save(input_path)
    output_path = tmp_path / "merged.parquet"
    script_path = tmp_path / "noop.py"
    script_path.write_text(
        "def process(sample, params, context):\n"
        "    return {'labels': {'n': 1}}\n",
        encoding="utf-8",
    )
    yaml_path = tmp_path / "p.yaml"
    yaml_path.write_text(
        yaml.dump(
            {
                "name": "api_shard",
                "input": {"manifest": str(input_path)},
                "output": {"manifest": str(output_path)},
                "runs_dir": str(tmp_path / "runs"),
                "execution": {"checkpoint_every": 0},
                "sharding": {"shards": 2, "parallel_shards": 2, "strategy": "duration-balanced"},
                "pipeline": [
                    {
                        "name": "tag",
                        "operator": "script.python",
                        "params": {"path": str(script_path)},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    cfg = PipelineConfig.from_yaml(yaml_path)
    out = run_sharded_pipeline(cfg, run_root=tmp_path / "shard_root")
    assert out.shard_count == 2
    assert len(out.manifest) == 3
    assert output_path.exists()
