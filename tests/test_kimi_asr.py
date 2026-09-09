from __future__ import annotations

import importlib.util
from pathlib import Path

import audio_engine.operators  # noqa: F401
import audio_engine.operators.asr.kimi as kimi_module
from audio_engine.core.manifest import Manifest
from audio_engine.core.operator import OperatorConfig
from audio_engine.core.pipeline import ExecutionConfig, PipelineConfig, PipelineRunner, PipelineStep
from audio_engine.core.registry import OperatorRegistry
from audio_engine.core.sample import Sample


def _samples(count: int) -> list[Sample]:
    return [
        Sample(
            id=f"s{index}",
            source_path=f"/audio/s{index}.wav",
            sha256=f"{index:064x}",
            audio={"resampled_16k": f"/audio/s{index}.wav"},
        )
        for index in range(count)
    ]


def _config(tmp_path: Path, **params) -> OperatorConfig:
    run_dir = params.pop("run_dir", None)
    return OperatorConfig(
        params={
            "input_audio_key": "resampled_16k",
            "api_base": "http://127.0.0.1:5554",
            "model": "kimi-audio",
            "model_version": "test",
            "concurrency": 2,
            "batch_size": 2,
            **params,
        },
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "derived",
        run_dir=Path(run_dir) if run_dir is not None else None,
    )


def test_kimi_batch_transcribes_via_vllm_and_reuses_cache(tmp_path: Path, monkeypatch):
    calls: list[str] = []

    def fake_transcribe(audio_path: str, settings: dict) -> dict:
        calls.append(audio_path)
        return {"text": f"text:{Path(audio_path).stem}", "language": "zh"}

    monkeypatch.setattr(kimi_module, "_call_vllm_transcription", fake_transcribe)
    operator = OperatorRegistry.get("asr.kimi_batch")
    config = _config(tmp_path)

    first = operator.process_batch(_samples(3), config)
    assert sorted(calls) == ["/audio/s0.wav", "/audio/s1.wav", "/audio/s2.wav"]
    assert [result.sample.get_transcript_text("kimi") for result in first] == [
        "text:s0",
        "text:s1",
        "text:s2",
    ]
    assert all(result.sample.is_completed("asr.kimi_batch") for result in first)

    second = operator.process_batch(_samples(3), config)
    assert all(result.cache_hit for result in second)
    assert len(calls) == 3


def test_kimi_batch_isolates_failed_audio(tmp_path: Path, monkeypatch):
    def fake_transcribe(audio_path: str, settings: dict) -> dict:
        if audio_path.endswith("s1.wav"):
            raise ValueError("broken audio")
        return {"text": f"text:{Path(audio_path).stem}", "language": "zh"}

    monkeypatch.setattr(kimi_module, "_call_vllm_transcription", fake_transcribe)
    results = OperatorRegistry.get("asr.kimi_batch").process_batch(
        _samples(3), _config(tmp_path, concurrency=3, batch_size=3)
    )

    assert results[0].sample.get_transcript_text("kimi") == "text:s0"
    assert results[1].sample.status["asr.kimi_batch"] == "failed"
    assert "broken audio" in results[1].sample.errors["asr.kimi_batch"]
    assert results[2].sample.get_transcript_text("kimi") == "text:s2"


def test_kimi_vllm_environment_and_operator_precedence(monkeypatch):
    monkeypatch.delenv("KIMI_ASR_API_BASES", raising=False)
    monkeypatch.setenv("KIMI_ASR_API_BASE", "http://environment:5554/v1")
    monkeypatch.setenv("KIMI_ASR_MODEL", "environment-model")
    env_settings = kimi_module._resolve_settings(OperatorConfig(params={}))
    assert env_settings["api_base"] == "http://environment:5554/v1"
    assert env_settings["api_bases"] == ["http://environment:5554/v1"]
    assert env_settings["model"] == "environment-model"

    explicit = kimi_module._resolve_settings(
        OperatorConfig(params={"api_base": "http://operator:5554", "model": "operator-model"})
    )
    assert explicit["api_base"] == "http://operator:5554"
    assert explicit["model"] == "operator-model"


def test_kimi_vllm_parses_multiple_api_bases(monkeypatch):
    monkeypatch.delenv("KIMI_ASR_API_BASE", raising=False)
    monkeypatch.setenv(
        "KIMI_ASR_API_BASES",
        "http://127.0.0.1:5554, http://127.0.0.1:5555",
    )
    settings = kimi_module._resolve_settings(OperatorConfig(params={}))
    assert settings["api_bases"] == [
        "http://127.0.0.1:5554",
        "http://127.0.0.1:5555",
    ]
    first = kimi_module.select_kimi_api_base(settings, "s0")
    second = kimi_module.select_kimi_api_base(settings, "s1")
    assert first in settings["api_bases"]
    assert second in settings["api_bases"]
    assert kimi_module.select_kimi_api_base(settings, "s0") == first


def test_kimi_batch_routes_samples_across_replicas(tmp_path: Path, monkeypatch):
    seen: list[str] = []

    def fake_transcribe(audio_path: str, settings: dict) -> dict:
        seen.append(settings["api_base"])
        return {"text": f"text:{Path(audio_path).stem}", "language": "zh"}

    monkeypatch.setattr(kimi_module, "_call_vllm_transcription", fake_transcribe)
    OperatorRegistry.get("asr.kimi_batch").process_batch(
        _samples(4),
        _config(
            tmp_path,
            api_bases=["http://127.0.0.1:5554", "http://127.0.0.1:5555"],
            concurrency=4,
            batch_size=4,
        ),
    )
    assert set(seen) <= {"http://127.0.0.1:5554", "http://127.0.0.1:5555"}
    assert len(seen) == 4


def test_kimi_batch_cache_includes_pad_bucket(tmp_path: Path, monkeypatch):
    calls: list[str] = []

    def fake_transcribe(audio_path: str, settings: dict) -> dict:
        calls.append(audio_path)
        return {"text": f"text:{Path(audio_path).stem}", "language": "zh"}

    monkeypatch.setattr(kimi_module, "_call_vllm_transcription", fake_transcribe)
    operator = OperatorRegistry.get("asr.kimi_batch")
    unpadded = _samples(1)
    operator.process_batch(unpadded, _config(tmp_path, input_audio_key="resampled_16k"))

    padded = _samples(1)
    padded[0].labels["kimi_pad_target_s"] = 3
    padded[0].labels["kimi_pad_mode"] = "padded"
    padded[0].audio["kimi_padded_16k"] = "/audio/s0.wav"
    operator.process_batch(
        padded, _config(tmp_path, input_audio_key="kimi_padded_16k")
    )
    assert len(calls) == 2


def test_kimi_batch_missing_padded_audio_fails_only_that_sample(tmp_path: Path, monkeypatch):
    def fake_transcribe(audio_path: str, settings: dict) -> dict:
        return {"text": f"text:{Path(audio_path).stem}", "language": "zh"}

    monkeypatch.setattr(kimi_module, "_call_vllm_transcription", fake_transcribe)
    samples = _samples(2)
    samples[0].audio = {}
    samples[1].audio["kimi_padded_16k"] = samples[1].audio["resampled_16k"]
    results = OperatorRegistry.get("asr.kimi_batch").process_batch(
        samples, _config(tmp_path, input_audio_key="kimi_padded_16k", concurrency=2, batch_size=2)
    )
    assert results[0].sample.status["asr.kimi_batch"] == "failed"
    assert results[1].sample.get_transcript_text("kimi") == "text:s1"


def test_kimi_batch_sends_one_pad_bucket_per_http_chunk(tmp_path: Path, monkeypatch):
    chunks: list[list[str]] = []

    def fake_many(paths, settings, sample_ids=None):
        chunks.append([Path(path).stem for path in paths])
        return [{"text": f"text:{Path(path).stem}", "language": "zh"} for path in paths]

    monkeypatch.setattr(kimi_module, "_transcribe_many", fake_many)
    samples = _samples(4)
    for sample, target in zip(samples, (3, 6, 3, 6)):
        sample.labels["kimi_pad_mode"] = "padded"
        sample.labels["kimi_pad_target_s"] = target
        sample.audio["kimi_padded_16k"] = sample.audio["resampled_16k"]
    OperatorRegistry.get("asr.kimi_batch").process_batch(
        samples,
        _config(tmp_path, input_audio_key="kimi_padded_16k", concurrency=4, batch_size=4),
    )
    assert chunks == [["s0", "s2"], ["s1", "s3"]]


def test_kimi_batch_binds_shard_to_one_replica(tmp_path: Path, monkeypatch):
    seen: list[str] = []

    def fake_transcribe(audio_path: str, settings: dict) -> dict:
        seen.append(settings["api_base"])
        return {"text": f"text:{Path(audio_path).stem}", "language": "zh"}

    monkeypatch.setattr(kimi_module, "_call_vllm_transcription", fake_transcribe)
    OperatorRegistry.get("asr.kimi_batch").process_batch(
        _samples(3),
        _config(
            tmp_path,
            api_bases=["http://127.0.0.1:5554", "http://127.0.0.1:5555"],
            concurrency=3,
            batch_size=3,
            run_dir=tmp_path / "shard-001",
        ),
    )
    assert seen == ["http://127.0.0.1:5555"] * 3


def test_kimi_batch_runs_over_parquet_pipeline(tmp_path: Path):
    input_path = tmp_path / "input.parquet"
    Manifest(_samples(3)).save(input_path)
    config = PipelineConfig(
        name="kimi_batch_test",
        input_manifest=str(input_path),
        steps=[
            PipelineStep(
                name="kimi_asr",
                operator="asr.kimi_batch",
                params={"input_audio_key": "resampled_16k", "mock": True, "concurrency": 2},
            )
        ],
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "derived",
        runs_dir=tmp_path / "runs",
        execution=ExecutionConfig(executor="sequential", workers=1, checkpoint_every=2),
    )

    runner = PipelineRunner(config)
    result = runner.run()

    assert len(result) == 3
    assert runner.metrics.to_dict()["by_step"]["kimi_asr"]["processed"] == 3
    assert all(sample.get_transcript_text("kimi").startswith("[mock:kimi:") for sample in result)


def _load_probe_module():
    script = Path(__file__).parents[1] / "scripts/probe_kimi_vllm.py"
    spec = importlib.util.spec_from_file_location("probe_kimi_vllm", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probe_discovers_single_file_and_directory(tmp_path: Path):
    probe = _load_probe_module()
    first = tmp_path / "b.WAV"
    second = tmp_path / "a.wav"
    ignored = tmp_path / "notes.txt"
    nested = tmp_path / "nested" / "c.wav"
    nested.parent.mkdir()
    for path in (first, second, ignored, nested):
        path.write_bytes(b"test")

    assert probe.discover_wavs(first) == [first.resolve()]
    assert probe.discover_wavs(tmp_path) == [second.resolve(), first.resolve()]
    assert probe.discover_wavs(tmp_path, recursive=True) == [
        second.resolve(),
        first.resolve(),
        nested.resolve(),
    ]


def test_probe_pad_and_repeat_flags(tmp_path: Path, monkeypatch):
    import numpy as np
    import soundfile as sf

    probe = _load_probe_module()
    wav = tmp_path / "short.wav"
    sf.write(str(wav), np.full(8000, 0.1, dtype=np.float32), 16000)
    calls: list[str] = []

    def fake_many(paths, settings, sample_ids=None):
        calls.extend(paths)
        return [{"text": "hello", "language": "zh"} for _ in paths]

    monkeypatch.setattr(kimi_module, "_transcribe_many", fake_many)
    monkeypatch.setattr(probe.kimi, "_transcribe_many", fake_many)
    assert probe.main([str(wav), "--pad", "--repeat", "2"]) == 0
    assert len(calls) == 2
    first_info = sf.info(calls[0])
    assert first_info.frames == 3 * 16000


def test_iter_pad_bucket_windows_keeps_buckets_separate():
    assert kimi_module.iter_pad_bucket_windows(
        [("padded", 3), ("padded", 6), ("padded", 3)],
        batch_size=4,
    ) == [[0, 2], [1]]
    assert kimi_module.iter_pad_bucket_windows(
        [("over_30s", None), ("over_30s", None)],
        batch_size=4,
    ) == [[0], [1]]
    assert kimi_module.iter_pad_bucket_windows(
        [("padded", 3), ("padded", 3), ("padded", 3)],
        batch_size=2,
    ) == [[0, 1], [2]]


def test_probe_mixed_pad_sends_one_bucket_per_http_chunk(tmp_path: Path, monkeypatch):
    import numpy as np
    import soundfile as sf

    probe = _load_probe_module()
    short_a = tmp_path / "a.wav"
    short_b = tmp_path / "b.wav"
    mid = tmp_path / "c.wav"
    sf.write(str(short_a), np.full(16000, 0.1, dtype=np.float32), 16000)
    sf.write(str(short_b), np.full(int(1.5 * 16000), 0.1, dtype=np.float32), 16000)
    sf.write(str(mid), np.full(5 * 16000, 0.1, dtype=np.float32), 16000)
    chunks: list[list[str]] = []

    def fake_many(paths, settings, sample_ids=None):
        chunks.append([Path(path).stem for path in paths])
        return [{"text": "hello", "language": "zh"} for _ in paths]

    monkeypatch.setattr(kimi_module, "_transcribe_many", fake_many)
    monkeypatch.setattr(probe.kimi, "_transcribe_many", fake_many)
    assert probe.main([str(tmp_path), "--pad", "--concurrency", "4"]) == 0
    assert chunks == [["a", "b"], ["c"]]


def _load_vllm_patch_module():
    script = Path(__file__).parents[1] / "scripts/patch_vllm_kimi_audio_batch.py"
    spec = importlib.util.spec_from_file_location("patch_vllm_kimi_audio_batch", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vllm_kimi_patch_applies_and_is_idempotent():
    patch = _load_vllm_patch_module()
    fake = "prefix\n" + patch.OLD_SNIPPET + "suffix\n"
    once = patch.apply_patch(fake)
    assert patch.MARKER in once
    assert "isinstance(input_features, (list, tuple))" in once
    assert patch.OLD_SNIPPET not in once
    assert patch.apply_patch(once) == once
