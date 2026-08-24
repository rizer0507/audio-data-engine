from __future__ import annotations

import json
from pathlib import Path

import audio_engine.operators  # noqa: F401
from audio_engine.core.manifest import Manifest
from audio_engine.core.pipeline import ExecutionConfig, PipelineConfig, PipelineRunner, PipelineStep
from audio_engine.core.sample import Sample


def test_python_script_pipeline_is_concurrent_logged_and_cache_aware(tmp_path: Path):
    script = tmp_path / "policy.py"
    script.write_text(
        "def process(sample, params, context):\n"
        "    context.log('evaluated', value=params['value'])\n"
        "    return {'labels': {'custom': params['value']}}\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "input.jsonl"
    Manifest([Sample(id=f"s{i}", source_path=f"s{i}.wav", sha256=str(i)) for i in range(12)]).save(
        manifest_path
    )

    def run(value: str, name: str):
        config = PipelineConfig(
            name=name,
            input_manifest=str(manifest_path),
            steps=[
                PipelineStep(
                    name="policy",
                    operator="script.python",
                    params={"path": str(script), "value": value},
                )
            ],
            execution=ExecutionConfig(executor="thread", workers=4, checkpoint_every=0),
            cache_dir=tmp_path / "cache",
            runs_dir=tmp_path / "runs",
        )
        runner = PipelineRunner(config)
        return runner, runner.run()

    first_runner, first = run("v1", "first")
    assert [sample.labels["custom"] for sample in first] == ["v1"] * 12
    assert first_runner.metrics.processed == 12
    events = [
        json.loads(line)
        for line in (first_runner.run_dir / "script_logs" / "policy.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(events) == 36  # started + script-owned event + finished
    assert {event["sample_id"] for event in events} == {f"s{i}" for i in range(12)}

    # A changed implementation must not reuse results cached for the old source.
    script.write_text(
        "def process(sample, params, context):\n"
        "    context.log('changed')\n"
        "    return {'labels': {'custom': 'changed'}}\n",
        encoding="utf-8",
    )
    second_runner, second = run("v1", "second")
    assert [sample.labels["custom"] for sample in second] == ["changed"] * 12
    assert second_runner.metrics.cache_hits == 0
