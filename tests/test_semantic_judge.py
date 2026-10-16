"""046 semantic.judge: concurrency, cache, resume, schema, multi-task."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

from audio_engine.core.manifest import Manifest
from audio_engine.core.operator import OperatorConfig
from audio_engine.core.pipeline import ExecutionConfig, PipelineConfig, PipelineRunner, PipelineStep
from audio_engine.core.sample import Sample
from audio_engine.core.semantic.client import ConcurrencyGate, SemanticChatClient
from audio_engine.core.semantic.concurrency import validate_concurrency_budget
from audio_engine.core.semantic.config import freeze_bundle, load_service_config
from audio_engine.core.semantic.evidence import build_families_context
from audio_engine.core.semantic.fingerprint import semantic_cache_key
from audio_engine.core.semantic.preflight import preflight_samples
from audio_engine.core.semantic.types import RUN_FAILED, RUN_MISSING, RUN_SUCCESS_EMPTY, RUN_SUCCESS_TEXT
from audio_engine.operators.semantic.judge import SemanticJudgeOperator


ROOT = Path(__file__).resolve().parents[1]
CONFLICT_TASK = ROOT / "configs/semantic/tasks/semantic_conflict_v1.yaml"
INTENT_TASK = ROOT / "configs/semantic/tasks/intent_v1.yaml"


def _sample(
    sid: str,
    *,
    transcripts: dict | None = None,
    labels: dict | None = None,
) -> Sample:
    return Sample(
        id=sid,
        source_path=f"/virtual/{sid}.wav",
        sha256=f"sha-{sid}",
        duration=1.5,
        transcripts=transcripts
        or {
            "qwen_1": {"text": "我不要", "model": "qwen", "status": "success"},
            "qwen_2": {"text": "我不要", "model": "qwen", "status": "success"},
            "glm_1": {"text": "我要", "model": "glm", "status": "success"},
            "glm_2": {"text": "我要", "model": "glm", "status": "success"},
            "sensevoice_1": {"text": "我不要", "model": "sv", "status": "success"},
            "sensevoice_2": {"text": "我不要", "model": "sv", "status": "success"},
        },
        labels=labels or {"segment_pcm_sha256": f"seg-{sid}", "is_vad_segment": True},
    )


@pytest.fixture()
def service_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "service.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "api_base": "http://127.0.0.1:9/v1",
                "model": "mock-model",
                "model_revision": "test-rev",
                "api_key_env": "SEMANTIC_JUDGE_API_KEY_TEST",
                "generation": {"max_tokens": 128, "temperature": 0.0},
                "retry": {"max_attempts": 2, "connect_timeout_s": 1, "read_timeout_s": 2, "request_timeout_s": 5},
                "max_concurrency_total": 8,
                "structured_output_mode": "prompt_schema_validate",
                "probe_on_start": False,
                "cache_namespace": "test",
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return path


def _write_task(tmp_path: Path, service_yaml: Path, *, task_id: str, base: Path) -> Path:
    raw = yaml.safe_load(base.read_text(encoding="utf-8"))
    raw["task_id"] = task_id
    raw["service_config"] = str(service_yaml)
    # Keep prompt/schema paths absolute so freeze works from any cwd.
    raw["prompts"]["system"] = str(ROOT / raw["prompts"]["system"])
    raw["prompts"]["user"] = str(ROOT / raw["prompts"]["user"])
    raw["output_schema"] = str(ROOT / raw["output_schema"])
    out = tmp_path / f"{task_id}.yaml"
    out.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return out


class _FakeHandler(BaseHTTPRequestHandler):
    delay_s = 0.05
    fail_codes: dict[str, list[int]] = {}
    peak_active = 0
    _active = 0
    _lock = threading.Lock()
    responses: dict[str, dict] = {}

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path.endswith("/models"):
            body = json.dumps({"data": [{"id": "mock-model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        payload = json.loads(raw.decode("utf-8"))
        with self._lock:
            type(self)._active += 1
            type(self).peak_active = max(type(self).peak_active, type(self)._active)
        try:
            time.sleep(self.delay_s)
            key = "default"
            user = ""
            for msg in payload.get("messages") or []:
                if msg.get("role") == "user":
                    user = str(msg.get("content") or "")
            if "样本标识" in user:
                # Extract sample id from rendered prompt when possible.
                import re

                m = re.search(r'样本标识：\s*"([^"]+)"', user)
                if m:
                    key = m.group(1)
            codes = self.fail_codes.get(key) or self.fail_codes.get("default")
            if codes:
                code = codes.pop(0)
                body = json.dumps({"error": f"forced {code}"}).encode()
                self.send_response(code)
                if code == 429:
                    self.send_header("Retry-After", "0")
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            result = self.responses.get(key) or {
                "conflict": True,
                "insufficient_evidence": False,
                "summary": f"mock-{key}",
                "differing_aspects": ["intent"],
            }
            content = json.dumps(result, ensure_ascii=False)
            body = json.dumps(
                {
                    "choices": [{"message": {"content": content}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        finally:
            with self._lock:
                type(self)._active -= 1


@pytest.fixture()
def fake_server():
    _FakeHandler.peak_active = 0
    _FakeHandler._active = 0
    _FakeHandler.fail_codes = {}
    _FakeHandler.responses = {}
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    yield f"http://{host}:{port}/v1", _FakeHandler
    server.shutdown()
    thread.join(timeout=2)


def test_evidence_status_distinctions():
    s = _sample(
        "e1",
        transcripts={
            "qwen_1": {"text": "你好", "status": "success"},
            "qwen_2": {"text": "", "status": "success"},
            "glm_1": {"text": "残留", "status": "failed"},
            "glm_2": {"text": "x"},  # missing key sensevoice
        },
    )
    from audio_engine.core.semantic.config import EvidenceConfig

    evidence = EvidenceConfig(
        model_families={
            "qwen": ["qwen_1", "qwen_2"],
            "glm": ["glm_1", "glm_2"],
            "sensevoice": ["sensevoice_1"],
        }
    )
    families, summary = build_families_context(s, evidence)
    assert families["qwen"]["runs"][0]["status"] == RUN_SUCCESS_TEXT
    assert families["qwen"]["runs"][1]["status"] == RUN_SUCCESS_EMPTY
    assert families["glm"]["runs"][0]["status"] == RUN_FAILED
    assert families["glm"]["runs"][0]["text"] is None  # residual not evidence
    assert families["sensevoice"]["runs"][0]["status"] == RUN_MISSING
    assert summary["success_text_runs"] == 2


def test_preflight_duplicate_ids():
    with pytest.raises(ValueError, match="duplicate id"):
        preflight_samples([_sample("a"), _sample("a")])


def test_concurrency_budget_rejected():
    with pytest.raises(ValueError, match="budget exceeded"):
        validate_concurrency_budget(
            max_concurrency_total=8,
            workers_per_shard=4,
            parallel_shards=4,
        )


def test_mock_two_tasks_coexist(tmp_path: Path, service_yaml: Path):
    conflict = _write_task(tmp_path, service_yaml, task_id="semantic_conflict_v1", base=CONFLICT_TASK)
    intent = _write_task(tmp_path, service_yaml, task_id="intent_v1", base=INTENT_TASK)
    sample = _sample("s1")
    op = SemanticJudgeOperator()
    cfg1 = OperatorConfig(
        params={"task_config": str(conflict)},
        cache_dir=tmp_path / "cache",
        mock=True,
    )
    cfg2 = OperatorConfig(
        params={"task_config": str(intent)},
        cache_dir=tmp_path / "cache",
        mock=True,
    )
    r1 = op.process(sample, cfg1)
    r2 = op.process(r1.sample, cfg2)
    results = r2.sample.labels["semantic_results"]
    assert "semantic_conflict_v1" in results
    assert "intent_v1" in results
    assert results["semantic_conflict_v1"]["status"] == "mock"
    assert results["intent_v1"]["status"] == "mock"
    assert results["intent_v1"]["result"]["intent"] in {
        "agree",
        "disagree",
        "ask_clarify",
        "request_action",
        "provide_info",
        "other",
        "unknown",
    }


def test_fake_http_concurrency_and_order(tmp_path: Path, service_yaml: Path, fake_server):
    api_base, handler = fake_server
    handler.delay_s = 0.08
    # Patch service api_base
    raw = yaml.safe_load(service_yaml.read_text(encoding="utf-8"))
    raw["api_base"] = api_base
    raw["max_concurrency_total"] = 4
    raw["structured_output_mode"] = "prompt_schema_validate"
    service_yaml.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    task = _write_task(tmp_path, service_yaml, task_id="semantic_conflict_v1", base=CONFLICT_TASK)

    samples = [_sample(f"s{i}") for i in range(12)]
    for s in samples:
        handler.responses[s.id] = {
            "conflict": s.id.endswith("1") or s.id.endswith("3"),
            "insufficient_evidence": False,
            "summary": s.id,
            "differing_aspects": ["intent"] if s.id.endswith("1") else ["none"],
        }

    in_path = tmp_path / "in.parquet"
    out_path = tmp_path / "out.parquet"
    Manifest(samples).save(in_path)

    cfg = PipelineConfig(
        name="semantic_judge_test",
        input_manifest=str(in_path),
        output_manifest=str(out_path),
        steps=[
            PipelineStep(
                name="semantic_conflict",
                operator="semantic.judge",
                params={"task_config": str(task)},
            )
        ],
        output_dir=tmp_path / "derived",
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs",
        execution=ExecutionConfig(executor="thread", workers=4, max_in_flight=8, checkpoint_every=0),
    )
    result = PipelineRunner(cfg).run()
    assert len(result.samples) == 12
    assert [s.id for s in result.samples] == [s.id for s in samples]
    assert handler.peak_active > 1
    assert handler.peak_active <= 4
    for s in result.samples:
        rec = s.labels["semantic_results"]["semantic_conflict_v1"]
        assert rec["status"] == "success"
        assert rec["result"]["summary"] == s.id


def test_retry_429_then_success(tmp_path: Path, service_yaml: Path, fake_server):
    api_base, handler = fake_server
    raw = yaml.safe_load(service_yaml.read_text(encoding="utf-8"))
    raw["api_base"] = api_base
    raw["retry"] = {
        "max_attempts": 3,
        "connect_timeout_s": 1,
        "read_timeout_s": 2,
        "request_timeout_s": 10,
        "backoff_base_s": 0.01,
        "backoff_max_s": 0.05,
    }
    service_yaml.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    task = _write_task(tmp_path, service_yaml, task_id="semantic_conflict_v1", base=CONFLICT_TASK)
    handler.fail_codes["s429"] = [429, 200]  # second attempt: wait, 200 is wrong path
    # Actually fail_codes pops and returns HTTP code; after 429 we need success on next.
    # Fix: only queue 429 once.
    handler.fail_codes["s429"] = [429]

    op = SemanticJudgeOperator()
    # Clear class runtime between tests
    SemanticJudgeOperator._runtime.clear()
    SemanticJudgeOperator._bundle_cache.clear()
    cfg = OperatorConfig(
        params={"task_config": str(task), "workers": 2},
        cache_dir=tmp_path / "cache",
        mock=False,
    )
    result = op.process(_sample("s429"), cfg)
    rec = result.sample.labels["semantic_results"]["semantic_conflict_v1"]
    assert rec["status"] == "success"
    assert rec["attempts"] >= 2


def test_auth_error_not_endless_retry(tmp_path: Path, service_yaml: Path, fake_server):
    api_base, handler = fake_server
    raw = yaml.safe_load(service_yaml.read_text(encoding="utf-8"))
    raw["api_base"] = api_base
    raw["retry"]["max_attempts"] = 5
    service_yaml.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    task = _write_task(tmp_path, service_yaml, task_id="semantic_conflict_v1", base=CONFLICT_TASK)
    handler.fail_codes["s401"] = [401, 401, 401, 401, 401]
    SemanticJudgeOperator._runtime.clear()
    SemanticJudgeOperator._bundle_cache.clear()
    op = SemanticJudgeOperator()
    cfg = OperatorConfig(
        params={"task_config": str(task), "workers": 1},
        cache_dir=tmp_path / "cache",
    )
    result = op.process(_sample("s401"), cfg)
    rec = result.sample.labels["semantic_results"]["semantic_conflict_v1"]
    assert rec["status"] == "request_error"
    assert result.sample.status.get("semantic.judge") == "failed"
    assert rec["attempts"] == 1 or "401" in str(rec["error_message"])


def test_cache_invalidates_on_transcript_change(tmp_path: Path, service_yaml: Path):
    task = _write_task(tmp_path, service_yaml, task_id="semantic_conflict_v1", base=CONFLICT_TASK)
    SemanticJudgeOperator._runtime.clear()
    SemanticJudgeOperator._bundle_cache.clear()
    op = SemanticJudgeOperator()
    cfg = OperatorConfig(params={"task_config": str(task)}, cache_dir=tmp_path / "cache", mock=True)
    s1 = _sample("c1")
    r1 = op.process(s1, cfg)
    assert not r1.cache_hit
    r2 = op.process(s1, cfg)
    assert r2.skipped or r2.cache_hit
    s2 = _sample("c1")
    s2.transcripts["glm_1"] = {"text": "完全不同", "status": "success"}
    # should_skip false due to fingerprint change
    assert not op.should_skip(s2, cfg)
    key1 = semantic_cache_key(s1, freeze_bundle(task), mock=True)
    key2 = semantic_cache_key(s2, freeze_bundle(task), mock=True)
    assert key1 != key2


def test_resume_retries_failed_rows(tmp_path: Path, service_yaml: Path, fake_server):
    api_base, handler = fake_server
    raw = yaml.safe_load(service_yaml.read_text(encoding="utf-8"))
    raw["api_base"] = api_base
    service_yaml.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    task = _write_task(tmp_path, service_yaml, task_id="semantic_conflict_v1", base=CONFLICT_TASK)

    samples = [_sample("ok1"), _sample("bad1"), _sample("ok2")]
    handler.fail_codes["bad1"] = [500, 500, 500]
    in_path = tmp_path / "in.parquet"
    out_path = tmp_path / "out.parquet"
    Manifest(samples).save(in_path)

    SemanticJudgeOperator._runtime.clear()
    SemanticJudgeOperator._bundle_cache.clear()
    # Keep paths short on Windows (atomic_path double-tmp can exceed MAX_PATH).
    runs_dir = tmp_path / "r"
    cfg = PipelineConfig(
        name="sem",
        input_manifest=str(in_path),
        output_manifest=str(out_path),
        steps=[
            PipelineStep(
                name="judge",
                operator="semantic.judge",
                params={"task_config": str(task)},
            )
        ],
        output_dir=tmp_path / "d",
        cache_dir=tmp_path / "c",
        runs_dir=runs_dir,
        execution=ExecutionConfig(
            executor="sequential", workers=1, checkpoint_every=10, fail_fast=False
        ),
    )
    runner = PipelineRunner(cfg)
    first = runner.run()
    run_dir = runner.run_dir
    statuses = {
        s.id: s.labels["semantic_results"]["semantic_conflict_v1"]["status"]
        for s in first.samples
    }
    assert statuses["ok1"] == "success"
    assert statuses["ok2"] == "success"
    assert statuses["bad1"] == "request_error"

    # Clear forced failures and resume.
    handler.fail_codes.pop("bad1", None)
    SemanticJudgeOperator._runtime.clear()
    cfg2 = PipelineConfig(
        name="sem",
        input_manifest=str(in_path),
        output_manifest=str(out_path),
        steps=cfg.steps,
        output_dir=tmp_path / "d",
        cache_dir=tmp_path / "c",
        runs_dir=runs_dir,
        resume=str(run_dir),
        execution=cfg.execution,
    )
    second = PipelineRunner(cfg2).run()
    fixed = {
        s.id: s.labels["semantic_results"]["semantic_conflict_v1"]["status"]
        for s in second.samples
    }
    assert fixed["bad1"] == "success"
    assert [s.id for s in second.samples] == ["ok1", "bad1", "ok2"]


def test_parquet_roundtrip_preserves_lineage(tmp_path: Path, service_yaml: Path):
    task = _write_task(tmp_path, service_yaml, task_id="semantic_conflict_v1", base=CONFLICT_TASK)
    SemanticJudgeOperator._runtime.clear()
    SemanticJudgeOperator._bundle_cache.clear()
    sample = _sample("p1")
    sample.labels["existing_tag"] = {"ok": True}
    op = SemanticJudgeOperator()
    cfg = OperatorConfig(params={"task_config": str(task)}, cache_dir=tmp_path / "cache", mock=True)
    updated = op.process(sample, cfg).sample
    path = tmp_path / "round.parquet"
    Manifest([updated]).save(path)
    loaded = Manifest.load(path).samples[0]
    assert loaded.labels["existing_tag"]["ok"] is True
    assert "semantic_conflict_v1" in loaded.labels["semantic_results"]
    assert loaded.transcripts["qwen_1"]["text"] == "我不要"
    assert loaded.labels.get("segment_pcm_sha256") == "seg-p1"


def test_mock_not_reused_in_real_mode(tmp_path: Path, service_yaml: Path, fake_server):
    api_base, handler = fake_server
    raw = yaml.safe_load(service_yaml.read_text(encoding="utf-8"))
    raw["api_base"] = api_base
    service_yaml.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    task = _write_task(tmp_path, service_yaml, task_id="semantic_conflict_v1", base=CONFLICT_TASK)
    SemanticJudgeOperator._runtime.clear()
    SemanticJudgeOperator._bundle_cache.clear()
    op = SemanticJudgeOperator()
    sample = _sample("m1")
    mock_cfg = OperatorConfig(
        params={"task_config": str(task)}, cache_dir=tmp_path / "cache", mock=True
    )
    mocked = op.process(sample, mock_cfg).sample
    assert mocked.labels["semantic_results"]["semantic_conflict_v1"]["status"] == "mock"

    real_cfg = OperatorConfig(
        params={"task_config": str(task), "workers": 1},
        cache_dir=tmp_path / "cache",
        mock=False,
    )
    assert op.should_retry_on_resume(mocked, real_cfg)
    assert not op.should_skip(mocked, real_cfg)
    live = op.process(mocked, real_cfg).sample
    assert live.labels["semantic_results"]["semantic_conflict_v1"]["status"] == "success"
    assert live.labels["semantic_results"]["semantic_conflict_v1"]["mock"] is False


def test_gate_shared_by_retries():
    gate = ConcurrencyGate(2)
    active = []

    def worker():
        with gate:
            active.append(gate.active)
            time.sleep(0.05)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert max(active) <= 2
    assert gate.peak <= 2
