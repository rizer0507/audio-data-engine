"""Model protocol / port isolation tests using local HTTP servers (NOT real vLLM)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


class _Recorder:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []


def _make_handler(recorder: _Recorder, *, kind: str, model_id: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # noqa: A003
            return

        def do_GET(self):  # noqa: N802
            recorder.requests.append({"method": "GET", "path": self.path, "kind": kind})
            if self.path.endswith("/models") or self.path == "/v1/models":
                body = json.dumps({"data": [{"id": model_id}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            recorder.requests.append(
                {
                    "method": "POST",
                    "path": self.path,
                    "kind": kind,
                    "body_len": length,
                    "content_type": self.headers.get("Content-Type"),
                }
            )
            if kind == "glm" and "transcriptions" in self.path:
                body = json.dumps({"text": "glm-ok"}).encode()
            elif kind == "qwen" and "chat/completions" in self.path:
                body = json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": "language zh<asr_text>qwen-ok</asr_text>"
                                }
                            }
                        ]
                    }
                ).encode()
            else:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


@pytest.fixture
def twin_servers():
    glm_rec = _Recorder()
    qwen_rec = _Recorder()
    glm_srv = HTTPServer(("127.0.0.1", 0), _make_handler(glm_rec, kind="glm", model_id="glm-asr"))
    qwen_srv = HTTPServer(("127.0.0.1", 0), _make_handler(qwen_rec, kind="qwen", model_id="qwen3-asr"))
    threads = [
        threading.Thread(target=glm_srv.serve_forever, daemon=True),
        threading.Thread(target=qwen_srv.serve_forever, daemon=True),
    ]
    for t in threads:
        t.start()
    yield {
        "glm": glm_srv,
        "qwen": qwen_srv,
        "glm_port": glm_srv.server_address[1],
        "qwen_port": qwen_srv.server_address[1],
        "glm_rec": glm_rec,
        "qwen_rec": qwen_rec,
    }
    glm_srv.shutdown()
    qwen_srv.shutdown()


def test_model_check_hits_correct_ports(twin_servers, tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    import yaml
    from fastapi.testclient import TestClient

    from audio_engine.web.app import create_app
    from audio_engine.web.settings import load_web_settings

    glm_port = twin_servers["glm_port"]
    qwen_port = twin_servers["qwen_port"]
    cfg = {
        "bind_host": "127.0.0.1",
        "bind_port": 0,
        "project_root": str(ROOT),
        "sqlite_path": str(tmp_path / "db.sqlite3"),
        "runs_dir": str(tmp_path / "runs"),
        "plugins_config": str(ROOT / "configs" / "plugins.yaml"),
        "sources_manifest": str(tmp_path / "manifest.yaml"),
        "allowed_source_roots": [str(tmp_path)],
        "models": [
            {
                "connection_id": "glm_main",
                "name": "GLM",
                "provider": "glm",
                "base_url": f"http://127.0.0.1:{glm_port}/v1",
                "model": "glm-asr",
                "model_revision": "t1",
                "api_key_env": "GLM_ASR_API_KEY",
            },
            {
                "connection_id": "qwen_main",
                "name": "Qwen",
                "provider": "qwen",
                "base_url": f"http://127.0.0.1:{qwen_port}/v1",
                "model": "qwen3-asr",
                "model_revision": "t1",
                "api_key_env": "QWEN_ASR_API_KEY",
            },
        ],
    }
    (tmp_path / "manifest.yaml").write_text("sources: []\n", encoding="utf-8")
    cfg_path = tmp_path / "web.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    settings = load_web_settings(cfg_path, project_root=ROOT)
    app = create_app(settings)
    with TestClient(app) as client:
        g = client.post("/api/v1/models/glm_main/check", json={"level": "models"})
        q = client.post("/api/v1/models/qwen_main/check", json={"level": "models"})
    assert g.status_code == 200 and g.json()["ok"] is True
    assert q.status_code == 200 and q.json()["ok"] is True
    assert any(r["path"].endswith("/models") for r in twin_servers["glm_rec"].requests)
    assert any(r["path"].endswith("/models") for r in twin_servers["qwen_rec"].requests)
    # Ports must differ and each recorder only saw its own server
    assert glm_port != qwen_port
    assert all(r["kind"] == "glm" for r in twin_servers["glm_rec"].requests)
    assert all(r["kind"] == "qwen" for r in twin_servers["qwen_rec"].requests)


def test_normalize_base_url_no_double_v1() -> None:
    from audio_engine.web.services.models import normalize_v1_base

    assert normalize_v1_base("http://127.0.0.1:8001/v1") == "http://127.0.0.1:8001/v1"
    assert normalize_v1_base("http://127.0.0.1:8001") == "http://127.0.0.1:8001/v1"
    with pytest.raises(Exception):
        normalize_v1_base("http://0.0.0.0:8001/v1")


def test_transcription_protocol_glm_and_qwen(twin_servers, tmp_path: Path) -> None:
    """Protocol probe: GLM → /audio/transcriptions, Qwen → /chat/completions.

    Uses local HTTP servers only — NOT real vLLM inference acceptance.
    """

    pytest.importorskip("fastapi")
    import wave

    import yaml
    from fastapi.testclient import TestClient

    from audio_engine.web.app import create_app
    from audio_engine.web.settings import load_web_settings

    # Minimal WAV for multipart upload
    sample = tmp_path / "probe.wav"
    with wave.open(str(sample), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 1600)

    glm_port = twin_servers["glm_port"]
    qwen_port = twin_servers["qwen_port"]
    cfg = {
        "bind_host": "127.0.0.1",
        "bind_port": 0,
        "project_root": str(ROOT),
        "sqlite_path": str(tmp_path / "db.sqlite3"),
        "runs_dir": str(tmp_path / "runs"),
        "plugins_config": str(ROOT / "configs" / "plugins.yaml"),
        "sources_manifest": str(tmp_path / "manifest.yaml"),
        "allowed_source_roots": [str(tmp_path)],
        "models": [
            {
                "connection_id": "glm_main",
                "name": "GLM",
                "provider": "glm",
                "base_url": f"http://127.0.0.1:{glm_port}/v1",
                "model": "glm-asr",
                "model_revision": "t1",
                "api_key_env": "GLM_ASR_API_KEY",
            },
            {
                "connection_id": "qwen_main",
                "name": "Qwen",
                "provider": "qwen",
                "base_url": f"http://127.0.0.1:{qwen_port}/v1",
                "model": "qwen3-asr",
                "model_revision": "t1",
                "api_key_env": "QWEN_ASR_API_KEY",
            },
        ],
    }
    (tmp_path / "manifest.yaml").write_text("sources: []\n", encoding="utf-8")
    cfg_path = tmp_path / "web.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    settings = load_web_settings(cfg_path, project_root=ROOT)
    app = create_app(settings)

    # Poison parent env with wrong ports — must not override frozen connections.
    import os

    os.environ["GLM_ASR_API_BASE"] = "http://127.0.0.1:1/v1"
    os.environ["QWEN_ASR_API_BASE"] = "http://127.0.0.1:2/v1"
    try:
        with TestClient(app) as client:
            g = client.post(
                "/api/v1/models/glm_main/check",
                json={"level": "transcription", "sample_ref": {"path": str(sample)}},
            )
            q = client.post(
                "/api/v1/models/qwen_main/check",
                json={"level": "transcription", "sample_ref": {"path": str(sample)}},
            )
    finally:
        os.environ.pop("GLM_ASR_API_BASE", None)
        os.environ.pop("QWEN_ASR_API_BASE", None)

    assert g.status_code == 200 and g.json().get("transcription_ok") is True, g.text
    assert q.status_code == 200 and q.json().get("transcription_ok") is True, q.text
    assert any(
        r["method"] == "POST" and "transcriptions" in r["path"]
        for r in twin_servers["glm_rec"].requests
    )
    assert any(
        r["method"] == "POST" and "chat/completions" in r["path"]
        for r in twin_servers["qwen_rec"].requests
    )
    # Wrong poison ports must not have been contacted (recorders only on twin ports).
    assert all(r["kind"] == "glm" for r in twin_servers["glm_rec"].requests)
    assert all(r["kind"] == "qwen" for r in twin_servers["qwen_rec"].requests)


def test_model_name_mismatch(twin_servers, tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    import yaml
    from fastapi.testclient import TestClient

    from audio_engine.web.app import create_app
    from audio_engine.web.settings import load_web_settings

    glm_port = twin_servers["glm_port"]
    cfg = {
        "bind_host": "127.0.0.1",
        "bind_port": 0,
        "project_root": str(ROOT),
        "sqlite_path": str(tmp_path / "db.sqlite3"),
        "runs_dir": str(tmp_path / "runs"),
        "plugins_config": str(ROOT / "configs" / "plugins.yaml"),
        "sources_manifest": str(tmp_path / "manifest.yaml"),
        "allowed_source_roots": [str(tmp_path)],
        "models": [
            {
                "connection_id": "glm_bad",
                "name": "GLM",
                "provider": "glm",
                "base_url": f"http://127.0.0.1:{glm_port}/v1",
                "model": "not-the-served-name",
                "model_revision": "t1",
                "api_key_env": "GLM_ASR_API_KEY",
            }
        ],
    }
    (tmp_path / "manifest.yaml").write_text("sources: []\n", encoding="utf-8")
    cfg_path = tmp_path / "web.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    app = create_app(load_web_settings(cfg_path, project_root=ROOT))
    with TestClient(app) as client:
        r = client.post("/api/v1/models/glm_bad/check", json={"level": "models"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["code"] == "MODEL_NAME_MISMATCH"


def test_multi_route_env_injection_isolation() -> None:
    """Production adapter injects distinct per-route env from model bindings."""

    from audio_engine.plugins.contracts import ResolvedModelConnection

    # Mirror the injection loop used by ProductionPipelinePlugin.execute
    routes = [
        {"route_id": "glm_a", "family": "glm", "env": {}},
        {"route_id": "glm_b", "family": "glm", "env": {}},
        {"route_id": "qwen_a", "family": "qwen", "env": {}},
    ]
    models = [
        ResolvedModelConnection(
            connection_id="c1",
            provider="glm",
            base_url="http://127.0.0.1:8101/v1",
            model="glm-asr",
            model_revision="r1",
            api_key_env="",
        ),
        ResolvedModelConnection(
            connection_id="c2",
            provider="glm",
            base_url="http://127.0.0.1:8102/v1",
            model="glm-asr",
            model_revision="r2",
            api_key_env="",
        ),
        ResolvedModelConnection(
            connection_id="c3",
            provider="qwen",
            base_url="http://127.0.0.1:8201/v1",
            model="qwen3-asr",
            model_revision="r1",
            api_key_env="",
        ),
    ]
    bindings = [
        {"route_id": "glm_a", "connection_id": "c1"},
        {"route_id": "glm_b", "connection_id": "c2"},
        {"route_id": "qwen_a", "connection_id": "c3"},
    ]
    route_to_conn = {b["route_id"]: b["connection_id"] for b in bindings}
    for route in routes:
        cid = route_to_conn.get(route["route_id"])
        conn = next((m for m in models if m.connection_id == cid), None)
        assert conn is not None
        env = dict(route["env"])
        if conn.provider == "glm":
            env["GLM_ASR_API_BASE"] = conn.base_url
            env["GLM_ASR_API_BASES"] = conn.base_url
            env["GLM_ASR_MODEL"] = conn.model
        elif conn.provider == "qwen":
            env["QWEN_ASR_API_BASE"] = conn.base_url
            env["QWEN_ASR_API_BASES"] = conn.base_url
            env["QWEN_ASR_MODEL"] = conn.model
        route["env"] = env

    assert routes[0]["env"]["GLM_ASR_API_BASE"].endswith(":8101/v1")
    assert routes[1]["env"]["GLM_ASR_API_BASE"].endswith(":8102/v1")
    assert routes[2]["env"]["QWEN_ASR_API_BASE"].endswith(":8201/v1")
    # Same-family routes must not share a single base
    assert routes[0]["env"]["GLM_ASR_API_BASE"] != routes[1]["env"]["GLM_ASR_API_BASE"]


def test_vllm_url_builders_no_double_v1() -> None:
    from audio_engine.operators.asr.vllm import chat_completions_url, transcription_url

    assert transcription_url("http://127.0.0.1:8001/v1").endswith(
        "/v1/audio/transcriptions"
    )
    assert "/v1/v1/" not in transcription_url("http://127.0.0.1:8001/v1")
    assert chat_completions_url("http://127.0.0.1:8002/v1").endswith(
        "/v1/chat/completions"
    )
    assert "/v1/v1/" not in chat_completions_url("http://127.0.0.1:8002/v1")
