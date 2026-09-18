"""Qwen3-ASR vLLM launch adapter (verified baseline from 2026-09-16)."""

from __future__ import annotations

from pathlib import Path

from audio_engine.core.stage1.adapters.base import (
    LaunchPlan,
    client_api_base,
    ensure_path_exists,
)
from audio_engine.core.stage1.identity import assert_served_model, fetch_model_identity
from audio_engine.core.stage1.process import (
    ServiceSession,
    build_cuda_env,
    start_owned_process,
    terminate_owned_process,
    wait_for_http_ready,
)
from audio_engine.core.stage1.probe import ProbeResult, probe_vllm_transcription
from audio_engine.core.stage1.runtime_config import Stage1RuntimeConfig


class QwenAdapter:
    family = "qwen"

    def check_paths(self, runtime: Stage1RuntimeConfig) -> list[str]:
        cfg = runtime.qwen
        errors: list[str] = []
        ensure_path_exists(cfg.model_path, "families.qwen.model_path", errors)
        ensure_path_exists(cfg.chat_template, "families.qwen.chat_template", errors)
        if cfg.vllm_bin is None:
            errors.append("families.qwen.vllm_bin 未配置")
        else:
            ensure_path_exists(cfg.vllm_bin, "families.qwen.vllm_bin", errors)
        return errors

    def plan(
        self,
        runtime: Stage1RuntimeConfig,
        *,
        gpu: int | str,
        port: int | None = None,
    ) -> LaunchPlan:
        runtime.require_ready(families=["qwen"])
        runtime.require_authorized_gpu(gpu)
        cfg = runtime.qwen
        assert cfg.vllm_bin is not None
        bind_port = int(port if port is not None else cfg.port)
        argv = [
            str(cfg.vllm_bin),
            "serve",
            str(cfg.model_path),
            "--gpu-memory-utilization",
            f"{cfg.gpu_memory_utilization:.2f}",
            "--host",
            cfg.host,
            "--port",
            str(bind_port),
            "--tensor-parallel-size",
            str(cfg.tensor_parallel_size),
            "--served-model-name",
            cfg.served_model_name,
            "--chat-template",
            str(cfg.chat_template),
        ]
        env = build_cuda_env(gpu)
        api_base = client_api_base(cfg.host, bind_port)
        client_env = {
            "QWEN_ASR_API_BASE": api_base,
            "QWEN_ASR_API_KEY": cfg.api_key,
            "QWEN_ASR_MODEL": cfg.served_model_name,
        }
        return LaunchPlan(
            family=self.family,
            argv=argv,
            env={"CUDA_VISIBLE_DEVICES": str(gpu)},
            client_env=client_env,
            api_base=api_base,
            port=bind_port,
            gpu=gpu,
            served_model_name=cfg.served_model_name,
            notes=[
                "保留 Qwen chat-template；勿套用到 GLM",
                f"子进程 CUDA_VISIBLE_DEVICES={gpu}",
            ],
            kind="vllm",
        )

    def start(
        self,
        runtime: Stage1RuntimeConfig,
        *,
        gpu: int | str,
        port: int | None = None,
        session_dir: Path,
        attach_existing: bool = False,
        dry_run: bool = False,
    ) -> ServiceSession:
        plan = self.plan(runtime, gpu=gpu, port=port)
        path_errors = self.check_paths(runtime)
        if path_errors and not attach_existing and not dry_run:
            raise FileNotFoundError("Qwen 启动路径检查失败:\n- " + "\n- ".join(path_errors))

        session_dir = Path(session_dir)
        session_dir.mkdir(parents=True, exist_ok=True)
        assert plan.api_base is not None

        if attach_existing:
            identity = assert_served_model(plan.api_base, runtime.qwen.served_model_name)
            session = ServiceSession(
                family=self.family,
                gpu=gpu,
                port=plan.port,
                api_base=plan.api_base,
                served_model_name=runtime.qwen.served_model_name,
                owned=False,
                client_env=plan.client_env,
                argv=plan.argv,
                attached=True,
                kind="vllm",
                extra={"model_ids": list(identity.model_ids)},
                session_dir=str(session_dir),
            )
            session.save(session_dir / "session.json")
            return session

        if dry_run:
            session = ServiceSession(
                family=self.family,
                gpu=gpu,
                port=plan.port,
                api_base=plan.api_base,
                served_model_name=runtime.qwen.served_model_name,
                owned=False,
                client_env=plan.client_env,
                argv=plan.argv,
                kind="vllm",
                extra={"dry_run": True},
                session_dir=str(session_dir),
            )
            session.save(session_dir / "session.json")
            return session

        # Refuse attach-by-accident: if something already answers with wrong identity, fail.
        try:
            existing = fetch_model_identity(plan.api_base, timeout_s=2.0)
        except Exception:
            existing = None
        if existing is not None:
            if existing.matches(runtime.qwen.served_model_name):
                raise RuntimeError(
                    f"端口 {plan.port} 已有匹配的 Qwen 服务；"
                    "请使用 --attach-existing 显式接入，避免重复启动或误杀"
                )
            raise RuntimeError(
                f"端口 {plan.port} 已被其他模型占用: ids={list(existing.model_ids)!r}"
            )

        full_env = build_cuda_env(gpu)
        log_path = session_dir / "vllm.log"
        managed = start_owned_process(plan.argv, env=full_env, log_path=log_path)
        session = ServiceSession(
            family=self.family,
            gpu=gpu,
            port=plan.port,
            api_base=plan.api_base,
            served_model_name=runtime.qwen.served_model_name,
            owned=True,
            client_env=plan.client_env,
            argv=plan.argv,
            pid=managed.pid,
            pgid=managed.pgid,
            log_path=str(log_path),
            session_dir=str(session_dir),
            kind="vllm",
        )
        session.save(session_dir / "session.json")
        try:
            wait_for_http_ready(
                plan.api_base,
                timeout_s=runtime.ready_timeout_s,
                poll_s=runtime.ready_poll_s,
                predicate=lambda url: assert_served_model(
                    url, runtime.qwen.served_model_name, timeout_s=5.0
                ),
            )
        except Exception:
            self.stop(session)
            raise
        session.save(session_dir / "session.json")
        return session

    def probe(
        self,
        runtime: Stage1RuntimeConfig,
        *,
        session: ServiceSession | None = None,
        api_base: str | None = None,
        audio_path: str | Path,
        gpu: int | str | None = None,
    ) -> ProbeResult:
        base = api_base or (session.api_base if session else None)
        if not base:
            raise ValueError("Qwen 探针需要 api_base 或已启动 session")
        assert_served_model(base, runtime.qwen.served_model_name)
        return probe_vllm_transcription(
            family=self.family,
            audio_path=audio_path,
            api_base=base,
            model=runtime.qwen.served_model_name,
            api_key=runtime.qwen.api_key,
        )

    def stop(self, session: ServiceSession) -> None:
        if not session.owned:
            return
        terminate_owned_process(pid=session.pid, pgid=session.pgid)
