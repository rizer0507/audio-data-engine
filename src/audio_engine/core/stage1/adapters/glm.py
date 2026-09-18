"""GLM-ASR vLLM launch adapter (independent env + FlashInfer baseline)."""

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


class GlmAdapter:
    family = "glm"

    def check_paths(self, runtime: Stage1RuntimeConfig) -> list[str]:
        cfg = runtime.glm
        errors: list[str] = []
        ensure_path_exists(cfg.model_path, "families.glm.model_path", errors)
        ensure_path_exists(cfg.python_bin, "families.glm.python_bin", errors)
        ensure_path_exists(cfg.vllm_bin, "families.glm.vllm_bin", errors)
        return errors

    def plan(
        self,
        runtime: Stage1RuntimeConfig,
        *,
        gpu: int | str,
        port: int | None = None,
    ) -> LaunchPlan:
        runtime.require_ready(families=["glm"])
        runtime.require_authorized_gpu(gpu)
        cfg = runtime.glm
        bind_port = int(port if port is not None else cfg.port)
        argv = [
            str(cfg.python_bin),
            str(cfg.vllm_bin),
            "serve",
            str(cfg.model_path),
            "--host",
            cfg.host,
            "--port",
            str(bind_port),
            "--served-model-name",
            cfg.served_model_name,
            "--tensor-parallel-size",
            str(cfg.tensor_parallel_size),
        ]
        if cfg.trust_remote_code:
            argv.append("--trust-remote-code")
        argv.extend(
            [
                "--dtype",
                cfg.dtype,
                "--max-model-len",
                str(cfg.max_model_len),
                "--max-num-seqs",
                str(cfg.max_num_seqs),
                "--gpu-memory-utilization",
                f"{cfg.gpu_memory_utilization:.2f}",
                "--limit-mm-per-prompt",
                cfg.limit_mm_per_prompt,
            ]
        )
        if cfg.no_enable_flashinfer_autotune:
            argv.append("--no-enable-flashinfer-autotune")
        argv.extend(["--kernel-config", cfg.kernel_config])

        # Launch env overlays only (CUDA injected separately); keep baseline keys.
        launch_env = {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            **cfg.env,
        }
        api_base = f"http://{cfg.client_host}:{bind_port}"
        # Prefer client_host; also normalize 0.0.0.0 bind for display consistency.
        _ = client_api_base(cfg.host, bind_port)
        client_env = {
            "GLM_ASR_API_BASE": api_base,
            "GLM_ASR_MODEL": cfg.served_model_name,
            "GLM_ASR_API_KEY": cfg.api_key,
        }
        return LaunchPlan(
            family=self.family,
            argv=argv,
            env=launch_env,
            client_env=client_env,
            api_base=api_base,
            port=bind_port,
            gpu=gpu,
            served_model_name=cfg.served_model_name,
            notes=[
                "使用独立 env_root 的 python+vllm；PYTHONNOUSERSITE / FlashInfer SAMPLER=0",
                "禁止附加 Qwen chat-template",
                f"启动前清除: {', '.join(cfg.unset_env)}",
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
            raise FileNotFoundError("GLM 启动路径检查失败:\n- " + "\n- ".join(path_errors))

        session_dir = Path(session_dir)
        session_dir.mkdir(parents=True, exist_ok=True)
        assert plan.api_base is not None
        cfg = runtime.glm

        if attach_existing:
            identity = assert_served_model(plan.api_base, cfg.served_model_name)
            session = ServiceSession(
                family=self.family,
                gpu=gpu,
                port=plan.port,
                api_base=plan.api_base,
                served_model_name=cfg.served_model_name,
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
                served_model_name=cfg.served_model_name,
                owned=False,
                client_env=plan.client_env,
                argv=plan.argv,
                kind="vllm",
                extra={"dry_run": True, "unset_env": list(cfg.unset_env)},
                session_dir=str(session_dir),
            )
            session.save(session_dir / "session.json")
            return session

        try:
            existing = fetch_model_identity(plan.api_base, timeout_s=2.0)
        except Exception:
            existing = None
        if existing is not None:
            if existing.matches(cfg.served_model_name):
                raise RuntimeError(
                    f"端口 {plan.port} 已有匹配的 GLM 服务；"
                    "请使用 --attach-existing 显式接入，避免重复启动或误杀"
                )
            raise RuntimeError(
                f"端口 {plan.port} 已被其他模型占用: ids={list(existing.model_ids)!r}"
            )

        full_env = build_cuda_env(
            gpu,
            overlays=cfg.env,
            unset=cfg.unset_env,
        )
        log_path = session_dir / "vllm.log"
        managed = start_owned_process(plan.argv, env=full_env, log_path=log_path)
        session = ServiceSession(
            family=self.family,
            gpu=gpu,
            port=plan.port,
            api_base=plan.api_base,
            served_model_name=cfg.served_model_name,
            owned=True,
            client_env=plan.client_env,
            argv=plan.argv,
            pid=managed.pid,
            pgid=managed.pgid,
            log_path=str(log_path),
            session_dir=str(session_dir),
            kind="vllm",
            extra={"unset_env": list(cfg.unset_env)},
        )
        session.save(session_dir / "session.json")
        try:
            wait_for_http_ready(
                plan.api_base,
                timeout_s=runtime.ready_timeout_s,
                poll_s=runtime.ready_poll_s,
                predicate=lambda url: assert_served_model(
                    url, cfg.served_model_name, timeout_s=5.0
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
            raise ValueError("GLM 探针需要 api_base 或已启动 session")
        assert_served_model(base, runtime.glm.served_model_name)
        return probe_vllm_transcription(
            family=self.family,
            audio_path=audio_path,
            api_base=base,
            model=runtime.glm.served_model_name,
            api_key=runtime.glm.api_key,
        )

    def stop(self, session: ServiceSession) -> None:
        if not session.owned:
            return
        terminate_owned_process(pid=session.pid, pgid=session.pgid)
