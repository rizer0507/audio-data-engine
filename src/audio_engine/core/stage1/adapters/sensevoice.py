"""SenseVoice local-load adapter (no vLLM daemon)."""

from __future__ import annotations

import os
from pathlib import Path

from audio_engine.core.stage1.adapters.base import LaunchPlan, ensure_path_exists
from audio_engine.core.stage1.process import ServiceSession, build_cuda_env
from audio_engine.core.stage1.probe import ProbeResult, require_probe_audio
from audio_engine.core.stage1.runtime_config import Stage1RuntimeConfig


class SenseVoiceAdapter:
    family = "sensevoice"

    def check_paths(self, runtime: Stage1RuntimeConfig) -> list[str]:
        cfg = runtime.sensevoice
        errors: list[str] = []
        ensure_path_exists(cfg.model_path, "families.sensevoice.model_path", errors)
        if runtime.engine_python is None:
            errors.append("engine_python 未配置（SenseVoice 随引擎解释器本地加载）")
        elif not runtime.engine_python.exists():
            errors.append(f"engine_python 不存在: {runtime.engine_python}")
        return errors

    def plan(
        self,
        runtime: Stage1RuntimeConfig,
        *,
        gpu: int | str,
        port: int | None = None,
    ) -> LaunchPlan:
        runtime.require_ready(families=["sensevoice"])
        runtime.require_authorized_gpu(gpu)
        cfg = runtime.sensevoice
        client_env = {
            "SENSEVOICE_MODEL_PATH": str(cfg.model_path),
            "CUDA_VISIBLE_DEVICES": str(gpu),
        }
        # Local load: argv documents the constrained worker environment for sharding.
        argv = [
            str(runtime.engine_python or "python"),
            "-c",
            (
                "import os; "
                f"assert os.environ.get('CUDA_VISIBLE_DEVICES') == '{gpu}', "
                "'SenseVoice worker GPU scope mismatch'"
            ),
        ]
        return LaunchPlan(
            family=self.family,
            argv=argv,
            env={"CUDA_VISIBLE_DEVICES": str(gpu)},
            client_env=client_env,
            api_base=None,
            port=None,
            gpu=gpu,
            served_model_name="sensevoice-small",
            notes=[
                "SenseVoice 本地 FunASR 加载，无独立 serve 端口",
                "流水线分片须将 gpus 约束为授权卡，并由 sharded_run 注入 CUDA_VISIBLE_DEVICES",
                f"逻辑 device={cfg.device}（相对可见卡）",
            ],
            kind="local",
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
        if path_errors and not dry_run:
            raise FileNotFoundError(
                "SenseVoice 路径检查失败:\n- " + "\n- ".join(path_errors)
            )
        session_dir = Path(session_dir)
        session_dir.mkdir(parents=True, exist_ok=True)
        session = ServiceSession(
            family=self.family,
            gpu=gpu,
            port=None,
            api_base=None,
            served_model_name="sensevoice-small",
            owned=True,
            client_env=plan.client_env,
            argv=plan.argv,
            session_dir=str(session_dir),
            kind="local",
            attached=bool(attach_existing),
            extra={
                "dry_run": dry_run,
                "model_path": str(runtime.sensevoice.model_path),
                "device": runtime.sensevoice.device,
            },
        )
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
        cfg = runtime.sensevoice
        audio = require_probe_audio(audio_path)
        assigned = gpu
        if assigned is None and session is not None:
            assigned = session.gpu
        if assigned is None:
            raise ValueError("SenseVoice 探针需要 gpu，以约束 CUDA_VISIBLE_DEVICES")

        # Constrain this process visibility for the probe load.
        previous = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ.update(build_cuda_env(assigned, overlays={
            "SENSEVOICE_MODEL_PATH": str(cfg.model_path),
        }))
        try:
            from audio_engine.operators.asr import sensevoice as sensevoice_op

            settings = {
                "model_path": str(cfg.model_path),
                "device": cfg.device,
                "language": cfg.language,
                "use_itn": cfg.use_itn,
                "disable_update": cfg.disable_update,
            }
            model = sensevoice_op._load_sensevoice_model(settings)
            raw = model.generate(
                input=str(audio),
                language=cfg.language,
                use_itn=cfg.use_itn,
            )
            text = ""
            if isinstance(raw, list) and raw:
                first = raw[0]
                if isinstance(first, dict):
                    text = str(first.get("text") or "")
                else:
                    text = str(first)
            elif isinstance(raw, dict):
                text = str(raw.get("text") or "")
            else:
                text = str(raw or "")
            parsed = sensevoice_op.parse_sensevoice_text(text)
            comparable = str(parsed.get("text") or text).strip()
            return ProbeResult(
                ok=bool(comparable or text.strip()),
                family=self.family,
                audio_path=str(audio),
                text=comparable or text.strip(),
                detail={"raw_text": text, "parsed": parsed, "gpu": assigned},
            )
        finally:
            try:
                from audio_engine.operators.asr import sensevoice as sensevoice_op

                sensevoice_op.release_cached_models()
            except Exception:
                pass
            if previous is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous

    def stop(self, session: ServiceSession) -> None:
        # Local adapter: drop in-process cache only; never kill unrelated workers.
        try:
            from audio_engine.operators.asr import sensevoice as sensevoice_op

            sensevoice_op.release_cached_models()
        except Exception:
            pass
