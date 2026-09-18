"""Load and validate stage-1 server runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class MissingConfigError(ValueError):
    """Raised when required deployment fields are absent."""

    def __init__(self, missing: list[str], *, path: Path | None = None) -> None:
        self.missing = list(missing)
        self.path = path
        prefix = f"{path}: " if path else ""
        detail = "\n".join(f"  - {item}" for item in self.missing)
        super().__init__(f"{prefix}部署配置缺项，请补齐后再启动:\n{detail}")


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, tuple, dict)) and len(value) == 0:
        return True
    return False


def _as_path(value: Any) -> Path | None:
    if _is_blank(value):
        return None
    return Path(str(value)).expanduser()


@dataclass(frozen=True)
class QwenFamilyConfig:
    enabled: bool
    model_path: Path
    chat_template: Path
    served_model_name: str
    vllm_bin: Path | None
    host: str
    port: int
    gpu_memory_utilization: float
    tensor_parallel_size: int
    api_key: str
    pipeline: str


@dataclass(frozen=True)
class GlmFamilyConfig:
    enabled: bool
    model_path: Path
    env_root: Path
    python_bin: Path
    vllm_bin: Path
    served_model_name: str
    host: str
    client_host: str
    port: int
    tensor_parallel_size: int
    dtype: str
    max_model_len: int
    max_num_seqs: int
    gpu_memory_utilization: float
    trust_remote_code: bool
    limit_mm_per_prompt: str
    no_enable_flashinfer_autotune: bool
    kernel_config: str
    env: dict[str, str]
    unset_env: tuple[str, ...]
    api_key: str
    pipeline: str


@dataclass(frozen=True)
class SenseVoiceFamilyConfig:
    enabled: bool
    model_path: Path
    device: str
    language: str
    use_itn: bool
    disable_update: bool
    pipeline: str


@dataclass(frozen=True)
class Stage1RuntimeConfig:
    path: Path
    authorized_gpus: tuple[int | str, ...] | None
    authorized_gpu_uuids: tuple[str, ...] | None
    engine_python: Path | None
    dnsmos_onnx_path: Path | None
    probe_audio_path: Path | None
    ready_timeout_s: float
    ready_poll_s: float
    session_root: Path
    qwen: QwenFamilyConfig
    glm: GlmFamilyConfig
    sensevoice: SenseVoiceFamilyConfig
    raw: dict[str, Any] = field(repr=False)
    scheduler_lease_root: Path = Path("runs/stage1/gpu_leases")
    scheduler_poll_interval_s: float = 5.0
    sensevoice_reserve_fraction: float = 0.35
    dual_gpu_scheduler: bool = True

    def missing_fields(
        self,
        *,
        families: list[str] | None = None,
        deploy: bool = False,
    ) -> list[str]:
        """Return blank required fields.

        ``families=None`` checks all families. ``families=[]`` checks only globals.
        ``deploy=True`` also requires engine_python and DNSMOS paths.
        """
        if families is None:
            wanted = {"qwen", "glm", "sensevoice"}
        else:
            wanted = {name.lower() for name in families}

        missing: list[str] = []
        if _is_blank(self.authorized_gpus):
            missing.append("authorized_gpus（授权 GPU 编号列表，禁止照抄历史卡号猜测）")
        if deploy:
            if self.engine_python is None:
                missing.append("engine_python（audio-data / SenseVoice 引擎解释器绝对路径）")
            if self.dnsmos_onnx_path is None:
                missing.append("dnsmos.onnx_path（DNSMOS ONNX 权重路径）")

        if "qwen" in wanted and self.qwen.enabled:
            if self.qwen.vllm_bin is None:
                missing.append("families.qwen.vllm_bin（Qwen vLLM 可执行文件绝对路径）")
            if _is_blank(str(self.qwen.model_path)):
                missing.append("families.qwen.model_path")
            if _is_blank(str(self.qwen.chat_template)):
                missing.append("families.qwen.chat_template")

        if "glm" in wanted and self.glm.enabled:
            glm_raw = (self.raw.get("families") or {}).get("glm") or {}
            has_explicit_bins = not _is_blank(glm_raw.get("python_bin")) and not _is_blank(
                glm_raw.get("vllm_bin")
            )
            if _is_blank(str(self.glm.env_root)) and not has_explicit_bins:
                missing.append("families.glm.env_root 或 python_bin/vllm_bin")
            if _is_blank(str(self.glm.model_path)):
                missing.append("families.glm.model_path")

        if "sensevoice" in wanted and self.sensevoice.enabled:
            if _is_blank(str(self.sensevoice.model_path)):
                missing.append("families.sensevoice.model_path")
            # SenseVoice 本地加载依赖引擎解释器；非 deploy 全量检查时仍单独要求
            if not deploy and self.engine_python is None:
                missing.append("engine_python（SenseVoice 本地加载所用解释器）")

        return missing

    def require_ready(
        self,
        *,
        families: list[str] | None = None,
        deploy: bool = False,
    ) -> None:
        missing = self.missing_fields(families=families, deploy=deploy)
        if missing:
            raise MissingConfigError(missing, path=self.path)

    def require_authorized_gpu(self, gpu: int | str) -> int | str:
        if _is_blank(self.authorized_gpus):
            raise MissingConfigError(
                ["authorized_gpus（授权 GPU 编号列表，禁止照抄历史卡号猜测）"],
                path=self.path,
            )
        assert self.authorized_gpus is not None
        normalized = _normalize_gpu_token(gpu)
        allowed = {_normalize_gpu_token(item) for item in self.authorized_gpus}
        if normalized not in allowed:
            raise ValueError(
                f"GPU {gpu!r} 不在 authorized_gpus={list(self.authorized_gpus)!r} 内；"
                "拒绝使用未授权卡"
            )
        return gpu

    def family(self, name: str) -> QwenFamilyConfig | GlmFamilyConfig | SenseVoiceFamilyConfig:
        key = name.lower().strip()
        if key == "qwen":
            return self.qwen
        if key == "glm":
            return self.glm
        if key in {"sensevoice", "sv"}:
            return self.sensevoice
        raise ValueError(f"未知模型家族: {name}（支持 qwen / glm / sensevoice）")


def _normalize_gpu_token(value: int | str) -> str:
    text = str(value).strip()
    if text.isdigit():
        return str(int(text))
    return text


def _require_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"配置缺少映射字段: {key}")
    return value


def _parse_authorized_gpus(raw: Any) -> tuple[int | str, ...] | None:
    if _is_blank(raw):
        return None
    if not isinstance(raw, (list, tuple)):
        raise ValueError("authorized_gpus 必须是列表")
    items: list[int | str] = []
    for item in raw:
        if isinstance(item, bool):
            raise ValueError(f"非法 GPU 项: {item!r}")
        if isinstance(item, int):
            items.append(item)
        else:
            text = str(item).strip()
            if not text:
                continue
            items.append(int(text) if text.isdigit() else text)
    return tuple(items) or None


def _parse_qwen(raw: dict[str, Any]) -> QwenFamilyConfig:
    return QwenFamilyConfig(
        enabled=bool(raw.get("enabled", True)),
        model_path=Path(str(raw.get("model_path") or "")),
        chat_template=Path(str(raw.get("chat_template") or "")),
        served_model_name=str(raw.get("served_model_name") or "qwen3-asr"),
        vllm_bin=_as_path(raw.get("vllm_bin")),
        host=str(raw.get("host") or "127.0.0.1"),
        port=int(raw.get("port") or 5555),
        gpu_memory_utilization=float(raw.get("gpu_memory_utilization") or 0.5),
        tensor_parallel_size=int(raw.get("tensor_parallel_size") or 1),
        api_key=str(raw.get("api_key") or "dummy"),
        pipeline=str(raw.get("pipeline") or "pipelines/qwen_asr_batch.yaml"),
    )


def _parse_glm(raw: dict[str, Any]) -> GlmFamilyConfig:
    env_root = _as_path(raw.get("env_root")) or Path("")
    python_bin = _as_path(raw.get("python_bin")) or (
        env_root / "bin" / "python" if str(env_root) else Path("")
    )
    vllm_bin = _as_path(raw.get("vllm_bin")) or (
        env_root / "bin" / "vllm" if str(env_root) else Path("")
    )
    env_raw = raw.get("env") or {}
    if not isinstance(env_raw, dict):
        raise ValueError("families.glm.env 必须是映射")
    unset = raw.get("unset_env") or []
    if not isinstance(unset, (list, tuple)):
        raise ValueError("families.glm.unset_env 必须是列表")
    return GlmFamilyConfig(
        enabled=bool(raw.get("enabled", True)),
        model_path=Path(str(raw.get("model_path") or "")),
        env_root=env_root,
        python_bin=python_bin,
        vllm_bin=vllm_bin,
        served_model_name=str(raw.get("served_model_name") or "glm-asr"),
        host=str(raw.get("host") or "0.0.0.0"),
        client_host=str(raw.get("client_host") or "127.0.0.1"),
        port=int(raw.get("port") or 5570),
        tensor_parallel_size=int(raw.get("tensor_parallel_size") or 1),
        dtype=str(raw.get("dtype") or "bfloat16"),
        max_model_len=int(raw.get("max_model_len") or 4096),
        max_num_seqs=int(raw.get("max_num_seqs") or 8),
        gpu_memory_utilization=float(raw.get("gpu_memory_utilization") or 0.90),
        trust_remote_code=bool(raw.get("trust_remote_code", True)),
        limit_mm_per_prompt=str(raw.get("limit_mm_per_prompt") or '{"audio":1}'),
        no_enable_flashinfer_autotune=bool(raw.get("no_enable_flashinfer_autotune", True)),
        kernel_config=str(
            raw.get("kernel_config")
            or '{"enable_jit_warmup":false,"enable_cutedsl_warmup":false}'
        ),
        env={str(k): str(v) for k, v in env_raw.items()},
        unset_env=tuple(str(item) for item in unset),
        api_key=str(raw.get("api_key") or "dummy"),
        pipeline=str(raw.get("pipeline") or "pipelines/glm_asr_batch.yaml"),
    )


def _parse_sensevoice(raw: dict[str, Any]) -> SenseVoiceFamilyConfig:
    return SenseVoiceFamilyConfig(
        enabled=bool(raw.get("enabled", True)),
        model_path=Path(str(raw.get("model_path") or "")),
        device=str(raw.get("device") or "cuda:0"),
        language=str(raw.get("language") or "auto"),
        use_itn=bool(raw.get("use_itn", True)),
        disable_update=bool(raw.get("disable_update", True)),
        pipeline=str(raw.get("pipeline") or "pipelines/sensevoice_asr_batch.yaml"),
    )


def load_runtime_config(path: str | Path) -> Stage1RuntimeConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"runtime 配置不存在: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"runtime 配置必须是 YAML 映射: {config_path}")

    families = _require_mapping(raw, "families")
    probe = raw.get("probe") or {}
    if not isinstance(probe, dict):
        raise ValueError("probe 必须是映射")
    session = raw.get("session") or {}
    if not isinstance(session, dict):
        raise ValueError("session 必须是映射")
    dnsmos = raw.get("dnsmos") or {}
    if not isinstance(dnsmos, dict):
        raise ValueError("dnsmos 必须是映射")

    uuids_raw = raw.get("authorized_gpu_uuids")
    uuids: tuple[str, ...] | None = None
    if not _is_blank(uuids_raw):
        if not isinstance(uuids_raw, (list, tuple)):
            raise ValueError("authorized_gpu_uuids 必须是列表")
        uuids = tuple(str(item).strip() for item in uuids_raw if str(item).strip())

    scheduler = raw.get("scheduler") or {}
    if not isinstance(scheduler, dict):
        raise ValueError("scheduler 必须是映射")

    return Stage1RuntimeConfig(
        path=config_path,
        authorized_gpus=_parse_authorized_gpus(raw.get("authorized_gpus")),
        authorized_gpu_uuids=uuids,
        engine_python=_as_path(raw.get("engine_python")),
        dnsmos_onnx_path=_as_path(dnsmos.get("onnx_path")),
        probe_audio_path=_as_path(probe.get("audio_path")),
        ready_timeout_s=float(probe.get("ready_timeout_s") or 300),
        ready_poll_s=float(probe.get("ready_poll_s") or 2.0),
        session_root=Path(str(session.get("root") or "runs/stage1/serve")),
        qwen=_parse_qwen(_require_mapping(families, "qwen")),
        glm=_parse_glm(_require_mapping(families, "glm")),
        sensevoice=_parse_sensevoice(_require_mapping(families, "sensevoice")),
        scheduler_lease_root=Path(
            str(scheduler.get("lease_root") or "runs/stage1/gpu_leases")
        ),
        scheduler_poll_interval_s=float(scheduler.get("poll_interval_s") or 5.0),
        sensevoice_reserve_fraction=float(
            scheduler.get("sensevoice_reserve_fraction") or 0.35
        ),
        dual_gpu_scheduler=bool(scheduler.get("dual_gpu", True)),
        raw=raw,
    )
