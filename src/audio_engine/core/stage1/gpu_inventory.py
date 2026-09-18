"""GPU inventory: UUID binding, VRAM snapshot, process ownership.

Utilization must NEVER be treated as an idle/loadable signal.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any, Callable


HEAVY_FAMILIES = frozenset({"qwen", "glm"})


@dataclass(frozen=True)
class GpuProcess:
    pid: int
    used_memory_mib: float | None
    name: str
    owned: bool = False  # filled by caller against lease/job pids


@dataclass(frozen=True)
class GpuDevice:
    index: str
    uuid: str
    memory_used_mib: float | None
    memory_total_mib: float | None
    utilization_gpu: float | None
    utilization_memory: float | None
    processes: tuple[GpuProcess, ...] = ()
    collect_ok: bool = True
    note: str = ""

    @property
    def free_memory_mib(self) -> float | None:
        if self.memory_used_mib is None or self.memory_total_mib is None:
            return None
        return max(0.0, float(self.memory_total_mib) - float(self.memory_used_mib))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["idle_by_util_forbidden"] = True
        data["free_memory_mib"] = self.free_memory_mib
        return data


@dataclass
class GpuBinding:
    """Authorized index ↔ UUID binding for a job."""

    tokens: list[str]  # normalized index or uuid tokens from request
    index_to_uuid: dict[str, str] = field(default_factory=dict)
    uuid_to_index: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    unknown: bool = False

    def resolve_index(self, token: int | str) -> str:
        text = str(token).strip()
        if text in self.index_to_uuid:
            return text
        if text in self.uuid_to_index:
            return self.uuid_to_index[text]
        if text.isdigit():
            return str(int(text))
        return text

    def expected_uuid(self, token: int | str) -> str | None:
        idx = self.resolve_index(token)
        return self.index_to_uuid.get(idx)


SnapshotFn = Callable[[list[str]], list[GpuDevice]]


def _parse_float(text: str) -> float | None:
    text = text.strip()
    if not text or text.upper() == "[N/A]":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def collect_gpu_snapshot(
    gpu_ids: list[str],
    *,
    timeout_s: float = 5.0,
) -> list[GpuDevice]:
    """Best-effort nvidia-smi snapshot. Failures → empty list (caller marks unknown)."""
    if not gpu_ids:
        return []
    if shutil.which("nvidia-smi") is None:
        return []
    ids = ",".join(str(g) for g in gpu_ids)
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={ids}",
                "--query-gpu=index,uuid,memory.used,memory.total,"
                "utilization.gpu,utilization.memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except Exception:
        return []
    if completed.returncode != 0:
        return []

    devices: list[GpuDevice] = []
    for line in completed.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        index = parts[0]
        uuid = parts[1]
        procs = _collect_processes_for_index(index, timeout_s=timeout_s)
        devices.append(
            GpuDevice(
                index=index,
                uuid=uuid,
                memory_used_mib=_parse_float(parts[2]),
                memory_total_mib=_parse_float(parts[3]),
                utilization_gpu=_parse_float(parts[4]),
                utilization_memory=_parse_float(parts[5]),
                processes=tuple(procs),
                collect_ok=True,
                note="利用率低不能作为空闲判据",
            )
        )
    return devices


def _collect_processes_for_index(index: str, *, timeout_s: float) -> list[GpuProcess]:
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-compute-apps=pid,used_gpu_memory,process_name",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except Exception:
        return []
    if completed.returncode != 0:
        return []
    procs: list[GpuProcess] = []
    for line in completed.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 1 or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        mem = _parse_float(parts[1]) if len(parts) > 1 else None
        name = parts[2] if len(parts) > 2 else ""
        procs.append(GpuProcess(pid=pid, used_memory_mib=mem, name=name))
    return procs


def bind_authorized_gpus(
    *,
    request_gpus: list[str],
    authorized_gpus: tuple[int | str, ...] | None,
    authorized_uuids: tuple[str, ...] | None,
    snapshot: list[GpuDevice] | None = None,
    snapshot_fn: SnapshotFn | None = None,
) -> GpuBinding:
    """Bind job GPU tokens to live UUIDs; detect drift vs configured UUIDs."""
    tokens = [str(g).strip() for g in request_gpus if str(g).strip()]
    binding = GpuBinding(tokens=tokens)
    if not tokens:
        binding.errors.append("request.gpus 为空")
        binding.unknown = True
        return binding

    devices = snapshot
    if devices is None:
        fn = snapshot_fn or collect_gpu_snapshot
        devices = fn(tokens)

    if not devices:
        binding.unknown = True
        binding.errors.append("GPU 采集失败或无 nvidia-smi；准入标记 unknown")
        # Still record configured UUID hints if provided.
        if authorized_uuids and authorized_gpus and len(authorized_uuids) == len(
            authorized_gpus
        ):
            for gpu, uuid in zip(authorized_gpus, authorized_uuids):
                idx = str(int(gpu)) if str(gpu).isdigit() else str(gpu)
                binding.index_to_uuid[idx] = str(uuid)
                binding.uuid_to_index[str(uuid)] = idx
        return binding

    for device in devices:
        binding.index_to_uuid[str(device.index)] = device.uuid
        binding.uuid_to_index[device.uuid] = str(device.index)

    if authorized_uuids and authorized_gpus:
        if len(authorized_uuids) != len(authorized_gpus):
            binding.errors.append(
                "authorized_gpu_uuids 长度必须与 authorized_gpus 一致"
            )
        else:
            for gpu, expected in zip(authorized_gpus, authorized_uuids):
                idx = str(int(gpu)) if str(gpu).isdigit() else str(gpu)
                live = binding.index_to_uuid.get(idx)
                if live and live != str(expected):
                    binding.errors.append(
                        f"GPU {idx} UUID 漂移: 配置={expected} 实测={live}"
                    )
    return binding


def family_vram_budget_mib(
    family: str,
    *,
    total_mib: float,
    gpu_memory_utilization: float | None = None,
    sensevoice_reserve_fraction: float = 0.35,
) -> float:
    """Conservative reservation; never assume Qwen+GLM fit together."""
    family = family.lower().strip()
    if family in {"qwen", "glm"}:
        util = float(gpu_memory_utilization if gpu_memory_utilization is not None else 0.5)
        # Headroom above configured util so we don't oversubscribe.
        return total_mib * min(0.98, util + 0.05)
    if family in {"sensevoice", "sv"}:
        return total_mib * sensevoice_reserve_fraction
    return total_mib * 0.5


def can_admit_family(
    device: GpuDevice | None,
    family: str,
    *,
    gpu_memory_utilization: float | None = None,
    foreign_pids: set[int] | None = None,
    owned_pids: set[int] | None = None,
    allow_unknown: bool = False,
) -> tuple[bool, str]:
    """VRAM + foreign-process admission. util==0 is irrelevant."""
    if device is None or not device.collect_ok:
        if allow_unknown:
            return True, "inventory_unknown_allowed"
        return False, "inventory_unknown"
    foreign_pids = foreign_pids or set()
    owned_pids = owned_pids or set()
    for proc in device.processes:
        if proc.pid in owned_pids:
            continue
        if proc.pid in foreign_pids or (
            foreign_pids == set() and proc.pid not in owned_pids
        ):
            # Any non-owned compute process blocks heavy admission by default.
            if family in HEAVY_FAMILIES and proc.pid not in owned_pids:
                return False, f"foreign_process_pid={proc.pid}"
    if device.memory_total_mib is None or device.free_memory_mib is None:
        if allow_unknown:
            return True, "memory_unknown_allowed"
        return False, "memory_unknown"
    need = family_vram_budget_mib(
        family,
        total_mib=float(device.memory_total_mib),
        gpu_memory_utilization=gpu_memory_utilization,
    )
    # Free memory alone is not enough if util is low but mem is full —
    # we already use free_memory. Reject if free < need * 0.9 (allow small reuse).
    if float(device.free_memory_mib) < need * 0.15 and family in HEAVY_FAMILIES:
        # Card already nearly full — do not treat util=0 as loadable.
        return False, (
            f"insufficient_free_vram free={device.free_memory_mib:.0f} "
            f"need~{need:.0f} (util ignored)"
        )
    return True, "ok"


def families_compatible_on_same_gpu(a: str, b: str) -> bool:
    """Qwen and GLM must not co-reside; default one heavy instance per card."""
    a = a.lower().strip()
    b = b.lower().strip()
    if a == b:
        return True  # resident reuse same family
    if a in HEAVY_FAMILIES and b in HEAVY_FAMILIES:
        return False
    if a in HEAVY_FAMILIES and b in HEAVY_FAMILIES | {"sensevoice", "sv"}:
        # Heavy + SenseVoice: do not assume fit; require exclusive by default.
        return False
    return False
