"""Dual-GPU scheduler: lease + admission + family queue with resident reuse.

Design (step 4):
- Default one heavy instance per authorized card (Qwen/GLM exclusive).
- Never treat GPU-Util==0 as idle/loadable.
- Prefer resident reuse for same-family second route / shards.
- Idle card claims next pending family (tail work) after resource release.
- Multi-job isolation via GpuLeaseStore (no foreign lease steal).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from audio_engine.core.stage1.cache_policy import FAMILY_RUN_ALIASES, REQUIRED_FAMILIES
from audio_engine.core.stage1.gpu_inventory import (
    HEAVY_FAMILIES,
    GpuBinding,
    GpuDevice,
    bind_authorized_gpus,
    can_admit_family,
    collect_gpu_snapshot,
    families_compatible_on_same_gpu,
)
from audio_engine.core.stage1.gpu_lease import GpuLease, GpuLeaseError, GpuLeaseStore


@dataclass
class FamilyWorkItem:
    family: str
    aliases: list[str]
    priority: float = 0.0  # higher first; remaining duration proxy
    load_cost: float = 1.0  # relative model load cost


@dataclass
class ScheduleDecision:
    family: str
    gpu_key: str
    gpu_uuid: str | None
    reason: str
    claim_wait_s: float
    resident_reuse: bool


@dataclass
class SchedulerMetrics:
    claim_waits_s: list[float] = field(default_factory=list)
    load_timings_s: dict[str, float] = field(default_factory=dict)
    busy_time_s: dict[str, float] = field(default_factory=dict)
    idle_reasons: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)

    def claim_delay_p95(self) -> float | None:
        if not self.claim_waits_s:
            return None
        ordered = sorted(self.claim_waits_s)
        idx = min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))
        return ordered[idx]

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_waits_s": list(self.claim_waits_s),
            "claim_delay_p95_s": self.claim_delay_p95(),
            "load_timings_s": dict(self.load_timings_s),
            "busy_time_s": dict(self.busy_time_s),
            "idle_reasons": list(self.idle_reasons[-50:]),
            "decisions": list(self.decisions[-50:]),
            "note": "性能数字仅在真实服务器基准中填写；本地 mock 不得当作吞吐结论",
        }


class DualGpuScheduler:
    """Coordinates up to two authorized GPUs for family-level work units."""

    def __init__(
        self,
        *,
        job_id: str,
        worker_token: str,
        gpus: list[str],
        binding: GpuBinding,
        lease_store: GpuLeaseStore,
        family_utils: dict[str, float] | None = None,
        poll_interval_s: float = 5.0,
        snapshot_fn: Callable[[list[str]], list[GpuDevice]] | None = None,
        owned_pids: set[int] | None = None,
        allow_unknown_inventory: bool = True,
    ) -> None:
        self.job_id = job_id
        self.worker_token = worker_token
        self.gpus = [str(g) for g in gpus]
        self.binding = binding
        self.lease_store = lease_store
        self.family_utils = family_utils or {"qwen": 0.5, "glm": 0.90, "sensevoice": 0.35}
        self.poll_interval_s = poll_interval_s
        self.snapshot_fn = snapshot_fn or collect_gpu_snapshot
        self.owned_pids = owned_pids or set()
        self.allow_unknown_inventory = allow_unknown_inventory
        self.metrics = SchedulerMetrics()
        self._lock = threading.Lock()
        self._pending: list[FamilyWorkItem] = []
        self._resident: dict[str, str] = {}  # gpu_key -> family
        self._active: dict[str, str] = {}  # gpu_key -> family
        self._waiting_since: dict[str, float] = {}  # family -> monotonic
        self._busy_started: dict[str, float] = {}

    @classmethod
    def from_runtime(
        cls,
        *,
        job_id: str,
        worker_token: str,
        gpus: list[str],
        runtime: Any,
        lease_root: Path | str | None = None,
        snapshot_fn: Callable[[list[str]], list[GpuDevice]] | None = None,
        allow_unknown_inventory: bool = True,
    ) -> DualGpuScheduler:
        snap = None
        if snapshot_fn is not None:
            snap = snapshot_fn([str(g) for g in gpus])
        binding = bind_authorized_gpus(
            request_gpus=[str(g) for g in gpus],
            authorized_gpus=getattr(runtime, "authorized_gpus", None),
            authorized_uuids=getattr(runtime, "authorized_gpu_uuids", None),
            snapshot=snap,
            snapshot_fn=snapshot_fn,
        )
        root = Path(
            lease_root
            or getattr(runtime, "scheduler_lease_root", None)
            or "runs/stage1/gpu_leases"
        )
        family_utils = {
            "qwen": float(runtime.qwen.gpu_memory_utilization),
            "glm": float(runtime.glm.gpu_memory_utilization),
            "sensevoice": float(
                getattr(runtime, "sensevoice_reserve_fraction", 0.35) or 0.35
            ),
        }
        poll = float(getattr(runtime, "scheduler_poll_interval_s", 5.0) or 5.0)
        return cls(
            job_id=job_id,
            worker_token=worker_token,
            gpus=[str(g) for g in gpus],
            binding=binding,
            lease_store=GpuLeaseStore(root),
            family_utils=family_utils,
            poll_interval_s=poll,
            snapshot_fn=snapshot_fn,
            allow_unknown_inventory=allow_unknown_inventory,
        )

    def enqueue_default_families(self, *, remaining: set[str] | None = None) -> None:
        wanted = remaining or set(REQUIRED_FAMILIES)
        items: list[FamilyWorkItem] = []
        # Prefer heavier / longer families first so the second card can claim tails.
        order = [("qwen", 3.0, 1.0), ("glm", 3.0, 1.2), ("sensevoice", 2.0, 0.6)]
        for family, priority, cost in order:
            if family not in wanted:
                continue
            items.append(
                FamilyWorkItem(
                    family=family,
                    aliases=list(FAMILY_RUN_ALIASES[family]),
                    priority=priority,
                    load_cost=cost,
                )
            )
        with self._lock:
            self._pending = items
            now = time.monotonic()
            for item in items:
                self._waiting_since.setdefault(item.family, now)

    def pending_families(self) -> list[str]:
        with self._lock:
            return [item.family for item in self._pending]

    def _device_map(self) -> dict[str, GpuDevice]:
        devices = self.snapshot_fn(self.gpus) if self.snapshot_fn else []
        return {str(d.index): d for d in devices}

    def _free_gpu_keys(self) -> list[str]:
        with self._lock:
            active = set(self._active)
        free: list[str] = []
        for gpu in self.gpus:
            key = self.binding.resolve_index(gpu)
            if key in active:
                continue
            foreign = self.lease_store.held_by_other_job(key, job_id=self.job_id)
            if foreign is not None:
                self.metrics.idle_reasons.append(
                    {
                        "gpu": key,
                        "reason": "foreign_lease",
                        "job_id": foreign.job_id,
                        "family": foreign.family,
                    }
                )
                continue
            free.append(key)
        return free

    def try_claim(self, *, mode: str = "run") -> ScheduleDecision | None:
        """Claim one pending family onto a free admissible GPU."""
        with self._lock:
            if not self._pending:
                return None
            pending = list(self._pending)
            resident = dict(self._resident)

        free = self._free_gpu_keys()
        if not free:
            return None

        devices = self._device_map()
        # Prefer: resident reuse for same family still pending (shouldn't happen
        # for family-level units), else highest priority onto any free card.
        pending_sorted = sorted(
            pending, key=lambda item: (-item.priority, item.load_cost, item.family)
        )

        for gpu_key in free:
            # Prefer assigning a family that matches nothing resident conflict.
            for item in pending_sorted:
                family = item.family
                resident_family = resident.get(gpu_key)
                if resident_family and not families_compatible_on_same_gpu(
                    resident_family, family
                ):
                    # Need previous family fully stopped before new heavy load.
                    if resident_family != family:
                        self.metrics.idle_reasons.append(
                            {
                                "gpu": gpu_key,
                                "reason": "incompatible_resident",
                                "resident": resident_family,
                                "wanted": family,
                            }
                        )
                        continue
                device = devices.get(gpu_key)
                ok, why = can_admit_family(
                    device,
                    family,
                    gpu_memory_utilization=self.family_utils.get(family),
                    owned_pids=self.owned_pids,
                    allow_unknown=self.allow_unknown_inventory,
                )
                if not ok:
                    self.metrics.idle_reasons.append(
                        {
                            "gpu": gpu_key,
                            "reason": why,
                            "family": family,
                            "util_ignored": True,
                        }
                    )
                    continue
                try:
                    lease = self.lease_store.acquire(
                        gpu_key,
                        job_id=self.job_id,
                        worker_token=self.worker_token,
                        family=family,
                        gpu_uuid=self.binding.index_to_uuid.get(gpu_key),
                        mode=mode,
                    )
                except GpuLeaseError as exc:
                    self.metrics.idle_reasons.append(
                        {"gpu": gpu_key, "reason": str(exc), "family": family}
                    )
                    continue

                wait_s = 0.0
                with self._lock:
                    started = self._waiting_since.pop(family, time.monotonic())
                    wait_s = max(0.0, time.monotonic() - started)
                    self._pending = [x for x in self._pending if x.family != family]
                    self._active[gpu_key] = family
                    self._resident[gpu_key] = family
                    self._busy_started[gpu_key] = time.monotonic()
                    self.metrics.claim_waits_s.append(wait_s)
                    decision = ScheduleDecision(
                        family=family,
                        gpu_key=gpu_key,
                        gpu_uuid=lease.gpu_uuid,
                        reason="claimed",
                        claim_wait_s=wait_s,
                        resident_reuse=resident_family == family,
                    )
                    self.metrics.decisions.append(
                        {
                            "family": family,
                            "gpu": gpu_key,
                            "uuid": lease.gpu_uuid,
                            "wait_s": wait_s,
                            "resident_reuse": decision.resident_reuse,
                        }
                    )
                return decision
        return None

    def release_gpu(self, gpu_key: str, *, clear_resident: bool = True) -> None:
        with self._lock:
            family = self._active.pop(gpu_key, None)
            started = self._busy_started.pop(gpu_key, None)
            if started is not None:
                self.metrics.busy_time_s[gpu_key] = self.metrics.busy_time_s.get(
                    gpu_key, 0.0
                ) + (time.monotonic() - started)
            if clear_resident:
                self._resident.pop(gpu_key, None)
        self.lease_store.release(
            gpu_key, worker_token=self.worker_token, job_id=self.job_id
        )

    def record_load_time(self, family: str, seconds: float) -> None:
        self.metrics.load_timings_s[family] = seconds

    def mark_family_waiting(self, family: str) -> None:
        with self._lock:
            self._waiting_since.setdefault(family, time.monotonic())

    def requeue(self, item: FamilyWorkItem) -> None:
        with self._lock:
            if any(x.family == item.family for x in self._pending):
                return
            self._pending.append(item)
            self._waiting_since.setdefault(item.family, time.monotonic())


def build_serial_baseline_plan(families: list[str], gpu: str) -> list[dict[str, Any]]:
    """Document-only serial plan for benchmark comparison (same decode config)."""
    return [
        {
            "order": idx,
            "family": family,
            "gpu": gpu,
            "mode": "serial_baseline",
            "aliases": list(FAMILY_RUN_ALIASES.get(family, [])),
        }
        for idx, family in enumerate(families)
    ]


def build_dual_gpu_plan(
    families: list[str], gpus: list[str]
) -> list[dict[str, Any]]:
    """Static illustration of dual-card claim order (not a performance claim)."""
    if len(gpus) < 2:
        return build_serial_baseline_plan(families, gpus[0] if gpus else "?")
    plan: list[dict[str, Any]] = []
    # First wave: first two families on two cards.
    for family, gpu in zip(families[:2], gpus[:2]):
        plan.append(
            {
                "wave": 1,
                "family": family,
                "gpu": gpu,
                "mode": "dual_parallel",
                "aliases": list(FAMILY_RUN_ALIASES.get(family, [])),
            }
        )
    # Tail: remaining families claimed by whichever card frees first (runtime).
    for family in families[2:]:
        plan.append(
            {
                "wave": 2,
                "family": family,
                "gpu": "<idle_card_claims>",
                "mode": "tail_claim",
                "aliases": list(FAMILY_RUN_ALIASES.get(family, [])),
            }
        )
    return plan
