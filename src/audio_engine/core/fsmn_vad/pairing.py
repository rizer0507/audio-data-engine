"""VAD event consumption matching ph_ai_model_asr_rt_cpu process_vad_and_asr."""

from __future__ import annotations

from typing import Any

from audio_engine.core.fsmn_vad.clip import clip_pcm16le, ms_to_byte_offset
from audio_engine.core.fsmn_vad.types import (
    BRANCH_COMPLETE,
    BRANCH_EOF,
    BRANCH_SPLIT_END,
    BYTES_PER_MS,
    COMPLETE_PENDING_DROP_MAX_MS,
    END_REASON_EOF,
    END_REASON_VAD,
    SOURCE_STATUS_ALL_SHORT,
    SOURCE_STATUS_COMPLETED,
    SOURCE_STATUS_NO_SPEECH,
    SPLIT_END_DROP_MAX_MS,
    SPLIT_END_REPORT_LOOKBACK_MS,
    SR,
    VadDiscarded,
    VadKeptSegment,
    VadRawInterval,
    VadStreamResult,
)


def _as_event_pairs(events: Any) -> list[tuple[float, float]]:
    if not events:
        return []
    pairs: list[tuple[float, float]] = []
    for item in events:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            raise ValueError(f"Invalid VAD event shape: {item!r}")
        pairs.append((float(item[0]), float(item[1])))
    return pairs


def _classify_status(
    *,
    raw_count: int,
    kept_count: int,
    discarded_count: int,
    short_count: int,
) -> str:
    if raw_count == 0 and kept_count == 0:
        return SOURCE_STATUS_NO_SPEECH
    if kept_count > 0:
        return SOURCE_STATUS_COMPLETED
    if discarded_count > 0 and short_count > 0:
        return SOURCE_STATUS_ALL_SHORT
    if discarded_count > 0:
        return SOURCE_STATUS_ALL_SHORT
    return SOURCE_STATUS_NO_SPEECH


class FenshenEventPairer:
    """Stateful consumer of model events + application EOF (send_end_result=true)."""

    def __init__(self, *, sample_rate: int = SR, bytes_per_ms: int = BYTES_PER_MS):
        if sample_rate != SR:
            # Byte-offset formula is fixed to 16 kHz PCM16 like the original service.
            raise ValueError(
                f"fenshen VAD pairing requires sample_rate={SR}, got {sample_rate}"
            )
        self.sample_rate = sample_rate
        self.bytes_per_ms = bytes_per_ms
        self.pending_start_ms: float | None = None
        self.emitted: set[tuple[float, float]] = set()
        self.raw_intervals: list[VadRawInterval] = []
        self.kept: list[VadKeptSegment] = []
        self.discarded: list[VadDiscarded] = []
        self.eof_fallback: list[VadKeptSegment] = []

    def consume_events(
        self,
        events: Any,
        pcm16le: bytes,
        *,
        received_bytes: int,
    ) -> None:
        if received_bytes < 0 or received_bytes > len(pcm16le) or received_bytes % 2:
            raise ValueError(f"Invalid received_bytes={received_bytes}")
        for start, end in _as_event_pairs(events):
            if start < -1 or end < -1:
                raise ValueError(f"Invalid VAD event: {start}, {end}")
            if start != -1 and end == -1:
                # Overwrite any existing pending (original service behaviour).
                self.pending_start_ms = start
                continue
            if start == -1 and end != -1:
                self._handle_split_end(end, pcm16le, received_bytes=received_bytes)
                continue
            if start != -1 and end != -1:
                self._handle_complete(start, end, pcm16le, received_bytes=received_bytes)
                continue
            # [-1, -1] and other no-ops: ignored like the original service.

    def finalize_application_end(
        self,
        pcm16le: bytes,
        *,
        received_bytes: int,
        send_end_result: bool = True,
    ) -> None:
        """Mirror process_vad_and_asr is_final + send_end_result branch."""
        if self.pending_start_ms is None:
            return
        if not send_end_result:
            return
        self._emit_eof(pcm16le, received_bytes=received_bytes)

    def to_result(self, event_log: list[dict[str, Any]]) -> VadStreamResult:
        short_count = sum(
            1
            for d in self.discarded
            if d.reason in {"short_split_end", "short_complete_pending"}
        )
        status = _classify_status(
            raw_count=len(self.raw_intervals),
            kept_count=len(self.kept),
            discarded_count=len(self.discarded),
            short_count=short_count,
        )
        return VadStreamResult(
            candidates=list(self.raw_intervals),
            kept=list(self.kept),
            discarded=list(self.discarded),
            eof_fallback=list(self.eof_fallback),
            event_log=list(event_log),
            status=status,
        )

    def _handle_split_end(
        self, end: float, pcm16le: bytes, *, received_bytes: int
    ) -> None:
        if self.pending_start_ms is None:
            return
        pending = self.pending_start_ms
        reported_start = pending - SPLIT_END_REPORT_LOOKBACK_MS
        reported_end = end
        if end - pending <= SPLIT_END_DROP_MAX_MS:
            self.discarded.append(
                VadDiscarded(
                    start_ms=pending,
                    end_ms=end,
                    end_reason=END_REASON_VAD,
                    reason="short_split_end",
                    clip_branch=BRANCH_SPLIT_END,
                    reported_start_ms=reported_start,
                    reported_end_ms=reported_end,
                )
            )
            self.pending_start_ms = None
            return

        start_byte = max(0, ms_to_byte_offset(pending, bytes_per_ms=self.bytes_per_ms))
        end_byte = min(received_bytes, ms_to_byte_offset(end, bytes_per_ms=self.bytes_per_ms))
        key = (reported_start, reported_end)
        if key in self.emitted:
            # Original service continues without clearing pending on duplicate.
            return

        self._publish(
            pcm_start_ms=pending,
            model_end_ms=end,
            reported_start_ms=reported_start,
            reported_end_ms=reported_end,
            start_byte=start_byte,
            end_byte=end_byte,
            received_bytes=received_bytes,
            pcm16le=pcm16le,
            end_reason=END_REASON_VAD,
            clip_branch=BRANCH_SPLIT_END,
            inferred_end_ms=None,
            extra_padding_ms=0.0,
        )
        self.emitted.add(key)
        self.pending_start_ms = None

    def _handle_complete(
        self, start: float, end: float, pcm16le: bytes, *, received_bytes: int
    ) -> None:
        reported_start = start
        reported_end = end
        if (
            self.pending_start_ms is not None
            and end - self.pending_start_ms <= COMPLETE_PENDING_DROP_MAX_MS
        ):
            self.discarded.append(
                VadDiscarded(
                    start_ms=self.pending_start_ms,
                    end_ms=end,
                    end_reason=END_REASON_VAD,
                    reason="short_complete_pending",
                    clip_branch=BRANCH_COMPLETE,
                    reported_start_ms=reported_start,
                    reported_end_ms=reported_end,
                )
            )
            self.pending_start_ms = None
            return

        start_byte = max(0, ms_to_byte_offset(start, bytes_per_ms=self.bytes_per_ms))
        # Original service computes end_pcm then crops pcm_bytes[start_pcm:] — buffer end.
        end_byte = received_bytes
        key = (reported_start, reported_end)
        if key in self.emitted:
            return

        self._publish(
            pcm_start_ms=start,
            model_end_ms=end,
            reported_start_ms=reported_start,
            reported_end_ms=reported_end,
            start_byte=start_byte,
            end_byte=end_byte,
            received_bytes=received_bytes,
            pcm16le=pcm16le,
            end_reason=END_REASON_VAD,
            clip_branch=BRANCH_COMPLETE,
            inferred_end_ms=None,
            extra_padding_ms=0.0,
        )
        self.emitted.add(key)
        self.pending_start_ms = None

    def _emit_eof(self, pcm16le: bytes, *, received_bytes: int) -> None:
        pending = self.pending_start_ms
        if pending is None:
            return
        start_byte = max(0, ms_to_byte_offset(pending, bytes_per_ms=self.bytes_per_ms))
        end_byte = received_bytes
        inferred_end_ms = received_bytes / float(self.bytes_per_ms)
        seg = self._publish(
            pcm_start_ms=pending,
            model_end_ms=None,
            reported_start_ms=pending,
            reported_end_ms=inferred_end_ms,
            start_byte=start_byte,
            end_byte=end_byte,
            received_bytes=received_bytes,
            pcm16le=pcm16le,
            end_reason=END_REASON_EOF,
            clip_branch=BRANCH_EOF,
            inferred_end_ms=inferred_end_ms,
            extra_padding_ms=0.0,
            into_eof_list=True,
        )
        if seg is not None:
            # Match original: pending is not cleared after EOF send; stream ends next.
            pass

    def _publish(
        self,
        *,
        pcm_start_ms: float,
        model_end_ms: float | None,
        reported_start_ms: float,
        reported_end_ms: float | None,
        start_byte: int,
        end_byte: int,
        received_bytes: int,
        pcm16le: bytes,
        end_reason: str,
        clip_branch: str,
        inferred_end_ms: float | None,
        extra_padding_ms: float,
        into_eof_list: bool = False,
    ) -> VadKeptSegment | None:
        start_byte = max(0, min(received_bytes, int(start_byte)))
        end_byte = max(start_byte, min(received_bytes, int(end_byte)))
        if start_byte % 2 or end_byte % 2:
            # PCM16 alignment; original int(ms*32) is always even for integer ms.
            start_byte -= start_byte % 2
            end_byte -= end_byte % 2
            end_byte = max(start_byte, end_byte)
        start_sample = start_byte // 2
        end_sample = end_byte // 2
        buffer_end_sample = received_bytes // 2
        raw_start = max(0, ms_to_byte_offset(pcm_start_ms, bytes_per_ms=self.bytes_per_ms))
        if clip_branch == BRANCH_SPLIT_END and model_end_ms is not None:
            raw_end = min(
                received_bytes,
                ms_to_byte_offset(model_end_ms, bytes_per_ms=self.bytes_per_ms),
            )
            boundary_clamped = start_byte != raw_start or end_byte != raw_end
        elif clip_branch == BRANCH_COMPLETE:
            boundary_clamped = start_byte != raw_start
        else:
            boundary_clamped = start_byte != raw_start or end_byte != received_bytes

        duration_ms = (end_sample - start_sample) * 1000.0 / self.sample_rate
        pcm = clip_pcm16le(pcm16le, start_sample, end_sample)
        raw = VadRawInterval(
            start_ms=pcm_start_ms,
            end_ms=model_end_ms,
            end_reason=end_reason,
            clip_branch=clip_branch,
            boundary_clamped=boundary_clamped,
            start_sample=start_sample,
            end_sample=end_sample,
            reported_start_ms=reported_start_ms,
            reported_end_ms=reported_end_ms,
        )
        self.raw_intervals.append(raw)
        seg = VadKeptSegment(
            start_ms=pcm_start_ms,
            end_ms=model_end_ms,
            inferred_end_ms=inferred_end_ms,
            end_reason=end_reason,
            start_sample=start_sample,
            end_sample=end_sample,
            duration_ms=duration_ms,
            boundary_clamped=boundary_clamped,
            pcm16le=pcm,
            extra_padding_ms=extra_padding_ms,
            clip_branch=clip_branch,
            reported_start_ms=reported_start_ms,
            reported_end_ms=reported_end_ms,
            buffer_end_sample=buffer_end_sample,
        )
        self.kept.append(seg)
        if into_eof_list:
            self.eof_fallback.append(seg)
        return seg


def pair_and_filter(
    event_log: list[dict[str, Any]],
    pcm16le: bytes,
    *,
    total_samples: int | None = None,
    include_eof_fallback_in_kept: bool = True,
    send_end_result: bool = True,
    sample_rate: int = SR,
) -> VadStreamResult:
    """Replay logged model events with original-service branch rules.

    ``include_eof_fallback_in_kept`` is accepted for compatibility; with
    ``send_end_result=True`` (fenshen default) EOF fallback is always published.
    """
    del include_eof_fallback_in_kept  # fenshen send_end_result governs publication
    if len(pcm16le) % 2:
        raise ValueError("PCM16 byte length must be even")
    total = total_samples if total_samples is not None else len(pcm16le) // 2
    if total == 0:
        return VadStreamResult(status="empty_audio", event_log=list(event_log))

    pairer = FenshenEventPairer(sample_rate=sample_rate)
    saw_application_end = False
    for entry in event_log:
        received = entry.get("received_bytes")
        if received is None:
            end_sample = entry.get("input_end_sample")
            if end_sample is None:
                received = len(pcm16le)
            else:
                received = int(end_sample) * 2
        received = min(int(received), len(pcm16le))
        if not entry.get("skipped_generate"):
            pairer.consume_events(
                entry.get("events") or [],
                pcm16le,
                received_bytes=received,
            )
        if entry.get("application_end"):
            pairer.finalize_application_end(
                pcm16le,
                received_bytes=received,
                send_end_result=send_end_result,
            )
            saw_application_end = True

    if not saw_application_end and send_end_result:
        # Offline callers / legacy logs may omit application_end; append EOF once.
        pairer.finalize_application_end(
            pcm16le,
            received_bytes=len(pcm16le),
            send_end_result=True,
        )

    return pairer.to_result(event_log)
