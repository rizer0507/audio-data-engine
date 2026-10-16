"""VAD event pairing, short-utterance filter, and EOF fallback handling."""

from __future__ import annotations

from typing import Any

from audio_engine.core.fsmn_vad.clip import (
    clip_pcm16le,
    compute_clip_bounds,
    meets_min_keep,
    ms_to_sample,
)
from audio_engine.core.fsmn_vad.types import (
    END_REASON_EOF,
    END_REASON_VAD,
    END_SILENCE_KEEP_MS,
    SOURCE_STATUS_ALL_SHORT,
    SOURCE_STATUS_COMPLETED,
    SOURCE_STATUS_ENDPOINT_AT_EOF,
    SOURCE_STATUS_EOF_WITHOUT_END,
    SOURCE_STATUS_NO_SPEECH,
    SR,
    VAD_FRAME_LENGTH_MS,
    VAD_FRAME_SHIFT_MS,
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
    endpoint_at_eof_count: int,
    short_count: int,
) -> str:
    if raw_count == 0:
        return SOURCE_STATUS_NO_SPEECH
    if kept_count > 0:
        return SOURCE_STATUS_COMPLETED
    if discarded_count > 0:
        if endpoint_at_eof_count > 0:
            return SOURCE_STATUS_ENDPOINT_AT_EOF
        if short_count > 0:
            return SOURCE_STATUS_ALL_SHORT
        return SOURCE_STATUS_EOF_WITHOUT_END
    return SOURCE_STATUS_NO_SPEECH


def pair_and_filter(
    event_log: list[dict[str, Any]],
    pcm16le: bytes,
    *,
    total_samples: int | None = None,
    include_eof_fallback_in_kept: bool = False,
    sample_rate: int = SR,
) -> VadStreamResult:
    """Retain END_SILENCE_KEEP_MS after model endpoints; drop clips shorter than 1s.

    End silence uses existing source audio only and is clamped at file EOF.
    Model endpoints reaching the last frontend frame are always discarded.
    Pending starts without a model end at EOF are always discarded.
    include_eof_fallback_in_kept is accepted for compatibility and cannot
    re-enable EOF fallback publication.
    """
    if len(pcm16le) % 2:
        raise ValueError("PCM16 byte length must be even")
    total = total_samples if total_samples is not None else len(pcm16le) // 2
    if total == 0:
        return VadStreamResult(status="empty_audio", event_log=list(event_log))

    # WavFrontendOnline uses 25 ms windows, 10 ms shifts and snip_edges=True.
    # Its EOF end timestamp is frame_count * shift, often 15-25 ms before
    # the actual sample EOF. Compare the raw model end, before retaining a tail.
    frame_length = ms_to_sample(VAD_FRAME_LENGTH_MS, sample_rate=sample_rate)
    frame_shift = ms_to_sample(VAD_FRAME_SHIFT_MS, sample_rate=sample_rate)
    frame_count = max(0, (total - frame_length) // frame_shift + 1)
    eof_endpoint_sample = frame_count * frame_shift

    pending: float | None = None
    seen: set[tuple[int, int, str]] = set()
    raw_intervals: list[VadRawInterval] = []
    kept: list[VadKeptSegment] = []
    discarded: list[VadDiscarded] = []
    eof_list: list[VadKeptSegment] = []

    def _emit(start_ms: float, end_ms: float, reason: str) -> None:
        if start_ms < 0 or end_ms <= start_ms:
            raise ValueError(f"Invalid interval: {start_ms}, {end_ms}")
        endpoint_at_eof = (
            reason == END_REASON_VAD
            and ms_to_sample(end_ms, sample_rate=sample_rate) >= eof_endpoint_sample
        )
        retained_end_ms = end_ms + END_SILENCE_KEEP_MS if reason == END_REASON_VAD else end_ms
        bounds = compute_clip_bounds(
            start_ms,
            retained_end_ms,
            total,
            sample_rate=sample_rate,
            end_is_eof=(reason == END_REASON_EOF),
        )
        if bounds.end_sample <= bounds.start_sample:
            raise ValueError("Interval has no samples after clipping")
        key = (
            bounds.start_sample,
            bounds.end_sample,
            "endpoint_at_eof" if endpoint_at_eof else reason,
        )
        if key in seen:
            return
        seen.add(key)

        raw_intervals.append(
            VadRawInterval(
                start_ms=start_ms,
                end_ms=end_ms,
                end_reason=reason,
                boundary_clamped=bounds.boundary_clamped,
                start_sample=bounds.start_sample,
                end_sample=bounds.end_sample,
            )
        )

        sample_count = bounds.end_sample - bounds.start_sample
        duration_ms = bounds.duration_ms
        if reason == END_REASON_EOF:
            ok = False
            reject_reason = "eof_without_end"
        elif endpoint_at_eof:
            ok = False
            reject_reason = "endpoint_at_eof"
        else:
            ok, reject_reason = meets_min_keep(0.0, duration_ms, sample_count)

        if not ok:
            discarded.append(
                VadDiscarded(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    end_reason=reason,
                    reason=reject_reason or "short_utterance",
                    start_sample=bounds.start_sample,
                    end_sample=bounds.end_sample,
                    duration_ms=duration_ms,
                )
            )
            return

        pcm = clip_pcm16le(pcm16le, bounds.start_sample, bounds.end_sample)
        seg = VadKeptSegment(
            start_ms=start_ms,
            end_ms=end_ms,
            inferred_end_ms=None,
            end_reason=reason,
            start_sample=bounds.start_sample,
            end_sample=bounds.end_sample,
            duration_ms=duration_ms,
            boundary_clamped=bounds.boundary_clamped,
            pcm16le=pcm,
            extra_padding_ms=max(
                0.0, bounds.end_sample * 1000.0 / sample_rate - end_ms
            ),
        )
        kept.append(seg)

    for entry in event_log:
        events = _as_event_pairs(entry.get("events") or [])
        for start, end in events:
            if start < -1 or end < -1 or (start == -1 and end == -1):
                raise ValueError(f"Invalid VAD event: {start}, {end}")
            if start != -1:
                if pending is not None and pending != start:
                    raise ValueError("New start before pending interval closed")
                pending = start
            if end != -1:
                if pending is None:
                    raise ValueError("VAD end without a start")
                _emit(pending, end, END_REASON_VAD)
                pending = None

    if pending is not None:
        eof_end_ms = total * 1000.0 / sample_rate
        _emit(pending, eof_end_ms, END_REASON_EOF)

    status = _classify_status(
        raw_count=len(raw_intervals),
        kept_count=len(kept),
        discarded_count=len(discarded),
        endpoint_at_eof_count=sum(d.reason == "endpoint_at_eof" for d in discarded),
        short_count=sum(d.reason == "short_utterance" for d in discarded),
    )

    return VadStreamResult(
        candidates=raw_intervals,
        kept=kept,
        discarded=discarded,
        eof_fallback=eof_list,
        event_log=list(event_log),
        status=status,
    )
