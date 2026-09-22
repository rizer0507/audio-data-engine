# -*- coding: utf-8 -*-
"""Tests for scripts/annotate_semantic_risk_transcripts.py."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audio_engine.core.manifest import Manifest
from audio_engine.core.sample import Sample

import annotate_semantic_risk_transcripts as mod


def _sample(
    sid: str,
    *,
    category: str,
    sha: str,
    wav: Path | None = None,
    candidate: str = "候选",
) -> Sample:
    audio = {}
    if wav is not None:
        audio["resampled_16k"] = str(wav)
    return Sample(
        id=sid,
        source_path=str(wav) if wav else f"/data/{sid}.pcm",
        sha256=sha,
        duration=1.0,
        audio=audio,
        labels={
            "category": category,
            "type": category,
            "candidate_text": candidate,
            "original_audio_sha256": sha,
        },
    )


def test_prepare_label_merge_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    batch = "tbatch"
    wav_a = tmp_path / "a.wav"
    wav_b = tmp_path / "b.wav"
    wav_c = tmp_path / "c.wav"
    for p in (wav_a, wav_b, wav_c):
        p.write_bytes(b"RIFF....WAVEfmt ")  # minimal placeholder

    classified = tmp_path / "datasets" / "stage1" / "derived" / f"{mod.DEFAULT_RULE_STEM}_{batch}.parquet"
    classified.parent.mkdir(parents=True, exist_ok=True)
    Manifest.from_samples(
        [
            _sample("s_risk", category="semantic_risk", sha="hash_risk", wav=wav_a),
            _sample("s_risk2", category="semantic_risk", sha="hash_risk2", wav=wav_b),
            _sample("s_gold", category="gold_candidate", sha="hash_gold", wav=wav_c),
        ]
    ).save(classified)

    rc = mod.main(
        [
            "prepare",
            "--batch",
            batch,
            "--classified",
            str(classified),
            "--no-link-wav",
        ]
    )
    assert rc == 0
    work = mod.default_work_dir(batch)
    q = pd.read_parquet(mod.queue_path(work))
    assert len(q) == 2
    assert set(q["sample_id"]) == {"s_risk", "s_risk2"}
    assert mod.playlist_path(work).is_file()

    # simulate annotations without interactive label
    ann = pd.DataFrame(
        [
            {
                "sample_id": "s_risk",
                "original_audio_sha256": "hash_risk",
                "batch": batch,
                "category": "semantic_risk",
                "queue_index": 1,
                "wav_path": str(wav_a),
                "candidate_text": "候选",
                "gold_text": "我不需要",
                "gold_kind": "speech",
                "decision": "accept",
                "status": "annotated",
                "annotator_id": "ann1",
                "annotated_at": "2026-09-18T00:00:00Z",
            }
        ]
    )
    mod.atomic_write_parquet(ann, mod.annotations_path(work))

    out = tmp_path / "merged.parquet"
    rc = mod.main(
        [
            "merge",
            "--batch",
            batch,
            "--classified",
            str(classified),
            "--output",
            str(out),
            "--overwrite",
        ]
    )
    assert rc == 0
    merged = Manifest.load(out)
    by_id = {s.id: s for s in merged.samples}
    assert by_id["s_risk"].labels["gold_text"] == "我不需要"
    assert by_id["s_risk"].labels["is_human_verified"] is True
    assert by_id["s_risk"].labels["category"] == "semantic_risk"
    # untouched rows keep no gold
    assert not by_id["s_gold"].labels.get("gold_text")
    assert not by_id["s_risk2"].labels.get("gold_text")
    report = json.loads(out.with_suffix(out.suffix + ".merge_report.json").read_text(encoding="utf-8"))
    assert report["applied"] == 1


def test_close_wav_prefers_closer_extension(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    opened = mod.OpenedWav(path=wav, via="cursor")

    calls: list[Path | None] = []

    def fake_ext(path, *, timeout=1.5):
        calls.append(path)
        return "closed_via:ext closed=1"

    monkeypatch.setattr(mod, "close_wav_via_closer_ext", fake_ext)
    monkeypatch.setattr(
        mod,
        "_focus_wav_in_editor",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("should not focus")),
    )
    monkeypatch.setattr(
        mod,
        "_send_close_hotkey",
        lambda: (_ for _ in ()).throw(AssertionError("should not hotkey")),
    )
    assert mod.close_wav_for_remote_player(opened) == "closed_via:ext closed=1"
    assert calls == [wav]


def test_close_wav_falls_back_to_hotkey(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    wav = tmp_path / "b.wav"
    wav.write_bytes(b"RIFF")
    opened = mod.OpenedWav(path=wav, via="cursor")
    monkeypatch.setattr(mod, "close_wav_via_closer_ext", lambda *_a, **_k: None)
    monkeypatch.setattr(mod, "_focus_wav_in_editor", lambda *_a, **_k: "cursor")
    monkeypatch.setattr(mod, "_send_close_hotkey", lambda: True)
    monkeypatch.setattr(mod.time, "sleep", lambda *_a, **_k: None)
    assert mod.close_wav_for_remote_player(opened) == "closed_via:hotkey after_focus:cursor"


def test_label_closes_wav_after_save(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    batch = "tbatch_close"
    wav = tmp_path / "c.wav"
    wav.write_bytes(b"RIFF")
    work = mod.default_work_dir(batch)
    work.mkdir(parents=True)
    mod.atomic_write_parquet(
        pd.DataFrame(
            [
                {
                    "queue_index": 1,
                    "sample_id": "s1",
                    "original_audio_sha256": "h1",
                    "batch": batch,
                    "category": "semantic_risk",
                    "wav_path": str(wav),
                    "source_path": str(wav),
                    "candidate_text": "",
                    "selected_family": "",
                    "duration": 1.0,
                    "status": "pending",
                }
            ]
        ),
        mod.queue_path(work),
    )

    opened = mod.OpenedWav(path=wav.resolve(), via="cursor")
    closes: list[str] = []

    monkeypatch.setattr(mod, "closer_healthcheck", lambda: True)
    monkeypatch.setattr(mod, "open_wav_for_remote_player", lambda p: opened)
    monkeypatch.setattr(
        mod,
        "close_wav_for_remote_player",
        lambda o, **_k: closes.append(str(o.path)) or "closed_via:test",
    )
    monkeypatch.setattr("builtins.input", lambda _prompt="": "人工转写")

    rc = mod.main(
        ["label", "--batch", batch, "--work-dir", str(work), "--annotator-id", "ann1"]
    )
    assert rc == 0
    assert closes == [str(wav.resolve())]
    ann = pd.read_parquet(mod.annotations_path(work))
    assert list(ann["gold_text"]) == ["人工转写"]


def test_merge_rejects_sha_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    batch = "tbatch2"
    wav = tmp_path / "x.wav"
    wav.write_bytes(b"RIFF")
    classified = tmp_path / "c.parquet"
    Manifest.from_samples(
        [_sample("s1", category="semantic_risk", sha="hash_ok", wav=wav)]
    ).save(classified)
    work = mod.default_work_dir(batch)
    work.mkdir(parents=True)
    mod.atomic_write_parquet(
        pd.DataFrame(
            [
                {
                    "sample_id": "s1",
                    "original_audio_sha256": "hash_WRONG",
                    "batch": batch,
                    "category": "semantic_risk",
                    "queue_index": 1,
                    "wav_path": str(wav),
                    "candidate_text": "",
                    "gold_text": "x",
                    "gold_kind": "speech",
                    "decision": "accept",
                    "status": "annotated",
                    "annotator_id": "a",
                    "annotated_at": "t",
                }
            ]
        ),
        mod.annotations_path(work),
    )
    out = tmp_path / "out.parquet"
    rc = mod.main(
        [
            "merge",
            "--batch",
            batch,
            "--classified",
            str(classified),
            "--work-dir",
            str(work),
            "--output",
            str(out),
            "--overwrite",
            "--strict",
        ]
    )
    assert rc == 1
