"""044 phase B: tagging is decoupled from selection_v3 business config.

Guards:
- core/tagging (except the migrate boundary) never imports selection_v3 / v2 / engine,
  including lazy function-level imports;
- core/evidence never imports tagging or selection business modules;
- free_tags_v2 loads and evaluates with every ``configs/selection`` file unreadable;
- the legacy ``selection_config`` key is refused with a migration hint;
- evidence constants stay in parity with ``selection_v3.types``;
- shared acoustic store / scorer semantics (no overwrite of success, no fake compute).
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
TAGGING_ROOT = ROOT / "src" / "audio_engine" / "core" / "tagging"
EVIDENCE_ROOT = ROOT / "src" / "audio_engine" / "core" / "evidence"
FREE_TAGS = ROOT / "configs" / "tagging" / "zh_asr_free_tags_v2.yaml"
SELECTION_V3_YAML = ROOT / "configs" / "selection" / "zh_asr_five_class_v3.yaml"

# Only these tagging modules may touch the selection config: they implement the explicit
# one-way migration (``migrate-config``) from an old selection YAML.
_MIGRATE_ALLOWLIST = {"migrate.py", "evidence_config.py"}
# What even the migrate boundary may import from selection business code.
_MIGRATE_ALLOWED_IMPORTS = {"audio_engine.core.selection_v3.config"}

_FORBIDDEN_FOR_TAGGING = (
    "audio_engine.core.selection_v3",
    "audio_engine.core.selection_v2",
    "audio_engine.core.selection_engine",
)
_FORBIDDEN_FOR_EVIDENCE = _FORBIDDEN_FOR_TAGGING + ("audio_engine.core.tagging",)


_KEYS = ("glm_1", "glm_2", "sensevoice_1", "sensevoice_2", "qwen_1", "qwen_2")


def _sample(sid: str, text_by_family: dict[str, str] | None = None, quality: dict | None = None):
    from audio_engine.core.sample import Sample

    fam = text_by_family or {}
    transcripts = {}
    for key in _KEYS:
        text = fam.get(key.rsplit("_", 1)[0], "")
        transcripts[key] = {"text": text, "status": "success", "raw_text": text}
    return Sample(
        id=sid,
        source_path=f"{sid}.wav",
        sha256="b" * 64,
        duration=2.0,
        transcripts=transcripts,
        quality=dict(quality or {}),
        audio={"resampled_16k": f"{sid}.wav"},
    )


def _py_files(root: Path):
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _imports_of(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: list[str] = []
    for node in ast.walk(tree):  # includes lazy imports inside functions
        if isinstance(node, ast.Import):
            out.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.append(node.module)
    return out


def _hits(mod: str, prefixes: tuple[str, ...]) -> bool:
    return any(mod == p or mod.startswith(p + ".") for p in prefixes)


def test_tagging_has_no_selection_business_imports_outside_migrate():
    offenders: list[str] = []
    for path in _py_files(TAGGING_ROOT):
        for mod in _imports_of(path):
            if not _hits(mod, _FORBIDDEN_FOR_TAGGING):
                continue
            if path.name in _MIGRATE_ALLOWLIST and mod in _MIGRATE_ALLOWED_IMPORTS:
                continue
            offenders.append(f"{path.relative_to(ROOT)} -> {mod}")
    assert not offenders, "tagging imports selection business:\n" + "\n".join(offenders)


def test_tagging_migrate_boundary_only_imports_config_lazily():
    """evidence_config.py may only import SelectionV3Config inside the migrate function."""
    path = TAGGING_ROOT / "evidence_config.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:  # module level only
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, "module", None) or ""
            assert not _hits(mod, _FORBIDDEN_FOR_TAGGING), f"module-level import of {mod}"


def test_evidence_has_no_reverse_dependency_on_tagging_or_selection():
    offenders: list[str] = []
    for path in _py_files(EVIDENCE_ROOT):
        for mod in _imports_of(path):
            if _hits(mod, _FORBIDDEN_FOR_EVIDENCE):
                offenders.append(f"{path.relative_to(ROOT)} -> {mod}")
    assert not offenders, "evidence reverse deps:\n" + "\n".join(offenders)


def test_free_tags_v2_runs_in_clean_process_without_selection_modules_or_yaml():
    """Load + evaluate in a subprocess where configs/selection reads fail and selection_v3
    must never be imported."""
    code = textwrap.dedent(
        f"""
        import builtins, io, sys
        from pathlib import Path

        _real_open = builtins.open
        _real_read_text = Path.read_text
        _real_read_bytes = Path.read_bytes

        def _blocked(p):
            s = str(p).replace("\\\\", "/")
            return "configs/selection/" in s

        def _open(file, *a, **k):
            if isinstance(file, (str, Path)) and _blocked(file):
                raise AssertionError(f"selection yaml read at runtime: {{file}}")
            return _real_open(file, *a, **k)

        def _read_text(self, *a, **k):
            if _blocked(self):
                raise AssertionError(f"selection yaml read at runtime: {{self}}")
            return _real_read_text(self, *a, **k)

        def _read_bytes(self, *a, **k):
            if _blocked(self):
                raise AssertionError(f"selection yaml read at runtime: {{self}}")
            return _real_read_bytes(self, *a, **k)

        builtins.open = _open
        io.open = _open
        Path.read_text = _read_text
        Path.read_bytes = _read_bytes

        from audio_engine.core.sample import Sample
        from audio_engine.core.tagging.config import load_tagging_config
        from audio_engine.core.tagging.engine import MultiTagEngine

        cfg = load_tagging_config(r"{FREE_TAGS.as_posix()}")
        assert cfg.policy_version == "free_tags_v2", cfg.policy_version
        assert cfg.evidence.content_digest
        engine = MultiTagEngine(cfg)
        sample = Sample(
            id="s1",
            source_path="s1.wav",
            sha256="b" * 64,
            duration=2.0,
            transcripts={{
                k: {{"text": "好", "status": "success", "raw_text": "好"}}
                for k in ("glm_1", "glm_2", "sensevoice_1", "sensevoice_2", "qwen_1", "qwen_2")
            }},
        )
        result = engine.evaluate_sample(sample, dry_run=True)
        assert result is not None
        bad = sorted(m for m in sys.modules if m.startswith("audio_engine.core.selection"))
        assert not bad, f"selection modules imported at runtime: {{bad}}"
        print("OK")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert proc.returncode == 0 and "OK" in proc.stdout, proc.stdout + proc.stderr


def test_committed_tagging_configs_have_no_selection_reference():
    for rel in (
        "configs/tagging/zh_asr_free_tags_v2.yaml",
        "configs/tagging/zh_asr_multi_tag_v1.yaml",
        "configs/tagging/evidence/zh_asr_families_v2.yaml",
    ):
        text = (ROOT / rel).read_text(encoding="utf-8")
        raw = yaml.safe_load(text)
        assert "selection_config" not in raw and "selection_config_path" not in raw, rel
        assert "configs/selection/" not in text.replace("\\", "/"), rel


def test_legacy_selection_config_key_rejected_with_migrate_hint(tmp_path: Path):
    from audio_engine.core.tagging.config import load_tagging_config

    legacy = tmp_path / "legacy.yaml"
    legacy.write_text(
        "policy_version: x\n"
        "selection_config: configs/selection/zh_asr_five_class_v3.yaml\n"
        "tags: [{tag_id: t, name: t}]\n"
        "rules: [{rule_id: r, tag_id: t, condition: {capability: true}}]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="migrate-config"):
        load_tagging_config(legacy)


def test_selection_key_rejected_even_with_evidence_present(tmp_path: Path):
    from audio_engine.core.tagging.config import load_tagging_config

    raw = yaml.safe_load(FREE_TAGS.read_text(encoding="utf-8"))
    raw["selection_config"] = "configs/selection/zh_asr_five_class_v3.yaml"
    raw["evidence_config"] = str((ROOT / raw["evidence_config"]).as_posix())
    target = tmp_path / "mixed.yaml"
    target.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="selection_config"):
        load_tagging_config(target)


def test_fingerprint_covers_full_evidence_content(tmp_path: Path):
    from audio_engine.core.tagging.config import load_tagging_config

    base = load_tagging_config(FREE_TAGS)

    raw = yaml.safe_load(FREE_TAGS.read_text(encoding="utf-8"))
    ev_src = ROOT / raw["evidence_config"]
    ev = yaml.safe_load(ev_src.read_text(encoding="utf-8"))
    ev["critical_short_responses"] = list(ev["critical_short_responses"]) + ["绝对不行"]
    # The copy lives outside the repo: make every referenced path absolute.
    for spec in ev["classify_text"]["echo"].values():
        for k, v in list(spec.items()):
            spec[k] = (ROOT / v).as_posix()
    for key in ("voicemail_patterns_path", "semantic_lexicon_path"):
        ev[key] = (ROOT / ev[key]).as_posix()
    for key in ("audio_energy", "silero_vad", "dnsmos_decision", "dnsmos_scoring"):
        if isinstance(ev.get(key), dict) and ev[key].get("config_path"):
            ev[key]["config_path"] = (ROOT / ev[key]["config_path"]).as_posix()
    ev_path = tmp_path / "evidence.yaml"
    ev_path.write_text(yaml.safe_dump(ev, allow_unicode=True), encoding="utf-8")
    raw["evidence_config"] = ev_path.as_posix()
    raw["library"]["path"] = (ROOT / raw["library"]["path"]).as_posix()
    raw["fragments"]["path"] = (ROOT / raw["fragments"]["path"]).as_posix()
    for key, value in list(raw.get("lexicons", {}).items()):
        if isinstance(value, dict) and "path" in value:
            value["path"] = (ROOT / value["path"]).as_posix()
        elif isinstance(value, str):
            raw["lexicons"][key] = (ROOT / value).as_posix()
    for key, value in list(raw.get("acoustic", {}).items()):
        raw["acoustic"][key] = (ROOT / value).as_posix()
    cfg_path = tmp_path / "tagging.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    changed = load_tagging_config(cfg_path)
    assert changed.content_fingerprint != base.content_fingerprint


def test_evidence_matches_effective_selection_v3_values():
    """Independent evidence yaml == effective five_class_v3 values (migration parity)."""
    from audio_engine.core.selection_v3.config import SelectionV3Config
    from audio_engine.core.tagging.config import load_tagging_config

    sel = SelectionV3Config.from_yaml(SELECTION_V3_YAML)
    ev = load_tagging_config(FREE_TAGS).evidence
    for name in (
        "model_families",
        "teacher_families",
        "target_family",
        "expected_runs_per_family",
        "run_aliases",
        "punctuation_to_strip",
        "critical_short_responses",
        "short_polarity_max_han_chars",
        "classify_text_policy",
        "classify_text_keep_digits",
        "negative_phrases",
        "positive_phrases",
        "critical_tokens",
        "filler_phrases",
        "affirmation_phrases",
        "profanity_or_reject",
        "dnsmos_weak_sig",
        "family_threshold",
    ):
        assert getattr(ev, name) == getattr(sel, name), name
    assert {k: v.fingerprint for k, v in ev.classify_text_echo.items()} == {
        k: v.fingerprint for k, v in sel.classify_text_echo.items()
    }


def test_route_config_protocol_satisfied_by_both_configs():
    from audio_engine.core.evidence.family_routes import collect_route_views
    from audio_engine.core.selection_v3.config import SelectionV3Config
    from audio_engine.core.tagging.config import load_tagging_config

    sel = SelectionV3Config.from_yaml(SELECTION_V3_YAML)
    ev = load_tagging_config(FREE_TAGS).evidence
    sample = _sample("p1", {"qwen": "不需要", "glm": "需要", "sensevoice": "需要"})
    a = [(v.run_id, v.family, v.classify_text, v.comparison_text, str(v.status)) for v in collect_route_views(sample, sel)]
    b = [(v.run_id, v.family, v.classify_text, v.comparison_text, str(v.status)) for v in collect_route_views(sample, ev)]
    assert a == b


def test_evidence_types_parity_with_selection_v3_types():
    from audio_engine.core.evidence import types as ev_types
    from audio_engine.core.selection_v3 import types as sel_types

    shared = [
        n
        for n in dir(ev_types)
        if n.isupper() and not n.startswith("FACT_") and not n.startswith("SCHEDULE_")
        and hasattr(sel_types, n)
    ]
    assert len(shared) > 40, "parity check must cover the mirrored constants"
    drift = {n: (getattr(ev_types, n), getattr(sel_types, n)) for n in shared if getattr(ev_types, n) != getattr(sel_types, n)}
    assert not drift, drift
    # Every constant mirrored from selection must exist there (no silent renames).
    for n in dir(ev_types):
        if n.startswith(("RUN_STATUS_", "FAMILY_STATE_", "ROUTE_", "EXCLUSION_", "DNSMOS_STATUS_", "NOISE_BAND_")):
            assert hasattr(sel_types, n), n


# --- shared store / scorer ---------------------------------------------------------


def test_not_required_never_overwrites_success():
    from audio_engine.core.evidence.store import merge_quality_fields

    existing = {"vad_status": "success", "vad_speech_state": "present", "dnsmos_status": "failed"}
    merged = merge_quality_fields(existing, {"vad_status": "not_required", "dnsmos_status": "not_required"})
    assert merged["vad_status"] == "success"
    assert merged["vad_speech_state"] == "present"
    assert merged["dnsmos_status"] == "failed"  # schedule label does not erase a failure either
    # a later real success does upgrade a failure
    upgraded = merge_quality_fields(merged, {"dnsmos_status": "success", "dnsmos_sig": 3.1})
    assert upgraded["dnsmos_status"] == "success" and upgraded["dnsmos_sig"] == 3.1
    # not_required is read as "no valid evidence", never as success
    from audio_engine.core.evidence.store import fact_from_quality

    assert fact_from_quality("vad", {"vad_status": "not_required"}).status == "not_computed"


def test_scorer_rejects_empty_result_and_counts_only_real_calls(tmp_path: Path):
    from audio_engine.core.evidence.scoring import RealAcousticScorer

    calls: list[str] = []

    def vad_ok(audio_path, cfg):
        calls.append("vad")
        return {"vad_status": "success", "vad_speech_state": "present"}

    def energy_empty(audio_path, cfg):
        calls.append("energy")
        return {}  # passthrough-like empty result

    def dnsmos_boom(audio_path, cfg):
        calls.append("dnsmos")
        raise RuntimeError("onnx exploded")

    scorer = RealAcousticScorer(backends={"vad": vad_ok, "energy": energy_empty, "dnsmos": dnsmos_boom})
    facts = scorer.compute("s1", {"vad", "energy", "dnsmos"}, audio_path_for=lambda dep: "x.wav")

    assert facts["vad"].status == "success"
    assert facts["energy"].status == "failed" and "empty" in (facts["energy"].error or "")
    assert facts["dnsmos"].status == "failed" and "onnx exploded" in (facts["dnsmos"].error or "")
    ledger = scorer.ledger
    assert dict(ledger.planned) == {"vad": 1, "energy": 1, "dnsmos": 1}
    assert ledger.actual["vad"] == 1
    assert ledger.actual["energy"] == 0 and ledger.rejected["energy"] == 1  # empty != computed
    assert ledger.actual["dnsmos"] == 1 and ledger.failed["dnsmos"] == 1

    # SUCCESS is cached: second request does not run the backend again.
    calls.clear()
    again = scorer.compute("s1", {"vad"}, audio_path_for=lambda dep: "x.wav")
    assert again["vad"].status == "success" and calls == []
    assert ledger.cache_hits["vad"] == 1


def test_scorer_reports_unavailable_backend_without_fake_success():
    from audio_engine.core.evidence.scoring import RealAcousticScorer

    scorer = RealAcousticScorer()  # default backends, no config paths
    facts = scorer.compute("s2", {"vad", "energy", "dnsmos"}, audio_path_for=lambda dep: "missing.wav")
    for dep in ("vad", "energy", "dnsmos"):
        assert facts[dep].status == "failed", dep
        assert facts[dep].error, dep
        assert scorer.ledger.unavailable[dep] == 1
        assert scorer.ledger.actual[dep] == 0


def test_real_adapter_does_not_treat_empty_passthrough_as_compute():
    from audio_engine.core.tagging.adapters import QualityPassthroughAdapter, RealAcousticAdapter
    from audio_engine.core.evidence.scoring import RealAcousticScorer

    sample = _sample("a1")
    passthrough = QualityPassthroughAdapter()
    view = passthrough.compute(sample, deps={"vad"}, params={})
    assert view.vad_status in (None, "", "not_computed") or view.vad_status != "success"

    scorer = RealAcousticScorer(
        backends={"vad": lambda p, c: {"vad_status": "success", "vad_speech_state": "present"}}
    )
    adapter = RealAcousticAdapter(scorer=scorer)
    sample = _sample("a2", quality={"vad_status": "not_required"})
    out = adapter.compute(sample, deps={"vad"}, params={})
    assert out.vad_status == "success"
    assert adapter.call_count("vad") == 1
    # already-success quality is reused, not recomputed
    adapter.compute(sample, deps={"vad"}, params={})
    assert adapter.call_count("vad") == 1
