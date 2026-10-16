"""Tag definition library: one-tag-one-file catalog, view, snapshot (042).

Viewing / listing / docs generation must not invoke ASR or acoustic inference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from audio_engine.core.tagging.config import (
    FragmentDef,
    LexiconSpec,
    RuleDef,
    TagDef,
    TaggingConfig,
    _load_yaml,
    _sha256_obj,
)
from audio_engine.core.tagging.paths import resolve_tagging_path


class TagLibraryError(ValueError):
    """Definition / selection contract error with file+entry context."""


@dataclass
class TagLibraryEntry:
    """One tag definition file (may contain multiple rules for the same tag)."""

    tag: TagDef
    rules: list[RuleDef]
    raw: dict[str, Any]
    source_path: str

    def as_view_dict(self, *, expand_fragments: dict[str, FragmentDef] | None = None) -> dict[str, Any]:
        rules_out = []
        for r in self.rules:
            cond = r.condition
            if expand_fragments:
                cond = expand_condition_fragments(cond, expand_fragments)
            rules_out.append(
                {
                    "rule_id": r.rule_id,
                    "version": r.version,
                    "tag_id": r.tag_id,
                    "enabled": r.enabled,
                    "description": r.description,
                    "text_binding": r.text_binding,
                    "condition": cond,
                }
            )
        return {
            "tag_id": self.tag.tag_id,
            "version": self.tag.version,
            "name": self.tag.name,
            "description": self.tag.description,
            "business_definition": self.tag.business_definition,
            "status": self.tag.status_label(),
            "enabled": self.tag.enabled,
            "archived": self.tag.archived,
            "examples": dict(self.tag.examples),
            "dependencies": {
                "fragments": list(self.tag.dependency_fragments),
                "lexicons": list(self.tag.dependency_lexicons),
                "requires_acoustic": self.tag.requires_acoustic,
            },
            "definition_path": self.source_path,
            "rules": rules_out,
        }


@dataclass
class TagLibrary:
    root: Path
    entries: dict[str, TagLibraryEntry] = field(default_factory=dict)

    def list_entries(self, *, include_disabled: bool = True) -> list[TagLibraryEntry]:
        items = list(self.entries.values())
        if not include_disabled:
            items = [e for e in items if e.tag.enabled and not e.tag.archived]
        return sorted(items, key=lambda e: e.tag.tag_id)

    def get(self, tag_id: str) -> TagLibraryEntry:
        if tag_id not in self.entries:
            raise TagLibraryError(f"unknown tag_id in library: {tag_id} (library={self.root})")
        return self.entries[tag_id]


def _status_flags(raw: dict[str, Any], *, path: str) -> tuple[bool, bool]:
    """Return (enabled, archived) from status or legacy flags."""
    if "status" in raw:
        status = str(raw.get("status") or "enabled").strip().lower()
        if status in {"enabled", "active", "on"}:
            return True, False
        if status in {"disabled", "off", "inactive"}:
            return False, False
        if status in {"archived", "archive"}:
            return False, True
        raise TagLibraryError(f"{path}: invalid status {status!r} (expected enabled|disabled|archived)")
    return bool(raw.get("enabled", True)), bool(raw.get("archived", False))


def _parse_tag_file(path: Path, *, expected_tag_id: str | None = None) -> TagLibraryEntry:
    raw = _load_yaml(path)
    if not isinstance(raw, dict):
        raise TagLibraryError(f"{path}: tag definition must be a mapping")
    tid = str(raw.get("tag_id") or "").strip()
    if not tid:
        raise TagLibraryError(f"{path}: tag_id is required")
    stem = path.stem
    if stem.startswith("_"):
        # Example / template files are skipped by catalog loader; still parseable.
        pass
    elif expected_tag_id and tid != expected_tag_id:
        raise TagLibraryError(f"{path}: tag_id {tid!r} must match filename stem {expected_tag_id!r}")
    elif tid != stem and not stem.startswith("_"):
        raise TagLibraryError(f"{path}: tag_id {tid!r} must match filename stem {stem!r}")

    enabled, archived = _status_flags(raw, path=str(path))
    version = str(raw.get("version") or "1")
    examples = raw.get("examples") or {}
    if examples is not None and not isinstance(examples, dict):
        raise TagLibraryError(f"{path}: examples must be a mapping")
    deps = raw.get("dependencies") or {}
    if deps is not None and not isinstance(deps, dict):
        raise TagLibraryError(f"{path}: dependencies must be a mapping")
    dep_frags = [str(x) for x in (deps.get("fragments") or [])]
    dep_lex = [str(x) for x in (deps.get("lexicons") or [])]
    req_ac = deps.get("requires_acoustic")
    if req_ac is None:
        req_ac = raw.get("requires_acoustic")

    tag = TagDef(
        tag_id=tid,
        name=str(raw.get("name") or tid),
        description=str(raw.get("description") or ""),
        version=version,
        enabled=enabled,
        archived=archived,
        business_definition=str(raw.get("business_definition") or raw.get("definition") or ""),
        examples=dict(examples or {}),
        definition_path=str(path.as_posix()),
        requires_acoustic=None if req_ac is None else bool(req_ac),
        dependency_fragments=dep_frags,
        dependency_lexicons=dep_lex,
    )

    rules_raw = raw.get("rules")
    if not isinstance(rules_raw, list) or not rules_raw:
        raise TagLibraryError(f"{path}: rules must be a non-empty list")
    rules: list[RuleDef] = []
    seen_rule: set[str] = set()
    for i, item in enumerate(rules_raw):
        if not isinstance(item, dict):
            raise TagLibraryError(f"{path}: rules[{i}] must be mapping")
        rid = str(item.get("rule_id") or "").strip()
        if not rid:
            raise TagLibraryError(f"{path}: rules[{i}] requires rule_id")
        if rid in seen_rule:
            raise TagLibraryError(f"{path}: duplicate rule_id {rid}")
        seen_rule.add(rid)
        rule_tag = str(item.get("tag_id") or tid).strip()
        if rule_tag != tid:
            raise TagLibraryError(
                f"{path}: rule {rid} tag_id {rule_tag!r} must equal file tag_id {tid!r}"
            )
        cond = item.get("condition")
        if not isinstance(cond, dict):
            raise TagLibraryError(f"{path}: rule {rid} requires condition mapping")
        binding = item.get("text_binding")
        if binding is not None and not isinstance(binding, dict):
            raise TagLibraryError(f"{path}: rule {rid} text_binding must be mapping")
        rules.append(
            RuleDef(
                rule_id=rid,
                version=str(item.get("version") or "1"),
                tag_id=tid,
                condition=cond,
                enabled=bool(item.get("enabled", True)),
                description=str(item.get("description") or ""),
                text_binding=binding,
            )
        )
    return TagLibraryEntry(tag=tag, rules=rules, raw=raw, source_path=str(path.as_posix()))


def load_tag_library(library_dir: str | Path, *, base_dir: Path | None = None) -> TagLibrary:
    root = resolve_tagging_path(library_dir, base_dir=base_dir)
    if not root.is_dir():
        raise TagLibraryError(f"library path is not a directory: {root}")
    entries: dict[str, TagLibraryEntry] = {}
    for path in sorted(root.glob("*.yaml")) + sorted(root.glob("*.yml")):
        if path.name.startswith("_"):
            continue
        entry = _parse_tag_file(path, expected_tag_id=path.stem)
        if entry.tag.tag_id in entries:
            raise TagLibraryError(
                f"duplicate tag_id {entry.tag.tag_id}: {entries[entry.tag.tag_id].source_path} and {entry.source_path}"
            )
        entries[entry.tag.tag_id] = entry
    return TagLibrary(root=root, entries=entries)


def expand_condition_fragments(
    node: Any,
    fragments: dict[str, FragmentDef],
    *,
    _stack: tuple[str, ...] = (),
) -> Any:
    """Inline fragment references for human-readable docs (no evaluation)."""
    if isinstance(node, dict):
        if "fragment" in node and len(node) == 1:
            fid = str(node["fragment"])
            if fid in _stack:
                return {"fragment": fid, "_cycle": True}
            frag = fragments.get(fid)
            if frag is None:
                return {"fragment": fid, "_missing": True}
            return {
                "fragment": fid,
                "version": frag.version,
                "expanded": expand_condition_fragments(
                    frag.condition, fragments, _stack=_stack + (fid,)
                ),
            }
        return {
            k: expand_condition_fragments(v, fragments, _stack=_stack) for k, v in node.items()
        }
    if isinstance(node, list):
        return [expand_condition_fragments(v, fragments, _stack=_stack) for v in node]
    return node


def collect_condition_deps(node: Any) -> tuple[set[str], set[str], bool]:
    """Return (fragment_ids, lexicon_ids, requires_acoustic)."""
    frags: set[str] = set()
    lexs: set[str] = set()
    acoustic = False
    acoustic_caps = {
        "voicemail_raw_hit",  # text but ok
        "vad_speech_present",
        "vad_speech_absent",
        "strict_all_channels_zero",
        "dnsmos_sig_low",
    }

    def walk(n: Any) -> None:
        nonlocal acoustic
        if isinstance(n, dict):
            if "fragment" in n:
                frags.add(str(n["fragment"]))
            if "lexicon" in n:
                lexs.add(str(n["lexicon"]))
            cap = n.get("capability")
            if cap is not None and str(cap) in acoustic_caps and str(cap) != "voicemail_raw_hit":
                acoustic = True
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)

    walk(node)
    return frags, lexs, acoustic


def select_from_library(
    library: TagLibrary,
    select_tags: list[dict[str, Any]],
    *,
    source_label: str,
) -> tuple[dict[str, TagDef], list[RuleDef]]:
    if not select_tags:
        raise TagLibraryError(
            f"{source_label}: library mode requires non-empty select_tags "
            "(refusing to implicitly enable the whole library)"
        )
    tags: dict[str, TagDef] = {}
    rules: list[RuleDef] = []
    seen: set[str] = set()
    for i, item in enumerate(select_tags):
        if isinstance(item, str):
            item = {"tag_id": item}
        if not isinstance(item, dict):
            raise TagLibraryError(f"{source_label}: select_tags[{i}] must be mapping or string")
        tid = str(item.get("tag_id") or "").strip()
        if not tid:
            raise TagLibraryError(f"{source_label}: select_tags[{i}] requires tag_id")
        if tid in seen:
            raise TagLibraryError(f"{source_label}: duplicate select_tags tag_id {tid}")
        seen.add(tid)
        if tid not in library.entries:
            raise TagLibraryError(
                f"{source_label}: select_tags references unknown tag_id {tid} "
                f"(library={library.root})"
            )
        entry = library.entries[tid]
        want_ver = item.get("version")
        if want_ver is not None and str(want_ver) != entry.tag.version:
            raise TagLibraryError(
                f"{source_label}: tag {tid} version mismatch: "
                f"requested {want_ver!r}, library has {entry.tag.version!r} "
                f"({entry.source_path})"
            )
        if entry.tag.archived:
            raise TagLibraryError(
                f"{source_label}: cannot select archived tag {tid} ({entry.source_path})"
            )
        if not entry.tag.enabled:
            raise TagLibraryError(
                f"{source_label}: cannot select disabled tag {tid} ({entry.source_path})"
            )
        tags[tid] = entry.tag
        rules.extend(entry.rules)
    return tags, rules


def lexicon_full_dict(lx: LexiconSpec) -> dict[str, Any]:
    return {
        "lexicon_id": lx.lexicon_id,
        "version": lx.version,
        "path": lx.path,
        "content_hash": lx.content_hash,
        "flags": lx.flags,
        "entries": [
            {
                "entry_id": e.entry_id,
                "pattern": e.pattern,
                "entry_type": e.entry_type,
                "match_mode": e.match_mode,
                "flags": e.flags,
            }
            for e in lx.entries
        ],
    }


def build_definition_snapshot(cfg: TaggingConfig) -> dict[str, Any]:
    """Full effective definitions + dependencies for publish / resume (042)."""
    used_frags: set[str] = set()
    used_lex: set[str] = set()
    for rule in cfg.rules:
        fr, lx, _ = collect_condition_deps(rule.condition)
        used_frags |= fr
        used_lex |= lx
    # Recursively include fragment→fragment refs
    pending = set(used_frags)
    while pending:
        fid = pending.pop()
        frag = cfg.fragments.get(fid)
        if frag is None:
            continue
        fr, lx, _ = collect_condition_deps(frag.condition)
        for nxt in fr:
            if nxt not in used_frags:
                used_frags.add(nxt)
                pending.add(nxt)
        used_lex |= lx

    fragments_payload = {
        fid: {
            "fragment_id": f.fragment_id,
            "version": f.version,
            "description": f.description,
            "condition": f.condition,
        }
        for fid, f in sorted(cfg.fragments.items())
        if fid in used_frags or not used_frags
    }
    # Always include all loaded fragments when selection used subset? Prefer used+declared.
    if not fragments_payload:
        fragments_payload = {
            fid: {
                "fragment_id": f.fragment_id,
                "version": f.version,
                "description": f.description,
                "condition": f.condition,
            }
            for fid, f in sorted(cfg.fragments.items())
        }

    lexicons_payload = {
        lid: lexicon_full_dict(lx)
        for lid, lx in sorted(cfg.lexicons.items())
        if lid in used_lex or lid in cfg.lexicons
    }

    tags_payload = {
        tid: {
            "tag_id": t.tag_id,
            "name": t.name,
            "description": t.description,
            "version": t.version,
            "enabled": t.enabled,
            "archived": t.archived,
            "business_definition": t.business_definition,
            "examples": dict(t.examples),
            "definition_path": t.definition_path,
            "requires_acoustic": t.requires_acoustic,
            "dependency_fragments": list(t.dependency_fragments),
            "dependency_lexicons": list(t.dependency_lexicons),
        }
        for tid, t in sorted(cfg.tags.items())
    }
    rules_payload = [
        {
            "rule_id": r.rule_id,
            "version": r.version,
            "tag_id": r.tag_id,
            "enabled": r.enabled,
            "description": r.description,
            "text_binding": r.text_binding,
            "condition": r.condition,
        }
        for r in cfg.rules
    ]
    payload = {
        "schema": "tagging_definition_snapshot_v1",
        "policy_version": cfg.policy_version,
        "content_fingerprint": cfg.content_fingerprint,
        "source_path": cfg.source_path,
        "evidence_config_path": cfg.evidence_config_path,
        "evidence": cfg.evidence.as_dict(),
        "evidence_mode": cfg.evidence_mode,
        "acoustic": dict(cfg.acoustic),
        "library_mode": cfg.library_mode,
        "library_path": cfg.library_path,
        "select_tags": list(cfg.select_tags),
        "tags": tags_payload,
        "rules": rules_payload,
        "fragments": fragments_payload,
        "lexicons": lexicons_payload,
    }
    payload["snapshot_fingerprint"] = _sha256_obj(payload)
    return payload


def save_definition_snapshot(path: Path, cfg: TaggingConfig) -> dict[str, Any]:
    snap = build_definition_snapshot(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(snap, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    return snap


def load_definition_snapshot(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"definition snapshot not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TagLibraryError(f"invalid definition snapshot: {p}")
    return data


def assert_resume_compatible(progress: dict[str, Any] | None, cfg: TaggingConfig) -> None:
    if not progress:
        return
    old = progress.get("definition_fingerprint") or progress.get("config_fingerprint")
    if not old:
        return
    if str(old) != str(cfg.content_fingerprint):
        raise TagLibraryError(
            "definition drift detected on resume: "
            f"progress fingerprint={old}, current={cfg.content_fingerprint}. "
            "Refuse to mix old progress with changed tag definitions; "
            "start a new run or recompute."
        )


# Re-export helpers used by config loader to avoid circular import issues at call sites.
__all__ = [
    "TagLibrary",
    "TagLibraryEntry",
    "TagLibraryError",
    "assert_resume_compatible",
    "build_definition_snapshot",
    "collect_condition_deps",
    "expand_condition_fragments",
    "load_definition_snapshot",
    "load_tag_library",
    "save_definition_snapshot",
    "select_from_library",
]
