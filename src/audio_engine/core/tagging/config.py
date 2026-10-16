"""Tagging config: tags, rules, fragments, lexicons (040)."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from audio_engine.core.tagging.evidence_config import (
    TagEvidenceConfig,
    load_tag_evidence_config,
)
from audio_engine.core.tagging.types import POLICY_VERSION


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_obj(obj: Any) -> str:
    return _sha256_text(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str))


def _load_yaml(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"tagging config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw


@dataclass
class LexiconEntry:
    entry_id: str
    pattern: str
    entry_type: str = "regex"  # regex | literal
    match_mode: str = "search"  # search | fullmatch
    flags: str = ""
    compiled: re.Pattern[str] | None = None


@dataclass
class LexiconSpec:
    lexicon_id: str
    version: str
    path: str
    content_hash: str
    flags: str = "IGNORECASE"
    entries: list[LexiconEntry] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "lexicon_id": self.lexicon_id,
            "version": self.version,
            "path": self.path,
            "content_hash": self.content_hash,
            "flags": self.flags,
            "entry_count": len(self.entries),
        }


@dataclass
class TagDef:
    tag_id: str
    name: str
    description: str = ""
    version: str = "1"
    enabled: bool = True
    archived: bool = False
    business_definition: str = ""
    examples: dict[str, Any] = field(default_factory=dict)
    definition_path: str = ""
    requires_acoustic: bool | None = None
    dependency_fragments: list[str] = field(default_factory=list)
    dependency_lexicons: list[str] = field(default_factory=list)

    def status_label(self) -> str:
        if self.archived:
            return "archived"
        if not self.enabled:
            return "disabled"
        return "enabled"

    def as_dict(self) -> dict[str, Any]:
        return {
            "tag_id": self.tag_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "enabled": self.enabled,
            "archived": self.archived,
            "status": self.status_label(),
            "business_definition": self.business_definition,
            "examples": dict(self.examples),
            "definition_path": self.definition_path,
            "requires_acoustic": self.requires_acoustic,
            "dependency_fragments": list(self.dependency_fragments),
            "dependency_lexicons": list(self.dependency_lexicons),
        }


@dataclass
class FragmentDef:
    fragment_id: str
    version: str
    condition: dict[str, Any]
    description: str = ""


@dataclass
class RuleDef:
    rule_id: str
    version: str
    tag_id: str
    condition: dict[str, Any]
    enabled: bool = True
    description: str = ""
    text_binding: dict[str, Any] | None = None


@dataclass
class TaggingConfig:
    policy_version: str
    evidence: TagEvidenceConfig
    evidence_mode: str  # existing | compute_acoustic
    tags: dict[str, TagDef]
    rules: list[RuleDef]
    fragments: dict[str, FragmentDef]
    lexicons: dict[str, LexiconSpec]
    evidence_config_path: str = ""
    # Deprecated alias kept for snapshot readability of pre-044 runs.
    selection_config_path: str = ""
    work_dir_template: str = "runs/tagging/{batch_id}/{policy_version}"
    acoustic: dict[str, Any] = field(default_factory=dict)
    content_fingerprint: str = ""
    source_path: str = ""
    library_mode: bool = False
    library_path: str = ""
    select_tags: list[dict[str, Any]] = field(default_factory=list)

    @property
    def selection(self) -> TagEvidenceConfig:
        """Compatibility alias: historical code used ``config.selection`` for family/route."""
        return self.evidence

    def enabled_rules(self) -> list[RuleDef]:
        out: list[RuleDef] = []
        for rule in self.rules:
            if not rule.enabled:
                continue
            tag = self.tags.get(rule.tag_id)
            if tag is None or not tag.enabled or tag.archived:
                continue
            out.append(rule)
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "evidence_config_path": self.evidence_config_path,
            "evidence": self.evidence.as_dict(),
            "evidence_mode": self.evidence_mode,
            "content_fingerprint": self.content_fingerprint,
            "source_path": self.source_path,
            "library_mode": self.library_mode,
            "library_path": self.library_path,
            "select_tags": list(self.select_tags),
            "tags": {tid: t.as_dict() for tid, t in self.tags.items()},
            "rules": [
                {
                    "rule_id": r.rule_id,
                    "version": r.version,
                    "tag_id": r.tag_id,
                    "enabled": r.enabled,
                    "description": r.description,
                    "text_binding": r.text_binding,
                    "condition": r.condition,
                }
                for r in self.rules
            ],
            "fragments": {
                fid: {
                    "fragment_id": f.fragment_id,
                    "version": f.version,
                    "description": f.description,
                    "condition": f.condition,
                }
                for fid, f in self.fragments.items()
            },
            "lexicons": {lid: lx.as_dict() for lid, lx in self.lexicons.items()},
            "acoustic": dict(self.acoustic),
        }


def _compile_flags(flags_name: str) -> int:
    name = str(flags_name or "").upper()
    flags = 0
    if "IGNORECASE" in name or "I" == name:
        flags |= re.IGNORECASE
    if "MULTILINE" in name:
        flags |= re.MULTILINE
    if "DOTALL" in name:
        flags |= re.DOTALL
    return flags


def load_lexicon(path: str | Path, *, lexicon_id: str | None = None) -> LexiconSpec:
    p = Path(path)
    raw = _load_yaml(p)
    if not isinstance(raw, dict):
        raise ValueError(f"lexicon must be mapping: {p}")
    lid = str(lexicon_id or raw.get("lexicon_id") or raw.get("name") or p.stem)
    version = str(raw.get("version") or "1")
    flags_name = str(raw.get("flags") or "IGNORECASE")
    base_flags = _compile_flags(flags_name)
    text = p.read_text(encoding="utf-8")
    content_hash = _sha256_text(text)
    entries: list[LexiconEntry] = []
    items = raw.get("entries")
    if items is None:
        # voicemail-compatible: patterns: [str, ...]
        patterns = raw.get("patterns") or []
        items = []
        for i, pat in enumerate(patterns):
            items.append(
                {
                    "entry_id": f"{lid}_{i+1}",
                    "pattern": pat,
                    "type": "regex",
                    "match_mode": "search",
                }
            )
    if not isinstance(items, list):
        raise ValueError(f"lexicon entries must be list: {p}")
    for i, item in enumerate(items):
        if isinstance(item, str):
            entry = {
                "entry_id": f"{lid}_{i+1}",
                "pattern": item,
                "type": "regex",
                "match_mode": "search",
            }
        elif isinstance(item, dict):
            entry = item
        else:
            raise ValueError(f"invalid lexicon entry at {p}#{i}")
        eid = str(entry.get("entry_id") or f"{lid}_{i+1}")
        pattern = str(entry.get("pattern") or entry.get("text") or "").strip()
        if not pattern:
            raise ValueError(f"empty lexicon pattern: {p} entry {eid}")
        etype = str(entry.get("type") or entry.get("entry_type") or "regex").lower()
        mode = str(entry.get("match_mode") or "search").lower()
        if mode not in {"search", "fullmatch"}:
            raise ValueError(f"invalid match_mode {mode} in {p} entry {eid}")
        eflags_name = str(entry.get("flags") or flags_name)
        eflags = _compile_flags(eflags_name)
        try:
            if etype == "literal":
                compiled = re.compile(re.escape(pattern), eflags or base_flags)
            elif etype == "regex":
                compiled = re.compile(pattern, eflags or base_flags)
            else:
                raise ValueError(f"unknown entry type {etype} in {p} entry {eid}")
        except re.error as exc:
            raise ValueError(f"invalid regex in lexicon {p} entry {eid}: {exc}") from exc
        entries.append(
            LexiconEntry(
                entry_id=eid,
                pattern=pattern,
                entry_type=etype,
                match_mode=mode,
                flags=eflags_name,
                compiled=compiled,
            )
        )
    return LexiconSpec(
        lexicon_id=lid,
        version=version,
        path=str(p.as_posix()),
        content_hash=content_hash,
        flags=flags_name,
        entries=entries,
    )


def _walk_condition_refs(node: Any, *, fragments: set[str], capabilities: set[str]) -> None:
    if isinstance(node, dict):
        if "fragment" in node:
            fragments.add(str(node["fragment"]))
        if "capability" in node:
            capabilities.add(str(node["capability"]))
        for key in ("all", "any"):
            if key in node and isinstance(node[key], list):
                for child in node[key]:
                    _walk_condition_refs(child, fragments=fragments, capabilities=capabilities)
        if "not" in node:
            _walk_condition_refs(node["not"], fragments=fragments, capabilities=capabilities)
        if "at_least" in node:
            body = node["at_least"]
            if isinstance(body, dict):
                _walk_condition_refs(body.get("of"), fragments=fragments, capabilities=capabilities)
        for k, v in node.items():
            if k in {"all", "any", "not", "at_least", "fragment", "capability"}:
                continue
            _walk_condition_refs(v, fragments=fragments, capabilities=capabilities)
    elif isinstance(node, list):
        for child in node:
            _walk_condition_refs(child, fragments=fragments, capabilities=capabilities)


def _detect_fragment_cycles(fragments: dict[str, FragmentDef]) -> None:
    graph: dict[str, set[str]] = {}
    for fid, frag in fragments.items():
        refs: set[str] = set()
        caps: set[str] = set()
        _walk_condition_refs(frag.condition, fragments=refs, capabilities=caps)
        graph[fid] = refs

    visiting: set[str] = set()
    visited: set[str] = set()

    def dfs(node: str, stack: list[str]) -> None:
        if node in visiting:
            cycle = " -> ".join(stack + [node])
            raise ValueError(f"fragment cycle detected: {cycle}")
        if node in visited:
            return
        visiting.add(node)
        for nxt in graph.get(node, ()):
            if nxt not in fragments:
                raise ValueError(f"fragment {node} references unknown fragment {nxt}")
            dfs(nxt, stack + [node])
        visiting.remove(node)
        visited.add(node)

    for fid in fragments:
        dfs(fid, [])


_KNOWN_CAPABILITIES = frozenset(
    {
        "family_all_dual_success",
        "family_dual_success",
        "family_has_successful_raw_text",
        "cross_family_text_disagreement",
        "family_state",
        "family_states_quantified",
        "texts_equivalent",
        "texts_not_equivalent",
        "lexicon_match",
        "single_han_char",
        "text_length",
        "han_char_count",
        "is_critical_short_response",
        "polarity_conflict",
        "critical_short_ambiguity",
        "voicemail_raw_hit",
        "vad_speech_present",
        "vad_speech_absent",
        "strict_all_channels_zero",
        "dnsmos_sig_low",
        "true",
        "false",
    }
)


def validate_condition_tree(
    node: Any,
    *,
    fragments: dict[str, FragmentDef],
    lexicons: dict[str, LexiconSpec],
    families: set[str],
    path: str = "condition",
) -> None:
    if not isinstance(node, dict):
        raise ValueError(f"{path}: condition must be mapping")
    keys = set(node.keys())
    combinators = {"all", "any", "not", "at_least", "fragment", "capability"}
    present = keys & combinators
    if len(present) != 1:
        # allow capability node with extra params
        if "capability" in keys:
            present = {"capability"}
        elif "fragment" in keys:
            present = {"fragment"}
        else:
            raise ValueError(f"{path}: need exactly one of {sorted(combinators)}")

    if "all" in present:
        items = node["all"]
        if not isinstance(items, list) or not items:
            raise ValueError(f"{path}.all must be non-empty list")
        for i, child in enumerate(items):
            validate_condition_tree(
                child, fragments=fragments, lexicons=lexicons, families=families, path=f"{path}.all[{i}]"
            )
        return
    if "any" in present:
        items = node["any"]
        if not isinstance(items, list) or not items:
            raise ValueError(f"{path}.any must be non-empty list")
        for i, child in enumerate(items):
            validate_condition_tree(
                child, fragments=fragments, lexicons=lexicons, families=families, path=f"{path}.any[{i}]"
            )
        return
    if "not" in present:
        validate_condition_tree(
            node["not"], fragments=fragments, lexicons=lexicons, families=families, path=f"{path}.not"
        )
        return
    if "at_least" in present:
        body = node["at_least"]
        if not isinstance(body, dict):
            raise ValueError(f"{path}.at_least must be mapping")
        n = body.get("n")
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"{path}.at_least.n must be positive int")
        of = body.get("of")
        if not isinstance(of, list) or not of:
            raise ValueError(f"{path}.at_least.of must be non-empty list")
        for i, child in enumerate(of):
            validate_condition_tree(
                child, fragments=fragments, lexicons=lexicons, families=families, path=f"{path}.at_least.of[{i}]"
            )
        return
    if "fragment" in present:
        fid = str(node["fragment"])
        if fid not in fragments:
            raise ValueError(f"{path}: unknown fragment {fid}")
        return

    cap = str(node.get("capability") or "")
    if cap not in _KNOWN_CAPABILITIES:
        raise ValueError(f"{path}: unknown capability {cap}")
    if cap == "family_has_successful_raw_text":
        family = node.get("family")
        if not isinstance(family, str) or not family.strip():
            raise ValueError(f"{path}: family_has_successful_raw_text requires family")
    if cap == "cross_family_text_disagreement":
        family_names = node.get("families")
        if (
            not isinstance(family_names, list)
            or len(family_names) < 2
            or any(not isinstance(family, str) or not family.strip() for family in family_names)
            or len(set(family_names)) != len(family_names)
        ):
            raise ValueError(
                f"{path}: cross_family_text_disagreement requires at least two distinct families"
            )
        if "include_empty" in node and not isinstance(node["include_empty"], bool):
            raise ValueError(f"{path}.include_empty must be bool")
    for key in ("family", "families", "left", "right", "text"):
        # family existence checked when present
        pass
    fam = node.get("family")
    if fam is not None and str(fam) not in families:
        raise ValueError(f"{path}: unknown family {fam}")
    fams = node.get("families")
    if fams is not None:
        if not isinstance(fams, list):
            raise ValueError(f"{path}.families must be list")
        for f in fams:
            if str(f) not in families:
                raise ValueError(f"{path}: unknown family {f}")
    lex = node.get("lexicon")
    if lex is not None and str(lex) not in lexicons:
        raise ValueError(f"{path}: unknown lexicon {lex}")
    if cap == "lexicon_match" and not lex:
        raise ValueError(f"{path}: lexicon_match requires lexicon")


def _load_fragments(raw: Any, *, base_dir: Path) -> dict[str, FragmentDef]:
    from audio_engine.core.tagging.paths import resolve_tagging_path

    out: dict[str, FragmentDef] = {}
    if raw is None:
        return out
    items: list[Any]
    if isinstance(raw, dict) and "path" in raw:
        frag_path = resolve_tagging_path(str(raw["path"]), base_dir=base_dir)
        data = _load_yaml(frag_path)
        items = data.get("fragments") if isinstance(data, dict) else data
    elif isinstance(raw, dict) and "fragments" in raw:
        items = raw["fragments"]
    elif isinstance(raw, list):
        items = raw
    else:
        items = []
    if not isinstance(items, list):
        raise ValueError("fragments must be a list")
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("fragment entry must be mapping")
        fid = str(item.get("fragment_id") or item.get("id") or "").strip()
        if not fid:
            raise ValueError("fragment requires fragment_id")
        version = str(item.get("version") or "1")
        cond = item.get("condition")
        if not isinstance(cond, dict):
            raise ValueError(f"fragment {fid} requires condition mapping")
        out[fid] = FragmentDef(
            fragment_id=fid,
            version=version,
            condition=cond,
            description=str(item.get("description") or ""),
        )
    return out


def _parse_inline_tag(item: dict[str, Any], *, source_path: str = "") -> TagDef:
    tid = str(item.get("tag_id") or "").strip()
    if not tid:
        raise ValueError("tag requires tag_id")
    if "status" in item:
        status = str(item.get("status") or "enabled").strip().lower()
        if status in {"enabled", "active", "on"}:
            enabled, archived = True, False
        elif status in {"disabled", "off", "inactive"}:
            enabled, archived = False, False
        elif status in {"archived", "archive"}:
            enabled, archived = False, True
        else:
            raise ValueError(f"tag {tid}: invalid status {status!r}")
    else:
        enabled = bool(item.get("enabled", True))
        archived = bool(item.get("archived", False))
    examples = item.get("examples") or {}
    if examples is not None and not isinstance(examples, dict):
        raise ValueError(f"tag {tid}: examples must be mapping")
    deps = item.get("dependencies") or {}
    if deps is not None and not isinstance(deps, dict):
        raise ValueError(f"tag {tid}: dependencies must be mapping")
    return TagDef(
        tag_id=tid,
        name=str(item.get("name") or tid),
        description=str(item.get("description") or ""),
        version=str(item.get("version") or "1"),
        enabled=enabled,
        archived=archived,
        business_definition=str(item.get("business_definition") or item.get("definition") or ""),
        examples=dict(examples or {}),
        definition_path=source_path,
        requires_acoustic=(
            None
            if (deps.get("requires_acoustic") if isinstance(deps, dict) else None) is None
            and item.get("requires_acoustic") is None
            else bool(
                deps.get("requires_acoustic")
                if isinstance(deps, dict) and deps.get("requires_acoustic") is not None
                else item.get("requires_acoustic")
            )
        ),
        dependency_fragments=[str(x) for x in ((deps or {}).get("fragments") or [])],
        dependency_lexicons=[str(x) for x in ((deps or {}).get("lexicons") or [])],
    )


def _parse_inline_rule(item: dict[str, Any], *, tags: dict[str, TagDef]) -> RuleDef:
    rid = str(item.get("rule_id") or "").strip()
    if not rid:
        raise ValueError("rule requires rule_id")
    tag_id = str(item.get("tag_id") or "").strip()
    if not tag_id:
        raise ValueError(f"rule {rid} requires tag_id")
    if tag_id not in tags:
        raise ValueError(f"rule {rid} references unknown tag {tag_id}")
    cond = item.get("condition")
    if not isinstance(cond, dict):
        raise ValueError(f"rule {rid} requires condition mapping")
    binding = item.get("text_binding")
    if binding is not None and not isinstance(binding, dict):
        raise ValueError(f"rule {rid} text_binding must be mapping")
    return RuleDef(
        rule_id=rid,
        version=str(item.get("version") or "1"),
        tag_id=tag_id,
        condition=cond,
        enabled=bool(item.get("enabled", True)),
        description=str(item.get("description") or ""),
        text_binding=binding,
    )


def load_tagging_config(path: str | Path) -> TaggingConfig:
    from audio_engine.core.tagging.library import (
        TagLibraryError,
        load_tag_library,
        select_from_library,
    )
    from audio_engine.core.tagging.paths import find_repo_root, resolve_tagging_path

    p = Path(path).resolve()
    raw = _load_yaml(p)
    if not isinstance(raw, dict):
        raise ValueError(f"tagging config must be mapping: {p}")

    repo_root = find_repo_root(p.parent)
    policy = str(raw.get("policy_version") or POLICY_VERSION)

    # 044: refuse implicit selection_config loading — migrate explicitly.
    legacy_selection = str(
        raw.get("selection_config") or raw.get("selection_config_path") or ""
    ).strip()
    evidence_path_raw = str(
        raw.get("evidence_config") or raw.get("evidence_config_path") or ""
    ).strip()
    evidence_inline = raw.get("evidence") if isinstance(raw.get("evidence"), dict) else None
    if legacy_selection:
        raise ValueError(
            f"{p}: legacy selection_config={legacy_selection!r} is no longer loaded at "
            "runtime. Migrate with `audio-data tagging migrate-config` to produce an "
            "independent evidence_config, then remove selection_config."
        )
    if evidence_inline is not None:
        # Write temp-less load via in-memory dump through load_tag_evidence_config path.
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as tmp:
            import yaml as _yaml

            _yaml.safe_dump(evidence_inline, tmp, allow_unicode=True)
            tmp_path = Path(tmp.name)
        try:
            evidence = load_tag_evidence_config(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
        evidence_path = f"inline:{p.as_posix()}"
    elif evidence_path_raw:
        evidence_resolved = resolve_tagging_path(
            evidence_path_raw, base_dir=p.parent, repo_root=repo_root
        )
        evidence = load_tag_evidence_config(evidence_resolved)
        evidence_path = str(evidence_resolved.as_posix())
    else:
        raise ValueError(
            f"{p}: evidence_config (path) or evidence (inline mapping) is required"
        )
    families = set(evidence.ordered_families())

    evidence_mode = str(raw.get("evidence_mode") or "existing").strip().lower()
    if evidence_mode not in {"existing", "compute_acoustic"}:
        raise ValueError(f"{p}: evidence_mode must be existing|compute_acoustic")

    library_raw = raw.get("library")
    select_raw = raw.get("select_tags")
    library_mode = library_raw is not None or select_raw is not None
    library_path = ""
    select_tags_norm: list[dict[str, Any]] = []

    tags: dict[str, TagDef] = {}
    rules: list[RuleDef] = []

    if library_mode:
        if library_raw is None:
            raise TagLibraryError(f"{p}: select_tags requires library.path")
        if isinstance(library_raw, str):
            lib_dir = library_raw
        elif isinstance(library_raw, dict):
            lib_dir = str(library_raw.get("path") or "").strip()
        else:
            raise TagLibraryError(f"{p}: library must be path string or mapping with path")
        if not lib_dir:
            raise TagLibraryError(f"{p}: library.path is required")
        library_path = str(
            resolve_tagging_path(lib_dir, base_dir=p.parent, repo_root=repo_root).as_posix()
        )
        library = load_tag_library(library_path, base_dir=p.parent)
        if select_raw is None:
            raise TagLibraryError(
                f"{p}: library mode requires select_tags "
                "(refusing to implicitly enable the whole library)"
            )
        if not isinstance(select_raw, list):
            raise TagLibraryError(f"{p}: select_tags must be a list")
        for item in select_raw:
            if isinstance(item, str):
                select_tags_norm.append({"tag_id": item})
            elif isinstance(item, dict):
                select_tags_norm.append(dict(item))
            else:
                raise TagLibraryError(f"{p}: select_tags entries must be string or mapping")
        tags, rules = select_from_library(library, select_tags_norm, source_label=str(p.as_posix()))

    # Inline tags/rules (compatibility). Conflict with library IDs is rejected.
    inline_tags_raw = raw.get("tags") or []
    inline_rules_raw = raw.get("rules") or []
    if library_mode and (inline_tags_raw or inline_rules_raw):
        inline_ids = set()
        for item in inline_tags_raw:
            if isinstance(item, dict) and item.get("tag_id"):
                inline_ids.add(str(item["tag_id"]))
        for item in inline_rules_raw:
            if isinstance(item, dict) and item.get("tag_id"):
                inline_ids.add(str(item["tag_id"]))
        overlap = inline_ids & set(tags)
        if overlap:
            raise TagLibraryError(
                f"{p}: inline tags/rules conflict with library-selected IDs: {sorted(overlap)}"
            )

    for item in inline_tags_raw:
        if not isinstance(item, dict):
            raise ValueError("tag entry must be mapping")
        tag = _parse_inline_tag(item, source_path=str(p.as_posix()))
        if tag.tag_id in tags:
            raise TagLibraryError(f"{p}: duplicate tag_id {tag.tag_id} (inline vs library)")
        tags[tag.tag_id] = tag

    fragments = _load_fragments(raw.get("fragments"), base_dir=p.parent)
    _detect_fragment_cycles(fragments)

    lexicons: dict[str, LexiconSpec] = {}
    lex_raw = raw.get("lexicons") or {}
    if not isinstance(lex_raw, dict):
        raise ValueError("lexicons must be mapping")
    for lid, spec in lex_raw.items():
        if isinstance(spec, str):
            lpath = resolve_tagging_path(spec, base_dir=p.parent, repo_root=repo_root)
            lexicons[str(lid)] = load_lexicon(lpath, lexicon_id=str(lid))
        elif isinstance(spec, dict):
            lpath_raw = spec.get("path")
            if not lpath_raw:
                raise ValueError(f"lexicon {lid} requires path")
            lpath = resolve_tagging_path(str(lpath_raw), base_dir=p.parent, repo_root=repo_root)
            lexicons[str(lid)] = load_lexicon(lpath, lexicon_id=str(lid))
        else:
            raise ValueError(f"invalid lexicon spec for {lid}")

    seen_rule_ids: set[str] = {r.rule_id for r in rules}
    for item in inline_rules_raw:
        if not isinstance(item, dict):
            raise ValueError("rule entry must be mapping")
        rule = _parse_inline_rule(item, tags=tags)
        if rule.rule_id in seen_rule_ids:
            raise TagLibraryError(f"{p}: duplicate rule_id {rule.rule_id}")
        seen_rule_ids.add(rule.rule_id)
        rules.append(rule)

    if not tags:
        raise ValueError(f"{p}: no tags selected or defined")
    if not rules:
        raise ValueError(f"{p}: no rules selected or defined")

    for rule in rules:
        validate_condition_tree(
            rule.condition,
            fragments=fragments,
            lexicons=lexicons,
            families=families,
            path=f"rule:{rule.rule_id}",
        )

    for fid, frag in fragments.items():
        validate_condition_tree(
            frag.condition,
            fragments=fragments,
            lexicons=lexicons,
            families=families,
            path=f"fragment:{fid}",
        )

    acoustic = dict(raw.get("acoustic") or {})
    # Resolve acoustic nested paths when present (voicemail etc. stay as configured strings
    # but fingerprint uses resolved content when file exists).
    acoustic_fp = dict(acoustic)
    for key in ("voicemail_patterns_path", "silero_vad_config", "audio_energy_config", "dnsmos_config"):
        if key in acoustic and acoustic[key]:
            try:
                resolved = resolve_tagging_path(
                    str(acoustic[key]), base_dir=p.parent, repo_root=repo_root
                )
                acoustic[key] = str(resolved.as_posix())
                acoustic_fp[key] = {
                    "path": str(resolved.as_posix()),
                    "content_hash": _sha256_text(resolved.read_text(encoding="utf-8")),
                }
            except FileNotFoundError:
                acoustic_fp[key] = {"path": str(acoustic[key]), "missing": True}

    work_dir = str(
        raw.get("work_dir_template")
        or "runs/tagging/{batch_id}/{policy_version}"
    )

    fingerprint_payload = {
        "policy_version": policy,
        "evidence_config_path": evidence_path,
        "evidence": evidence.as_dict(),
        "evidence_mode": evidence_mode,
        "library_mode": library_mode,
        "library_path": library_path,
        "select_tags": select_tags_norm,
        "tags": {
            tid: {
                "version": t.version,
                "enabled": t.enabled,
                "archived": t.archived,
                "name": t.name,
                "description": t.description,
                "business_definition": t.business_definition,
                "examples": t.examples,
                "definition_path": t.definition_path,
            }
            for tid, t in sorted(tags.items())
        },
        "rules": [
            {
                "rule_id": r.rule_id,
                "version": r.version,
                "tag_id": r.tag_id,
                "enabled": r.enabled,
                "description": r.description,
                "text_binding": r.text_binding,
                "condition": r.condition,
            }
            for r in rules
        ],
        "fragments": {
            k: {"version": v.version, "condition": v.condition} for k, v in sorted(fragments.items())
        },
        "lexicons": {
            k: {"version": v.version, "content_hash": v.content_hash, "entries": [
                {"entry_id": e.entry_id, "pattern": e.pattern, "entry_type": e.entry_type, "match_mode": e.match_mode}
                for e in v.entries
            ]}
            for k, v in sorted(lexicons.items())
        },
        "acoustic": acoustic_fp,
    }
    cfg = TaggingConfig(
        policy_version=policy,
        evidence=evidence,
        evidence_config_path=evidence_path,
        evidence_mode=evidence_mode,
        tags=tags,
        rules=rules,
        fragments=fragments,
        lexicons=lexicons,
        work_dir_template=work_dir,
        acoustic=acoustic,
        content_fingerprint=_sha256_obj(fingerprint_payload),
        source_path=str(p.as_posix()),
        library_mode=library_mode,
        library_path=library_path,
        select_tags=select_tags_norm,
    )
    return cfg
