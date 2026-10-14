"""Tagging config: tags, rules, fragments, lexicons (040)."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from audio_engine.core.selection_v3.config import SelectionV3Config
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
    enabled: bool = True
    archived: bool = False


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
    selection_config_path: str
    selection: SelectionV3Config
    evidence_mode: str  # existing | compute_acoustic
    tags: dict[str, TagDef]
    rules: list[RuleDef]
    fragments: dict[str, FragmentDef]
    lexicons: dict[str, LexiconSpec]
    work_dir_template: str = "runs/tagging/{batch_id}/{policy_version}"
    acoustic: dict[str, Any] = field(default_factory=dict)
    content_fingerprint: str = ""
    source_path: str = ""

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
            "selection_config_path": self.selection_config_path,
            "evidence_mode": self.evidence_mode,
            "content_fingerprint": self.content_fingerprint,
            "source_path": self.source_path,
            "tags": {
                tid: {
                    "tag_id": t.tag_id,
                    "name": t.name,
                    "description": t.description,
                    "enabled": t.enabled,
                    "archived": t.archived,
                }
                for tid, t in self.tags.items()
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
    out: dict[str, FragmentDef] = {}
    if raw is None:
        return out
    items: list[Any]
    if isinstance(raw, dict) and "path" in raw:
        frag_path = Path(str(raw["path"]))
        if not frag_path.is_absolute():
            frag_path = (base_dir / frag_path).resolve() if False else Path(frag_path)
        # Prefer repo-relative path as-is (cwd = repo root in this project).
        frag_path = Path(str(raw["path"]))
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


def load_tagging_config(path: str | Path) -> TaggingConfig:
    p = Path(path)
    raw = _load_yaml(p)
    if not isinstance(raw, dict):
        raise ValueError(f"tagging config must be mapping: {p}")

    policy = str(raw.get("policy_version") or POLICY_VERSION)
    selection_path = str(raw.get("selection_config") or raw.get("selection_config_path") or "").strip()
    if not selection_path:
        raise ValueError(f"{p}: selection_config is required")
    selection = SelectionV3Config.from_yaml(selection_path)
    families = set(selection.ordered_families())

    evidence_mode = str(raw.get("evidence_mode") or "existing").strip().lower()
    if evidence_mode not in {"existing", "compute_acoustic"}:
        raise ValueError(f"{p}: evidence_mode must be existing|compute_acoustic")

    tags: dict[str, TagDef] = {}
    for item in raw.get("tags") or []:
        if not isinstance(item, dict):
            raise ValueError("tag entry must be mapping")
        tid = str(item.get("tag_id") or "").strip()
        if not tid:
            raise ValueError("tag requires tag_id")
        tags[tid] = TagDef(
            tag_id=tid,
            name=str(item.get("name") or tid),
            description=str(item.get("description") or ""),
            enabled=bool(item.get("enabled", True)),
            archived=bool(item.get("archived", False)),
        )

    fragments = _load_fragments(raw.get("fragments"), base_dir=p.parent)
    _detect_fragment_cycles(fragments)

    lexicons: dict[str, LexiconSpec] = {}
    lex_raw = raw.get("lexicons") or {}
    if not isinstance(lex_raw, dict):
        raise ValueError("lexicons must be mapping")
    for lid, spec in lex_raw.items():
        if isinstance(spec, str):
            lexicons[str(lid)] = load_lexicon(spec, lexicon_id=str(lid))
        elif isinstance(spec, dict):
            lpath = spec.get("path")
            if not lpath:
                raise ValueError(f"lexicon {lid} requires path")
            lexicons[str(lid)] = load_lexicon(lpath, lexicon_id=str(lid))
        else:
            raise ValueError(f"invalid lexicon spec for {lid}")

    rules: list[RuleDef] = []
    for item in raw.get("rules") or []:
        if not isinstance(item, dict):
            raise ValueError("rule entry must be mapping")
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
        validate_condition_tree(
            cond, fragments=fragments, lexicons=lexicons, families=families, path=f"rule:{rid}"
        )
        binding = item.get("text_binding")
        if binding is not None and not isinstance(binding, dict):
            raise ValueError(f"rule {rid} text_binding must be mapping")
        rules.append(
            RuleDef(
                rule_id=rid,
                version=str(item.get("version") or "1"),
                tag_id=tag_id,
                condition=cond,
                enabled=bool(item.get("enabled", True)),
                description=str(item.get("description") or ""),
                text_binding=binding,
            )
        )

    # Validate fragment trees too.
    for fid, frag in fragments.items():
        validate_condition_tree(
            frag.condition,
            fragments=fragments,
            lexicons=lexicons,
            families=families,
            path=f"fragment:{fid}",
        )

    acoustic = dict(raw.get("acoustic") or {})
    work_dir = str(
        raw.get("work_dir_template")
        or "runs/tagging/{batch_id}/{policy_version}"
    )
    fingerprint_payload = {
        "policy_version": policy,
        "selection_config": selection_path,
        "evidence_mode": evidence_mode,
        "tags": raw.get("tags"),
        "rules": raw.get("rules"),
        "fragments": {k: {"version": v.version, "condition": v.condition} for k, v in fragments.items()},
        "lexicons": {k: v.content_hash for k, v in lexicons.items()},
        "acoustic": acoustic,
    }
    cfg = TaggingConfig(
        policy_version=policy,
        selection_config_path=selection_path,
        selection=selection,
        evidence_mode=evidence_mode,
        tags=tags,
        rules=rules,
        fragments=fragments,
        lexicons=lexicons,
        work_dir_template=work_dir,
        acoustic=acoustic,
        content_fingerprint=_sha256_obj(fingerprint_payload),
        source_path=str(p.as_posix()),
    )
    return cfg
