"""Generate / check human-readable tag library docs (042). No ASR / acoustic calls."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from audio_engine.core.tagging.library import (
    TagLibrary,
    expand_condition_fragments,
    load_tag_library,
)
from audio_engine.core.tagging.paths import find_repo_root, resolve_tagging_path


DEFAULT_DOC_REL = Path("docs/02-规范规则/tag规则库.md")
MARKER_BEGIN = "<!-- tag-library-doc:begin -->"
MARKER_END = "<!-- tag-library-doc:end -->"
SOURCE_HASH_PREFIX = "source_content_hash:"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _yaml_block(obj: Any) -> str:
    text = yaml.safe_dump(obj, allow_unicode=True, sort_keys=False, default_flow_style=False)
    return text.rstrip() + "\n"


def _examples_md(examples: dict[str, Any]) -> str:
    if not examples:
        return "_（未提供例子）_\n"
    lines: list[str] = []
    for key in ("hit", "miss", "unknown"):
        items = examples.get(key)
        title = {"hit": "命中", "miss": "不命中", "unknown": "证据不足 / 无法判断"}[key]
        lines.append(f"- **{title}** (`{key}`):")
        if not items:
            lines.append("  - （无）")
            continue
        if not isinstance(items, list):
            items = [items]
        for it in items:
            if isinstance(it, dict):
                desc = str(it.get("description") or it.get("note") or it)
                note = it.get("notes") or it.get("disclaimer")
                lines.append(f"  - {desc}")
                if note:
                    lines.append(f"    - 说明：{note}")
            else:
                lines.append(f"  - {it}")
    lines.append("")
    lines.append("> 例子为规则示意，不能宣称为未做过的真实模型验证。")
    lines.append("")
    return "\n".join(lines)


def compute_library_source_hash(library: TagLibrary, *, fragments_path: Path | None = None) -> str:
    payload: dict[str, Any] = {"entries": {}}
    for tid, entry in sorted(library.entries.items()):
        payload["entries"][tid] = {
            "path": entry.source_path,
            "text_hash": _sha256_text(Path(entry.source_path).read_text(encoding="utf-8")),
        }
    if fragments_path and fragments_path.is_file():
        payload["fragments"] = {
            "path": str(fragments_path.as_posix()),
            "text_hash": _sha256_text(fragments_path.read_text(encoding="utf-8")),
        }
    return _sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def render_tag_library_markdown(
    library: TagLibrary,
    *,
    fragments: dict[str, Any] | None = None,
    fragments_path: Path | None = None,
    source_hash: str = "",
) -> str:
    from audio_engine.core.tagging.config import FragmentDef

    frag_defs: dict[str, FragmentDef] = {}
    if fragments:
        for fid, f in fragments.items():
            if isinstance(f, FragmentDef):
                frag_defs[fid] = f
            elif isinstance(f, dict):
                frag_defs[fid] = FragmentDef(
                    fragment_id=str(f.get("fragment_id") or fid),
                    version=str(f.get("version") or "1"),
                    condition=dict(f.get("condition") or {}),
                    description=str(f.get("description") or ""),
                )

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines: list[str] = [
        "# Tag 规则库总览",
        "",
        "> 本文档为**派生产物**，由 `audio-data tags docs --generate` 从 `configs/tagging/library/` 生成。",
        "> 修改定义后请重新生成；可用 `audio-data tags docs --check` 检查是否过期。",
        "> 生成与查看过程**不调用 ASR / 声学推理**。",
        "",
        f"生成时间：{now}",
        f"定义库目录：`{library.root.as_posix()}`",
        f"{SOURCE_HASH_PREFIX} `{source_hash}`",
        "",
        MARKER_BEGIN,
        "",
        "## 目录",
        "",
        "| tag_id | 名称 | 版本 | 状态 | 规则数 | 定义文件 |",
        "|--------|------|------|------|--------|----------|",
    ]
    for entry in library.list_entries(include_disabled=True):
        rel = entry.source_path
        try:
            rel = str(Path(entry.source_path).resolve().relative_to(find_repo_root()).as_posix())
        except Exception:
            pass
        lines.append(
            f"| `{entry.tag.tag_id}` | {entry.tag.name} | `{entry.tag.version}` | "
            f"{entry.tag.status_label()} | {len(entry.rules)} | [`{Path(rel).name}`](../../{rel}) |"
        )
    lines.extend(["", "---", ""])

    if frag_defs:
        lines.append("## 公共条件片段")
        lines.append("")
        if fragments_path:
            lines.append(f"来源：`{fragments_path.as_posix()}`")
            lines.append("")
        for fid in sorted(frag_defs):
            f = frag_defs[fid]
            lines.append(f"### `{fid}` (v{f.version})")
            lines.append("")
            if f.description:
                lines.append(f.description)
                lines.append("")
            lines.append("```yaml")
            lines.append(_yaml_block(f.condition).rstrip())
            lines.append("```")
            lines.append("")

    for entry in library.list_entries(include_disabled=True):
        t = entry.tag
        lines.append(f"## `{t.tag_id}` — {t.name}")
        lines.append("")
        lines.append(f"- **版本**：`{t.version}`")
        lines.append(f"- **状态**：{t.status_label()}")
        lines.append(f"- **定义文件**：`{t.definition_path}`")
        lines.append("")
        if t.business_definition:
            lines.append("### 业务定义")
            lines.append("")
            lines.append(t.business_definition.strip())
            lines.append("")
        if t.description:
            lines.append("### 摘要")
            lines.append("")
            lines.append(t.description.strip())
            lines.append("")
        lines.append("### 依赖")
        lines.append("")
        lines.append(f"- 公共片段：{', '.join(f'`{x}`' for x in t.dependency_fragments) or '（无显式声明）'}")
        lines.append(f"- 词表：{', '.join(f'`{x}`' for x in t.dependency_lexicons) or '（无显式声明）'}")
        ac = t.requires_acoustic
        lines.append(f"- 需要声学证据：{'是' if ac else ('否' if ac is False else '未声明（以规则条件为准）')}")
        lines.append("")
        lines.append("### 例子")
        lines.append("")
        lines.append(_examples_md(t.examples))
        lines.append("### 规则")
        lines.append("")
        for rule in entry.rules:
            lines.append(f"#### `{rule.rule_id}` (v{rule.version})")
            lines.append("")
            lines.append(f"- enabled: `{rule.enabled}`")
            if rule.description:
                lines.append(f"- 说明：{rule.description}")
            if rule.text_binding:
                lines.append("- text_binding:")
                lines.append("```yaml")
                lines.append(_yaml_block(rule.text_binding).rstrip())
                lines.append("```")
            lines.append("- 完整条件（展开公共片段）：")
            lines.append("```yaml")
            expanded = expand_condition_fragments(rule.condition, frag_defs)
            lines.append(_yaml_block(expanded).rstrip())
            lines.append("```")
            lines.append("")
        lines.append("---")
        lines.append("")

    lines.append(MARKER_END)
    lines.append("")
    return "\n".join(lines)


def generate_tag_library_docs(
    *,
    library_dir: str | Path = "configs/tagging/library",
    fragments_path: str | Path = "configs/tagging/fragments_v1.yaml",
    output: str | Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    root = repo_root or find_repo_root()
    lib = load_tag_library(library_dir, base_dir=root)
    frag_path = resolve_tagging_path(fragments_path, base_dir=root, repo_root=root)
    from audio_engine.core.tagging.config import _load_fragments

    fragments = _load_fragments({"path": str(frag_path)}, base_dir=root)
    source_hash = compute_library_source_hash(lib, fragments_path=frag_path)
    text = render_tag_library_markdown(
        lib, fragments=fragments, fragments_path=frag_path, source_hash=source_hash
    )
    out = Path(output) if output else (root / DEFAULT_DOC_REL)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return {
        "ok": True,
        "output": str(out.as_posix()),
        "tag_count": len(lib.entries),
        "source_content_hash": source_hash,
        "asr_or_acoustic_invoked": False,
    }


def check_tag_library_docs(
    *,
    library_dir: str | Path = "configs/tagging/library",
    fragments_path: str | Path = "configs/tagging/fragments_v1.yaml",
    doc_path: str | Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    root = repo_root or find_repo_root()
    lib = load_tag_library(library_dir, base_dir=root)
    frag_path = resolve_tagging_path(fragments_path, base_dir=root, repo_root=root)
    expected = compute_library_source_hash(lib, fragments_path=frag_path)
    doc = Path(doc_path) if doc_path else (root / DEFAULT_DOC_REL)
    if not doc.is_absolute():
        doc = root / doc
    if not doc.is_file():
        return {
            "ok": False,
            "stale": True,
            "reason": "doc_missing",
            "doc_path": str(doc.as_posix()),
            "expected_source_content_hash": expected,
            "asr_or_acoustic_invoked": False,
        }
    text = doc.read_text(encoding="utf-8")
    found = None
    for line in text.splitlines():
        if SOURCE_HASH_PREFIX in line:
            # source_content_hash: `abc`
            part = line.split(SOURCE_HASH_PREFIX, 1)[1].strip().strip("`").strip()
            found = part
            break
    stale = found != expected
    return {
        "ok": not stale,
        "stale": stale,
        "doc_path": str(doc.as_posix()),
        "doc_source_content_hash": found,
        "expected_source_content_hash": expected,
        "asr_or_acoustic_invoked": False,
    }
