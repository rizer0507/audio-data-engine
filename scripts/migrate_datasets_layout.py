#!/usr/bin/env python3
"""Migrate legacy flat ``datasets/`` to the staged layout (009).

Server-side helper for projects that still only have ``datasets/manifests/``.

What it does
------------
1. Rename existing ``datasets/`` → ``datasets-bak/`` (keeps all old content).
2. Create a blank staged ``datasets/`` tree::

     datasets/
       README.md
       stage1/cleaned|asr|derived/
       stage3/eval_sets|asr|derived|reports/
       manifests/          # restored from bak (compatibility layer)
       shards/             # optional empty

3. Copy ``datasets-bak/manifests/`` → ``datasets/manifests/`` so old
   parquet/jsonl stay readable via dual-resolve.
4. Check whether the **codebase** is fully synced to the post-009 layout
   (source_naming staged roots, key pipelines, etc.).

Examples
--------
::

  # Check only (no rename)
  python scripts/migrate_datasets_layout.py --check-only

  # Dry-run then apply
  python scripts/migrate_datasets_layout.py --dry-run
  python scripts/migrate_datasets_layout.py

  # Custom project root / bak name
  python scripts/migrate_datasets_layout.py --root /data2/.../audio-data-engine
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT_DEFAULT = Path(__file__).resolve().parents[1]

# Blank staged dirs required by source_naming / 009.
STAGED_DIRS: tuple[str, ...] = (
    "stage1/cleaned",
    "stage1/asr",
    "stage1/derived",
    "stage3/eval_sets",
    "stage3/asr",
    "stage3/derived",
    "stage3/reports",
    "manifests",
    "shards",
)

README_STUB = """# datasets — 可复用业务产物目录

> 本目录由 ``scripts/migrate_datasets_layout.py`` 初始化为 staged 空白骨架。
> 旧产物在同级 ``datasets-bak/``；``manifests/`` 为兼容层（已从 bak 拷回）。

```text
datasets/
  stage1/
    cleaned/     cleaned_{BATCH}.parquet
    asr/         {alias}_asr_{BATCH}.parquet
    derived/     multi_asr_* / classified_* / summary_* / …
  stage3/
    eval_sets/   eval_{BATCH}.parquet
    asr/         {alias}_asr_eval_{BATCH}.parquet
    derived/     eval_aggregate_* / eval_metrics_*
    reports/     {eval_name}/evaluation.{json,xlsx}
  manifests/     兼容层：旧文件可读；新写入默认落 stage*
```

命名权威：``src/audio_engine/core/source_naming.py``。
"""


@dataclass
class CheckItem:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class CheckReport:
    items: list[CheckItem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(item.ok for item in self.items)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.items.append(CheckItem(name=name, ok=ok, detail=detail))


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def check_codebase_synced(root: Path) -> CheckReport:
    """Verify key markers that the updated (staged datasets) codebase is present."""
    report = CheckReport()

    naming = root / "src" / "audio_engine" / "core" / "source_naming.py"
    text = _read_text(naming)
    report.add(
        "source_naming.py exists",
        naming.is_file(),
        str(naming.relative_to(root)) if naming.is_file() else "missing",
    )
    for marker in (
        "STAGE1_CLEANED_DIR",
        "STAGE1_ASR_DIR",
        "STAGE3_EVAL_SETS_DIR",
        "STAGE3_REPORTS_DIR",
        "staged_manifest_path",
        "resolve_existing_manifest",
    ):
        report.add(
            f"source_naming has {marker}",
            marker in text,
            "ok" if marker in text else "marker not found — code likely not synced",
        )

    # Pipelines should write staged paths, not only flat manifests.
    pipeline_checks = {
        "pipelines/data_cleaning_source_A.yaml": "datasets/stage1/cleaned",
        "pipelines/qwen_asr_batch.yaml": "datasets/stage1/asr",
        "pipelines/multi_asr_aggregate.yaml": "datasets/stage1/derived",
        "pipelines/classify_dataset.yaml": "datasets/stage1/derived",
        "pipelines/eval_aggregate.yaml": "datasets/stage3",
        "pipelines/eval_metric_pipeline.yaml": "datasets/stage3",
    }
    for rel, needle in pipeline_checks.items():
        path = root / rel
        body = _read_text(path)
        present = path.is_file()
        staged = needle in body
        report.add(
            f"pipeline staged path: {rel}",
            present and staged,
            (
                "ok"
                if present and staged
                else ("file missing" if not present else f"missing substring {needle!r}")
            ),
        )

    # New write path helper must prefer stage over flat manifests for cleaned_*.
    if "STAGE1_CLEANED_DIR" in text and "manifest_dir_for_stem" in text:
        report.add("manifest_dir_for_stem present", True, "ok")
    else:
        report.add(
            "manifest_dir_for_stem present",
            False,
            "source_naming incomplete — pull latest code first",
        )

    return report


def check_datasets_layout(root: Path, *, datasets_name: str = "datasets") -> CheckReport:
    """Check blank/staged datasets skeleton under project root."""
    report = CheckReport()
    ds = root / datasets_name
    report.add(f"{datasets_name}/ exists", ds.is_dir(), str(ds))
    if not ds.is_dir():
        return report
    for rel in STAGED_DIRS:
        path = ds / rel
        report.add(f"{datasets_name}/{rel}/", path.is_dir(), str(path))
    readme = ds / "README.md"
    report.add(f"{datasets_name}/README.md", readme.is_file(), str(readme))
    return report


def print_report(title: str, report: CheckReport) -> None:
    print(f"\n=== {title} ===")
    for item in report.items:
        mark = "OK " if item.ok else "FAIL"
        extra = f"  ({item.detail})" if item.detail else ""
        print(f"  [{mark}] {item.name}{extra}")
    print(f"  → {'PASS' if report.ok else 'NOT READY'}")


def ensure_dir(path: Path, *, dry_run: bool) -> None:
    if dry_run:
        print(f"  [dry-run] mkdir {path}")
        return
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, content: str, *, dry_run: bool) -> None:
    if dry_run:
        print(f"  [dry-run] write {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def copy_tree(src: Path, dest: Path, *, dry_run: bool) -> int:
    """Copy files under src into dest. Returns number of files copied."""
    if not src.is_dir():
        print(f"  [skip] no source dir: {src}")
        return 0
    count = 0
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        target = dest / rel
        if dry_run:
            print(f"  [dry-run] copy {path} -> {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        count += 1
    return count


def migrate(
    root: Path,
    *,
    bak_name: str,
    dry_run: bool,
    force: bool,
) -> int:
    datasets = root / "datasets"
    bak = root / bak_name

    print(f"Project root: {root}")
    print(f"datasets   : {datasets}")
    print(f"backup as  : {bak}")

    code = check_codebase_synced(root)
    print_report("Code sync check (updated project markers)", code)
    if not code.ok and not force:
        print(
            "\n[ABORT] Codebase does not look fully synced to the staged-datasets "
            "revision. Sync code from USB / git first, then re-run.\n"
            "        Or pass --force to migrate datasets layout anyway."
        )
        return 2

    if bak.exists():
        print(f"\n[ABORT] Backup already exists: {bak}")
        print("        Remove or rename it first, or pass a different --bak-name.")
        return 1

    if not datasets.exists():
        print("\n[INFO] No existing datasets/; will create blank staged tree only.")
    else:
        # Detect already-staged: avoid double migration unless forced.
        already_staged = (datasets / "stage1" / "cleaned").is_dir() and (
            datasets / "stage3" / "eval_sets"
        ).is_dir()
        if already_staged and not force:
            print(
                "\n[ABORT] datasets/ already has stage1/stage3 layout. "
                "Nothing to migrate. Use --force to bak+recreate anyway."
            )
            layout = check_datasets_layout(root)
            print_report("Current datasets layout", layout)
            return 1

        print(f"\n[1/3] Rename {datasets.name} -> {bak.name}")
        if dry_run:
            print(f"  [dry-run] rename {datasets} -> {bak}")
        else:
            datasets.rename(bak)

    print("\n[2/3] Create blank staged datasets/")
    ensure_dir(datasets, dry_run=dry_run)
    for rel in STAGED_DIRS:
        ensure_dir(datasets / rel, dry_run=dry_run)
        gitkeep = datasets / rel / ".gitkeep"
        if not dry_run and not gitkeep.exists():
            gitkeep.write_text("", encoding="utf-8")
        elif dry_run:
            print(f"  [dry-run] touch {gitkeep}")
    write_text(datasets / "README.md", README_STUB, dry_run=dry_run)

    print("\n[3/3] Restore manifests/ from backup (compatibility layer)")
    bak_manifests = bak / "manifests"
    new_manifests = datasets / "manifests"
    ensure_dir(new_manifests, dry_run=dry_run)
    if bak_manifests.is_dir():
        n = copy_tree(bak_manifests, new_manifests, dry_run=dry_run)
        print(f"  copied {n} file(s) from {bak_manifests}")
    else:
        # Legacy only-flat tree: entire old datasets was manifests-like.
        # If bak has parquet directly under bak/ (unusual), leave them in bak only.
        print(f"  [warn] no {bak_manifests}; manifests/ left empty")
        print("         Old files remain under bak; resolve may need manual copy.")

    layout = check_datasets_layout(root)
    print_report("New datasets layout", layout)

    print("\nDone." if not dry_run else "\nDry-run done (no changes written).")
    print("Notes:")
    print(f"  - Full old tree: {bak}")
    print(f"  - Compatible manifests: {new_manifests}")
    print("  - New pipeline writes go to stage1/* and stage3/*")
    print("  - Optional: later move bak/manifests files into staged dirs by stem")
    return 0 if layout.ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Backup datasets/ -> datasets-bak/, create blank staged datasets/, "
            "restore manifests/, and check codebase sync for 009 layout."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT_DEFAULT,
        help=f"project root (default: {ROOT_DEFAULT})",
    )
    parser.add_argument(
        "--bak-name",
        default="datasets-bak",
        help="backup directory name under project root (default: datasets-bak)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print actions without renaming/writing",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="only run sync + layout checks; do not migrate",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="migrate even if code-sync check fails or stage dirs already exist",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not root.is_dir():
        print(f"[ERROR] root is not a directory: {root}", file=sys.stderr)
        return 1

    if args.bak_name in {".", "..", "datasets"} or "/" in args.bak_name or "\\" in args.bak_name:
        print(f"[ERROR] invalid --bak-name: {args.bak_name!r}", file=sys.stderr)
        return 1

    if args.check_only:
        code = check_codebase_synced(root)
        print_report("Code sync check", code)
        layout = check_datasets_layout(root)
        print_report("datasets layout", layout)
        bak = root / args.bak_name
        print(f"\nBackup dir present: {bak.is_dir()} ({bak})")
        if (root / "datasets" / "manifests").is_dir():
            n = sum(1 for p in (root / "datasets" / "manifests").rglob("*") if p.is_file())
            print(f"datasets/manifests files: {n}")
        return 0 if code.ok and layout.ok else 2

    return migrate(
        root,
        bak_name=args.bak_name,
        dry_run=args.dry_run,
        force=args.force,
    )


if __name__ == "__main__":
    raise SystemExit(main())
