"""Developer CLI sharing the same plugin registry as Web.

  python -m audio_engine.plugins list
  python -m audio_engine.plugins show --kind pipeline --id <id>
  python -m audio_engine.plugins validate --request <request.json>
  python -m audio_engine.plugins plan --request <request.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from audio_engine.plugins.contracts import PluginContext, parse_task_request
from audio_engine.plugins.registry import get_registry


def _project_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    # Prefer repo root when running from source checkout.
    here = Path(__file__).resolve()
    candidate = here.parents[3]  # src/audio_engine/plugins -> repo
    if (candidate / "configs").is_dir() and (candidate / "pipelines").is_dir():
        return candidate
    return Path.cwd().resolve()


def cmd_list(args: argparse.Namespace) -> int:
    root = _project_root(args.project_root)
    registry = get_registry(root)
    catalog = registry.refresh(args.config)
    items = catalog.list(kind=args.kind, purpose=args.purpose, status=args.status)
    payload = {
        "generation": catalog.generation,
        "count": len(items),
        "diagnostics": catalog.diagnostics,
        "items": [
            {
                "kind": d.kind,
                "id": d.id,
                "version": d.version,
                "name": d.name,
                "adapter": d.adapter,
                "status": d.status,
                "runnable": d.runnable,
                "purpose": d.capabilities.purpose,
                "supports_cancel": d.capabilities.supports_cancel,
                "supports_resume": d.capabilities.supports_resume,
                "supports_tag_selection": d.capabilities.supports_tag_selection,
                "definition_path": d.definition_path,
            }
            for d in items
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    root = _project_root(args.project_root)
    registry = get_registry(root)
    registry.refresh(args.config)
    desc = registry.describe(args.kind, args.id, args.version)
    print(json.dumps(desc.as_dict(), ensure_ascii=False, indent=2))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    root = _project_root(args.project_root)
    registry = get_registry(root)
    registry.refresh(args.config)
    data = json.loads(Path(args.request).read_text(encoding="utf-8"))
    task = parse_task_request(data)
    context = PluginContext(project_root=root, readonly=True)
    report = registry.validate_task(task, context)
    print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    return 0 if report.ok else 2


def cmd_plan(args: argparse.Namespace) -> int:
    root = _project_root(args.project_root)
    registry = get_registry(root)
    registry.refresh(args.config)
    data = json.loads(Path(args.request).read_text(encoding="utf-8"))
    task = parse_task_request(data)
    context = PluginContext(project_root=root, readonly=True)
    try:
        plan = registry.plan_task(task, context)
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(plan.as_dict(), ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m audio_engine.plugins")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--config", default=None, help="path to configs/plugins.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list discovered plugins")
    p_list.add_argument("--kind", choices=["pipeline", "tag"], default=None)
    p_list.add_argument("--purpose", default=None)
    p_list.add_argument("--status", default=None)
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="show one plugin descriptor")
    p_show.add_argument("--kind", choices=["pipeline", "tag"], required=True)
    p_show.add_argument("--id", required=True)
    p_show.add_argument("--version", default=None)
    p_show.set_defaults(func=cmd_show)

    p_val = sub.add_parser("validate", help="validate a task request JSON")
    p_val.add_argument("--request", required=True)
    p_val.set_defaults(func=cmd_validate)

    p_plan = sub.add_parser("plan", help="build a read-only execution plan")
    p_plan.add_argument("--request", required=True)
    p_plan.set_defaults(func=cmd_plan)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
