"""Strict, sandboxed Jinja rendering for semantic prompts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jinja2 import (
    BaseLoader,
    Environment,
    StrictUndefined,
    TemplateError,
    TemplateNotFound,
    select_autoescape,
)


class _RootBoundLoader(BaseLoader):
    """Only load templates from a frozen root directory."""

    def __init__(self, root: Path, frozen: dict[str, str]):
        self.root = root.resolve()
        self.frozen = frozen

    def get_source(self, environment: Environment, template: str):
        rel = template.replace("\\", "/")
        if rel in self.frozen:
            source = self.frozen[rel]
            path = str(self.root / rel)
            return source, path, lambda: True
        target = (self.root / rel).resolve()
        try:
            target.relative_to(self.root)
        except ValueError as exc:
            raise TemplateNotFound(template) from exc
        if not target.is_file():
            raise TemplateNotFound(template)
        source = target.read_text(encoding="utf-8")
        return source, str(target), lambda: True


def _tojson_filter(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def build_environment(template_root: Path, frozen_files: dict[str, str]) -> Environment:
    env = Environment(
        loader=_RootBoundLoader(template_root, frozen_files),
        undefined=StrictUndefined,
        # from_string defaults to escaping unless default_for_string=False.
        autoescape=select_autoescape(default_for_string=False, default=False),
        auto_reload=False,
        enable_async=False,
    )
    env.filters["tojson"] = _tojson_filter
    # Block attribute access that could escape the sandbox via common attrs.
    env.globals.clear()
    return env


class TemplateRenderer:
    def __init__(
        self,
        *,
        template_root: Path,
        user_template: str,
        system_template: str | None,
        included: dict[str, str],
    ):
        frozen = dict(included)
        frozen["__user__.jinja"] = user_template
        if system_template is not None:
            frozen["__system__.jinja"] = system_template
        self.env = build_environment(template_root, frozen)
        self._user = self.env.from_string(user_template)
        self._system = (
            self.env.from_string(system_template) if system_template is not None else None
        )

    def render(self, context: dict[str, Any]) -> tuple[str | None, str]:
        try:
            user = self._user.render(context)
            system = self._system.render(context) if self._system is not None else None
        except TemplateError as exc:
            raise RuntimeError(f"template_error: {exc}") from exc
        return system, user
