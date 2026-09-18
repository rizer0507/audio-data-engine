"""Load allowed warehouse categories from config (extensible; not hard-coded branches)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml

from audio_engine.core.selection_v3.types import BUSINESS_CATEGORIES_FIVE_CLASS


DEFAULT_CATEGORIES_CONFIG = Path("configs/warehouse/categories_five_class_v2_2.yaml")


def load_allowed_categories(path: str | Path | None = None) -> frozenset[str]:
    """Return allowed final/reviewed categories.

    Missing config falls back to current five-class business set so default
    batches keep working; adding categories only requires updating the YAML.
    """
    cfg_path = Path(path) if path else DEFAULT_CATEGORIES_CONFIG
    if not cfg_path.is_file():
        return frozenset(BUSINESS_CATEGORIES_FIVE_CLASS)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"categories config must be a mapping: {cfg_path}")
    items = raw.get("allowed_categories") or raw.get("categories") or []
    if not isinstance(items, list) or not items:
        raise ValueError(f"categories config missing allowed_categories list: {cfg_path}")
    return frozenset(str(x).strip() for x in items if str(x).strip())


def validate_category_value(
    value: str,
    allowed: Iterable[str] | None,
) -> str | None:
    """Return error message if value is set and not allowed; else None."""
    text = str(value or "").strip()
    if not text:
        return None
    if allowed is None:
        return None
    allowed_set = {str(x) for x in allowed}
    if text not in allowed_set:
        return f"category {text!r} not in allowed_categories"
    return None


def merge_category_params(params: dict[str, Any]) -> frozenset[str] | None:
    """Resolve allowed categories from operator/CLI params."""
    if params.get("allowed_categories"):
        return frozenset(str(x).strip() for x in params["allowed_categories"] if str(x).strip())
    path = params.get("categories_config") or params.get("categories_path")
    if path:
        return load_allowed_categories(path)
    # Default: five-class file if present, else built-in set.
    return load_allowed_categories(None)
