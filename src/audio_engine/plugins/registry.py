"""Plugin adapter registry, definition discovery, and atomic catalog refresh."""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from audio_engine.core.tagging.library import TagLibraryError, load_tag_library
from audio_engine.plugins.adapters import (
    TagBatchCoordinator,
    create_production_pipeline,
    create_tag_definition,
    create_yaml_pipeline,
)
from audio_engine.plugins.contracts import (
    API_VERSION,
    ExecutionPlan,
    Plugin,
    PluginContext,
    PluginDescriptor,
    PluginKind,
    PluginRef,
    PluginRequest,
    PluginResult,
    PluginStatus,
    TaskRequest,
    ValidationIssue,
    ValidationReport,
    parse_task_request,
    task_to_plugin_request,
    validate_plugin_id,
)
from audio_engine.plugins.definition_view import (
    apply_description_timestamp,
    bump_version_string,
    enrich_descriptor_dict,
    find_select_tag_refs,
    validate_definition_draft,
)
from audio_engine.plugins.schema_util import assert_schema_or_raise
from audio_engine.plugins.snapshot import sha256_file, sha256_obj

AdapterFactory = Callable[..., Plugin]


class PluginStatusError(Exception):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


class PluginDefinitionError(Exception):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


def _atomic_write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)
        Path(tmp_name).replace(path)
    finally:
        if Path(tmp_name).exists():
            Path(tmp_name).unlink(missing_ok=True)


@dataclass
class CatalogEntry:
    descriptor: PluginDescriptor
    plugin: Plugin


@dataclass
class PluginCatalog:
    generation: int
    api_version: str
    entries: dict[str, CatalogEntry] = field(default_factory=dict)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    plugins_config_path: str = ""
    plugins_config_digest: str = ""

    def list(
        self,
        *,
        kind: PluginKind | None = None,
        purpose: str | None = None,
        status: str | None = None,
        include_non_runnable: bool = True,
    ) -> list[PluginDescriptor]:
        items = [e.descriptor for e in self.entries.values()]
        if kind:
            items = [d for d in items if d.kind == kind]
        if purpose:
            items = [d for d in items if d.capabilities.purpose == purpose]
        if status:
            items = [d for d in items if d.status == status]
        if not include_non_runnable:
            items = [d for d in items if d.runnable]
        return sorted(items, key=lambda d: (d.kind, d.id, d.version))

    def get(self, kind: PluginKind, plugin_id: str, version: str | None = None) -> CatalogEntry:
        if version:
            key = f"{kind}:{plugin_id}@{version}"
            if key not in self.entries:
                raise KeyError(f"plugin not found: {key}")
            return self.entries[key]
        matches = [
            e
            for e in self.entries.values()
            if e.descriptor.kind == kind and e.descriptor.id == plugin_id
        ]
        if not matches:
            raise KeyError(f"plugin not found: {kind}:{plugin_id}")
        if len(matches) > 1:
            # Prefer enabled current; still unique id within generation.
            enabled = [m for m in matches if m.descriptor.status == "enabled"]
            matches = enabled or matches
        return matches[0]

    def as_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "api_version": self.api_version,
            "plugins_config_path": self.plugins_config_path,
            "plugins_config_digest": self.plugins_config_digest,
            "count": len(self.entries),
            "items": [e.as_dict() for e in self.list()],
            "diagnostics": list(self.diagnostics),
        }


class PluginRegistry:
    """Process-wide registry with atomic catalog generations."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = Path(project_root).resolve() if project_root else Path.cwd().resolve()
        self._adapters: dict[str, AdapterFactory] = {}
        self._lock = threading.RLock()
        self._catalog: PluginCatalog | None = None
        self._generation = 0
        self._builtins_registered = False
        self._tag_runtime: Path | None = None
        self._tag_library: Path | None = None

    def register_adapter(
        self,
        adapter_id: str,
        factory: AdapterFactory,
        *,
        replace: bool = False,
    ) -> None:
        aid = str(adapter_id or "").strip()
        if not aid:
            raise ValueError("adapter_id must be non-empty")
        with self._lock:
            if aid in self._adapters and not replace:
                # Idempotent for identical factory; reject conflicting rebind.
                if self._adapters[aid] is factory:
                    return
                raise ValueError(f"adapter already registered: {aid}")
            self._adapters[aid] = factory

    def ensure_builtins(self) -> None:
        with self._lock:
            if self._builtins_registered:
                return
            self.register_adapter("yaml_pipeline", create_yaml_pipeline)
            self.register_adapter("production_pipeline", create_production_pipeline)
            self.register_adapter("tag_definition", create_tag_definition)
            self._builtins_registered = True

    @property
    def catalog(self) -> PluginCatalog:
        with self._lock:
            if self._catalog is None:
                raise RuntimeError("plugin catalog not loaded; call refresh() first")
            return self._catalog

    def refresh(self, plugins_config: Path | str | None = None) -> PluginCatalog:
        self.ensure_builtins()
        cfg_path = Path(plugins_config) if plugins_config else self.project_root / "configs" / "plugins.yaml"
        if not cfg_path.is_absolute():
            cfg_path = (self.project_root / cfg_path).resolve()
        else:
            cfg_path = cfg_path.resolve()

        with self._lock:
            new_catalog, fatal = self._build_catalog(cfg_path)
            if fatal:
                # Keep previous generation when refresh cannot form a consistent catalog.
                raise RuntimeError(
                    "plugin catalog refresh failed: "
                    + "; ".join(f"{i['code']}: {i['message']}" for i in fatal)
                )
            self._generation += 1
            new_catalog.generation = self._generation
            for entry in new_catalog.entries.values():
                entry.descriptor.catalog_generation = self._generation
                # Re-bind adapters that store generation.
                if hasattr(entry.plugin, "catalog_generation"):
                    entry.plugin.catalog_generation = self._generation  # type: ignore[attr-defined]
            self._catalog = new_catalog
            return new_catalog

    def _build_catalog(self, cfg_path: Path) -> tuple[PluginCatalog, list[dict[str, Any]]]:
        diagnostics: list[dict[str, Any]] = []
        fatal: list[dict[str, Any]] = []
        if not cfg_path.is_file():
            fatal.append(
                {
                    "code": "PLUGINS_CONFIG_MISSING",
                    "message": f"plugins config not found: {cfg_path}",
                }
            )
            return PluginCatalog(generation=0, api_version=API_VERSION), fatal

        try:
            raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            fatal.append({"code": "PLUGINS_CONFIG_PARSE", "message": str(exc)})
            return PluginCatalog(generation=0, api_version=API_VERSION), fatal

        if not isinstance(raw, dict):
            fatal.append(
                {"code": "PLUGINS_CONFIG_INVALID", "message": "root must be a mapping"}
            )
            return PluginCatalog(generation=0, api_version=API_VERSION), fatal

        api_version = str(raw.get("api_version") or API_VERSION)
        if api_version != API_VERSION:
            fatal.append(
                {
                    "code": "API_VERSION_UNSUPPORTED",
                    "message": f"plugins.yaml api_version {api_version!r} != {API_VERSION!r}",
                }
            )
            return PluginCatalog(generation=0, api_version=API_VERSION), fatal

        # Extension modules (trusted in-repo only). Bind adapters to *this*
        # registry instance via register_extension_adapter during import.
        global _REFRESHING_REGISTRY
        with _REFRESHING_LOCK:
            _REFRESHING_REGISTRY = self
        try:
            for mod in raw.get("extension_modules") or []:
                try:
                    # Re-import so register side-effects run each refresh.
                    name = str(mod)
                    if name in sys.modules:
                        importlib.reload(sys.modules[name])
                    else:
                        importlib.import_module(name)
                except Exception as exc:  # noqa: BLE001
                    diagnostics.append(
                        {
                            "code": "EXTENSION_MODULE_LOAD_FAILED",
                            "message": str(exc),
                            "module": str(mod),
                        }
                    )
        finally:
            with _REFRESHING_LOCK:
                _REFRESHING_REGISTRY = None

        entries: dict[str, CatalogEntry] = {}
        seen_ids: dict[tuple[str, str], str] = {}

        for i, item in enumerate(raw.get("pipelines") or []):
            if not isinstance(item, dict):
                diagnostics.append(
                    {
                        "code": "PIPELINE_REGISTRATION_INVALID",
                        "message": f"pipelines[{i}] must be a mapping",
                    }
                )
                continue
            try:
                pid = validate_plugin_id(str(item.get("id") or ""))
                version = str(item.get("version") or "1")
                adapter_id = str(item.get("adapter") or "").strip()
                if not adapter_id:
                    raise ValueError("adapter is required")
                if adapter_id not in self._adapters:
                    raise ValueError(f"unknown adapter: {adapter_id}")
                schema = item.get("parameters_schema") or {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {},
                }
                schema_issues = assert_schema_or_raise(
                    schema, definition_path=f"configs/plugins.yaml#pipelines[{i}]"
                )
                for issue in schema_issues:
                    diagnostics.append(issue.as_dict())
                if schema_issues:
                    continue
                sk = ("pipeline", pid)
                if sk in seen_ids:
                    fatal.append(
                        {
                            "code": "DUPLICATE_PLUGIN_ID",
                            "message": f"duplicate pipeline id {pid}",
                            "definition_path": "configs/plugins.yaml",
                        }
                    )
                    continue
                seen_ids[sk] = version
                factory = self._adapters[adapter_id]
                plugin = factory(
                    item,
                    project_root=self.project_root,
                    catalog_generation=0,
                )
                desc = plugin.describe()
                key = desc.ref().key()
                entries[key] = CatalogEntry(descriptor=desc, plugin=plugin)
            except Exception as exc:  # noqa: BLE001
                diagnostics.append(
                    {
                        "code": "PIPELINE_ADAPTER_ERROR",
                        "message": str(exc),
                        "index": i,
                        "id": item.get("id"),
                    }
                )

        # Tag libraries
        for i, lib in enumerate(raw.get("tag_libraries") or []):
            if not isinstance(lib, dict):
                diagnostics.append(
                    {
                        "code": "TAG_LIBRARY_INVALID",
                        "message": f"tag_libraries[{i}] must be a mapping",
                    }
                )
                continue
            lib_path = Path(str(lib.get("path") or ""))
            if not lib_path.is_absolute():
                lib_path = self.project_root / lib_path
            runtime = Path(
                str(lib.get("runtime_config") or "configs/tagging/zh_asr_free_tags_v2.yaml")
            )
            if not runtime.is_absolute():
                runtime = self.project_root / runtime
            self._tag_library = lib_path
            self._tag_runtime = runtime
            try:
                library = load_tag_library(lib_path, base_dir=self.project_root)
            except TagLibraryError as exc:
                diagnostics.append(
                    {
                        "code": "TAG_LIBRARY_LOAD_FAILED",
                        "message": str(exc),
                        "path": str(lib_path),
                    }
                )
                continue
            except Exception as exc:  # noqa: BLE001
                diagnostics.append(
                    {
                        "code": "TAG_LIBRARY_LOAD_FAILED",
                        "message": str(exc),
                        "path": str(lib_path),
                    }
                )
                continue

            for entry in library.list_entries(include_disabled=True):
                try:
                    sk = ("tag", entry.tag.tag_id)
                    if sk in seen_ids:
                        fatal.append(
                            {
                                "code": "DUPLICATE_PLUGIN_ID",
                                "message": f"duplicate tag id {entry.tag.tag_id}",
                                "definition_path": entry.source_path,
                            }
                        )
                        continue
                    seen_ids[sk] = entry.tag.version
                    plugin = create_tag_definition(
                        entry_raw=entry.raw,
                        library_root=lib_path,
                        runtime_config=runtime,
                        project_root=self.project_root,
                        catalog_generation=0,
                        source_path=entry.source_path,
                    )
                    desc = plugin.describe()
                    entries[desc.ref().key()] = CatalogEntry(descriptor=desc, plugin=plugin)
                except Exception as exc:  # noqa: BLE001
                    diagnostics.append(
                        {
                            "code": "TAG_ADAPTER_ERROR",
                            "message": str(exc),
                            "definition_path": entry.source_path,
                        }
                    )

        if fatal:
            return (
                PluginCatalog(generation=0, api_version=API_VERSION, diagnostics=diagnostics),
                fatal,
            )

        catalog = PluginCatalog(
            generation=0,
            api_version=api_version,
            entries=entries,
            diagnostics=diagnostics,
            plugins_config_path=str(cfg_path.relative_to(self.project_root).as_posix())
            if cfg_path.is_relative_to(self.project_root)
            else str(cfg_path),
            plugins_config_digest=sha256_file(cfg_path),
        )
        return catalog, []

    def get_plugin(self, kind: PluginKind, plugin_id: str, version: str | None = None) -> Plugin:
        return self.catalog.get(kind, plugin_id, version).plugin

    def describe(self, kind: PluginKind, plugin_id: str, version: str | None = None) -> PluginDescriptor:
        return self.catalog.get(kind, plugin_id, version).descriptor

    def describe_detail(
        self, kind: PluginKind, plugin_id: str, version: str | None = None
    ) -> dict[str, Any]:
        """Descriptor dict plus summary / editable / edit_schema for the detail UI."""

        entry = self.catalog.get(kind, plugin_id, version)
        return enrich_descriptor_dict(entry.descriptor.as_dict(), entry.plugin)

    def set_status(
        self,
        kind: PluginKind,
        plugin_id: str,
        status: PluginStatus,
        *,
        expected_revision: int | None = None,
        plugins_config: Path | str | None = None,
    ) -> dict[str, Any]:
        """Persist enabled|archived to the unique config source and refresh catalog.

        Does not modify rules, versions, or trigger inference. Idempotent when
        already at the requested status. Old ``disabled`` is never auto-enabled.
        """

        if status not in {"enabled", "archived"}:
            raise PluginStatusError(
                "PLUGIN_STATUS_INVALID",
                "status must be enabled|archived",
                {"status": status},
            )
        self.ensure_builtins()
        catalog = self.ensure_catalog_loaded(plugins_config)
        if expected_revision is not None and int(expected_revision) != catalog.generation:
            raise PluginStatusError(
                "REVISION_CONFLICT",
                "plugin catalog revision mismatch; reload and retry",
                {"current_revision": catalog.generation, "current_generation": catalog.generation},
            )
        try:
            entry = catalog.get(kind, plugin_id)
        except KeyError as exc:
            raise PluginStatusError("PLUGIN_NOT_FOUND", f"{kind}/{plugin_id}") from exc

        desc = entry.descriptor
        # Idempotent: already at target status.
        if desc.status == status:
            return {
                "descriptor": desc.as_dict(),
                "revision": catalog.generation,
                "generation": catalog.generation,
                "unchanged": True,
            }

        cfg_path = Path(plugins_config) if plugins_config else Path(catalog.plugins_config_path)
        if not cfg_path.is_absolute():
            cfg_path = (self.project_root / cfg_path).resolve()

        if kind == "pipeline":
            self._persist_pipeline_status(cfg_path, plugin_id, status)
        elif kind == "tag":
            def_path = desc.definition_path
            if not def_path:
                raise PluginStatusError(
                    "PLUGIN_DEFINITION_PATH_MISSING",
                    f"tag {plugin_id} has no definition_path to persist status",
                )
            tag_path = Path(def_path)
            if not tag_path.is_absolute():
                tag_path = (self.project_root / tag_path).resolve()
            self._persist_tag_status(tag_path, status)
        else:
            raise PluginStatusError("PLUGIN_KIND_UNSUPPORTED", f"unsupported kind: {kind}")

        new_catalog = self.refresh(cfg_path)
        new_desc = new_catalog.get(kind, plugin_id).descriptor
        return {
            "descriptor": new_desc.as_dict(),
            "revision": new_catalog.generation,
            "generation": new_catalog.generation,
            "unchanged": False,
        }

    def ensure_catalog_loaded(self, plugins_config: Path | str | None = None) -> PluginCatalog:
        with self._lock:
            if self._catalog is None:
                return self.refresh(plugins_config)
            return self._catalog

    def _runtime_config_paths(self, catalog: PluginCatalog) -> list[Path]:
        cfg_path = Path(catalog.plugins_config_path)
        if not cfg_path.is_absolute():
            cfg_path = (self.project_root / cfg_path).resolve()
        paths: list[Path] = []
        try:
            raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            return paths
        for item in raw.get("tag_libraries") or []:
            if isinstance(item, dict) and item.get("runtime_config"):
                paths.append(Path(str(item["runtime_config"])))
        return paths

    def validate_definition(
        self,
        kind: PluginKind,
        plugin_id: str,
        draft: dict[str, Any],
        *,
        expected_revision: int | None = None,
        expected_definition_digest: str | None = None,
        version: str | None = None,
        plugins_config: Path | str | None = None,
    ) -> dict[str, Any]:
        self.ensure_builtins()
        catalog = self.ensure_catalog_loaded(plugins_config)
        if expected_revision is not None and int(expected_revision) != catalog.generation:
            raise PluginDefinitionError(
                "REVISION_CONFLICT",
                "plugin catalog revision mismatch; reload and retry",
                {
                    "current_revision": catalog.generation,
                    "current_generation": catalog.generation,
                },
            )
        try:
            entry = catalog.get(kind, plugin_id, version)
        except KeyError as exc:
            raise PluginDefinitionError("PLUGIN_NOT_FOUND", f"{kind}/{plugin_id}") from exc
        if (
            expected_definition_digest is not None
            and expected_definition_digest != entry.descriptor.definition_digest
        ):
            raise PluginDefinitionError(
                "DEFINITION_DIGEST_CONFLICT",
                "definition_digest mismatch; reload and retry",
                {
                    "current_definition_digest": entry.descriptor.definition_digest,
                },
            )
        result = validate_definition_draft(entry.plugin, draft)
        affected: list[dict[str, Any]] = []
        if result.get("changes_execution_logic") and kind == "tag":
            old_ver = entry.descriptor.version
            new_ver = bump_version_string(old_ver)
            affected = find_select_tag_refs(
                self.project_root,
                self._runtime_config_paths(catalog),
                tag_id=plugin_id,
                version=old_ver,
            )
            for ref in affected:
                ref["new_version"] = new_ver
        result["affected_references"] = affected
        result["revision"] = catalog.generation
        result["definition_digest"] = entry.descriptor.definition_digest
        return result

    def save_definition(
        self,
        kind: PluginKind,
        plugin_id: str,
        draft: dict[str, Any],
        *,
        expected_revision: int | None = None,
        expected_definition_digest: str | None = None,
        version: str | None = None,
        plugins_config: Path | str | None = None,
    ) -> dict[str, Any]:
        """Validate then atomically persist allowed definition fields to original files."""

        self.ensure_builtins()
        with self._lock:
            catalog = self.ensure_catalog_loaded(plugins_config)
            if expected_revision is not None and int(expected_revision) != catalog.generation:
                raise PluginDefinitionError(
                    "REVISION_CONFLICT",
                    "plugin catalog revision mismatch; reload and retry",
                    {
                        "current_revision": catalog.generation,
                        "current_generation": catalog.generation,
                    },
                )
            try:
                entry = catalog.get(kind, plugin_id, version)
            except KeyError as exc:
                raise PluginDefinitionError("PLUGIN_NOT_FOUND", f"{kind}/{plugin_id}") from exc
            if (
                expected_definition_digest is not None
                and expected_definition_digest != entry.descriptor.definition_digest
            ):
                raise PluginDefinitionError(
                    "DEFINITION_DIGEST_CONFLICT",
                    "definition_digest mismatch; reload and retry",
                    {
                        "current_definition_digest": entry.descriptor.definition_digest,
                    },
                )

            result = validate_definition_draft(entry.plugin, draft)
            if not result.get("ok"):
                raise PluginDefinitionError(
                    "DEFINITION_VALIDATION_FAILED",
                    "definition draft failed validation",
                    {
                        "field_errors": result.get("field_errors") or [],
                        "warnings": result.get("warnings") or [],
                    },
                )

            changes_exec = bool(result.get("changes_execution_logic"))
            cfg_path = Path(plugins_config) if plugins_config else Path(catalog.plugins_config_path)
            if not cfg_path.is_absolute():
                cfg_path = (self.project_root / cfg_path).resolve()

            archived_status = entry.descriptor.status
            old_version = entry.descriptor.version
            new_version = old_version
            affected: list[dict[str, Any]] = []

            if kind == "tag":
                preview = result.get("preview_entry")
                if not isinstance(preview, dict):
                    raise PluginDefinitionError(
                        "DEFINITION_VALIDATION_FAILED", "missing preview_entry"
                    )
                # Preserve archive/enabled/disabled — never auto-enable on save.
                preview["status"] = archived_status
                preview.pop("enabled", None)
                preview.pop("archived", None)
                apply_description_timestamp(
                    preview=preview,
                    previous=dict(entry.plugin.entry_raw),  # type: ignore[attr-defined]
                    changed=bool(result.get("description_changed")),
                )

                if changes_exec:
                    new_version = bump_version_string(old_version)
                    preview["version"] = new_version
                    for rule in preview.get("rules") or []:
                        if isinstance(rule, dict):
                            rule["version"] = bump_version_string(str(rule.get("version") or "1"))
                    affected = find_select_tag_refs(
                        self.project_root,
                        self._runtime_config_paths(catalog),
                        tag_id=plugin_id,
                        version=old_version,
                    )
                    if not affected and old_version != new_version:
                        # Still ok — tag may not be selected in any runtime yet.
                        pass
                    # Update references together with definition.
                    for ref in affected:
                        self._persist_select_tag_version(
                            Path(ref["path"]),
                            tag_id=plugin_id,
                            old_version=old_version,
                            new_version=new_version,
                        )
                        ref["new_version"] = new_version

                def_path = entry.descriptor.definition_path
                if not def_path:
                    raise PluginDefinitionError(
                        "PLUGIN_DEFINITION_PATH_MISSING",
                        f"tag {plugin_id} has no definition_path",
                    )
                tag_path = Path(def_path)
                if not tag_path.is_absolute():
                    tag_path = (self.project_root / tag_path).resolve()
                try:
                    _atomic_write_yaml(tag_path, preview)
                except OSError as exc:
                    raise PluginDefinitionError(
                        "DEFINITION_PERSIST_FAILED",
                        f"cannot write tag definition: {exc}",
                        {"path": str(tag_path)},
                    ) from exc

            elif kind == "pipeline":
                preview_reg = result.get("preview_registration")
                if not isinstance(preview_reg, dict):
                    raise PluginDefinitionError(
                        "DEFINITION_VALIDATION_FAILED", "missing preview_registration"
                    )
                apply_description_timestamp(
                    preview=preview_reg,
                    previous=dict(entry.plugin.registration),  # type: ignore[attr-defined]
                    changed=bool(result.get("description_changed")),
                )
                if changes_exec:
                    new_version = bump_version_string(old_version)
                    preview_reg["version"] = new_version
                # Never change status via definition save.
                preview_reg["status"] = archived_status if archived_status in {"enabled", "archived", "disabled"} else entry.descriptor.status
                self._persist_pipeline_registration(cfg_path, plugin_id, preview_reg)
                # Description-only edit: do not rewrite pipeline step YAML.
            else:
                raise PluginDefinitionError("PLUGIN_KIND_UNSUPPORTED", f"unsupported kind: {kind}")

            new_catalog = self.refresh(cfg_path)
            # After version bump, look up by new version
            try:
                new_entry = new_catalog.get(kind, plugin_id, new_version)
            except KeyError:
                new_entry = new_catalog.get(kind, plugin_id)
            detail = enrich_descriptor_dict(new_entry.descriptor.as_dict(), new_entry.plugin)
            return {
                "descriptor": detail,
                "revision": new_catalog.generation,
                "generation": new_catalog.generation,
                "changes_execution_logic": changes_exec,
                "affected_references": affected,
                "previous_version": old_version,
                "version": new_entry.descriptor.version,
            }

    def _persist_select_tag_version(
        self,
        rel_or_abs: Path,
        *,
        tag_id: str,
        old_version: str,
        new_version: str,
    ) -> None:
        path = rel_or_abs if rel_or_abs.is_absolute() else (self.project_root / rel_or_abs)
        path = path.resolve()
        if not path.is_file():
            raise PluginDefinitionError(
                "REFERENCE_UPDATE_FAILED",
                f"runtime config missing for reference update: {path}",
            )
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # noqa: BLE001
            raise PluginDefinitionError("REFERENCE_UPDATE_FAILED", str(exc)) from exc
        select = raw.get("select_tags")
        if not isinstance(select, list):
            raise PluginDefinitionError(
                "REFERENCE_UPDATE_FAILED",
                f"select_tags missing in {path}",
            )
        found = False
        for item in select:
            if (
                isinstance(item, dict)
                and str(item.get("tag_id") or "") == tag_id
                and str(item.get("version") or "") == old_version
            ):
                item["version"] = new_version
                found = True
        if not found:
            raise PluginDefinitionError(
                "REFERENCE_INCONSISTENT",
                f"could not update select_tags for {tag_id}@{old_version} in {path}",
                {"path": str(path), "tag_id": tag_id, "old_version": old_version},
            )
        try:
            _atomic_write_yaml(path, raw)
        except OSError as exc:
            raise PluginDefinitionError(
                "REFERENCE_UPDATE_FAILED",
                f"cannot write runtime config: {exc}",
                {"path": str(path)},
            ) from exc

    def _persist_pipeline_registration(
        self, cfg_path: Path, plugin_id: str, registration: dict[str, Any]
    ) -> None:
        if not cfg_path.is_file():
            raise PluginDefinitionError("PLUGINS_CONFIG_MISSING", f"plugins config not found: {cfg_path}")
        try:
            raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # noqa: BLE001
            raise PluginDefinitionError("PLUGINS_CONFIG_PARSE", str(exc)) from exc
        if not isinstance(raw, dict):
            raise PluginDefinitionError("PLUGINS_CONFIG_INVALID", "root must be a mapping")
        pipelines = raw.get("pipelines") or []
        if not isinstance(pipelines, list):
            raise PluginDefinitionError("PLUGINS_CONFIG_INVALID", "pipelines must be a list")
        found = False
        allowed_keys = {
            "description",
            "description_updated_at",
            "version",
            "status",
        }
        for item in pipelines:
            if isinstance(item, dict) and str(item.get("id") or "") == plugin_id:
                for key in allowed_keys:
                    if key in registration:
                        item[key] = registration[key]
                found = True
                break
        if not found:
            raise PluginDefinitionError(
                "PLUGIN_NOT_FOUND", f"pipeline id not in plugins.yaml: {plugin_id}"
            )
        try:
            _atomic_write_yaml(cfg_path, raw)
        except OSError as exc:
            raise PluginDefinitionError(
                "DEFINITION_PERSIST_FAILED",
                f"cannot write plugins.yaml: {exc}",
                {"path": str(cfg_path)},
            ) from exc

    def _persist_pipeline_status(self, cfg_path: Path, plugin_id: str, status: PluginStatus) -> None:
        if not cfg_path.is_file():
            raise PluginStatusError("PLUGINS_CONFIG_MISSING", f"plugins config not found: {cfg_path}")
        try:
            raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # noqa: BLE001
            raise PluginStatusError("PLUGINS_CONFIG_PARSE", str(exc)) from exc
        if not isinstance(raw, dict):
            raise PluginStatusError("PLUGINS_CONFIG_INVALID", "root must be a mapping")
        pipelines = raw.get("pipelines") or []
        if not isinstance(pipelines, list):
            raise PluginStatusError("PLUGINS_CONFIG_INVALID", "pipelines must be a list")
        found = False
        for item in pipelines:
            if isinstance(item, dict) and str(item.get("id") or "") == plugin_id:
                item["status"] = status
                found = True
                break
        if not found:
            raise PluginStatusError("PLUGIN_NOT_FOUND", f"pipeline id not in plugins.yaml: {plugin_id}")
        try:
            _atomic_write_yaml(cfg_path, raw)
        except OSError as exc:
            raise PluginStatusError(
                "PLUGIN_STATUS_PERSIST_FAILED",
                f"cannot write plugins.yaml: {exc}",
                {"path": str(cfg_path)},
            ) from exc

    def _persist_tag_status(self, tag_path: Path, status: PluginStatus) -> None:
        if not tag_path.is_file():
            raise PluginStatusError("TAG_DEFINITION_MISSING", f"tag definition not found: {tag_path}")
        try:
            raw = yaml.safe_load(tag_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # noqa: BLE001
            raise PluginStatusError("TAG_DEFINITION_PARSE", str(exc)) from exc
        if not isinstance(raw, dict):
            raise PluginStatusError("TAG_DEFINITION_INVALID", "tag definition must be a mapping")
        raw["status"] = status
        # Drop legacy boolean flags so status is the single source of truth.
        raw.pop("enabled", None)
        raw.pop("archived", None)
        try:
            _atomic_write_yaml(tag_path, raw)
        except OSError as exc:
            raise PluginStatusError(
                "PLUGIN_STATUS_PERSIST_FAILED",
                f"cannot write tag definition: {exc}",
                {"path": str(tag_path)},
            ) from exc

    def validate_task(self, task: TaskRequest, context: PluginContext) -> ValidationReport:
        report = ValidationReport(ok=True)
        catalog = self.catalog
        if task.catalog_generation is not None and task.catalog_generation != catalog.generation:
            report.ok = False
            report.errors.append(
                ValidationIssue(
                    code="CATALOG_GENERATION_MISMATCH",
                    message=(
                        f"request generation {task.catalog_generation} != "
                        f"current {catalog.generation}"
                    ),
                    field_path="catalog_generation",
                )
            )
            return report

        if task.task_type == "pipeline":
            if task.root_plugin is None:
                report.ok = False
                report.errors.append(
                    ValidationIssue(
                        code="ROOT_PLUGIN_REQUIRED",
                        message="pipeline task requires root_plugin",
                        field_path="root_plugin",
                    )
                )
                return report
            try:
                entry = catalog.get(
                    task.root_plugin.kind, task.root_plugin.id, task.root_plugin.version
                )
            except KeyError:
                report.ok = False
                report.errors.append(
                    ValidationIssue(
                        code="PLUGIN_NOT_FOUND",
                        message=f"root plugin not found: {task.root_plugin.key()}",
                        field_path="root_plugin",
                        plugin_ref=task.root_plugin.as_dict(),
                    )
                )
                return report
            preq = task_to_plugin_request(task, task.root_plugin)
            report.extend(entry.plugin.validate(preq, context))
            for tag in task.tags:
                try:
                    tag_entry = catalog.get("tag", tag.id, tag.version)
                except KeyError:
                    report.ok = False
                    report.errors.append(
                        ValidationIssue(
                            code="TAG_NOT_FOUND",
                            message=f"tag not found: {tag.key()}",
                            field_path="tags",
                            plugin_ref=tag.as_dict(),
                        )
                    )
                    continue
                report.extend(
                    tag_entry.plugin.validate(
                        PluginRequest(plugin=tag, input=task.input), context
                    )
                )
            return report

        # tagging
        if task.root_plugin is not None:
            report.ok = False
            report.errors.append(
                ValidationIssue(
                    code="ROOT_PLUGIN_MUST_BE_NULL",
                    message="tagging task requires root_plugin=null",
                    field_path="root_plugin",
                )
            )
        if not task.tags:
            report.ok = False
            report.errors.append(
                ValidationIssue(
                    code="TAGS_REQUIRED",
                    message="tagging task requires non-empty tags",
                    field_path="tags",
                )
            )
        if task.input.kind != "dataset":
            report.ok = False
            report.errors.append(
                ValidationIssue(
                    code="INPUT_KIND_UNSUPPORTED",
                    message="tagging task requires dataset input",
                    field_path="input.kind",
                )
            )
        for tag in task.tags:
            try:
                tag_entry = catalog.get("tag", tag.id, tag.version)
            except KeyError:
                report.ok = False
                report.errors.append(
                    ValidationIssue(
                        code="TAG_NOT_FOUND",
                        message=f"tag not found: {tag.key()}",
                        field_path="tags",
                        plugin_ref=tag.as_dict(),
                    )
                )
                continue
            report.extend(
                tag_entry.plugin.validate(
                    PluginRequest(plugin=tag, input=task.input, parameters=task.parameters),
                    context,
                )
            )
        return report

    def plan_task(self, task: TaskRequest, context: PluginContext) -> ExecutionPlan:
        report = self.validate_task(task, context)
        if not report.ok:
            raise ValueError(
                "validate failed: " + "; ".join(e.message for e in report.errors)
            )
        context = PluginContext(
            project_root=context.project_root,
            job_id=context.job_id,
            attempt_id=context.attempt_id,
            work_dir=context.work_dir,
            snapshot_dir=context.snapshot_dir,
            model_connections=context.model_connections,
            secret_resolver=context.secret_resolver,
            cancel_token=context.cancel_token,
            on_progress=context.on_progress,
            on_log=context.on_log,
            readonly=True,
        )
        if task.task_type == "pipeline":
            assert task.root_plugin is not None
            entry = self.catalog.get(
                task.root_plugin.kind, task.root_plugin.id, task.root_plugin.version
            )
            preq = task_to_plugin_request(task, task.root_plugin)
            plan = entry.plugin.plan(preq, context)
            plan.task_request = task.as_dict()
            return plan

        # tagging batch
        if self._tag_runtime is None or self._tag_library is None:
            raise RuntimeError("tag library not configured in plugins.yaml")
        coordinator = TagBatchCoordinator(
            runtime_config=self._tag_runtime,
            library_root=self._tag_library,
            project_root=self.project_root,
            catalog_generation=self.catalog.generation,
        )
        # Use first tag as plugin shell for PluginRequest fields.
        shell = PluginRequest(
            plugin=task.tags[0],
            input=task.input,
            parameters=dict(task.parameters),
            model_bindings=list(task.model_bindings),
            tags=list(task.tags),
            execution=task.execution,
            catalog_generation=task.catalog_generation,
        )
        plan = coordinator.plan_tags(task.tags, shell, context)
        plan.task_request = task.as_dict()
        plan.sub_plans = [
            {"plugin": t.as_dict(), "role": "selected_tag"} for t in task.tags
        ]
        return plan

    def execute_plan(self, plan: ExecutionPlan, context: PluginContext) -> PluginResult:
        task_data = plan.task_request
        if task_data and task_data.get("task_type") == "tagging":
            if self._tag_runtime is None or self._tag_library is None:
                raise RuntimeError("tag library not configured")
            coordinator = TagBatchCoordinator(
                runtime_config=self._tag_runtime,
                library_root=self._tag_library,
                project_root=self.project_root,
                catalog_generation=self.catalog.generation,
            )
            return coordinator.execute(plan, context)

        entry = self.catalog.get(plan.plugin.kind, plan.plugin.id, plan.plugin.version)
        return entry.plugin.execute(plan, context)

    def definition_digest_map(self) -> dict[str, str]:
        return {
            e.descriptor.ref().key(): e.descriptor.definition_digest
            for e in self.catalog.entries.values()
        }

    def catalog_content_digest(self) -> str:
        return sha256_obj(self.definition_digest_map())


_DEFAULT_REGISTRY: PluginRegistry | None = None
_DEFAULT_LOCK = threading.Lock()
_REFRESHING_REGISTRY: PluginRegistry | None = None
_REFRESHING_LOCK = threading.Lock()


def register_extension_adapter(adapter_id: str, factory: AdapterFactory) -> None:
    """Register an adapter from a trusted extension_modules import.

    Binds to the registry currently refreshing (Web/Worker instance), falling
    back to the process default registry. Uses replace=True so reload on
    refresh does not fail on a new factory object identity.
    """

    with _REFRESHING_LOCK:
        target = _REFRESHING_REGISTRY
    if target is None:
        target = get_registry()
    target.register_adapter(adapter_id, factory, replace=True)


def get_registry(project_root: Path | str | None = None) -> PluginRegistry:
    global _DEFAULT_REGISTRY
    root = Path(project_root).resolve() if project_root else Path.cwd().resolve()
    with _DEFAULT_LOCK:
        if _DEFAULT_REGISTRY is None or _DEFAULT_REGISTRY.project_root != root:
            _DEFAULT_REGISTRY = PluginRegistry(project_root=root)
        return _DEFAULT_REGISTRY


def load_catalog(
    project_root: Path | str | None = None,
    plugins_config: Path | str | None = None,
) -> PluginCatalog:
    registry = get_registry(project_root)
    return registry.refresh(plugins_config)
