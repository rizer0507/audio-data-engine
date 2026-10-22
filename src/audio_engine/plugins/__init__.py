"""Unified plugin interface for pipelines and tags."""

from audio_engine.plugins.contracts import (
    API_VERSION,
    ExecutionPlan,
    PluginContext,
    PluginDescriptor,
    PluginRef,
    PluginRequest,
    PluginResult,
    TaskRequest,
    ValidationReport,
    parse_task_request,
)
from audio_engine.plugins.registry import (
    PluginCatalog,
    PluginRegistry,
    get_registry,
    load_catalog,
    register_extension_adapter,
)

__all__ = [
    "API_VERSION",
    "ExecutionPlan",
    "PluginCatalog",
    "PluginContext",
    "PluginDescriptor",
    "PluginRef",
    "PluginRegistry",
    "PluginRequest",
    "PluginResult",
    "TaskRequest",
    "ValidationReport",
    "get_registry",
    "load_catalog",
    "parse_task_request",
    "register_extension_adapter",
]
