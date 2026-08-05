# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Mode config, mode defaults, and task type hints handler."""

from __future__ import annotations

from typing import Any, Dict, Optional, Type

from victor.core.verticals.extension_handlers.base import (
    BaseExtensionHandler,
    ExtensionLoaderContext,
)
from victor.core.verticals.extension_handlers.registry import ExtensionHandlerRegistry


@ExtensionHandlerRegistry.register
class ModeConfigHandler(BaseExtensionHandler):
    """Loads mode configuration provider for a vertical.

    Resolution order:
    1. Entry point: victor.mode_configs group
    2. Fallback: _resolve_class_or_factory_extension with suffix "mode_config"
    """

    extension_type = "mode_config"

    @classmethod
    def load(cls, ctx: Type[ExtensionLoaderContext]) -> Optional[Any]:
        provider = ctx._load_named_entry_point_extension(
            "mode_config_provider", "victor.mode_configs"
        )
        if provider is not None:
            return provider

        from victor.core.verticals.vertical_metadata import VerticalMetadata

        metadata = VerticalMetadata.from_class(ctx)
        vertical_prefix = metadata.class_prefix
        return ctx._resolve_class_or_factory_extension(
            "mode_config_provider",
            "mode_config",
            class_name=f"{vertical_prefix}ModeConfigProvider",
        )

    @classmethod
    def load_defaults(cls, ctx: Type[ExtensionLoaderContext]) -> Dict[str, Any]:
        """Get default mode configurations (fast, thorough, explore)."""
        return {
            "fast": {
                "name": "fast",
                "tool_budget": 10,
                "max_iterations": 20,
                "temperature": 0.7,
                "description": "Quick responses with limited tool usage",
            },
            "thorough": {
                "name": "thorough",
                "tool_budget": 50,
                "max_iterations": 50,
                "temperature": 0.7,
                "description": "Comprehensive analysis with extensive tool usage",
            },
            "explore": {
                "name": "explore",
                "tool_budget": 30,
                "max_iterations": 30,
                "temperature": 0.9,
                "description": "Exploratory mode with higher creativity",
            },
        }

    @classmethod
    def load_task_type_hints(cls, ctx: Type[ExtensionLoaderContext]) -> Dict[str, Any]:
        """Get default task type hints (edit, search, explain, debug, implement)."""
        return {
            "edit": {
                "task_type": "edit",
                "hint": "[EDIT MODE] Read target files first, then make focused modifications.",
                "tool_budget": 15,
                "priority_tools": ["read", "edit", "grep"],
            },
            "search": {
                "task_type": "search",
                "hint": "[SEARCH MODE] Use semantic search and grep for efficient discovery.",
                "tool_budget": 10,
                "priority_tools": ["grep", "code_search", "ls"],
            },
            "explain": {
                "task_type": "explain",
                "hint": "[EXPLAIN MODE] Read relevant code and provide clear explanations.",
                "tool_budget": 8,
                "priority_tools": ["read", "grep", "overview"],
            },
            "debug": {
                "task_type": "debug",
                "hint": "[DEBUG MODE] Investigate systematically, check logs and error messages.",
                "tool_budget": 20,
                "priority_tools": ["read", "grep", "shell", "run_tests"],
            },
            "implement": {
                "task_type": "implement",
                "hint": "[IMPLEMENT MODE] Plan first, implement incrementally, verify each step.",
                "tool_budget": 30,
                "priority_tools": ["read", "write", "edit", "shell"],
            },
        }
