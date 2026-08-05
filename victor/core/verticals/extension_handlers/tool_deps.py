# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Tool dependency, tool graph, and tiered tool config handlers."""

from __future__ import annotations

from typing import Any, Optional, Type

from victor.core.verticals.extension_handlers.base import (
    BaseExtensionHandler,
    ExtensionLoaderContext,
)
from victor.core.verticals.extension_handlers.registry import ExtensionHandlerRegistry


@ExtensionHandlerRegistry.register
class ToolDepsHandler(BaseExtensionHandler):
    """Loads tool dependency provider for a vertical.

    Resolution order:
    1. Entry-point loader: load_tool_dependency_provider_from_entry_points
    2. Fallback: create_vertical_tool_dependency_provider factory
    """

    extension_type = "tool_deps"

    @classmethod
    def load(cls, ctx: Type[ExtensionLoaderContext]) -> Optional[Any]:
        try:
            from victor.framework.entry_point_loader import (
                load_tool_dependency_provider_from_entry_points,
            )

            return ctx._load_cached_optional_extension(
                "tool_dependency_provider",
                lambda: load_tool_dependency_provider_from_entry_points(ctx.name),
            )
        except ImportError:
            try:
                from victor.core.tool_dependency_loader import (
                    create_vertical_tool_dependency_provider,
                )

                return ctx._load_cached_optional_extension(
                    "tool_dependency_provider",
                    lambda: create_vertical_tool_dependency_provider(ctx.name),
                )
            except (ImportError, ValueError):
                return None

    @classmethod
    def load_tool_graph(cls, ctx: Type[ExtensionLoaderContext]) -> Optional[Any]:
        """Stub — override in vertical for custom tool execution graph."""
        return None

    @classmethod
    def load_tiered_tool_config(cls, ctx: Type[ExtensionLoaderContext]) -> Optional[Any]:
        """Stub — actual implementation in VerticalMetadataProvider."""
        return None
