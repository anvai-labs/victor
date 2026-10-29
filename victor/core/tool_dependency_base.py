"""Base Tool Dependency Provider — backward compatibility shim.

Canonical implementations have been promoted to victor-contracts.
This module re-exports them for backward compatibility.

Usage (preferred — import from SDK):
    from victor_contracts.verticals.tool_dependencies import BaseToolDependencyProvider

Usage (backward compat — still works):
    from victor.core.tool_dependency_base import BaseToolDependencyProvider
"""

from __future__ import annotations

# Re-export from SDK when available. The module may not exist yet if
# the SDK hasn't been updated with the tool_dependencies subpackage.
try:
    from victor_contracts.verticals.tool_dependencies import (  # noqa: F401
        BaseToolDependencyProvider,
        EmptyToolDependencyProvider,
        ToolDependencyConfig,
        ToolDependencyLoadError,
        create_vertical_tool_dependency_provider,
    )
except ImportError:
    # SDK doesn't have tool_dependencies yet — provide stubs
    from typing import Any, List

    class BaseToolDependencyProvider:  # type: ignore[no-redef]
        """Stub — install updated victor-contracts for full implementation."""

        def get_dependencies(self, tool_name: str) -> List[str]:
            return []

    class EmptyToolDependencyProvider(BaseToolDependencyProvider):  # type: ignore[no-redef]
        pass

    class ToolDependencyConfig:  # type: ignore[no-redef]
        pass

    class ToolDependencyLoadError(Exception):  # type: ignore[no-redef]
        pass

    def create_vertical_tool_dependency_provider(
        *args: Any, **kwargs: Any
    ) -> BaseToolDependencyProvider:
        return EmptyToolDependencyProvider()


__all__ = [
    "BaseToolDependencyProvider",
    "EmptyToolDependencyProvider",
    "ToolDependencyConfig",
    "ToolDependencyLoadError",
    "create_vertical_tool_dependency_provider",
]
