# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compatibility wrapper for the SDK-owned tool naming registry.

New definition-layer code should import these symbols from `victor_contracts`
or `victor_contracts.constants`. This module remains for backward compatibility
with existing victor-ai callers during the migration.
"""

from __future__ import annotations

from typing import Any

try:
    # Direct re-exports from the SDK -- identity-preserving so that
    # ``from victor.tools.tool_names import ToolNames`` is the *same*
    # object as ``from victor_contracts import ToolNames``.
    from victor_contracts.constants import (
        ToolNames,
        ToolNameEntry,
        CANONICAL_TO_ALIASES,
        TOOL_ALIASES,
        get_canonical_name,
        get_aliases,
        is_valid_tool_name,
        get_all_canonical_names,
        get_name_mapping,
    )
except ImportError:
    # SDK not installed -- provide minimal fallbacks so the module can
    # still be imported in environments that only have victor-ai.

    class _SDKFallback:
        """Fallback for when SDK is not available."""

        def __init__(self, name: str):
            self._name = name

        def __getattr__(self, name: str) -> Any:
            return self

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return None

        def __eq__(self, other: Any) -> bool:
            return isinstance(other, str) and other == self._name

        def __hash__(self) -> int:
            return hash(self._name)

        def __repr__(self) -> str:
            return f"<ToolNames.{self._name} (SDK not available)>"

    class _SDKLazyLoader:
        """Lazy loader that provides attribute-like access without the SDK."""

        def __getattr__(self, name: str) -> Any:
            return _SDKFallback(name)

    ToolNames = _SDKLazyLoader()  # type: ignore[assignment,misc]
    ToolNameEntry = type("ToolNameEntry", (), {})  # type: ignore[assignment,misc]
    CANONICAL_TO_ALIASES: dict = {}  # type: ignore[no-redef]
    TOOL_ALIASES: dict = {}  # type: ignore[no-redef]

    def get_canonical_name(name: str) -> str:  # type: ignore[no-redef]
        return name

    def get_aliases(name: str) -> list:  # type: ignore[no-redef]
        return []

    def is_valid_tool_name(name: str) -> bool:  # type: ignore[no-redef]
        return bool(name)

    def get_all_canonical_names() -> list:  # type: ignore[no-redef]
        return []

    def get_name_mapping() -> dict:  # type: ignore[no-redef]
        return {}


__all__ = [
    "ToolNames",
    "ToolNameEntry",
    "TOOL_ALIASES",
    "CANONICAL_TO_ALIASES",
    "get_canonical_name",
    "get_aliases",
    "is_valid_tool_name",
    "get_all_canonical_names",
    "get_name_mapping",
]
