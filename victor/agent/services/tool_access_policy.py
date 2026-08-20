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

"""Tool availability and alias policy for ``ToolService``."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional


class ToolAccessPolicy:
    """Owns enabled-tool filtering, registrar discovery, and alias resolution."""

    def __init__(
        self,
        *,
        get_registrar: Callable[[], Optional[Any]],
        get_selector: Callable[[], Optional[Any]],
        logger: logging.Logger,
    ):
        self._get_registrar = get_registrar
        self._get_selector = get_selector
        self._logger = logger
        self.enabled_tools: Optional[set[str]] = None

    def set_enabled_tools(self, tools: set[str]) -> None:
        """Set the explicitly enabled tools for this session."""
        self.enabled_tools = set(tools) if tools else None
        self._logger.info("Enabled tools: %s", sorted(tools) if tools else "all")

        selector = self._get_selector()
        if selector is not None and hasattr(selector, "set_enabled_tools"):
            selector.set_enabled_tools(tools or set())
            self._logger.debug("Propagated enabled tools to selector")

    def get_enabled_tools(self) -> set[str]:
        """Return explicitly enabled tools or all available tools."""
        if self.enabled_tools is not None:
            return self.enabled_tools.copy()
        return self.get_available_tools()

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Return whether a tool is currently enabled."""
        if self.enabled_tools is None:
            return True
        return tool_name in self.enabled_tools

    def get_available_tools(self) -> set[str]:
        """Return all registered tool names from the bound registrar."""
        registrar = self._get_registrar()
        if not registrar:
            return set()

        try:
            if hasattr(registrar, "get_registered_tools"):
                return set(registrar.get_registered_tools())
            if hasattr(registrar, "get_tool_names"):
                return set(registrar.get_tool_names())
            if hasattr(registrar, "list_tools"):
                names: set[str] = set()
                for tool in registrar.list_tools():
                    if isinstance(tool, str):
                        names.add(tool)
                        continue
                    tool_name = getattr(tool, "name", None)
                    if isinstance(tool_name, str) and tool_name:
                        names.add(tool_name)
                return names
        except Exception as exc:
            self._logger.warning("Failed to get available tools: %s", exc)
        return set()

    def resolve_tool_alias(self, tool_name: str) -> str:
        """Resolve a model-supplied tool name to the canonical enabled tool name."""
        from victor.tools.core_tool_aliases import normalize_model_tool_name
        from victor.tools.tool_names import ToolNames, get_canonical_name

        normalized_tool_name = normalize_model_tool_name(tool_name)
        canonical = get_canonical_name(normalized_tool_name)
        shell_aliases = {
            "shell",
            "run",
            "bash",
            "execute",
            "cmd",
            "execute_bash",
        }

        try:
            if hasattr(ToolNames, "SHELL"):
                shell_aliases.add(str(ToolNames.SHELL))
        except Exception:
            pass

        if canonical not in shell_aliases and normalized_tool_name not in shell_aliases:
            if canonical != tool_name:
                self._logger.debug("Resolved '%s' to canonical '%s'", tool_name, canonical)
            return canonical

        try:
            shell_canonical = str(ToolNames.SHELL) if hasattr(ToolNames, "SHELL") else "shell"
        except Exception:
            shell_canonical = "shell"

        if self.is_tool_enabled(shell_canonical):
            self._logger.debug("Resolved '%s' to '%s' (shell enabled)", tool_name, shell_canonical)
            return shell_canonical

        self._logger.debug(
            "Shell tool not enabled for '%s', using canonical '%s'",
            tool_name,
            canonical,
        )
        return canonical


__all__ = ["ToolAccessPolicy"]
