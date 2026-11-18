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

"""Unified MCP tool inventory (FEP-0049-followup B1).

Single source of truth for what the MCP server exposes. This replaces the
three divergent hand-rolled scans that drifted apart:

- ``victor/ui/commands/mcp.py`` (@tool functions only),
- ``victor/integrations/mcp/server.py`` (functions + BaseTool, own
  exclusion set, silent except-pass on instantiation failures),
- ``ToolRegistry.discover_lightweight()`` (its own 12-file exclusion list).

Deterministic: modules are scanned in sorted order and the result is sorted
by tool name, so the same tools directory always yields the same inventory.
Both tool kinds are handled: ``@tool``-decorated functions (registered as-is;
the registry unwraps ``.Tool``) and ``BaseTool`` subclasses instantiated with
a zero-argument constructor. Instantiation failures are LOGGED with the tool
name and counted — never silently swallowed.
"""

import importlib
import inspect
import logging
import os
from pathlib import Path
from typing import Any, List

# Pinned union of the three historical exclusion lists (do not trim
# casually: each entry exists because one of the three former scans
# excluded it).
DISCOVERY_EXCLUDED_FILES = frozenset(
    {
        "__init__.py",
        "base.py",
        "decorators.py",
        "semantic_selector.py",
        "common.py",
        "enums.py",
        "registry.py",
        "metadata.py",
        "metadata_registry.py",
        "tool_names.py",
        "output_utils.py",
        "shared_ast_utils.py",
        "dependency_graph.py",
        "plugin_registry.py",
    }
)

logger = logging.getLogger(__name__)

DEFAULT_TOOLS_DIR = str(Path(__file__).resolve().parents[2] / "tools")


def discover_tools(tools_dir: str | None = None) -> List[Any]:
    """Single source of truth for the MCP tool inventory.

    Scans ``victor/tools/*.py`` (or *tools_dir*, whose file stems must map to
    ``victor.tools.<stem>`` modules) for ``@tool``-decorated functions and
    ``BaseTool`` subclasses.

    Args:
        tools_dir: Override the scanned directory (defaults to the in-repo
            ``victor/tools``). Module names are still resolved as
            ``victor.tools.<stem>``.

    Returns:
        Registrable tool objects (decorated functions and zero-arg
        ``BaseTool`` instances), deterministically sorted by tool name.
        Instantiation failures are logged with the tool name and skipped.
    """
    from victor.tools.base import BaseTool

    directory = tools_dir or DEFAULT_TOOLS_DIR
    discovered: list[tuple[str, Any]] = []
    failed_instantiations: list[str] = []

    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".py") or filename in DISCOVERY_EXCLUDED_FILES:
            continue
        module_name = f"victor.tools.{filename[:-3]}"
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            logger.warning(f"Tool discovery skipped module {module_name}: {exc}")
            continue

        for _attr, obj in sorted(inspect.getmembers(module)):
            if inspect.isfunction(obj) and getattr(obj, "_is_tool", False):
                tool = getattr(obj, "Tool", None)
                name = getattr(tool, "name", None)
                if name:
                    discovered.append((name, obj))
            elif (
                inspect.isclass(obj)
                and issubclass(obj, BaseTool)
                and obj is not BaseTool
                and hasattr(obj, "name")
            ):
                try:
                    instance = obj()
                except Exception as exc:
                    failed_instantiations.append(obj.__name__)
                    logger.warning(
                        f"Tool discovery: instantiation failed for {obj.__name__} "
                        f"in {module_name} ({exc}); skipped"
                    )
                    continue
                discovered.append((instance.name, instance))

    if failed_instantiations:
        logger.warning(
            f"Tool discovery: {len(failed_instantiations)} tool(s) failed "
            f"instantiation and were skipped: {sorted(failed_instantiations)}"
        )

    # Deterministic inventory: sorted by tool name; a name discovered twice
    # keeps its first (alphabetically-first module) occurrence.
    seen: set[str] = set()
    ordered: List[Any] = []
    for name, obj in sorted(discovered, key=lambda pair: pair[0]):
        if name in seen:
            continue
        seen.add(name)
        ordered.append(obj)
    return ordered


def register_discovered_tools(registry: Any, tools_dir: str | None = None) -> int:
    """Register the full discovered inventory into *registry*.

    Args:
        registry: A ``ToolRegistry`` (or anything exposing ``register``).
        tools_dir: Optional discovery override, passed to :func:`discover_tools`.

    Returns:
        Number of tools handed to ``registry.register``.
    """
    count = 0
    for tool in discover_tools(tools_dir):
        registry.register(tool)
        count += 1
    return count
