"""Canonical accessors for tool-selection configuration."""

from __future__ import annotations

import os
from typing import Any

# Truthy spellings for the legacy flat env override. A set-but-empty value is OFF
# (never an error) - deliberately not routed through pydantic bool coercion.
_TRUTHY = {"1", "true", "yes", "on"}


def is_semantic_tool_selection_enabled(settings: Any, default: bool = True) -> bool:
    """Return semantic-selection enablement from the canonical nested config.

    Preferred order:
    1. ``settings.tool_selection.use_semantic_tool_selection`` (canonical)
    2. ``settings.tools.use_semantic_tool_selection`` (older nested mirror)
    3. flat ``settings.use_semantic_tool_selection`` (legacy compatibility)
    """

    tool_selection = getattr(settings, "tool_selection", None)
    if tool_selection is not None and hasattr(tool_selection, "use_semantic_tool_selection"):
        return bool(tool_selection.use_semantic_tool_selection)

    tools = getattr(settings, "tools", None)
    if tools is not None and hasattr(tools, "use_semantic_tool_selection"):
        return bool(tools.use_semantic_tool_selection)

    return bool(getattr(settings, "use_semantic_tool_selection", default))


def is_tool_selection_enabled(settings: Any = None, *, config_override: Any = None) -> bool:
    """Return per-turn tool-pruning enablement from the single decision point.

    Per-turn pruning narrows the toolset offered to the LLM each turn. It is
    OPT-IN (default off: every enabled registered tool reaches the call as-is)
    because a narrow supply starves agentic loops that need tools the ranker
    did not surface.

    Precedence:
    1. ``config_override`` (explicit DI; ``None`` = defer)
    2. ``settings.tools.tool_selection_enabled is True`` (canonical field;
       ``False`` defers so the legacy flat env can still opt in)
    3. flat env ``VICTOR_TOOL_SELECTION`` (legacy compat; a set-but-empty value
       is off - deliberately not routed through pydantic bool coercion, which
       would reject ``""``)
    4. False
    """

    if config_override is not None:
        return bool(config_override)

    tools = getattr(settings, "tools", None)
    if tools is not None and getattr(tools, "tool_selection_enabled", False):
        return True

    return os.environ.get("VICTOR_TOOL_SELECTION", "").strip().lower() in _TRUTHY


__all__ = ["is_semantic_tool_selection_enabled", "is_tool_selection_enabled"]
