# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""AgenticLoop executor selection (mirrors FEP-0012 ``DecisionBackend``).

A single ``executor`` enum config replaces direct reads of the
``USE_STATEGRAPH_AGENTIC_LOOP`` feature flag for choosing the AgenticLoop
executor. ``AUTO`` (default) preserves prior behavior by consulting that flag;
``STATEGRAPH`` forces the StateGraph executor regardless of the flag.

Values:
- ``AUTO`` (default): use the StateGraph executor iff
  ``USE_STATEGRAPH_AGENTIC_LOOP`` is enabled (legacy behavior).
- ``STATEGRAPH``: always use the StateGraph executor.

This keeps the user-facing config surface to ONE value with explicit semantics,
aligned with the FEP-0012 "one knob" pattern.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class AgenticLoopExecutor(str, Enum):
    """Which executor runs ``AgenticLoop.run()`` / ``stream()``."""

    AUTO = "auto"
    STATEGRAPH = "stategraph"

    @classmethod
    def parse(cls, value: Any) -> "AgenticLoopExecutor":
        """Coerce a raw config value to an executor (falls back to AUTO)."""
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            return cls.AUTO


def resolve_agentic_loop_executor() -> AgenticLoopExecutor:
    """Resolve the AgenticLoop executor.

    Explicit ``STATEGRAPH`` wins. ``AUTO`` (default) consults the legacy
    ``USE_STATEGRAPH_AGENTIC_LOOP`` feature flag, preserving every existing
    config.

    Returns:
        ``STATEGRAPH`` if the StateGraph executor should run, else ``AUTO``.
    """
    # Lazy import avoids a settings <-> executor module cycle.
    from victor.config.agentic_loop_settings import AgenticLoopSettings

    backend = AgenticLoopExecutor.parse(AgenticLoopSettings().executor)
    if backend is AgenticLoopExecutor.STATEGRAPH:
        return AgenticLoopExecutor.STATEGRAPH

    # AUTO -> legacy flag-based selection.
    try:
        from victor.core.feature_flags import FeatureFlag, is_feature_enabled

        if is_feature_enabled(FeatureFlag.USE_STATEGRAPH_AGENTIC_LOOP):
            return AgenticLoopExecutor.STATEGRAPH
    except Exception:
        pass
    return AgenticLoopExecutor.AUTO


def use_stategraph_executor() -> bool:
    """True when the StateGraph AgenticLoop executor should run."""
    return resolve_agentic_loop_executor() is AgenticLoopExecutor.STATEGRAPH
