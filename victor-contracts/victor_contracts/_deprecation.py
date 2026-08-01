"""Deprecation helpers for the ``victor_contracts`` package.

Single source of truth for the deprecated runtime bridge modules and their
replacement pointers, as classified in ``CONTRACT_STABILITY.md``. Deprecated
bridges emit ``DeprecationWarning`` on attribute access starting with the
0.9.0 release and are removed no earlier than 0.10.0.
"""

from __future__ import annotations

REMOVAL_VERSION = "0.10.0"

# Deprecated bridge module name -> replacement pointer (per CONTRACT_STABILITY.md).
DEPRECATED_BRIDGE_REPLACEMENTS: dict[str, str] = {
    "agent_spec_runtime": "victor.agent.specs.models",
    "graph_runtime": "victor.framework.graph",
    "handler_runtime": (
        "victor_contracts.workflow_runtime (register_compute_handler) "
        "or victor.framework.handler_registry"
    ),
    "subagent_runtime": "victor.agent.subagents.protocols",
    "tool_runtime": "victor.framework.tools",
    "workflow_executor_runtime": "victor_contracts.workflow_runtime",
}


def deprecated_bridge_message(module: str) -> str:
    """Build the ``DeprecationWarning`` message for a deprecated bridge module.

    Args:
        module: Bare module name (e.g. ``"graph_runtime"``); must be one of
            the keys in ``DEPRECATED_BRIDGE_REPLACEMENTS``.

    Returns:
        The warning message announcing removal in ``REMOVAL_VERSION`` and the
        replacement import path.
    """
    replacement = DEPRECATED_BRIDGE_REPLACEMENTS[module]
    return (
        f"victor_contracts.{module} is deprecated and will be removed in "
        f"victor-contracts {REMOVAL_VERSION}; use {replacement}"
    )
