"""Canonical compute handler registry for workflow execution.

This module is the authoritative home for:
- ComputeHandler protocol
- register_compute_handler / get_compute_handler / list_compute_handlers

victor.workflows.executor re-exports these for backward compatibility.
New code should import directly from this module.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol

from victor_contracts.workflows import ExecutorNodeStatus, NodeResult

# Prefix identifying chain handlers in YAML workflow references
# ("chain:<vertical>:<name>"). Canonical home; victor.workflows.executor
# re-exports it for backward compatibility until 0.11.0.
CHAIN_HANDLER_PREFIX = "chain:"

if TYPE_CHECKING:
    from victor.workflows.definition import ComputeNode
    from victor.tools.registry import ToolRegistry
    from victor.workflows.context import WorkflowContext

logger = logging.getLogger(__name__)


class ComputeHandler(Protocol):
    """Protocol for custom compute node handlers.

    Register handlers with register_compute_handler() to extend workflow
    execution with domain-specific logic.

    Example:
        async def rl_decision_handler(
            node: ComputeNode,
            context: WorkflowContext,
            tool_registry: ToolRegistry,
        ) -> NodeResult:
            decision = policy.predict(features)
            return NodeResult(node.id, ExecutorNodeStatus.COMPLETED, output=decision)

        register_compute_handler("rl_decision", rl_decision_handler)
    """

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        """Execute custom handler logic."""
        ...


# Global registry — module-level singleton, process-scoped
_compute_handlers: Dict[str, ComputeHandler] = {}


def register_compute_handler(name: str, handler: ComputeHandler) -> None:
    """Register a custom compute handler by name.

    Args:
        name: Handler name referenced in YAML as ``handler: <name>``
        handler: Async callable implementing the ComputeHandler protocol
    """
    _compute_handlers[name] = handler
    logger.debug(f"Registered compute handler: {name}")


def get_compute_handler(name: str) -> Optional[ComputeHandler]:
    """Return a registered compute handler or None if not found."""
    return _compute_handlers.get(name)


def list_compute_handlers() -> List[str]:
    """Return names of all registered compute handlers."""
    return list(_compute_handlers.keys())


__all__ = [
    "ComputeHandler",
    "register_compute_handler",
    "get_compute_handler",
    "list_compute_handlers",
]
