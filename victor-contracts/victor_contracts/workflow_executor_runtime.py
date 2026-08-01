"""SDK host adapters for workflow executor runtime helpers."""

from __future__ import annotations

import importlib
import warnings
from typing import TYPE_CHECKING, Any

from victor_contracts._deprecation import deprecated_bridge_message

if TYPE_CHECKING:
    from victor.workflows.definition import ComputeNode
    from victor.workflows.executor import (
        ExecutorNodeStatus,
        NodeResult,
        WorkflowContext,
        WorkflowExecutor,
        WorkflowResult,
        register_compute_handler,
    )

__all__ = [
    "WorkflowExecutor",
    "WorkflowContext",
    "WorkflowResult",
    "NodeResult",
    "ExecutorNodeStatus",
    "register_compute_handler",
    "ComputeNode",
]

_LAZY_IMPORTS = {
    "WorkflowExecutor": "victor.workflows.executor",
    "WorkflowContext": "victor.workflows.executor",
    "WorkflowResult": "victor.workflows.executor",
    "NodeResult": "victor.workflows.executor",
    "ExecutorNodeStatus": "victor.workflows.executor",
    "register_compute_handler": "victor.workflows.executor",
    "ComputeNode": "victor.workflows.definition",
}


def __getattr__(name: str) -> Any:
    """Resolve workflow executor helpers lazily from the Victor host runtime."""
    warnings.warn(
        deprecated_bridge_message("workflow_executor_runtime"),
        DeprecationWarning,
        stacklevel=2,
    )
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(
            f"module 'victor_contracts.workflow_executor_runtime' has no attribute {name!r}"
        )

    module = importlib.import_module(module_name)
    return getattr(module, name)
