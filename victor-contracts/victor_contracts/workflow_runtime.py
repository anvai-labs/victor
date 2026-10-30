"""SDK host adapters for workflow runtime helpers.

These names are intentionally hosted in the SDK so extracted verticals can
depend on ``victor_contracts`` instead of importing ``victor.framework`` directly.
They remain runtime-backed adapters rather than pure definition-layer
contracts, so they should be treated as a compatibility seam on the migration
path toward descriptor-first workflows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from victor_contracts._lazy import make_lazy_getattr

if TYPE_CHECKING:
    from victor.framework.workflows import BaseYAMLWorkflowProvider
    from victor.workflows.definition import (
        ComputeNode,
        WorkflowBuilder,
        WorkflowDefinition,
        workflow,
    )
    from victor.workflows.executor import (
        ExecutorNodeStatus,
        NodeResult,
        WorkflowContext,
        WorkflowExecutor,
        WorkflowResult,
        register_compute_handler,
    )

__all__ = [
    "BaseYAMLWorkflowProvider",
    "ComputeNode",
    "ExecutorNodeStatus",
    "NodeResult",
    "WorkflowBuilder",
    "WorkflowContext",
    "WorkflowDefinition",
    "WorkflowExecutor",
    "WorkflowResult",
    "register_compute_handler",
    "workflow",
]

_LAZY_IMPORTS = {
    "BaseYAMLWorkflowProvider": "victor.framework.workflows",
    "ComputeNode": "victor.workflows.definition",
    "ExecutorNodeStatus": "victor_contracts.workflows",
    "NodeResult": "victor_contracts.workflows",
    "WorkflowBuilder": "victor.workflows.definition",
    "WorkflowContext": "victor.workflows.context",
    "WorkflowDefinition": "victor.workflows.definition",
    "WorkflowExecutor": "victor.workflows.unified_executor",
    "WorkflowResult": "victor.workflows.context",
    "register_compute_handler": "victor.workflows.compute_registry",
    "workflow": "victor.workflows.definition",
}


__getattr__ = make_lazy_getattr(__name__, _LAZY_IMPORTS)
