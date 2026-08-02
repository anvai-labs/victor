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

"""HITL (Human-in-the-Loop) Coordinator.

Handles workflow execution with HITL approval nodes, integrating
with the HITLExecutor and HITLHandler infrastructure.

Features:
- Execute workflows with HITL approval nodes
- Support for custom HITL handlers (CLI, API, etc.)
- Track HITL request history
- Timeout handling with configurable fallback behavior
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)

if TYPE_CHECKING:
    from victor.framework.workflow_engine import WorkflowExecutionResult
    from victor.workflows.hitl import HITLHandler, HITLExecutor

logger = logging.getLogger(__name__)


class HITLCoordinator:
    """Coordinator for Human-in-the-Loop workflow execution.

    Handles workflow execution with HITL approval nodes, providing
    a unified interface for different HITL handler implementations
    (CLI, API, Slack, etc.).

    Example:
        coordinator = HITLCoordinator()

        # Set custom handler
        coordinator.set_handler(my_cli_handler)

        # Execute with HITL
        result = await coordinator.execute(
            "workflow.yaml",
            initial_state={"data": "value"},
        )

        # Check HITL requests
        for request in result.hitl_requests:
            print(f"Request {request['id']}: {request['status']}")
    """

    def __init__(
        self,
        handler: Optional["HITLHandler"] = None,
        timeout_seconds: int = 300,
    ) -> None:
        """Initialize the HITL coordinator.

        Args:
            handler: Optional custom HITL handler
            timeout_seconds: Default timeout for HITL requests
        """
        self._handler = handler
        self._timeout_seconds = timeout_seconds
        self._executor: Optional["HITLExecutor"] = None

    def set_handler(self, handler: "HITLHandler") -> None:
        """Set custom HITL handler.

        Args:
            handler: HITLHandler for approval nodes
        """
        self._handler = handler
        # Reset executor to use new handler
        self._executor = None

    def set_timeout(self, timeout_seconds: int) -> None:
        """Set default timeout for HITL requests.

        Args:
            timeout_seconds: Timeout in seconds
        """
        self._timeout_seconds = timeout_seconds

    def _get_executor(self) -> "HITLExecutor":
        """Get or create HITL executor."""
        if self._executor is None:
            from victor.workflows.hitl import HITLExecutor, DefaultHITLHandler

            handler = self._handler or DefaultHITLHandler()
            self._executor = HITLExecutor(handler=handler)
        return self._executor

    async def execute(
        self,
        yaml_path: Union[str, Path],
        initial_state: Optional[Dict[str, Any]] = None,
        approval_callback: Optional[Callable[[Dict[str, Any]], bool]] = None,
        **kwargs: Any,
    ) -> "WorkflowExecutionResult":
        """Execute workflow with HITL approval nodes.

        Loads and executes a YAML-defined workflow, pausing at HITL
        nodes for human approval/input.

        Args:
            yaml_path: Path to YAML workflow file
            initial_state: Initial workflow state
            approval_callback: Optional callback for approval decisions
            **kwargs: Additional execution parameters

        Returns:
            WorkflowExecutionResult with HITL request history
        """
        from victor.framework.workflow_engine import WorkflowExecutionResult
        from victor.workflows.yaml_loader import load_workflow_from_file
        from victor.workflows.hitl import HITLExecutor, DefaultHITLHandler

        start_time = time.time()
        hitl_requests: List[Dict[str, Any]] = []

        try:
            # Load workflow
            workflow_def = load_workflow_from_file(str(yaml_path))

            # Handle dict result
            if isinstance(workflow_def, dict):
                if not workflow_def:
                    raise ValueError(f"No workflows found in {yaml_path}")
                workflow_def = next(iter(workflow_def.values()))

            # Create HITL handler
            handler = self._handler or DefaultHITLHandler()

            # Create HITL executor
            executor = HITLExecutor(
                workflow=workflow_def,
                handler=handler,
            )

            # Execute with HITL
            result = await executor.execute(initial_state or {})

            duration = time.time() - start_time

            return WorkflowExecutionResult(
                success=result.success,
                final_state=result.final_state,
                nodes_executed=result.nodes_executed,
                duration_seconds=duration,
                hitl_requests=hitl_requests,
            )

        except Exception as e:
            logger.error(f"HITL workflow failed: {e}")
            return WorkflowExecutionResult(
                success=False,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )

    async def execute_with_definition(
        self,
        workflow: Any,
        initial_state: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "WorkflowExecutionResult":
        """Execute a WorkflowDefinition with HITL nodes.

        Args:
            workflow: WorkflowDefinition to execute
            initial_state: Initial workflow state
            **kwargs: Additional execution parameters

        Returns:
            WorkflowExecutionResult with HITL request history
        """
        from victor.framework.workflow_engine import WorkflowExecutionResult
        from victor.workflows.hitl import HITLExecutor, DefaultHITLHandler

        start_time = time.time()
        hitl_requests: List[Dict[str, Any]] = []

        try:
            # Create HITL handler
            handler = self._handler or DefaultHITLHandler()

            # Create HITL executor
            executor = HITLExecutor(
                workflow=workflow,
                handler=handler,
            )

            # Execute with HITL
            result = await executor.execute(initial_state or {})

            duration = time.time() - start_time

            return WorkflowExecutionResult(
                success=result.success,
                final_state=result.final_state,
                nodes_executed=result.nodes_executed,
                duration_seconds=duration,
                hitl_requests=hitl_requests,
            )

        except Exception as e:
            logger.error(f"HITL workflow failed: {e}")
            return WorkflowExecutionResult(
                success=False,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )


__all__ = ["HITLCoordinator"]
