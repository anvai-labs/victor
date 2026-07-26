# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Delegation handler for processing delegation requests.

This module provides DelegationHandler, which is responsible for:
- Validating delegation requests
- Selecting appropriate delegate roles and tools
- Spawning delegate agents via SubAgentOrchestrator
- Tracking active delegations
- Returning results to requesting agents

The handler integrates with Victor's SubAgent infrastructure to reuse
existing agent spawning and execution logic.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set

from victor.agent.subagents.base import SubAgentRole, SubAgentResult
from victor.agent.subagents.orchestrator import (
    ROLE_DEFAULT_BUDGETS,
    ROLE_DEFAULT_TOOLS,
    SubAgentOrchestrator,
)
from victor.observability.run_kind import RunKind, tagged_run
from victor.agent.delegation.protocol import (
    DelegationPriority,
    DelegationRequest,
    DelegationResponse,
    DelegationStatus,
)

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator

logger = logging.getLogger(__name__)


# Mapping from role name strings to SubAgentRole enum
ROLE_MAPPING: Dict[str, SubAgentRole] = {
    "researcher": SubAgentRole.RESEARCHER,
    "research": SubAgentRole.RESEARCHER,
    "planner": SubAgentRole.PLANNER,
    "plan": SubAgentRole.PLANNER,
    "executor": SubAgentRole.EXECUTOR,
    "execute": SubAgentRole.EXECUTOR,
    "implement": SubAgentRole.EXECUTOR,
    "reviewer": SubAgentRole.REVIEWER,
    "review": SubAgentRole.REVIEWER,
    "tester": SubAgentRole.TESTER,
    "test": SubAgentRole.TESTER,
}


class ActiveDelegation:
    """Tracks an active delegation.

    Attributes:
        request: Original delegation request
        delegate_id: ID of the spawned delegate
        start_time: When delegation started
        task: Asyncio task (for cancellation)
    """

    def __init__(
        self,
        request: DelegationRequest,
        delegate_id: str,
    ):
        self.request = request
        self.delegate_id = delegate_id
        self.start_time = time.time()
        self.task: Optional[asyncio.Task] = None
        self.result: Optional[DelegationResponse] = None


class DelegationHandler:
    """Handles delegation requests from agents.

    Provides the core logic for processing delegation requests, spawning
    appropriate delegate agents, and returning results.

    Attributes:
        orchestrator: Parent orchestrator
        sub_agent_orchestrator: Underlying sub-agent infrastructure
        max_concurrent: Maximum concurrent delegations

    Example:
        handler = DelegationHandler(orchestrator)

        response = await handler.handle(DelegationRequest(
            task="Find all API endpoints",
            suggested_role="researcher",
            tool_budget=15,
        ))

        if response.success:
            print(response.result)
    """

    def __init__(
        self,
        orchestrator: "AgentOrchestrator",
        sub_agent_orchestrator: Optional[SubAgentOrchestrator] = None,
        max_concurrent: int = 4,
    ):
        """Initialize delegation handler.

        Args:
            orchestrator: Parent orchestrator
            sub_agent_orchestrator: Optional sub-agent orchestrator
            max_concurrent: Maximum concurrent delegations
        """
        self.orchestrator = orchestrator
        self.sub_agents = sub_agent_orchestrator or SubAgentOrchestrator(orchestrator)
        self.max_concurrent = max_concurrent
        self._active: Dict[str, ActiveDelegation] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._on_complete: Optional[Callable[[DelegationResponse], None]] = None

        logger.info(f"DelegationHandler initialized (max_concurrent={max_concurrent})")

    def set_completion_callback(
        self,
        callback: Callable[[DelegationResponse], None],
    ) -> None:
        """Set callback for delegation completion.

        Args:
            callback: Function to call when a delegation completes
        """
        self._on_complete = callback

    async def handle(self, request: DelegationRequest) -> DelegationResponse:
        """Handle a delegation request.

        Validates the request, spawns a delegate agent, and returns the result.
        If await_result is False, returns immediately with a pending status.

        Args:
            request: Delegation request to handle

        Returns:
            DelegationResponse with result or status
        """
        logger.info(
            f"Handling delegation from '{request.from_agent}': "
            f"{request.task[:50]}... (role={request.suggested_role})"
        )

        # Validate request
        validation_error = self._validate_request(request)
        if validation_error:
            return DelegationResponse.rejected(request.delegation_id, validation_error)

        # Check concurrent limit
        if len(self._active) >= self.max_concurrent:
            return DelegationResponse.rejected(
                request.delegation_id,
                f"Maximum concurrent delegations ({self.max_concurrent}) reached",
            )

        # Determine role
        role = self._resolve_role(request)
        if role is None:
            return DelegationResponse.rejected(
                request.delegation_id,
                f"Unknown role: {request.suggested_role}",
            )

        # Create delegation tracking
        delegate_id = f"delegate_{uuid.uuid4().hex[:8]}"
        delegation = ActiveDelegation(request, delegate_id)
        self._active[request.delegation_id] = delegation

        try:
            if request.await_result:
                # Synchronous: wait for result
                return await self._execute_delegation(delegation, role)
            else:
                # Asynchronous: start and return immediately
                delegation.task = asyncio.create_task(self._execute_delegation(delegation, role))
                return DelegationResponse.pending(request.delegation_id, delegate_id)

        except Exception as e:
            logger.error(f"Delegation failed: {e}", exc_info=True)
            self._active.pop(request.delegation_id, None)
            return DelegationResponse(
                delegation_id=request.delegation_id,
                accepted=True,
                status=DelegationStatus.FAILED,
                delegate_id=delegate_id,
                error=str(e),
            )

    @tagged_run(RunKind.DELEGATE)
    async def _execute_delegation(
        self,
        delegation: ActiveDelegation,
        role: SubAgentRole,
    ) -> DelegationResponse:
        """Execute a delegation.

        Everything the delegate emits is tagged as delegate work. Without it a
        delegate is indistinguishable from a benchmark run downstream: both
        carry the shared agentic loop's turn-budget notice, and classifying on
        that text overstated the eval share of trace evidence roughly twofold.

        Args:
            delegation: Active delegation tracking
            role: Role for the delegate

        Returns:
            DelegationResponse with result
        """
        request = delegation.request

        async with self._semaphore:
            try:
                # Build context for delegate
                context = self._build_delegate_context(request)

                # Determine tools
                tools = request.required_tools or ROLE_DEFAULT_TOOLS.get(role, ["read", "ls"])

                # Calculate timeout
                timeout = request.deadline_seconds or 300

                # Spawn delegate
                logger.debug(f"Spawning {role.value} delegate for task: {request.task[:50]}...")

                result = await self.sub_agents.spawn(
                    role=role,
                    task=context,
                    tool_budget=request.tool_budget,
                    allowed_tools=tools,
                    timeout_seconds=int(timeout),
                )

                # Convert result
                response = self._convert_result(request, delegation, result)

                # Notify callback
                if self._on_complete:
                    try:
                        self._on_complete(response)
                    except Exception as e:
                        logger.debug(f"Completion callback failed: {e}")

                return response

            except asyncio.TimeoutError:
                return DelegationResponse(
                    delegation_id=request.delegation_id,
                    accepted=True,
                    status=DelegationStatus.TIMEOUT,
                    delegate_id=delegation.delegate_id,
                    error=f"Delegation timed out after {request.deadline_seconds}s",
                    duration_seconds=time.time() - delegation.start_time,
                )

            finally:
                # Clean up
                self._active.pop(request.delegation_id, None)

    def _validate_request(self, request: DelegationRequest) -> Optional[str]:
        """Validate a delegation request.

        Args:
            request: Request to validate

        Returns:
            Error message if invalid, None if valid
        """
        if not request.task or not request.task.strip():
            return "Task cannot be empty"

        if request.tool_budget < 1:
            return "Tool budget must be at least 1"

        if request.tool_budget > 100:
            return "Tool budget exceeds maximum (100)"

        if request.deadline_seconds is not None and request.deadline_seconds < 5:
            return "Deadline too short (minimum 5 seconds)"

        return None

    def _resolve_role(self, request: DelegationRequest) -> Optional[SubAgentRole]:
        """Resolve the role for a delegation.

        Uses suggested_role if valid, otherwise infers from task keywords.

        Args:
            request: Delegation request

        Returns:
            Resolved SubAgentRole or None
        """
        # Try suggested role first
        if request.suggested_role:
            role_str = request.suggested_role.lower()
            if role_str in ROLE_MAPPING:
                return ROLE_MAPPING[role_str]

        # Infer from task keywords
        task_lower = request.task.lower()

        if any(kw in task_lower for kw in ["find", "search", "research", "explore", "discover"]):
            return SubAgentRole.RESEARCHER
        elif any(kw in task_lower for kw in ["plan", "design", "architect", "organize"]):
            return SubAgentRole.PLANNER
        elif any(
            kw in task_lower for kw in ["implement", "create", "build", "write", "edit", "fix"]
        ):
            return SubAgentRole.EXECUTOR
        elif any(kw in task_lower for kw in ["review", "check", "verify", "validate", "audit"]):
            return SubAgentRole.REVIEWER
        elif any(kw in task_lower for kw in ["test", "spec", "assert", "mock"]):
            return SubAgentRole.TESTER

        # Default to executor
        return SubAgentRole.EXECUTOR

    def _build_delegate_context(self, request: DelegationRequest) -> str:
        """Build context string for the delegate.

        Args:
            request: Delegation request

        Returns:
            Context string for the delegate
        """
        lines = [
            "You are a delegate agent spawned to complete a specific task.",
            "",
            "## Task",
            request.task,
        ]

        if request.parent_goal:
            lines.extend(
                [
                    "",
                    "## Parent Agent's Goal",
                    request.parent_goal,
                ]
            )

        if request.context:
            lines.extend(
                [
                    "",
                    "## Additional Context",
                ]
            )
            for key, value in request.context.items():
                lines.append(f"- **{key}**: {value}")

        lines.extend(
            [
                "",
                "## Instructions",
                "1. Focus exclusively on the assigned task",
                "2. Be thorough but efficient with tool usage",
                "3. Summarize your findings clearly",
                "4. Report any blockers or issues encountered",
            ]
        )

        return "\n".join(lines)

    def _convert_result(
        self,
        request: DelegationRequest,
        delegation: ActiveDelegation,
        result: SubAgentResult,
    ) -> DelegationResponse:
        """Convert SubAgentResult to DelegationResponse.

        Args:
            request: Original request
            delegation: Delegation tracking
            result: SubAgent result

        Returns:
            DelegationResponse
        """
        status = DelegationStatus.COMPLETED if result.success else DelegationStatus.FAILED

        # Extract discoveries from result
        discoveries = []
        if result.success and result.summary:
            for line in result.summary.split("\n"):
                line = line.strip()
                if any(
                    line.lower().startswith(prefix)
                    for prefix in ["found", "discovered", "identified", "located"]
                ):
                    discoveries.append(line)

        return DelegationResponse(
            delegation_id=request.delegation_id,
            accepted=True,
            status=status,
            delegate_id=delegation.delegate_id,
            result=result.summary if result.success else None,
            error=result.error,
            tool_calls_used=result.tool_calls_used,
            duration_seconds=result.duration_seconds,
            discoveries=discoveries,
        )

    async def cancel(self, delegation_id: str) -> bool:
        """Cancel an active delegation.

        Args:
            delegation_id: ID of delegation to cancel

        Returns:
            True if cancelled, False if not found
        """
        delegation = self._active.get(delegation_id)
        if not delegation:
            return False

        if delegation.task and not delegation.task.done():
            delegation.task.cancel()

        self._active.pop(delegation_id, None)
        logger.info(f"Cancelled delegation: {delegation_id}")
        return True

    def get_status(self, delegation_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a delegation.

        Args:
            delegation_id: Delegation ID

        Returns:
            Status dictionary or None
        """
        delegation = self._active.get(delegation_id)
        if not delegation:
            return None

        return {
            "delegation_id": delegation_id,
            "delegate_id": delegation.delegate_id,
            "from_agent": delegation.request.from_agent,
            "task": delegation.request.task[:100],
            "role": delegation.request.suggested_role,
            "running": delegation.task is not None and not delegation.task.done(),
            "duration": time.time() - delegation.start_time,
        }

    def get_active_count(self) -> int:
        """Get number of active delegations.

        Returns:
            Count of active delegations
        """
        return len(self._active)

    async def wait_for_completion(
        self,
        delegation_id: str,
        timeout: Optional[float] = None,
    ) -> Optional[DelegationResponse]:
        """Wait for a delegation to complete.

        Args:
            delegation_id: Delegation ID
            timeout: Maximum wait time

        Returns:
            DelegationResponse or None if not found/timeout
        """
        delegation = self._active.get(delegation_id)
        if not delegation:
            return None

        if delegation.task is None:
            return delegation.result

        try:
            if timeout:
                return await asyncio.wait_for(delegation.task, timeout=timeout)
            else:
                return await delegation.task
        except asyncio.TimeoutError:
            return None


__all__ = [
    "DelegationHandler",
    "ActiveDelegation",
    "ROLE_MAPPING",
]
