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

"""Human-in-the-Loop (HITL) Protocol for agent workflows.

This module provides infrastructure for human approval flows and agent
interruption/resume capabilities. It enables workflows where critical
operations require human approval before proceeding.

Example:
    from victor.framework.hitl import HITLController, ApprovalStatus

    # Create controller with custom approval handler
    async def my_approval_handler(request):
        # Send to Slack, email, etc.
        response = await notify_human(request)
        return response.status, response.message, response.user

    controller = HITLController(approval_handler=my_approval_handler)

    # Request approval for dangerous operation
    request = controller.request_approval(
        title="Delete Database",
        description="This will delete all user data",
        context={"database": "production", "tables": ["users", "orders"]},
        timeout_seconds=300,
    )

    # Wait for human response
    result = await controller.wait_for_approval(request.id)
    if result.is_approved:
        # Proceed with operation
        pass
    else:
        # Handle rejection
        pass

    # Or use interrupt/resume for pausing workflows
    checkpoint_id = controller.interrupt(context={"step": 5})
    # ... later ...
    controller.resume(checkpoint_id)
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

from victor.framework.execution_checkpoint import (
    ApprovalState,
    normalize_execution_checkpoint_context,
)


class ApprovalStatus(str, Enum):
    """Status of an approval request.

    Attributes:
        PENDING: Request is awaiting human response
        APPROVED: Request was approved by human
        REJECTED: Request was rejected by human
        TIMEOUT: Request timed out without response
    """

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


@dataclass
class ApprovalRequest:
    """A request for human approval.

    Represents a pending decision that requires human input before
    the agent can proceed with an operation.

    Attributes:
        id: Unique identifier for this request
        title: Short title describing what needs approval
        description: Detailed description of the operation
        context: Additional context data for the human reviewer
        timeout_seconds: How long to wait for approval (default: 300)
        status: Current status of the request
        response: Human's response message (if any)
        responder: Identity of the human who responded (if any)
        created_at: Unix timestamp when request was created
    """

    id: str
    title: str
    description: str
    context: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 300
    status: ApprovalStatus = ApprovalStatus.PENDING
    response: Optional[str] = None
    responder: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    @property
    def is_pending(self) -> bool:
        """Check if request is still pending."""
        return self.status == ApprovalStatus.PENDING

    @property
    def is_approved(self) -> bool:
        """Check if request was approved."""
        return self.status == ApprovalStatus.APPROVED

    @property
    def is_rejected(self) -> bool:
        """Check if request was rejected."""
        return self.status == ApprovalStatus.REJECTED

    @property
    def is_timeout(self) -> bool:
        """Check if request timed out."""
        return self.status == ApprovalStatus.TIMEOUT

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "context": self.context,
            "timeout_seconds": self.timeout_seconds,
            "status": self.status.value,
            "response": self.response,
            "responder": self.responder,
            "created_at": self.created_at,
        }


# Type alias for approval handler functions
ApprovalHandler = Callable[
    [ApprovalRequest],
    Awaitable[Tuple[ApprovalStatus, Optional[str], Optional[str]]],
]


@dataclass
class HITLCheckpoint:
    """HITLCheckpoint for Human-in-the-Loop pause/resume functionality.

    Renamed from Checkpoint to be semantically distinct:
    - GitCheckpoint (victor.agent.checkpoints): Git stash-based
    - ExecutionCheckpoint (victor.agent.time_aware_executor): Time/progress tracking
    - WorkflowCheckpoint (victor.framework.graph): Workflow state persistence
    - HITLCheckpoint (here): Pause/resume with context for human interaction

    Attributes:
        id: Unique checkpoint identifier
        context: Workflow state at time of interrupt
        created_at: Unix timestamp when checkpoint was created
    """

    id: str
    context: Dict[str, Any]
    created_at: float = field(default_factory=time.time)


class HITLController:
    """Controller for Human-in-the-Loop interactions.

    Manages approval requests and workflow pause/resume functionality.
    Can be configured with custom approval handlers for integration
    with external notification systems (Slack, email, etc.).

    Attributes:
        is_paused: Whether the controller is currently paused

    Example:
        controller = HITLController()

        # Interrupt workflow
        checkpoint_id = controller.interrupt(context={"step": 5})

        # Later resume
        controller.resume(checkpoint_id)

        # Request approval
        request = controller.request_approval(
            title="Deploy",
            description="Deploy to production",
        )
        result = await controller.wait_for_approval(request.id)
    """

    def __init__(
        self,
        approval_handler: Optional[ApprovalHandler] = None,
    ):
        """Initialize HITLController.

        Args:
            approval_handler: Optional async function to handle approval requests.
                              Should return (status, response, responder) tuple.
        """
        self._approval_handler = approval_handler
        self._paused = False
        self._checkpoints: Dict[str, HITLCheckpoint] = {}
        self._requests: Dict[str, ApprovalRequest] = {}

        # Callbacks
        self._on_pause_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        self._on_resume_callbacks: List[Callable[[str], None]] = []
        self._on_approval_request_callbacks: List[Callable[[ApprovalRequest], None]] = []

    # =========================================================================
    # Pause/Resume Properties and Methods
    # =========================================================================

    @property
    def is_paused(self) -> bool:
        """Check if controller is currently paused."""
        return self._paused

    def interrupt(self, context: Optional[Dict[str, Any]] = None) -> str:
        """Interrupt the workflow and create a checkpoint.

        Pauses the workflow and stores the current context for later
        resumption.

        Args:
            context: Optional workflow state to preserve

        Returns:
            HITLCheckpoint ID for resuming later
        """
        checkpoint_id = f"cp_{uuid.uuid4().hex}"
        checkpoint_context = normalize_execution_checkpoint_context(context)
        checkpoint = HITLCheckpoint(
            id=checkpoint_id,
            context=checkpoint_context,
        )
        self._checkpoints[checkpoint_id] = checkpoint
        self._paused = True

        # Notify callbacks
        for callback in self._on_pause_callbacks:
            callback(checkpoint_id, checkpoint.context)

        return checkpoint_id

    def resume(self, checkpoint_id: str) -> Dict[str, Any]:
        """Resume from a checkpoint.

        Args:
            checkpoint_id: ID of the checkpoint to resume from

        Returns:
            The context stored in the checkpoint

        Raises:
            ValueError: If checkpoint_id is not found
        """
        if checkpoint_id not in self._checkpoints:
            raise ValueError(f"Invalid checkpoint ID: {checkpoint_id}")

        checkpoint = self._checkpoints[checkpoint_id]
        self._paused = False

        # Notify callbacks
        for callback in self._on_resume_callbacks:
            callback(checkpoint_id)

        return checkpoint.context

    def get_checkpoint_context(self, checkpoint_id: str) -> Dict[str, Any]:
        """Get the context stored in a checkpoint.

        Args:
            checkpoint_id: ID of the checkpoint

        Returns:
            The context dictionary

        Raises:
            ValueError: If checkpoint_id is not found
        """
        if checkpoint_id not in self._checkpoints:
            raise ValueError(f"Invalid checkpoint ID: {checkpoint_id}")

        return self._checkpoints[checkpoint_id].context

    # =========================================================================
    # Approval Request Methods
    # =========================================================================

    def request_approval(
        self,
        title: str,
        description: str,
        context: Optional[Dict[str, Any]] = None,
        timeout_seconds: int = 300,
    ) -> ApprovalRequest:
        """Create a new approval request.

        Args:
            title: Short title for the approval
            description: Detailed description
            context: Additional context data
            timeout_seconds: Timeout for the request

        Returns:
            The created ApprovalRequest
        """
        request_id = f"req_{uuid.uuid4().hex}"
        request = ApprovalRequest(
            id=request_id,
            title=title,
            description=description,
            context=normalize_execution_checkpoint_context(context),
            timeout_seconds=timeout_seconds,
        )
        self._requests[request_id] = request

        # Notify callbacks
        for callback in self._on_approval_request_callbacks:
            callback(request)

        return request

    def get_request(self, request_id: str) -> ApprovalRequest:
        """Get an approval request by ID.

        Args:
            request_id: ID of the request

        Returns:
            The ApprovalRequest

        Raises:
            ValueError: If request_id is not found
        """
        if request_id not in self._requests:
            raise ValueError(f"Invalid request ID: {request_id}")

        return self._requests[request_id]

    def get_pending_requests(self) -> List[ApprovalRequest]:
        """Get all pending approval requests.

        Returns:
            List of pending ApprovalRequest objects
        """
        return [req for req in self._requests.values() if req.status == ApprovalStatus.PENDING]

    def respond_to_request(
        self,
        request_id: str,
        approved: bool,
        response: Optional[str] = None,
        responder: Optional[str] = None,
    ) -> ApprovalRequest:
        """Respond to an approval request.

        Args:
            request_id: ID of the request to respond to
            approved: Whether to approve or reject
            response: Optional response message
            responder: Identity of the responder

        Returns:
            Updated ApprovalRequest

        Raises:
            ValueError: If request_id is not found
        """
        if request_id not in self._requests:
            raise ValueError(f"Invalid request ID: {request_id}")

        request = self._requests[request_id]
        status = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
        approval_state = ApprovalState.APPROVED if approved else ApprovalState.REJECTED

        # Create updated request (dataclass is not mutable by default in this style)
        updated = ApprovalRequest(
            id=request.id,
            title=request.title,
            description=request.description,
            context=normalize_execution_checkpoint_context(
                request.context,
                approval_state=approval_state,
            ),
            timeout_seconds=request.timeout_seconds,
            status=status,
            response=response,
            responder=responder,
            created_at=request.created_at,
        )
        self._requests[request_id] = updated

        return updated

    async def wait_for_approval(
        self,
        request_id: str,
        timeout: Optional[float] = None,
        poll_interval: float = 0.1,
    ) -> ApprovalRequest:
        """Wait for an approval request to be resolved.

        Polls the request status until it is no longer pending or
        the timeout is reached.

        Args:
            request_id: ID of the request to wait for
            timeout: Timeout in seconds (uses request's timeout if None)
            poll_interval: How often to check status

        Returns:
            The resolved ApprovalRequest

        Raises:
            ValueError: If request_id is not found
        """
        if request_id not in self._requests:
            raise ValueError(f"Invalid request ID: {request_id}")

        request = self._requests[request_id]
        effective_timeout = timeout if timeout is not None else request.timeout_seconds
        start_time = time.time()

        while True:
            current = self._requests[request_id]
            if current.status != ApprovalStatus.PENDING:
                return current

            elapsed = time.time() - start_time
            if elapsed >= effective_timeout:
                # Mark as timeout
                timed_out = ApprovalRequest(
                    id=current.id,
                    title=current.title,
                    description=current.description,
                    context=current.context,
                    timeout_seconds=current.timeout_seconds,
                    status=ApprovalStatus.TIMEOUT,
                    response="Request timed out",
                    responder=None,
                    created_at=current.created_at,
                )
                self._requests[request_id] = timed_out
                return timed_out

            await asyncio.sleep(poll_interval)

    async def process_approval(self, request_id: str) -> ApprovalRequest:
        """Process an approval request using the configured handler.

        If no custom handler is configured, uses default approval.

        Args:
            request_id: ID of the request to process

        Returns:
            Updated ApprovalRequest

        Raises:
            ValueError: If request_id is not found
        """
        if request_id not in self._requests:
            raise ValueError(f"Invalid request ID: {request_id}")

        request = self._requests[request_id]

        if self._approval_handler:
            status, response, responder = await self._approval_handler(request)
        else:
            # Default: auto-approve
            status = ApprovalStatus.APPROVED
            response = "Auto-approved (no handler configured)"
            responder = "system"

        updated = ApprovalRequest(
            id=request.id,
            title=request.title,
            description=request.description,
            context=normalize_execution_checkpoint_context(
                request.context,
                approval_state=_approval_status_to_checkpoint_state(status),
            ),
            timeout_seconds=request.timeout_seconds,
            status=status,
            response=response,
            responder=responder,
            created_at=request.created_at,
        )
        self._requests[request_id] = updated

        return updated

    def _default_approval(self, request_id: str) -> ApprovalRequest:
        """Default approval for testing (synchronous).

        Auto-approves the request for testing purposes.

        Args:
            request_id: ID of the request to approve

        Returns:
            Updated ApprovalRequest
        """
        return self.respond_to_request(
            request_id=request_id,
            approved=True,
            response="Auto-approved for testing",
            responder="test-system",
        )

    # =========================================================================
    # Callback Registration
    # =========================================================================

    def on_pause(
        self,
        callback: Callable[[str, Dict[str, Any]], None],
    ) -> None:
        """Register callback for pause events.

        Args:
            callback: Function called with (checkpoint_id, context) on pause
        """
        self._on_pause_callbacks.append(callback)

    def on_resume(self, callback: Callable[[str], None]) -> None:
        """Register callback for resume events.

        Args:
            callback: Function called with checkpoint_id on resume
        """
        self._on_resume_callbacks.append(callback)

    def on_approval_request(
        self,
        callback: Callable[[ApprovalRequest], None],
    ) -> None:
        """Register callback for new approval requests.

        Args:
            callback: Function called with ApprovalRequest on creation
        """
        self._on_approval_request_callbacks.append(callback)


def _approval_status_to_checkpoint_state(
    status: ApprovalStatus,
) -> Optional[ApprovalState]:
    if status == ApprovalStatus.APPROVED:
        return ApprovalState.APPROVED
    if status == ApprovalStatus.REJECTED:
        return ApprovalState.REJECTED
    return None


__all__ = [
    "ApprovalStatus",
    "ApprovalRequest",
    "ApprovalHandler",
    "HITLCheckpoint",
    "HITLController",
]
