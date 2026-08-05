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

"""Delegation protocol data structures.

This module defines the core data structures for the delegation protocol:
- DelegationRequest: What the delegating agent wants done
- DelegationResponse: Result from the delegated task
- DelegationPriority: Urgency levels for delegation

These are immutable dataclasses designed for serialization and transport
across process boundaries (for background delegation).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class DelegationPriority(Enum):
    """Priority level for delegated tasks.

    Higher priority tasks may be executed sooner or given more resources.

    - LOW: Background task, can wait
    - NORMAL: Standard priority (default)
    - HIGH: Should be executed promptly
    - URGENT: Critical task, execute immediately
    """

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class DelegationStatus(Enum):
    """Status of a delegation request.

    - PENDING: Request received, not yet started
    - RUNNING: Delegate is executing
    - COMPLETED: Successfully completed
    - FAILED: Execution failed
    - CANCELLED: Cancelled before completion
    - TIMEOUT: Exceeded deadline
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class DelegationRequest:
    """Request from one agent to delegate work to another.

    Encapsulates everything needed to spawn a delegate agent and
    execute a task on behalf of the requesting agent.

    Attributes:
        task: Clear description of what the delegate should accomplish
        from_agent: ID of the agent making the request
        suggested_role: Preferred role for the delegate (researcher, planner, etc.)
        priority: Urgency level
        required_tools: Specific tools the delegate needs
        tool_budget: Maximum tool calls for the delegate
        context: Additional context to pass to the delegate
        deadline_seconds: Maximum time allowed for execution
        await_result: Whether to wait for completion
        parent_goal: Optional context about the parent agent's goal
        delegation_id: Unique identifier for this delegation

    Example:
        request = DelegationRequest(
            task="Find all files that import the User model",
            from_agent="main_agent",
            suggested_role="researcher",
            tool_budget=10,
        )
    """

    task: str
    from_agent: str = "main"
    suggested_role: Optional[str] = None
    priority: DelegationPriority = DelegationPriority.NORMAL
    required_tools: Optional[List[str]] = None
    tool_budget: int = 10
    context: Optional[Dict[str, Any]] = None
    deadline_seconds: Optional[float] = None
    await_result: bool = True
    parent_goal: Optional[str] = None
    delegation_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def __post_init__(self):
        """Validate request parameters."""
        if not self.task or not self.task.strip():
            raise ValueError("Delegation task cannot be empty")
        if self.tool_budget < 1:
            raise ValueError("Tool budget must be at least 1")
        if self.deadline_seconds is not None and self.deadline_seconds <= 0:
            raise ValueError("Deadline must be positive")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "delegation_id": self.delegation_id,
            "task": self.task,
            "from_agent": self.from_agent,
            "suggested_role": self.suggested_role,
            "priority": self.priority.value,
            "required_tools": self.required_tools,
            "tool_budget": self.tool_budget,
            "context": self.context,
            "deadline_seconds": self.deadline_seconds,
            "await_result": self.await_result,
            "parent_goal": self.parent_goal,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DelegationRequest":
        """Create from dictionary."""
        priority_str = data.get("priority", "normal")
        priority = DelegationPriority(priority_str)
        return cls(
            task=data["task"],
            from_agent=data.get("from_agent", "main"),
            suggested_role=data.get("suggested_role"),
            priority=priority,
            required_tools=data.get("required_tools"),
            tool_budget=data.get("tool_budget", 10),
            context=data.get("context"),
            deadline_seconds=data.get("deadline_seconds"),
            await_result=data.get("await_result", True),
            parent_goal=data.get("parent_goal"),
            delegation_id=data.get("delegation_id", uuid.uuid4().hex[:12]),
        )


@dataclass
class DelegationResponse:
    """Response from a delegation.

    Contains the result (or error) from the delegated task execution,
    along with metrics about the execution.

    Attributes:
        delegation_id: ID of the original request
        accepted: Whether the delegation was accepted
        status: Current status of the delegation
        delegate_id: ID of the spawned delegate agent
        result: Result string from successful execution
        error: Error message if failed
        tool_calls_used: Number of tool calls made
        duration_seconds: Execution time
        discoveries: Key findings from the delegate

    Example:
        if response.accepted and response.status == DelegationStatus.COMPLETED:
            print(response.result)
        else:
            print(f"Delegation failed: {response.error}")
    """

    delegation_id: str
    accepted: bool
    status: DelegationStatus = DelegationStatus.PENDING
    delegate_id: Optional[str] = None
    result: Optional[str] = None
    error: Optional[str] = None
    tool_calls_used: int = 0
    duration_seconds: float = 0.0
    discoveries: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Check if delegation completed successfully."""
        return self.accepted and self.status == DelegationStatus.COMPLETED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "delegation_id": self.delegation_id,
            "accepted": self.accepted,
            "status": self.status.value,
            "delegate_id": self.delegate_id,
            "result": self.result,
            "error": self.error,
            "tool_calls_used": self.tool_calls_used,
            "duration_seconds": self.duration_seconds,
            "discoveries": self.discoveries,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DelegationResponse":
        """Create from dictionary."""
        status_str = data.get("status", "pending")
        status = DelegationStatus(status_str)
        return cls(
            delegation_id=data["delegation_id"],
            accepted=data["accepted"],
            status=status,
            delegate_id=data.get("delegate_id"),
            result=data.get("result"),
            error=data.get("error"),
            tool_calls_used=data.get("tool_calls_used", 0),
            duration_seconds=data.get("duration_seconds", 0.0),
            discoveries=data.get("discoveries", []),
        )

    @classmethod
    def rejected(cls, delegation_id: str, reason: str) -> "DelegationResponse":
        """Create a rejected response."""
        return cls(
            delegation_id=delegation_id,
            accepted=False,
            status=DelegationStatus.FAILED,
            error=reason,
        )

    @classmethod
    def pending(cls, delegation_id: str, delegate_id: str) -> "DelegationResponse":
        """Create a pending response (for fire-and-forget)."""
        return cls(
            delegation_id=delegation_id,
            accepted=True,
            status=DelegationStatus.PENDING,
            delegate_id=delegate_id,
        )


__all__ = [
    "DelegationPriority",
    "DelegationStatus",
    "DelegationRequest",
    "DelegationResponse",
]
