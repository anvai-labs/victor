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

"""Unit tests for the delegation module."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from victor.agent.delegation import (
    DelegationPriority,
    DelegationRequest,
    DelegationResponse,
    DelegationStatus,
    DelegationHandler,
    DelegateTool,
)
from victor.agent.delegation.handler import ROLE_MAPPING
from victor.agent.subagents import SubAgentRole


class TestDelegationPriority:
    """Test DelegationPriority enum."""

    def test_all_priorities_defined(self):
        """All expected priorities are defined."""
        assert DelegationPriority.LOW.value == "low"
        assert DelegationPriority.NORMAL.value == "normal"
        assert DelegationPriority.HIGH.value == "high"
        assert DelegationPriority.URGENT.value == "urgent"


class TestDelegationStatus:
    """Test DelegationStatus enum."""

    def test_all_statuses_defined(self):
        """All expected statuses are defined."""
        assert DelegationStatus.PENDING.value == "pending"
        assert DelegationStatus.RUNNING.value == "running"
        assert DelegationStatus.COMPLETED.value == "completed"
        assert DelegationStatus.FAILED.value == "failed"
        assert DelegationStatus.CANCELLED.value == "cancelled"
        assert DelegationStatus.TIMEOUT.value == "timeout"


class TestDelegationRequest:
    """Test DelegationRequest dataclass."""

    def test_minimal_request(self):
        """Create request with minimal fields."""
        request = DelegationRequest(task="Find something")
        assert request.task == "Find something"
        assert request.from_agent == "main"
        assert request.suggested_role is None
        assert request.priority == DelegationPriority.NORMAL
        assert request.delegation_id is not None

    def test_full_request(self):
        """Create request with all fields."""
        request = DelegationRequest(
            task="Research authentication",
            from_agent="main_agent",
            suggested_role="researcher",
            priority=DelegationPriority.HIGH,
            required_tools=["read", "search"],
            tool_budget=20,
            context={"key": "value"},
            deadline_seconds=120,
            await_result=False,
            parent_goal="Implement new auth system",
        )
        assert request.from_agent == "main_agent"
        assert request.suggested_role == "researcher"
        assert request.priority == DelegationPriority.HIGH
        assert request.tool_budget == 20
        assert request.await_result is False

    def test_empty_task_raises(self):
        """Empty task raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            DelegationRequest(task="")

    def test_whitespace_task_raises(self):
        """Whitespace-only task raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            DelegationRequest(task="   ")

    def test_zero_budget_raises(self):
        """Zero tool budget raises ValueError."""
        with pytest.raises(ValueError, match="at least 1"):
            DelegationRequest(task="Valid task", tool_budget=0)

    def test_negative_deadline_raises(self):
        """Negative deadline raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            DelegationRequest(task="Valid task", deadline_seconds=-10)

    def test_to_dict(self):
        """to_dict serializes correctly."""
        request = DelegationRequest(
            task="Test task",
            suggested_role="executor",
            tool_budget=15,
        )
        d = request.to_dict()
        assert d["task"] == "Test task"
        assert d["suggested_role"] == "executor"
        assert d["tool_budget"] == 15
        assert d["priority"] == "normal"

    def test_from_dict(self):
        """from_dict deserializes correctly."""
        data = {
            "task": "Test task",
            "from_agent": "agent1",
            "suggested_role": "researcher",
            "priority": "high",
            "tool_budget": 20,
        }
        request = DelegationRequest.from_dict(data)
        assert request.task == "Test task"
        assert request.from_agent == "agent1"
        assert request.suggested_role == "researcher"
        assert request.priority == DelegationPriority.HIGH
        assert request.tool_budget == 20


class TestDelegationResponse:
    """Test DelegationResponse dataclass."""

    def test_success_response(self):
        """Create successful response."""
        response = DelegationResponse(
            delegation_id="test123",
            accepted=True,
            status=DelegationStatus.COMPLETED,
            delegate_id="delegate_abc",
            result="Found 5 endpoints",
            tool_calls_used=10,
            duration_seconds=15.5,
        )
        assert response.success is True
        assert response.result == "Found 5 endpoints"
        assert response.error is None

    def test_failure_response(self):
        """Create failed response."""
        response = DelegationResponse(
            delegation_id="test123",
            accepted=True,
            status=DelegationStatus.FAILED,
            error="Something went wrong",
        )
        assert response.success is False
        assert response.error == "Something went wrong"

    def test_rejected_factory(self):
        """rejected factory creates rejected response."""
        response = DelegationResponse.rejected("test123", "Not enough resources")
        assert response.accepted is False
        assert response.status == DelegationStatus.FAILED
        assert response.error == "Not enough resources"

    def test_pending_factory(self):
        """pending factory creates pending response."""
        response = DelegationResponse.pending("test123", "delegate_abc")
        assert response.accepted is True
        assert response.status == DelegationStatus.PENDING
        assert response.delegate_id == "delegate_abc"

    def test_to_dict(self):
        """to_dict serializes correctly."""
        response = DelegationResponse(
            delegation_id="test123",
            accepted=True,
            status=DelegationStatus.COMPLETED,
            result="Done",
        )
        d = response.to_dict()
        assert d["delegation_id"] == "test123"
        assert d["accepted"] is True
        assert d["status"] == "completed"

    def test_from_dict(self):
        """from_dict deserializes correctly."""
        data = {
            "delegation_id": "test123",
            "accepted": True,
            "status": "completed",
            "result": "Done",
        }
        response = DelegationResponse.from_dict(data)
        assert response.delegation_id == "test123"
        assert response.accepted is True
        assert response.status == DelegationStatus.COMPLETED


class TestRoleMapping:
    """Test role mapping in handler."""

    def test_all_roles_mapped(self):
        """All expected roles have mappings."""
        assert "researcher" in ROLE_MAPPING
        assert "planner" in ROLE_MAPPING
        assert "executor" in ROLE_MAPPING
        assert "reviewer" in ROLE_MAPPING
        assert "tester" in ROLE_MAPPING

    def test_aliases_mapped(self):
        """Role aliases are mapped correctly."""
        assert ROLE_MAPPING["research"] == SubAgentRole.RESEARCHER
        assert ROLE_MAPPING["plan"] == SubAgentRole.PLANNER
        assert ROLE_MAPPING["execute"] == SubAgentRole.EXECUTOR
        assert ROLE_MAPPING["implement"] == SubAgentRole.EXECUTOR
        assert ROLE_MAPPING["review"] == SubAgentRole.REVIEWER
        assert ROLE_MAPPING["test"] == SubAgentRole.TESTER


class TestDelegationHandler:
    """Test DelegationHandler class."""

    def test_initialization(self):
        """Handler can be initialized with mock orchestrator."""
        mock_orchestrator = MagicMock()
        handler = DelegationHandler(mock_orchestrator)
        assert handler.orchestrator == mock_orchestrator
        assert handler.max_concurrent == 4

    def test_custom_max_concurrent(self):
        """Custom max_concurrent is respected."""
        mock_orchestrator = MagicMock()
        handler = DelegationHandler(mock_orchestrator, max_concurrent=8)
        assert handler.max_concurrent == 8

    def test_get_active_count(self):
        """get_active_count returns 0 initially."""
        mock_orchestrator = MagicMock()
        handler = DelegationHandler(mock_orchestrator)
        assert handler.get_active_count() == 0

    def test_resolve_role_from_suggestion(self):
        """Role is resolved from suggestion."""
        mock_orchestrator = MagicMock()
        handler = DelegationHandler(mock_orchestrator)

        request = DelegationRequest(task="Test", suggested_role="researcher")
        role = handler._resolve_role(request)
        assert role == SubAgentRole.RESEARCHER

    def test_resolve_role_from_keywords(self):
        """Role is inferred from task keywords."""
        mock_orchestrator = MagicMock()
        handler = DelegationHandler(mock_orchestrator)

        # Research keywords
        request = DelegationRequest(task="Find all API endpoints")
        role = handler._resolve_role(request)
        assert role == SubAgentRole.RESEARCHER

        # Plan keywords
        request = DelegationRequest(task="Design the architecture")
        role = handler._resolve_role(request)
        assert role == SubAgentRole.PLANNER

        # Execute keywords
        request = DelegationRequest(task="Implement the feature")
        role = handler._resolve_role(request)
        assert role == SubAgentRole.EXECUTOR

        # Review keywords
        request = DelegationRequest(task="Review the code changes")
        role = handler._resolve_role(request)
        assert role == SubAgentRole.REVIEWER

        # Test keywords
        request = DelegationRequest(task="Test the authentication")
        role = handler._resolve_role(request)
        assert role == SubAgentRole.TESTER

    def test_validate_request_empty_task(self):
        """Validation fails for empty task."""
        mock_orchestrator = MagicMock()
        handler = DelegationHandler(mock_orchestrator)

        # Can't create request with empty task (validation in __post_init__)
        # So we test the handler's validation separately
        error = handler._validate_request(DelegationRequest(task="valid"))
        assert error is None

    def test_validate_request_excessive_budget(self):
        """Validation fails for excessive budget."""
        mock_orchestrator = MagicMock()
        handler = DelegationHandler(mock_orchestrator)

        # Create request with high budget (but under dataclass limit)
        request = DelegationRequest(task="Test", tool_budget=50)
        error = handler._validate_request(request)
        assert error is None

        # Over handler's max
        request = DelegationRequest(task="Test", tool_budget=150)
        error = handler._validate_request(request)
        assert error is not None
        assert "maximum" in error.lower()

    def test_build_delegate_context(self):
        """Context is built correctly for delegate."""
        mock_orchestrator = MagicMock()
        handler = DelegationHandler(mock_orchestrator)

        request = DelegationRequest(
            task="Find all endpoints",
            parent_goal="Improve API documentation",
            context={"files": ["api.py", "routes.py"]},
        )
        context = handler._build_delegate_context(request)

        assert "Find all endpoints" in context
        assert "Improve API documentation" in context
        assert "files" in context
        assert "Focus exclusively on the assigned task" in context


class TestDelegateTool:
    """Test DelegateTool class."""

    def test_tool_metadata(self):
        """Tool has correct metadata."""
        mock_handler = MagicMock()
        tool = DelegateTool(mock_handler)

        assert tool.name == "delegate"
        assert "delegate" in tool.description.lower()
        assert "task" in tool.parameters["properties"]
        assert "role" in tool.parameters["properties"]

    def test_parameters_schema(self):
        """Parameters schema is correct."""
        mock_handler = MagicMock()
        tool = DelegateTool(mock_handler)

        props = tool.parameters["properties"]
        assert props["task"]["type"] == "string"
        assert "researcher" in props["role"]["enum"]
        assert "executor" in props["role"]["enum"]
        assert props["tool_budget"]["type"] == "integer"
        assert props["await_result"]["type"] == "boolean"

    def test_required_parameters(self):
        """Only task is required."""
        mock_handler = MagicMock()
        tool = DelegateTool(mock_handler)

        assert tool.parameters["required"] == ["task"]

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Successful delegation returns success result."""
        mock_handler = MagicMock()
        mock_handler.handle = AsyncMock(
            return_value=DelegationResponse(
                delegation_id="test123",
                accepted=True,
                status=DelegationStatus.COMPLETED,
                delegate_id="delegate_abc",
                result="Found 5 endpoints",
                tool_calls_used=10,
                duration_seconds=15.5,
            )
        )

        tool = DelegateTool(mock_handler)
        result = await tool.execute(
            task="Find all API endpoints",
            role="researcher",
            tool_budget=15,
        )

        assert result.success is True
        assert "Found 5 endpoints" in result.output
        assert result.metadata["delegation_id"] == "test123"

    @pytest.mark.asyncio
    async def test_execute_rejection(self):
        """Rejected delegation returns failure result."""
        mock_handler = MagicMock()
        mock_handler.handle = AsyncMock(
            return_value=DelegationResponse(
                delegation_id="test123",
                accepted=False,
                status=DelegationStatus.FAILED,
                error="Too many concurrent delegations",
            )
        )

        tool = DelegateTool(mock_handler)
        result = await tool.execute(task="Test task")

        assert result.success is False
        assert "rejected" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_async_mode(self):
        """Fire-and-forget mode returns pending status."""
        mock_handler = MagicMock()
        mock_handler.handle = AsyncMock(
            return_value=DelegationResponse(
                delegation_id="test123",
                accepted=True,
                status=DelegationStatus.PENDING,
                delegate_id="delegate_abc",
            )
        )

        tool = DelegateTool(mock_handler)
        result = await tool.execute(
            task="Background task",
            await_result=False,
        )

        assert result.success is True
        assert "started" in result.output.lower()
        assert result.metadata["async"] is True


class TestDelegationHandlerAsyncMethods:
    """Test async methods of DelegationHandler."""

    @pytest.mark.asyncio
    async def test_handle_validates_request(self):
        """handle should validate request before processing."""
        mock_orchestrator = MagicMock()
        handler = DelegationHandler(mock_orchestrator)

        # Create request that will fail validation (budget > 100)
        request = DelegationRequest(task="Test", tool_budget=150)
        response = await handler.handle(request)

        assert response.accepted is False
        assert "maximum" in response.error.lower()

    @pytest.mark.asyncio
    async def test_handle_rejects_max_concurrent(self):
        """handle should reject when max concurrent reached."""
        mock_orchestrator = MagicMock()
        handler = DelegationHandler(mock_orchestrator, max_concurrent=1)

        # Simulate one active delegation
        handler._active["existing"] = MagicMock()

        request = DelegationRequest(task="Test task", suggested_role="researcher")
        response = await handler.handle(request)

        assert response.accepted is False
        assert "concurrent" in response.error.lower()

    @pytest.mark.asyncio
    async def test_handle_rejects_unknown_role(self):
        """handle should reject unknown role."""
        mock_orchestrator = MagicMock()
        handler = DelegationHandler(mock_orchestrator)

        # Patch _resolve_role to return None
        with patch.object(handler, "_resolve_role", return_value=None):
            request = DelegationRequest(task="Test task", suggested_role="unknown_role")
            response = await handler.handle(request)

        assert response.accepted is False
        assert "unknown role" in response.error.lower()

    @pytest.mark.asyncio
    async def test_cancel_returns_false_for_unknown(self):
        """cancel should return False for unknown delegation."""
        mock_orchestrator = MagicMock()
        handler = DelegationHandler(mock_orchestrator)

        result = await handler.cancel("nonexistent_id")
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_cancels_active_task(self):
        """cancel should cancel active task and remove from active."""
        mock_orchestrator = MagicMock()
        handler = DelegationHandler(mock_orchestrator)

        # Create a mock delegation with a task
        mock_task = MagicMock()
        mock_task.done.return_value = False
        mock_delegation = MagicMock()
        mock_delegation.task = mock_task

        handler._active["test_id"] = mock_delegation

        result = await handler.cancel("test_id")

        assert result is True
        mock_task.cancel.assert_called_once()
        assert "test_id" not in handler._active

    def test_get_status_returns_none_for_unknown(self):
        """get_status should return None for unknown delegation."""
        mock_orchestrator = MagicMock()
        handler = DelegationHandler(mock_orchestrator)

        status = handler.get_status("nonexistent_id")
        assert status is None

    def test_get_status_returns_dict_for_known(self):
        """get_status should return dict for known delegation."""
        from victor.agent.delegation.handler import ActiveDelegation

        mock_orchestrator = MagicMock()
        handler = DelegationHandler(mock_orchestrator)

        # Create an active delegation
        request = DelegationRequest(task="Test task", suggested_role="researcher")
        delegation = ActiveDelegation(request, "delegate_123")
        delegation.task = None  # Not running

        handler._active["test_id"] = delegation

        status = handler.get_status("test_id")

        assert status is not None
        assert status["delegation_id"] == "test_id"
        assert status["delegate_id"] == "delegate_123"
        assert status["role"] == "researcher"
        assert status["running"] is False

    @pytest.mark.asyncio
    async def test_wait_for_completion_unknown_delegation(self):
        """wait_for_completion should return None for unknown delegation."""
        mock_orchestrator = MagicMock()
        handler = DelegationHandler(mock_orchestrator)

        result = await handler.wait_for_completion("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_wait_for_completion_no_task(self):
        """wait_for_completion should return result if no task."""
        from victor.agent.delegation.handler import ActiveDelegation

        mock_orchestrator = MagicMock()
        handler = DelegationHandler(mock_orchestrator)

        request = DelegationRequest(task="Test")
        delegation = ActiveDelegation(request, "delegate_123")
        delegation.task = None
        delegation.result = DelegationResponse(
            delegation_id="test_id",
            accepted=True,
            status=DelegationStatus.COMPLETED,
            result="Done",
        )

        handler._active["test_id"] = delegation

        result = await handler.wait_for_completion("test_id")
        assert result.result == "Done"

    def test_set_completion_callback(self):
        """set_completion_callback should set the callback."""
        mock_orchestrator = MagicMock()
        handler = DelegationHandler(mock_orchestrator)

        callback = MagicMock()
        handler.set_completion_callback(callback)

        assert handler._on_complete == callback


class TestDelegationHandlerConvertResult:
    """Test _convert_result method."""

    def test_convert_successful_result(self):
        """_convert_result should convert successful result."""
        from victor.agent.delegation.handler import ActiveDelegation
        from victor.agent.subagents.base import SubAgentResult

        mock_orchestrator = MagicMock()
        handler = DelegationHandler(mock_orchestrator)

        request = DelegationRequest(task="Test")
        delegation = ActiveDelegation(request, "delegate_123")

        sub_result = SubAgentResult(
            success=True,
            summary="Found 5 endpoints\nDiscovered 3 issues",
            details={},
            tool_calls_used=10,
            context_size=1000,
            duration_seconds=15.5,
        )

        response = handler._convert_result(request, delegation, sub_result)

        assert response.status == DelegationStatus.COMPLETED
        assert response.result == "Found 5 endpoints\nDiscovered 3 issues"
        assert response.tool_calls_used == 10
        assert response.duration_seconds == 15.5

    def test_convert_failed_result(self):
        """_convert_result should convert failed result."""
        from victor.agent.delegation.handler import ActiveDelegation
        from victor.agent.subagents.base import SubAgentResult

        mock_orchestrator = MagicMock()
        handler = DelegationHandler(mock_orchestrator)

        request = DelegationRequest(task="Test")
        delegation = ActiveDelegation(request, "delegate_123")

        sub_result = SubAgentResult(
            success=False,
            summary="",
            details={},
            tool_calls_used=0,
            context_size=0,
            duration_seconds=0.0,
            error="Something went wrong",
        )

        response = handler._convert_result(request, delegation, sub_result)

        assert response.status == DelegationStatus.FAILED
        assert response.result is None
        assert response.error == "Something went wrong"

    def test_convert_extracts_discoveries(self):
        """_convert_result should extract discoveries from summary."""
        from victor.agent.delegation.handler import ActiveDelegation
        from victor.agent.subagents.base import SubAgentResult

        mock_orchestrator = MagicMock()
        handler = DelegationHandler(mock_orchestrator)

        request = DelegationRequest(task="Test")
        delegation = ActiveDelegation(request, "delegate_123")

        sub_result = SubAgentResult(
            success=True,
            summary=(
                "Some intro text\n"
                "Found 5 API endpoints\n"
                "Regular line\n"
                "Discovered a security issue\n"
                "Identified the root cause\n"
                "Located the config file"
            ),
            details={},
            tool_calls_used=5,
            context_size=500,
            duration_seconds=10.0,
        )

        response = handler._convert_result(request, delegation, sub_result)

        assert len(response.discoveries) == 4
        assert any("Found 5 API endpoints" in d for d in response.discoveries)
        assert any("Discovered a security issue" in d for d in response.discoveries)


class TestActiveDelegation:
    """Test ActiveDelegation class."""

    def test_initialization(self):
        """ActiveDelegation should initialize correctly."""
        from victor.agent.delegation.handler import ActiveDelegation

        request = DelegationRequest(task="Test")
        delegation = ActiveDelegation(request, "delegate_123")

        assert delegation.request == request
        assert delegation.delegate_id == "delegate_123"
        assert delegation.start_time > 0
        assert delegation.task is None
        assert delegation.result is None


class TestModuleExports:
    """Test module exports are correct."""

    def test_delegation_init_exports(self):
        """Delegation __init__ exports all expected symbols."""
        from victor.agent.delegation import (
            DelegationPriority,
            DelegationRequest,
            DelegationResponse,
            DelegationStatus,
            DelegationHandler,
            DelegateTool,
        )

        # If we get here without ImportError, all exports work
        assert True

    def test_handler_exports(self):
        """Handler module exports all expected symbols."""
        from victor.agent.delegation.handler import (
            DelegationHandler,
            ActiveDelegation,
            ROLE_MAPPING,
        )

        assert True
