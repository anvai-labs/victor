# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Human-in-the-Loop (HITL) workflow system.

Tests cover:
- HITLNode creation and serialization
- HITLRequest/Response dataclasses
- HITLExecutor with mock handlers
- WorkflowBuilder HITL methods
- SafetyChecker HITL integration
"""

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from victor.workflows.hitl import (
    HITLNodeType,
    HITLFallback,
    HITLStatus,
    HITLRequest,
    HITLResponse,
    HITLNode,
    HITLExecutor,
    DefaultHITLHandler,
)
from victor.workflows.definition import WorkflowNodeType, WorkflowBuilder


class TestHITLRequest:
    """Tests for HITLRequest dataclass."""

    def test_create_request(self):
        """Test creating a HITL request."""
        request = HITLRequest(
            request_id="req_123",
            node_id="approval_node",
            hitl_type=HITLNodeType.APPROVAL,
            prompt="Proceed with changes?",
            context={"files": ["main.py"]},
            timeout=60.0,
        )

        assert request.request_id == "req_123"
        assert request.hitl_type == HITLNodeType.APPROVAL
        assert request.prompt == "Proceed with changes?"
        assert request.timeout == 60.0
        assert request.fallback == HITLFallback.ABORT

    def test_request_serialization(self):
        """Test request to_dict serialization."""
        request = HITLRequest(
            request_id="req_456",
            node_id="choice_node",
            hitl_type=HITLNodeType.CHOICE,
            prompt="Select option",
            choices=["A", "B", "C"],
            default_value="A",
        )

        data = request.to_dict()

        assert data["request_id"] == "req_456"
        assert data["hitl_type"] == "choice"
        assert data["choices"] == ["A", "B", "C"]
        assert data["default_value"] == "A"


class TestHITLResponse:
    """Tests for HITLResponse dataclass."""

    def test_create_approved_response(self):
        """Test creating an approved response."""
        response = HITLResponse(
            request_id="req_123",
            status=HITLStatus.APPROVED,
            approved=True,
        )

        assert response.approved is True
        assert response.status == HITLStatus.APPROVED

    def test_create_rejected_response(self):
        """Test creating a rejected response."""
        response = HITLResponse(
            request_id="req_123",
            status=HITLStatus.REJECTED,
            approved=False,
            reason="Too risky",
        )

        assert response.approved is False
        assert response.reason == "Too risky"

    def test_response_serialization(self):
        """Test response to_dict serialization."""
        response = HITLResponse(
            request_id="req_789",
            status=HITLStatus.MODIFIED,
            approved=True,
            modifications={"path": "/new/path"},
        )

        data = response.to_dict()

        assert data["status"] == "modified"
        assert data["modifications"] == {"path": "/new/path"}


class TestHITLNode:
    """Tests for HITLNode workflow node."""

    def test_create_approval_node(self):
        """Test creating an approval node."""
        node = HITLNode(
            id="approve_delete",
            name="Approve Delete",
            hitl_type=HITLNodeType.APPROVAL,
            prompt="Confirm file deletion?",
            context_keys=["file_path", "reason"],
            timeout=120.0,
        )

        assert node.id == "approve_delete"
        assert node.hitl_type == HITLNodeType.APPROVAL
        assert node.node_type == WorkflowNodeType.HITL
        assert node.timeout == 120.0

    def test_create_choice_node(self):
        """Test creating a choice node."""
        node = HITLNode(
            id="select_strategy",
            name="Select Strategy",
            hitl_type=HITLNodeType.CHOICE,
            prompt="Select approach:",
            choices=["conservative", "aggressive", "balanced"],
            default_value="balanced",
        )

        assert node.choices == ["conservative", "aggressive", "balanced"]
        assert node.default_value == "balanced"

    def test_node_serialization(self):
        """Test node to_dict serialization."""
        node = HITLNode(
            id="review_code",
            name="Review Code",
            hitl_type=HITLNodeType.REVIEW,
            prompt="Review the changes",
            context_keys=["diff", "file_list"],
        )

        data = node.to_dict()

        assert data["id"] == "review_code"
        assert data["hitl_type"] == "review"
        assert data["type"] == "hitl"
        assert data["_hitl_node"] is True

    def test_create_request_from_context(self):
        """Test creating a request from workflow context."""
        node = HITLNode(
            id="confirm_deploy",
            name="Confirm Deploy",
            hitl_type=HITLNodeType.CONFIRMATION,
            prompt="Deploy to production?",
            context_keys=["environment", "version"],
        )

        context = {
            "environment": "production",
            "version": "1.2.3",
            "extra_data": "not included",
        }

        request = node.create_request(context)

        assert request.node_id == "confirm_deploy"
        assert request.hitl_type == HITLNodeType.CONFIRMATION
        assert request.context == {"environment": "production", "version": "1.2.3"}
        assert "extra_data" not in request.context

    def test_validate_response(self):
        """Test response validation with custom validator."""
        node = HITLNode(
            id="input_number",
            name="Input Number",
            hitl_type=HITLNodeType.INPUT,
            prompt="Enter a number:",
            validator=lambda x: isinstance(x, int) and x > 0,
        )

        valid_response = HITLResponse(
            request_id="req_1",
            status=HITLStatus.APPROVED,
            value=42,
        )
        invalid_response = HITLResponse(
            request_id="req_2",
            status=HITLStatus.APPROVED,
            value=-1,
        )

        assert node.validate_response(valid_response) is True
        assert node.validate_response(invalid_response) is False


class TestHITLExecutor:
    """Tests for HITLExecutor."""

    @pytest.fixture
    def mock_handler(self):
        """Create a mock HITL handler."""
        handler = MagicMock()
        handler.request_human_input = AsyncMock()
        return handler

    @pytest.fixture
    def executor(self, mock_handler):
        """Create executor with mock handler."""
        return HITLExecutor(mock_handler)

    @pytest.mark.asyncio
    async def test_execute_approval_approved(self, executor, mock_handler):
        """Test executing approval node that gets approved."""
        mock_handler.request_human_input.return_value = HITLResponse(
            request_id="req_1",
            status=HITLStatus.APPROVED,
            approved=True,
        )

        node = HITLNode(
            id="test_approval",
            name="Test Approval",
            hitl_type=HITLNodeType.APPROVAL,
            prompt="Approve?",
            timeout=1.0,
        )

        response = await executor.execute_hitl_node(node, {})

        assert response.approved is True
        assert response.status == HITLStatus.APPROVED
        mock_handler.request_human_input.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_approval_rejected(self, executor, mock_handler):
        """Test executing approval node that gets rejected."""
        mock_handler.request_human_input.return_value = HITLResponse(
            request_id="req_1",
            status=HITLStatus.REJECTED,
            approved=False,
            reason="Not now",
        )

        node = HITLNode(
            id="test_approval",
            name="Test Approval",
            hitl_type=HITLNodeType.APPROVAL,
            prompt="Approve?",
        )

        response = await executor.execute_hitl_node(node, {})

        assert response.approved is False
        assert response.reason == "Not now"

    @pytest.mark.asyncio
    async def test_execute_timeout_abort(self, executor, mock_handler):
        """Test timeout with ABORT fallback."""
        # Handler never returns (simulated by raising TimeoutError)
        mock_handler.request_human_input.side_effect = asyncio.TimeoutError()

        node = HITLNode(
            id="test_timeout",
            name="Test Timeout",
            hitl_type=HITLNodeType.APPROVAL,
            prompt="Approve?",
            timeout=0.01,
            fallback=HITLFallback.ABORT,
        )

        response = await executor.execute_hitl_node(node, {})

        assert response.approved is False
        assert response.status == HITLStatus.TIMEOUT

    @pytest.mark.asyncio
    async def test_execute_timeout_continue(self, executor, mock_handler):
        """Test timeout with CONTINUE fallback."""
        mock_handler.request_human_input.side_effect = asyncio.TimeoutError()

        node = HITLNode(
            id="test_timeout",
            name="Test Timeout Continue",
            hitl_type=HITLNodeType.APPROVAL,
            prompt="Approve?",
            timeout=0.01,
            fallback=HITLFallback.CONTINUE,
            default_value="default_action",
        )

        response = await executor.execute_hitl_node(node, {})

        assert response.approved is True
        assert response.status == HITLStatus.TIMEOUT
        assert response.value == "default_action"

    @pytest.mark.asyncio
    async def test_execute_with_validation_failure(self, executor, mock_handler):
        """Test response that fails validation."""
        mock_handler.request_human_input.return_value = HITLResponse(
            request_id="req_1",
            status=HITLStatus.APPROVED,
            approved=True,
            value=-5,  # Will fail validation
        )

        node = HITLNode(
            id="test_validate",
            name="Test Validate",
            hitl_type=HITLNodeType.INPUT,
            prompt="Enter positive number:",
            validator=lambda x: x > 0,
        )

        response = await executor.execute_hitl_node(node, {})

        assert response.approved is False
        assert response.status == HITLStatus.REJECTED
        assert "validation failed" in response.reason.lower()


class TestWorkflowBuilderHITL:
    """Tests for WorkflowBuilder HITL methods."""

    def test_add_hitl_approval(self):
        """Test adding HITL approval node."""
        workflow = (
            WorkflowBuilder("test_workflow")
            .add_agent("analyze", "researcher", "Analyze code")
            .add_hitl_approval(
                "approve_changes",
                prompt="Proceed with changes?",
                context_keys=["changes"],
                timeout=120.0,
            )
            .add_agent("implement", "executor", "Make changes")
            .build()
        )

        assert len(workflow.nodes) == 3
        hitl_node = workflow.nodes["approve_changes"]
        assert hitl_node.hitl_type == HITLNodeType.APPROVAL
        assert hitl_node.timeout == 120.0

    def test_add_hitl_choice(self):
        """Test adding HITL choice node."""
        workflow = (
            WorkflowBuilder("choice_workflow")
            .add_hitl_choice(
                "select_approach",
                prompt="Choose approach:",
                choices=["fast", "safe", "balanced"],
                default_value="balanced",
            )
            .build()
        )

        hitl_node = workflow.nodes["select_approach"]
        assert hitl_node.hitl_type == HITLNodeType.CHOICE
        assert hitl_node.choices == ["fast", "safe", "balanced"]
        assert hitl_node.default_value == "balanced"

    def test_add_hitl_review(self):
        """Test adding HITL review node."""
        workflow = (
            WorkflowBuilder("review_workflow")
            .add_hitl_review(
                "code_review",
                prompt="Review the code changes",
                context_keys=["diff", "files"],
            )
            .build()
        )

        hitl_node = workflow.nodes["code_review"]
        assert hitl_node.hitl_type == HITLNodeType.REVIEW

    def test_add_hitl_confirmation(self):
        """Test adding HITL confirmation node."""
        workflow = (
            WorkflowBuilder("confirm_workflow")
            .add_hitl_confirmation(
                "confirm_deploy",
                prompt="Deploy to production?",
                timeout=60.0,
                fallback=HITLFallback.ABORT,
            )
            .build()
        )

        hitl_node = workflow.nodes["confirm_deploy"]
        assert hitl_node.hitl_type == HITLNodeType.CONFIRMATION
        assert hitl_node.fallback == HITLFallback.ABORT


class TestSafetyCheckerHITLIntegration:
    """Tests for SafetyChecker HITL integration."""

    def test_create_hitl_confirmation_callback(self):
        """Test creating HITL confirmation callback."""
        from victor.agent.safety import create_hitl_confirmation_callback

        callback = create_hitl_confirmation_callback(timeout=30.0, fallback="abort")

        assert callable(callback)
        # Callback should be async
        import inspect

        assert inspect.iscoroutinefunction(callback)

    @pytest.mark.asyncio
    async def test_hitl_callback_approved(self):
        """Test HITL callback when approved."""
        from victor.agent.safety import (
            create_hitl_confirmation_callback,
            ConfirmationRequest,
            OperationalRiskLevel,
        )

        # Create mock handler that approves
        mock_handler = MagicMock()
        mock_handler.request_human_input = AsyncMock(
            return_value=HITLResponse(
                request_id="test",
                status=HITLStatus.APPROVED,
                approved=True,
            )
        )

        callback = create_hitl_confirmation_callback(
            hitl_handler=mock_handler,
            timeout=1.0,
        )

        request = ConfirmationRequest(
            tool_name="execute_bash",
            risk_level=OperationalRiskLevel.HIGH,
            description="rm -rf dangerous",
            details=["Deletes files"],
            arguments={"command": "rm -rf temp"},
        )

        result = await callback(request)

        assert result is True
        mock_handler.request_human_input.assert_called_once()

    @pytest.mark.asyncio
    async def test_hitl_callback_rejected(self):
        """Test HITL callback when rejected."""
        from victor.agent.safety import (
            create_hitl_confirmation_callback,
            ConfirmationRequest,
            OperationalRiskLevel,
        )

        mock_handler = MagicMock()
        mock_handler.request_human_input = AsyncMock(
            return_value=HITLResponse(
                request_id="test",
                status=HITLStatus.REJECTED,
                approved=False,
                reason="Too dangerous",
            )
        )

        callback = create_hitl_confirmation_callback(
            hitl_handler=mock_handler,
            timeout=1.0,
        )

        request = ConfirmationRequest(
            tool_name="execute_bash",
            risk_level=OperationalRiskLevel.CRITICAL,
            description="Format disk",
            details=["Critical operation"],
            arguments={"command": "mkfs"},
        )

        result = await callback(request)

        assert result is False
