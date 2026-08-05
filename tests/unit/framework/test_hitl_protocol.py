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

"""Tests for victor.framework.hitl module.

These tests verify the Human-in-the-Loop (HITL) protocol that enables
human approval flows and agent interruption/resume capabilities.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

# =============================================================================
# ApprovalStatus Enum Tests
# =============================================================================


class TestApprovalStatus:
    """Tests for ApprovalStatus enum values."""

    def test_pending_status_exists(self):
        """ApprovalStatus should have PENDING value."""
        from victor.framework.hitl import ApprovalStatus

        assert ApprovalStatus.PENDING.value == "pending"

    def test_approved_status_exists(self):
        """ApprovalStatus should have APPROVED value."""
        from victor.framework.hitl import ApprovalStatus

        assert ApprovalStatus.APPROVED.value == "approved"

    def test_rejected_status_exists(self):
        """ApprovalStatus should have REJECTED value."""
        from victor.framework.hitl import ApprovalStatus

        assert ApprovalStatus.REJECTED.value == "rejected"

    def test_timeout_status_exists(self):
        """ApprovalStatus should have TIMEOUT value."""
        from victor.framework.hitl import ApprovalStatus

        assert ApprovalStatus.TIMEOUT.value == "timeout"


# =============================================================================
# ApprovalRequest Dataclass Tests
# =============================================================================


class TestApprovalRequest:
    """Tests for ApprovalRequest dataclass."""

    def test_approval_request_required_fields(self):
        """ApprovalRequest should have required fields."""
        from victor.framework.hitl import ApprovalRequest, ApprovalStatus

        request = ApprovalRequest(
            id="req-123",
            title="Deploy to Production",
            description="Deploy version 1.2.3 to production servers",
        )

        assert request.id == "req-123"
        assert request.title == "Deploy to Production"
        assert request.description == "Deploy version 1.2.3 to production servers"

    def test_approval_request_default_context(self):
        """ApprovalRequest should have empty context by default."""
        from victor.framework.hitl import ApprovalRequest

        request = ApprovalRequest(
            id="req-123",
            title="Test",
            description="Test description",
        )

        assert request.context == {}

    def test_approval_request_with_context(self):
        """ApprovalRequest should store context dict."""
        from victor.framework.hitl import ApprovalRequest

        context = {"files": ["a.py", "b.py"], "changes": 42}
        request = ApprovalRequest(
            id="req-123",
            title="Test",
            description="Test description",
            context=context,
        )

        assert request.context == context
        assert request.context["files"] == ["a.py", "b.py"]

    def test_approval_request_default_timeout(self):
        """ApprovalRequest should have 300 second timeout by default."""
        from victor.framework.hitl import ApprovalRequest

        request = ApprovalRequest(
            id="req-123",
            title="Test",
            description="Test description",
        )

        assert request.timeout_seconds == 300

    def test_approval_request_custom_timeout(self):
        """ApprovalRequest should accept custom timeout."""
        from victor.framework.hitl import ApprovalRequest

        request = ApprovalRequest(
            id="req-123",
            title="Test",
            description="Test description",
            timeout_seconds=60,
        )

        assert request.timeout_seconds == 60

    def test_approval_request_default_status(self):
        """ApprovalRequest should have PENDING status by default."""
        from victor.framework.hitl import ApprovalRequest, ApprovalStatus

        request = ApprovalRequest(
            id="req-123",
            title="Test",
            description="Test description",
        )

        assert request.status == ApprovalStatus.PENDING

    def test_approval_request_response_fields(self):
        """ApprovalRequest should have response and responder fields."""
        from victor.framework.hitl import ApprovalRequest, ApprovalStatus

        request = ApprovalRequest(
            id="req-123",
            title="Test",
            description="Test description",
            status=ApprovalStatus.APPROVED,
            response="Looks good to me",
            responder="user@example.com",
        )

        assert request.status == ApprovalStatus.APPROVED
        assert request.response == "Looks good to me"
        assert request.responder == "user@example.com"

    def test_approval_request_is_pending(self):
        """ApprovalRequest should have is_pending property."""
        from victor.framework.hitl import ApprovalRequest, ApprovalStatus

        pending = ApprovalRequest(
            id="req-1",
            title="Test",
            description="Test",
        )
        approved = ApprovalRequest(
            id="req-2",
            title="Test",
            description="Test",
            status=ApprovalStatus.APPROVED,
        )

        assert pending.is_pending is True
        assert approved.is_pending is False

    def test_approval_request_is_approved(self):
        """ApprovalRequest should have is_approved property."""
        from victor.framework.hitl import ApprovalRequest, ApprovalStatus

        pending = ApprovalRequest(
            id="req-1",
            title="Test",
            description="Test",
        )
        approved = ApprovalRequest(
            id="req-2",
            title="Test",
            description="Test",
            status=ApprovalStatus.APPROVED,
        )

        assert pending.is_approved is False
        assert approved.is_approved is True

    def test_approval_request_is_rejected(self):
        """ApprovalRequest should have is_rejected property."""
        from victor.framework.hitl import ApprovalRequest, ApprovalStatus

        pending = ApprovalRequest(
            id="req-1",
            title="Test",
            description="Test",
        )
        rejected = ApprovalRequest(
            id="req-2",
            title="Test",
            description="Test",
            status=ApprovalStatus.REJECTED,
        )

        assert pending.is_rejected is False
        assert rejected.is_rejected is True


# =============================================================================
# HITLController Tests
# =============================================================================


class TestHITLControllerPause:
    """Tests for HITLController pause/resume functionality."""

    def test_is_paused_initially_false(self):
        """HITLController should not be paused initially."""
        from victor.framework.hitl import HITLController

        controller = HITLController()

        assert controller.is_paused is False

    def test_interrupt_sets_paused_true(self):
        """interrupt() should set is_paused to True."""
        from victor.framework.hitl import HITLController

        controller = HITLController()
        controller.interrupt()

        assert controller.is_paused is True

    def test_interrupt_returns_checkpoint_id(self):
        """interrupt() should return a checkpoint ID."""
        from victor.framework.hitl import HITLController

        controller = HITLController()
        checkpoint_id = controller.interrupt()

        assert checkpoint_id is not None
        assert isinstance(checkpoint_id, str)
        assert len(checkpoint_id) > 0

    def test_interrupt_with_context(self):
        """interrupt() should accept optional context."""
        from victor.framework.hitl import HITLController

        controller = HITLController()
        context = {"step": 5, "tools_used": ["read_file", "write_file"]}
        checkpoint_id = controller.interrupt(context=context)

        assert checkpoint_id is not None
        assert controller.is_paused is True

    def test_resume_sets_paused_false(self):
        """resume() should set is_paused to False."""
        from victor.framework.hitl import HITLController

        controller = HITLController()
        checkpoint_id = controller.interrupt()
        controller.resume(checkpoint_id)

        assert controller.is_paused is False

    def test_resume_with_invalid_checkpoint_raises(self):
        """resume() with invalid checkpoint should raise ValueError."""
        from victor.framework.hitl import HITLController

        controller = HITLController()
        controller.interrupt()

        with pytest.raises(ValueError):
            controller.resume("invalid-checkpoint-id")

    def test_multiple_interrupts_return_different_ids(self):
        """Multiple interrupts should return different checkpoint IDs."""
        from victor.framework.hitl import HITLController

        controller = HITLController()
        id1 = controller.interrupt()
        controller.resume(id1)

        id2 = controller.interrupt()
        controller.resume(id2)

        assert id1 != id2

    def test_get_checkpoint_context(self):
        """get_checkpoint_context() should return stored context."""
        from victor.framework.hitl import HITLController

        controller = HITLController()
        context = {"iteration": 10, "last_tool": "code_search"}
        checkpoint_id = controller.interrupt(context=context)

        retrieved = controller.get_checkpoint_context(checkpoint_id)

        assert retrieved == context

    def test_interrupt_normalizes_execution_checkpoint_context(self):
        """interrupt() should store trace-safe execution checkpoint metadata."""
        from victor.framework.execution_checkpoint import ExecutionCheckpoint
        from victor.framework.hitl import HITLController

        checkpoint = ExecutionCheckpoint.create(
            session_id="session-1",
            graph_checkpoint_id="graph-ckpt-1",
            conversation_checkpoint_id="conversation-ckpt-1",
        )
        controller = HITLController()
        checkpoint_id = controller.interrupt(
            context={"execution_checkpoint": checkpoint, "step": 5}
        )

        stored = controller.get_checkpoint_context(checkpoint_id)

        assert stored["execution_checkpoint_metadata"] == checkpoint.to_trace_metadata()
        assert stored["execution_checkpoint"] == checkpoint.to_dict()
        assert stored["step"] == 5

    def test_resume_returns_normalized_execution_checkpoint_context(self):
        """resume() should return the normalized HITL checkpoint context."""
        from victor.framework.execution_checkpoint import ExecutionCheckpoint
        from victor.framework.hitl import HITLController

        checkpoint = ExecutionCheckpoint.create(
            session_id="session-1",
            graph_checkpoint_id="graph-ckpt-1",
        )
        controller = HITLController()
        checkpoint_id = controller.interrupt(context={"execution_checkpoint": checkpoint})

        resumed = controller.resume(checkpoint_id)

        assert resumed["execution_checkpoint_metadata"] == checkpoint.to_trace_metadata()
        assert resumed["execution_checkpoint"] == checkpoint.to_dict()


# =============================================================================
# HITLController Approval Tests
# =============================================================================


class TestHITLControllerApproval:
    """Tests for HITLController approval request functionality."""

    def test_request_approval_creates_request(self):
        """request_approval() should create an ApprovalRequest."""
        from victor.framework.hitl import HITLController, ApprovalStatus

        controller = HITLController()
        request = controller.request_approval(
            title="Test Approval",
            description="Please approve this test",
        )

        assert request is not None
        assert request.title == "Test Approval"
        assert request.description == "Please approve this test"
        assert request.status == ApprovalStatus.PENDING

    def test_request_approval_generates_id(self):
        """request_approval() should generate unique IDs."""
        from victor.framework.hitl import HITLController

        controller = HITLController()
        req1 = controller.request_approval(
            title="Test 1",
            description="First",
        )
        req2 = controller.request_approval(
            title="Test 2",
            description="Second",
        )

        assert req1.id != req2.id

    def test_request_approval_with_context(self):
        """request_approval() should accept context dict."""
        from victor.framework.hitl import HITLController

        controller = HITLController()
        context = {"action": "delete", "target": "database"}
        request = controller.request_approval(
            title="Dangerous Operation",
            description="Delete entire database",
            context=context,
        )

        assert request.context == context

    def test_request_approval_normalizes_execution_checkpoint_context(self):
        """Approval requests should carry trace-safe execution checkpoint metadata."""
        from victor.framework.execution_checkpoint import ExecutionCheckpoint
        from victor.framework.hitl import HITLController

        checkpoint = ExecutionCheckpoint.create(
            session_id="session-1",
            graph_checkpoint_id="graph-ckpt-1",
            conversation_checkpoint_id="conversation-ckpt-1",
            approval_state="pending",
        )
        controller = HITLController()
        request = controller.request_approval(
            title="Review write",
            description="Approve file-changing step",
            context={"execution_checkpoint": checkpoint, "risk": "write"},
        )

        assert request.context["execution_checkpoint_metadata"] == checkpoint.to_trace_metadata()
        assert request.context["execution_checkpoint"] == checkpoint.to_dict()
        assert request.context["risk"] == "write"

    def test_request_approval_with_timeout(self):
        """request_approval() should accept custom timeout."""
        from victor.framework.hitl import HITLController

        controller = HITLController()
        request = controller.request_approval(
            title="Quick Decision",
            description="Needs fast approval",
            timeout_seconds=30,
        )

        assert request.timeout_seconds == 30

    def test_get_pending_requests(self):
        """get_pending_requests() should return pending requests."""
        from victor.framework.hitl import HITLController, ApprovalStatus

        controller = HITLController()
        req1 = controller.request_approval(
            title="Test 1",
            description="First",
        )
        req2 = controller.request_approval(
            title="Test 2",
            description="Second",
        )

        pending = controller.get_pending_requests()

        assert len(pending) == 2
        assert all(r.status == ApprovalStatus.PENDING for r in pending)

    def test_respond_to_request_approved(self):
        """respond_to_request() should update request to approved."""
        from victor.framework.hitl import HITLController, ApprovalStatus

        controller = HITLController()
        request = controller.request_approval(
            title="Test",
            description="Test",
        )

        updated = controller.respond_to_request(
            request_id=request.id,
            approved=True,
            response="LGTM",
            responder="admin",
        )

        assert updated.status == ApprovalStatus.APPROVED
        assert updated.response == "LGTM"
        assert updated.responder == "admin"

    def test_respond_to_request_approved_updates_execution_checkpoint_metadata(self):
        """Approved requests should update checkpoint approval metadata."""
        from victor.framework.execution_checkpoint import ExecutionCheckpoint
        from victor.framework.hitl import HITLController

        checkpoint = ExecutionCheckpoint.create(
            session_id="session-1",
            graph_checkpoint_id="graph-ckpt-1",
            approval_state="pending",
        )
        controller = HITLController()
        request = controller.request_approval(
            title="Test",
            description="Test",
            context={"execution_checkpoint": checkpoint},
        )

        updated = controller.respond_to_request(
            request_id=request.id,
            approved=True,
        )

        assert updated.context["execution_checkpoint_metadata"]["approval_state"] == "approved"
        assert updated.context["execution_checkpoint"]["approval_state"] == "approved"

    def test_respond_to_request_rejected(self):
        """respond_to_request() should update request to rejected."""
        from victor.framework.hitl import HITLController, ApprovalStatus

        controller = HITLController()
        request = controller.request_approval(
            title="Test",
            description="Test",
        )

        updated = controller.respond_to_request(
            request_id=request.id,
            approved=False,
            response="Not safe",
            responder="admin",
        )

        assert updated.status == ApprovalStatus.REJECTED
        assert updated.response == "Not safe"

    def test_respond_to_request_rejected_updates_execution_checkpoint_metadata(self):
        """Rejected requests should update checkpoint approval metadata."""
        from victor.framework.execution_checkpoint import ExecutionCheckpoint
        from victor.framework.hitl import HITLController

        checkpoint = ExecutionCheckpoint.create(
            session_id="session-1",
            graph_checkpoint_id="graph-ckpt-1",
            approval_state="pending",
        )
        controller = HITLController()
        request = controller.request_approval(
            title="Test",
            description="Test",
            context={"execution_checkpoint": checkpoint},
        )

        updated = controller.respond_to_request(
            request_id=request.id,
            approved=False,
        )

        assert updated.context["execution_checkpoint_metadata"]["approval_state"] == "rejected"
        assert updated.context["execution_checkpoint"]["approval_state"] == "rejected"

    def test_respond_to_invalid_request_raises(self):
        """respond_to_request() with invalid ID should raise ValueError."""
        from victor.framework.hitl import HITLController

        controller = HITLController()

        with pytest.raises(ValueError):
            controller.respond_to_request(
                request_id="nonexistent",
                approved=True,
            )


# =============================================================================
# HITLController Timeout Tests
# =============================================================================


class TestHITLControllerTimeout:
    """Tests for HITLController timeout handling."""

    @pytest.mark.asyncio
    async def test_wait_for_approval_returns_request(self):
        """wait_for_approval() should return the request."""
        from victor.framework.hitl import HITLController, ApprovalStatus

        async def auto_approve(controller, request_id):
            """Approve after short delay."""
            await asyncio.sleep(0.01)
            controller.respond_to_request(
                request_id=request_id,
                approved=True,
            )

        controller = HITLController()
        request = controller.request_approval(
            title="Test",
            description="Test",
        )

        # Start auto-approve in background
        asyncio.create_task(auto_approve(controller, request.id))

        result = await controller.wait_for_approval(request.id, timeout=1.0)

        assert result.status == ApprovalStatus.APPROVED

    @pytest.mark.asyncio
    async def test_wait_for_approval_timeout(self):
        """wait_for_approval() should timeout and set status."""
        from victor.framework.hitl import HITLController, ApprovalStatus

        controller = HITLController()
        request = controller.request_approval(
            title="Test",
            description="Test",
        )

        result = await controller.wait_for_approval(request.id, timeout=0.01)

        assert result.status == ApprovalStatus.TIMEOUT

    @pytest.mark.asyncio
    async def test_wait_for_approval_invalid_id_raises(self):
        """wait_for_approval() with invalid ID should raise ValueError."""
        from victor.framework.hitl import HITLController

        controller = HITLController()

        with pytest.raises(ValueError):
            await controller.wait_for_approval("nonexistent", timeout=1.0)


# =============================================================================
# HITLController with Custom Handler Tests
# =============================================================================


class TestHITLControllerHandler:
    """Tests for HITLController with custom approval handler."""

    @pytest.mark.asyncio
    async def test_custom_approval_handler(self):
        """HITLController should use custom approval handler."""
        from victor.framework.hitl import HITLController, ApprovalStatus

        async def custom_handler(request):
            """Auto-approve all requests."""
            return ApprovalStatus.APPROVED, "Auto-approved", "system"

        controller = HITLController(approval_handler=custom_handler)
        request = controller.request_approval(
            title="Test",
            description="Test",
        )

        result = await controller.process_approval(request.id)

        assert result.status == ApprovalStatus.APPROVED
        assert result.response == "Auto-approved"
        assert result.responder == "system"

    @pytest.mark.asyncio
    async def test_custom_approval_handler_updates_execution_checkpoint_metadata(self):
        """Custom approval handlers should update checkpoint approval metadata."""
        from victor.framework.execution_checkpoint import ExecutionCheckpoint
        from victor.framework.hitl import HITLController, ApprovalStatus

        async def custom_handler(request):
            """Auto-approve all requests."""
            return ApprovalStatus.APPROVED, "Auto-approved", "system"

        checkpoint = ExecutionCheckpoint.create(
            session_id="session-1",
            graph_checkpoint_id="graph-ckpt-1",
            approval_state="pending",
        )
        controller = HITLController(approval_handler=custom_handler)
        request = controller.request_approval(
            title="Test",
            description="Test",
            context={"execution_checkpoint": checkpoint},
        )

        result = await controller.process_approval(request.id)

        assert result.context["execution_checkpoint_metadata"]["approval_state"] == "approved"
        assert result.context["execution_checkpoint"]["approval_state"] == "approved"

    @pytest.mark.asyncio
    async def test_custom_rejection_handler(self):
        """HITLController should use custom rejection handler."""
        from victor.framework.hitl import HITLController, ApprovalStatus

        async def reject_handler(request):
            """Reject all requests."""
            return ApprovalStatus.REJECTED, "Policy violation", "policy-bot"

        controller = HITLController(approval_handler=reject_handler)
        request = controller.request_approval(
            title="Dangerous",
            description="Dangerous operation",
        )

        result = await controller.process_approval(request.id)

        assert result.status == ApprovalStatus.REJECTED
        assert result.responder == "policy-bot"

    def test_default_approval_for_testing(self):
        """_default_approval() should auto-approve for testing."""
        from victor.framework.hitl import HITLController, ApprovalStatus

        controller = HITLController()
        request = controller.request_approval(
            title="Test",
            description="Test",
        )

        # Use synchronous default approval
        controller._default_approval(request.id)
        updated = controller.get_request(request.id)

        assert updated.status == ApprovalStatus.APPROVED


# =============================================================================
# HITLController Events Tests
# =============================================================================


class TestHITLControllerEvents:
    """Tests for HITLController event callbacks."""

    def test_on_pause_callback(self):
        """on_pause callback should be called on interrupt."""
        from victor.framework.hitl import HITLController

        callback_called = []

        def on_pause(checkpoint_id, context):
            callback_called.append((checkpoint_id, context))

        controller = HITLController()
        controller.on_pause(on_pause)
        context = {"step": 1}
        checkpoint_id = controller.interrupt(context=context)

        assert len(callback_called) == 1
        assert callback_called[0][0] == checkpoint_id
        assert callback_called[0][1] == context

    def test_on_resume_callback(self):
        """on_resume callback should be called on resume."""
        from victor.framework.hitl import HITLController

        callback_called = []

        def on_resume(checkpoint_id):
            callback_called.append(checkpoint_id)

        controller = HITLController()
        controller.on_resume(on_resume)
        checkpoint_id = controller.interrupt()
        controller.resume(checkpoint_id)

        assert len(callback_called) == 1
        assert callback_called[0] == checkpoint_id

    def test_on_approval_request_callback(self):
        """on_approval_request callback should be called on new request."""
        from victor.framework.hitl import HITLController

        callback_called = []

        def on_request(request):
            callback_called.append(request)

        controller = HITLController()
        controller.on_approval_request(on_request)
        request = controller.request_approval(
            title="Test",
            description="Test",
        )

        assert len(callback_called) == 1
        assert callback_called[0].id == request.id


# =============================================================================
# Framework Export Tests
# =============================================================================


class TestHITLExports:
    """Tests for HITL exports from framework."""

    def test_hitl_exported_from_framework(self):
        """HITL types should be exported from victor.framework.hitl."""
        from victor.framework.hitl import (
            ApprovalStatus,
            ApprovalRequest,
            HITLController,
        )

        assert ApprovalStatus is not None
        assert ApprovalRequest is not None
        assert HITLController is not None
