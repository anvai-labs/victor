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

"""Integration tests for HITL approval request lifecycle and flows.

These tests verify the complete approval request lifecycle including:
- Request creation and lifecycle
- Custom approval handlers
- Rejection flow with reason propagation
- Timeout handling with different fallback behaviors
- Multiple concurrent approval requests
"""

import asyncio
import pytest
import time
from typing import Dict, Any, Optional, Tuple

from victor.framework.hitl import (
    HITLController,
    ApprovalStatus,
    ApprovalRequest,
    ApprovalHandler,
)
from victor.workflows.hitl import (
    HITLNode,
    HITLNodeType,
    HITLFallback,
    HITLStatus,
    HITLRequest,
    HITLResponse,
    HITLExecutor,
)

# =============================================================================
# Approval Request Lifecycle Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.hitl
class TestApprovalRequestLifecycle:
    """Test approval request creation and lifecycle management."""

    def test_request_creation_with_full_context(self):
        """Approval requests should capture full context at creation time."""
        controller = HITLController()

        context = {
            "operation": "database_migration",
            "target_tables": ["users", "orders", "products"],
            "estimated_duration": "15 minutes",
            "risk_level": "high",
            "rollback_available": True,
        }

        request = controller.request_approval(
            title="Database Migration",
            description="Migrate production database schema with user data",
            context=context,
            timeout_seconds=600,
        )

        assert request.id.startswith("req_")
        assert request.title == "Database Migration"
        assert request.description == "Migrate production database schema with user data"
        assert request.context == context
        assert request.timeout_seconds == 600
        assert request.status == ApprovalStatus.PENDING
        assert request.is_pending is True
        assert request.created_at <= time.time()

    def test_request_lifecycle_pending_to_approved(self):
        """Request should transition from PENDING to APPROVED correctly."""
        controller = HITLController()

        request = controller.request_approval(
            title="Deploy Feature",
            description="Deploy new authentication feature",
        )

        # Initially pending
        assert request.status == ApprovalStatus.PENDING
        assert request.is_pending is True
        assert request.is_approved is False

        # Approve the request
        updated = controller.respond_to_request(
            request_id=request.id,
            approved=True,
            response="Deployment approved after security review",
            responder="security-team@company.com",
        )

        # Verify transition
        assert updated.status == ApprovalStatus.APPROVED
        assert updated.is_approved is True
        assert updated.is_pending is False
        assert updated.response == "Deployment approved after security review"
        assert updated.responder == "security-team@company.com"

    def test_request_lifecycle_pending_to_rejected(self):
        """Request should transition from PENDING to REJECTED correctly."""
        controller = HITLController()

        request = controller.request_approval(
            title="Delete User Data",
            description="Permanent deletion of inactive user accounts",
            context={"accounts_affected": 15000},
        )

        # Reject the request
        updated = controller.respond_to_request(
            request_id=request.id,
            approved=False,
            response="Rejected: GDPR compliance review required first",
            responder="compliance@company.com",
        )

        # Verify transition
        assert updated.status == ApprovalStatus.REJECTED
        assert updated.is_rejected is True
        assert updated.is_approved is False
        assert "GDPR compliance" in updated.response

    def test_request_can_be_retrieved_after_response(self):
        """Request should be retrievable and show updated status."""
        controller = HITLController()

        request = controller.request_approval(
            title="Test Request",
            description="Test",
        )

        controller.respond_to_request(
            request_id=request.id,
            approved=True,
        )

        # Retrieve and verify
        retrieved = controller.get_request(request.id)
        assert retrieved.status == ApprovalStatus.APPROVED

    def test_pending_requests_list_updates_correctly(self):
        """Pending requests list should reflect current state."""
        controller = HITLController()

        # Create multiple requests
        req1 = controller.request_approval(title="Request 1", description="First")
        req2 = controller.request_approval(title="Request 2", description="Second")
        req3 = controller.request_approval(title="Request 3", description="Third")

        # All should be pending
        pending = controller.get_pending_requests()
        assert len(pending) == 3

        # Approve one, reject another
        controller.respond_to_request(req1.id, approved=True)
        controller.respond_to_request(req2.id, approved=False)

        # Only one should remain pending
        pending = controller.get_pending_requests()
        assert len(pending) == 1
        assert pending[0].id == req3.id


# =============================================================================
# Custom Approval Handler Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.hitl
class TestCustomApprovalHandlers:
    """Test custom approval handlers for different scenarios."""

    @pytest.mark.asyncio
    async def test_handler_receives_full_request_context(self):
        """Custom handler should receive complete request information."""
        received_requests = []

        async def tracking_handler(
            request: ApprovalRequest,
        ) -> Tuple[ApprovalStatus, Optional[str], Optional[str]]:
            received_requests.append(request)
            return ApprovalStatus.APPROVED, "Handled", "handler"

        controller = HITLController(approval_handler=tracking_handler)

        request = controller.request_approval(
            title="Test Title",
            description="Test Description",
            context={"key": "value"},
            timeout_seconds=120,
        )

        await controller.process_approval(request.id)

        assert len(received_requests) == 1
        received = received_requests[0]
        assert received.title == "Test Title"
        assert received.description == "Test Description"
        assert received.context == {"key": "value"}
        assert received.timeout_seconds == 120

    @pytest.mark.asyncio
    async def test_policy_based_approval_handler(self):
        """Handler can implement policy-based approval logic."""

        async def policy_handler(
            request: ApprovalRequest,
        ) -> Tuple[ApprovalStatus, Optional[str], Optional[str]]:
            # Policy: Auto-approve low-risk, manual review for high-risk
            risk_level = request.context.get("risk_level", "unknown")

            if risk_level == "low":
                return (
                    ApprovalStatus.APPROVED,
                    "Auto-approved (low risk)",
                    "policy-engine",
                )
            elif risk_level == "high":
                return (
                    ApprovalStatus.REJECTED,
                    "High risk requires manual review",
                    "policy-engine",
                )
            else:
                return ApprovalStatus.PENDING, None, None

        controller = HITLController(approval_handler=policy_handler)

        # Low risk should auto-approve
        low_risk = controller.request_approval(
            title="Minor Change",
            description="Update configuration",
            context={"risk_level": "low"},
        )
        result = await controller.process_approval(low_risk.id)
        assert result.is_approved is True
        assert "Auto-approved" in result.response

        # High risk should reject for manual review
        high_risk = controller.request_approval(
            title="Critical Change",
            description="Modify production database",
            context={"risk_level": "high"},
        )
        result = await controller.process_approval(high_risk.id)
        assert result.is_rejected is True
        assert "manual review" in result.response.lower()

    @pytest.mark.asyncio
    async def test_handler_with_external_service_simulation(self):
        """Handler can simulate external service calls (Slack, email, etc.)."""
        notification_log = []

        async def notification_handler(
            request: ApprovalRequest,
        ) -> Tuple[ApprovalStatus, Optional[str], Optional[str]]:
            # Simulate sending notification to external service
            notification_log.append(
                {
                    "service": "slack",
                    "channel": "#approvals",
                    "title": request.title,
                    "timestamp": time.time(),
                }
            )

            # Simulate waiting for response (instant in test)
            await asyncio.sleep(0.01)

            # Simulate receiving approval from external service
            notification_log.append(
                {
                    "service": "slack",
                    "action": "approved",
                    "user": "alice@company.com",
                }
            )

            return ApprovalStatus.APPROVED, "Approved via Slack", "alice@company.com"

        controller = HITLController(approval_handler=notification_handler)

        request = controller.request_approval(
            title="Deployment Request",
            description="Deploy to staging",
        )

        result = await controller.process_approval(request.id)

        assert result.is_approved is True
        assert result.responder == "alice@company.com"
        assert len(notification_log) == 2
        assert notification_log[0]["service"] == "slack"

    @pytest.mark.asyncio
    async def test_default_handler_auto_approves(self):
        """Controller without custom handler should auto-approve."""
        controller = HITLController()  # No custom handler

        request = controller.request_approval(
            title="Test",
            description="Test",
        )

        result = await controller.process_approval(request.id)

        assert result.is_approved is True
        assert "Auto-approved" in result.response
        assert result.responder == "system"


# =============================================================================
# Rejection Flow Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.hitl
class TestRejectionFlowWithReasonPropagation:
    """Test rejection flows and reason propagation."""

    def test_rejection_reason_preserved(self):
        """Rejection reason should be preserved through the flow."""
        controller = HITLController()

        request = controller.request_approval(
            title="Delete Repository",
            description="Permanently delete code repository",
            context={"repository": "company/core-api"},
        )

        rejection_reason = (
            "Rejected: Repository contains unreleased features. "
            "Please coordinate with product team before deletion."
        )

        updated = controller.respond_to_request(
            request_id=request.id,
            approved=False,
            response=rejection_reason,
            responder="engineering-lead@company.com",
        )

        assert updated.is_rejected is True
        assert updated.response == rejection_reason
        assert "unreleased features" in updated.response
        assert updated.responder == "engineering-lead@company.com"

    @pytest.mark.asyncio
    async def test_rejection_from_custom_handler_with_detailed_reason(self):
        """Custom handler should propagate detailed rejection reasons."""

        async def detailed_rejection_handler(
            request: ApprovalRequest,
        ) -> Tuple[ApprovalStatus, Optional[str], Optional[str]]:
            # Check multiple conditions and provide detailed rejection reason
            reasons = []

            if "production" in str(request.context).lower():
                reasons.append("Production environment requires additional approval")

            if request.context.get("changes_count", 0) > 100:
                reasons.append(f"Large change set ({request.context.get('changes_count')} files)")

            if not request.context.get("tests_passed", False):
                reasons.append("Tests must pass before approval")

            if reasons:
                return ApprovalStatus.REJECTED, "; ".join(reasons), "validation-bot"
            return ApprovalStatus.APPROVED, "All checks passed", "validation-bot"

        controller = HITLController(approval_handler=detailed_rejection_handler)

        request = controller.request_approval(
            title="Large Production Deployment",
            description="Deploy changes to production",
            context={
                "environment": "production",
                "changes_count": 150,
                "tests_passed": False,
            },
        )

        result = await controller.process_approval(request.id)

        assert result.is_rejected is True
        assert "Production environment" in result.response
        assert "150 files" in result.response
        assert "Tests must pass" in result.response

    def test_multiple_rejections_preserve_individual_reasons(self):
        """Multiple rejections should each preserve their own reasons."""
        controller = HITLController()

        requests = [
            controller.request_approval(
                title=f"Request {i}",
                description=f"Description {i}",
            )
            for i in range(3)
        ]

        reasons = [
            "Security review pending",
            "Budget approval needed",
            "Legal review required",
        ]

        for request, reason in zip(requests, reasons):
            controller.respond_to_request(
                request_id=request.id,
                approved=False,
                response=reason,
            )

        # Verify each request has its own reason
        for request, expected_reason in zip(requests, reasons):
            retrieved = controller.get_request(request.id)
            assert retrieved.response == expected_reason


# =============================================================================
# Timeout Handling Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.hitl
class TestTimeoutHandlingWithFallbackBehaviors:
    """Test timeout handling with different fallback behaviors."""

    @pytest.mark.asyncio
    async def test_timeout_sets_status_correctly(self):
        """Timeout should set request status to TIMEOUT."""
        controller = HITLController()

        request = controller.request_approval(
            title="Quick Decision Needed",
            description="Urgent approval required",
            timeout_seconds=0.01,  # Very short timeout
        )

        result = await controller.wait_for_approval(request.id, timeout=0.02)

        assert result.status == ApprovalStatus.TIMEOUT
        assert result.is_timeout is True
        assert "timed out" in result.response.lower()

    @pytest.mark.asyncio
    async def test_workflow_hitl_timeout_with_continue_fallback(self):
        """CONTINUE fallback should allow workflow to proceed with default."""

        class NeverRespondsHandler:
            async def request_human_input(self, request: HITLRequest) -> HITLResponse:
                await asyncio.sleep(10.0)  # Will be cancelled by timeout
                raise AssertionError("Should not reach here")

        node = HITLNode(
            id="timeout_continue",
            name="Timeout Continue Test",
            hitl_type=HITLNodeType.CONFIRMATION,
            prompt="Confirm to continue",
            timeout=0.01,  # Very short timeout
            fallback=HITLFallback.CONTINUE,
            default_value="default_action",
        )

        executor = HITLExecutor(handler=NeverRespondsHandler())
        response = await executor.execute_hitl_node(node, {})

        assert response.status == HITLStatus.TIMEOUT
        assert response.approved is True  # CONTINUE allows proceeding
        assert response.value == "default_action"
        assert "continuing with default" in response.reason.lower()

    @pytest.mark.asyncio
    async def test_workflow_hitl_timeout_with_abort_fallback(self):
        """ABORT fallback should prevent workflow from proceeding."""

        class NeverRespondsHandler:
            async def request_human_input(self, request: HITLRequest) -> HITLResponse:
                await asyncio.sleep(10.0)
                raise AssertionError("Should not reach here")

        node = HITLNode(
            id="timeout_abort",
            name="Timeout Abort Test",
            hitl_type=HITLNodeType.APPROVAL,
            prompt="Approve this action",
            timeout=0.01,
            fallback=HITLFallback.ABORT,
        )

        executor = HITLExecutor(handler=NeverRespondsHandler())
        response = await executor.execute_hitl_node(node, {})

        assert response.status == HITLStatus.TIMEOUT
        assert response.approved is False  # ABORT prevents proceeding
        assert "0.01" in response.reason  # Includes timeout duration

    @pytest.mark.asyncio
    async def test_workflow_hitl_timeout_with_skip_fallback(self):
        """SKIP fallback should skip the node entirely."""

        class NeverRespondsHandler:
            async def request_human_input(self, request: HITLRequest) -> HITLResponse:
                await asyncio.sleep(10.0)
                raise AssertionError("Should not reach here")

        node = HITLNode(
            id="timeout_skip",
            name="Timeout Skip Test",
            hitl_type=HITLNodeType.REVIEW,
            prompt="Review this content",
            timeout=0.01,
            fallback=HITLFallback.SKIP,
        )

        executor = HITLExecutor(handler=NeverRespondsHandler())
        response = await executor.execute_hitl_node(node, {})

        assert response.status == HITLStatus.SKIPPED
        assert response.approved is True  # SKIP allows proceeding
        assert "skipping" in response.reason.lower()

    @pytest.mark.asyncio
    async def test_response_before_timeout_prevents_timeout(self):
        """Response received before timeout should prevent timeout status."""

        class QuickHandler:
            async def request_human_input(self, request: HITLRequest) -> HITLResponse:
                await asyncio.sleep(0.01)  # Quick response
                return HITLResponse(
                    request_id=request.request_id,
                    status=HITLStatus.APPROVED,
                    approved=True,
                )

        node = HITLNode(
            id="quick_response",
            name="Quick Response Test",
            hitl_type=HITLNodeType.APPROVAL,
            prompt="Approve quickly",
            timeout=1.0,  # Longer timeout
            fallback=HITLFallback.ABORT,
        )

        executor = HITLExecutor(handler=QuickHandler())
        response = await executor.execute_hitl_node(node, {})

        assert response.status == HITLStatus.APPROVED
        assert response.approved is True


# =============================================================================
# Concurrent Approval Requests Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.hitl
class TestConcurrentApprovalRequests:
    """Test handling of multiple concurrent approval requests."""

    def test_multiple_requests_tracked_independently(self):
        """Multiple concurrent requests should be tracked independently."""
        controller = HITLController()

        # Create multiple requests concurrently
        requests = []
        for i in range(5):
            req = controller.request_approval(
                title=f"Concurrent Request {i}",
                description=f"Request number {i}",
                context={"index": i},
            )
            requests.append(req)

        # All should be pending
        pending = controller.get_pending_requests()
        assert len(pending) == 5

        # Each should have unique ID
        ids = [req.id for req in requests]
        assert len(set(ids)) == 5

    @pytest.mark.asyncio
    async def test_concurrent_wait_for_approvals(self):
        """Multiple wait_for_approval calls should work concurrently."""
        controller = HITLController()

        # Create multiple requests
        requests = [
            controller.request_approval(
                title=f"Request {i}",
                description=f"Description {i}",
            )
            for i in range(3)
        ]

        # Approve them with different delays
        async def approve_with_delay(request_id: str, delay: float, approved: bool):
            await asyncio.sleep(delay)
            controller.respond_to_request(
                request_id=request_id,
                approved=approved,
                response=f"Responded after {delay}s",
            )

        # Start approval tasks
        approve_tasks = [
            asyncio.create_task(approve_with_delay(requests[0].id, 0.05, True)),
            asyncio.create_task(approve_with_delay(requests[1].id, 0.03, False)),
            asyncio.create_task(approve_with_delay(requests[2].id, 0.07, True)),
        ]

        # Wait for all approvals concurrently
        wait_tasks = [
            asyncio.create_task(controller.wait_for_approval(req.id, timeout=1.0))
            for req in requests
        ]

        results = await asyncio.gather(*wait_tasks)

        # Verify results
        assert results[0].is_approved is True
        assert results[1].is_rejected is True
        assert results[2].is_approved is True

        # Clean up approve tasks
        await asyncio.gather(*approve_tasks)

    @pytest.mark.asyncio
    async def test_mixed_concurrent_responses(self):
        """Concurrent requests should handle mixed response types correctly."""
        controller = HITLController()

        # Create requests with different expected outcomes
        timeout_req = controller.request_approval(
            title="Will Timeout",
            description="This will timeout",
            timeout_seconds=0.05,
        )

        approve_req = controller.request_approval(
            title="Will Be Approved",
            description="This will be approved",
        )

        reject_req = controller.request_approval(
            title="Will Be Rejected",
            description="This will be rejected",
        )

        async def respond_to_requests():
            await asyncio.sleep(0.01)
            controller.respond_to_request(approve_req.id, approved=True)
            controller.respond_to_request(reject_req.id, approved=False)
            # Note: timeout_req is intentionally not responded to

        # Start response task
        respond_task = asyncio.create_task(respond_to_requests())

        # Wait for all (timeout_req will timeout)
        timeout_result, approve_result, reject_result = await asyncio.gather(
            controller.wait_for_approval(timeout_req.id, timeout=0.1),
            controller.wait_for_approval(approve_req.id, timeout=1.0),
            controller.wait_for_approval(reject_req.id, timeout=1.0),
        )

        await respond_task

        assert timeout_result.status == ApprovalStatus.TIMEOUT
        assert approve_result.status == ApprovalStatus.APPROVED
        assert reject_result.status == ApprovalStatus.REJECTED

    def test_pending_requests_filter_by_status(self):
        """get_pending_requests should only return pending requests."""
        controller = HITLController()

        # Create 5 requests
        requests = [
            controller.request_approval(title=f"Request {i}", description=f"Desc {i}")
            for i in range(5)
        ]

        # Approve 2, reject 1, leave 2 pending
        controller.respond_to_request(requests[0].id, approved=True)
        controller.respond_to_request(requests[1].id, approved=True)
        controller.respond_to_request(requests[2].id, approved=False)

        pending = controller.get_pending_requests()

        assert len(pending) == 2
        pending_ids = {req.id for req in pending}
        assert requests[3].id in pending_ids
        assert requests[4].id in pending_ids
        assert requests[0].id not in pending_ids
        assert requests[1].id not in pending_ids
        assert requests[2].id not in pending_ids


# =============================================================================
# Callback Integration Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.hitl
class TestApprovalCallbackIntegration:
    """Test callback integration for approval events."""

    def test_on_approval_request_callback_fires(self):
        """Callback should fire when approval request is created."""
        callback_calls = []

        def on_request(request: ApprovalRequest):
            callback_calls.append(
                {
                    "event": "request_created",
                    "id": request.id,
                    "title": request.title,
                }
            )

        controller = HITLController()
        controller.on_approval_request(on_request)

        controller.request_approval(
            title="Callback Test",
            description="Testing callback",
        )

        assert len(callback_calls) == 1
        assert callback_calls[0]["event"] == "request_created"
        assert callback_calls[0]["title"] == "Callback Test"

    def test_multiple_callbacks_all_fire(self):
        """Multiple registered callbacks should all fire."""
        callback1_calls = []
        callback2_calls = []

        controller = HITLController()
        controller.on_approval_request(lambda req: callback1_calls.append(req.id))
        controller.on_approval_request(lambda req: callback2_calls.append(req.title))

        controller.request_approval(title="Multi Callback", description="Test")

        assert len(callback1_calls) == 1
        assert len(callback2_calls) == 1
        assert callback2_calls[0] == "Multi Callback"

    def test_callback_receives_context(self):
        """Callback should have access to request context."""
        received_context = []

        def context_callback(request: ApprovalRequest):
            received_context.append(request.context.copy())

        controller = HITLController()
        controller.on_approval_request(context_callback)

        controller.request_approval(
            title="Context Test",
            description="Testing context in callback",
            context={"key": "value", "nested": {"a": 1}},
        )

        assert len(received_context) == 1
        assert received_context[0]["key"] == "value"
        assert received_context[0]["nested"]["a"] == 1
