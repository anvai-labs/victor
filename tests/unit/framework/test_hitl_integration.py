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

"""Integration tests for HITL Protocol and Personas with Teams.

These tests verify integration between:
- HITLController for human approval flows
- Persona system for agent characterization
- TeamMemberSpec for attaching personas to team members
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# =============================================================================
# HITL with Workflow Integration Tests
# =============================================================================


class TestHITLWorkflowIntegration:
    """Tests for HITL integration with workflow execution."""

    @pytest.mark.asyncio
    async def test_hitl_interrupts_workflow(self):
        """HITL should interrupt workflow and allow resume."""
        from victor.framework.hitl import HITLController

        controller = HITLController()
        workflow_steps_completed = []

        async def run_workflow():
            """Simulated workflow that can be interrupted."""
            workflow_steps_completed.append("step1")

            if controller.is_paused:
                return "interrupted"

            workflow_steps_completed.append("step2")

            if controller.is_paused:
                return "interrupted"

            workflow_steps_completed.append("step3")
            return "completed"

        # Run first part
        workflow_steps_completed.append("step1")

        # Interrupt the workflow
        checkpoint_id = controller.interrupt(
            context={"completed_steps": workflow_steps_completed.copy()}
        )

        assert controller.is_paused is True

        # Verify checkpoint context
        ctx = controller.get_checkpoint_context(checkpoint_id)
        assert ctx["completed_steps"] == ["step1"]

        # Resume workflow
        controller.resume(checkpoint_id)
        assert controller.is_paused is False

    @pytest.mark.asyncio
    async def test_hitl_approval_in_workflow(self):
        """HITL should handle approval requests within workflow."""
        from victor.framework.hitl import HITLController, ApprovalStatus

        controller = HITLController()
        workflow_log = []

        async def dangerous_operation():
            """Operation requiring approval."""
            workflow_log.append("requesting_approval")

            request = controller.request_approval(
                title="Delete Files",
                description="Delete all temporary files",
                context={"files": ["temp1.txt", "temp2.txt"]},
                timeout_seconds=5,
            )

            # Simulate external approval (in real scenario, would be async wait)
            controller.respond_to_request(
                request_id=request.id,
                approved=True,
                response="Approved by test",
                responder="test_user",
            )

            updated = controller.get_request(request.id)

            if updated.is_approved:
                workflow_log.append("operation_executed")
                return True
            else:
                workflow_log.append("operation_skipped")
                return False

        result = await dangerous_operation()

        assert result is True
        assert workflow_log == ["requesting_approval", "operation_executed"]

    @pytest.mark.asyncio
    async def test_hitl_rejection_stops_workflow(self):
        """HITL rejection should stop dangerous operation."""
        from victor.framework.hitl import HITLController, ApprovalStatus

        controller = HITLController()
        workflow_log = []

        async def dangerous_operation():
            """Operation that gets rejected."""
            workflow_log.append("requesting_approval")

            request = controller.request_approval(
                title="Drop Database",
                description="Drop production database",
                context={"database": "production"},
            )

            # Reject the request
            controller.respond_to_request(
                request_id=request.id,
                approved=False,
                response="Too dangerous",
                responder="admin",
            )

            updated = controller.get_request(request.id)

            if updated.is_approved:
                workflow_log.append("operation_executed")
                return True
            else:
                workflow_log.append("operation_rejected")
                return False

        result = await dangerous_operation()

        assert result is False
        assert workflow_log == ["requesting_approval", "operation_rejected"]

    @pytest.mark.asyncio
    async def test_hitl_timeout_handling(self):
        """HITL should handle timeout gracefully."""
        from victor.framework.hitl import HITLController, ApprovalStatus

        controller = HITLController()

        request = controller.request_approval(
            title="Test Timeout",
            description="This will timeout",
            timeout_seconds=1,
        )

        # Wait for timeout
        result = await controller.wait_for_approval(request.id, timeout=0.01)

        assert result.status == ApprovalStatus.TIMEOUT


# =============================================================================
# Persona with Team Member Integration Tests
# =============================================================================


class TestPersonaTeamMemberIntegration:
    """Tests for Persona integration with TeamMemberSpec."""

    def test_persona_attached_to_team_member_spec(self):
        """TeamMemberSpec should work with persona-like attributes."""
        from victor.framework.teams import TeamMemberSpec
        from victor.framework.personas import Persona, get_persona

        # Get a built-in persona for reference
        senior_dev = get_persona("senior_developer")

        # Create team member spec with persona-like attributes
        spec = TeamMemberSpec(
            role="executor",
            goal="Implement the authentication system",
            name=senior_dev.name if senior_dev else "Senior Developer",
            backstory="You are an expert in security and authentication patterns.",
            tool_budget=25,
        )

        assert spec.name == "Senior Developer"
        assert "expert" in spec.backstory.lower()

    def test_persona_provides_system_prompt_for_team_member(self):
        """Persona should provide system prompt for team member."""
        from victor.framework.personas import Persona

        persona = Persona(
            name="Security Expert",
            background="Expert in application security and authentication.",
            communication_style="professional",
            expertise_areas=("security", "authentication", "encryption"),
        )

        prompt_section = persona.get_system_prompt_section()

        # Can be injected into team member's prompt
        assert "Security Expert" in prompt_section
        assert "security" in prompt_section.lower()

    def test_multiple_personas_for_team(self):
        """Multiple personas can be used for different team roles."""
        from victor.framework.personas import get_persona

        researcher_persona = get_persona("mentor")
        developer_persona = get_persona("senior_developer")
        reviewer_persona = get_persona("code_reviewer")

        personas = [researcher_persona, developer_persona, reviewer_persona]

        # All personas should be distinct and available
        assert all(p is not None for p in personas)
        assert len(set(p.name for p in personas)) == 3  # All unique


# =============================================================================
# End-to-End Approval Flow Tests
# =============================================================================


class TestApprovalFlowEndToEnd:
    """End-to-end tests for approval flow."""

    @pytest.mark.asyncio
    async def test_full_approval_flow_with_callback(self):
        """Test complete approval flow with callbacks."""
        from victor.framework.hitl import HITLController, ApprovalStatus

        # Track events
        events = []

        controller = HITLController()

        # Register callbacks
        controller.on_approval_request(lambda req: events.append(("request_created", req.id)))

        # Create request
        request = controller.request_approval(
            title="Deploy to Production",
            description="Deploy v1.2.3 to production environment",
            context={"version": "1.2.3", "environment": "production"},
        )

        assert ("request_created", request.id) in events

        # Simulate human approval
        controller.respond_to_request(
            request_id=request.id,
            approved=True,
            response="Deployment approved after review",
            responder="ops_team@company.com",
        )

        # Verify final state
        final = controller.get_request(request.id)
        assert final.status == ApprovalStatus.APPROVED
        assert final.response == "Deployment approved after review"
        assert final.responder == "ops_team@company.com"

    @pytest.mark.asyncio
    async def test_concurrent_approval_requests(self):
        """Test handling multiple concurrent approval requests."""
        from victor.framework.hitl import HITLController, ApprovalStatus

        controller = HITLController()

        # Create multiple requests
        requests = []
        for i in range(3):
            req = controller.request_approval(
                title=f"Request {i}",
                description=f"Description for request {i}",
            )
            requests.append(req)

        # All should be pending
        pending = controller.get_pending_requests()
        assert len(pending) == 3

        # Approve first, reject second, leave third pending
        controller.respond_to_request(requests[0].id, approved=True)
        controller.respond_to_request(requests[1].id, approved=False)

        # Check states
        assert controller.get_request(requests[0].id).is_approved
        assert controller.get_request(requests[1].id).is_rejected
        assert controller.get_request(requests[2].id).is_pending

        # Only one should remain pending
        pending = controller.get_pending_requests()
        assert len(pending) == 1
        assert pending[0].id == requests[2].id

    @pytest.mark.asyncio
    async def test_custom_approval_handler_end_to_end(self):
        """Test end-to-end with custom approval handler."""
        from victor.framework.hitl import HITLController, ApprovalStatus

        approval_log = []

        async def policy_based_handler(request):
            """Auto-approve or reject based on context."""
            approval_log.append(f"Evaluating: {request.title}")

            # Policy: Reject anything with "production" in context
            if "production" in str(request.context).lower():
                return (
                    ApprovalStatus.REJECTED,
                    "Production changes require manual review",
                    "policy-bot",
                )
            else:
                return ApprovalStatus.APPROVED, "Auto-approved by policy", "policy-bot"

        controller = HITLController(approval_handler=policy_based_handler)

        # Safe request
        safe_req = controller.request_approval(
            title="Update Dev Config",
            description="Update development configuration",
            context={"environment": "development"},
        )
        safe_result = await controller.process_approval(safe_req.id)

        # Dangerous request
        prod_req = controller.request_approval(
            title="Update Prod Config",
            description="Update production configuration",
            context={"environment": "production"},
        )
        prod_result = await controller.process_approval(prod_req.id)

        # Verify results
        assert safe_result.is_approved
        assert prod_result.is_rejected
        assert len(approval_log) == 2


# =============================================================================
# HITL + Persona Combined Integration
# =============================================================================


class TestHITLPersonaCombined:
    """Tests for combined HITL and Persona usage."""

    @pytest.mark.asyncio
    async def test_persona_influences_approval_message(self):
        """Persona formatting should be usable in approval messages."""
        from victor.framework.hitl import HITLController
        from victor.framework.personas import Persona

        formal_persona = Persona(
            name="Formal Bot",
            background="A formal assistant.",
            communication_style="formal",
        )

        controller = HITLController()

        # Format a message using persona
        raw_message = "requesting approval for deployment"
        formatted = formal_persona.format_message(raw_message)

        request = controller.request_approval(
            title="Deployment Request",
            description=formatted,
        )

        # Formatted message should have proper capitalization and punctuation
        assert request.description[0].isupper()
        assert request.description.endswith(".")

    @pytest.mark.asyncio
    async def test_checkpoint_stores_persona_context(self):
        """Checkpoint should be able to store persona information."""
        from victor.framework.hitl import HITLController
        from victor.framework.personas import Persona, get_persona

        controller = HITLController()
        persona = get_persona("mentor")

        # Include persona info in checkpoint context
        checkpoint_id = controller.interrupt(
            context={
                "current_persona": persona.name if persona else "default",
                "step": "teaching_phase",
                "topics_covered": ["variables", "loops"],
            }
        )

        # Retrieve and verify
        ctx = controller.get_checkpoint_context(checkpoint_id)
        assert ctx["current_persona"] == "Mentor"
        assert ctx["step"] == "teaching_phase"
        assert "loops" in ctx["topics_covered"]


# =============================================================================
# Error Handling Integration Tests
# =============================================================================


class TestHITLErrorHandling:
    """Tests for HITL error handling in integration scenarios."""

    def test_invalid_checkpoint_resume_error(self):
        """Invalid checkpoint should raise clear error."""
        from victor.framework.hitl import HITLController

        controller = HITLController()
        controller.interrupt()  # Create valid checkpoint

        with pytest.raises(ValueError, match="Invalid checkpoint"):
            controller.resume("nonexistent-checkpoint-id")

    def test_invalid_request_response_error(self):
        """Invalid request response should raise clear error."""
        from victor.framework.hitl import HITLController

        controller = HITLController()

        with pytest.raises(ValueError, match="Invalid request"):
            controller.respond_to_request(
                request_id="nonexistent-request",
                approved=True,
            )

    @pytest.mark.asyncio
    async def test_approval_handler_exception_handling(self):
        """Custom handler exceptions should be handled gracefully."""
        from victor.framework.hitl import HITLController

        async def failing_handler(request):
            """Handler that raises an exception."""
            raise RuntimeError("Handler failed!")

        controller = HITLController(approval_handler=failing_handler)

        request = controller.request_approval(
            title="Test",
            description="Test",
        )

        # Should raise the exception (caller is responsible for handling)
        with pytest.raises(RuntimeError, match="Handler failed"):
            await controller.process_approval(request.id)
