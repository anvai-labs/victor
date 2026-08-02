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

"""Integration tests for HITL nodes in workflow graphs.

These tests verify:
- HITL nodes integration with workflow graphs
- Approval gates blocking workflow progression
- Context key filtering for approval requests
- HITL combined with conditional edges
"""

import asyncio
import pytest
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from victor.workflows.hitl import (
    HITLNode,
    HITLNodeType,
    HITLFallback,
    HITLStatus,
    HITLRequest,
    HITLResponse,
    HITLExecutor,
    HITLHandler,
)
from victor.workflows.definition import (
    WorkflowBuilder,
    WorkflowDefinition,
    WorkflowNodeType,
    ConditionNode,
)

# =============================================================================
# HITL Nodes in Workflow Graphs Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.hitl
class TestHITLNodesInWorkflowGraphs:
    """Test HITL nodes integration with workflow graph definitions."""

    def test_workflow_builder_creates_approval_node(self):
        """WorkflowBuilder should create HITL approval nodes correctly."""
        workflow = (
            WorkflowBuilder("test_workflow", "Test workflow with HITL")
            .add_agent("analyze", "researcher", "Analyze the code")
            .add_hitl_approval(
                "approve_changes",
                prompt="Do you approve these changes?",
                context_keys=["analysis_result", "files_affected"],
                timeout=300.0,
            )
            .add_agent("implement", "executor", "Implement changes")
            .build()
        )

        # Verify workflow structure
        assert workflow.name == "test_workflow"
        assert len(workflow.nodes) == 3

        # Verify HITL node
        hitl_node = workflow.get_node("approve_changes")
        assert hitl_node is not None
        assert hitl_node.node_type == WorkflowNodeType.HITL
        assert isinstance(hitl_node, HITLNode)
        assert hitl_node.hitl_type == HITLNodeType.APPROVAL
        assert hitl_node.prompt == "Do you approve these changes?"
        assert hitl_node.timeout == 300.0

    def test_workflow_builder_creates_choice_node(self):
        """WorkflowBuilder should create HITL choice nodes correctly."""
        workflow = (
            WorkflowBuilder("choice_workflow")
            .add_agent("analyze", "researcher", "Analyze options")
            .add_hitl_choice(
                "select_option",
                prompt="Select an option:",
                choices=["Option A", "Option B", "Option C"],
                default_value="Option A",
                timeout=60.0,
                fallback="continue",
            )
            .add_agent("execute", "executor", "Execute selected option")
            .build()
        )

        hitl_node = workflow.get_node("select_option")
        assert isinstance(hitl_node, HITLNode)
        assert hitl_node.hitl_type == HITLNodeType.CHOICE
        assert hitl_node.choices == ["Option A", "Option B", "Option C"]
        assert hitl_node.default_value == "Option A"
        assert hitl_node.fallback == HITLFallback.CONTINUE

    def test_workflow_builder_creates_review_node(self):
        """WorkflowBuilder should create HITL review nodes correctly."""
        workflow = (
            WorkflowBuilder("review_workflow")
            .add_agent("generate", "executor", "Generate code")
            .add_hitl_review(
                "review_code",
                prompt="Review the generated code",
                context_keys=["generated_code", "file_path"],
                timeout=600.0,
                fallback="abort",
            )
            .add_agent("finalize", "executor", "Finalize changes")
            .build()
        )

        hitl_node = workflow.get_node("review_code")
        assert isinstance(hitl_node, HITLNode)
        assert hitl_node.hitl_type == HITLNodeType.REVIEW
        assert hitl_node.timeout == 600.0
        assert hitl_node.fallback == HITLFallback.ABORT

    def test_workflow_builder_creates_confirmation_node(self):
        """WorkflowBuilder should create HITL confirmation nodes correctly."""
        workflow = (
            WorkflowBuilder("confirm_workflow")
            .add_agent("prepare", "executor", "Prepare deployment")
            .add_hitl_confirmation(
                "confirm_deploy",
                prompt="Press Enter to deploy",
                timeout=30.0,
                fallback="continue",
            )
            .add_agent("deploy", "executor", "Deploy to production")
            .build()
        )

        hitl_node = workflow.get_node("confirm_deploy")
        assert isinstance(hitl_node, HITLNode)
        assert hitl_node.hitl_type == HITLNodeType.CONFIRMATION
        assert hitl_node.timeout == 30.0
        assert hitl_node.fallback == HITLFallback.CONTINUE

    def test_hitl_node_auto_chains_in_workflow(self):
        """HITL nodes should auto-chain with other nodes in workflow."""
        workflow = (
            WorkflowBuilder("chain_test")
            .add_agent("step1", "researcher", "First step")
            .add_hitl_approval("approve", "Approve?")
            .add_agent("step2", "executor", "Second step")
            .build()
        )

        # Verify chaining
        step1 = workflow.get_node("step1")
        approve = workflow.get_node("approve")
        step2 = workflow.get_node("step2")

        assert "approve" in step1.next_nodes
        assert "step2" in approve.next_nodes


# =============================================================================
# Approval Gates Blocking Workflow Progression Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.hitl
class TestApprovalGatesBlockingProgression:
    """Test approval gates that block workflow progression."""

    @pytest.mark.asyncio
    async def test_approval_gate_blocks_until_approved(self):
        """Approval gate should block workflow until approved."""

        class BlockingHandler:
            def __init__(self):
                self.approval_event = asyncio.Event()
                self.should_approve = True

            async def request_human_input(self, request: HITLRequest) -> HITLResponse:
                # Wait for external signal to proceed
                await self.approval_event.wait()

                if self.should_approve:
                    return HITLResponse(
                        request_id=request.request_id,
                        status=HITLStatus.APPROVED,
                        approved=True,
                    )
                else:
                    return HITLResponse(
                        request_id=request.request_id,
                        status=HITLStatus.REJECTED,
                        approved=False,
                        reason="Rejected by test",
                    )

        handler = BlockingHandler()
        node = HITLNode(
            id="blocking_gate",
            name="Blocking Gate",
            hitl_type=HITLNodeType.APPROVAL,
            prompt="Approve to continue",
            timeout=5.0,
            fallback=HITLFallback.ABORT,
        )

        executor = HITLExecutor(handler=handler)
        workflow_continued = asyncio.Event()

        async def execute_workflow():
            response = await executor.execute_hitl_node(node, {})
            if response.approved:
                workflow_continued.set()
            return response

        # Start workflow execution
        task = asyncio.create_task(execute_workflow())

        # Verify workflow is blocked
        await asyncio.sleep(0.05)
        assert not workflow_continued.is_set()

        # Release the approval
        handler.approval_event.set()

        # Workflow should continue
        response = await task
        assert response.approved is True
        assert workflow_continued.is_set()

    @pytest.mark.asyncio
    async def test_rejection_gate_stops_workflow(self):
        """Rejection should stop workflow progression."""

        class RejectHandler:
            async def request_human_input(self, request: HITLRequest) -> HITLResponse:
                return HITLResponse(
                    request_id=request.request_id,
                    status=HITLStatus.REJECTED,
                    approved=False,
                    reason="User rejected the operation",
                )

        node = HITLNode(
            id="reject_gate",
            name="Reject Gate",
            hitl_type=HITLNodeType.APPROVAL,
            prompt="Approve dangerous operation?",
            timeout=5.0,
            fallback=HITLFallback.ABORT,
        )

        executor = HITLExecutor(handler=RejectHandler())
        response = await executor.execute_hitl_node(node, {})

        assert response.approved is False
        assert response.status == HITLStatus.REJECTED
        assert "rejected" in response.reason.lower()

    @pytest.mark.asyncio
    async def test_multiple_sequential_approval_gates(self):
        """Multiple sequential approval gates should each block separately."""

        approval_count = {"count": 0}

        class CountingHandler:
            async def request_human_input(self, request: HITLRequest) -> HITLResponse:
                approval_count["count"] += 1
                return HITLResponse(
                    request_id=request.request_id,
                    status=HITLStatus.APPROVED,
                    approved=True,
                )

        handler = CountingHandler()
        executor = HITLExecutor(handler=handler)

        # Create multiple gates
        gates = [
            HITLNode(
                id=f"gate_{i}",
                name=f"Gate {i}",
                hitl_type=HITLNodeType.APPROVAL,
                prompt=f"Approve gate {i}?",
                timeout=5.0,
            )
            for i in range(3)
        ]

        # Execute each gate sequentially
        for gate in gates:
            response = await executor.execute_hitl_node(gate, {})
            assert response.approved is True

        assert approval_count["count"] == 3


# =============================================================================
# Context Key Filtering Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.hitl
class TestContextKeyFilteringForApprovalRequests:
    """Test context key filtering for HITL approval requests."""

    def test_context_keys_filter_workflow_context(self):
        """HITLNode should filter context to specified keys."""
        node = HITLNode(
            id="filtered_context",
            name="Filtered Context Test",
            hitl_type=HITLNodeType.APPROVAL,
            prompt="Review filtered context",
            context_keys=["public_data", "summary"],
            timeout=60.0,
        )

        workflow_context = {
            "public_data": "This should be visible",
            "summary": "This should also be visible",
            "internal_state": "This should NOT be visible",
            "secrets": "This should definitely NOT be visible",
            "metadata": {"internal": True},
        }

        request = node.create_request(workflow_context)

        # Only specified keys should be in request context
        assert "public_data" in request.context
        assert "summary" in request.context
        assert "internal_state" not in request.context
        assert "secrets" not in request.context
        assert "metadata" not in request.context

        # Values should match
        assert request.context["public_data"] == "This should be visible"
        assert request.context["summary"] == "This should also be visible"

    def test_empty_context_keys_produces_empty_context(self):
        """Empty context_keys should produce empty request context."""
        node = HITLNode(
            id="no_context",
            name="No Context Test",
            hitl_type=HITLNodeType.CONFIRMATION,
            prompt="Confirm",
            context_keys=[],
            timeout=60.0,
        )

        workflow_context = {
            "data": "should not appear",
            "more_data": "also should not appear",
        }

        request = node.create_request(workflow_context)

        assert request.context == {}

    def test_missing_context_keys_ignored(self):
        """Missing context keys should be silently ignored."""
        node = HITLNode(
            id="partial_context",
            name="Partial Context Test",
            hitl_type=HITLNodeType.REVIEW,
            prompt="Review",
            context_keys=["existing_key", "missing_key", "another_missing"],
            timeout=60.0,
        )

        workflow_context = {
            "existing_key": "I exist",
            "other_key": "Not requested",
        }

        request = node.create_request(workflow_context)

        assert "existing_key" in request.context
        assert request.context["existing_key"] == "I exist"
        assert "missing_key" not in request.context
        assert "another_missing" not in request.context
        assert "other_key" not in request.context

    def test_nested_context_values_preserved(self):
        """Nested values should be preserved in filtered context."""
        node = HITLNode(
            id="nested_context",
            name="Nested Context Test",
            hitl_type=HITLNodeType.APPROVAL,
            prompt="Approve",
            context_keys=["config", "results"],
            timeout=60.0,
        )

        workflow_context = {
            "config": {
                "level1": {"level2": {"value": "deep_value"}},
                "list_data": [1, 2, 3],
            },
            "results": ["result1", "result2"],
            "private": "hidden",
        }

        request = node.create_request(workflow_context)

        assert request.context["config"]["level1"]["level2"]["value"] == "deep_value"
        assert request.context["config"]["list_data"] == [1, 2, 3]
        assert request.context["results"] == ["result1", "result2"]


# =============================================================================
# HITL Combined with Conditional Edges Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.hitl
class TestHITLCombinedWithConditionalEdges:
    """Test HITL nodes combined with conditional workflow edges."""

    def test_workflow_with_hitl_before_condition(self):
        """HITL node can precede conditional branching."""

        def router(ctx: Dict[str, Any]) -> str:
            if ctx.get("approved"):
                return "proceed"
            return "abort"

        workflow = (
            WorkflowBuilder("conditional_after_hitl")
            .add_agent("analyze", "researcher", "Analyze")
            .add_hitl_approval(
                "approve",
                "Approve to proceed?",
                next_nodes=[],  # Explicit empty since we'll use condition
            )
            .add_condition("decide", router, {"proceed": "execute", "abort": "cleanup"})
            .add_agent("execute", "executor", "Execute approved action", next_nodes=["end"])
            .add_agent("cleanup", "executor", "Cleanup on rejection", next_nodes=["end"])
            .add_agent("end", "executor", "Finalize")
            .build()
        )

        # Verify structure
        assert len(workflow.nodes) == 6
        assert workflow.get_node("decide").node_type == WorkflowNodeType.CONDITION

    def test_workflow_with_condition_before_hitl(self):
        """Conditional edge can route to different HITL nodes."""

        def severity_router(ctx: Dict[str, Any]) -> str:
            severity = ctx.get("severity", "low")
            if severity == "high":
                return "high_approval"
            elif severity == "medium":
                return "medium_approval"
            return "auto_approve"

        workflow = (
            WorkflowBuilder("hitl_routing")
            .add_agent("assess", "researcher", "Assess severity")
            .add_condition(
                "route_by_severity",
                severity_router,
                {
                    "high_approval": "manager_approve",
                    "medium_approval": "team_approve",
                    "auto_approve": "execute",
                },
            )
            .add_hitl_approval(
                "manager_approve",
                "Manager approval required for high severity",
                timeout=600.0,
                next_nodes=["execute"],
            )
            .add_hitl_approval(
                "team_approve",
                "Team approval required",
                timeout=300.0,
                next_nodes=["execute"],
            )
            .add_agent("execute", "executor", "Execute action")
            .build()
        )

        # Verify routing
        condition_node = workflow.get_node("route_by_severity")
        assert condition_node.branches["high_approval"] == "manager_approve"
        assert condition_node.branches["medium_approval"] == "team_approve"
        assert condition_node.branches["auto_approve"] == "execute"

    @pytest.mark.asyncio
    async def test_hitl_response_affects_conditional_routing(self):
        """HITL response can affect subsequent conditional routing."""

        class ModifyingHandler:
            async def request_human_input(self, request: HITLRequest) -> HITLResponse:
                # Simulate user choosing "manual" path
                return HITLResponse(
                    request_id=request.request_id,
                    status=HITLStatus.APPROVED,
                    approved=True,
                    value="manual",  # User's choice
                )

        node = HITLNode(
            id="path_choice",
            name="Path Choice",
            hitl_type=HITLNodeType.CHOICE,
            prompt="Select path:",
            choices=["automatic", "manual", "skip"],
            timeout=60.0,
        )

        executor = HITLExecutor(handler=ModifyingHandler())
        response = await executor.execute_hitl_node(node, {})

        assert response.value == "manual"

        # This value could then be used by a condition node to route
        def path_router(ctx: Dict[str, Any]) -> str:
            return ctx.get("user_choice", "automatic")

        # Simulate updating context with HITL response
        workflow_context = {"user_choice": response.value}
        next_node = path_router(workflow_context)

        assert next_node == "manual"

    def test_complex_workflow_with_multiple_hitl_and_conditions(self):
        """Complex workflow with multiple HITL nodes and conditions."""

        def review_outcome(ctx: Dict[str, Any]) -> str:
            if ctx.get("approved"):
                return "approved"
            return "rejected"

        def deploy_type(ctx: Dict[str, Any]) -> str:
            return ctx.get("deploy_target", "staging")

        workflow = (
            WorkflowBuilder("complex_hitl_workflow", "Multi-stage approval workflow")
            # Stage 1: Analysis
            .add_agent("analyze", "researcher", "Analyze changes")
            .add_hitl_review(
                "review_analysis",
                "Review analysis results",
                context_keys=["analysis_output"],
            )
            # Stage 2: Decision point
            .add_condition(
                "review_decision",
                review_outcome,
                {"approved": "prepare_deploy", "rejected": "revise"},
            )
            # Approved path
            .add_agent(
                "prepare_deploy",
                "executor",
                "Prepare deployment",
                next_nodes=["deploy_target_decision"],
            )
            .add_condition(
                "deploy_target_decision",
                deploy_type,
                {"staging": "staging_approval", "production": "production_approval"},
            )
            .add_hitl_approval(
                "staging_approval",
                "Approve staging deployment?",
                timeout=120.0,
                next_nodes=["deploy"],
            )
            .add_hitl_approval(
                "production_approval",
                "Approve PRODUCTION deployment?",
                timeout=300.0,
                next_nodes=["deploy"],
            )
            .add_agent("deploy", "executor", "Execute deployment")
            # Rejected path
            .add_agent("revise", "executor", "Revise based on feedback", next_nodes=["analyze"])
            .build()
        )

        # Verify workflow has all expected nodes (9 nodes)
        # analyze, review_analysis, review_decision, prepare_deploy,
        # deploy_target_decision, staging_approval, production_approval,
        # deploy, revise
        assert len(workflow.nodes) == 9

        # Verify HITL nodes
        assert isinstance(workflow.get_node("review_analysis"), HITLNode)
        assert isinstance(workflow.get_node("staging_approval"), HITLNode)
        assert isinstance(workflow.get_node("production_approval"), HITLNode)

        # Verify condition nodes
        assert isinstance(workflow.get_node("review_decision"), ConditionNode)
        assert isinstance(workflow.get_node("deploy_target_decision"), ConditionNode)


# =============================================================================
# HITL Node Validation Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.hitl
class TestHITLNodeValidation:
    """Test HITL node validation functionality."""

    @pytest.mark.asyncio
    async def test_validator_accepts_valid_input(self):
        """Validator should accept valid input."""

        class ValidInputHandler:
            async def request_human_input(self, request: HITLRequest) -> HITLResponse:
                return HITLResponse(
                    request_id=request.request_id,
                    status=HITLStatus.APPROVED,
                    approved=True,
                    value=100,  # Valid: between 0 and 200
                )

        def validate_range(value: Any) -> bool:
            return isinstance(value, int) and 0 <= value <= 200

        node = HITLNode(
            id="validated_input",
            name="Validated Input",
            hitl_type=HITLNodeType.INPUT,
            prompt="Enter a number between 0 and 200:",
            timeout=60.0,
            validator=validate_range,
        )

        executor = HITLExecutor(handler=ValidInputHandler())
        response = await executor.execute_hitl_node(node, {})

        assert response.status == HITLStatus.APPROVED
        assert response.value == 100

    @pytest.mark.asyncio
    async def test_validator_rejects_invalid_input(self):
        """Validator should reject invalid input."""

        class InvalidInputHandler:
            async def request_human_input(self, request: HITLRequest) -> HITLResponse:
                return HITLResponse(
                    request_id=request.request_id,
                    status=HITLStatus.APPROVED,
                    approved=True,
                    value=500,  # Invalid: outside range
                )

        def validate_range(value: Any) -> bool:
            return isinstance(value, int) and 0 <= value <= 200

        node = HITLNode(
            id="validated_input",
            name="Validated Input",
            hitl_type=HITLNodeType.INPUT,
            prompt="Enter a number between 0 and 200:",
            timeout=60.0,
            validator=validate_range,
        )

        executor = HITLExecutor(handler=InvalidInputHandler())
        response = await executor.execute_hitl_node(node, {})

        assert response.status == HITLStatus.REJECTED
        assert response.approved is False
        assert "validation failed" in response.reason.lower()

    def test_node_validate_response_method(self):
        """HITLNode.validate_response should work correctly."""

        def string_validator(value: Any) -> bool:
            return isinstance(value, str) and len(value) >= 3

        node = HITLNode(
            id="string_validation",
            name="String Validation",
            hitl_type=HITLNodeType.INPUT,
            prompt="Enter at least 3 characters:",
            validator=string_validator,
        )

        valid_response = HITLResponse(
            request_id="test",
            status=HITLStatus.APPROVED,
            approved=True,
            value="hello",
        )

        invalid_response = HITLResponse(
            request_id="test",
            status=HITLStatus.APPROVED,
            approved=True,
            value="hi",  # Too short
        )

        null_value_response = HITLResponse(
            request_id="test",
            status=HITLStatus.APPROVED,
            approved=True,
            value=None,
        )

        assert node.validate_response(valid_response) is True
        assert node.validate_response(invalid_response) is False
        assert node.validate_response(null_value_response) is True  # Null skips validation


# =============================================================================
# HITL Serialization Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.hitl
class TestHITLNodeSerialization:
    """Test HITL node serialization for workflow persistence."""

    def test_hitl_node_to_dict(self):
        """HITLNode should serialize to dictionary correctly."""
        node = HITLNode(
            id="serialize_test",
            name="Serialization Test",
            hitl_type=HITLNodeType.APPROVAL,
            prompt="Test prompt",
            context_keys=["key1", "key2"],
            timeout=120.0,
            fallback=HITLFallback.CONTINUE,
            default_value="default",
            next_nodes=["next_node"],
        )

        serialized = node.to_dict()

        assert serialized["id"] == "serialize_test"
        assert serialized["name"] == "Serialization Test"
        assert serialized["type"] == "hitl"
        assert serialized["hitl_type"] == "approval"
        assert serialized["prompt"] == "Test prompt"
        assert serialized["context_keys"] == ["key1", "key2"]
        assert serialized["timeout"] == 120.0
        assert serialized["fallback"] == "continue"
        assert serialized["default_value"] == "default"
        assert serialized["next_nodes"] == ["next_node"]
        assert serialized["_hitl_node"] is True

    def test_hitl_request_to_dict(self):
        """HITLRequest should serialize correctly."""
        node = HITLNode(
            id="request_serialize",
            name="Request Serialize",
            hitl_type=HITLNodeType.CHOICE,
            prompt="Choose:",
            choices=["A", "B", "C"],
            timeout=60.0,
            fallback=HITLFallback.ABORT,
        )

        request = node.create_request({"data": "value"})
        serialized = request.to_dict()

        assert "request_id" in serialized
        assert serialized["node_id"] == "request_serialize"
        assert serialized["hitl_type"] == "choice"
        assert serialized["prompt"] == "Choose:"
        assert serialized["choices"] == ["A", "B", "C"]
        assert serialized["timeout"] == 60.0
        assert serialized["fallback"] == "abort"
        assert "created_at" in serialized

    def test_hitl_response_to_dict(self):
        """HITLResponse should serialize correctly."""
        response = HITLResponse(
            request_id="req_123",
            status=HITLStatus.APPROVED,
            approved=True,
            value="selected_option",
            reason="User selected option",
        )

        serialized = response.to_dict()

        assert serialized["request_id"] == "req_123"
        assert serialized["status"] == "approved"
        assert serialized["approved"] is True
        assert serialized["value"] == "selected_option"
        assert serialized["reason"] == "User selected option"
        assert "responded_at" in serialized
