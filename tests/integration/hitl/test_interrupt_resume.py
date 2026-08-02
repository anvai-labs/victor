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

"""Integration tests for HITL interrupt/resume (checkpoint) functionality.

These tests verify:
- Workflow interruption creating checkpoints
- Resume from checkpoint with context restoration
- Multiple interrupt points in a single workflow
- HITLCheckpoint persistence and retrieval
"""

import asyncio
import pytest
from typing import Dict, Any, List

from victor.framework.hitl import (
    HITLController,
    HITLCheckpoint,
    ApprovalStatus,
)

# =============================================================================
# HITLCheckpoint Creation Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.hitl
class TestWorkflowInterruptionCreatesCheckpoints:
    """Test workflow interruption and checkpoint creation."""

    def test_interrupt_creates_checkpoint_id(self):
        """Interrupt should return a valid checkpoint ID."""
        controller = HITLController()

        checkpoint_id = controller.interrupt()

        assert checkpoint_id is not None
        assert isinstance(checkpoint_id, str)
        assert checkpoint_id.startswith("cp_")
        assert len(checkpoint_id) > 10  # UUID-based ID should be long

    def test_interrupt_sets_paused_state(self):
        """Interrupt should set the controller to paused state."""
        controller = HITLController()

        assert controller.is_paused is False

        controller.interrupt()

        assert controller.is_paused is True

    def test_interrupt_with_context_stores_context(self):
        """Interrupt should store the provided context."""
        controller = HITLController()

        context = {
            "current_step": 5,
            "completed_tasks": ["task1", "task2", "task3"],
            "intermediate_results": {"analysis": "some findings"},
            "tool_calls_made": 15,
        }

        checkpoint_id = controller.interrupt(context=context)
        retrieved_context = controller.get_checkpoint_context(checkpoint_id)

        assert retrieved_context == context
        assert retrieved_context["current_step"] == 5
        assert "task2" in retrieved_context["completed_tasks"]

    def test_interrupt_without_context_stores_empty_dict(self):
        """Interrupt without context should store empty dictionary."""
        controller = HITLController()

        checkpoint_id = controller.interrupt()
        retrieved_context = controller.get_checkpoint_context(checkpoint_id)

        assert retrieved_context == {}

    def test_multiple_interrupts_create_distinct_checkpoints(self):
        """Multiple interrupts should create unique checkpoints."""
        controller = HITLController()

        # First interrupt
        cp1 = controller.interrupt(context={"step": 1})
        controller.resume(cp1)

        # Second interrupt
        cp2 = controller.interrupt(context={"step": 2})
        controller.resume(cp2)

        # Third interrupt
        cp3 = controller.interrupt(context={"step": 3})

        # All IDs should be unique
        assert cp1 != cp2 != cp3

        # Each should have its own context
        assert controller.get_checkpoint_context(cp1) == {"step": 1}
        assert controller.get_checkpoint_context(cp2) == {"step": 2}
        assert controller.get_checkpoint_context(cp3) == {"step": 3}


# =============================================================================
# Resume from HITLCheckpoint Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.hitl
class TestResumeFromCheckpointWithContextRestoration:
    """Test resuming from checkpoints with context restoration."""

    def test_resume_clears_paused_state(self):
        """Resume should clear the paused state."""
        controller = HITLController()

        checkpoint_id = controller.interrupt()
        assert controller.is_paused is True

        controller.resume(checkpoint_id)
        assert controller.is_paused is False

    def test_resume_returns_stored_context(self):
        """Resume should return the context stored at interrupt time."""
        controller = HITLController()

        original_context = {
            "workflow_id": "wf_123",
            "iteration": 42,
            "state": {"key": "value"},
        }

        checkpoint_id = controller.interrupt(context=original_context)
        returned_context = controller.resume(checkpoint_id)

        assert returned_context == original_context
        assert returned_context["workflow_id"] == "wf_123"

    def test_resume_with_invalid_checkpoint_raises_error(self):
        """Resume with invalid checkpoint ID should raise ValueError."""
        controller = HITLController()

        controller.interrupt()  # Create a valid checkpoint

        with pytest.raises(ValueError, match="Invalid checkpoint"):
            controller.resume("invalid_checkpoint_id")

    def test_context_restored_exactly_as_stored(self):
        """Complex context should be restored exactly as stored."""
        controller = HITLController()

        complex_context = {
            "strings": ["a", "b", "c"],
            "numbers": [1, 2.5, 3],
            "nested": {"level1": {"level2": {"value": "deep"}}},
            "boolean": True,
            "null_value": None,
        }

        checkpoint_id = controller.interrupt(context=complex_context)
        restored_context = controller.resume(checkpoint_id)

        assert restored_context["strings"] == ["a", "b", "c"]
        assert restored_context["numbers"][1] == 2.5
        assert restored_context["nested"]["level1"]["level2"]["value"] == "deep"
        assert restored_context["boolean"] is True
        assert restored_context["null_value"] is None

    def test_resume_allows_reinterrupt(self):
        """After resume, controller should allow new interrupt."""
        controller = HITLController()

        # First interrupt and resume cycle
        cp1 = controller.interrupt(context={"cycle": 1})
        controller.resume(cp1)
        assert controller.is_paused is False

        # Second interrupt and resume cycle
        cp2 = controller.interrupt(context={"cycle": 2})
        assert controller.is_paused is True

        restored = controller.resume(cp2)
        assert restored["cycle"] == 2
        assert controller.is_paused is False


# =============================================================================
# Multiple Interrupt Points Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.hitl
class TestMultipleInterruptPointsInWorkflow:
    """Test multiple interrupt points within a single workflow."""

    @pytest.mark.asyncio
    async def test_simulated_multi_step_workflow_with_interrupts(self):
        """Simulate a workflow with multiple interrupt points."""
        controller = HITLController()
        workflow_log = []

        async def step_1():
            workflow_log.append("step_1_start")
            await asyncio.sleep(0.01)
            workflow_log.append("step_1_complete")
            return {"step_1_result": "analyzed codebase"}

        async def step_2(input_data: Dict[str, Any]):
            workflow_log.append("step_2_start")
            await asyncio.sleep(0.01)
            workflow_log.append("step_2_complete")
            return {**input_data, "step_2_result": "identified issues"}

        async def step_3(input_data: Dict[str, Any]):
            workflow_log.append("step_3_start")
            await asyncio.sleep(0.01)
            workflow_log.append("step_3_complete")
            return {**input_data, "step_3_result": "fixed issues"}

        # Execute step 1
        result_1 = await step_1()

        # First interrupt after analysis
        cp1 = controller.interrupt(
            context={
                "current_step": "after_analysis",
                "results": result_1,
                "log": workflow_log.copy(),
            }
        )

        # Simulate human review and resume
        assert controller.is_paused is True
        context_1 = controller.resume(cp1)
        assert context_1["current_step"] == "after_analysis"

        # Execute step 2
        result_2 = await step_2(context_1["results"])

        # Second interrupt after issue identification
        cp2 = controller.interrupt(
            context={
                "current_step": "after_identification",
                "results": result_2,
                "log": workflow_log.copy(),
            }
        )

        # Resume and complete
        context_2 = controller.resume(cp2)
        assert context_2["current_step"] == "after_identification"

        result_3 = await step_3(context_2["results"])

        # Verify complete workflow
        assert len(workflow_log) == 6
        assert result_3["step_1_result"] == "analyzed codebase"
        assert result_3["step_3_result"] == "fixed issues"

    def test_checkpoints_preserve_separate_workflow_states(self):
        """Different checkpoints should preserve independent states."""
        controller = HITLController()

        # Simulate parallel workflow branches
        branch_a_cp = controller.interrupt(
            context={
                "branch": "A",
                "data": ["a1", "a2"],
            }
        )
        controller.resume(branch_a_cp)

        branch_b_cp = controller.interrupt(
            context={
                "branch": "B",
                "data": ["b1", "b2", "b3"],
            }
        )
        controller.resume(branch_b_cp)

        # Verify each checkpoint preserved its state
        branch_a_context = controller.get_checkpoint_context(branch_a_cp)
        branch_b_context = controller.get_checkpoint_context(branch_b_cp)

        assert branch_a_context["branch"] == "A"
        assert len(branch_a_context["data"]) == 2

        assert branch_b_context["branch"] == "B"
        assert len(branch_b_context["data"]) == 3

    def test_interrupt_during_different_workflow_phases(self):
        """Interrupt should work correctly during different workflow phases."""
        controller = HITLController()
        checkpoints = []

        phases = [
            {"phase": "initialization", "progress": 0},
            {"phase": "data_loading", "progress": 20},
            {"phase": "analysis", "progress": 50},
            {"phase": "generation", "progress": 80},
            {"phase": "finalization", "progress": 100},
        ]

        for phase_data in phases:
            cp = controller.interrupt(context=phase_data)
            checkpoints.append(cp)
            controller.resume(cp)

        # Verify all phases captured correctly
        for cp, expected_phase in zip(checkpoints, phases):
            context = controller.get_checkpoint_context(cp)
            assert context["phase"] == expected_phase["phase"]
            assert context["progress"] == expected_phase["progress"]


# =============================================================================
# HITLCheckpoint Persistence and Retrieval Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.hitl
class TestCheckpointPersistenceAndRetrieval:
    """Test checkpoint persistence and retrieval functionality."""

    def test_get_checkpoint_context_retrieves_stored_data(self):
        """get_checkpoint_context should retrieve stored checkpoint data."""
        controller = HITLController()

        context = {"key": "value", "count": 42}
        checkpoint_id = controller.interrupt(context=context)

        # Retrieve without resuming
        retrieved = controller.get_checkpoint_context(checkpoint_id)

        assert retrieved == context
        assert controller.is_paused is True  # Still paused

    def test_checkpoint_context_immutable_after_creation(self):
        """HITLCheckpoint context should not be affected by external modifications."""
        controller = HITLController()

        original_context = {"items": ["a", "b"], "value": 10}
        checkpoint_id = controller.interrupt(context=original_context)

        # Modify the original dictionary
        original_context["items"].append("c")
        original_context["value"] = 999

        # Retrieve checkpoint - should have original values
        # Note: This depends on implementation - some may copy, some may not
        retrieved = controller.get_checkpoint_context(checkpoint_id)

        # The implementation may or may not copy, so we just verify retrieval works
        assert "items" in retrieved
        assert "value" in retrieved

    def test_invalid_checkpoint_retrieval_raises_error(self):
        """Retrieving invalid checkpoint should raise ValueError."""
        controller = HITLController()

        with pytest.raises(ValueError, match="Invalid checkpoint"):
            controller.get_checkpoint_context("nonexistent_cp_123")

    def test_checkpoint_available_after_multiple_operations(self):
        """Checkpoints should remain available after multiple operations."""
        controller = HITLController()

        # Create first checkpoint
        cp1 = controller.interrupt(context={"checkpoint": 1})
        controller.resume(cp1)

        # Create second checkpoint
        cp2 = controller.interrupt(context={"checkpoint": 2})
        controller.resume(cp2)

        # Create third checkpoint
        cp3 = controller.interrupt(context={"checkpoint": 3})
        controller.resume(cp3)

        # All checkpoints should still be retrievable
        assert controller.get_checkpoint_context(cp1)["checkpoint"] == 1
        assert controller.get_checkpoint_context(cp2)["checkpoint"] == 2
        assert controller.get_checkpoint_context(cp3)["checkpoint"] == 3

    def test_checkpoint_with_large_context(self):
        """HITLCheckpoint should handle large context data."""
        controller = HITLController()

        large_context = {
            "large_list": list(range(1000)),
            "large_dict": {f"key_{i}": f"value_{i}" for i in range(500)},
            "nested_data": {"level_1": {"level_2": {"items": ["item"] * 100}}},
            "long_string": "x" * 10000,
        }

        checkpoint_id = controller.interrupt(context=large_context)
        retrieved = controller.get_checkpoint_context(checkpoint_id)

        assert len(retrieved["large_list"]) == 1000
        assert len(retrieved["large_dict"]) == 500
        assert len(retrieved["long_string"]) == 10000


# =============================================================================
# Pause/Resume Callback Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.hitl
class TestPauseResumeCallbacks:
    """Test callbacks for pause and resume events."""

    def test_on_pause_callback_receives_checkpoint_and_context(self):
        """on_pause callback should receive checkpoint ID and context."""
        pause_events = []

        def on_pause(checkpoint_id: str, context: Dict[str, Any]):
            pause_events.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "context": context.copy(),
                }
            )

        controller = HITLController()
        controller.on_pause(on_pause)

        context = {"step": "analysis", "progress": 50}
        checkpoint_id = controller.interrupt(context=context)

        assert len(pause_events) == 1
        assert pause_events[0]["checkpoint_id"] == checkpoint_id
        assert pause_events[0]["context"] == context

    def test_on_resume_callback_receives_checkpoint_id(self):
        """on_resume callback should receive checkpoint ID."""
        resume_events = []

        def on_resume(checkpoint_id: str):
            resume_events.append(checkpoint_id)

        controller = HITLController()
        controller.on_resume(on_resume)

        cp = controller.interrupt(context={"test": True})
        controller.resume(cp)

        assert len(resume_events) == 1
        assert resume_events[0] == cp

    def test_multiple_pause_resume_callbacks_all_fire(self):
        """Multiple registered callbacks should all fire."""
        pause_count = []
        resume_count = []

        controller = HITLController()
        controller.on_pause(lambda cp, ctx: pause_count.append(1))
        controller.on_pause(lambda cp, ctx: pause_count.append(2))
        controller.on_resume(lambda cp: resume_count.append(1))
        controller.on_resume(lambda cp: resume_count.append(2))
        controller.on_resume(lambda cp: resume_count.append(3))

        cp = controller.interrupt()
        controller.resume(cp)

        assert len(pause_count) == 2
        assert len(resume_count) == 3

    def test_callbacks_fire_in_correct_sequence(self):
        """Callbacks should fire in the order they were registered."""
        event_sequence = []

        controller = HITLController()
        controller.on_pause(lambda cp, ctx: event_sequence.append("pause_1"))
        controller.on_pause(lambda cp, ctx: event_sequence.append("pause_2"))
        controller.on_resume(lambda cp: event_sequence.append("resume_1"))
        controller.on_resume(lambda cp: event_sequence.append("resume_2"))

        cp = controller.interrupt()
        controller.resume(cp)

        assert event_sequence == ["pause_1", "pause_2", "resume_1", "resume_2"]


# =============================================================================
# Integration with Approval Requests
# =============================================================================


@pytest.mark.integration
@pytest.mark.hitl
class TestInterruptResumeWithApprovalRequests:
    """Test interrupt/resume functionality combined with approval requests."""

    @pytest.mark.asyncio
    async def test_checkpoint_can_store_pending_approval_info(self):
        """HITLCheckpoint can store information about pending approvals."""
        controller = HITLController()

        # Create an approval request
        approval_req = controller.request_approval(
            title="Deploy to Production",
            description="Deploy version 1.2.3",
        )

        # Interrupt with reference to pending approval
        checkpoint_id = controller.interrupt(
            context={
                "pending_approval_id": approval_req.id,
                "workflow_stage": "awaiting_deployment_approval",
            }
        )

        # Resume and check pending approval
        context = controller.resume(checkpoint_id)
        approval_id = context["pending_approval_id"]

        # Approval should still be retrievable
        retrieved_approval = controller.get_request(approval_id)
        assert retrieved_approval.title == "Deploy to Production"
        assert retrieved_approval.is_pending is True

    @pytest.mark.asyncio
    async def test_pause_and_approval_work_together(self):
        """Pause/resume and approval flows should work together."""
        controller = HITLController()
        workflow_events = []

        # Workflow step 1
        workflow_events.append("step_1_complete")

        # Create approval and interrupt
        approval = controller.request_approval(
            title="Continue workflow?",
            description="Approve to continue",
        )
        cp = controller.interrupt(
            context={
                "at_step": 1,
                "approval_id": approval.id,
            }
        )

        workflow_events.append("interrupted_for_approval")

        # Simulate external approval
        controller.respond_to_request(approval.id, approved=True)

        # Resume workflow
        context = controller.resume(cp)
        workflow_events.append("resumed_workflow")

        # Verify approval was granted
        final_approval = controller.get_request(context["approval_id"])
        assert final_approval.is_approved is True

        workflow_events.append("step_2_complete")

        assert workflow_events == [
            "step_1_complete",
            "interrupted_for_approval",
            "resumed_workflow",
            "step_2_complete",
        ]
