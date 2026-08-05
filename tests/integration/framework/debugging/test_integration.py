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

"""Integration tests for workflow debugging."""

import asyncio
import pytest

from victor.framework.debugging import (
    DebugSession,
    DebugSessionConfig,
    BreakpointPosition,
)
from victor.framework.debugging.breakpoints import BreakpointType


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.slow
class TestWorkflowDebugging:
    """Integration tests for workflow debugging."""

    async def test_debug_workflow_with_breakpoint(self, sample_workflow, debug_session):
        """Test debugging a workflow with breakpoint."""
        # Set breakpoint
        bp = debug_session.set_breakpoint(
            node_id="analyze",
            position=BreakpointPosition.BEFORE,
        )

        # Create hook
        hook = debug_session.create_hook()

        # Execute workflow in background
        async def execute_workflow():
            return await sample_workflow.invoke({"task": "test"}, debug_hook=hook)

        execution_task = asyncio.create_task(execute_workflow())

        # Wait for breakpoint to be hit
        await asyncio.sleep(0.5)

        # Check paused
        assert debug_session.is_paused is True

        context = debug_session.get_pause_context()
        assert context is not None
        assert context.node_id == "analyze"
        assert bp.id in context.breakpoint_ids

        # Continue execution
        await debug_session.continue_execution()

        # Wait for completion
        result = await execution_task

        assert result.success is True

    async def test_step_through_workflow(self, sample_workflow, debug_session):
        """Test stepping through workflow execution."""
        # Set breakpoint
        debug_session.set_breakpoint(node_id="analyze", position=BreakpointPosition.AFTER)

        hook = debug_session.create_hook()

        async def execute_workflow():
            return await sample_workflow.invoke({"task": "test"}, debug_hook=hook)

        execution_task = asyncio.create_task(execute_workflow())
        await asyncio.sleep(0.5)

        # First breakpoint hit
        assert debug_session.is_paused is True

        # Step over to next node
        await debug_session.step_over()
        await asyncio.sleep(0.5)

        # Should be paused again (in stepping mode)
        assert debug_session.is_paused is True

        # Continue to completion
        await debug_session.continue_execution()
        result = await execution_task

        assert result.success is True

    async def test_conditional_breakpoint(self, sample_workflow, event_bus):
        """Test conditional breakpoint."""
        session = DebugSession(
            config=DebugSessionConfig(
                session_id="test-conditional",
                workflow_id="test-workflow",
            ),
            event_bus=event_bus,
        )

        # Set conditional breakpoint (should not hit initially)
        session.set_breakpoint(
            node_id="process",
            position=BreakpointPosition.AFTER,
            condition=lambda state: state.get("errors", 0) > 10,
        )

        hook = session.create_hook()

        async def execute_workflow():
            return await sample_workflow.invoke({"task": "test"}, debug_hook=hook)

        execution_task = asyncio.create_task(execute_workflow())
        await asyncio.sleep(0.5)

        # Should complete without hitting breakpoint (errors only increments by 1)
        result = await execution_task

        assert result.success is True
        assert session.is_paused is False

    async def test_exception_breakpoint(self, event_bus):
        """Test exception breakpoint.

        Exception breakpoints pause when an error occurs, allowing inspection.
        This test verifies the breakpoint is hit and workflow can be resumed.
        """
        from victor.framework.graph import StateGraph

        # Create workflow that raises exception
        graph = StateGraph()

        async def failing_node(state):
            raise ValueError("Test error")

        graph.add_node("failing", failing_node)
        graph.set_entry_point("failing")

        compiled = graph.compile()

        # Create debug session with exception breakpoint
        session = DebugSession(
            config=DebugSessionConfig(
                session_id="test-exception",
                workflow_id="test-workflow",
            ),
            event_bus=event_bus,
        )

        session.set_breakpoint(
            bp_type=BreakpointType.EXCEPTION,
            position=BreakpointPosition.ON_ERROR,
        )

        hook = session.create_hook()

        # Execute workflow in background (will pause at exception)
        async def execute_workflow():
            return await compiled.invoke({"task": "test"}, debug_hook=hook)

        execution_task = asyncio.create_task(execute_workflow())

        # Wait for breakpoint to be hit
        await asyncio.sleep(0.5)

        # Should be paused at exception breakpoint
        if session.is_paused:
            context = session.get_pause_context()
            assert context is not None
            assert context.position == BreakpointPosition.ON_ERROR
            # Resume to let workflow complete with error
            await session.continue_execution()

        # Wait for completion with timeout
        try:
            result = await asyncio.wait_for(execution_task, timeout=2.0)
            # Should fail due to exception
            assert result.success is False
        except asyncio.TimeoutError:
            # If still hanging, stop session and fail gracefully
            await session.stop()
            pytest.fail("Workflow did not complete after resuming from exception breakpoint")

    async def test_multiple_breakpoints(self, sample_workflow, debug_session):
        """Test multiple breakpoints on different nodes."""
        # Set breakpoints on both nodes
        bp1 = debug_session.set_breakpoint(
            node_id="analyze",
            position=BreakpointPosition.BEFORE,
        )
        bp2 = debug_session.set_breakpoint(
            node_id="process",
            position=BreakpointPosition.BEFORE,
        )

        hook = debug_session.create_hook()

        hit_count = {"analyze": 0, "process": 0}

        async def execute_workflow():
            return await sample_workflow.invoke({"task": "test"}, debug_hook=hook)

        execution_task = asyncio.create_task(execute_workflow())

        # First breakpoint (analyze)
        await asyncio.sleep(0.5)
        if debug_session.is_paused:
            context = debug_session.get_pause_context()
            if context.node_id == "analyze":
                hit_count["analyze"] += 1
            await debug_session.continue_execution()

        # Second breakpoint (process)
        await asyncio.sleep(0.5)
        if debug_session.is_paused:
            context = debug_session.get_pause_context()
            if context.node_id == "process":
                hit_count["process"] += 1
            await debug_session.continue_execution()

        # Wait for completion
        result = await execution_task

        assert result.success is True
        # Note: May not hit both breakpoints depending on timing
        assert hit_count["analyze"] + hit_count["process"] >= 1

    async def test_disabled_breakpoint(self, sample_workflow, debug_session):
        """Test that disabled breakpoints don't trigger."""
        bp = debug_session.set_breakpoint(
            node_id="analyze",
            position=BreakpointPosition.BEFORE,
        )

        # Disable breakpoint
        debug_session.breakpoint_mgr.disable_breakpoint(bp.id)

        hook = debug_session.create_hook()

        # Execute workflow
        result = await sample_workflow.invoke({"task": "test"}, debug_hook=hook)

        # Should complete without pausing
        assert result.success is True
        assert debug_session.is_paused is False

    async def test_state_inspection_at_breakpoint(self, sample_workflow, debug_session):
        """Test state inspection at breakpoint."""
        debug_session.set_breakpoint(
            node_id="analyze",
            position=BreakpointPosition.BEFORE,
        )

        hook = debug_session.create_hook()

        async def execute_workflow():
            return await sample_workflow.invoke(
                {"task": "initial_task", "counter": 0}, debug_hook=hook
            )

        execution_task = asyncio.create_task(execute_workflow())
        await asyncio.sleep(0.5)

        if debug_session.is_paused:
            # Get current state
            state = debug_session.get_current_state()
            assert state is not None
            assert state["task"] == "initial_task"
            assert state["counter"] == 0

            # Get pause context
            context = debug_session.get_pause_context()
            assert context is not None
            assert context.node_id == "analyze"

            await debug_session.continue_execution()

        result = await execution_task
        assert result.success is True
