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

"""Unit tests for execution control."""

import asyncio
import pytest

from victor.framework.debugging.execution import (
    ExecutionController,
    ExecutionState,
    StepMode,
    PauseContext,
)
from victor.framework.debugging.breakpoints import (
    BreakpointPosition,
    BreakpointType,
    WorkflowBreakpoint,
)


@pytest.mark.unit
class TestExecutionController:
    """Test ExecutionController."""

    def test_initial_state(self, execution_controller):
        """Test initial execution state."""
        assert execution_controller.state == ExecutionState.RUNNING
        assert execution_controller.pause_context is None

    async def test_pause_and_continue(self, execution_controller, sample_state):
        """Test pause and continue flow."""
        assert execution_controller.state == ExecutionState.RUNNING

        # Pause in background task
        async def pause_task():
            await execution_controller.pause(
                node_id="analyze",
                state=sample_state,
                position=BreakpointPosition.BEFORE,
                breakpoints=[],
            )

        pause_task = asyncio.create_task(pause_task())

        # Wait a bit for pause to occur
        await asyncio.sleep(0.1)

        assert execution_controller.state == ExecutionState.PAUSED

        # Resume
        await execution_controller.continue_execution()

        # Wait for pause to complete
        await pause_task

        assert execution_controller.state == ExecutionState.RUNNING

    async def test_step_over(self, execution_controller, sample_state):
        """Test step over command."""

        # Pause first
        async def pause_task():
            await execution_controller.pause(
                node_id="analyze",
                state=sample_state,
                position=BreakpointPosition.BEFORE,
                breakpoints=[],
            )

        pause_task = asyncio.create_task(pause_task())
        await asyncio.sleep(0.1)

        # Step over
        await execution_controller.step_over()

        assert execution_controller.state == ExecutionState.STEPPING
        assert execution_controller._step_mode == StepMode.STEP_OVER

        # Let pause complete
        await pause_task

    async def test_step_into(self, execution_controller, sample_state):
        """Test step into command."""

        # Pause first
        async def pause_task():
            await execution_controller.pause(
                node_id="analyze",
                state=sample_state,
                position=BreakpointPosition.BEFORE,
                breakpoints=[],
            )

        pause_task = asyncio.create_task(pause_task())
        await asyncio.sleep(0.1)

        # Step into
        await execution_controller.step_into()

        assert execution_controller.state == ExecutionState.STEPPING
        assert execution_controller._step_mode == StepMode.STEP_INTO

        # Let pause complete
        await pause_task

    async def test_step_out(self, execution_controller, sample_state):
        """Test step out command."""

        # Pause first
        async def pause_task():
            await execution_controller.pause(
                node_id="analyze",
                state=sample_state,
                position=BreakpointPosition.BEFORE,
                breakpoints=[],
            )

        pause_task = asyncio.create_task(pause_task())
        await asyncio.sleep(0.1)

        # Step out
        await execution_controller.step_out()

        assert execution_controller.state == ExecutionState.STEPPING
        assert execution_controller._step_mode == StepMode.STEP_OUT

        # Let pause complete
        await pause_task

    async def test_should_pause_with_breakpoints(self, execution_controller, sample_state):
        """Test should_pause with breakpoints."""
        bp = WorkflowBreakpoint(
            id="bp-1",
            type=BreakpointType.NODE,
            position=BreakpointPosition.BEFORE,
            node_id="analyze",
        )

        should = execution_controller.should_pause(
            node_id="analyze",
            state=sample_state,
            breakpoints=[bp],
        )

        assert should is True

    async def test_should_pause_without_breakpoints(self, execution_controller, sample_state):
        """Test should_pause without breakpoints."""
        should = execution_controller.should_pause(
            node_id="analyze",
            state=sample_state,
            breakpoints=[],
        )

        assert should is False

    async def test_pause_immediately(self, execution_controller):
        """Test pause_immediately sets state to PAUSED."""
        await execution_controller.pause_immediately()

        assert execution_controller.state == ExecutionState.PAUSED

    def test_terminate(self, execution_controller):
        """Test terminate sets state to TERMINATED."""
        execution_controller.terminate()

        assert execution_controller.state == ExecutionState.TERMINATED

    async def test_pause_context_capture(self, execution_controller, sample_state):
        """Test pause context captures state correctly."""
        bp = WorkflowBreakpoint(
            id="bp-1",
            type=BreakpointType.NODE,
            position=BreakpointPosition.BEFORE,
            node_id="analyze",
        )

        async def pause_task():
            return await execution_controller.pause(
                node_id="analyze",
                state=sample_state,
                position=BreakpointPosition.BEFORE,
                breakpoints=[bp],
            )

        pause_task = asyncio.create_task(pause_task())
        await asyncio.sleep(0.1)

        context = execution_controller.pause_context

        assert context is not None
        assert context.node_id == "analyze"
        assert context.position == BreakpointPosition.BEFORE
        assert context.state == sample_state
        assert bp.id in context.breakpoint_ids

        # Resume to complete
        await execution_controller.continue_execution()
        await pause_task

    async def test_continue_when_not_paused(self, execution_controller):
        """Test continue when not paused does nothing."""
        # Should not raise exception
        await execution_controller.continue_execution()

        assert execution_controller.state == ExecutionState.RUNNING

    async def test_step_commands_when_not_paused(self, execution_controller):
        """Test step commands when not paused do nothing."""
        await execution_controller.step_over()
        await execution_controller.step_into()
        await execution_controller.step_out()

        # All should no-op since not paused
        assert execution_controller.state == ExecutionState.RUNNING


@pytest.mark.unit
class TestPauseContext:
    """Test PauseContext dataclass."""

    def test_pause_context_creation(self, sample_state):
        """Test creating pause context."""
        context = PauseContext(
            session_id="test-session",
            node_id="analyze",
            position=BreakpointPosition.BEFORE,
            state=sample_state,
            breakpoint_ids=["bp-1"],
            timestamp=1234567890.0,
        )

        assert context.session_id == "test-session"
        assert context.node_id == "analyze"
        assert context.position == BreakpointPosition.BEFORE
        assert context.state == sample_state
        assert context.breakpoint_ids == ["bp-1"]
        assert context.timestamp == 1234567890.0

    def test_pause_context_with_error(self, sample_state):
        """Test pause context with error."""
        error = Exception("Test error")

        context = PauseContext(
            session_id="test-session",
            node_id="analyze",
            position=BreakpointPosition.ON_ERROR,
            state=sample_state,
            breakpoint_ids=["bp-1"],
            timestamp=1234567890.0,
            error=error,
        )

        assert context.error == error

    def test_pause_context_to_dict(self, sample_state):
        """Test pause context serialization."""
        context = PauseContext(
            session_id="test-session",
            node_id="analyze",
            position=BreakpointPosition.BEFORE,
            state=sample_state,
            breakpoint_ids=["bp-1"],
            timestamp=1234567890.0,
        )

        data = context.to_dict()

        assert data["session_id"] == "test-session"
        assert data["node_id"] == "analyze"
        assert data["position"] == "before"
        assert "state_keys" in data
        assert data["breakpoint_ids"] == ["bp-1"]
