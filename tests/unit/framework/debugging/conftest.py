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

"""Shared fixtures for debugging module tests."""

import pytest

from victor.framework.debugging.breakpoints import (
    BreakpointManager,
    BreakpointStorage,
    BreakpointType,
    BreakpointPosition,
    WorkflowBreakpoint,
)
from victor.framework.debugging.execution import (
    ExecutionController,
    ExecutionState,
    StepMode,
)
from victor.framework.debugging.inspector import StateInspector
from victor.framework.debugging.session import DebugSession, DebugSessionConfig
from victor.core.events import ObservabilityBus


@pytest.fixture
def event_bus():
    """ObservabilityBus instance for testing."""
    return ObservabilityBus()


@pytest.fixture
def breakpoint_storage():
    """BreakpointStorage instance."""
    return BreakpointStorage()


@pytest.fixture
def breakpoint_manager(event_bus):
    """BreakpointManager instance."""
    return BreakpointManager(event_bus)


@pytest.fixture
def execution_controller():
    """ExecutionController instance."""
    return ExecutionController(session_id="test-session")


@pytest.fixture
def state_inspector():
    """StateInspector instance."""
    return StateInspector()


@pytest.fixture
def sample_state():
    """Sample workflow state for testing."""
    return {
        "task": "Analyze code",
        "file_path": "/tmp/test.py",
        "errors": 0,
        "results": [],
        "metadata": {"iteration": 1},
    }


@pytest.fixture
def debug_session(event_bus):
    """DebugSession instance."""
    config = DebugSessionConfig(
        session_id="test-session",
        workflow_id="test-workflow",
    )
    return DebugSession(config=config, event_bus=event_bus)


@pytest.fixture
def sample_workflow():
    """Create sample StateGraph for testing."""
    from victor.framework.graph import StateGraph

    graph = StateGraph()

    async def node_analyze(state):
        state["analysis"] = "complete"
        return state

    async def node_process(state):
        state["errors"] = state.get("errors", 0) + 1
        return state

    graph.add_node("analyze", node_analyze)
    graph.add_node("process", node_process)
    graph.add_edge("analyze", "process")
    graph.set_entry_point("analyze")

    return graph.compile()
