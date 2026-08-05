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

"""Unit tests for breakpoint management."""

import pytest

from victor.framework.debugging.breakpoints import (
    BreakpointManager,
    BreakpointPosition,
    BreakpointStorage,
    BreakpointType,
    WorkflowBreakpoint,
)


@pytest.mark.unit
class TestWorkflowBreakpoint:
    """Test WorkflowBreakpoint dataclass."""

    def test_node_breakpoint_creation(self):
        """Test creating a node breakpoint."""
        bp = WorkflowBreakpoint(
            id="bp-1",
            type=BreakpointType.NODE,
            position=BreakpointPosition.BEFORE,
            node_id="analyze",
        )

        assert bp.id == "bp-1"
        assert bp.type == BreakpointType.NODE
        assert bp.position == BreakpointPosition.BEFORE
        assert bp.node_id == "analyze"
        assert bp.enabled is True
        assert bp.hit_count == 0

    def test_conditional_breakpoint_creation(self):
        """Test creating a conditional breakpoint."""

        def condition(state):
            return state.get("errors", 0) > 5

        bp = WorkflowBreakpoint(
            id="bp-2",
            type=BreakpointType.CONDITIONAL,
            position=BreakpointPosition.AFTER,
            node_id="process",
            condition=condition,
        )

        assert bp.type == BreakpointType.CONDITIONAL
        assert bp.condition == condition

    def test_should_hit_node_breakpoint(self, sample_state):
        """Test node breakpoint hit detection."""
        bp = WorkflowBreakpoint(
            id="bp-1",
            type=BreakpointType.NODE,
            position=BreakpointPosition.BEFORE,
            node_id="analyze",
        )

        assert bp.should_hit(sample_state, "analyze") is True
        assert bp.should_hit(sample_state, "process") is False

    def test_should_hit_conditional_breakpoint(self, sample_state):
        """Test conditional breakpoint hit detection."""
        bp = WorkflowBreakpoint(
            id="bp-2",
            type=BreakpointType.CONDITIONAL,
            position=BreakpointPosition.AFTER,
            node_id="process",
            condition=lambda state: state.get("errors", 0) > 5,
        )

        assert bp.should_hit(sample_state, "process") is False

        sample_state["errors"] = 10
        assert bp.should_hit(sample_state, "process") is True

    def test_should_hit_exception_breakpoint(self, sample_state):
        """Test exception breakpoint hit detection."""
        bp = WorkflowBreakpoint(
            id="bp-3",
            type=BreakpointType.EXCEPTION,
            position=BreakpointPosition.ON_ERROR,
        )

        assert bp.should_hit(sample_state, "analyze") is False

        error = Exception("Test error")
        assert bp.should_hit(sample_state, "analyze", error) is True

    def test_ignore_count(self, sample_state):
        """Test breakpoint ignore count."""
        bp = WorkflowBreakpoint(
            id="bp-4",
            type=BreakpointType.NODE,
            position=BreakpointPosition.BEFORE,
            node_id="analyze",
            ignore_count=2,
        )

        # First two hits should be ignored
        assert bp.should_hit(sample_state, "analyze") is False
        assert bp.hit_count == 1

        assert bp.should_hit(sample_state, "analyze") is False
        assert bp.hit_count == 2

        # Third hit should trigger
        assert bp.should_hit(sample_state, "analyze") is True
        assert bp.hit_count == 3

    def test_disabled_breakpoint(self, sample_state):
        """Test disabled breakpoint doesn't hit."""
        bp = WorkflowBreakpoint(
            id="bp-5",
            type=BreakpointType.NODE,
            position=BreakpointPosition.BEFORE,
            node_id="analyze",
            enabled=False,
        )

        assert bp.should_hit(sample_state, "analyze") is False

    def test_to_dict(self):
        """Test breakpoint serialization."""
        bp = WorkflowBreakpoint(
            id="bp-1",
            type=BreakpointType.NODE,
            position=BreakpointPosition.BEFORE,
            node_id="analyze",
        )

        data = bp.to_dict()

        assert data["id"] == "bp-1"
        assert data["type"] == "node"
        assert data["position"] == "before"
        assert data["node_id"] == "analyze"


@pytest.mark.unit
class TestBreakpointStorage:
    """Test BreakpointStorage."""

    def test_add_and_get_breakpoint(self, breakpoint_storage):
        """Test adding and retrieving breakpoint."""
        bp = WorkflowBreakpoint(
            id="bp-1",
            type=BreakpointType.NODE,
            position=BreakpointPosition.BEFORE,
            node_id="analyze",
        )

        breakpoint_storage.add(bp)
        retrieved = breakpoint_storage.get("bp-1")

        assert retrieved is not None
        assert retrieved.id == "bp-1"

    def test_remove_breakpoint(self, breakpoint_storage):
        """Test removing breakpoint."""
        bp = WorkflowBreakpoint(
            id="bp-1",
            type=BreakpointType.NODE,
            position=BreakpointPosition.BEFORE,
            node_id="analyze",
        )

        breakpoint_storage.add(bp)
        removed = breakpoint_storage.remove("bp-1")

        assert removed is not None
        assert removed.id == "bp-1"
        assert breakpoint_storage.get("bp-1") is None

    def test_list_all_breakpoints(self, breakpoint_storage):
        """Test listing all breakpoints."""
        bp1 = WorkflowBreakpoint(
            id="bp-1",
            type=BreakpointType.NODE,
            position=BreakpointPosition.BEFORE,
            node_id="analyze",
        )
        bp2 = WorkflowBreakpoint(
            id="bp-2",
            type=BreakpointType.NODE,
            position=BreakpointPosition.BEFORE,
            node_id="process",
        )

        breakpoint_storage.add(bp1)
        breakpoint_storage.add(bp2)

        all_bps = breakpoint_storage.list_all()

        assert len(all_bps) == 2
        assert bp1 in all_bps
        assert bp2 in all_bps

    def test_get_breakpoints_for_node(self, breakpoint_storage):
        """Test getting breakpoints for specific node."""
        bp1 = WorkflowBreakpoint(
            id="bp-1",
            type=BreakpointType.NODE,
            position=BreakpointPosition.BEFORE,
            node_id="analyze",
        )
        bp2 = WorkflowBreakpoint(
            id="bp-2",
            type=BreakpointType.NODE,
            position=BreakpointPosition.BEFORE,
            node_id="process",
        )

        breakpoint_storage.add(bp1)
        breakpoint_storage.add(bp2)

        analyze_bps = breakpoint_storage.get_for_node("analyze")

        assert len(analyze_bps) == 1
        assert analyze_bps[0].id == "bp-1"

    def test_clear_breakpoints(self, breakpoint_storage):
        """Test clearing all breakpoints."""
        bp = WorkflowBreakpoint(
            id="bp-1",
            type=BreakpointType.NODE,
            position=BreakpointPosition.BEFORE,
            node_id="analyze",
        )

        breakpoint_storage.add(bp)
        breakpoint_storage.clear()

        assert len(breakpoint_storage.list_all()) == 0


@pytest.mark.unit
class TestBreakpointManager:
    """Test BreakpointManager."""

    async def test_set_node_breakpoint(self, breakpoint_manager):
        """Test setting a node breakpoint."""
        bp = breakpoint_manager.set_breakpoint(
            node_id="analyze",
            position=BreakpointPosition.BEFORE,
        )

        assert bp.node_id == "analyze"
        assert bp.position == BreakpointPosition.BEFORE
        assert bp.enabled is True
        assert bp.id is not None

    async def test_set_conditional_breakpoint(self, breakpoint_manager):
        """Test setting a conditional breakpoint."""

        def condition(state):
            return state.get("errors", 0) > 5

        bp = breakpoint_manager.set_breakpoint(
            node_id="process",
            position=BreakpointPosition.AFTER,
            condition=condition,
        )

        assert bp.type == BreakpointType.CONDITIONAL
        assert bp.condition == condition

    async def test_clear_breakpoint(self, breakpoint_manager):
        """Test clearing a breakpoint."""
        bp = breakpoint_manager.set_breakpoint(node_id="analyze")

        cleared = breakpoint_manager.clear_breakpoint(bp.id)

        assert cleared is True
        assert breakpoint_manager.storage.get(bp.id) is None

    async def test_list_breakpoints(self, breakpoint_manager):
        """Test listing breakpoints."""
        bp1 = breakpoint_manager.set_breakpoint(node_id="analyze")
        bp2 = breakpoint_manager.set_breakpoint(node_id="process")

        all_bps = breakpoint_manager.list_breakpoints()

        assert len(all_bps) == 2
        assert bp1 in all_bps
        assert bp2 in all_bps

    async def test_enable_disable_breakpoint(self, breakpoint_manager):
        """Test enabling/disabling breakpoint."""
        bp = breakpoint_manager.set_breakpoint(node_id="analyze")

        breakpoint_manager.disable_breakpoint(bp.id)
        assert breakpoint_manager.storage.get(bp.id).enabled is False

        breakpoint_manager.enable_breakpoint(bp.id)
        assert breakpoint_manager.storage.get(bp.id).enabled is True

    async def test_evaluate_breakpoints_hit(self, breakpoint_manager, sample_state):
        """Test breakpoint evaluation when hit."""
        bp = breakpoint_manager.set_breakpoint(
            node_id="analyze",
            position=BreakpointPosition.BEFORE,
        )

        hit_bps = breakpoint_manager.evaluate_breakpoints(
            state=sample_state,
            node_id="analyze",
            position=BreakpointPosition.BEFORE,
        )

        assert len(hit_bps) == 1
        assert hit_bps[0].id == bp.id
        assert hit_bps[0].hit_count == 1

    async def test_evaluate_breakpoints_miss(self, breakpoint_manager, sample_state):
        """Test breakpoint evaluation when missed."""
        bp = breakpoint_manager.set_breakpoint(
            node_id="analyze",
            position=BreakpointPosition.BEFORE,
        )

        hit_bps = breakpoint_manager.evaluate_breakpoints(
            state=sample_state,
            node_id="process",  # Wrong node
            position=BreakpointPosition.BEFORE,
        )

        assert len(hit_bps) == 0
        assert bp.hit_count == 0

    async def test_list_breakpoints_with_filters(self, breakpoint_manager):
        """Test listing breakpoints with filters."""
        bp1 = breakpoint_manager.set_breakpoint(node_id="analyze")
        bp2 = breakpoint_manager.set_breakpoint(node_id="process")

        # Filter by node_id
        analyze_bps = breakpoint_manager.list_breakpoints(node_id="analyze")
        assert len(analyze_bps) == 1
        assert analyze_bps[0].id == bp1.id

        # Filter by enabled_only
        breakpoint_manager.disable_breakpoint(bp1.id)
        enabled_bps = breakpoint_manager.list_breakpoints(enabled_only=True)
        assert len(enabled_bps) == 1
        assert enabled_bps[0].id == bp2.id
