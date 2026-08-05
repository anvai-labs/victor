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

"""TDD Patterns for Multi-Agent Teams.

This module demonstrates Test-Driven Development patterns for multi-agent
systems using Victor's unified teams architecture. Each test class shows
a specific pattern with red-green-refactor examples.

Usage:
    pytest tests/examples/test_multiagent_tdd_patterns.py -v

TDD Patterns Covered:
1. Agent Role Testing - Verify agents behave according to their roles
2. Formation Pattern Testing - Test team formation strategies
3. Communication Testing - Verify inter-agent messaging
4. Concurrent Execution Testing - Test parallel agent execution
5. Pipeline Testing - Verify chained agent outputs
6. Observability Testing - Test event emission during execution
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.teams import (
    AgentMessage,
    UnifiedTeamCoordinator,
    ITeamCoordinator,
    ITeamMember,
    MemberResult,
    MessageType,
    TeamFormation,
    TeamMessageBus,
    TeamResult,
    TeamSharedMemory,
    UnifiedTeamCoordinator,
    create_coordinator,
)

# =============================================================================
# Pattern 1: Mock Team Members for Testing
# =============================================================================


@dataclass
class MockAgent:
    """Mock agent for TDD testing of multi-agent systems.

    This mock implements ITeamMember protocol and allows:
    - Configurable task outputs
    - Controlled execution timing
    - Message tracking
    - Failure simulation
    """

    id: str
    output: str = "Task completed"
    delay: float = 0.0
    should_fail: bool = False
    execution_count: int = field(default=0, init=False)
    received_messages: List[AgentMessage] = field(default_factory=list)
    task_history: List[str] = field(default_factory=list)
    context_history: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def role(self) -> MagicMock:
        """Return a mock role for protocol compliance."""
        return MagicMock(name=self.id)

    async def execute_task(self, task: str, context: Dict[str, Any]) -> str:
        """Execute a task and track execution."""
        self.execution_count += 1
        self.task_history.append(task)
        self.context_history.append(context.copy())

        if self.delay > 0:
            await asyncio.sleep(self.delay)

        if self.should_fail:
            raise RuntimeError(f"Agent {self.id} failed")

        return self.output

    async def receive_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Receive and optionally respond to a message."""
        self.received_messages.append(message)
        return AgentMessage(
            sender_id=self.id,
            recipient_id=message.sender_id,
            content=f"Acknowledged: {message.content}",
            message_type=MessageType.RESPONSE,
        )


# =============================================================================
# Pattern 2: Agent Role TDD
# =============================================================================


@pytest.mark.unit
class TestAgentRoleTDD:
    """TDD pattern: Testing agent roles and capabilities.

    RED: Write failing tests for expected role behaviors
    GREEN: Implement roles that pass tests
    REFACTOR: Consolidate common role patterns
    """

    def test_researcher_discovers_information(self):
        """RED -> GREEN: Researcher agents should produce discoveries."""
        # Arrange: Create a researcher-type agent
        researcher = MockAgent(
            id="researcher_1",
            output="Discovered: API endpoint at /api/v2/users",
        )

        # Act: Execute research task
        result = asyncio.run(
            researcher.execute_task(
                "Find all API endpoints in the codebase",
                {"codebase_path": "/project"},
            )
        )

        # Assert: Researcher produces discovery output
        assert "Discovered" in result
        assert researcher.execution_count == 1
        assert "API endpoints" in researcher.task_history[0]

    def test_executor_modifies_code(self):
        """RED -> GREEN: Executor agents should produce code changes."""
        # Arrange: Create an executor-type agent
        executor = MockAgent(
            id="executor_1",
            output="Modified: Added authentication to endpoint",
        )

        # Act: Execute modification task
        result = asyncio.run(
            executor.execute_task(
                "Add authentication middleware to /api/v2/users",
                {"file_path": "/project/api/routes.py"},
            )
        )

        # Assert: Executor produces modification output
        assert "Modified" in result
        assert "authentication" in result

    def test_reviewer_provides_feedback(self):
        """RED -> GREEN: Reviewer agents should produce code review feedback."""
        # Arrange: Create a reviewer-type agent
        reviewer = MockAgent(
            id="reviewer_1",
            output="Review: Code looks good, minor issue on line 42",
        )

        # Act: Execute review task
        result = asyncio.run(
            reviewer.execute_task(
                "Review the authentication changes",
                {"diff": "--- old\n+++ new"},
            )
        )

        # Assert: Reviewer produces feedback
        assert "Review" in result
        assert "line 42" in result


# =============================================================================
# Pattern 3: Team Formation TDD
# =============================================================================


@pytest.mark.unit
class TestTeamFormationTDD:
    """TDD pattern: Testing different team formations.

    Each formation has distinct execution semantics that should be tested.
    """

    @pytest.fixture
    def coordinator(self) -> ITeamCoordinator:
        """Create a fresh coordinator for each test."""
        return create_coordinator(lightweight=True)

    @pytest.mark.asyncio
    async def test_sequential_executes_in_order(self, coordinator):
        """RED -> GREEN: Sequential should execute agents in add order."""
        # Arrange: Create agents that track execution order
        execution_order = []

        agent1 = MockAgent(id="first", output="First done")
        agent2 = MockAgent(id="second", output="Second done")

        # Track original execute_task
        original1 = agent1.execute_task
        original2 = agent2.execute_task

        async def track1(task, ctx):
            execution_order.append("first")
            return await original1(task, ctx)

        async def track2(task, ctx):
            execution_order.append("second")
            return await original2(task, ctx)

        agent1.execute_task = track1
        agent2.execute_task = track2

        # Act: Execute with sequential formation
        coordinator.add_member(agent1).add_member(agent2)
        coordinator.set_formation(TeamFormation.SEQUENTIAL)
        result = await coordinator.execute_task("Test task", {})

        # Assert: Execution order matches add order
        assert execution_order == ["first", "second"]
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_parallel_executes_concurrently(self, coordinator):
        """RED -> GREEN: Parallel should execute agents concurrently."""
        # Arrange: Create agents with delays
        start_times = {}
        end_times = {}

        async def tracked_execute(agent_id, delay):
            start_times[agent_id] = asyncio.get_event_loop().time()
            await asyncio.sleep(delay)
            end_times[agent_id] = asyncio.get_event_loop().time()
            return f"{agent_id} done"

        agent1 = MockAgent(id="parallel_1")
        agent2 = MockAgent(id="parallel_2")

        agent1.execute_task = lambda t, c: tracked_execute("parallel_1", 0.1)
        agent2.execute_task = lambda t, c: tracked_execute("parallel_2", 0.1)

        # Act: Execute with parallel formation
        coordinator.add_member(agent1).add_member(agent2)
        coordinator.set_formation(TeamFormation.PARALLEL)
        result = await coordinator.execute_task("Test task", {})

        # Assert: Agents started at approximately the same time
        assert result["success"] is True
        time_diff = abs(start_times["parallel_1"] - start_times["parallel_2"])
        assert time_diff < 0.05  # Within 50ms of each other

    @pytest.mark.asyncio
    async def test_pipeline_chains_outputs(self, coordinator):
        """RED -> GREEN: Pipeline should chain agent outputs."""
        # Arrange: Create pipeline stages
        stage1 = MockAgent(id="stage1", output="Stage1Output")
        stage2 = MockAgent(id="stage2", output="Stage2Output")

        # Act: Execute with pipeline formation
        coordinator.add_member(stage1).add_member(stage2)
        coordinator.set_formation(TeamFormation.PIPELINE)
        result = await coordinator.execute_task("Initial input", {})

        # Assert: Final output is from last stage
        assert result["success"] is True
        assert result["final_output"] == "Stage2Output"


# =============================================================================
# Pattern 4: Agent Communication TDD
# =============================================================================


@pytest.mark.unit
class TestAgentCommunicationTDD:
    """TDD pattern: Testing inter-agent communication."""

    @pytest.fixture
    def message_bus(self) -> TeamMessageBus:
        """Create a fresh message bus for each test."""
        return TeamMessageBus("test_team")

    @pytest.mark.asyncio
    async def test_broadcast_reaches_all_agents(self, message_bus):
        """RED -> GREEN: Broadcast should deliver to all registered agents."""
        # Arrange: Register multiple agents
        message_bus.register_agent("sender")
        message_bus.register_agent("receiver_1")
        message_bus.register_agent("receiver_2")

        # Act: Broadcast a message
        broadcast = AgentMessage(
            sender_id="sender",
            recipient_id=None,  # Broadcast
            content="Team update",
            message_type=MessageType.STATUS,
        )
        await message_bus.send(broadcast)

        # Assert: All receivers got the message
        msg1 = await message_bus.receive("receiver_1", timeout=1.0)
        msg2 = await message_bus.receive("receiver_2", timeout=1.0)
        sender_msg = await message_bus.receive("sender", timeout=0)

        assert msg1 is not None
        assert msg2 is not None
        assert sender_msg is None  # Sender doesn't get their own broadcast

    @pytest.mark.asyncio
    async def test_direct_message_reaches_recipient(self, message_bus):
        """RED -> GREEN: Direct messages should only reach the recipient."""
        # Arrange: Register agents
        message_bus.register_agent("sender")
        message_bus.register_agent("target")
        message_bus.register_agent("other")

        # Act: Send a direct message
        direct = AgentMessage(
            sender_id="sender",
            recipient_id="target",
            content="Private message",
            message_type=MessageType.TASK,
        )
        await message_bus.send(direct)

        # Assert: Only target receives the message
        target_msg = await message_bus.receive("target", timeout=1.0)
        other_msg = await message_bus.receive("other", timeout=0)

        assert target_msg is not None
        assert target_msg.content == "Private message"
        assert other_msg is None


# =============================================================================
# Pattern 5: Shared Memory TDD
# =============================================================================


@pytest.mark.unit
class TestSharedMemoryTDD:
    """TDD pattern: Testing shared memory between agents."""

    @pytest.fixture
    def shared_memory(self) -> TeamSharedMemory:
        """Create fresh shared memory for each test."""
        return TeamSharedMemory()

    def test_agents_can_share_data(self, shared_memory):
        """RED -> GREEN: Agents should be able to share data."""
        # Act: One agent stores data
        shared_memory.set("research_findings", {"endpoints": ["/api/v1"]}, "researcher")

        # Assert: Another agent can read it
        data = shared_memory.get("research_findings")
        assert data["endpoints"] == ["/api/v1"]

    def test_tracks_contributors(self, shared_memory):
        """RED -> GREEN: Should track which agent contributed data."""
        # Act: Multiple agents contribute
        shared_memory.set("findings", {"initial": "data"}, "agent_1")
        shared_memory.update("findings", {"extra": "data"}, "agent_2")

        # Assert: Contributors are tracked
        contributors = shared_memory.get_contributors("findings")
        assert "agent_1" in contributors
        assert "agent_2" in contributors

    def test_append_to_list(self, shared_memory):
        """RED -> GREEN: Should append to list values."""
        # Arrange: Initialize a list
        shared_memory.set("discoveries", [], "researcher")

        # Act: Multiple agents append
        shared_memory.append("discoveries", "Found API", "researcher_1")
        shared_memory.append("discoveries", "Found DB", "researcher_2")

        # Assert: Both items present
        discoveries = shared_memory.get("discoveries")
        assert len(discoveries) == 2
        assert "Found API" in discoveries
        assert "Found DB" in discoveries


# =============================================================================
# Pattern 6: Concurrent Agent Testing
# =============================================================================


@pytest.mark.unit
class TestConcurrentAgentsTDD:
    """TDD pattern: Testing concurrent agent execution."""

    @pytest.mark.asyncio
    async def test_concurrent_agents_complete_independently(self):
        """RED -> GREEN: Concurrent agents should complete independently."""
        # Arrange: Create agents with different delays
        fast_agent = MockAgent(id="fast", output="Fast done", delay=0.01)
        slow_agent = MockAgent(id="slow", output="Slow done", delay=0.1)

        # Act: Execute concurrently
        async def run_agent(agent):
            return await agent.execute_task("Task", {})

        start = asyncio.get_event_loop().time()
        results = await asyncio.gather(
            run_agent(fast_agent),
            run_agent(slow_agent),
        )
        duration = asyncio.get_event_loop().time() - start

        # Assert: Both completed, total time is max of delays
        assert len(results) == 2
        assert "Fast done" in results
        assert "Slow done" in results
        assert duration < 0.15  # Less than sum of delays

    @pytest.mark.asyncio
    async def test_one_failure_doesnt_block_others(self):
        """RED -> GREEN: One agent failing shouldn't block others."""
        # Arrange: Create mixed agents
        good_agent = MockAgent(id="good", output="Success")
        bad_agent = MockAgent(id="bad", should_fail=True)

        # Act: Execute with parallel coordinator
        coordinator = create_coordinator(lightweight=True)
        coordinator.add_member(good_agent).add_member(bad_agent)
        coordinator.set_formation(TeamFormation.PARALLEL)

        result = await coordinator.execute_task("Task", {})

        # Assert: Good agent succeeded even though bad agent failed
        assert "good" in result["member_results"]
        assert result["member_results"]["good"].success is True
        assert result["member_results"]["bad"].success is False


# =============================================================================
# Pattern 7: Integration Testing with Unified Coordinator
# =============================================================================


@pytest.mark.integration
class TestUnifiedCoordinatorTDD:
    """TDD pattern: Integration tests for UnifiedTeamCoordinator."""

    @pytest.fixture
    def coordinator(self) -> UnifiedTeamCoordinator:
        """Create coordinator without observability for testing."""
        return UnifiedTeamCoordinator(enable_observability=False)

    @pytest.mark.asyncio
    async def test_full_team_workflow(self, coordinator):
        """RED -> GREEN: Complete team workflow should work end-to-end."""
        # Arrange: Create a realistic team
        researcher = MockAgent(
            id="researcher",
            output="Found: Authentication module at /auth",
        )
        executor = MockAgent(
            id="executor",
            output="Modified: Added rate limiting to auth module",
        )
        reviewer = MockAgent(
            id="reviewer",
            output="Approved: Changes look good",
        )

        # Act: Build and execute team
        coordinator.add_member(researcher)
        coordinator.add_member(executor)
        coordinator.add_member(reviewer)
        coordinator.set_formation(TeamFormation.PIPELINE)

        result = await coordinator.execute_task(
            "Add rate limiting to authentication",
            {"priority": "high"},
        )

        # Assert: Pipeline completed successfully
        assert result["success"] is True
        assert len(result["member_results"]) == 3
        assert result["final_output"] == "Approved: Changes look good"

    @pytest.mark.asyncio
    async def test_fluent_api_chaining(self, coordinator):
        """RED -> GREEN: Fluent API should support method chaining."""
        # Act: Use fluent API
        agent = MockAgent(id="test")
        result_coordinator = (
            coordinator.add_member(agent)
            .set_formation(TeamFormation.SEQUENTIAL)
            .clear()
            .add_member(agent)
        )

        # Assert: Chaining returns same coordinator
        assert result_coordinator is coordinator
        assert len(coordinator.members) == 1


# =============================================================================
# Pattern 8: Factory Function Testing
# =============================================================================


@pytest.mark.unit
class TestFactoryFunctionTDD:
    """TDD pattern: Testing create_coordinator factory."""

    def test_default_requires_orchestrator(self):
        """Default production coordinators require an orchestrator."""
        with pytest.raises(ValueError, match="orchestrator is required"):
            create_coordinator()

    def test_lightweight_creates_unified(self):
        """Lightweight mode creates a coordinator without an orchestrator."""
        coordinator = create_coordinator(lightweight=True)
        assert isinstance(coordinator, UnifiedTeamCoordinator)

    def test_respects_observability_flag(self):
        """Factory forwards the observability flag to production coordinators."""
        orchestrator = MagicMock()
        with_obs = create_coordinator(
            orchestrator=orchestrator,
            enable_observability=True,
        )
        without_obs = create_coordinator(
            orchestrator=orchestrator,
            enable_observability=False,
        )

        assert with_obs._observability_enabled is True
        assert without_obs._observability_enabled is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
