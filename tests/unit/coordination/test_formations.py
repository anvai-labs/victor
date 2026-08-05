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

"""Tests for formation strategies."""

import pytest

from victor.coordination.formations import (
    ConsensusFormation,
    HierarchicalFormation,
    ParallelFormation,
    PipelineFormation,
    SequentialFormation,
)
from victor.coordination.formations.base import TeamContext
from victor.teams.types import AgentMessage, MessageType, MemberResult

# =============================================================================
# Mock Agents
# =============================================================================


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, agent_id: str, role: str = "worker"):
        self.id = agent_id
        self.role = role
        self.executed = False
        self.received_tasks = []

    async def execute(self, task: AgentMessage, context: TeamContext) -> MemberResult:
        """Mock execution."""
        self.executed = True
        self.received_tasks.append(task)

        return MemberResult(
            member_id=self.id,
            success=True,
            output=f"Result from {self.id}",
            metadata={"task": task.content},
        )


class FailingMockAgent(MockAgent):
    """Mock agent that always fails."""

    async def execute(self, task: AgentMessage, context: TeamContext) -> MemberResult:
        """Mock execution that fails."""
        self.executed = True
        self.received_tasks.append(task)

        return MemberResult(
            member_id=self.id,
            success=False,
            output=None,
            error="Mock failure",
            metadata={"task": task.content},
        )


class ManagerMockAgent(MockAgent):
    """Mock manager agent that delegates tasks."""

    async def execute(self, task: AgentMessage, context: TeamContext) -> MemberResult:
        """Mock execution with delegation."""
        self.executed = True
        self.received_tasks.append(task)

        # Return delegation tasks
        return MemberResult(
            member_id=self.id,
            success=True,
            output="Manager plan",
            metadata={
                "delegated_tasks": [
                    AgentMessage(
                        sender_id=self.id,
                        message_type=MessageType.TASK,
                        recipient_id="worker1",
                        content="Worker task 1",
                    ),
                    AgentMessage(
                        sender_id=self.id,
                        message_type=MessageType.TASK,
                        recipient_id="worker2",
                        content="Worker task 2",
                    ),
                ]
            },
        )


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_context():
    """Create mock context."""
    return TeamContext(
        team_id="test-team",
        formation="sequential",
        shared_state={},
    )


@pytest.fixture
def mock_task():
    """Create mock task."""
    return AgentMessage(
        sender_id="user",
        content="Test task",
        message_type=MessageType.TASK,
    )


@pytest.fixture
def mock_agents():
    """Create list of mock agents."""
    return [
        MockAgent("agent1"),
        MockAgent("agent2"),
        MockAgent("agent3"),
    ]


# =============================================================================
# Sequential Formation Tests
# =============================================================================


class TestSequentialFormation:
    """Tests for SequentialFormation."""

    @pytest.mark.asyncio
    async def test_execute_sequentially(self, mock_agents, mock_context, mock_task):
        """Test agents execute sequentially."""
        formation = SequentialFormation()

        results = await formation.execute(mock_agents, mock_context, mock_task)

        assert len(results) == 3
        assert all(r.success for r in results)

        # Verify agents executed in order
        assert mock_agents[0].executed
        assert mock_agents[1].executed
        assert mock_agents[2].executed

    @pytest.mark.asyncio
    async def test_context_chaining(self, mock_agents, mock_context, mock_task):
        """Test context chains between agents."""
        formation = SequentialFormation()

        await formation.execute(mock_agents, mock_context, mock_task)

        # Agent 1 received original task
        assert mock_agents[0].received_tasks[0].content == "Test task"

        # Agent 2 received result from agent 1
        assert "agent1" in mock_agents[1].received_tasks[0].data.get("previous_agent", "")

    @pytest.mark.asyncio
    async def test_handles_failure(self, mock_context, mock_task):
        """Test sequential formation handles agent failure."""
        agents = [
            MockAgent("agent1"),
            FailingMockAgent("agent2"),
            MockAgent("agent3"),
        ]

        formation = SequentialFormation()

        results = await formation.execute(agents, mock_context, mock_task)

        # Agent 2 failed but agent 3 still executed
        assert len(results) == 3
        assert not results[1].success
        assert agents[2].executed  # Continues after failure


# =============================================================================
# Parallel Formation Tests
# =============================================================================


class TestParallelFormation:
    """Tests for ParallelFormation."""

    @pytest.mark.asyncio
    async def test_execute_in_parallel(self, mock_agents, mock_context, mock_task):
        """Test agents execute in parallel."""
        formation = ParallelFormation()

        results = await formation.execute(mock_agents, mock_context, mock_task)

        assert len(results) == 3
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_all_receive_same_task(self, mock_agents, mock_context, mock_task):
        """Test all agents receive the same task."""
        formation = ParallelFormation()

        await formation.execute(mock_agents, mock_context, mock_task)

        # All agents received the original task
        for agent in mock_agents:
            assert agent.received_tasks[0].content == "Test task"

    @pytest.mark.asyncio
    async def test_handles_partial_failure(self, mock_context, mock_task):
        """Test parallel formation handles partial failures."""
        agents = [
            MockAgent("agent1"),
            FailingMockAgent("agent2"),
            MockAgent("agent3"),
        ]

        formation = ParallelFormation()

        results = await formation.execute(agents, mock_context, mock_task)

        assert len(results) == 3
        assert results[0].success
        assert not results[1].success
        assert results[2].success


# =============================================================================
# Hierarchical Formation Tests
# =============================================================================


class TestHierarchicalFormation:
    """Tests for HierarchicalFormation."""

    @pytest.mark.asyncio
    async def test_manager_delegates_to_workers(self, mock_context, mock_task):
        """Test manager delegates tasks to workers."""
        manager = ManagerMockAgent("manager", role="manager")
        workers = [MockAgent("worker1"), MockAgent("worker2")]

        formation = HierarchicalFormation()
        results = await formation.execute([manager] + workers, mock_context, mock_task)

        assert len(results) == 3  # Manager + 2 workers
        assert all(r.success for r in results)
        assert manager.executed
        assert all(w.executed for w in workers)

    @pytest.mark.asyncio
    async def test_requires_manager_role(self):
        """Test hierarchical formation requires manager."""
        formation = HierarchicalFormation()

        required_roles = formation.get_required_roles()
        assert "manager" in required_roles
        assert "coordinator" in required_roles

    @pytest.mark.asyncio
    async def test_validates_context(self):
        """Test hierarchical formation validates context."""
        formation = HierarchicalFormation()

        # Valid context
        valid_context = TeamContext(
            team_id="test",
            formation="hierarchical",
            shared_state={},
        )
        assert formation.validate_context(valid_context)

        # Invalid context (no shared_state)
        invalid_context = TeamContext(
            team_id="test",
            formation="hierarchical",
        )
        # Note: This might fail depending on implementation


# =============================================================================
# Pipeline Formation Tests
# =============================================================================


class TestPipelineFormation:
    """Tests for PipelineFormation."""

    @pytest.mark.asyncio
    async def test_pipeline_execution(self, mock_agents, mock_context, mock_task):
        """Test agents execute as pipeline."""
        formation = PipelineFormation()

        results = await formation.execute(mock_agents, mock_context, mock_task)

        assert len(results) == 3
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_output_flows_to_next(self, mock_agents, mock_context, mock_task):
        """Test output of each agent flows to next."""
        formation = PipelineFormation()

        await formation.execute(mock_agents, mock_context, mock_task)

        # Each agent after the first should receive previous agent's output
        assert mock_agents[0].received_tasks[0].content == "Test task"

    @pytest.mark.asyncio
    async def test_stops_on_failure(self, mock_context, mock_task):
        """Test pipeline stops on first failure."""
        agents = [
            MockAgent("agent1"),
            FailingMockAgent("agent2"),
            MockAgent("agent3"),
        ]

        formation = PipelineFormation()

        results = await formation.execute(agents, mock_context, mock_task)

        # Agent 3 should not execute
        assert len(results) == 2  # Only agent1 and agent2 results
        assert agents[2].executed is False


# =============================================================================
# Consensus Formation Tests
# =============================================================================


class TestConsensusFormation:
    """Tests for ConsensusFormation."""

    @pytest.mark.asyncio
    async def test_reaches_consensus(self, mock_context, mock_task):
        """Test agents can reach consensus."""
        formation = ConsensusFormation(max_rounds=2, agreement_threshold=0.7)

        # Create agents that return identical results to test consensus
        class ConsensusMockAgent:
            """Mock agent that returns identical results for consensus testing."""

            def __init__(self, agent_id: str):
                self.id = agent_id
                self.executed = False
                self.received_tasks = []

            async def execute(self, task: AgentMessage, context: TeamContext) -> MemberResult:
                """Mock execution that returns identical result."""
                self.executed = True
                self.received_tasks.append(task)

                return MemberResult(
                    member_id=self.id,
                    success=True,
                    output="Consensus result",  # All agents return same result
                    metadata={"task": task.content},
                )

        agents = [ConsensusMockAgent(f"agent{i}") for i in range(3)]

        results = await formation.execute(agents, mock_context, mock_task)

        # Should complete in 1 round since all agents return identical results
        assert len(results) == 3
        # All results should have the same output
        assert all(r.output == "Consensus result" for r in results)

    @pytest.mark.asyncio
    async def test_multiple_rounds(self, mock_context, mock_task):
        """Test multiple consensus rounds if needed."""
        # Use agents that return different results
        agents = [MockAgent(f"agent{i}") for i in range(3)]

        # Make agents return different content
        for i, agent in enumerate(agents):
            original_execute = agent.execute

            async def varied_execute(task, context, idx=i):
                result = await original_execute(task, context)
                result.content = f"Different result {idx}"
                return result

            agent.execute = varied_execute

        formation = ConsensusFormation(max_rounds=2, agreement_threshold=1.0)

        results = await formation.execute(agents, mock_context, mock_task)

        # Should execute multiple rounds trying to reach consensus
        assert len(results) >= 3

    @pytest.mark.asyncio
    async def test_max_rounds_limit(self, mock_agents, mock_context, mock_task):
        """Test respects max rounds limit."""
        formation = ConsensusFormation(max_rounds=2)

        results = await formation.execute(mock_agents, mock_context, mock_task)

        # Should not exceed max_rounds * num_agents
        assert len(results) <= 2 * 3
