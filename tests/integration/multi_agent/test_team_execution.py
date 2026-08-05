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

"""Integration tests for multi-agent team execution.

These tests verify the end-to-end behavior of team execution across
different formation patterns: SEQUENTIAL, PARALLEL, HIERARCHICAL,
PIPELINE, and CONSENSUS.

Tests exercise the real implementations in:
- victor/framework/team_coordinator.py
- victor/framework/agent_protocols.py
- victor/framework/agent_roles.py
- victor/framework/personas.py
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import pytest

from victor.framework.agent_protocols import (
    AgentCapability,
    AgentMessage,
    IAgentPersona,
    IAgentRole,
    ITeamMember,
    MessageType,
    TeamFormation,
)
from victor.framework.agent_roles import (
    ExecutorRole,
    ManagerRole,
    ResearcherRole,
    ReviewerRole,
    get_role,
)
from victor.framework.personas import Persona, get_persona
from victor.teams import (
    UnifiedTeamCoordinator,
    MemberResult,
    TeamResult,
)

# =============================================================================
# Test Fixtures and Mock Implementations
# =============================================================================


@dataclass
class IntegrationTestRole:
    """Role for integration testing with configurable capabilities."""

    name: str = "test_role"
    capabilities: Set[AgentCapability] = field(
        default_factory=lambda: {AgentCapability.READ, AgentCapability.COMMUNICATE}
    )
    allowed_tools: Set[str] = field(default_factory=lambda: {"read_file"})
    tool_budget: int = 15

    def get_system_prompt_section(self) -> str:
        return f"You are a {self.name} agent for integration testing."


class IntegrationTestAgent:
    """Full-featured test agent for integration testing.

    Tracks execution details for verification while simulating
    realistic agent behavior with delays and state tracking.
    """

    def __init__(
        self,
        agent_id: str,
        role: Optional[IAgentRole] = None,
        persona: Optional[IAgentPersona] = None,
        delay: float = 0.0,
        output_prefix: str = "",
        fail_on_task: bool = False,
    ):
        self._id = agent_id
        self._role = role or IntegrationTestRole(name=agent_id)
        self._persona = persona
        self.delay = delay
        self.output_prefix = output_prefix or agent_id
        self.fail_on_task = fail_on_task

        # Tracking for test assertions
        self.executed_tasks: List[str] = []
        self.received_contexts: List[Dict[str, Any]] = []
        self.received_messages: List[AgentMessage] = []
        self.execution_timestamps: List[float] = []

    @property
    def id(self) -> str:
        return self._id

    @property
    def role(self) -> IAgentRole:
        return self._role

    @property
    def persona(self) -> Optional[IAgentPersona]:
        return self._persona

    async def execute_task(self, task: str, context: Dict[str, Any]) -> str:
        """Execute a task with timing and state tracking."""
        self.execution_timestamps.append(time.time())
        self.executed_tasks.append(task)
        self.received_contexts.append(context.copy())

        # Track execution order if tracker provided
        if "execution_tracker" in context:
            context["execution_tracker"].append(self.id)

        # Simulate processing time
        if self.delay > 0:
            await asyncio.sleep(self.delay)

        # Simulate failure if configured
        if self.fail_on_task:
            raise RuntimeError(f"Agent {self.id} failed task: {task}")

        # Generate output, potentially using previous output in pipeline
        previous_output = context.get("previous_output", "")
        if previous_output:
            return f"{self.output_prefix}: Processed [{previous_output}] for task: {task}"
        return f"{self.output_prefix}: Completed task: {task}"

    async def receive_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Receive and respond to a message."""
        self.received_messages.append(message)

        # Format response using persona if available
        content = f"Acknowledged: {message.content}"
        if self._persona:
            content = self._persona.format_message(content)

        return AgentMessage(
            sender_id=self.id,
            recipient_id=message.sender_id,
            content=content,
            message_type=MessageType.RESULT,
        )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def coordinator() -> UnifiedTeamCoordinator:
    """Create a fresh coordinator for each test."""
    return UnifiedTeamCoordinator()


@pytest.fixture
def execution_tracker() -> List[str]:
    """Create execution order tracker."""
    return []


@pytest.fixture
def three_agent_team() -> List[IntegrationTestAgent]:
    """Create a basic three-agent team."""
    return [
        IntegrationTestAgent("agent_1", delay=0.01),
        IntegrationTestAgent("agent_2", delay=0.01),
        IntegrationTestAgent("agent_3", delay=0.01),
    ]


@pytest.fixture
def hierarchical_team() -> tuple:
    """Create a team with manager and workers."""
    manager_role = ManagerRole()
    worker_role = ExecutorRole()

    manager = IntegrationTestAgent("manager", role=manager_role, delay=0.01)
    worker1 = IntegrationTestAgent("worker_1", role=worker_role, delay=0.01)
    worker2 = IntegrationTestAgent("worker_2", role=worker_role, delay=0.01)

    return manager, [worker1, worker2]


@pytest.fixture
def pipeline_team() -> List[IntegrationTestAgent]:
    """Create a pipeline team with specialized roles."""
    researcher = IntegrationTestAgent(
        "researcher",
        role=ResearcherRole(),
        output_prefix="Research",
        delay=0.01,
    )
    executor = IntegrationTestAgent(
        "executor",
        role=ExecutorRole(),
        output_prefix="Implementation",
        delay=0.01,
    )
    reviewer = IntegrationTestAgent(
        "reviewer",
        role=ReviewerRole(),
        output_prefix="Review",
        delay=0.01,
    )
    return [researcher, executor, reviewer]


# =============================================================================
# SEQUENTIAL Formation Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.agents
class TestSequentialFormation:
    """Integration tests for SEQUENTIAL team formation."""

    @pytest.mark.asyncio
    async def test_sequential_execution_order(
        self,
        coordinator: UnifiedTeamCoordinator,
        three_agent_team: List[IntegrationTestAgent],
        execution_tracker: List[str],
    ):
        """Agents execute in the order they were added."""
        for agent in three_agent_team:
            coordinator.add_member(agent)

        coordinator.set_formation(TeamFormation.SEQUENTIAL)

        result = await coordinator.execute_task(
            "Complete analysis task",
            {"execution_tracker": execution_tracker},
        )

        assert result["success"] is True
        assert execution_tracker == ["agent_1", "agent_2", "agent_3"]

    @pytest.mark.asyncio
    async def test_sequential_all_agents_receive_same_task(
        self,
        coordinator: UnifiedTeamCoordinator,
        three_agent_team: List[IntegrationTestAgent],
    ):
        """All agents receive the same task description."""
        for agent in three_agent_team:
            coordinator.add_member(agent)

        coordinator.set_formation(TeamFormation.SEQUENTIAL)
        task = "Analyze authentication module"

        await coordinator.execute_task(task, {})

        for agent in three_agent_team:
            assert len(agent.executed_tasks) == 1
            assert agent.executed_tasks[0] == task

    @pytest.mark.asyncio
    async def test_sequential_collects_all_results(
        self,
        coordinator: UnifiedTeamCoordinator,
        three_agent_team: List[IntegrationTestAgent],
    ):
        """Results are collected from all agents."""
        for agent in three_agent_team:
            coordinator.add_member(agent)

        coordinator.set_formation(TeamFormation.SEQUENTIAL)

        result = await coordinator.execute_task("Test task", {})

        assert "member_results" in result
        assert len(result["member_results"]) == 3
        assert all(agent.id in result["member_results"] for agent in three_agent_team)

    @pytest.mark.asyncio
    async def test_sequential_handles_agent_failure(
        self,
        coordinator: UnifiedTeamCoordinator,
    ):
        """Sequential formation handles individual agent failures."""
        agent1 = IntegrationTestAgent("agent_1")
        agent2 = IntegrationTestAgent("agent_2", fail_on_task=True)
        agent3 = IntegrationTestAgent("agent_3")

        coordinator.add_member(agent1).add_member(agent2).add_member(agent3)
        coordinator.set_formation(TeamFormation.SEQUENTIAL)

        result = await coordinator.execute_task("Test task", {})

        # Overall should report failure
        assert result["success"] is False
        # Failed agent should have error recorded
        assert result["member_results"]["agent_2"].success is False
        assert result["member_results"]["agent_2"].error is not None

    @pytest.mark.asyncio
    async def test_sequential_timing_is_additive(
        self,
        coordinator: UnifiedTeamCoordinator,
    ):
        """Sequential execution time is sum of individual delays."""
        # Each agent takes 0.05s
        agents = [IntegrationTestAgent(f"agent_{i}", delay=0.05) for i in range(3)]

        for agent in agents:
            coordinator.add_member(agent)

        coordinator.set_formation(TeamFormation.SEQUENTIAL)

        start = time.time()
        await coordinator.execute_task("Test task", {})
        duration = time.time() - start

        # Should take at least 0.15s (3 x 0.05s)
        assert duration >= 0.15


# =============================================================================
# PARALLEL Formation Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.agents
class TestParallelFormation:
    """Integration tests for PARALLEL team formation."""

    @pytest.mark.asyncio
    async def test_parallel_concurrent_execution(
        self,
        coordinator: UnifiedTeamCoordinator,
    ):
        """Agents execute concurrently with parallel formation."""
        # Each agent takes 0.1s
        agents = [IntegrationTestAgent(f"agent_{i}", delay=0.1) for i in range(3)]

        for agent in agents:
            coordinator.add_member(agent)

        coordinator.set_formation(TeamFormation.PARALLEL)

        start = time.time()
        await coordinator.execute_task("Parallel task", {})
        duration = time.time() - start

        # If parallel, should take ~0.1s, not ~0.3s
        assert duration < 0.2

    @pytest.mark.asyncio
    async def test_parallel_all_agents_execute(
        self,
        coordinator: UnifiedTeamCoordinator,
        three_agent_team: List[IntegrationTestAgent],
    ):
        """All agents execute the task in parallel formation."""
        for agent in three_agent_team:
            coordinator.add_member(agent)

        coordinator.set_formation(TeamFormation.PARALLEL)

        result = await coordinator.execute_task("Parallel analysis", {})

        for agent in three_agent_team:
            assert len(agent.executed_tasks) == 1

        assert len(result["member_results"]) == 3

    @pytest.mark.asyncio
    async def test_parallel_timestamps_overlap(
        self,
        coordinator: UnifiedTeamCoordinator,
    ):
        """Agent execution timestamps should overlap in parallel."""
        agents = [IntegrationTestAgent(f"agent_{i}", delay=0.1) for i in range(3)]

        for agent in agents:
            coordinator.add_member(agent)

        coordinator.set_formation(TeamFormation.PARALLEL)

        await coordinator.execute_task("Test task", {})

        # Get all start timestamps
        timestamps = [agent.execution_timestamps[0] for agent in agents]

        # In parallel, timestamps should be very close (within 0.05s)
        max_diff = max(timestamps) - min(timestamps)
        assert max_diff < 0.05

    @pytest.mark.asyncio
    async def test_parallel_handles_mixed_failures(
        self,
        coordinator: UnifiedTeamCoordinator,
    ):
        """Parallel formation handles some agents failing."""
        agent1 = IntegrationTestAgent("agent_1")
        agent2 = IntegrationTestAgent("agent_2", fail_on_task=True)
        agent3 = IntegrationTestAgent("agent_3")

        coordinator.add_member(agent1).add_member(agent2).add_member(agent3)
        coordinator.set_formation(TeamFormation.PARALLEL)

        result = await coordinator.execute_task("Test task", {})

        # Overall should fail
        assert result["success"] is False
        # Successful agents recorded
        assert result["member_results"]["agent_1"].success is True
        assert result["member_results"]["agent_3"].success is True
        # Failed agent recorded
        assert result["member_results"]["agent_2"].success is False

    @pytest.mark.asyncio
    async def test_parallel_collects_all_results(
        self,
        coordinator: UnifiedTeamCoordinator,
        three_agent_team: List[IntegrationTestAgent],
    ):
        """Parallel execution collects results from all agents."""
        for agent in three_agent_team:
            coordinator.add_member(agent)

        coordinator.set_formation(TeamFormation.PARALLEL)

        result = await coordinator.execute_task("Test task", {})

        assert result["success"] is True
        assert len(result["member_results"]) == 3


# =============================================================================
# HIERARCHICAL Formation Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.agents
class TestHierarchicalFormation:
    """Integration tests for HIERARCHICAL team formation."""

    @pytest.mark.asyncio
    async def test_hierarchical_manager_executes_first(
        self,
        coordinator: UnifiedTeamCoordinator,
        hierarchical_team: tuple,
        execution_tracker: List[str],
    ):
        """Manager executes before workers in hierarchical formation."""
        manager, workers = hierarchical_team

        coordinator.add_member(manager)
        for worker in workers:
            coordinator.add_member(worker)

        coordinator.set_manager(manager)
        coordinator.set_formation(TeamFormation.HIERARCHICAL)

        await coordinator.execute_task(
            "Hierarchical task",
            {"execution_tracker": execution_tracker},
        )

        # Manager should be first
        assert execution_tracker[0] == "manager"

    @pytest.mark.asyncio
    async def test_hierarchical_workers_execute_after_manager(
        self,
        coordinator: UnifiedTeamCoordinator,
        hierarchical_team: tuple,
        execution_tracker: List[str],
    ):
        """Workers execute after manager in hierarchical formation."""
        manager, workers = hierarchical_team

        coordinator.add_member(manager)
        for worker in workers:
            coordinator.add_member(worker)

        coordinator.set_manager(manager)
        coordinator.set_formation(TeamFormation.HIERARCHICAL)

        await coordinator.execute_task(
            "Hierarchical task",
            {"execution_tracker": execution_tracker},
        )

        # Workers should come after manager
        assert set(execution_tracker[1:]) == {"worker_1", "worker_2"}

    @pytest.mark.asyncio
    async def test_hierarchical_auto_selects_manager_by_capability(
        self,
        coordinator: UnifiedTeamCoordinator,
        execution_tracker: List[str],
    ):
        """Auto-selects manager based on DELEGATE capability."""
        # Agent with DELEGATE capability should be auto-selected as manager
        manager_role = IntegrationTestRole(
            name="manager",
            capabilities={AgentCapability.DELEGATE, AgentCapability.COMMUNICATE},
        )
        worker_role = IntegrationTestRole(
            name="worker",
            capabilities={AgentCapability.READ, AgentCapability.WRITE},
        )

        worker1 = IntegrationTestAgent("worker_1", role=worker_role)
        manager = IntegrationTestAgent("manager", role=manager_role)
        worker2 = IntegrationTestAgent("worker_2", role=worker_role)

        # Add manager in the middle
        coordinator.add_member(worker1).add_member(manager).add_member(worker2)
        coordinator.set_formation(TeamFormation.HIERARCHICAL)

        await coordinator.execute_task(
            "Auto-select test",
            {"execution_tracker": execution_tracker},
        )

        # Manager should still be first despite being added second
        assert execution_tracker[0] == "manager"

    @pytest.mark.asyncio
    async def test_hierarchical_with_explicit_manager(
        self,
        coordinator: UnifiedTeamCoordinator,
        execution_tracker: List[str],
    ):
        """Explicit manager is used when set."""
        agents = [IntegrationTestAgent(f"agent_{i}") for i in range(3)]

        for agent in agents:
            coordinator.add_member(agent)

        # Set agent_2 as explicit manager
        coordinator.set_manager(agents[1])
        coordinator.set_formation(TeamFormation.HIERARCHICAL)

        await coordinator.execute_task(
            "Explicit manager test",
            {"execution_tracker": execution_tracker},
        )

        assert execution_tracker[0] == "agent_1"

    @pytest.mark.asyncio
    async def test_hierarchical_all_results_collected(
        self,
        coordinator: UnifiedTeamCoordinator,
        hierarchical_team: tuple,
    ):
        """All results are collected in hierarchical formation."""
        manager, workers = hierarchical_team

        coordinator.add_member(manager)
        for worker in workers:
            coordinator.add_member(worker)

        coordinator.set_manager(manager)
        coordinator.set_formation(TeamFormation.HIERARCHICAL)

        result = await coordinator.execute_task("Test task", {})

        assert result["success"] is True
        assert len(result["member_results"]) == 3


# =============================================================================
# PIPELINE Formation Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.agents
class TestPipelineFormation:
    """Integration tests for PIPELINE team formation."""

    @pytest.mark.asyncio
    async def test_pipeline_sequential_order(
        self,
        coordinator: UnifiedTeamCoordinator,
        pipeline_team: List[IntegrationTestAgent],
        execution_tracker: List[str],
    ):
        """Pipeline executes in sequence."""
        for agent in pipeline_team:
            coordinator.add_member(agent)

        coordinator.set_formation(TeamFormation.PIPELINE)

        await coordinator.execute_task(
            "Pipeline task",
            {"execution_tracker": execution_tracker},
        )

        assert execution_tracker == ["researcher", "executor", "reviewer"]

    @pytest.mark.asyncio
    async def test_pipeline_passes_output_to_next_stage(
        self,
        coordinator: UnifiedTeamCoordinator,
        pipeline_team: List[IntegrationTestAgent],
    ):
        """Pipeline passes previous output to next stage."""
        for agent in pipeline_team:
            coordinator.add_member(agent)

        coordinator.set_formation(TeamFormation.PIPELINE)

        result = await coordinator.execute_task("Feature implementation", {})

        # Second stage should have received previous_output
        executor = pipeline_team[1]
        assert len(executor.received_contexts) == 1
        assert "previous_output" in executor.received_contexts[0]
        assert executor.received_contexts[0]["previous_output"] != ""

    @pytest.mark.asyncio
    async def test_pipeline_final_output(
        self,
        coordinator: UnifiedTeamCoordinator,
        pipeline_team: List[IntegrationTestAgent],
    ):
        """Pipeline returns final stage output."""
        for agent in pipeline_team:
            coordinator.add_member(agent)

        coordinator.set_formation(TeamFormation.PIPELINE)

        result = await coordinator.execute_task("Pipeline task", {})

        assert "final_output" in result
        # Final output should be from the last agent (reviewer)
        assert "Review" in result["final_output"]

    @pytest.mark.asyncio
    async def test_pipeline_stops_on_failure(
        self,
        coordinator: UnifiedTeamCoordinator,
        execution_tracker: List[str],
    ):
        """Pipeline stops when a stage fails."""
        agent1 = IntegrationTestAgent("stage_1")
        agent2 = IntegrationTestAgent("stage_2", fail_on_task=True)
        agent3 = IntegrationTestAgent("stage_3")

        coordinator.add_member(agent1).add_member(agent2).add_member(agent3)
        coordinator.set_formation(TeamFormation.PIPELINE)

        result = await coordinator.execute_task(
            "Pipeline task",
            {"execution_tracker": execution_tracker},
        )

        # Pipeline should stop at failed stage
        assert result["success"] is False
        assert "stage_1" in execution_tracker
        assert "stage_2" in execution_tracker
        # Stage 3 should not execute
        assert "stage_3" not in execution_tracker

    @pytest.mark.asyncio
    async def test_pipeline_output_chain(
        self,
        coordinator: UnifiedTeamCoordinator,
    ):
        """Each stage output becomes input for the next."""
        agents = [IntegrationTestAgent(f"stage_{i}", output_prefix=f"Stage{i}") for i in range(3)]

        for agent in agents:
            coordinator.add_member(agent)

        coordinator.set_formation(TeamFormation.PIPELINE)

        result = await coordinator.execute_task("Chain test", {})

        # Verify the chain
        # Stage 2 should reference Stage 1's output
        stage2_context = agents[1].received_contexts[0]
        assert "Stage0" in stage2_context.get("previous_output", "")

        # Stage 3 should reference Stage 2's output
        stage3_context = agents[2].received_contexts[0]
        assert "Stage1" in stage3_context.get("previous_output", "")


# =============================================================================
# CONSENSUS Formation Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.agents
class TestConsensusFormation:
    """Integration tests for CONSENSUS team formation."""

    @pytest.mark.asyncio
    async def test_consensus_all_agents_execute(
        self,
        coordinator: UnifiedTeamCoordinator,
        three_agent_team: List[IntegrationTestAgent],
    ):
        """All agents execute in consensus formation."""
        for agent in three_agent_team:
            coordinator.add_member(agent)

        coordinator.set_formation(TeamFormation.CONSENSUS)

        result = await coordinator.execute_task("Consensus task", {})

        for agent in three_agent_team:
            assert len(agent.executed_tasks) == 1

    @pytest.mark.asyncio
    async def test_consensus_achieved_when_all_succeed(
        self,
        coordinator: UnifiedTeamCoordinator,
        three_agent_team: List[IntegrationTestAgent],
    ):
        """Consensus is achieved when all agents succeed."""
        for agent in three_agent_team:
            coordinator.add_member(agent)

        coordinator.set_formation(TeamFormation.CONSENSUS)

        result = await coordinator.execute_task("Consensus task", {})

        assert result["success"] is True
        assert result.get("consensus_achieved", False) is True

    @pytest.mark.asyncio
    async def test_consensus_fails_when_any_agent_fails(
        self,
        coordinator: UnifiedTeamCoordinator,
    ):
        """Consensus fails when any agent fails."""
        agent1 = IntegrationTestAgent("agent_1")
        agent2 = IntegrationTestAgent("agent_2", fail_on_task=True)
        agent3 = IntegrationTestAgent("agent_3")

        coordinator.add_member(agent1).add_member(agent2).add_member(agent3)
        coordinator.set_formation(TeamFormation.CONSENSUS)

        result = await coordinator.execute_task("Consensus task", {})

        assert result["success"] is False
        assert result.get("consensus_achieved", True) is False


# =============================================================================
# Cross-Formation Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.agents
class TestCrossFormation:
    """Tests comparing behavior across formations."""

    @pytest.mark.asyncio
    async def test_formation_affects_execution_pattern(
        self,
        coordinator: UnifiedTeamCoordinator,
    ):
        """Different formations produce different execution patterns."""
        results = {}

        for formation in [TeamFormation.SEQUENTIAL, TeamFormation.PARALLEL]:
            coord = UnifiedTeamCoordinator()
            tracker: List[str] = []

            agents = [IntegrationTestAgent(f"agent_{i}", delay=0.05) for i in range(3)]

            for agent in agents:
                coord.add_member(agent)

            coord.set_formation(formation)

            start = time.time()
            await coord.execute_task("Test task", {"execution_tracker": tracker})
            duration = time.time() - start

            results[formation] = {"duration": duration, "order": tracker.copy()}

        # Sequential should be slower than parallel
        assert (
            results[TeamFormation.SEQUENTIAL]["duration"]
            > results[TeamFormation.PARALLEL]["duration"]
        )

    @pytest.mark.asyncio
    async def test_all_formations_produce_results(
        self,
        coordinator: UnifiedTeamCoordinator,
    ):
        """All formations produce member results."""
        for formation in TeamFormation:
            coord = UnifiedTeamCoordinator()

            # For hierarchical, need a manager
            if formation == TeamFormation.HIERARCHICAL:
                manager_role = IntegrationTestRole(
                    name="manager",
                    capabilities={
                        AgentCapability.DELEGATE,
                        AgentCapability.COMMUNICATE,
                    },
                )
                manager = IntegrationTestAgent("manager", role=manager_role)
                worker = IntegrationTestAgent("worker")
                coord.add_member(manager).add_member(worker)
                coord.set_manager(manager)
            else:
                coord.add_member(IntegrationTestAgent("agent_1"))
                coord.add_member(IntegrationTestAgent("agent_2"))

            coord.set_formation(formation)

            result = await coord.execute_task("Test task", {})

            assert "member_results" in result
            # REFLECTION collapses the generator+critic loop into a single
            # synthesized result; other formations return one result per member.
            expected_min = 1 if formation == TeamFormation.REFLECTION else 2
            assert len(result["member_results"]) >= expected_min

    @pytest.mark.asyncio
    async def test_formation_switching(
        self,
        coordinator: UnifiedTeamCoordinator,
        three_agent_team: List[IntegrationTestAgent],
    ):
        """Formation can be switched between tasks."""
        for agent in three_agent_team:
            coordinator.add_member(agent)

        # Execute with sequential
        coordinator.set_formation(TeamFormation.SEQUENTIAL)
        result1 = await coordinator.execute_task("Sequential task", {})

        # Clear agent state
        for agent in three_agent_team:
            agent.executed_tasks.clear()

        # Execute with parallel
        coordinator.set_formation(TeamFormation.PARALLEL)
        result2 = await coordinator.execute_task("Parallel task", {})

        # Both should succeed
        assert result1["success"] is True
        assert result2["success"] is True
        assert result1["formation"] == "sequential"
        assert result2["formation"] == "parallel"


# =============================================================================
# Real Role Integration Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.agents
class TestRealRoleIntegration:
    """Tests with real role implementations from agent_roles.py."""

    @pytest.mark.asyncio
    async def test_team_with_real_roles(
        self,
        coordinator: UnifiedTeamCoordinator,
    ):
        """Team with actual ManagerRole, ResearcherRole, etc."""
        manager = IntegrationTestAgent("manager", role=ManagerRole())
        researcher = IntegrationTestAgent("researcher", role=ResearcherRole())
        executor = IntegrationTestAgent("executor", role=ExecutorRole())
        reviewer = IntegrationTestAgent("reviewer", role=ReviewerRole())

        coordinator.add_member(manager)
        coordinator.add_member(researcher)
        coordinator.add_member(executor)
        coordinator.add_member(reviewer)

        coordinator.set_manager(manager)
        coordinator.set_formation(TeamFormation.HIERARCHICAL)

        result = await coordinator.execute_task(
            "Implement authentication feature",
            {"target_dir": "src/auth/"},
        )

        assert result["success"] is True
        assert len(result["member_results"]) == 4

    @pytest.mark.asyncio
    async def test_role_capabilities_preserved(
        self,
        coordinator: UnifiedTeamCoordinator,
    ):
        """Agent roles preserve their capabilities."""
        researcher = IntegrationTestAgent("researcher", role=ResearcherRole())
        executor = IntegrationTestAgent("executor", role=ExecutorRole())

        coordinator.add_member(researcher).add_member(executor)
        coordinator.set_formation(TeamFormation.SEQUENTIAL)

        await coordinator.execute_task("Test capabilities", {})

        # Verify role capabilities are preserved
        assert AgentCapability.SEARCH in researcher.role.capabilities
        assert AgentCapability.WRITE in executor.role.capabilities
        assert AgentCapability.WRITE not in researcher.role.capabilities


# =============================================================================
# Persona Integration Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.agents
class TestPersonaIntegration:
    """Tests with real persona implementations."""

    @pytest.mark.asyncio
    async def test_team_with_personas(
        self,
        coordinator: UnifiedTeamCoordinator,
    ):
        """Team members can have personas."""
        senior_dev = get_persona("senior_developer")
        code_reviewer = get_persona("code_reviewer")

        agent1 = IntegrationTestAgent("dev", persona=senior_dev)
        agent2 = IntegrationTestAgent("reviewer", persona=code_reviewer)

        coordinator.add_member(agent1).add_member(agent2)
        coordinator.set_formation(TeamFormation.PIPELINE)

        result = await coordinator.execute_task("Code review task", {})

        assert result["success"] is True
        assert agent1.persona is not None
        assert agent2.persona is not None

    @pytest.mark.asyncio
    async def test_persona_formats_messages(
        self,
        coordinator: UnifiedTeamCoordinator,
    ):
        """Personas format messages according to their style."""
        formal_persona = Persona(
            name="Formal Agent",
            background="Test",
            communication_style="formal",
        )
        casual_persona = Persona(
            name="Casual Agent",
            background="Test",
            communication_style="casual",
        )

        agent1 = IntegrationTestAgent("formal", persona=formal_persona)
        agent2 = IntegrationTestAgent("casual", persona=casual_persona)

        coordinator.add_member(agent1).add_member(agent2)

        message = AgentMessage(
            sender_id="coordinator",
            recipient_id="all",
            content="hello there",
            message_type=MessageType.TASK,
        )

        responses = await coordinator.broadcast(message)

        # Responses should be formatted by persona
        assert len(responses) == 2


# =============================================================================
# Fluent API Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.agents
class TestFluentAPI:
    """Tests for fluent/chaining API."""

    def test_fluent_team_building(self):
        """Team can be built with fluent API."""
        agent1 = IntegrationTestAgent("agent_1")
        agent2 = IntegrationTestAgent("agent_2")
        manager = IntegrationTestAgent("manager", role=ManagerRole())

        coordinator = (
            UnifiedTeamCoordinator()
            .add_member(manager)
            .add_member(agent1)
            .add_member(agent2)
            .set_manager(manager)
            .set_formation(TeamFormation.HIERARCHICAL)
        )

        assert len(coordinator.members) == 3
        assert coordinator.manager == manager
        assert coordinator.formation == TeamFormation.HIERARCHICAL

    @pytest.mark.asyncio
    async def test_fluent_build_and_execute(self):
        """Build and execute team in one chain."""
        agent1 = IntegrationTestAgent("agent_1")
        agent2 = IntegrationTestAgent("agent_2")

        coordinator = (
            UnifiedTeamCoordinator()
            .add_member(agent1)
            .add_member(agent2)
            .set_formation(TeamFormation.PARALLEL)
        )

        result = await coordinator.execute_task("Fluent task", {})

        assert result["success"] is True
        assert len(result["member_results"]) == 2
