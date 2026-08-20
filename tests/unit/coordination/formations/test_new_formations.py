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

"""Tests for new formation strategies.

Tests for:
- ReflectionFormation
- DynamicRouterFormation
- MultiLevelHierarchyFormation
- AdaptiveFormation
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, Mock

from victor.coordination.formations.reflection import ReflectionFormation
from victor.coordination.formations.dynamic_router import DynamicRouterFormation
from victor.coordination.formations.multi_level_hierarchy import (
    MultiLevelHierarchyFormation,
    HierarchyNode,
)
from victor.coordination.formations.adaptive import (
    AdaptiveFormation,
    AdaptationStrategy,
)
from victor.coordination.formations.base import TeamContext
from victor.teams.types import AgentMessage, MessageType


@pytest.fixture
def mock_agent():
    """Create a mock agent."""
    agent = MagicMock()
    agent.id = "test-agent"
    agent.execute = AsyncMock(return_value="Agent response")
    return agent


@pytest.fixture
def mock_generator():
    """Create a mock generator agent for reflection formation."""
    agent = MagicMock()
    agent.id = "generator"
    agent.execute = AsyncMock(side_effect=["Draft 1", "Draft 2", "Final draft"])
    return agent


@pytest.fixture
def mock_critic():
    """Create a mock critic agent for reflection formation."""
    agent = MagicMock()
    agent.id = "critic"
    agent.execute = AsyncMock(side_effect=["Needs improvement", "Better but fix X", "Good!"])
    return agent


@pytest.fixture
def team_context():
    """Create a team context."""
    return TeamContext("test-team", "test-formation")


# ==================== ReflectionFormation Tests ====================


class TestReflectionFormation:
    """Tests for ReflectionFormation."""

    @pytest.mark.asyncio
    async def test_reflection_formation_basic_execution(
        self, team_context, mock_generator, mock_critic
    ):
        """Test basic reflection formation execution."""
        formation = ReflectionFormation(max_iterations=3)

        team_context.set("generator", mock_generator)
        team_context.set("critic", mock_critic)

        task = AgentMessage(
            sender_id="test", content="Write a solution", message_type=MessageType.TASK
        )

        results = await formation.execute([], team_context, task)

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].output == "Final draft"
        assert results[0].metadata["iterations"] == 3
        assert results[0].metadata["satisfied"] is True

    @pytest.mark.asyncio
    async def test_reflection_formation_early_termination(
        self, team_context, mock_generator, mock_critic
    ):
        """Test early termination when critic is satisfied."""
        formation = ReflectionFormation(max_iterations=5)

        team_context.set("generator", mock_generator)
        team_context.set("critic", mock_critic)

        task = AgentMessage(
            sender_id="test", content="Write a solution", message_type=MessageType.TASK
        )

        results = await formation.execute([], team_context, task)

        # Should stop after 3 iterations even though max is 5
        assert results[0].metadata["iterations"] == 3

    @pytest.mark.asyncio
    async def test_reflection_formation_custom_keywords(
        self, team_context, mock_generator, mock_critic
    ):
        """Test custom satisfaction keywords."""
        # Use custom keyword that won't appear in critic responses
        formation = ReflectionFormation(max_iterations=3, satisfaction_keywords=["perfect"])

        team_context.set("generator", mock_generator)
        team_context.set("critic", mock_critic)

        task = AgentMessage(
            sender_id="test", content="Write a solution", message_type=MessageType.TASK
        )

        results = await formation.execute([], team_context, task)

        # Should run all iterations without early termination
        assert results[0].metadata["iterations"] == 3
        assert results[0].metadata["satisfied"] is False

    @pytest.mark.asyncio
    async def test_reflection_formation_missing_generator(self, team_context, mock_critic):
        """Test error handling when generator is missing."""
        formation = ReflectionFormation()

        team_context.set("critic", mock_critic)
        # Don't set generator

        task = AgentMessage(
            sender_id="test", content="Write a solution", message_type=MessageType.TASK
        )

        results = await formation.execute([], team_context, task)

        assert len(results) == 1
        assert results[0].success is False
        assert "Missing generator or critic" in results[0].error

    @pytest.mark.asyncio
    async def test_reflection_formation_generator_failure(
        self, team_context, mock_generator, mock_critic
    ):
        """Test error handling when generator fails."""
        formation = ReflectionFormation()

        team_context.set("generator", mock_generator)
        team_context.set("critic", mock_critic)

        # Make generator fail on second call
        mock_generator.execute.side_effect = ["Draft 1", Exception("Generator failed")]

        task = AgentMessage(
            sender_id="test", content="Write a solution", message_type=MessageType.TASK
        )

        results = await formation.execute([], team_context, task)

        assert results[0].success is False
        assert "Generator failed" in results[0].error

    def test_reflection_formation_validate_context(self, team_context):
        """Test context validation."""
        formation = ReflectionFormation()

        # Empty context should fail validation
        assert formation.validate_context(team_context) is False

        # Context with both agents should pass
        mock_generator = MagicMock()
        mock_critic = MagicMock()
        team_context.set("generator", mock_generator)
        team_context.set("critic", mock_critic)

        assert formation.validate_context(team_context) is True

    def test_reflection_formation_get_required_roles(self):
        """Test getting required roles."""
        formation = ReflectionFormation()
        roles = formation.get_required_roles()

        assert roles == ["generator", "critic"]

    def test_reflection_formation_supports_early_termination(self):
        """Test early termination support."""
        formation = ReflectionFormation()
        assert formation.supports_early_termination() is True


# ==================== DynamicRouterFormation Tests ====================


class TestDynamicRouterFormation:
    """Tests for DynamicRouterFormation."""

    @pytest.fixture
    def mock_coder(self):
        """Create a mock coder agent."""
        agent = MagicMock()
        agent.id = "coder"
        agent.execute = AsyncMock(return_value="Code solution")
        return agent

    @pytest.fixture
    def mock_researcher(self):
        """Create a mock researcher agent."""
        agent = MagicMock()
        agent.id = "researcher"
        agent.execute = AsyncMock(return_value="Research results")
        return agent

    @pytest.fixture
    def mock_analyst(self):
        """Create a mock analyst agent."""
        agent = MagicMock()
        agent.id = "analyst"
        agent.execute = AsyncMock(return_value="Analysis results")
        return agent

    @pytest.mark.asyncio
    async def test_dynamic_router_category_routing(
        self, team_context, mock_coder, mock_researcher, mock_analyst
    ):
        """Test routing based on task category."""
        formation = DynamicRouterFormation(
            category_to_role={"coding": "coder", "research": "researcher"}
        )

        team_context.set("coder", mock_coder)
        team_context.set("researcher", mock_researcher)
        team_context.set("analyst", mock_analyst)

        # Task with coding keyword
        task = AgentMessage(
            sender_id="test",
            content="Implement a function to parse data",
            message_type=MessageType.TASK,
        )

        results = await formation.execute([], team_context, task)

        assert results[0].success is True
        assert results[0].metadata["selected_agent"] == "coder"
        assert results[0].output == "Code solution"

    @pytest.mark.asyncio
    async def test_dynamic_router_keyword_fallback(
        self, team_context, mock_coder, mock_researcher, mock_analyst
    ):
        """Test keyword-based fallback routing."""
        formation = DynamicRouterFormation()

        team_context.set("coder", mock_coder)
        team_context.set("researcher", mock_researcher)
        team_context.set("analyst", mock_analyst)

        # Task with 'analyze' keyword
        task = AgentMessage(
            sender_id="test",
            content="Analyze the data trends",
            message_type=MessageType.TASK,
        )

        results = await formation.execute([], team_context, task)

        assert results[0].success is True
        assert results[0].metadata["selected_agent"] == "analyst"
        assert results[0].metadata["routing_method"] == "keyword"

    @pytest.mark.asyncio
    async def test_dynamic_router_custom_mappings(self, team_context, mock_coder, mock_researcher):
        """Test custom category and keyword mappings."""
        formation = DynamicRouterFormation(
            category_to_role={"custom": "coder"},
            keyword_to_agent={"custom_keyword": "coder"},
        )

        team_context.set("coder", mock_coder)
        team_context.set("researcher", mock_researcher)

        # Task with custom keyword
        task = AgentMessage(
            sender_id="test",
            content="Use custom_keyword to process",
            message_type=MessageType.TASK,
        )

        results = await formation.execute([], team_context, task)

        assert results[0].success is True
        assert results[0].metadata["selected_agent"] == "coder"

    @pytest.mark.asyncio
    async def test_dynamic_router_no_matching_agent(self, team_context, mock_coder):
        """Test routing when no specific agent matches."""
        formation = DynamicRouterFormation()

        team_context.set("coder", mock_coder)

        # Task without clear routing signals
        task = AgentMessage(
            sender_id="test",
            content="Do something generic",
            message_type=MessageType.TASK,
        )

        results = await formation.execute([], team_context, task)

        # Should still route to first available agent
        assert results[0].success is True

    @pytest.mark.asyncio
    async def test_dynamic_router_agent_failure(self, team_context, mock_coder, mock_researcher):
        """Test error handling when selected agent fails."""
        formation = DynamicRouterFormation()

        team_context.set("coder", mock_coder)

        # Make coder fail
        mock_coder.execute.side_effect = Exception("Execution failed")

        task = AgentMessage(
            sender_id="test",
            content="Implement a function",
            message_type=MessageType.TASK,
        )

        results = await formation.execute([], team_context, task)

        assert results[0].success is False
        assert "Execution failed" in results[0].error

    def test_dynamic_router_validate_context(self, team_context):
        """Test context validation."""
        formation = DynamicRouterFormation()

        # Empty context should fail validation
        assert formation.validate_context(team_context) is False

        # Context with agent should pass
        mock_agent = MagicMock()
        mock_agent.execute = MagicMock()
        mock_agent.id = "test"
        team_context.set("agent", mock_agent)

        assert formation.validate_context(team_context) is True


# ==================== MultiLevelHierarchyFormation Tests ====================


class TestMultiLevelHierarchyFormation:
    """Tests for MultiLevelHierarchyFormation."""

    @pytest.mark.asyncio
    async def test_hierarchy_single_node(self, team_context, mock_agent):
        """Test hierarchy with single node (no children)."""
        leaf = HierarchyNode(agent=mock_agent)
        formation = MultiLevelHierarchyFormation(hierarchy=leaf)

        task = AgentMessage(
            sender_id="test",
            content="Complete this task",
            message_type=MessageType.TASK,
        )

        results = await formation.execute([], team_context, task)

        assert results[0].success is True
        assert results[0].metadata["hierarchy_levels"] == 1
        assert results[0].metadata["nodes_executed"] == 1
        mock_agent.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_hierarchy_two_levels(self, team_context, mock_agent):
        """Test hierarchy with two levels (coordinator + workers)."""
        # Create leaf nodes
        worker1 = MagicMock()
        worker1.id = "worker1"
        worker1.execute = AsyncMock(return_value="Worker 1 result")

        worker2 = MagicMock()
        worker2.id = "worker2"
        worker2.execute = AsyncMock(return_value="Worker 2 result")

        # Create coordinator with children
        coordinator = MagicMock()
        coordinator.id = "coordinator"
        coordinator.execute = AsyncMock(return_value="Coordinator result")

        leaf1 = HierarchyNode(agent=worker1)
        leaf2 = HierarchyNode(agent=worker2)
        lead = HierarchyNode(agent=coordinator, children=[leaf1, leaf2])

        formation = MultiLevelHierarchyFormation(hierarchy=lead)

        task = AgentMessage(
            sender_id="test",
            content="Complete this task",
            message_type=MessageType.TASK,
        )

        results = await formation.execute([], team_context, task)

        assert results[0].success is True
        assert results[0].metadata["hierarchy_levels"] == 2
        assert results[0].metadata["nodes_executed"] == 3

    @pytest.mark.asyncio
    async def test_hierarchy_three_levels(self, team_context):
        """Test hierarchy with three levels."""
        # Create workers
        worker1 = MagicMock()
        worker1.id = "worker1"
        worker1.execute = AsyncMock(return_value="Worker 1")

        worker2 = MagicMock()
        worker2.id = "worker2"
        worker2.execute = AsyncMock(return_value="Worker 2")

        # Create team lead
        lead = MagicMock()
        lead.id = "team_lead"
        lead.execute = AsyncMock(return_value="Lead result")

        # Create coordinator
        coordinator = MagicMock()
        coordinator.id = "coordinator"
        coordinator.execute = AsyncMock(return_value="Coordinator result")

        # Build hierarchy
        leaf1 = HierarchyNode(agent=worker1)
        leaf2 = HierarchyNode(agent=worker2)
        lead_node = HierarchyNode(agent=lead, children=[leaf1, leaf2])
        coordinator_node = HierarchyNode(agent=coordinator, children=[lead_node])

        formation = MultiLevelHierarchyFormation(hierarchy=coordinator_node)

        task = AgentMessage(
            sender_id="test",
            content="Complete complex task",
            message_type=MessageType.TASK,
        )

        results = await formation.execute([], team_context, task)

        assert results[0].success is True
        assert results[0].metadata["hierarchy_levels"] == 3
        assert results[0].metadata["nodes_executed"] == 4

    def test_hierarchy_node_depth(self, mock_agent):
        """Test HierarchyNode depth calculation."""
        # Single node
        leaf = HierarchyNode(agent=mock_agent)
        assert leaf.get_depth() == 1

        # Two levels
        coordinator = MagicMock()
        two_level = HierarchyNode(agent=coordinator, children=[leaf])
        assert two_level.get_depth() == 2

    def test_hierarchy_node_parent_reference(self, mock_agent):
        """Test HierarchyNode parent references are set correctly."""
        coordinator = MagicMock()
        worker = MagicMock()

        leaf = HierarchyNode(agent=worker)
        lead = HierarchyNode(agent=coordinator, children=[leaf])

        assert leaf.parent is lead
        assert lead.level == 0
        assert leaf.level == 1

    def test_hierarchy_validate_valid(self, mock_agent):
        """Test validation of valid hierarchy."""
        leaf = HierarchyNode(agent=mock_agent)
        formation = MultiLevelHierarchyFormation(hierarchy=leaf)

        team_context = TeamContext("test", "hierarchy")
        assert formation.validate_context(team_context) is True

    def test_hierarchy_validate_invalid(self):
        """Test validation of invalid hierarchy."""
        # Node with None agent
        leaf = HierarchyNode(agent=None)
        formation = MultiLevelHierarchyFormation(hierarchy=leaf)

        team_context = TeamContext("test", "hierarchy")
        assert formation.validate_context(team_context) is False

    @pytest.mark.asyncio
    async def test_hierarchy_task_splitting_line_strategy(self, team_context):
        """Test task splitting with line strategy."""
        worker1 = MagicMock()
        worker1.id = "worker1"
        worker1.execute = AsyncMock(return_value="Part 1")

        worker2 = MagicMock()
        worker2.id = "worker2"
        worker2.execute = AsyncMock(return_value="Part 2")

        lead = MagicMock()
        lead.id = "lead"
        lead.execute = AsyncMock(return_value="Lead result")

        leaf1 = HierarchyNode(agent=worker1)
        leaf2 = HierarchyNode(agent=worker2)
        lead_node = HierarchyNode(agent=lead, children=[leaf1, leaf2])

        formation = MultiLevelHierarchyFormation(hierarchy=lead_node, split_strategy="line")

        # Multi-line task
        task = AgentMessage(
            sender_id="test",
            content="Line 1\nLine 2\nLine 3\nLine 4",
            message_type=MessageType.TASK,
        )

        results = await formation.execute([], team_context, task)

        assert results[0].success is True
        # Both workers should be called
        worker1.execute.assert_called_once()
        worker2.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_hierarchy_result_aggregation(self, team_context):
        """Test result aggregation from child nodes."""
        worker1 = MagicMock()
        worker1.id = "worker1"
        worker1.execute = AsyncMock(return_value="Result 1")

        worker2 = MagicMock()
        worker2.id = "worker2"
        worker2.execute = AsyncMock(return_value="Result 2")

        lead = MagicMock()
        lead.id = "lead"
        lead.execute = AsyncMock(return_value="Lead result")

        leaf1 = HierarchyNode(agent=worker1)
        leaf2 = HierarchyNode(agent=worker2)
        lead_node = HierarchyNode(agent=lead, children=[leaf1, leaf2])

        formation = MultiLevelHierarchyFormation(hierarchy=lead_node)

        task = AgentMessage(
            sender_id="test", content="Complete task", message_type=MessageType.TASK
        )

        results = await formation.execute([], team_context, task)

        assert results[0].success is True
        # Result should be aggregated
        assert results[0].output != ""


# ==================== AdaptiveFormation Tests ====================


class TestAdaptiveFormation:
    """Tests for AdaptiveFormation."""

    @pytest.fixture
    def mock_agents(self):
        """Create multiple mock agents."""
        agents = []
        for i in range(3):
            agent = MagicMock()
            agent.id = f"agent{i}"
            agent.execute = AsyncMock(return_value=f"Result from agent{i}")
            agents.append(agent)
        return agents

    @pytest.mark.asyncio
    async def test_adaptive_formation_initial_selection(self, team_context, mock_agents):
        """Test initial formation selection based on task size."""
        formation = AdaptiveFormation(adaptation_strategy="performance")

        for agent in mock_agents:
            team_context.set(agent.id, agent)

        # Small task - should use orchestration
        small_task = AgentMessage(
            sender_id="test", content="Small task", message_type=MessageType.TASK
        )
        results = await formation.execute([], team_context, small_task)

        assert results[0].success is True
        assert "current_formation" in results[0].metadata

    @pytest.mark.asyncio
    async def test_adaptive_formation_large_task(self, team_context, mock_agents):
        """Test formation selection for large tasks."""
        formation = AdaptiveFormation(adaptation_strategy="performance")

        for agent in mock_agents:
            team_context.set(agent.id, agent)

        # Large task (>1000 chars) - should use hierarchy
        large_task = AgentMessage(
            sender_id="test", content="X" * 1500, message_type=MessageType.TASK
        )
        results = await formation.execute([], team_context, large_task)

        assert results[0].success is True
        assert results[0].metadata["current_formation"] in [
            "sequential",
            "hierarchical",
            "consensus",
        ]

    @pytest.mark.asyncio
    async def test_adaptive_formation_performance_switching(self, team_context):
        """Test formation switching based on performance."""
        formation = AdaptiveFormation(
            adaptation_strategy="performance",
            performance_threshold=0.5,
            max_duration_seconds=1.0,
        )

        # Add agents to context
        for i in range(3):
            agent = MagicMock()
            agent.id = f"agent{i}"
            # Slow execution to trigger switching
            agent.execute = AsyncMock(
                side_effect=lambda x, **kwargs: None  # Simulate slow execution
            )
            team_context.set(agent.id, agent)

        task = AgentMessage(sender_id="test", content="Test task", message_type=MessageType.TASK)

        results = await formation.execute([], team_context, task)

        # Check metadata
        assert "current_formation" in results[0].metadata
        assert "performance_score" in results[0].metadata

    @pytest.mark.asyncio
    async def test_adaptive_formation_error_rate_strategy(self, team_context):
        """Test error rate adaptation strategy."""
        formation = AdaptiveFormation(adaptation_strategy="error_rate", max_switches=2)

        # Add agents
        for i in range(3):
            agent = MagicMock()
            agent.id = f"agent{i}"
            agent.execute = AsyncMock(return_value=f"Result {i}")
            team_context.set(agent.id, agent)

        task = AgentMessage(sender_id="test", content="Test task", message_type=MessageType.TASK)

        results = await formation.execute([], team_context, task)

        assert results[0].success is True
        assert results[0].metadata["adaptation_strategy"] == "error_rate"

    @pytest.mark.asyncio
    async def test_adaptive_formation_max_switches_limit(self, team_context):
        """Test that formation switching respects max_switches limit."""
        formation = AdaptiveFormation(adaptation_strategy="performance", max_switches=1)

        # Add agents
        for i in range(3):
            agent = MagicMock()
            agent.id = f"agent{i}"
            agent.execute = AsyncMock(return_value=f"Result {i}")
            team_context.set(agent.id, agent)

        task = AgentMessage(sender_id="test", content="Test task", message_type=MessageType.TASK)

        results = await formation.execute([], team_context, task)

        # Should not exceed max_switches
        assert results[0].metadata["formation_switches"] <= 1

    @pytest.mark.asyncio
    async def test_adaptive_formation_formation_history_tracking(self, team_context, mock_agents):
        """Test that formation history is tracked correctly."""
        formation = AdaptiveFormation(adaptation_strategy="performance")

        for agent in mock_agents:
            team_context.set(agent.id, agent)

        task = AgentMessage(sender_id="test", content="Test task", message_type=MessageType.TASK)

        results = await formation.execute([], team_context, task)

        # Should have formation history
        assert "formation_history" in results[0].metadata
        assert len(results[0].metadata["formation_history"]) >= 1

    def test_adaptive_formation_validate_context(self, team_context):
        """Test context validation."""
        formation = AdaptiveFormation()

        # Empty context should fail validation
        assert formation.validate_context(team_context) is False

        # Context with 1 agent should pass
        mock_agent = MagicMock()
        mock_agent.execute = MagicMock()
        mock_agent.id = "test"
        team_context.set("agent", mock_agent)

        assert formation.validate_context(team_context) is True

    def test_adaptive_formation_get_performance_summary(self, team_context):
        """Test getting performance summary."""
        formation = AdaptiveFormation()

        summary = formation.get_performance_summary()

        assert "formations_tried" in summary
        assert "total_switches" in summary
        assert "current_formation" in summary
        assert "adaptation_strategy" in summary

    def test_adaptive_formation_supports_early_termination(self):
        """Test early termination support."""
        formation = AdaptiveFormation()
        assert formation.supports_early_termination() is True

    @pytest.mark.asyncio
    async def test_adaptive_formation_custom_formation_cycle(self, team_context):
        """Test custom formation cycle."""
        formation = AdaptiveFormation(formation_cycle=["sequential", "consensus"])

        # Add agents
        for i in range(2):
            agent = MagicMock()
            agent.id = f"agent{i}"
            agent.execute = AsyncMock(return_value=f"Result {i}")
            team_context.set(agent.id, agent)

        task = AgentMessage(sender_id="test", content="Test task", message_type=MessageType.TASK)

        results = await formation.execute([], team_context, task)

        assert results[0].success is True
        # Should use one of the formations in the custom cycle
        assert results[0].metadata["current_formation"] in [
            "sequential",
            "consensus",
        ]

    @pytest.mark.asyncio
    async def test_adaptive_formation_respects_initial_formation_hint(self, team_context):
        """Explicit formation hints should override the task-size heuristic."""
        formation = AdaptiveFormation()

        for i in range(2):
            agent = MagicMock()
            agent.id = f"agent{i}"
            agent.execute = AsyncMock(return_value=f"Result {i}")
            team_context.set(agent.id, agent)

        team_context.set("formation_hint", "consensus")
        small_task = AgentMessage(
            sender_id="test",
            content="Small task",
            message_type=MessageType.TASK,
        )

        results = await formation.execute([], team_context, small_task)

        assert results[0].success is True
        assert results[0].metadata["current_formation"] == "consensus"
