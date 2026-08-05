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

"""Unit tests for the team-step workflow adapter."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

import victor.framework.workflows.nodes as nodes_module
from victor.framework.workflows.nodes import TeamStep, TeamStepConfig
from victor.framework.state_merging import MergeMode, StateMergeError
from victor.teams.types import TeamConfig, TeamFormation, TeamMember
from victor.agent.subagents import SubAgentRole


class TestTeamStepCompatibilityAliases:
    """Tests for deprecated TeamNode* compatibility aliases."""

    def test_preferred_aliases(self):
        """Legacy config alias emits a deprecation warning and maps to TeamStepConfig."""
        with pytest.warns(DeprecationWarning, match="TeamNodeConfig"):
            alias = nodes_module.TeamNodeConfig

        assert TeamStepConfig is alias

    def test_preferred_step_alias(self):
        """Legacy step alias emits a deprecation warning and maps to TeamStep."""
        with pytest.warns(DeprecationWarning, match="TeamNode"):
            alias = nodes_module.TeamNode

        assert TeamStep is alias

    def test_alias_warning_mentions_removal_milestone(self):
        """Legacy alias warnings should publish the concrete removal milestone."""
        with pytest.warns(DeprecationWarning) as caught:
            _ = nodes_module.TeamNode

        message = str(caught[0].message)
        assert "v0.9.0" in message
        assert "2027-03-31" in message
        assert "TeamStep" in message


class TestTeamStepConfig:
    """Tests for TeamStepConfig dataclass."""

    def test_default_config(self):
        """Test creating default config."""
        config = TeamStepConfig()

        assert config.timeout_seconds is None
        assert config.merge_strategy == "dict"
        assert config.merge_mode == MergeMode.TEAM_WINS
        assert config.output_key == "team_result"
        assert config.continue_on_error is True
        assert config.validate_before_merge is False
        assert config.required_keys is None

    def test_custom_config(self):
        """Test creating custom config."""
        config = TeamStepConfig(
            timeout_seconds=300,
            merge_strategy="list",
            merge_mode=MergeMode.GRAPH_WINS,
            output_key="custom_result",
            continue_on_error=False,
            validate_before_merge=True,
            required_keys=["key1", "key2"],
        )

        assert config.timeout_seconds == 300
        assert config.merge_strategy == "list"
        assert config.merge_mode == MergeMode.GRAPH_WINS
        assert config.output_key == "custom_result"
        assert config.continue_on_error is False
        assert config.validate_before_merge is True
        assert config.required_keys == ["key1", "key2"]


class TestTeamStep:
    """Tests for TeamStep."""

    @pytest.fixture
    def sample_members(self):
        """Create sample team members."""
        return [
            TeamMember(
                id="researcher",
                role=SubAgentRole.RESEARCHER,
                name="Researcher",
                goal="Find information",
                tool_budget=10,
            ),
            TeamMember(
                id="reviewer",
                role=SubAgentRole.REVIEWER,
                name="Reviewer",
                goal="Review and refine report",
                tool_budget=15,
            ),
        ]

    @pytest.fixture
    def sample_config(self):
        """Create sample team-step config."""
        return TeamStepConfig(
            timeout_seconds=300,
            merge_strategy="dict",
            merge_mode=MergeMode.TEAM_WINS,
            output_key="team_result",
        )

    @pytest.fixture
    def sample_team_step(self, sample_members, sample_config):
        """Create a sample TeamStep."""
        return TeamStep(
            id="test_team",
            name="Test Team",
            goal="Test goal",
            team_formation=TeamFormation.SEQUENTIAL,
            members=sample_members,
            config=sample_config,
            max_iterations=30,
            total_tool_budget=50,
        )

    def test_team_step_creation(self, sample_team_step):
        """Test creating a TeamStep."""
        assert sample_team_step.id == "test_team"
        assert sample_team_step.name == "Test Team"
        assert sample_team_step.goal == "Test goal"
        assert sample_team_step.team_formation == TeamFormation.SEQUENTIAL
        assert len(sample_team_step.members) == 2
        assert sample_team_step.max_iterations == 30
        assert sample_team_step.total_tool_budget == 50

    def test_team_step_to_dict(self, sample_team_step):
        """Test serializing TeamStep to dictionary."""
        data = sample_team_step.to_dict()

        assert data["id"] == "test_team"
        assert data["name"] == "Test Team"
        assert data["type"] == "team"
        assert data["goal"] == "Test goal"
        assert data["team_formation"] == "sequential"
        assert len(data["members"]) == 2
        assert data["max_iterations"] == 30
        assert data["total_tool_budget"] == 50

    def test_team_step_from_dict(self):
        """Test deserializing TeamStep from dictionary."""
        data = {
            "id": "test_team",
            "name": "Test Team",
            "goal": "Test goal",
            "team_formation": "sequential",
            "members": [
                {
                    "id": "researcher",
                    "role": "researcher",
                    "name": "Researcher",
                    "goal": "Find information",
                    "tool_budget": 10,
                },
                {
                    "id": "reviewer",
                    "role": "reviewer",
                    "name": "Reviewer",
                    "goal": "Review and refine report",
                    "tool_budget": 15,
                },
            ],
            "config": {
                "timeout_seconds": 300,
                "merge_strategy": "dict",
                "merge_mode": "team_wins",
                "output_key": "team_result",
            },
            "max_iterations": 30,
            "total_tool_budget": 50,
        }

        node = TeamStep.from_dict(data)

        assert node.id == "test_team"
        assert node.name == "Test Team"
        assert node.goal == "Test goal"
        assert node.team_formation == TeamFormation.SEQUENTIAL
        assert len(node.members) == 2
        assert node.members[0].role == SubAgentRole.RESEARCHER
        assert node.members[1].role == SubAgentRole.REVIEWER
        assert node.max_iterations == 30
        assert node.total_tool_budget == 50

    def test_build_goal_with_substitution(self, sample_members, sample_config):
        """Test building goal with context variable substitution."""
        # Create a node with goal containing placeholders
        node = TeamStep(
            id="test_team",
            name="Test Team",
            goal="Work on: ${user_task} with priority: ${priority}",
            team_formation=TeamFormation.SEQUENTIAL,
            members=sample_members,
            config=sample_config,
            max_iterations=30,
            total_tool_budget=50,
        )

        graph_state = {
            "user_task": "Implement feature X",
            "priority": "high",
        }

        goal = node._build_goal(graph_state)

        assert "user_task" not in goal  # Should be substituted
        assert "Implement feature X" in goal
        assert "high" in goal

    def test_extract_context(self, sample_team_step):
        """Test extracting context from graph state."""
        graph_state = {
            "user_task": "Implement X",
            "_internal": "secret",
            "_task": "task",
        }

        context = sample_team_step._extract_context(graph_state)

        # Should include user_task
        assert "user_task" in context
        # Should exclude _internal
        assert "_internal" not in context
        # Should include _task (exception)
        assert "_task" in context

    @pytest.mark.asyncio
    async def test_execute_async_success(self, sample_team_step):
        """Test successful team execution."""
        # Mock the team coordinator
        mock_coordinator = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.final_output = "Team completed successfully"
        mock_result.formation = TeamFormation.SEQUENTIAL
        mock_result.total_duration = 10.0
        mock_result.total_tool_calls = 25
        mock_result.member_results = {}
        mock_result.shared_context = {}

        with patch(
            "victor.teams.create_coordinator",
            return_value=mock_coordinator,
        ):
            with patch.object(
                mock_coordinator,
                "run",
            ) as mock_run:
                mock_run.return_value = mock_result

                graph_state = {"user_task": "Test task"}
                result = await sample_team_step.execute_async(None, graph_state)

                # Should have team result
                assert "team_result" in result
                assert result["team_result"]["success"] is True
                # Note: actual output format may differ, just check success

    @pytest.mark.asyncio
    async def test_execute_team_uses_delegate_follow_up_capability(self, sample_team_step):
        """TeamStep should execute delegate follow-up contracts through the protocol."""

        class FollowUpCoordinator:
            def __init__(self) -> None:
                self.seen_contract = None
                self.seen_step_id = None
                self.execute_team_config = AsyncMock(side_effect=AssertionError("wrong path"))

            async def execute_follow_up_request(self, request):
                raise AssertionError(f"unexpected direct request: {request}")

            async def execute_follow_up_contract(self, contract, *, step_id=None):
                self.seen_contract = contract
                self.seen_step_id = step_id
                return {
                    "success": True,
                    "final_output": "retry completed",
                    "formation": "parallel",
                    "total_tool_calls": 3,
                    "shared_context": {"delegate_selected_step": "resume_delegate_retry"},
                    "member_results": {
                        "tester": {
                            "member_id": "tester",
                            "success": True,
                            "output": "fixed validation",
                            "tool_calls_used": 3,
                        }
                    },
                }

        coordinator = FollowUpCoordinator()
        follow_up_contract = {
            "primary_step_id": "resume_delegate_retry",
            "next_steps": [],
        }
        team_config = TeamConfig(
            name="Delegate Follow Up",
            goal="Resume delegate work",
            members=sample_team_step.members,
            formation=TeamFormation.PARALLEL,
            shared_context={
                "delegate_follow_up_contract": follow_up_contract,
                "delegate_next_step_id": "resume_delegate_retry",
            },
        )

        result = await sample_team_step._execute_team(coordinator, team_config)

        assert coordinator.seen_contract == follow_up_contract
        assert coordinator.seen_step_id == "resume_delegate_retry"
        coordinator.execute_team_config.assert_not_called()
        assert result.success is True
        assert result.final_output == "retry completed"
        assert result.formation == TeamFormation.PARALLEL
        assert result.total_tool_calls == 3
        assert result.shared_context["delegate_selected_step"] == "resume_delegate_retry"
        assert result.member_results["tester"].output == "fixed validation"

    @pytest.mark.asyncio
    async def test_execute_async_resumes_delegate_follow_up_contract(self, sample_team_step):
        """Workflow execution should route follow-up graph state through the capability."""

        class FollowUpCoordinator:
            def __init__(self) -> None:
                self.seen_contract = None
                self.seen_step_id = None

            async def execute_follow_up_request(self, request):
                raise AssertionError(f"unexpected direct request: {request}")

            async def execute_follow_up_contract(self, contract, *, step_id=None):
                self.seen_contract = contract
                self.seen_step_id = step_id
                return {
                    "success": True,
                    "final_output": "merge approved",
                    "shared_context": {"delegate_selected_step": "approve_merge_execution"},
                    "member_results": {},
                }

        coordinator = FollowUpCoordinator()
        follow_up_contract = {
            "primary_step_id": "approve_merge_execution",
            "next_steps": [{"step_id": "approve_merge_execution"}],
        }

        with patch("victor.teams.create_coordinator", return_value=coordinator):
            result = await sample_team_step.execute_async(
                None,
                {
                    "delegate_follow_up_contract": follow_up_contract,
                    "delegate_next_step_id": "approve_merge_execution",
                },
            )

        assert coordinator.seen_contract == follow_up_contract
        assert coordinator.seen_step_id == "approve_merge_execution"
        assert result["team_result"]["success"] is True
        assert result["team_output"] == "merge approved"
        assert result["_team_shared_context"]["delegate_selected_step"] == "approve_merge_execution"

    @pytest.mark.asyncio
    async def test_execute_async_timeout(self, sample_team_step):
        """Test team execution with timeout."""

        # Mock that times out
        async def timeout_execute(*args, **kwargs):
            await asyncio.sleep(2)  # Longer than timeout
            return MagicMock()

        with patch(
            "victor.teams.create_coordinator",
            return_value=MagicMock(),
        ):
            with patch.object(
                sample_team_step,
                "_execute_team",
                side_effect=timeout_execute,
            ):
                # Set short timeout
                sample_team_step.config.timeout_seconds = 0.1

                graph_state = {"user_task": "Test task"}
                result = await sample_team_step.execute_async(None, graph_state)

                # Should have error but continue (continue_on_error=True)
                assert "_error" in result
                assert "_timeout" in result
                assert "timed out" in result["_error"].lower()

    @pytest.mark.asyncio
    async def test_execute_async_exception_continue(self, sample_team_step):
        """Test team execution with exception and continue_on_error."""
        with patch(
            "victor.teams.create_coordinator",
            return_value=MagicMock(),
        ):
            with patch.object(
                sample_team_step,
                "_execute_team",
                side_effect=Exception("Team execution failed"),
            ):
                sample_team_step.config.continue_on_error = True
                graph_state = {"user_task": "Test task"}
                result = await sample_team_step.execute_async(None, graph_state)

                # Should have error but continue
                assert "_error" in result
                assert "Team execution failed" in result["_error"]

    @pytest.mark.asyncio
    async def test_execute_async_exception_raise(self, sample_team_step):
        """Test team execution with exception and continue_on_error=False."""
        with patch(
            "victor.teams.create_coordinator",
            return_value=MagicMock(),
        ):
            with patch.object(
                sample_team_step,
                "_execute_team",
                side_effect=Exception("Team execution failed"),
            ):
                sample_team_step.config.continue_on_error = False
                graph_state = {"user_task": "Test task"}

                with pytest.raises(Exception, match="Team execution failed"):
                    await sample_team_step.execute_async(None, graph_state)

    def test_merge_team_result_team_wins(self, sample_team_step):
        """Test merging team result with team_wins mode."""
        from victor.teams.types import TeamResult

        graph_state = {"key": "graph_value"}
        team_result = MagicMock()
        team_result.success = True
        team_result.final_output = "Team output"
        team_result.shared_context = {"key": "team_value"}
        team_result.formation = TeamFormation.SEQUENTIAL

        result = sample_team_step._merge_team_result(graph_state, team_result)

        # Team wins on conflict
        assert result["key"] == "team_value"

    def test_merge_team_result_graph_wins(self, sample_team_step):
        """Test merging team result with graph_wins mode."""
        from victor.teams.types import TeamResult

        sample_team_step.config.merge_mode = MergeMode.GRAPH_WINS

        graph_state = {"key": "graph_value"}
        team_result = MagicMock()
        team_result.success = True
        team_result.final_output = "Team output"
        team_result.shared_context = {"key": "team_value"}
        team_result.formation = TeamFormation.SEQUENTIAL

        result = sample_team_step._merge_team_result(graph_state, team_result)

        # Graph wins on conflict
        assert result["key"] == "graph_value"

    def test_merge_team_result_error_mode(self, sample_team_step):
        """Test merging team result with error mode raises on conflict."""
        from victor.teams.types import TeamResult

        sample_team_step.config.merge_mode = MergeMode.ERROR

        graph_state = {"key": "graph_value"}
        team_result = MagicMock()
        team_result.success = True
        team_result.final_output = "Team output"
        team_result.shared_context = {"key": "team_value"}
        team_result.formation = TeamFormation.SEQUENTIAL

        with pytest.raises(StateMergeError):
            sample_team_step._merge_team_result(graph_state, team_result)


class TestTeamStepFormations:
    """Tests for different team formations."""

    @pytest.fixture
    def hierarchical_members(self):
        """Create hierarchical team members."""
        return [
            TeamMember(
                id="manager",
                role=SubAgentRole.PLANNER,
                name="Manager",
                goal="Coordinate work",
                is_manager=True,
                can_delegate=True,
                tool_budget=10,
            ),
            TeamMember(
                id="worker1",
                role=SubAgentRole.EXECUTOR,
                name="Worker 1",
                goal="Execute tasks",
                reports_to="manager",
                tool_budget=20,
            ),
            TeamMember(
                id="worker2",
                role=SubAgentRole.EXECUTOR,
                name="Worker 2",
                goal="Execute tasks",
                reports_to="manager",
                tool_budget=20,
            ),
        ]

    def test_sequential_formation(self, hierarchical_members):
        """Test creating a sequential team step."""
        node = TeamStep(
            id="sequential_team",
            name="Sequential Team",
            goal="Execute sequentially",
            team_formation=TeamFormation.SEQUENTIAL,
            members=hierarchical_members[:2],
        )

        assert node.team_formation == TeamFormation.SEQUENTIAL

    def test_parallel_formation(self, hierarchical_members):
        """Test creating a parallel team step."""
        node = TeamStep(
            id="parallel_team",
            name="Parallel Team",
            goal="Execute in parallel",
            team_formation=TeamFormation.PARALLEL,
            members=hierarchical_members,
        )

        assert node.team_formation == TeamFormation.PARALLEL

    def test_hierarchical_formation(self, hierarchical_members):
        """Test creating a hierarchical team step."""
        node = TeamStep(
            id="hierarchical_team",
            name="Hierarchical Team",
            goal="Execute with hierarchy",
            team_formation=TeamFormation.HIERARCHICAL,
            members=hierarchical_members,
        )

        assert node.team_formation == TeamFormation.HIERARCHICAL

    def test_pipeline_formation(self, hierarchical_members):
        """Test creating a pipeline team step."""
        node = TeamStep(
            id="pipeline_team",
            name="Pipeline Team",
            goal="Execute as pipeline",
            team_formation=TeamFormation.PIPELINE,
            members=hierarchical_members[:2],
        )

        assert node.team_formation == TeamFormation.PIPELINE

    def test_consensus_formation(self, hierarchical_members):
        """Test creating a consensus team step."""
        node = TeamStep(
            id="consensus_team",
            name="Consensus Team",
            goal="Execute with consensus",
            team_formation=TeamFormation.CONSENSUS,
            members=hierarchical_members[:3],
        )

        assert node.team_formation == TeamFormation.CONSENSUS


class TestTeamStepSyncExecute:
    """Tests for synchronous execute method."""

    @pytest.fixture
    def simple_team_step(self):
        """Create a simple team step for sync tests."""
        return TeamStep(
            id="test_team",
            name="Test Team",
            goal="Test goal",
            team_formation=TeamFormation.SEQUENTIAL,
            members=[
                TeamMember(
                    id="member1",
                    role=SubAgentRole.EXECUTOR,
                    name="Member 1",
                    goal="Execute",
                    tool_budget=10,
                )
            ],
        )

    def test_sync_execute_wraps_async(self, simple_team_step):
        """Test that sync execute properly wraps async execute."""
        coro = object()

        with (
            patch.object(
                simple_team_step,
                "execute_async",
                new=Mock(return_value=coro),
            ) as mock_execute,
            patch.object(
                nodes_module,
                "run_sync",
                return_value={"team_result": "ok"},
            ) as mock_run_sync,
        ):
            graph_state = {"key": "value"}

            result = simple_team_step.execute(None, graph_state)

            mock_execute.assert_called_once()
            mock_run_sync.assert_called_once_with(coro)
            assert result == {"team_result": "ok"}

    @pytest.mark.asyncio
    async def test_sync_execute_uses_worker_thread_bridge_inside_running_loop(
        self,
        simple_team_step,
    ):
        """Test that sync execute uses the shared bridge from async contexts."""
        coro = object()

        with (
            patch.object(
                simple_team_step,
                "execute_async",
                new=Mock(return_value=coro),
            ) as mock_execute,
            patch.object(
                nodes_module,
                "run_sync",
                return_value={"team_result": "ok"},
            ) as mock_run_sync,
        ):
            result = simple_team_step.execute(None, {"key": "value"})

        mock_execute.assert_called_once_with(None, {"key": "value"})
        mock_run_sync.assert_called_once_with(coro)
        assert result == {"team_result": "ok"}


class TestTeamStepRichPersona:
    """Tests for rich persona support in TeamStep members."""

    def test_team_member_with_rich_persona(self):
        """Test creating team member with rich persona attributes."""
        member = TeamMember(
            id="expert",
            role=SubAgentRole.RESEARCHER,
            name="Expert Researcher",
            goal="Conduct deep research",
            backstory="20 years of experience in the field",
            expertise=["research", "analysis", "writing"],
            personality="methodical and thorough",
            max_delegation_depth=2,
        )

        assert member.backstory == "20 years of experience in the field"
        assert member.expertise == ["research", "analysis", "writing"]
        assert member.personality == "methodical and thorough"
        assert member.max_delegation_depth == 2

    def test_team_member_to_system_prompt(self):
        """Test generating system prompt from rich persona."""
        member = TeamMember(
            id="expert",
            role=SubAgentRole.RESEARCHER,
            name="Expert Researcher",
            goal="Conduct research",
            backstory="10 years experience",
            expertise=["analysis"],
            personality="thorough",
        )

        prompt = member.to_system_prompt()

        assert "# Role: Expert Researcher" in prompt
        assert "## Goal" in prompt
        assert "Conduct research" in prompt
        assert "## Background" in prompt
        assert "10 years experience" in prompt
        assert "## Expertise" in prompt
        assert "analysis" in prompt
        assert "## Communication Style" in prompt
        assert "thorough" in prompt
