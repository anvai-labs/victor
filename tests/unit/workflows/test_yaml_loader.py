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

"""Tests for YAML workflow loader."""

import tempfile
from pathlib import Path

import pytest

from victor.workflows import (
    AgentNode,
    ConditionNode,
    ParallelNode,
    TransformNode,
)
from victor.workflows.yaml_loader import (
    YAMLWorkflowConfig,
    YAMLWorkflowError,
    YAMLWorkflowProvider,
    load_workflow_from_dict,
    load_workflow_from_file,
    load_workflow_from_yaml,
    load_workflows_from_directory,
    _create_simple_condition,
    _parse_constraints,
    _parse_value,
)


class TestParseValue:
    """Tests for value parsing."""

    def test_parse_string_double_quotes(self):
        assert _parse_value('"hello"') == "hello"

    def test_parse_string_single_quotes(self):
        assert _parse_value("'hello'") == "hello"

    def test_parse_int(self):
        assert _parse_value("42") == 42

    def test_parse_float(self):
        assert _parse_value("3.14") == 3.14

    def test_parse_bool_true(self):
        assert _parse_value("true") is True
        assert _parse_value("True") is True

    def test_parse_bool_false(self):
        assert _parse_value("false") is False
        assert _parse_value("False") is False

    def test_parse_none(self):
        assert _parse_value("none") is None
        assert _parse_value("None") is None

    def test_parse_unquoted_string(self):
        assert _parse_value("hello") == "hello"


class TestSimpleCondition:
    """Tests for simple condition parsing."""

    def test_truthy_check(self):
        condition = _create_simple_condition("has_errors")
        assert condition({"has_errors": True}) == "true"
        assert condition({"has_errors": False}) == "false"
        assert condition({}) == "false"

    def test_equality(self):
        condition = _create_simple_condition("status == 'active'")
        assert condition({"status": "active"}) == "true"
        assert condition({"status": "inactive"}) == "false"

    def test_inequality(self):
        condition = _create_simple_condition("count != 0")
        assert condition({"count": 1}) == "true"
        assert condition({"count": 0}) == "false"

    def test_greater_than(self):
        condition = _create_simple_condition("score > 50")
        assert condition({"score": 60}) == "true"
        assert condition({"score": 40}) == "false"

    def test_less_than(self):
        condition = _create_simple_condition("errors < 5")
        assert condition({"errors": 3}) == "true"
        assert condition({"errors": 10}) == "false"

    def test_in_operator(self):
        condition = _create_simple_condition("status in [active, pending, review]")
        assert condition({"status": "active"}) == "true"
        assert condition({"status": "completed"}) == "false"


class TestParseConstraints:
    """Tests for constraint parsing."""

    def test_none_string_allows_all(self):
        """constraints: none allows everything."""
        constraints = _parse_constraints("none")
        assert constraints.llm_allowed is True
        assert constraints.network_allowed is True
        assert constraints.write_allowed is True

    def test_none_in_list_allows_all(self):
        """constraints: [none] allows everything."""
        constraints = _parse_constraints(["none"])
        assert constraints.llm_allowed is True
        assert constraints.network_allowed is True
        assert constraints.write_allowed is True

    def test_none_cannot_combine_with_others(self):
        """constraints: [none, llm] is invalid."""
        with pytest.raises(YAMLWorkflowError) as exc_info:
            _parse_constraints(["none", "llm"])
        assert "cannot be combined" in str(exc_info.value)

    def test_list_blocks_specified(self):
        """constraints: [llm, network] blocks those items."""
        constraints = _parse_constraints(["llm", "network"])
        assert constraints.llm_allowed is False
        assert constraints.network_allowed is False
        assert constraints.write_allowed is True  # Not in list

    def test_list_case_insensitive(self):
        """constraints: [LLM, NETWORK] works."""
        constraints = _parse_constraints(["LLM", "NETWORK"])
        assert constraints.llm_allowed is False
        assert constraints.network_allowed is False

    def test_empty_list_allows_all(self):
        """constraints: [] allows everything."""
        constraints = _parse_constraints([])
        assert constraints.llm_allowed is True
        assert constraints.network_allowed is True
        assert constraints.write_allowed is True

    def test_missing_constraints_uses_defaults(self):
        """No constraints = TaskConstraints defaults."""
        constraints = _parse_constraints(None)
        assert constraints.llm_allowed is False  # Default
        assert constraints.network_allowed is True  # Default
        assert constraints.write_allowed is False  # Default

    def test_dict_format_legacy_support(self):
        """Dict format still works for backwards compatibility."""
        constraints = _parse_constraints(
            {
                "llm_allowed": True,
                "network_allowed": False,
                "write_allowed": True,
            }
        )
        assert constraints.llm_allowed is True
        assert constraints.network_allowed is False
        assert constraints.write_allowed is True

    def test_timeout_passthrough(self):
        """Timeout is passed through."""
        constraints = _parse_constraints(["llm"], timeout=120.0)
        assert constraints.timeout == 120.0

    def test_invalid_string_raises_error(self):
        """constraints: 'invalid' raises error."""
        with pytest.raises(YAMLWorkflowError) as exc_info:
            _parse_constraints("invalid")
        # Error message includes valid values from enum
        assert "Invalid constraint string" in str(exc_info.value)
        assert "Valid values" in str(exc_info.value)


class TestLoadWorkflowFromDict:
    """Tests for loading workflows from dictionaries."""

    def test_load_simple_workflow(self):
        data = {
            "description": "Test workflow",
            "nodes": [
                {
                    "id": "start",
                    "type": "agent",
                    "role": "researcher",
                    "goal": "Research the topic",
                    "tool_budget": 20,
                }
            ],
        }
        workflow = load_workflow_from_dict(data, "test_workflow")
        assert workflow.name == "test_workflow"
        assert workflow.description == "Test workflow"
        assert len(workflow.nodes) == 1
        assert "start" in workflow.nodes
        assert isinstance(workflow.nodes["start"], AgentNode)
        assert workflow.nodes["start"].role == "researcher"

    def test_load_workflow_with_multiple_nodes(self):
        data = {
            "description": "Multi-node workflow",
            "nodes": [
                {
                    "id": "analyze",
                    "type": "agent",
                    "role": "analyst",
                    "goal": "Analyze data",
                    "next": ["decide"],
                },
                {
                    "id": "decide",
                    "type": "condition",
                    "condition": "has_issues",
                    "branches": {"true": "fix", "false": "done"},
                },
                {
                    "id": "fix",
                    "type": "agent",
                    "role": "executor",
                    "goal": "Fix issues",
                    "next": ["done"],
                },
                {
                    "id": "done",
                    "type": "agent",
                    "role": "writer",
                    "goal": "Write report",
                },
            ],
        }
        workflow = load_workflow_from_dict(data, "test")
        assert len(workflow.nodes) == 4
        assert isinstance(workflow.nodes["decide"], ConditionNode)
        assert workflow.nodes["decide"].branches == {"true": "fix", "false": "done"}

    def test_load_parallel_node(self):
        data = {
            "nodes": [
                {
                    "id": "parallel_step",
                    "type": "parallel",
                    "parallel_nodes": ["task_a", "task_b"],
                    "join_strategy": "all",
                    "next": ["merge"],
                },
                {"id": "task_a", "type": "agent", "role": "executor", "goal": "Task A"},
                {"id": "task_b", "type": "agent", "role": "executor", "goal": "Task B"},
                {
                    "id": "merge",
                    "type": "agent",
                    "role": "writer",
                    "goal": "Merge results",
                },
            ],
        }
        workflow = load_workflow_from_dict(data, "parallel_test")
        assert isinstance(workflow.nodes["parallel_step"], ParallelNode)
        assert workflow.nodes["parallel_step"].parallel_nodes == ["task_a", "task_b"]

    def test_load_transform_node(self):
        data = {
            "nodes": [
                {
                    "id": "transform_step",
                    "type": "transform",
                    "transform": "result = ctx.input",
                    "next": ["next_step"],
                },
                {
                    "id": "next_step",
                    "type": "agent",
                    "role": "executor",
                    "goal": "Use result",
                },
            ],
        }
        workflow = load_workflow_from_dict(data, "transform_test")
        assert isinstance(workflow.nodes["transform_step"], TransformNode)

    def test_load_with_metadata(self):
        data = {
            "description": "With metadata",
            "metadata": {"version": "1.0", "author": "test"},
            "nodes": [{"id": "start", "type": "agent", "role": "executor", "goal": "Do work"}],
        }
        workflow = load_workflow_from_dict(data, "metadata_test")
        assert workflow.metadata["version"] == "1.0"
        assert workflow.metadata["author"] == "test"

    def test_missing_node_id_raises_error(self):
        data = {"nodes": [{"type": "agent", "role": "executor", "goal": "No ID"}]}
        with pytest.raises(YAMLWorkflowError, match="missing required 'id' field"):
            load_workflow_from_dict(data, "bad_workflow")


class TestLoadWorkflowFromYAML:
    """Tests for loading workflows from YAML strings."""

    def test_load_single_workflow(self):
        yaml_content = """
workflows:
  code_review:
    description: "Review code quality"
    nodes:
      - id: analyze
        type: agent
        role: researcher
        goal: "Find code patterns"
        tool_budget: 20
      - id: report
        type: agent
        role: writer
        goal: "Summarize findings"
"""
        workflow = load_workflow_from_yaml(yaml_content, "code_review")
        assert workflow.name == "code_review"
        assert workflow.description == "Review code quality"
        assert len(workflow.nodes) == 2

    def test_load_all_workflows(self):
        yaml_content = """
workflows:
  workflow_a:
    nodes:
      - id: step1
        type: agent
        role: executor
        goal: "Task A"
  workflow_b:
    nodes:
      - id: step1
        type: agent
        role: executor
        goal: "Task B"
"""
        workflows = load_workflow_from_yaml(yaml_content)
        assert isinstance(workflows, dict)
        assert "workflow_a" in workflows
        assert "workflow_b" in workflows

    def test_invalid_yaml_raises_error(self):
        yaml_content = "not: valid: yaml: content:"
        with pytest.raises(YAMLWorkflowError, match="Invalid YAML"):
            load_workflow_from_yaml(yaml_content)

    def test_missing_workflow_raises_error(self):
        yaml_content = """
workflows:
  existing:
    nodes:
      - id: step
        type: agent
        role: executor
        goal: "Task"
"""
        with pytest.raises(YAMLWorkflowError, match="not found"):
            load_workflow_from_yaml(yaml_content, "nonexistent")


class TestLoadWorkflowFromFile:
    """Tests for loading workflows from files."""

    def test_load_from_file(self):
        yaml_content = """
workflows:
  test:
    description: "File test"
    nodes:
      - id: start
        type: agent
        role: executor
        goal: "From file"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            workflow = load_workflow_from_file(f.name, "test")
            assert workflow.name == "test"
            assert workflow.description == "File test"

            Path(f.name).unlink()

    def test_file_not_found_raises_error(self):
        with pytest.raises(YAMLWorkflowError, match="File not found"):
            load_workflow_from_file("/nonexistent/path/workflow.yaml")


class TestLoadWorkflowsFromDirectory:
    """Tests for loading workflows from a directory."""

    def test_load_from_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two workflow files
            for i, name in enumerate(["workflow_a", "workflow_b"]):
                content = f"""
workflows:
  {name}:
    description: "Workflow {i}"
    nodes:
      - id: step
        type: agent
        role: executor
        goal: "Task {i}"
"""
                Path(tmpdir, f"{name}.yaml").write_text(content)

            workflows = load_workflows_from_directory(tmpdir)
            assert "workflow_a" in workflows
            assert "workflow_b" in workflows


class TestYAMLWorkflowProvider:
    """Tests for the YAML workflow provider."""

    def test_provider_from_file(self):
        yaml_content = """
workflows:
  feature:
    description: "Feature workflow"
    nodes:
      - id: implement
        type: agent
        role: executor
        goal: "Implement feature"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            provider = YAMLWorkflowProvider.from_file(f.name)
            assert "feature" in provider.list_workflows()
            workflow = provider.get_workflow("feature")
            assert workflow is not None
            assert workflow.description == "Feature workflow"

            Path(f.name).unlink()

    def test_provider_get_workflows(self):
        yaml_content = """
workflows:
  test:
    nodes:
      - id: step
        type: agent
        role: executor
        goal: "Task"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            provider = YAMLWorkflowProvider.from_file(f.name)
            workflows = provider.get_workflows()
            assert "test" in workflows

            Path(f.name).unlink()


class TestCustomConditionRegistry:
    """Tests for custom condition functions."""

    def test_custom_condition(self):
        def custom_logic(ctx):
            return "proceed" if ctx.get("custom_flag") else "skip"

        config = YAMLWorkflowConfig(condition_registry={"custom_check": custom_logic})

        data = {
            "nodes": [
                {"id": "start", "type": "agent", "role": "executor", "goal": "Start"},
                {
                    "id": "check",
                    "type": "condition",
                    "condition": "custom_check",
                    "branches": {"proceed": "action", "skip": "end"},
                },
                {"id": "action", "type": "agent", "role": "executor", "goal": "Action"},
                {"id": "end", "type": "agent", "role": "executor", "goal": "End"},
            ],
        }
        workflow = load_workflow_from_dict(data, "custom_test", config)
        condition_node = workflow.nodes["check"]
        assert condition_node.evaluate({"custom_flag": True}) == "action"
        assert condition_node.evaluate({"custom_flag": False}) == "end"


class TestHITLNodes:
    """Tests for HITL node loading."""

    def test_load_hitl_approval(self):
        data = {
            "nodes": [
                {
                    "id": "approve",
                    "type": "hitl",
                    "hitl_type": "approval",
                    "prompt": "Approve changes?",
                    "timeout": 300,
                    "fallback": "abort",
                }
            ],
        }
        workflow = load_workflow_from_dict(data, "hitl_test")
        from victor.workflows.hitl import HITLNode

        assert isinstance(workflow.nodes["approve"], HITLNode)

    def test_load_hitl_choice(self):
        data = {
            "nodes": [
                {
                    "id": "select",
                    "type": "hitl",
                    "hitl_type": "choice",
                    "prompt": "Select option",
                    "choices": ["option_a", "option_b", "option_c"],
                    "default_value": "option_a",
                }
            ],
        }
        workflow = load_workflow_from_dict(data, "choice_test")
        from victor.workflows.hitl import HITLNode

        hitl_node = workflow.nodes["select"]
        assert isinstance(hitl_node, HITLNode)
        assert hitl_node.choices == ["option_a", "option_b", "option_c"]


TEAM_EXAMPLE = (
    Path(__file__).resolve().parents[3]
    / "verticals/victor-coding/victor_coding/workflows/team_node_example.yaml"
)


def _team_workflow(**overrides):
    node = {
        "id": "crew",
        "type": "team",
        "goal": "Ship the feature",
        "members": [
            {"id": "r", "role": "researcher", "goal": "Research", "tools": ["read"]},
            {"id": "w", "role": "Executor", "goal": "Write", "tool_budget": 7},
        ],
    }
    node.update(overrides)
    return {"nodes": [node, {"id": "done", "type": "transform", "transform": "x = 1"}]}


class TestTeamNodes:
    """``type: team`` YAML nodes build ``TeamStepWorkflow`` steps."""

    def test_example_loads_with_expected_fields(self):
        from victor.workflows.definition import TeamStepWorkflow

        workflow = load_workflow_from_file(TEAM_EXAMPLE)["team_node_demo"]

        small = workflow.nodes["small_team"]
        large = workflow.nodes["large_team"]
        assert isinstance(small, TeamStepWorkflow)
        assert isinstance(large, TeamStepWorkflow)
        assert small.team_formation == "sequential"
        assert large.team_formation == "pipeline"
        assert [m["id"] for m in large.members] == [
            "architect",
            "researcher",
            "implementer",
            "reviewer",
        ]
        assert [m["tool_budget"] for m in large.members] == [20, 15, 40, 15]
        assert large.members[0]["allowed_tools"] == ["read", "grep", "overview", "code_search"]
        assert "tools" not in large.members[0]
        assert large.total_tool_budget == 100
        assert small.total_tool_budget == 50
        assert small.max_iterations == 30
        assert small.timeout_seconds == 300.0
        assert small.merge_mode == "team_wins"
        assert small.output_key == "team_result"
        assert small.continue_on_error is True
        assert small.next_nodes == ["finalize"]
        assert small.name == "Small Implementation Team"
        assert "Implement the feature: $ctx.user_task" in small.goal

    def test_example_team_goal_placeholders_are_filled_by_team_step(self):
        """Team goals use $ctx.<key>: the loader env-expands ${...} and TeamStep
        does not render {{...}}."""
        from victor.framework.workflows.nodes import TeamStep

        node = load_workflow_from_file(TEAM_EXAMPLE)["team_node_demo"].nodes["small_team"]
        step = TeamStep(id=node.id, name=node.name, goal=node.goal, team_formation=None, members=[])
        goal = step._build_goal({"user_task": "add retries", "task_analysis": "medium"})
        assert "Implement the feature: add retries" in goal
        assert "Task Analysis: medium" in goal
        assert "{{" not in goal and "$ctx." not in goal

    def test_example_compiles(self):
        from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

        workflow = load_workflow_from_file(TEAM_EXAMPLE)["team_node_demo"]
        compiled = UnifiedWorkflowCompiler().compile_definition(workflow)
        assert compiled is not None

    def test_loaded_members_build_executor_members(self):
        """The loaded shape is what TeamStepExecutor consumes (no silent defaults)."""
        from victor.agent.subagents import SubAgentRole
        from victor.framework.state_merging import MergeMode
        from victor.teams.types import TeamFormation, TeamMember
        from victor.workflows.executors.team import TeamStepExecutor

        node = load_workflow_from_file(TEAM_EXAMPLE)["team_node_demo"].nodes["large_team"]
        executor = TeamStepExecutor()
        members = executor._build_members(node.members, SubAgentRole, TeamMember)
        assert [m.role for m in members] == [
            SubAgentRole.PLANNER,
            SubAgentRole.RESEARCHER,
            SubAgentRole.EXECUTOR,
            SubAgentRole.REVIEWER,
        ]
        assert members[0].allowed_tools == ["read", "grep", "overview", "code_search"]
        assert members[2].tool_budget == 40
        assert members[0].expertise == [
            "architecture",
            "design-patterns",
            "scalability",
            "security",
        ]
        assert executor._resolve_team_formation(node.team_formation, TeamFormation) is (
            TeamFormation.PIPELINE
        )
        assert executor._resolve_merge_mode(node.merge_mode, MergeMode) is MergeMode.TEAM_WINS

    def test_every_accepted_value_is_honoured_by_the_executor(self):
        """Loader whitelists must not admit values the executor silently degrades."""
        from victor.agent.subagents import SubAgentRole
        from victor.framework.state_merging import MergeMode
        from victor.teams.types import TeamFormation, TeamMember
        from victor.workflows.executors.team import TeamStepExecutor
        from victor.workflows.generation.validator import SchemaValidator
        from victor.workflows.yaml_loader import (
            TEAM_FORMATIONS,
            TEAM_MEMBER_ROLES,
            TEAM_MERGE_MODES,
        )

        executor = TeamStepExecutor()
        for formation in TEAM_FORMATIONS:
            assert executor._resolve_team_formation(formation, TeamFormation).value == formation
        for mode in TEAM_MERGE_MODES:
            assert executor._resolve_merge_mode(mode, MergeMode).value == mode
        # writer and analyst are deliberate executor aliases (agent-node vocabulary).
        aliases = {"writer": "executor", "analyst": "researcher"}
        for role in TEAM_MEMBER_ROLES:
            (member,) = executor._build_members(
                [{"role": role, "goal": "g"}], SubAgentRole, TeamMember
            )
            assert member.role.value == aliases.get(role, role)
        assert set(TEAM_FORMATIONS) == SchemaValidator.VALID_TEAM_FORMATIONS

    def test_defaults(self):
        node = load_workflow_from_dict(_team_workflow(), "t").nodes["crew"]
        assert node.team_formation == "sequential"
        assert node.merge_strategy == "dict"
        assert node.merge_mode == "team_wins"
        assert node.output_key == "team_result"
        assert node.timeout_seconds is None
        assert node.max_iterations == 50
        assert node.total_tool_budget == 100
        assert node.shared_context == {}
        assert node.name == "crew"
        assert node.next_nodes == ["done"]  # auto-chained like other node types
        assert node.members[1]["role"] == "executor"  # normalised case
        assert node.members[0]["allowed_tools"] == ["read"]

    def test_output_alias_timeout_alias_and_shared_context(self):
        node = load_workflow_from_dict(
            _team_workflow(output="out", timeout=12, shared_context={"k": 1}), "t"
        ).nodes["crew"]
        assert node.output_key == "out"
        assert node.timeout_seconds == 12.0
        assert node.shared_context == {"k": 1}

    def test_missing_member_ids_match_executor_generated_ids(self):
        from victor.agent.subagents import SubAgentRole
        from victor.teams.types import TeamMember
        from victor.workflows.executors.team import TeamStepExecutor

        members = [{"role": "writer", "goal": "a"}, {"role": "analyst", "goal": "b"}]
        node = load_workflow_from_dict(_team_workflow(members=members), "t").nodes["crew"]
        assert [m["id"] for m in node.members] == ["executor_0", "researcher_1"]
        anonymous = [{k: v for k, v in m.items() if k != "id"} for m in node.members]
        built = TeamStepExecutor()._build_members(anonymous, SubAgentRole, TeamMember)
        assert [m.id for m in built] == ["executor_0", "researcher_1"]

    def test_hierarchical_team_with_one_supervisor_builds_team_config(self):
        from victor.agent.subagents import SubAgentRole
        from victor.teams.types import TeamConfig, TeamFormation, TeamMember
        from victor.workflows.executors.team import TeamStepExecutor

        members = [
            {"id": "lead", "role": "planner", "goal": "Lead", "agent_category": "supervisor"},
            {"id": "dev", "goal": "Build", "delegation_targets": ["lead"]},
        ]
        node = load_workflow_from_dict(
            _team_workflow(team_formation="hierarchical", members=members), "t"
        ).nodes["crew"]
        built = TeamStepExecutor()._build_members(node.members, SubAgentRole, TeamMember)
        config = TeamConfig(
            name="crew", goal=node.goal, members=built, formation=TeamFormation.HIERARCHICAL
        )
        assert [m.id for m in config.members if m.is_supervisor] == ["lead"]

    def test_yaml_does_not_mutate_input_members(self):
        data = _team_workflow()
        load_workflow_from_dict(data, "t")
        assert data["nodes"][0]["members"][0]["tools"] == ["read"]

    @pytest.mark.parametrize(
        "overrides,message",
        [
            ({"goal": None}, "requires a non-empty 'goal'"),
            ({"goal": "   "}, "requires a non-empty 'goal'"),
            ({"members": []}, "non-empty 'members' list"),
            ({"members": None}, "non-empty 'members' list"),
            ({"members": {"a": {}}}, "non-empty 'members' list"),
            ({"team_formation": "round_robin"}, "invalid team_formation 'round_robin'"),
            ({"team_formation": "reflection"}, "invalid team_formation"),
            ({"merge_mode": "loser_wins"}, "invalid merge_mode"),
            ({"merge_strategy": "custom"}, "invalid merge_strategy"),
            ({"max_iterations": 0}, "max_iterations must be an integer >= 1"),
            ({"total_tool_budget": "lots"}, "total_tool_budget must be an integer"),
            ({"total_tool_budget": True}, "total_tool_budget must be an integer"),
            ({"timeout_seconds": -5}, "timeout_seconds must be a positive finite number"),
            ({"timeout_seconds": 0}, "timeout_seconds must be a positive finite number"),
            ({"timeout_seconds": 0, "timeout": 30}, "timeout_seconds must be a positive"),
            ({"timeout_seconds": float("inf")}, "timeout_seconds must be a positive finite"),
            ({"timeout_seconds": float("nan")}, "timeout_seconds must be a positive finite"),
            ({"timeout_seconds": "soon"}, "timeout_seconds must be a positive finite number"),
            ({"timeout": True}, "timeout must be a positive finite number"),
            ({"output_key": 5}, "output_key must be a string"),
            ({"name": None}, "name must be a string"),
            ({"members": [{"id": ["a"], "goal": "g"}]}, "member #0 id must be a non-empty string"),
            ({"members": [{"id": 1, "goal": "g"}]}, "member #0 id must be a non-empty string"),
            (
                {"members": [{"goal": "g", "provider": "openai"}]},
                "member #0 has unsupported keys ['provider']",
            ),
            ({"members": [{"goal": "g", "tool_budgte": 3}]}, "unsupported keys ['tool_budgte']"),
            ({"members": [{"goal": "g", "is_manager": "no"}]}, "is_manager must be a boolean"),
            ({"members": [{"goal": "g", "can_delegate": 1}]}, "can_delegate must be a boolean"),
            ({"members": [{"goal": "g", "expertise": "security"}]}, "expertise must be a list"),
            ({"members": [{"goal": "g", "backstory": ["x"]}]}, "backstory must be a string"),
            (
                {"members": [{"id": "a", "goal": "g", "delegation_targets": ["ghost"]}]},
                "delegation_targets ['ghost'] are not members",
            ),
            (
                # The executor would name the second member executor_1.
                {"members": [{"id": "executor_1", "goal": "g"}, {"goal": "h"}]},
                "duplicate member ids",
            ),
            (
                {"team_formation": "hierarchical"},
                "exactly one supervisor",
            ),
            (
                {
                    "team_formation": "hierarchical",
                    "members": [
                        {"id": "a", "goal": "g", "is_manager": True},
                        {"id": "b", "goal": "h", "agent_category": "supervisor"},
                    ],
                },
                "supervisors ['a', 'b']",
            ),
            (
                {
                    "team_formation": "hierarchical",
                    "members": [{"id": "a", "goal": "g", "is_manager": True}],
                },
                "at least 2 members",
            ),
            ({"shared_context": [1]}, "shared_context must be a mapping"),
            ({"continue_on_error": "yes"}, "continue_on_error must be a boolean"),
            ({"members": ["researcher"]}, "member #0 must be a mapping"),
            ({"members": [{"id": "a", "role": "executor"}]}, "member 'a' requires a non-empty"),
            ({"members": [{"role": "wizard", "goal": "g"}]}, "member #0 role 'wizard'"),
            (
                {"members": [{"goal": "g", "agent_category": "boss"}]},
                "agent_category 'boss'",
            ),
            ({"members": [{"goal": "g", "tool_budget": 0}]}, "tool_budget must be an integer"),
            ({"members": [{"goal": "g", "tools": "read"}]}, "tools must be a list"),
            (
                {"members": [{"goal": "g", "tools": ["a"], "allowed_tools": ["b"]}]},
                "both 'tools' and 'allowed_tools'",
            ),
            (
                {"members": [{"id": "a", "goal": "g"}, {"id": "a", "goal": "h"}]},
                "duplicate member ids",
            ),
        ],
    )
    def test_malformed_team_node_is_rejected(self, overrides, message):
        with pytest.raises(YAMLWorkflowError, match="Team node 'crew'") as exc_info:
            load_workflow_from_dict(_team_workflow(**overrides), "t")
        assert message in str(exc_info.value)

    def test_team_listed_in_unknown_type_error(self):
        with pytest.raises(YAMLWorkflowError, match="'team'"):
            load_workflow_from_dict({"nodes": [{"id": "x", "type": "squad"}]}, "t")
