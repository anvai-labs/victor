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

"""Unit tests for readable task planning schema."""

import json
import logging

import pytest

from victor.agent.planning.readable_schema import (
    ReadableTaskPlan,
    TaskComplexity,
    TaskPlannerContext,
    _validate_readable_plan_from_candidates,
    generate_task_plan,
    plan_to_session_context,
    plan_to_workflow_yaml,
)
from victor.agent.planning.base import (
    ExecutionPlan,
    PlanStep,
    StepType,
)


class TestReadableTaskPlan:
    """Tests for ReadableTaskPlan model."""

    def test_simple_plan_validation(self):
        """Test validation of simple readable plan."""
        json_str = '{"name":"Fix bug","complexity":"simple","desc":"Fix login bug","steps":[[1,"analyze","Find bug","grep"],[2,"feature","Fix","write"]]}'

        plan = ReadableTaskPlan.model_validate_json(json_str)

        assert plan.name == "Fix bug"
        assert plan.complexity == TaskComplexity.SIMPLE
        assert plan.desc == "Fix login bug"
        assert len(plan.steps) == 2
        assert plan.approval is False

    def test_moderate_plan_validation(self):
        """Test validation of moderate plan with dependencies."""
        json_str = """{"name":"Add auth","complexity":"moderate","desc":"OAuth2 login",
        "steps":[[1,"research","Research patterns","overview"],[2,"feature","Create module","write,test"],
        [3,"test","Verify","pytest",[2]]]}"""

        plan = ReadableTaskPlan.model_validate_json(json_str)

        assert plan.complexity == TaskComplexity.MODERATE
        assert len(plan.steps) == 3

        # Convert to execution plan
        exec_plan = plan.to_execution_plan()
        assert exec_plan.goal == "OAuth2 login"
        assert len(exec_plan.steps) == 3

    def test_complex_plan_with_approval(self):
        """Test complex plan requiring approval."""
        json_str = """{"name":"Deploy API","complexity":"complex","desc":"Deploy to production",
        "steps":[[1,"test","Run tests","pytest"],[2,"deploy","Deploy","kubectl",[1]]],
        "approval":true}"""

        plan = ReadableTaskPlan.model_validate_json(json_str)

        assert plan.approval is True
        assert plan.complexity == TaskComplexity.COMPLEX

        exec_plan = plan.to_execution_plan()
        # Deployment step should require approval
        deploy_step = [s for s in exec_plan.steps if s.id == "2"][0]
        assert deploy_step.requires_approval is True

    def test_to_execution_plan(self):
        """Test conversion to full ExecutionPlan."""
        plan = ReadableTaskPlan(
            name="Feature X",
            complexity=TaskComplexity.MODERATE,
            desc="Implement feature X",
            steps=[
                [1, "feature", "Create module", "write"],
                [2, "test", "Test", "pytest", [1]],
            ],
        )

        exec_plan = plan.to_execution_plan()

        assert exec_plan.goal == "Implement feature X"
        assert len(exec_plan.steps) == 2
        assert exec_plan.metadata["task_name"] == "Feature X"
        assert exec_plan.metadata["complexity"] == "moderate"
        assert exec_plan.steps[0].allowed_tools == ["write"]
        assert exec_plan.steps[1].allowed_tools == ["pytest"]
        assert exec_plan.steps[1].context["tools"] == ["pytest"]

    def test_from_execution_plan(self):
        """Test creation of readable plan from ExecutionPlan."""
        exec_plan = ExecutionPlan(
            id="test-plan",
            goal="Implement auth",
            steps=[
                PlanStep(
                    id="1",
                    description="Create auth module",
                    step_type=StepType.IMPLEMENTATION,
                    depends_on=[],
                    estimated_tool_calls=10,
                ),
            ],
            metadata={"task_name": "auth", "complexity": "simple"},
        )

        readable = ReadableTaskPlan.from_execution_plan(exec_plan)

        assert readable.name == "auth"
        assert readable.complexity == TaskComplexity.SIMPLE
        assert len(readable.steps) == 1

    def test_from_execution_plan_preserves_allowed_tools(self):
        """Test readable conversion preserves explicit step tool hints."""
        exec_plan = ExecutionPlan(
            id="test-plan",
            goal="Inventory Rust workspaces",
            steps=[
                PlanStep(
                    id="1",
                    description="Inventory Rust source files",
                    step_type=StepType.RESEARCH,
                    allowed_tools=["grep", "shell"],
                ),
            ],
            metadata={"task_name": "rust-inventory", "complexity": "simple"},
        )

        readable = ReadableTaskPlan.from_execution_plan(exec_plan)

        assert readable.steps[0] == [
            1,
            "research",
            "Inventory Rust source files",
            "grep,shell",
            [],
        ]

    def test_to_yaml(self):
        """Test conversion to YAML format."""
        plan = ReadableTaskPlan(
            name="Task X",
            complexity=TaskComplexity.SIMPLE,
            desc="Test task",
            steps=[[1, "test", "Run tests", "pytest"]],
            duration="15min",
        )

        yaml_str = plan.to_yaml()

        assert "workflows:" in yaml_str
        assert "Task X:" in yaml_str
        assert "Test task" in yaml_str

    def test_to_markdown(self):
        """Test conversion to markdown display format."""
        plan = ReadableTaskPlan(
            name="Fix Login Bug",
            complexity=TaskComplexity.SIMPLE,
            desc="Fix authentication bug",
            steps=[[1, "analyze", "Find bug", "grep"], [2, "feature", "Fix", "write"]],
            duration="30min",
        )

        markdown = plan.to_markdown()

        assert "# Fix Login Bug" in markdown
        assert "**Description**: Fix authentication bug" in markdown
        assert "**Complexity**: simple" in markdown
        assert "## Steps" in markdown

    def test_get_llm_prompt(self):
        """Test LLM prompt generation."""
        prompt = ReadableTaskPlan.get_llm_prompt()

        assert "JSON" in prompt
        assert "name" in prompt
        assert "complexity" in prompt
        assert "steps" in prompt
        assert "research, planning, feature" in prompt

    def test_step_type_mappings(self):
        """Test all step type string mappings to StepType enum."""
        type_mappings = [
            ("research", StepType.RESEARCH),
            ("planning", StepType.PLANNING),
            ("feature", StepType.IMPLEMENTATION),
            ("implementation", StepType.IMPLEMENTATION),
            ("bugfix", StepType.IMPLEMENTATION),
            ("refactor", StepType.IMPLEMENTATION),
            ("test", StepType.TESTING),
            ("testing", StepType.TESTING),
            ("review", StepType.REVIEW),
            ("deploy", StepType.DEPLOYMENT),
            ("deployment", StepType.DEPLOYMENT),
            ("analyze", StepType.RESEARCH),
            ("analysis", StepType.RESEARCH),
            ("doc", StepType.RESEARCH),
            ("documentation", StepType.RESEARCH),
        ]

        for type_str, expected_type in type_mappings:
            # Create a minimal plan with this step type
            plan = ReadableTaskPlan(
                name="Test",
                complexity=TaskComplexity.SIMPLE,
                desc="Test",
                steps=[[1, type_str, "Test step"]],
            )
            exec_plan = plan.to_execution_plan()
            assert exec_plan.steps[0].step_type == expected_type, f"Failed for {type_str}"

    def test_token_efficiency(self):
        """Test that readable schema uses fewer tokens than verbose JSON."""
        # Verbose JSON (what NOT to use)
        verbose = {
            "task_name": "Add authentication",
            "complexity": "moderate",
            "description": "Implement OAuth2 login",
            "steps": [
                {
                    "id": "1",
                    "type": "feature",
                    "description": "Create auth module",
                    "tools": ["write", "test"],
                    "dependencies": [],
                }
            ],
        }

        # Readable JSON (what TO use)
        readable = {
            "name": "Add auth",
            "complexity": "moderate",
            "desc": "OAuth2 login",
            "steps": [[1, "feature", "Create module", "write,test"]],
        }

        verbose_tokens = len(json.dumps(verbose))
        readable_tokens = len(json.dumps(readable))

        # Readable should be significantly smaller
        assert readable_tokens < verbose_tokens
        # Rough estimate: readable should be ~60-70% of verbose
        assert readable_tokens < verbose_tokens * 0.8


class TestTaskPlannerContext:
    """Tests for TaskPlannerContext session management."""

    def test_context_initialization(self):
        """Test context initialization."""
        ctx = TaskPlannerContext()

        assert ctx.current_plan is None
        assert len(ctx.plans_history) == 0
        assert len(ctx.approved_plans) == 0

    def test_set_and_approve_plan(self):
        """Test setting and approving a plan."""
        ctx = TaskPlannerContext()

        plan = ExecutionPlan(
            id="test",
            goal="Test goal",
            steps=[],
        )

        ctx.set_plan(plan)
        assert ctx.current_plan == plan

        ctx.approve_plan()
        assert len(ctx.approved_plans) == 1

    def test_archive_plan(self):
        """Test archiving current plan."""
        ctx = TaskPlannerContext()

        plan = ExecutionPlan(
            id="test",
            goal="Test goal",
            steps=[],
        )

        ctx.set_plan(plan)
        ctx.archive_plan()

        assert ctx.current_plan is None
        assert len(ctx.plans_history) == 1
        assert ctx.plans_history[0] == plan

    def test_get_plan_summary(self):
        """Test getting plan summary."""
        ctx = TaskPlannerContext()

        plan = ExecutionPlan(
            id="test",
            goal="Test goal",
            steps=[],
            metadata={"task_name": "test", "complexity": "simple"},
        )

        ctx.set_plan(plan)
        summary = ctx.get_plan_summary()

        assert summary["current_plan"] == "Test goal"
        assert summary["total_plans"] == 1
        assert summary["approved_plans"] == 0

    def test_to_context_dict(self):
        """Test exporting context to dictionary."""
        ctx = TaskPlannerContext()

        plan = ExecutionPlan(
            id="test",
            goal="Test goal",
            steps=[
                PlanStep(
                    id="1",
                    description="Step 1",
                    step_type=StepType.RESEARCH,
                    depends_on=[],
                )
            ],
            metadata={"task_name": "test"},
        )

        ctx.set_plan(plan)
        context_dict = ctx.to_context_dict()

        assert "task_planner" in context_dict
        assert context_dict["task_planner"]["active"] is True
        assert "current_plan" in context_dict["task_planner"]
        assert context_dict["task_planner"]["current_plan"]["goal"] == "Test goal"


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_plan_to_workflow_yaml(self):
        """Test conversion of plan to workflow YAML."""
        plan = ReadableTaskPlan(
            name="Feature X",
            complexity=TaskComplexity.MODERATE,
            desc="Implement feature X",
            steps=[
                [1, "feature", "Create module", "write"],
                [2, "test", "Test", "pytest", [1]],
            ],
        )

        yaml_str = plan_to_workflow_yaml(plan)

        assert "workflows:" in yaml_str
        assert "Feature X:" in yaml_str
        assert "Implement feature X" in yaml_str

    def test_plan_to_session_context(self):
        """Test adding plan to session context."""
        plan = ReadableTaskPlan(
            name="Task X",
            complexity=TaskComplexity.SIMPLE,
            desc="Test task",
            steps=[[1, "test", "Run tests", "pytest"]],
        )

        context = plan_to_session_context(plan, "session-123")

        assert context["session_id"] == "session-123"
        assert "task_plan" in context
        assert context["task_plan"]["name"] == "Task X"
        assert context["task_plan"]["complexity"] == "simple"

    def test_roundtrip_conversion(self):
        """Test roundtrip: Readable -> Execution -> Readable."""
        original_plan = ReadableTaskPlan(
            name="Roundtrip Test",
            complexity=TaskComplexity.MODERATE,
            desc="Test roundtrip conversion",
            steps=[
                [1, "research", "Research", "overview"],
                [2, "feature", "Implement", "write", [1]],
                [3, "test", "Test", "pytest", [2]],
            ],
            duration="1hr",
            approval=True,
        )

        # Convert to execution plan
        exec_plan = original_plan.to_execution_plan()

        # Convert back to readable
        restored_plan = ReadableTaskPlan.from_execution_plan(exec_plan)

        # Verify key fields match
        assert restored_plan.name == original_plan.name
        assert restored_plan.complexity == original_plan.complexity
        assert restored_plan.desc == original_plan.desc
        # Note: approval flag might not be preserved in roundtrip

    def test_invalid_json_validation(self):
        """Test that invalid JSON is rejected."""
        # Missing required field 'name'
        invalid_json = '{"complexity":"simple","desc":"test","steps":[]}'

        with pytest.raises(Exception):
            ReadableTaskPlan.model_validate_json(invalid_json)

    def test_invalid_step_data(self):
        """Test that invalid step data is rejected."""
        # Step data too short (needs at least 3 elements: id, type, desc)
        invalid_json = '{"name":"test","complexity":"simple","desc":"test","steps":[[1]]}'

        with pytest.raises(ValueError):
            ReadableTaskPlan.model_validate_json(invalid_json)

    def test_plan_candidate_validation_prefers_outer_plan_over_nested_step_array(self):
        """Schema-aware extraction should not validate a nested step array as the plan."""
        content = """
        Draft first step: [1, "analyze", "Read rust/Cargo.toml", "read", [], "tool"]

        {
          "name": "Rust review",
          "complexity": "complex",
          "desc": "Review Rust workspace",
          "steps": [[1, "analyze", "Read rust/Cargo.toml", "read", [], "tool"]]
        }
        """
        primary_json = '[1, "analyze", "Read rust/Cargo.toml", "read", [], "tool"]'

        plan = _validate_readable_plan_from_candidates(primary_json, content)

        assert plan.name == "Rust review"
        assert len(plan.steps) == 1


@pytest.mark.asyncio
async def test_generate_task_plan_recoverable_retry_does_not_log_warning(caplog):
    class FakeProvider:
        model = "fake-model"

        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return {"content": '{"name":"bad","complexity":"complex","desc":"missing steps"}'}
            return {
                "content": (
                    '{"name":"ok","complexity":"complex","desc":"valid",'
                    '"steps":[[1,"analyze","Read manifest","read",[],"tool"]]}'
                )
            }

    with caplog.at_level(logging.WARNING, logger="victor.agent.planning.readable_schema"):
        plan = await generate_task_plan(
            provider=FakeProvider(),
            user_request="review rust",
            complexity=TaskComplexity.COMPLEX,
            max_retries=1,
        )

    assert plan.name == "ok"
    assert not caplog.records


class TestTokenEfficiency:
    """Tests for token efficiency metrics."""

    def test_simple_plan_token_count(self):
        """Test token count for simple plan."""
        # A typical simple task plan with readable keywords
        readable_json = '{"name":"Fix bug","complexity":"simple","desc":"Fix login bug","steps":[[1,"analyze","Find bug","grep"],[2,"feature","Fix","write"]]}'

        # Count characters (rough proxy for tokens)
        char_count = len(readable_json)
        # Still very compact even with readable keywords
        assert char_count < 140

    def test_moderate_plan_token_count(self):
        """Test token count for moderate plan."""
        moderate_json = """{"name":"Add auth","complexity":"moderate","desc":"OAuth2 login",
        "steps":[[1,"research","Research","overview"],[2,"feature","Create module","write,test"],
        [3,"test","Verify","pytest",[2]]]}"""

        char_count = len(moderate_json)
        assert char_count < 220  # Still very compact

    def test_complex_plan_token_count(self):
        """Test token count for complex plan."""
        complex_json = """{"name":"Deploy API","complexity":"complex","desc":"Production deployment",
        "steps":[[1,"test","Test","pytest"],[2,"analyze","Plan","overview"],[3,"feature","Create config","write"],
        [4,"doc","Document","write"],[5,"test","Integration test","pytest",[4]],[6,"deploy","Deploy","kubectl",[5]]]}"""

        char_count = len(complex_json)
        assert char_count < 350  # Complex but still compact

    def test_token_savings_vs_verbose_json(self):
        """Test token savings compared to verbose JSON."""
        # Verbose version (what NOT to use)
        verbose = {
            "task_name": "Add authentication",
            "complexity": "moderate",
            "description": "Implement OAuth2 login with JWT tokens",
            "steps": [
                {
                    "step_id": 1,
                    "type": "feature",
                    "description": "Create auth module structure",
                    "tools": ["write", "test"],
                    "depends_on": [],
                },
                {
                    "step_id": 2,
                    "type": "feature",
                    "description": "Implement JWT validation",
                    "tools": ["write", "test"],
                    "depends_on": [1],
                },
            ],
        }

        # Readable version
        readable = {
            "name": "Add auth",
            "complexity": "moderate",
            "desc": "OAuth2 login with JWT",
            "steps": [
                [1, "feature", "Create auth structure", "write,test"],
                [2, "feature", "Implement JWT", "write,test", [1]],
            ],
        }

        verbose_size = len(json.dumps(verbose))
        readable_size = len(json.dumps(readable))

        # Readable should be 40-60% smaller
        savings_percent = (1 - readable_size / verbose_size) * 100
        assert savings_percent > 30, f"Expected >30% savings, got {savings_percent}%"

        print("\nToken efficiency:")
        print(f"Verbose JSON: {verbose_size} chars")
        print(f"Readable JSON: {readable_size} chars")
        print(f"Savings: {savings_percent:.1f}%")


class TestRichDictStepParsing:
    """Tests for _parse_step_dict — rich dict step format with execution node fields."""

    def _make_plan(self, steps: list) -> ReadableTaskPlan:
        return ReadableTaskPlan(
            name="Test plan",
            complexity=TaskComplexity.MODERATE,
            desc="Rich dict step parsing",
            steps=steps,
        )

    # ------------------------------------------------------------------
    # compute node
    # ------------------------------------------------------------------

    def test_compute_node_execution_field(self) -> None:
        plan = self._make_plan(
            [
                {
                    "id": "1",
                    "type": "analyze",
                    "desc": "Run checklist",
                    "exec": "compute",
                    "node": "rust_best_practices",
                }
            ]
        )
        exec_plan = plan.to_execution_plan()
        step = exec_plan.steps[0]

        assert step.execution == "compute"
        assert step.context["node"] == "rust_best_practices"
        assert step.context["execution"] == "compute"

    def test_compute_node_execution_alias(self) -> None:
        plan = self._make_plan(
            [{"id": "1", "type": "analyze", "desc": "Compute", "execution": "compute"}]
        )
        step = plan.to_execution_plan().steps[0]
        assert step.execution == "compute"

    # ------------------------------------------------------------------
    # tool node with produces
    # ------------------------------------------------------------------

    def test_tool_node_with_produces(self) -> None:
        plan = self._make_plan(
            [
                {
                    "id": "2",
                    "type": "analyze",
                    "desc": "List workspace members",
                    "tools": ["read"],
                    "exec": "tool",
                    "produces": "workspace_members",
                }
            ]
        )
        step = plan.to_execution_plan().steps[0]

        assert step.execution == "tool"
        assert step.context["produces"] == "workspace_members"
        assert step.allowed_tools == ["read"]

    # ------------------------------------------------------------------
    # loop node
    # ------------------------------------------------------------------

    def test_loop_node_with_loop_over(self) -> None:
        plan = self._make_plan(
            [
                {
                    "id": "3",
                    "type": "feature",
                    "desc": "Lint each crate",
                    "exec": "loop",
                    "loop_over": "workspace_members",
                    "deps": ["2"],
                }
            ]
        )
        step = plan.to_execution_plan().steps[0]

        assert step.execution == "loop"
        assert step.context["loop_over"] == "workspace_members"
        assert step.depends_on == ["2"]

    def test_loop_node_with_static_items(self) -> None:
        plan = self._make_plan(
            [
                {
                    "id": "4",
                    "type": "feature",
                    "desc": "Audit crates",
                    "exec": "loop",
                    "items": ["core", "util", "cli"],
                }
            ]
        )
        step = plan.to_execution_plan().steps[0]

        assert step.execution == "loop"
        assert step.context["items"] == ["core", "util", "cli"]

    def test_loop_node_with_exit_criteria(self) -> None:
        plan = self._make_plan(
            [
                {
                    "id": "5",
                    "type": "feature",
                    "desc": "Find failing crate",
                    "exec": "loop",
                    "loop_over": "workspace_members",
                    "exit": ["error found", "test failure"],
                }
            ]
        )
        step = plan.to_execution_plan().steps[0]

        assert step.exit_criteria == ["error found", "test failure"]

    def test_loop_node_exit_criteria_alias(self) -> None:
        plan = self._make_plan(
            [
                {
                    "id": "5",
                    "type": "feature",
                    "desc": "Find issue",
                    "exec": "loop",
                    "items": ["a", "b"],
                    "exit_criteria": ["found"],
                }
            ]
        )
        step = plan.to_execution_plan().steps[0]
        assert step.exit_criteria == ["found"]

    def test_loop_step_default_tool_calls_is_15(self) -> None:
        """Loop steps without explicit tool_calls default to 15 (per-iteration budget)."""
        plan = self._make_plan(
            [
                {
                    "id": "6",
                    "type": "analyze",
                    "desc": "Review each crate",
                    "exec": "loop",
                    "loop_over": "workspace_members",
                }
            ]
        )
        step = plan.to_execution_plan().steps[0]
        assert (
            step.estimated_tool_calls == 15
        ), "Loop steps must default to 15 tool calls per iteration, not 10"

    def test_loop_step_explicit_tool_calls_respected(self) -> None:
        """Explicit tool_calls on a loop step overrides the 15 default."""
        plan = self._make_plan(
            [
                {
                    "id": "6",
                    "type": "analyze",
                    "desc": "Deep per-crate review",
                    "exec": "loop",
                    "loop_over": "workspace_members",
                    "tool_calls": 20,
                }
            ]
        )
        step = plan.to_execution_plan().steps[0]
        assert step.estimated_tool_calls == 20

    def test_explicit_tool_calls_overrides_default(self) -> None:
        """Plans MAY request a larger budget via explicit tool_calls.

        Replaces the former rust-specific auto-bump heuristic.  The framework
        no longer infers "deep per-target review" from description vocabulary
        — the planner sets ``tool_calls`` explicitly when a step needs more
        budget (as the generic complex few-shot example demonstrates).
        """
        plan = self._make_plan(
            [
                {
                    "id": "8a",
                    "type": "analyze",
                    "desc": "Deep per-target review iterating over each review target",
                    "tools": ["read", "grep", "code_search"],
                    "produces": "per_target_findings",
                    "tool_calls": 25,
                }
            ]
        )
        step = plan.to_execution_plan().steps[0]
        assert step.estimated_tool_calls == 25

    def test_non_loop_step_default_tool_calls_is_10(self) -> None:
        """Non-loop steps retain the default of 10 tool calls."""
        plan = self._make_plan(
            [
                {
                    "id": "7",
                    "type": "analyze",
                    "desc": "Analyze codebase",
                    "exec": "tool",
                }
            ]
        )
        step = plan.to_execution_plan().steps[0]
        assert step.estimated_tool_calls == 10

    def test_loop_with_static_items_default_tool_calls_is_15(self) -> None:
        """Loop steps using static items (not loop_over) also default to 15."""
        plan = self._make_plan(
            [
                {
                    "id": "8",
                    "type": "analyze",
                    "desc": "Check each module",
                    "exec": "loop",
                    "items": ["core", "util", "cli"],
                }
            ]
        )
        step = plan.to_execution_plan().steps[0]
        assert step.estimated_tool_calls == 15

    # ------------------------------------------------------------------
    # conditional node
    # ------------------------------------------------------------------

    def test_conditional_node_fields(self) -> None:
        plan = self._make_plan(
            [
                {
                    "id": "6",
                    "type": "analyze",
                    "desc": "Check crate count",
                    "exec": "conditional",
                    "condition_on": "workspace_members",
                    "condition": "multiple",
                    "produces": "is_workspace",
                    "branches": {"true": ["7a"], "false": ["7b"]},
                }
            ]
        )
        step = plan.to_execution_plan().steps[0]

        assert step.execution == "conditional"
        assert step.context["condition_on"] == "workspace_members"
        assert step.context["condition"] == "multiple"
        assert step.context["produces"] == "is_workspace"
        assert step.context["branches"] == {"true": ["7a"], "false": ["7b"]}

    def test_conditional_node_default_condition_is_non_empty(self) -> None:
        plan = self._make_plan(
            [
                {
                    "id": "6",
                    "type": "analyze",
                    "desc": "Check value",
                    "exec": "conditional",
                    "condition_on": "some_key",
                }
            ]
        )
        step = plan.to_execution_plan().steps[0]
        assert step.context["condition"] == "non_empty"

    def test_conditional_node_branches_values_are_string_lists(self) -> None:
        plan = self._make_plan(
            [
                {
                    "id": "7",
                    "type": "analyze",
                    "desc": "Branch",
                    "exec": "conditional",
                    "condition_on": "x",
                    "branches": {"true": [8, 9], "false": [10]},
                }
            ]
        )
        step = plan.to_execution_plan().steps[0]
        # Branch IDs should be coerced to strings
        assert step.context["branches"]["true"] == ["8", "9"]
        assert step.context["branches"]["false"] == ["10"]

    # ------------------------------------------------------------------
    # approval node
    # ------------------------------------------------------------------

    def test_approval_node_execution_type(self) -> None:
        plan = self._make_plan(
            [
                {
                    "id": "8",
                    "type": "deployment",
                    "desc": "Deploy to production",
                    "exec": "approval",
                }
            ]
        )
        step = plan.to_execution_plan().steps[0]
        assert step.execution == "approval"

    # ------------------------------------------------------------------
    # tools as list vs comma-string
    # ------------------------------------------------------------------

    def test_tools_as_list_parsed_correctly(self) -> None:
        plan = self._make_plan(
            [
                {
                    "id": "1",
                    "type": "analyze",
                    "desc": "Step",
                    "tools": ["read", "grep", "write"],
                }
            ]
        )
        step = plan.to_execution_plan().steps[0]
        assert step.allowed_tools == ["read", "grep", "write"]

    def test_tools_as_comma_string(self) -> None:
        plan = self._make_plan(
            [
                {
                    "id": "1",
                    "type": "analyze",
                    "desc": "Step",
                    "tools": "read, grep, write",
                }
            ]
        )
        step = plan.to_execution_plan().steps[0]
        assert step.allowed_tools == ["read", "grep", "write"]

    # ------------------------------------------------------------------
    # dependencies via dict key "deps"
    # ------------------------------------------------------------------

    def test_deps_key_parsed_as_depends_on(self) -> None:
        plan = self._make_plan(
            [
                {"id": "1", "type": "analyze", "desc": "A"},
                {"id": "2", "type": "feature", "desc": "B", "deps": ["1"]},
            ]
        )
        steps = plan.to_execution_plan().steps
        step_b = next(s for s in steps if s.id == "2")
        assert step_b.depends_on == ["1"]

    def test_depends_on_alias_also_works(self) -> None:
        plan = self._make_plan(
            [
                {"id": "1", "type": "analyze", "desc": "A"},
                {"id": "2", "type": "feature", "desc": "B", "depends_on": ["1"]},
            ]
        )
        steps = plan.to_execution_plan().steps
        step_b = next(s for s in steps if s.id == "2")
        assert step_b.depends_on == ["1"]

    # ------------------------------------------------------------------
    # mixed compact + rich dict steps in the same plan
    # ------------------------------------------------------------------

    def test_mixed_list_and_dict_steps(self) -> None:
        plan = self._make_plan(
            [
                [1, "analyze", "Inventory", "read"],
                {
                    "id": "2",
                    "type": "feature",
                    "desc": "Compute checklist",
                    "exec": "compute",
                    "node": "rust_best_practices",
                    "deps": ["1"],
                },
            ]
        )
        steps = plan.to_execution_plan().steps
        assert len(steps) == 2
        assert steps[0].execution == ""
        assert steps[1].execution == "compute"
        assert steps[1].context["node"] == "rust_best_practices"


class TestStepEnrichment:
    """Tests for _enrich_step_dicts — the inference layer that fills in missing exec/produces/etc."""

    def _make_plan(self, steps):
        return ReadableTaskPlan(
            name="test",
            complexity=TaskComplexity.COMPLEX,
            desc="Test plan",
            steps=steps,
        )

    def test_description_alias_accepted_by_validator(self) -> None:
        """'description' is accepted as an alias for 'desc' in dict steps."""
        plan = self._make_plan(
            [
                {
                    "id": "1",
                    "type": "analyze",
                    "description": "Inventory workspace members",
                    "tools": "read",
                }
            ]
        )
        steps = plan.to_execution_plan().steps
        assert steps[0].description == "Inventory workspace members"

    def test_description_alias_parsed_into_step_description(self) -> None:
        """Step.description is correctly set when the dict uses 'description' not 'desc'."""
        plan = self._make_plan(
            [
                {
                    "id": "1",
                    "type": "analyze",
                    "description": "Find all modules",
                    "tools": "grep",
                }
            ]
        )
        assert plan.to_execution_plan().steps[0].description == "Find all modules"

    def test_infer_conditional_exec_from_route_prefix(self) -> None:
        """'Route: ...' prefix triggers exec=conditional inference."""
        plan = self._make_plan(
            [
                {
                    "id": "3",
                    "type": "analyze",
                    "desc": "Inventory all workspace members",
                    "tools": "shell",
                },
                {
                    "id": "4",
                    "type": "analyze",
                    "desc": "Route: determine if this is a multi-crate workspace or single crate",
                    "tools": "",
                },
                {
                    "id": "5a",
                    "type": "analyze",
                    "desc": "Loop over each workspace member",
                    "tools": "read",
                },
                {
                    "id": "5b",
                    "type": "analyze",
                    "desc": "Review single crate directly",
                    "tools": "read",
                },
            ]
        )
        steps = plan.to_execution_plan().steps
        routing_step = next(s for s in steps if s.id == "4")
        assert (
            routing_step.execution == "conditional"
        ), "Routing step should be inferred as conditional"

    def test_infer_loop_exec_from_for_each(self) -> None:
        """'For each X perform ...' triggers exec=loop inference."""
        plan = self._make_plan(
            [
                {
                    "id": "1",
                    "type": "analyze",
                    "desc": "For each module perform a review",
                    "tools": "read",
                }
            ]
        )
        step = plan.to_execution_plan().steps[0]
        assert step.execution == "loop"

    def test_infer_loop_exec_from_loop_over(self) -> None:
        """'Loop over each X ...' triggers exec=loop inference."""
        plan = self._make_plan(
            [
                {
                    "id": "1",
                    "type": "analyze",
                    "desc": "Loop over each workspace member crate performing deep review",
                    "tools": "read",
                }
            ]
        )
        step = plan.to_execution_plan().steps[0]
        assert step.execution == "loop"

    def test_synthesis_report_with_for_each_wording_does_not_infer_loop(self) -> None:
        """Report synthesis may mention 'for each finding' without being a loop."""
        plan = self._make_plan(
            [
                {
                    "id": "11",
                    "type": "doc",
                    "desc": (
                        "Synthesize all findings into a prioritized report with "
                        "concrete code examples for each suggested optimization"
                    ),
                    "tools": ["write"],
                }
            ]
        )
        step = plan.to_execution_plan().steps[0]
        assert step.execution != "loop"
        assert step.context["produces"] == "final_report"

    def test_infer_approval_exec(self) -> None:
        """'Present ... to user for review' triggers exec=approval inference."""
        plan = self._make_plan(
            [
                {
                    "id": "1",
                    "type": "review",
                    "desc": "Present the best practices checklist to user for review and approval",
                    "tools": "",
                }
            ]
        )
        step = plan.to_execution_plan().steps[0]
        assert step.execution == "approval"

    def test_review_step_parsing_infers_final_presentation_as_approval(self) -> None:
        """Final report presentation should be an approval checkpoint even if enrichment missed it."""
        plan = ReadableTaskPlan(
            name="Review presentation",
            complexity=TaskComplexity.COMPLEX,
            desc="Present final report",
            steps=[
                {
                    "id": "11",
                    "type": "review",
                    "desc": "Present consolidated report to user for feedback and discuss next steps for remediation",
                    "tools": [],
                    "deps": ["10"],
                }
            ],
        )

        step = plan.to_execution_plan().steps[0]

        assert step.execution == "approval"

    def test_explicit_exec_not_overwritten_by_inference(self) -> None:
        """Explicit exec field is never overwritten by the enrichment pass."""
        plan = self._make_plan(
            [
                {
                    "id": "1",
                    "type": "analyze",
                    "desc": "Route: determine if multi-crate",
                    "exec": "agent",
                    "tools": "",
                }
            ]
        )
        step = plan.to_execution_plan().steps[0]
        assert step.execution == "agent"

    def test_produces_inferred_from_inventory_description(self) -> None:
        """Steps that 'inventory workspace members' get produces inferred when a loop references it."""
        plan = self._make_plan(
            [
                {
                    "id": "1",
                    "type": "analyze",
                    "desc": "Inventory all workspace members and crate directories",
                    "tools": "shell,read",
                },
                {
                    "id": "2",
                    "type": "analyze",
                    "desc": "Loop over each workspace member crate performing deep review",
                    "tools": "read",
                },
            ]
        )
        exec_plan = plan.to_execution_plan()
        producer = exec_plan.steps[0]
        # The inventory step should have a produces key pointing to the collection
        assert producer.context.get("produces"), "inventory step should have produces inferred"
        # The loop step should reference that same key
        loop_step = exec_plan.steps[1]
        assert loop_step.context.get("loop_over") == producer.context["produces"]

    def test_branches_inferred_from_sibling_step_ids(self) -> None:
        """Conditional step 7 gets branches={true:[8a],false:[8b]} from sibling IDs."""
        plan = self._make_plan(
            [
                {
                    "id": "7",
                    "type": "analyze",
                    "desc": "Route: determine if multi-crate workspace or single crate",
                    "tools": "",
                },
                {
                    "id": "8a",
                    "type": "analyze",
                    "desc": "Loop over each workspace member crate",
                    "tools": "read",
                },
                {
                    "id": "8b",
                    "type": "analyze",
                    "desc": "Review single crate directly",
                    "tools": "read",
                },
            ]
        )
        exec_plan = plan.to_execution_plan()
        router = next(s for s in exec_plan.steps if s.id == "7")
        branches = router.context.get("branches", {})
        assert "8a" in branches.get("true", []) or "8a" in branches.get(
            "false", []
        ), "8a should appear in one of the branches"
        assert "8b" in branches.get("true", []) or "8b" in branches.get(
            "false", []
        ), "8b should appear in one of the branches"
        # The loop step (8a) should be in the 'true' branch (runs when multi-crate)
        assert "8a" in branches.get("true", []), "loop step 8a should be in true branch"
        assert "8b" in branches.get("false", []), "single-crate step 8b should be in false branch"

    def test_full_rust_workspace_plan_without_exec_annotations(self) -> None:
        """End-to-end: a plan generated by a model that omits exec fields is correctly enriched."""
        # Mirrors the actual GLM-5.1 output from the console transcript
        plan = self._make_plan(
            [
                {
                    "id": "3",
                    "type": "analyze",
                    "description": "Inventory all workspace members and crate directories from Cargo.toml parsing and directory listing",
                    "tools": ["shell", "read"],
                },
                {
                    "id": "4",
                    "type": "analyze",
                    "description": "For each workspace member, read its individual Cargo.toml to map dependencies",
                    "tools": "read",
                },
                {
                    "id": "5",
                    "type": "doc",
                    "description": "Create comprehensive Rust best practices checklist",
                    "tools": [],
                },
                {
                    "id": "6",
                    "type": "review",
                    "description": "Present the best practices checklist to user for review and approval before beginning workspace analysis",
                    "tools": [],
                },
                {
                    "id": "7",
                    "type": "analyze",
                    "description": "Route: determine if this is a multi-crate workspace or single crate to select review strategy",
                    "tools": [],
                },
                {
                    "id": "8a",
                    "type": "analyze",
                    "description": "Loop over each workspace member crate performing deep review",
                    "tools": ["read", "grep"],
                },
                {
                    "id": "8b",
                    "type": "analyze",
                    "description": "Review single crate directly",
                    "tools": ["read", "grep"],
                },
            ]
        )
        exec_plan = plan.to_execution_plan()
        by_id = {s.id: s for s in exec_plan.steps}

        # Step 3 (inventory) should produce the workspace collection
        assert by_id["3"].context.get("produces"), "inventory step should have produces"

        # Step 6 (present checklist to user) should be approval
        assert by_id["6"].execution == "approval", "checklist-presentation step should be approval"

        # Step 7 (route) should be conditional with multi-crate condition
        router = by_id["7"]
        assert router.execution == "conditional", "routing step should be conditional"
        assert (
            router.context.get("condition") == "multiple"
        ), "multi-crate routing should use 'multiple' condition"
        assert router.context.get("branches"), "routing step should have branches"
        assert "8a" in router.context["branches"].get("true", []), "loop path in true branch"
        assert "8b" in router.context["branches"].get(
            "false", []
        ), "single-crate path in false branch"

        # Step 8a (loop) should be loop with loop_over set
        assert by_id["8a"].execution == "loop", "step 8a should be loop"
        assert by_id["8a"].context.get("loop_over"), "loop step should have loop_over"

        # Step 8a loop_over key should match step 3 produces key
        assert (
            by_id["8a"].context["loop_over"] == by_id["3"].context["produces"]
        ), "loop_over must match produces so plan_state flows correctly"
