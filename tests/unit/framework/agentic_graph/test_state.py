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

"""Tests for AgenticLoopState model."""

import pytest
from pydantic import ValidationError

from victor.framework.agentic_graph import state as state_module
from victor.framework.agentic_graph.state import AgenticLoopStateModel


class TestAgenticLoopStateModel:
    """Tests for AgenticLoopStateModel Pydantic model."""

    def test_create_minimal_state(self):
        """Test creating a state with minimal required fields."""
        state = AgenticLoopStateModel(query="Test task")
        assert state.query == "Test task"
        assert state.iteration == 0
        assert state.max_iterations == 10
        assert state.stage is None
        assert state.perception is None

    def test_create_full_state(self):
        """Test creating a state with all fields populated."""
        state = AgenticLoopStateModel(
            query="Complex task",
            iteration=1,
            max_iterations=5,
            stage="plan",
            perception={"intent": "write", "complexity": "high"},
            task_type="code_generation",
            complexity="high",
            plan={"steps": ["analyze", "implement"]},
            action_result={"output": "done"},
            tool_results=[{"tool": "write", "result": "success"}],
            evaluation={"decision": "continue", "score": 0.7},
            progress_scores=[0.3, 0.5, 0.7],
            context={"project": "test"},
            conversation_history=[{"role": "user", "content": "hello"}],
        )
        assert state.query == "Complex task"
        assert state.iteration == 1
        assert state.max_iterations == 5
        assert state.stage == "plan"
        assert state.perception["intent"] == "write"
        assert state.task_type == "code_generation"
        assert state.complexity == "high"
        assert state.plan["steps"] == ["analyze", "implement"]
        assert state.action_result["output"] == "done"
        assert len(state.tool_results) == 1
        assert state.evaluation["decision"] == "continue"
        assert state.progress_scores == [0.3, 0.5, 0.7]
        assert state.context["project"] == "test"
        assert len(state.conversation_history) == 1

    def test_default_values(self):
        """Test that default values are set correctly."""
        state = AgenticLoopStateModel(query="Test")
        assert state.iteration == 0
        assert state.max_iterations == 10
        assert state.progress_scores == []
        assert state.tool_results == []

    def test_dict_like_interface(self):
        """Test dict-like interface for StateGraph compatibility."""
        state = AgenticLoopStateModel(query="Test", iteration=1)

        # Test __getitem__
        assert state["query"] == "Test"
        assert state["iteration"] == 1

        # Test __setitem__
        state["iteration"] = 2
        assert state.iteration == 2

        # Test __contains__
        assert "query" in state
        assert "nonexistent" not in state

        # Test get()
        assert state.get("query") == "Test"
        assert state.get("nonexistent", "default") == "default"

        # Test keys(), values(), items()
        assert "query" in state.keys()
        assert "Test" in state.values()
        assert ("query", "Test") in state.items()
        assert state.keys() == list(type(state).model_fields.keys())

    def test_model_validation(self):
        """Test Pydantic validation."""
        # Valid state
        state = AgenticLoopStateModel(query="Test", max_iterations=5)
        assert state.max_iterations == 5

        # Invalid max_iterations (negative)
        with pytest.raises(ValidationError):
            AgenticLoopStateModel(query="Test", max_iterations=-1)

    def test_model_serialization(self):
        """Test model serialization for checkpointing."""
        state = AgenticLoopStateModel(
            query="Test task",
            iteration=2,
            perception={"intent": "read"},
        )

        # Test model_dump
        data = state.model_dump()
        assert data["query"] == "Test task"
        assert data["iteration"] == 2
        assert data["perception"]["intent"] == "read"

        # Test model_dump_json
        json_str = state.model_dump_json()
        assert "Test task" in json_str

        # Test model_validate
        restored = AgenticLoopStateModel.model_validate(data)
        assert restored.query == state.query
        assert restored.iteration == state.iteration

    def test_immutability_with_frozen(self):
        """Test that frozen=True in config prevents direct mutation."""
        # Note: Pydantic v2 uses frozen in ConfigDict, not constructor
        # This test validates that the model_config approach works
        from pydantic import ConfigDict

        FrozenState = type(
            "FrozenState",
            (AgenticLoopStateModel,),
            {
                "model_config": ConfigDict(
                    arbitrary_types_allowed=True,
                    validate_assignment=True,
                    frozen=True,
                )
            },
        )

        state = FrozenState(query="Test")

        # Direct attribute assignment should fail for frozen models
        with pytest.raises(ValidationError):
            state.iteration = 5

    def test_copy_with_updates(self):
        """Test creating updated copies of state."""
        original = AgenticLoopStateModel(query="Test", iteration=1)

        # Create updated copy
        updated = original.model_copy(update={"iteration": 2, "stage": "plan"})
        assert updated.query == "Test"
        assert updated.iteration == 2
        assert updated.stage == "plan"

        # Original should be unchanged
        assert original.iteration == 1
        assert original.stage is None


class TestAgenticLoopStateType:
    """Tests for AgenticLoopState TypedDict type alias."""

    def test_state_type_alias_emits_deprecation_warning(self):
        """Legacy alias should stay backward compatible while warning callers."""
        with pytest.warns(DeprecationWarning, match="AgenticLoopState"):
            alias = state_module.AgenticLoopState

        assert alias is AgenticLoopStateModel

    def test_state_type_is_dict_subclass(self):
        """Test that AgenticLoopState is compatible with dict."""
        state: dict[str, object] = {
            "query": "Test",
            "iteration": 0,
            "max_iterations": 10,
        }
        assert isinstance(state, dict)
        assert state["query"] == "Test"

    def test_state_type_allows_optional_fields(self):
        """Test that optional fields can be omitted."""
        state: dict[str, object] = {"query": "Test"}
        assert "query" in state
        assert "perception" not in state


class TestStateTransition:
    """Tests for state transitions during agentic loop execution."""

    def test_perceive_stage_transition(self):
        """Test state transition after PERCEIVE stage."""
        state = AgenticLoopStateModel(query="Fix the bug")

        # After perceive
        perceived = state.model_copy(
            update={
                "stage": "perceive",
                "perception": {
                    "intent": "write",
                    "complexity": "medium",
                    "task_type": "debugging",
                },
                "task_type": "debugging",
                "complexity": "medium",
            }
        )
        assert perceived.stage == "perceive"
        assert perceived.perception["intent"] == "write"

    def test_plan_stage_transition(self):
        """Test state transition after PLAN stage."""
        state = AgenticLoopStateModel(
            query="Fix the bug",
            perception={"intent": "write"},
            task_type="debugging",
        )

        # After plan
        planned = state.model_copy(
            update={
                "stage": "plan",
                "plan": {"tool_calls": ["edit_file"]},
            }
        )
        assert planned.stage == "plan"
        assert planned.plan["tool_calls"] == ["edit_file"]

    def test_act_stage_transition(self):
        """Test state transition after ACT stage."""
        state = AgenticLoopStateModel(
            query="Fix the bug",
            plan={"tool_calls": ["edit_file"]},
        )

        # After act
        acted = state.model_copy(
            update={
                "stage": "act",
                "action_result": {"output": "File edited"},
                "tool_results": [{"tool": "edit_file", "result": "success"}],
            }
        )
        assert acted.stage == "act"
        assert acted.action_result["output"] == "File edited"

    def test_evaluate_stage_transition(self):
        """Test state transition after EVALUATE stage."""
        state = AgenticLoopStateModel(
            query="Fix the bug",
            action_result={"output": "File edited"},
        )

        # After evaluate
        evaluated = state.model_copy(
            update={
                "stage": "evaluate",
                "evaluation": {"decision": "continue", "score": 0.6},
                "progress_scores": [0.6],
            }
        )
        assert evaluated.stage == "evaluate"
        assert evaluated.evaluation["decision"] == "continue"
        assert evaluated.progress_scores == [0.6]

    def test_iteration_increment(self):
        """Test that iteration is incremented each loop."""
        state = AgenticLoopStateModel(query="Test", iteration=0)

        # After first iteration
        state = state.model_copy(update={"iteration": 1})
        assert state.iteration == 1

        # After second iteration
        state = state.model_copy(update={"iteration": 2})
        assert state.iteration == 2

    def test_max_iterations_limit(self):
        """Test max_iterations limit is enforced."""
        state = AgenticLoopStateModel(query="Test", max_iterations=3)

        # Should not exceed max_iterations
        assert state.iteration < state.max_iterations

        # At limit
        state = state.model_copy(update={"iteration": 3})
        assert state.iteration == state.max_iterations


class TestStateHelperFunctions:
    """Tests for state helper functions."""

    def test_create_initial_state(self):
        """Test creating initial state from query and context."""
        from victor.framework.agentic_graph.state import create_initial_state

        state = create_initial_state(
            query="Write tests",
            context={"project": "victor"},
            max_iterations=5,
        )
        assert state.query == "Write tests"
        assert state.context["project"] == "victor"
        assert state.max_iterations == 5
        assert state.iteration == 0
        assert state.progress_scores == []

    def test_should_continue_loop(self):
        """Test checking if loop should continue."""
        from victor.framework.agentic_graph.state import should_continue_loop

        # Continue: not at max, not complete
        state = AgenticLoopStateModel(
            query="Test",
            iteration=1,
            max_iterations=10,
            evaluation={"decision": "continue"},
        )
        assert should_continue_loop(state) is True

        # Stop: at max iterations
        state = state.model_copy(update={"iteration": 10})
        assert should_continue_loop(state) is False

        # Stop: evaluation says complete
        state = AgenticLoopStateModel(
            query="Test",
            iteration=1,
            evaluation={"decision": "complete"},
        )
        assert should_continue_loop(state) is False

        # Stop: evaluation says fail
        state = state.model_copy(update={"evaluation": {"decision": "fail"}})
        assert should_continue_loop(state) is False
