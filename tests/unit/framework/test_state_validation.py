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

"""Tests for StateGraph Pydantic state validation."""

import pytest
from typing_extensions import TypedDict
from pydantic import BaseModel, Field, ValidationError, ConfigDict

from victor.framework.graph import StateGraph, StateValidator, StateValidationError

# =============================================================================
# Test State Models
# =============================================================================


class PydanticState(BaseModel):
    """Pydantic state model for validation."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    messages: list[str] = Field(default_factory=list)
    task: str = Field(default="", description="Task description")
    count: int = 0


class PydanticStateStrict(BaseModel):
    """Pydantic state with truly required fields."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    messages: list[str] = Field(default_factory=list)
    task: str = Field(..., description="Required task")  # Required field
    count: int = 0


class TypedDictState(TypedDict):
    """TypedDict state for comparison."""

    messages: list[str]
    task: str
    count: int


# =============================================================================
# Test StateValidator
# =============================================================================


class TestStateValidator:
    """Tests for StateValidator class."""

    def test_validate_pydantic_state_valid(self):
        """validate should pass for valid Pydantic state."""
        validator = StateValidator(PydanticState, strict=False)

        state = {"messages": [], "task": "test", "count": 0}
        errors = validator.validate(state)

        assert errors == []

    def test_validate_pydantic_state_missing_required(self):
        """validate should fail for missing required field."""
        validator = StateValidator(PydanticStateStrict, strict=False)

        state = {"messages": [], "count": 0}  # Missing 'task'
        errors = validator.validate(state)

        assert len(errors) > 0
        assert any("task" in e for e in errors)

    def test_validate_pydantic_state_wrong_type(self):
        """validate should fail for wrong type."""
        validator = StateValidator(PydanticState, strict=False)

        state = {"messages": [], "task": "test", "count": "wrong"}  # Should be int
        errors = validator.validate(state)

        assert len(errors) > 0
        assert any("count" in e for e in errors)

    def test_validate_typeddict_state_valid(self):
        """validate should pass for valid TypedDict state."""
        validator = StateValidator(TypedDictState, strict=False)

        state = {"messages": [], "task": "test", "count": 0}
        errors = validator.validate(state)

        assert errors == []

    def test_validate_typeddict_state_missing_required(self):
        """validate should fail for missing required field in TypedDict."""
        validator = StateValidator(TypedDictState, strict=False)

        state = {"messages": []}  # Missing 'task' and 'count'
        errors = validator.validate(state)

        # TypedDict validation checks for missing required keys
        # All fields in TypedDict are required by default
        assert len(errors) >= 2  # At least 2 missing fields

    def test_validate_no_schema(self):
        """validate should pass when no schema provided."""
        validator = StateValidator(None, strict=False)

        state = {"any": "value", "random": "stuff"}
        errors = validator.validate(state)

        assert errors == []

    def test_validate_pydantic_with_extra_fields(self):
        """validate should handle extra fields gracefully."""
        validator = StateValidator(PydanticState, strict=False)

        state = {"messages": [], "task": "test", "count": 0, "extra": "ignored"}
        # Pydantic allows extra fields by default
        errors = validator.validate(state)

        # Extra fields are ignored by default
        assert len(errors) == 0 or any("extra" in e.lower() for e in errors)


# =============================================================================
# Test StateGraph Integration
# =============================================================================


class TestStateGraphValidation:
    """Tests for StateGraph validation integration."""

    @pytest.mark.asyncio
    async def test_invoke_with_valid_pydantic_state(self):
        """invoke should work with valid Pydantic state."""
        from victor.framework.graph import END

        async def increment_node(state):
            state["count"] += 1
            return state

        graph = StateGraph(PydanticState)
        graph.add_node("inc", increment_node)
        graph.add_edge("inc", END)
        graph.set_entry_point("inc")

        compiled = graph.compile()

        # Enable validation
        from victor.framework.config import GraphConfig, ValidationConfig

        config = GraphConfig(
            validation=ValidationConfig(
                enabled=True,
                strict=False,
                validate_on_entry=True,
                validate_after_nodes=True,
            )
        )

        state = {"messages": [], "task": "test", "count": 0}
        result = await compiled.invoke(state, config=config)

        assert result.success is True
        assert result.state["count"] == 1

    @pytest.mark.asyncio
    async def test_invoke_with_invalid_state_strict_mode(self):
        """invoke should raise in strict mode with invalid state."""
        from victor.framework.graph import END

        async def increment_node(state):
            state["count"] += 1
            return state

        # Use strict state with required field
        graph = StateGraph(PydanticStateStrict)
        graph.add_node("inc", increment_node)
        graph.add_edge("inc", END)
        graph.set_entry_point("inc")

        compiled = graph.compile()

        # Enable strict validation
        from victor.framework.config import GraphConfig, ValidationConfig

        config = GraphConfig(
            validation=ValidationConfig(enabled=True, strict=True, validate_on_entry=True)
        )

        # Missing required field 'task'
        state = {"messages": [], "count": 0}

        with pytest.raises(StateValidationError):
            await compiled.invoke(state, config=config)

    @pytest.mark.asyncio
    async def test_invoke_with_invalid_state_lenient_mode(self):
        """invoke should log errors in lenient mode but continue."""
        from victor.framework.graph import END

        async def increment_node(state):
            state["count"] += 1
            return state

        graph = StateGraph(PydanticState)
        graph.add_node("inc", increment_node)
        graph.add_edge("inc", END)
        graph.set_entry_point("inc")

        compiled = graph.compile()

        # Enable lenient validation (non-strict)
        from victor.framework.config import GraphConfig, ValidationConfig

        config = GraphConfig(
            validation=ValidationConfig(
                enabled=True, strict=False, validate_on_entry=True, log_errors=True
            )
        )

        # Missing required field 'task' - should log warning but continue
        state = {"messages": [], "count": 0}
        result = await compiled.invoke(state, config=config)

        # In lenient mode, execution continues
        # (May fail later if node expects the field)

    @pytest.mark.asyncio
    async def test_invoke_backward_compatible_no_validation(self):
        """invoke should work without validation (backward compatible)."""
        from victor.framework.graph import END

        async def increment_node(state):
            state["count"] += 1
            return state

        # TypedDict state (no validation by default)
        graph = StateGraph(TypedDictState)
        graph.add_node("inc", increment_node)
        graph.add_edge("inc", END)
        graph.set_entry_point("inc")

        compiled = graph.compile()

        state = {"messages": [], "task": "test", "count": 0}
        result = await compiled.invoke(state)

        assert result.success is True
        assert result.state["count"] == 1

    @pytest.mark.asyncio
    async def test_invoke_validation_disabled_by_default(self):
        """invoke should not validate when validation disabled (default)."""
        from victor.framework.graph import END

        async def increment_node(state):
            state["count"] += 1
            return state

        graph = StateGraph(PydanticState)
        graph.add_node("inc", increment_node)
        graph.add_edge("inc", END)
        graph.set_entry_point("inc")

        compiled = graph.compile()

        # Invalid state but validation disabled
        state = {"messages": [], "count": 0}  # Missing 'task'
        result = await compiled.invoke(state)

        # Should succeed (no validation by default)
        assert result.success is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
