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

"""Unit tests for EscapeHatchRegistry."""

import pytest
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from victor.framework.escape_hatch_registry import (
    ConditionFunction,
    TransformFunction,
    EscapeHatchRegistry,
    get_escape_hatch_registry,
    condition,
    transform,
)


# Sample functions for testing
def sample_condition(ctx: Dict[str, Any]) -> str:
    """Sample condition function."""
    if ctx.get("value", 0) > 10:
        return "high"
    return "low"


def sample_transform(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Sample transform function."""
    return {"result": ctx.get("a", 0) + ctx.get("b", 0)}


class TestConditionFunctionProtocol:
    """Tests for ConditionFunction protocol."""

    def test_sample_condition_matches_protocol(self):
        """Test that sample condition matches the protocol."""
        # Protocol check - should accept the function
        fn: ConditionFunction = sample_condition
        result = fn({"value": 15})
        assert result == "high"


class TestTransformFunctionProtocol:
    """Tests for TransformFunction protocol."""

    def test_sample_transform_matches_protocol(self):
        """Test that sample transform matches the protocol."""
        # Protocol check - should accept the function
        fn: TransformFunction = sample_transform
        result = fn({"a": 5, "b": 3})
        assert result == {"result": 8}


class TestEscapeHatchRegistry:
    """Tests for EscapeHatchRegistry class."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset singleton before each test."""
        EscapeHatchRegistry.reset_instance()
        yield
        EscapeHatchRegistry.reset_instance()

    def test_singleton_instance(self):
        """Test registry is a singleton."""
        reg1 = EscapeHatchRegistry.get_instance()
        reg2 = EscapeHatchRegistry.get_instance()
        assert reg1 is reg2

    def test_reset_instance(self):
        """Test reset creates new instance."""
        reg1 = EscapeHatchRegistry.get_instance()
        reg1.register_condition("test", sample_condition)

        EscapeHatchRegistry.reset_instance()
        reg2 = EscapeHatchRegistry.get_instance()

        assert reg1 is not reg2
        assert reg2.get_condition("test") is None

    # --- Condition Registration Tests ---

    def test_register_condition_global(self):
        """Test registering a global condition."""
        registry = EscapeHatchRegistry()

        registry.register_condition("my_cond", sample_condition)

        assert registry.get_condition("my_cond") is sample_condition

    def test_register_condition_with_vertical(self):
        """Test registering a condition for a specific vertical."""
        registry = EscapeHatchRegistry()

        registry.register_condition("my_cond", sample_condition, vertical="coding")

        assert registry.get_condition("my_cond", vertical="coding") is sample_condition
        # Should not be in global
        assert registry.get_condition("my_cond") is None

    def test_register_condition_duplicate_raises(self):
        """Test duplicate registration raises error."""
        registry = EscapeHatchRegistry()
        registry.register_condition("my_cond", sample_condition)

        with pytest.raises(ValueError, match="already registered"):
            registry.register_condition("my_cond", sample_condition)

    def test_register_condition_duplicate_with_replace(self):
        """Test duplicate registration with replace flag."""
        registry = EscapeHatchRegistry()

        def cond1(ctx):
            return "first"

        def cond2(ctx):
            return "second"

        registry.register_condition("my_cond", cond1)
        registry.register_condition("my_cond", cond2, replace=True)

        fn = registry.get_condition("my_cond")
        assert fn({}) == "second"

    # --- Transform Registration Tests ---

    def test_register_transform_global(self):
        """Test registering a global transform."""
        registry = EscapeHatchRegistry()

        registry.register_transform("my_trans", sample_transform)

        assert registry.get_transform("my_trans") is sample_transform

    def test_register_transform_with_vertical(self):
        """Test registering a transform for a specific vertical."""
        registry = EscapeHatchRegistry()

        registry.register_transform("my_trans", sample_transform, vertical="research")

        assert registry.get_transform("my_trans", vertical="research") is sample_transform
        # Should not be in global
        assert registry.get_transform("my_trans") is None

    def test_register_transform_duplicate_raises(self):
        """Test duplicate registration raises error."""
        registry = EscapeHatchRegistry()
        registry.register_transform("my_trans", sample_transform)

        with pytest.raises(ValueError, match="already registered"):
            registry.register_transform("my_trans", sample_transform)

    def test_register_transform_duplicate_with_replace(self):
        """Test duplicate registration with replace flag."""
        registry = EscapeHatchRegistry()

        def trans1(ctx):
            return {"v": 1}

        def trans2(ctx):
            return {"v": 2}

        registry.register_transform("my_trans", trans1)
        registry.register_transform("my_trans", trans2, replace=True)

        fn = registry.get_transform("my_trans")
        assert fn({}) == {"v": 2}

    # --- Bulk Registration Tests ---

    def test_register_from_vertical(self):
        """Test bulk registration from vertical."""
        registry = EscapeHatchRegistry()

        conditions = {
            "cond1": lambda ctx: "a",
            "cond2": lambda ctx: "b",
        }
        transforms = {
            "trans1": lambda ctx: {"x": 1},
            "trans2": lambda ctx: {"y": 2},
        }

        cond_count, trans_count = registry.register_from_vertical(
            "coding",
            conditions=conditions,
            transforms=transforms,
        )

        assert cond_count == 2
        assert trans_count == 2
        assert registry.get_condition("cond1", vertical="coding") is not None
        assert registry.get_transform("trans1", vertical="coding") is not None

    def test_register_from_vertical_empty(self):
        """Test bulk registration with no items."""
        registry = EscapeHatchRegistry()

        cond_count, trans_count = registry.register_from_vertical("coding")

        assert cond_count == 0
        assert trans_count == 0

    # --- Get Registry For Vertical Tests ---

    def test_get_registry_for_vertical_returns_both_dicts(self):
        """Test get_registry_for_vertical returns both dicts."""
        registry = EscapeHatchRegistry()

        registry.register_condition("cond1", sample_condition, vertical="coding")
        registry.register_transform("trans1", sample_transform, vertical="coding")

        conditions, transforms = registry.get_registry_for_vertical("coding")

        assert "cond1" in conditions
        assert "trans1" in transforms

    def test_get_registry_for_vertical_includes_global(self):
        """Test get_registry_for_vertical includes global entries."""
        registry = EscapeHatchRegistry()

        # Register global
        registry.register_condition("global_cond", sample_condition)
        registry.register_transform("global_trans", sample_transform)

        # Register vertical-specific
        registry.register_condition("vert_cond", sample_condition, vertical="coding")

        conditions, transforms = registry.get_registry_for_vertical("coding")

        assert "global_cond" in conditions
        assert "vert_cond" in conditions
        assert "global_trans" in transforms

    def test_get_registry_for_vertical_excludes_global(self):
        """Test get_registry_for_vertical can exclude global entries."""
        registry = EscapeHatchRegistry()

        # Register global
        registry.register_condition("global_cond", sample_condition)

        # Register vertical-specific
        registry.register_condition("vert_cond", sample_condition, vertical="coding")

        conditions, transforms = registry.get_registry_for_vertical(
            "coding",
            include_global=False,
        )

        assert "global_cond" not in conditions
        assert "vert_cond" in conditions

    def test_get_registry_for_vertical_overrides_global(self):
        """Test vertical-specific entries override global."""
        registry = EscapeHatchRegistry()

        def global_fn(ctx):
            return "global"

        def vertical_fn(ctx):
            return "vertical"

        registry.register_condition("shared", global_fn)
        registry.register_condition("shared", vertical_fn, vertical="coding")

        conditions, _ = registry.get_registry_for_vertical("coding")

        # Vertical-specific should override global
        assert conditions["shared"]({}) == "vertical"

    # --- Lookup Tests ---

    def test_get_condition_checks_vertical_first(self):
        """Test get_condition checks vertical-specific first."""
        registry = EscapeHatchRegistry()

        def global_fn(ctx):
            return "global"

        def vertical_fn(ctx):
            return "vertical"

        registry.register_condition("shared", global_fn)
        registry.register_condition("shared", vertical_fn, vertical="coding")

        # Without vertical, returns global
        assert registry.get_condition("shared")({}) == "global"

        # With vertical, returns vertical-specific
        assert registry.get_condition("shared", vertical="coding")({}) == "vertical"

    def test_get_condition_falls_back_to_global(self):
        """Test get_condition falls back to global if not in vertical."""
        registry = EscapeHatchRegistry()

        registry.register_condition("global_only", sample_condition)

        # Should find in global when vertical specified but not found
        fn = registry.get_condition("global_only", vertical="coding")
        assert fn is sample_condition

    def test_get_condition_returns_none_if_not_found(self):
        """Test get_condition returns None if not found."""
        registry = EscapeHatchRegistry()

        assert registry.get_condition("nonexistent") is None
        assert registry.get_condition("nonexistent", vertical="coding") is None

    def test_get_transform_checks_vertical_first(self):
        """Test get_transform checks vertical-specific first."""
        registry = EscapeHatchRegistry()

        def global_fn(ctx):
            return {"source": "global"}

        def vertical_fn(ctx):
            return {"source": "vertical"}

        registry.register_transform("shared", global_fn)
        registry.register_transform("shared", vertical_fn, vertical="research")

        # Without vertical, returns global
        assert registry.get_transform("shared")({})["source"] == "global"

        # With vertical, returns vertical-specific
        assert registry.get_transform("shared", vertical="research")({})["source"] == "vertical"

    # --- List Tests ---

    def test_list_conditions_all(self):
        """Test list_conditions returns all conditions."""
        registry = EscapeHatchRegistry()

        registry.register_condition("global1", sample_condition)
        registry.register_condition("v1", sample_condition, vertical="coding")
        registry.register_condition("v2", sample_condition, vertical="research")

        names = registry.list_conditions()

        assert set(names) == {"global1", "v1", "v2"}

    def test_list_conditions_by_vertical(self):
        """Test list_conditions filtered by vertical."""
        registry = EscapeHatchRegistry()

        registry.register_condition("global1", sample_condition)
        registry.register_condition("c1", sample_condition, vertical="coding")
        registry.register_condition("c2", sample_condition, vertical="coding")
        registry.register_condition("r1", sample_condition, vertical="research")

        names = registry.list_conditions(vertical="coding")

        assert set(names) == {"c1", "c2"}

    def test_list_transforms_all(self):
        """Test list_transforms returns all transforms."""
        registry = EscapeHatchRegistry()

        registry.register_transform("global1", sample_transform)
        registry.register_transform("v1", sample_transform, vertical="coding")

        names = registry.list_transforms()

        assert set(names) == {"global1", "v1"}

    def test_list_transforms_by_vertical(self):
        """Test list_transforms filtered by vertical."""
        registry = EscapeHatchRegistry()

        registry.register_transform("global1", sample_transform)
        registry.register_transform("c1", sample_transform, vertical="coding")

        names = registry.list_transforms(vertical="coding")

        assert set(names) == {"c1"}

    def test_list_verticals(self):
        """Test list_verticals returns all registered verticals."""
        registry = EscapeHatchRegistry()

        registry.register_condition("c1", sample_condition, vertical="coding")
        registry.register_transform("t1", sample_transform, vertical="research")
        registry.register_condition("c2", sample_condition, vertical="devops")

        verticals = registry.list_verticals()

        assert set(verticals) == {"coding", "research", "devops"}

    # --- Clear Tests ---

    def test_clear_all(self):
        """Test clear removes all entries."""
        registry = EscapeHatchRegistry()

        registry.register_condition("c1", sample_condition)
        registry.register_condition("c2", sample_condition, vertical="coding")
        registry.register_transform("t1", sample_transform)

        registry.clear()

        assert len(registry.list_conditions()) == 0
        assert len(registry.list_transforms()) == 0

    def test_clear_by_vertical(self):
        """Test clear with vertical only clears that vertical."""
        registry = EscapeHatchRegistry()

        registry.register_condition("c1", sample_condition)
        registry.register_condition("c2", sample_condition, vertical="coding")
        registry.register_condition("c3", sample_condition, vertical="research")

        registry.clear(vertical="coding")

        assert registry.get_condition("c1") is not None
        assert registry.get_condition("c2", vertical="coding") is None
        assert registry.get_condition("c3", vertical="research") is not None

    # --- Discover Tests ---

    def test_discover_from_vertical_imports_and_registers(self):
        """Test discover_from_vertical imports and registers escape hatches."""
        registry = EscapeHatchRegistry()
        fake_module = type(
            "_FakeEscapeHatches",
            (),
            {
                "CONDITIONS": {"tests_passing": sample_condition},
                "TRANSFORMS": {"merge_results": sample_transform},
            },
        )()

        with patch(
            "victor.framework.escape_hatch_registry.importlib.import_module",
            return_value=fake_module,
        ):
            cond_count, trans_count = registry.discover_from_vertical("coding")

        assert cond_count > 0
        assert trans_count > 0

        assert registry.get_condition("tests_passing", vertical="coding") is not None

    def test_discover_from_vertical_raises_for_invalid(self):
        """Test discover_from_vertical raises for invalid vertical."""
        registry = EscapeHatchRegistry()

        with pytest.raises(ImportError):
            registry.discover_from_vertical("nonexistent_vertical_xyz")


class TestModuleLevelFunctions:
    """Tests for module-level convenience functions."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset singleton before each test."""
        EscapeHatchRegistry.reset_instance()
        yield
        EscapeHatchRegistry.reset_instance()

    def test_get_escape_hatch_registry_returns_singleton(self):
        """Test get_escape_hatch_registry returns singleton."""
        reg1 = get_escape_hatch_registry()
        reg2 = get_escape_hatch_registry()
        assert reg1 is reg2


class TestConditionDecorator:
    """Tests for @condition decorator."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset singleton before each test."""
        EscapeHatchRegistry.reset_instance()
        yield
        EscapeHatchRegistry.reset_instance()

    def test_condition_decorator_registers_function(self):
        """Test @condition decorator registers the function."""

        @condition("my_test_cond")
        def my_condition(ctx: Dict[str, Any]) -> str:
            return "test"

        registry = get_escape_hatch_registry()
        assert registry.get_condition("my_test_cond") is my_condition

    def test_condition_decorator_with_vertical(self):
        """Test @condition decorator with vertical."""

        @condition("my_test_cond", vertical="coding")
        def my_condition(ctx: Dict[str, Any]) -> str:
            return "test"

        registry = get_escape_hatch_registry()
        assert registry.get_condition("my_test_cond", vertical="coding") is my_condition
        assert registry.get_condition("my_test_cond") is None

    def test_condition_decorator_returns_original_function(self):
        """Test @condition decorator returns the original function."""

        @condition("my_test_cond")
        def my_condition(ctx: Dict[str, Any]) -> str:
            return "result"

        # Function should still be callable directly
        assert my_condition({}) == "result"


class TestTransformDecorator:
    """Tests for @transform decorator."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset singleton before each test."""
        EscapeHatchRegistry.reset_instance()
        yield
        EscapeHatchRegistry.reset_instance()

    def test_transform_decorator_registers_function(self):
        """Test @transform decorator registers the function."""

        @transform("my_test_trans")
        def my_transform(ctx: Dict[str, Any]) -> Dict[str, Any]:
            return {"result": "test"}

        registry = get_escape_hatch_registry()
        assert registry.get_transform("my_test_trans") is my_transform

    def test_transform_decorator_with_vertical(self):
        """Test @transform decorator with vertical."""

        @transform("my_test_trans", vertical="research")
        def my_transform(ctx: Dict[str, Any]) -> Dict[str, Any]:
            return {"result": "test"}

        registry = get_escape_hatch_registry()
        assert registry.get_transform("my_test_trans", vertical="research") is my_transform
        assert registry.get_transform("my_test_trans") is None

    def test_transform_decorator_returns_original_function(self):
        """Test @transform decorator returns the original function."""

        @transform("my_test_trans")
        def my_transform(ctx: Dict[str, Any]) -> Dict[str, Any]:
            return {"value": 42}

        # Function should still be callable directly
        assert my_transform({}) == {"value": 42}
