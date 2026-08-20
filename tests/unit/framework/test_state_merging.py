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

"""Unit tests for state merging strategies."""

import pytest

from victor.framework.state_merging import (
    DictMergeStrategy,
    ListMergeStrategy,
    CustomMergeStrategy,
    SelectiveMergeStrategy,
    MergeMode,
    StateMergeError,
    validate_merged_state,
    create_merge_strategy,
    dict_merge_strategy,
    list_merge_strategy,
    custom_merge_strategy,
)


class TestDictMergeStrategy:
    """Tests for DictMergeStrategy."""

    def test_merge_new_keys(self):
        """Test merging new keys from team state."""
        graph_state = {"key1": "value1"}
        team_state = {"key2": "value2"}

        strategy = DictMergeStrategy(mode=MergeMode.TEAM_WINS)
        merged = strategy.merge(graph_state, team_state)

        assert merged == {"key1": "value1", "key2": "value2"}

    def test_merge_conflict_team_wins(self):
        """Test conflict resolution with team_wins mode."""
        graph_state = {"key": "graph_value"}
        team_state = {"key": "team_value"}

        strategy = DictMergeStrategy(mode=MergeMode.TEAM_WINS)
        merged = strategy.merge(graph_state, team_state)

        assert merged["key"] == "team_value"

    def test_merge_conflict_graph_wins(self):
        """Test conflict resolution with graph_wins mode."""
        graph_state = {"key": "graph_value"}
        team_state = {"key": "team_value"}

        strategy = DictMergeStrategy(mode=MergeMode.GRAPH_WINS)
        merged = strategy.merge(graph_state, team_state)

        assert merged["key"] == "graph_value"

    def test_merge_conflict_error(self):
        """Test conflict resolution with error mode."""
        graph_state = {"key": "graph_value"}
        team_state = {"key": "team_value"}

        strategy = DictMergeStrategy(mode=MergeMode.ERROR)

        with pytest.raises(StateMergeError) as exc_info:
            strategy.merge(graph_state, team_state)

        assert exc_info.value.key == "key"
        assert exc_info.value.graph_value == "graph_value"
        assert exc_info.value.team_value == "team_value"

    def test_merge_nested_dicts(self):
        """Test merging nested dictionaries."""
        graph_state = {
            "level1": {
                "key1": "value1",
                "shared": "graph",
            }
        }
        team_state = {
            "level1": {
                "key2": "value2",
                "shared": "team",
            }
        }

        strategy = DictMergeStrategy(mode=MergeMode.TEAM_WINS)
        merged = strategy.merge(graph_state, team_state)

        assert merged["level1"]["key1"] == "value1"
        assert merged["level1"]["key2"] == "value2"
        assert merged["level1"]["shared"] == "team"

    def test_merge_mode_merge_compatible_types(self):
        """Test merge mode with compatible types."""
        graph_state = {"list": [1, 2]}
        team_state = {"list": [3, 4]}

        strategy = DictMergeStrategy(mode=MergeMode.MERGE)
        merged = strategy.merge(graph_state, team_state)

        assert merged["list"] == [1, 2, 3, 4]

    def test_merge_mode_merge_incompatible_types(self):
        """Test merge mode with incompatible types falls back to team."""
        graph_state = {"key": "string"}
        team_state = {"key": 123}

        strategy = DictMergeStrategy(mode=MergeMode.MERGE)
        merged = strategy.merge(graph_state, team_state)

        # Should fall back to team value
        assert merged["key"] == 123


class TestListMergeStrategy:
    """Tests for ListMergeStrategy."""

    def test_merge_lists_concatenate(self):
        """Test merging lists by concatenation."""
        graph_state = {"items": [1, 2]}
        team_state = {"items": [3, 4]}

        strategy = ListMergeStrategy(mode=MergeMode.TEAM_WINS)
        merged = strategy.merge(graph_state, team_state)

        assert merged["items"] == [1, 2, 3, 4]

    def test_merge_lists_deduplicate(self):
        """Test merging lists with deduplication."""
        graph_state = {"items": [1, 2, 3]}
        team_state = {"items": [3, 4, 5]}

        strategy = ListMergeStrategy(mode=MergeMode.TEAM_WINS)
        merged = strategy.merge(graph_state, team_state, deduplicate=True)

        assert merged["items"] == [1, 2, 3, 4, 5]

    def test_merge_lists_with_list_keys(self):
        """Test merging with explicit list keys."""
        graph_state = {"items": [1, 2], "other": "value"}
        team_state = {"items": [3, 4], "other": "new_value"}

        strategy = ListMergeStrategy(mode=MergeMode.TEAM_WINS)
        merged = strategy.merge(graph_state, team_state, list_keys=["items"])

        assert merged["items"] == [1, 2, 3, 4]
        assert merged["other"] == "new_value"  # Non-list uses default merge

    def test_merge_non_list_keys(self):
        """Test that non-list keys use default merge behavior."""
        graph_state = {"key": "value"}
        team_state = {"key": "new_value"}

        strategy = ListMergeStrategy(mode=MergeMode.TEAM_WINS)
        merged = strategy.merge(graph_state, team_state)

        assert merged["key"] == "new_value"


class TestCustomMergeStrategy:
    """Tests for CustomMergeStrategy."""

    def test_custom_conflict_resolver(self):
        """Test custom conflict resolver function."""
        graph_state = {"key": "graph"}
        team_state = {"key": "team"}

        def resolve_conflict(key, graph_val, team_val):
            if key == "key":
                return f"{graph_val}+{team_val}"
            return team_val

        strategy = CustomMergeStrategy(conflict_resolver=resolve_conflict, mode=MergeMode.TEAM_WINS)
        merged = strategy.merge(graph_state, team_state)

        assert merged["key"] == "graph+team"

    def test_custom_resolver_returns_none(self):
        """Test that None from resolver falls back to mode."""
        graph_state = {"key": "graph"}
        team_state = {"key": "team"}

        def resolve_conflict(key, graph_val, team_val):
            return None  # Fall back to default

        strategy = CustomMergeStrategy(
            conflict_resolver=resolve_conflict, mode=MergeMode.GRAPH_WINS
        )
        merged = strategy.merge(graph_state, team_state)

        assert merged["key"] == "graph"  # Graph wins from fallback mode

    def test_custom_resolver_with_nested_dicts(self):
        """Test custom resolver doesn't prevent recursive dict merging."""
        graph_state = {"level1": {"key1": "value1", "shared": "graph"}}
        team_state = {"level1": {"key2": "value2", "shared": "team"}}

        # Custom resolver only applies to top-level conflicts
        def resolve_conflict(key, graph_val, team_val):
            if key == "special":
                return "custom"
            return None

        strategy = CustomMergeStrategy(conflict_resolver=resolve_conflict, mode=MergeMode.TEAM_WINS)
        merged = strategy.merge(graph_state, team_state)

        # Nested dict should still be merged recursively
        assert merged["level1"]["key1"] == "value1"
        assert merged["level1"]["key2"] == "value2"
        assert merged["level1"]["shared"] == "team"


class TestSelectiveMergeStrategy:
    """Tests for SelectiveMergeStrategy."""

    def test_merge_only_selected_keys(self):
        """Test merging only specified keys."""
        graph_state = {"key1": "value1", "key2": "value2", "key3": "value3"}
        team_state = {"key1": "new1", "key2": "new2", "key3": "new3"}

        strategy = SelectiveMergeStrategy(keys_to_merge=["key1", "key2"], mode=MergeMode.TEAM_WINS)
        merged = strategy.merge(graph_state, team_state)

        assert merged["key1"] == "new1"
        assert merged["key2"] == "new2"
        assert merged["key3"] == "value3"  # Not merged

    def test_merge_missing_key_in_team_state(self):
        """Test handling of missing keys in team state."""
        graph_state = {"key1": "value1", "key2": "value2"}
        team_state = {"key2": "new2"}  # key1 missing

        strategy = SelectiveMergeStrategy(keys_to_merge=["key1", "key2"], mode=MergeMode.TEAM_WINS)
        merged = strategy.merge(graph_state, team_state)

        assert merged["key1"] == "value1"  # Unchanged
        assert merged["key2"] == "new2"

    def test_recursive_false(self):
        """Test selective merge without recursion."""
        graph_state = {"nested": {"key": "value"}}
        team_state = {"nested": {"new_key": "new_value"}}

        strategy = SelectiveMergeStrategy(
            keys_to_merge=["nested"], mode=MergeMode.TEAM_WINS, recursive=False
        )
        merged = strategy.merge(graph_state, team_state)

        # Without recursion, entire dict is replaced
        assert merged["nested"] == {"new_key": "new_value"}


class TestValidateMergedState:
    """Tests for validate_merged_state function."""

    def test_validate_with_required_keys(self):
        """Test validation with required keys."""
        merged_state = {"key1": "value1", "key2": "value2"}

        # Should pass
        assert validate_merged_state(merged_state, required_keys=["key1", "key2"])

    def test_validate_missing_required_key(self):
        """Test validation fails with missing required key."""
        merged_state = {"key1": "value1"}

        with pytest.raises(StateMergeError) as exc_info:
            validate_merged_state(merged_state, required_keys=["key1", "key2"])

        assert "Missing required keys" in str(exc_info.value)

    def test_validate_with_forbidden_keys(self):
        """Test validation with forbidden keys."""
        merged_state = {"key1": "value1"}

        # Should pass
        assert validate_merged_state(merged_state, forbidden_keys=["key2"])

    def test_validate_forbidden_key_present(self):
        """Test validation fails with forbidden key present."""
        merged_state = {"key1": "value1", "_internal": "secret"}

        with pytest.raises(StateMergeError) as exc_info:
            validate_merged_state(merged_state, forbidden_keys=["_internal"])

        assert "Forbidden keys present" in str(exc_info.value)

    def test_validate_with_custom_validator(self):
        """Test validation with custom validator function."""
        merged_state = {"count": 5}

        # Should pass
        assert validate_merged_state(merged_state, validators={"count": lambda x: x > 0})

    def test_validate_custom_validator_fails(self):
        """Test validation fails with custom validator."""
        merged_state = {"count": -1}

        with pytest.raises(StateMergeError) as exc_info:
            validate_merged_state(merged_state, validators={"count": lambda x: x > 0})

        assert "Validation failed for key 'count'" in str(exc_info.value)


class TestCreateMergeStrategy:
    """Tests for create_merge_strategy factory function."""

    def test_create_dict_strategy(self):
        """Test creating dict merge strategy."""
        strategy = create_merge_strategy("dict", mode=MergeMode.TEAM_WINS)
        assert isinstance(strategy, DictMergeStrategy)

    def test_create_list_strategy(self):
        """Test creating list merge strategy."""
        strategy = create_merge_strategy("list", mode=MergeMode.TEAM_WINS)
        assert isinstance(strategy, ListMergeStrategy)

    def test_create_custom_strategy(self):
        """Test creating custom merge strategy."""
        strategy = create_merge_strategy("custom", mode=MergeMode.TEAM_WINS)
        assert isinstance(strategy, CustomMergeStrategy)

    def test_create_selective_strategy(self):
        """Test creating selective merge strategy."""
        strategy = create_merge_strategy(
            "selective", mode=MergeMode.TEAM_WINS, keys_to_merge=["key1"]
        )
        assert isinstance(strategy, SelectiveMergeStrategy)

    def test_create_unknown_strategy(self):
        """Test creating unknown strategy type raises error."""
        with pytest.raises(ValueError) as exc_info:
            create_merge_strategy("unknown")

        assert "Unknown merge strategy type" in str(exc_info.value)


class TestDefaultStrategies:
    """Tests for default strategy instances."""

    def test_dict_merge_strategy_default(self):
        """Test default dict merge strategy."""
        graph_state = {"key": "graph"}
        team_state = {"key": "team"}

        merged = dict_merge_strategy.merge(graph_state, team_state)
        assert merged["key"] == "team"  # TEAM_WINS by default

    def test_list_merge_strategy_default(self):
        """Test default list merge strategy."""
        graph_state = {"items": [1, 2]}
        team_state = {"items": [3, 4]}

        merged = list_merge_strategy.merge(graph_state, team_state)
        assert merged["items"] == [1, 2, 3, 4]

    def test_custom_merge_strategy_default(self):
        """Test default custom merge strategy."""
        # Default has no resolver, falls back to TEAM_WINS
        graph_state = {"key": "graph"}
        team_state = {"key": "team"}

        merged = custom_merge_strategy.merge(graph_state, team_state)
        assert merged["key"] == "team"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_merge_empty_graph_state(self):
        """Test merging with empty graph state."""
        graph_state = {}
        team_state = {"key": "value"}

        strategy = DictMergeStrategy(mode=MergeMode.TEAM_WINS)
        merged = strategy.merge(graph_state, team_state)

        assert merged == {"key": "value"}

    def test_merge_empty_team_state(self):
        """Test merging with empty team state."""
        graph_state = {"key": "value"}
        team_state = {}

        strategy = DictMergeStrategy(mode=MergeMode.TEAM_WINS)
        merged = strategy.merge(graph_state, team_state)

        assert merged == {"key": "value"}

    def test_merge_both_empty(self):
        """Test merging when both states are empty."""
        graph_state = {}
        team_state = {}

        strategy = DictMergeStrategy(mode=MergeMode.TEAM_WINS)
        merged = strategy.merge(graph_state, team_state)

        assert merged == {}

    def test_merge_preserves_graph_state_immutability(self):
        """Test that graph state is not modified during merge."""
        original_graph = {"key": "value"}
        graph_state = original_graph.copy()
        team_state = {"key2": "value2"}

        strategy = DictMergeStrategy(mode=MergeMode.TEAM_WINS)
        merged = strategy.merge(graph_state, team_state)

        # Original should be unchanged
        assert graph_state == original_graph
        # Merged should have both
        assert merged == {"key": "value", "key2": "value2"}
