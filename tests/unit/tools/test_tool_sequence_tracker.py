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

"""Tests for ToolSequenceTracker."""

from victor.agent.tool_sequence_tracker import (
    COMMON_TOOL_SEQUENCES,
    WORKFLOW_PATTERNS,
    SequenceTrackerConfig,
    ToolSequenceTracker,
    TransitionStats,
    create_sequence_tracker,
)


class TestSequenceTrackerConfig:
    """Tests for SequenceTrackerConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SequenceTrackerConfig()

        assert config.use_predefined_patterns is True
        assert config.learning_rate == 0.3
        assert config.decay_factor == 0.95
        assert config.max_history == 50
        assert config.boost_multiplier == 1.15
        assert config.min_observations == 3

    def test_custom_values(self):
        """Test custom configuration values."""
        config = SequenceTrackerConfig(
            use_predefined_patterns=False,
            learning_rate=0.5,
            max_history=100,
        )

        assert config.use_predefined_patterns is False
        assert config.learning_rate == 0.5
        assert config.max_history == 100


class TestTransitionStats:
    """Tests for TransitionStats."""

    def test_default_values(self):
        """Test default stats values."""
        stats = TransitionStats()

        assert stats.count == 0
        assert stats.success_rate == 1.0
        assert stats.avg_time_between == 0.0


class TestToolSequenceTrackerInit:
    """Tests for ToolSequenceTracker initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        tracker = ToolSequenceTracker()

        assert tracker.config.use_predefined_patterns is True
        assert len(tracker._history) == 0
        # Should have predefined patterns loaded
        assert len(tracker._transitions) > 0

    def test_initialization_without_predefined(self):
        """Test initialization without predefined patterns."""
        config = SequenceTrackerConfig(use_predefined_patterns=False)
        tracker = ToolSequenceTracker(config)

        assert len(tracker._transitions) == 0

    def test_predefined_patterns_loaded(self):
        """Test that predefined patterns are correctly loaded."""
        tracker = ToolSequenceTracker()

        # Check a known pattern
        assert "read" in tracker._transitions
        assert "edit" in tracker._transitions["read"]


class TestRecordExecution:
    """Tests for recording tool executions."""

    def test_record_single_execution(self):
        """Test recording a single tool execution."""
        tracker = ToolSequenceTracker()
        tracker.record_execution("read_file")

        assert len(tracker._history) == 1
        assert tracker._history[0] == "read"

    def test_record_multiple_executions(self):
        """Test recording multiple tool executions."""
        tracker = ToolSequenceTracker()
        tracker.record_execution("read_file")
        tracker.record_execution("edit_files")
        tracker.record_execution("run_tests")

        assert len(tracker._history) == 3
        assert tracker._history == ["read", "edit", "test"]

    def test_transition_updated(self):
        """Test that transitions are updated on execution."""
        config = SequenceTrackerConfig(use_predefined_patterns=False)
        tracker = ToolSequenceTracker(config)

        tracker.record_execution("read_file")
        tracker.record_execution("edit_files")

        # Transition should be recorded
        assert tracker._transitions["read"]["edit"].count == 1

    def test_history_trimmed(self):
        """Test that history is trimmed at max_history."""
        config = SequenceTrackerConfig(max_history=5)
        tracker = ToolSequenceTracker(config)

        for i in range(10):
            tracker.record_execution(f"tool_{i}")

        assert len(tracker._history) == 5
        assert tracker._history[0] == "tool_5"  # Oldest trimmed

    def test_success_rate_updated(self):
        """Test that success rate is tracked."""
        config = SequenceTrackerConfig(use_predefined_patterns=False, learning_rate=1.0)
        tracker = ToolSequenceTracker(config)

        tracker.record_execution("read_file")
        tracker.record_execution("edit_files", success=True)
        assert tracker._transitions["read"]["edit"].success_rate == 1.0

        tracker.record_execution("edit_files", success=False)
        # With learning_rate=1.0, it should be 0.0
        assert tracker._transitions["edit"]["edit"].success_rate == 0.0


class TestGetNextSuggestions:
    """Tests for getting next tool suggestions."""

    def test_no_suggestions_empty_history(self):
        """Test that empty history returns no suggestions."""
        tracker = ToolSequenceTracker()
        suggestions = tracker.get_next_suggestions()

        assert suggestions == []

    def test_suggestions_from_predefined(self):
        """Test suggestions from predefined patterns."""
        tracker = ToolSequenceTracker()
        tracker.record_execution("read_file")

        suggestions = tracker.get_next_suggestions(top_k=3)

        assert len(suggestions) > 0
        # edit should be suggested after read
        tool_names = [s[0] for s in suggestions]
        assert "edit" in tool_names

    def test_top_k_limit(self):
        """Test that top_k limits results."""
        tracker = ToolSequenceTracker()
        tracker.record_execution("read_file")

        suggestions = tracker.get_next_suggestions(top_k=2)

        assert len(suggestions) <= 2

    def test_exclude_tools(self):
        """Test that excluded tools are not suggested."""
        tracker = ToolSequenceTracker()
        tracker.record_execution("read_file")

        suggestions = tracker.get_next_suggestions(top_k=10, exclude_tools={"edit_files"})

        tool_names = [s[0] for s in suggestions]
        assert "edit" not in tool_names

    def test_confidence_ordering(self):
        """Test that suggestions are ordered by confidence."""
        tracker = ToolSequenceTracker()
        tracker.record_execution("read_file")

        suggestions = tracker.get_next_suggestions(top_k=5)

        if len(suggestions) >= 2:
            confidences = [s[1] for s in suggestions]
            # Should be in descending order
            assert confidences == sorted(confidences, reverse=True)

    def test_workflow_pattern_boost(self):
        """Test that multi-step workflow patterns boost suggestions."""
        tracker = ToolSequenceTracker()
        tracker.record_execution("read_file")
        tracker.record_execution("edit_files")

        suggestions = tracker.get_next_suggestions(top_k=5)

        # run_tests should be boosted due to workflow pattern
        tool_names = [s[0] for s in suggestions]
        assert "test" in tool_names

    def test_alias_inputs_normalize_to_canonical_suggestions(self):
        """Legacy aliases should feed canonical tracker state."""
        tracker = ToolSequenceTracker()
        tracker.record_execution("read_file")
        suggestions = tracker.get_next_suggestions(top_k=3)
        assert any(tool == "edit" for tool, _ in suggestions)


class TestApplyConfidenceBoost:
    """Tests for applying confidence boosts."""

    def test_no_boost_empty_history(self):
        """Test that empty history applies no boosts."""
        tracker = ToolSequenceTracker()
        scores = {"read_file": 0.5, "edit_files": 0.3}

        boosted = tracker.apply_confidence_boost(scores)

        assert boosted == scores

    def test_boost_applied(self):
        """Test that boosts are applied to likely next tools."""
        tracker = ToolSequenceTracker()
        tracker.record_execution("read_file")

        scores = {"edit_files": 0.5, "git": 0.3}
        boosted = tracker.apply_confidence_boost(scores)

        # edit_files should be boosted
        assert boosted["edit_files"] > scores["edit_files"]

    def test_boost_proportional(self):
        """Test that higher confidence tools get higher boosts."""
        tracker = ToolSequenceTracker()
        tracker.record_execution("read_file")

        scores = {"edit_files": 0.5, "code_search": 0.5}
        boosted = tracker.apply_confidence_boost(scores)

        # edit_files has higher sequence confidence than code_search
        # so it should get a bigger boost
        edit_boost = boosted["edit_files"] / scores["edit_files"]
        search_boost = boosted["code_search"] / scores["code_search"]
        assert edit_boost >= search_boost

    def test_max_boost_respected(self):
        """Test that max_boost limits the boost amount."""
        tracker = ToolSequenceTracker()
        tracker.record_execution("read_file")

        scores = {"edit_files": 1.0}
        boosted = tracker.apply_confidence_boost(scores, max_boost=0.10)

        # Boost should be at most 10% * boost_multiplier
        max_expected = 1.0 * (1.0 + 0.10 * tracker.config.boost_multiplier)
        assert boosted["edit_files"] <= max_expected


class TestGetWorkflowProgress:
    """Tests for workflow progress detection."""

    def test_no_progress_short_history(self):
        """Test that short history returns no workflow progress."""
        tracker = ToolSequenceTracker()
        tracker.record_execution("read_file")

        progress = tracker.get_workflow_progress()

        assert progress is None

    def test_file_editing_workflow_detected(self):
        """Test detection of file editing workflow."""
        tracker = ToolSequenceTracker()
        tracker.record_execution("read_file")
        tracker.record_execution("edit_files")
        tracker.record_execution("run_tests")

        progress = tracker.get_workflow_progress()

        assert progress is not None
        workflow_name, pct = progress
        assert workflow_name == "file_editing"
        assert pct >= 0.5  # At least 50% through

    def test_code_exploration_workflow_detected(self):
        """Test detection of code exploration workflow."""
        tracker = ToolSequenceTracker()
        tracker.record_execution("list_directory")
        tracker.record_execution("code_search")
        tracker.record_execution("read_file")

        progress = tracker.get_workflow_progress()

        assert progress is not None
        workflow_name, pct = progress
        assert workflow_name == "code_exploration"


class TestGetStatistics:
    """Tests for tracker statistics."""

    def test_empty_statistics(self):
        """Test statistics with empty history."""
        config = SequenceTrackerConfig(use_predefined_patterns=False)
        tracker = ToolSequenceTracker(config)

        stats = tracker.get_statistics()

        assert stats["history_length"] == 0
        assert stats["unique_tools_used"] == 0
        assert stats["total_transitions"] == 0

    def test_statistics_after_executions(self):
        """Test statistics after recording executions."""
        config = SequenceTrackerConfig(use_predefined_patterns=False)
        tracker = ToolSequenceTracker(config)

        tracker.record_execution("read_file")
        tracker.record_execution("edit_files")
        tracker.record_execution("read_file")

        stats = tracker.get_statistics()

        assert stats["history_length"] == 3
        assert stats["unique_tools_used"] == 2
        assert stats["total_transitions"] == 2


class TestClearAndReset:
    """Tests for clear and reset operations."""

    def test_clear_history(self):
        """Test clearing session history."""
        tracker = ToolSequenceTracker()
        tracker.record_execution("read_file")
        tracker.record_execution("edit_files")

        tracker.clear_history()

        assert len(tracker._history) == 0
        # Transitions should still exist
        assert len(tracker._transitions) > 0

    def test_reset_all(self):
        """Test full reset."""
        config = SequenceTrackerConfig(use_predefined_patterns=False)
        tracker = ToolSequenceTracker(config)
        tracker.record_execution("read_file")
        tracker.record_execution("edit_files")

        tracker.reset()

        assert len(tracker._history) == 0
        assert len(tracker._transitions) == 0

    def test_reset_reloads_predefined(self):
        """Test that reset reloads predefined patterns if configured."""
        tracker = ToolSequenceTracker()
        # Clear transitions
        tracker._transitions.clear()

        tracker.reset()

        # Should have predefined patterns again
        assert len(tracker._transitions) > 0


class TestFactoryFunction:
    """Tests for create_sequence_tracker factory."""

    def test_default_creation(self):
        """Test default factory creation."""
        tracker = create_sequence_tracker()

        assert tracker.config.use_predefined_patterns is True
        assert tracker.config.learning_rate == 0.3

    def test_custom_creation(self):
        """Test factory with custom options."""
        tracker = create_sequence_tracker(
            use_predefined=False,
            learning_rate=0.5,
        )

        assert tracker.config.use_predefined_patterns is False
        assert tracker.config.learning_rate == 0.5


class TestPredefinedPatterns:
    """Tests for predefined patterns data."""

    def test_common_sequences_not_empty(self):
        """Test that common sequences are defined."""
        assert len(COMMON_TOOL_SEQUENCES) > 0

    def test_workflow_patterns_not_empty(self):
        """Test that workflow patterns are defined."""
        assert len(WORKFLOW_PATTERNS) > 0

    def test_common_sequences_have_valid_weights(self):
        """Test that all weights are valid (0.0-1.0)."""
        for tool, transitions in COMMON_TOOL_SEQUENCES.items():
            for next_tool, weight in transitions:
                assert 0.0 <= weight <= 1.0, f"Invalid weight for {tool} -> {next_tool}"

    def test_workflow_patterns_have_valid_format(self):
        """Test that workflow patterns have correct format."""
        for pattern, suggested, weight in WORKFLOW_PATTERNS:
            assert isinstance(pattern, list)
            assert len(pattern) >= 2
            assert isinstance(suggested, str)
            assert 0.0 <= weight <= 1.0


class TestIntegration:
    """Integration tests for the complete workflow."""

    def test_full_editing_workflow(self):
        """Test a complete file editing workflow."""
        tracker = ToolSequenceTracker()

        # Simulate a typical editing workflow
        tracker.record_execution("code_search")
        suggestions = tracker.get_next_suggestions(top_k=3)
        assert any(s[0] == "read" for s in suggestions)

        tracker.record_execution("read_file")
        suggestions = tracker.get_next_suggestions(top_k=3)
        assert any(s[0] == "edit" for s in suggestions)

        tracker.record_execution("edit_files")
        suggestions = tracker.get_next_suggestions(top_k=3)
        assert any(s[0] == "test" for s in suggestions)

        tracker.record_execution("run_tests")
        suggestions = tracker.get_next_suggestions(top_k=3)
        # git or edit_files should be suggested
        tool_names = [s[0] for s in suggestions]
        assert "git" in tool_names or "edit_files" in tool_names

    def test_boost_improves_selection(self):
        """Test that boosts improve tool selection for likely tools."""
        tracker = ToolSequenceTracker()
        tracker.record_execution("read_file")

        # Start with equal scores
        scores = {
            "edit_files": 0.5,
            "web_search": 0.5,
            "docker": 0.5,
        }

        boosted = tracker.apply_confidence_boost(scores)

        # edit_files should now rank higher due to sequence boost
        assert boosted["edit_files"] > boosted["web_search"]
        assert boosted["edit_files"] > boosted["docker"]
