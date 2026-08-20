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

"""Tests for SemanticToolSelector sequence tracking integration."""

from unittest.mock import MagicMock
from pathlib import Path
import tempfile

from victor.tools.semantic_selector import SemanticToolSelector


class TestSequenceTrackingInit:
    """Tests for sequence tracking initialization."""

    def test_default_sequence_tracking_enabled(self):
        """Test that sequence tracking is enabled by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            selector = SemanticToolSelector(cache_dir=Path(tmpdir))

            assert selector._sequence_tracking is True
            assert selector._sequence_tracker is not None

    def test_sequence_tracking_disabled(self):
        """Test that sequence tracking can be disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            selector = SemanticToolSelector(cache_dir=Path(tmpdir), sequence_tracking=False)

            assert selector._sequence_tracking is False
            assert selector._sequence_tracker is None

    def test_sequence_tracker_type(self):
        """Test that the sequence tracker is the correct type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            selector = SemanticToolSelector(cache_dir=Path(tmpdir))

            from victor.agent.tool_sequence_tracker import ToolSequenceTracker

            assert isinstance(selector._sequence_tracker, ToolSequenceTracker)


class TestRecordToolUsage:
    """Tests for tool usage recording with sequence tracking."""

    def test_record_tool_usage_updates_sequence_tracker(self):
        """Test that recording tool execution updates the sequence tracker."""
        with tempfile.TemporaryDirectory() as tmpdir:
            selector = SemanticToolSelector(cache_dir=Path(tmpdir))

            # Record a tool execution
            selector.record_tool_execution("read_file", success=True)

            # Check that the sequence tracker was updated
            assert len(selector._sequence_tracker._history) == 1
            assert selector._sequence_tracker._history[0] == "read"

    def test_record_tool_usage_records_success(self):
        """Test that success/failure is recorded correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            selector = SemanticToolSelector(cache_dir=Path(tmpdir))

            # Record successful and failed executions
            selector.record_tool_execution("read_file", success=True)
            selector.record_tool_execution("edit_files", success=False)

            # Check history updated
            assert len(selector._sequence_tracker._history) == 2

    def test_record_tool_usage_disabled_tracking(self):
        """Test that recording works when sequence tracking is disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            selector = SemanticToolSelector(cache_dir=Path(tmpdir), sequence_tracking=False)

            # Should not raise even though tracker is None
            selector._record_tool_usage("read_file", "test query", success=True)

            # Verify it didn't crash and usage cache was still updated
            assert "read_file" in selector._tool_usage_cache


class TestGetSequenceBoost:
    """Tests for sequence boost calculation."""

    def test_sequence_boost_after_read_file(self):
        """Test that edit_files gets boosted after read_file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            selector = SemanticToolSelector(cache_dir=Path(tmpdir))

            # Record read_file execution
            selector.record_tool_execution("read_file", success=True)

            # edit_files should get a boost
            boost = selector._get_sequence_boost("edit_files")
            assert boost > 0.0
            assert boost <= 0.15  # Max boost is 0.15

    def test_no_boost_for_unrelated_tool(self):
        """Test that unrelated tools don't get boosted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            selector = SemanticToolSelector(cache_dir=Path(tmpdir))

            # Record read_file execution
            selector.record_tool_execution("read_file", success=True)

            # docker should not get a boost (not a common follow-up to read_file)
            boost = selector._get_sequence_boost("docker")
            assert boost == 0.0

    def test_no_boost_with_empty_history(self):
        """Test that no boost is applied with empty history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            selector = SemanticToolSelector(cache_dir=Path(tmpdir))

            # No history recorded
            boost = selector._get_sequence_boost("edit_files")
            assert boost == 0.0

    def test_no_boost_when_disabled(self):
        """Test that no boost is applied when tracking is disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            selector = SemanticToolSelector(cache_dir=Path(tmpdir), sequence_tracking=False)

            boost = selector._get_sequence_boost("edit_files")
            assert boost == 0.0


class TestApplySequenceBoosts:
    """Tests for applying sequence boosts to similarity scores."""

    def test_apply_sequence_boosts_increases_scores(self):
        """Test that sequence boosts increase relevant tool scores."""
        with tempfile.TemporaryDirectory() as tmpdir:
            selector = SemanticToolSelector(cache_dir=Path(tmpdir))

            # Record read_file execution
            selector.record_tool_execution("read_file", success=True)

            # Create mock tools with scores
            tool1 = MagicMock()
            tool1.name = "edit_files"
            tool2 = MagicMock()
            tool2.name = "docker"

            similarities = [(tool1, 0.5), (tool2, 0.5)]

            # Apply boosts
            boosted = selector._apply_sequence_boosts(similarities)

            # edit_files should have higher score now
            edit_score = next(score for t, score in boosted if t.name == "edit_files")
            docker_score = next(score for t, score in boosted if t.name == "docker")

            assert edit_score > 0.5  # Should be boosted
            assert docker_score == 0.5  # Should stay same

    def test_apply_sequence_boosts_when_disabled(self):
        """Test that no boosts are applied when tracking is disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            selector = SemanticToolSelector(cache_dir=Path(tmpdir), sequence_tracking=False)

            tool1 = MagicMock()
            tool1.name = "edit_files"

            similarities = [(tool1, 0.5)]

            # Apply boosts
            boosted = selector._apply_sequence_boosts(similarities)

            # Score should remain unchanged
            assert boosted[0][1] == 0.5


class TestGetNextToolSuggestions:
    """Tests for getting next tool suggestions."""

    def test_get_next_tool_suggestions_after_read(self):
        """Test getting suggestions after read_file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            selector = SemanticToolSelector(cache_dir=Path(tmpdir))

            # Record read_file
            selector.record_tool_execution("read_file", success=True)

            suggestions = selector.get_next_tool_suggestions(top_k=3)

            assert len(suggestions) > 0
            tool_names = [name for name, _ in suggestions]
            assert "edit_files" in tool_names  # Common follow-up to read_file

    def test_get_next_tool_suggestions_empty_history(self):
        """Test getting suggestions with no history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            selector = SemanticToolSelector(cache_dir=Path(tmpdir))

            suggestions = selector.get_next_tool_suggestions()
            assert suggestions == []

    def test_get_next_tool_suggestions_disabled(self):
        """Test getting suggestions when tracking is disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            selector = SemanticToolSelector(cache_dir=Path(tmpdir), sequence_tracking=False)

            suggestions = selector.get_next_tool_suggestions()
            assert suggestions == []


class TestGetCurrentWorkflow:
    """Tests for workflow detection."""

    def test_detect_file_editing_workflow(self):
        """Test detection of file editing workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            selector = SemanticToolSelector(cache_dir=Path(tmpdir))

            # Simulate file editing workflow
            selector.record_tool_execution("read_file", success=True)
            selector.record_tool_execution("edit_files", success=True)
            selector.record_tool_execution("run_tests", success=True)

            workflow = selector.get_current_workflow()

            assert workflow is not None
            workflow_name, progress = workflow
            assert workflow_name == "file_editing"
            assert progress >= 0.5

    def test_no_workflow_short_history(self):
        """Test that no workflow is detected with short history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            selector = SemanticToolSelector(cache_dir=Path(tmpdir))

            selector.record_tool_execution("read_file", success=True)

            workflow = selector.get_current_workflow()
            assert workflow is None

    def test_no_workflow_when_disabled(self):
        """Test that no workflow is detected when tracking is disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            selector = SemanticToolSelector(cache_dir=Path(tmpdir), sequence_tracking=False)

            workflow = selector.get_current_workflow()
            assert workflow is None


class TestClearSessionState:
    """Tests for clearing session state."""

    def test_clear_session_state_clears_history(self):
        """Test that clearing session state clears sequence history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            selector = SemanticToolSelector(cache_dir=Path(tmpdir))

            # Record some executions
            selector.record_tool_execution("read_file", success=True)
            selector.record_tool_execution("edit_files", success=True)

            assert len(selector._sequence_tracker._history) == 2

            # Clear session
            selector.clear_session_state()

            assert len(selector._sequence_tracker._history) == 0

    def test_clear_session_state_clears_warnings(self):
        """Test that clearing session state clears warnings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            selector = SemanticToolSelector(cache_dir=Path(tmpdir))

            selector._warned_tools.add("test_tool")
            selector._last_cost_warnings = ["warning1"]

            selector.clear_session_state()

            assert len(selector._warned_tools) == 0
            assert len(selector._last_cost_warnings) == 0


class TestClassificationToolStats:
    """Tests for classification tool stats with sequence tracking."""

    def test_stats_include_sequence_tracking(self):
        """Test that stats include sequence tracking info."""
        with tempfile.TemporaryDirectory() as tmpdir:
            selector = SemanticToolSelector(cache_dir=Path(tmpdir))

            # Record some executions
            selector.record_tool_execution("read_file", success=True)
            selector.record_tool_execution("edit_files", success=True)

            stats = selector.get_classification_tool_stats()

            assert "sequence_tracking" in stats
            assert stats["sequence_tracking"]["enabled"] is True
            assert stats["sequence_tracking"]["history_length"] == 2
            assert stats["sequence_tracking"]["unique_tools_used"] == 2

    def test_stats_when_tracking_disabled(self):
        """Test stats when sequence tracking is disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            selector = SemanticToolSelector(cache_dir=Path(tmpdir), sequence_tracking=False)

            stats = selector.get_classification_tool_stats()

            assert stats["sequence_tracking"]["enabled"] is False


class TestWorkflowPatternBoost:
    """Tests for multi-step workflow pattern boosting."""

    def test_run_tests_boosted_after_edit_workflow(self):
        """Test that run_tests is boosted in read → edit workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            selector = SemanticToolSelector(cache_dir=Path(tmpdir))

            # Simulate read → edit workflow
            selector.record_tool_execution("read_file", success=True)
            selector.record_tool_execution("edit_files", success=True)

            # run_tests should now be suggested with high confidence
            suggestions = selector.get_next_tool_suggestions(top_k=5)
            tool_names = [name for name, _ in suggestions]

            assert "run_tests" in tool_names

    def test_git_boosted_after_test_workflow(self):
        """Test that git is boosted after edit → test workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            selector = SemanticToolSelector(cache_dir=Path(tmpdir))

            # Simulate edit → test workflow
            selector.record_tool_execution("edit_files", success=True)
            selector.record_tool_execution("run_tests", success=True)

            # git should now be suggested
            suggestions = selector.get_next_tool_suggestions(top_k=5)
            tool_names = [name for name, _ in suggestions]

            assert "git" in tool_names
