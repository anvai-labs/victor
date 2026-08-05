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

"""Tests for SynthesisCheckpoint.

Covers GAP Phase 3: Synthesis checkpoint detection.
"""

import pytest

from victor.agent.synthesis_checkpoint import (
    CheckpointResult,
    SynthesisCheckpoint,
    ToolCountCheckpoint,
    DuplicateToolCheckpoint,
    SimilarArgsCheckpoint,
    TimeoutApproachingCheckpoint,
    NoProgressCheckpoint,
    ErrorRateCheckpoint,
    UnifiedTaskTrackerCheckpoint,  # Phase 3: New checkpoint
    CompositeSynthesisCheckpoint,
    create_default_checkpoint,
    create_aggressive_checkpoint,
    create_relaxed_checkpoint,
    get_checkpoint_for_complexity,
)


class TestCheckpointResult:
    """Tests for CheckpointResult dataclass."""

    def test_basic_result(self):
        """Test basic CheckpointResult creation."""
        result = CheckpointResult(should_synthesize=True, reason="Test reason")
        assert result.should_synthesize is True
        assert result.reason == "Test reason"
        assert result.suggested_prompt is None
        assert result.priority == 0

    def test_result_with_all_fields(self):
        """Test CheckpointResult with all fields."""
        result = CheckpointResult(
            should_synthesize=True,
            reason="Limit reached",
            suggested_prompt="Please synthesize",
            priority=5,
            metadata={"count": 10},
        )
        assert result.should_synthesize is True
        assert result.priority == 5
        assert result.metadata["count"] == 10


class TestToolCountCheckpoint:
    """Tests for ToolCountCheckpoint."""

    @pytest.fixture
    def checkpoint(self):
        return ToolCountCheckpoint(max_calls=5)

    def test_under_limit(self, checkpoint):
        """Test checkpoint doesn't trigger under limit."""
        history = [{"tool": f"tool_{i}"} for i in range(3)]
        result = checkpoint.check(history, {})

        assert result.should_synthesize is False
        assert "3/5" in result.reason

    def test_at_limit(self, checkpoint):
        """Test checkpoint triggers at limit."""
        history = [{"tool": f"tool_{i}"} for i in range(5)]
        result = checkpoint.check(history, {})

        assert result.should_synthesize is True
        assert "5 tool calls" in result.reason
        assert result.suggested_prompt is not None
        assert result.priority == 5

    def test_over_limit(self, checkpoint):
        """Test checkpoint triggers over limit."""
        history = [{"tool": f"tool_{i}"} for i in range(10)]
        result = checkpoint.check(history, {})

        assert result.should_synthesize is True

    def test_empty_history(self, checkpoint):
        """Test with empty history."""
        result = checkpoint.check([], {})
        assert result.should_synthesize is False


class TestDuplicateToolCheckpoint:
    """Tests for DuplicateToolCheckpoint."""

    @pytest.fixture
    def checkpoint(self):
        return DuplicateToolCheckpoint(threshold=3)

    def test_no_duplicates(self, checkpoint):
        """Test no duplicates doesn't trigger."""
        history = [
            {"tool": "read"},
            {"tool": "grep"},
            {"tool": "ls"},
        ]
        result = checkpoint.check(history, {})
        assert result.should_synthesize is False

    def test_duplicates_trigger(self, checkpoint):
        """Test consecutive duplicates trigger synthesis."""
        history = [
            {"tool": "read"},
            {"tool": "read"},
            {"tool": "read"},
        ]
        result = checkpoint.check(history, {})

        assert result.should_synthesize is True
        assert "read" in result.reason
        assert result.priority == 8

    def test_non_consecutive_duplicates(self, checkpoint):
        """Test non-consecutive duplicates don't trigger."""
        history = [
            {"tool": "read"},
            {"tool": "grep"},
            {"tool": "read"},
        ]
        result = checkpoint.check(history, {})
        assert result.should_synthesize is False

    def test_too_few_calls(self, checkpoint):
        """Test with too few calls."""
        history = [{"tool": "read"}]
        result = checkpoint.check(history, {})
        assert result.should_synthesize is False


class TestSimilarArgsCheckpoint:
    """Tests for SimilarArgsCheckpoint."""

    @pytest.fixture
    def checkpoint(self):
        return SimilarArgsCheckpoint(window_size=5, similarity_threshold=0.7)

    def test_no_similar_args(self, checkpoint):
        """Test different paths don't trigger."""
        history = [
            {"tool": "read", "args": {"path": "file1.py"}},
            {"tool": "read", "args": {"path": "file2.py"}},
            {"tool": "read", "args": {"path": "file3.py"}},
        ]
        result = checkpoint.check(history, {})
        assert result.should_synthesize is False

    def test_repeated_paths_trigger(self, checkpoint):
        """Test repeated paths trigger synthesis."""
        history = [
            {"tool": "read", "args": {"path": "same.py"}},
            {"tool": "read", "args": {"path": "same.py"}},
            {"tool": "read", "args": {"path": "same.py"}},
            {"tool": "read", "args": {"path": "same.py"}},
        ]
        result = checkpoint.check(history, {})

        assert result.should_synthesize is True
        assert "similar paths" in result.reason.lower() or "same" in result.reason.lower()

    def test_repeated_queries_trigger(self, checkpoint):
        """Test repeated queries trigger synthesis."""
        history = [
            {"tool": "grep", "args": {"query": "same query"}},
            {"tool": "grep", "args": {"query": "same query"}},
            {"tool": "grep", "args": {"query": "same query"}},
            {"tool": "grep", "args": {"query": "same query"}},
        ]
        result = checkpoint.check(history, {})

        assert result.should_synthesize is True

    def test_too_few_calls(self, checkpoint):
        """Test with too few calls."""
        history = [{"tool": "read", "args": {"path": "a.py"}}]
        result = checkpoint.check(history, {})
        assert result.should_synthesize is False


class TestTimeoutApproachingCheckpoint:
    """Tests for TimeoutApproachingCheckpoint."""

    @pytest.fixture
    def checkpoint(self):
        return TimeoutApproachingCheckpoint(warning_threshold=0.7, critical_threshold=0.9)

    def test_plenty_of_time(self, checkpoint):
        """Test with plenty of time remaining."""
        context = {"elapsed_time": 30, "timeout": 180}
        result = checkpoint.check([], context)

        assert result.should_synthesize is False
        assert "remaining" in result.reason.lower()

    def test_warning_threshold(self, checkpoint):
        """Test warning threshold triggers."""
        context = {"elapsed_time": 140, "timeout": 180}  # ~78%
        result = checkpoint.check([], context)

        assert result.should_synthesize is True
        assert result.priority == 7

    def test_critical_threshold(self, checkpoint):
        """Test critical threshold triggers with higher priority."""
        context = {"elapsed_time": 170, "timeout": 180}  # ~94%
        result = checkpoint.check([], context)

        assert result.should_synthesize is True
        assert result.priority == 10
        assert "critical" in result.reason.lower()

    def test_no_timeout(self, checkpoint):
        """Test with no timeout configured."""
        context = {"elapsed_time": 100, "timeout": 0}
        result = checkpoint.check([], context)
        assert result.should_synthesize is False


class TestNoProgressCheckpoint:
    """Tests for NoProgressCheckpoint."""

    @pytest.fixture
    def checkpoint(self):
        return NoProgressCheckpoint(window_size=4)

    def test_normal_progress(self, checkpoint):
        """Test normal progress doesn't trigger."""
        history = [
            {"tool": "read", "success": True, "result": "content"},
            {"tool": "grep", "success": True, "result": "matches"},
            {"tool": "ls", "success": True, "result": "files"},
            {"tool": "read", "success": True, "result": "more content"},
        ]
        result = checkpoint.check(history, {})
        assert result.should_synthesize is False

    def test_many_failures_trigger(self, checkpoint):
        """Test many failures trigger synthesis."""
        history = [
            {"tool": "read", "success": False, "error": "not found"},
            {"tool": "grep", "success": False, "error": "pattern error"},
            {"tool": "ls", "success": False, "error": "permission denied"},
            {"tool": "read", "success": True, "result": "ok"},
        ]
        result = checkpoint.check(history, {})

        assert result.should_synthesize is True
        assert "failed" in result.reason.lower()

    def test_empty_results_trigger(self, checkpoint):
        """Test empty results trigger synthesis."""
        history = [
            {"tool": "grep", "success": True, "result": "[]"},
            {"tool": "grep", "success": True, "result": "{}"},
            {"tool": "grep", "success": True, "result": ""},
            {"tool": "grep", "success": True, "result": "null"},
        ]
        result = checkpoint.check(history, {})

        assert result.should_synthesize is True
        assert "empty" in result.reason.lower()

    def test_too_few_calls(self, checkpoint):
        """Test with too few calls."""
        history = [{"tool": "read", "success": False}]
        result = checkpoint.check(history, {})
        assert result.should_synthesize is False


class TestErrorRateCheckpoint:
    """Tests for ErrorRateCheckpoint."""

    @pytest.fixture
    def checkpoint(self):
        return ErrorRateCheckpoint(error_threshold=0.5, min_calls=4)

    def test_low_error_rate(self, checkpoint):
        """Test low error rate doesn't trigger."""
        history = [
            {"tool": "a", "success": True},
            {"tool": "b", "success": True},
            {"tool": "c", "success": True},
            {"tool": "d", "success": False},
        ]
        result = checkpoint.check(history, {})
        assert result.should_synthesize is False

    def test_high_error_rate_triggers(self, checkpoint):
        """Test high error rate triggers synthesis."""
        history = [
            {"tool": "a", "success": False},
            {"tool": "b", "success": False},
            {"tool": "c", "success": False},
            {"tool": "d", "success": True},
        ]
        result = checkpoint.check(history, {})

        assert result.should_synthesize is True
        assert "75%" in result.reason or "error rate" in result.reason.lower()
        assert result.priority == 9

    def test_too_few_calls(self, checkpoint):
        """Test with too few calls."""
        history = [{"tool": "a", "success": False}]
        result = checkpoint.check(history, {})
        assert result.should_synthesize is False


class TestCompositeSynthesisCheckpoint:
    """Tests for CompositeSynthesisCheckpoint."""

    def test_no_checkpoints(self):
        """Test composite with no checkpoints."""
        composite = CompositeSynthesisCheckpoint([])
        result = composite.check([], {})
        assert result.should_synthesize is False

    def test_single_checkpoint_triggers(self):
        """Test composite with single triggering checkpoint."""
        composite = CompositeSynthesisCheckpoint([ToolCountCheckpoint(max_calls=2)])

        history = [{"tool": "a"}, {"tool": "b"}]
        result = composite.check(history, {})

        assert result.should_synthesize is True
        assert result.metadata.get("checkpoint") == "tool_count"

    def test_priority_ordering(self):
        """Test higher priority checkpoint result is returned."""
        composite = CompositeSynthesisCheckpoint(
            [
                ToolCountCheckpoint(max_calls=2),  # priority 5
                ErrorRateCheckpoint(error_threshold=0.3, min_calls=2),  # priority 9
            ]
        )

        history = [
            {"tool": "a", "success": False},
            {"tool": "b", "success": False},
        ]
        result = composite.check(history, {})

        assert result.should_synthesize is True
        assert result.priority == 9  # Error rate has higher priority

    def test_multiple_triggers_tracked(self):
        """Test multiple triggers are tracked in metadata."""
        composite = CompositeSynthesisCheckpoint(
            [
                ToolCountCheckpoint(max_calls=2),
                DuplicateToolCheckpoint(threshold=2),
            ]
        )

        history = [{"tool": "read"}, {"tool": "read"}]
        result = composite.check(history, {})

        assert result.should_synthesize is True
        # Should mention other triggers
        if "other_triggers" in result.metadata:
            assert len(result.metadata["other_triggers"]) >= 0

    def test_add_checkpoint_fluent(self):
        """Test fluent add_checkpoint method."""
        composite = (
            CompositeSynthesisCheckpoint()
            .add_checkpoint(ToolCountCheckpoint(5))
            .add_checkpoint(DuplicateToolCheckpoint(3))
        )

        assert len(composite._checkpoints) == 2


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_default_checkpoint(self):
        """Test default checkpoint creation.

        Phase 3: Checkpoint count reduced from 6 to 4 (replaced 3 primitive checkpoints
        with 1 UnifiedTaskTrackerCheckpoint).
        """
        checkpoint = create_default_checkpoint()
        assert isinstance(checkpoint, CompositeSynthesisCheckpoint)
        assert len(checkpoint._checkpoints) == 4  # Phase 3: Was 6, now 4

    def test_create_aggressive_checkpoint(self):
        """Test aggressive checkpoint has lower thresholds."""
        aggressive = create_aggressive_checkpoint()
        default = create_default_checkpoint()

        # Aggressive should have lower tool count threshold
        aggressive_count = next(
            c for c in aggressive._checkpoints if isinstance(c, ToolCountCheckpoint)
        )
        default_count = next(c for c in default._checkpoints if isinstance(c, ToolCountCheckpoint))

        assert aggressive_count.max_calls < default_count.max_calls

    def test_create_relaxed_checkpoint(self):
        """Test relaxed checkpoint has higher thresholds."""
        relaxed = create_relaxed_checkpoint()
        default = create_default_checkpoint()

        relaxed_count = next(c for c in relaxed._checkpoints if isinstance(c, ToolCountCheckpoint))
        default_count = next(c for c in default._checkpoints if isinstance(c, ToolCountCheckpoint))

        assert relaxed_count.max_calls > default_count.max_calls

    def test_get_checkpoint_for_complexity_simple(self):
        """Test simple complexity gets aggressive checkpoint."""
        checkpoint = get_checkpoint_for_complexity("simple")
        tool_count = next(c for c in checkpoint._checkpoints if isinstance(c, ToolCountCheckpoint))
        assert tool_count.max_calls == 5  # Aggressive setting

    def test_get_checkpoint_for_complexity_complex(self):
        """Test complex complexity gets relaxed checkpoint."""
        checkpoint = get_checkpoint_for_complexity("complex")
        tool_count = next(c for c in checkpoint._checkpoints if isinstance(c, ToolCountCheckpoint))
        assert tool_count.max_calls == 20  # Relaxed setting

    def test_get_checkpoint_for_complexity_medium(self):
        """Test medium complexity gets default checkpoint."""
        checkpoint = get_checkpoint_for_complexity("medium")
        tool_count = next(c for c in checkpoint._checkpoints if isinstance(c, ToolCountCheckpoint))
        assert tool_count.max_calls == 12  # Default setting


class TestEdgeCases:
    """Tests for edge cases."""

    def test_checkpoint_handles_missing_fields(self):
        """Test checkpoints handle missing history fields gracefully."""
        checkpoint = create_default_checkpoint()
        history = [
            {},  # Empty dict
            {"tool": "read"},  # Missing other fields
            {"tool": "grep", "args": None},  # None args
        ]
        # Should not raise
        result = checkpoint.check(history, {})
        assert isinstance(result, CheckpointResult)

    def test_composite_handles_checkpoint_error(self):
        """Test composite handles individual checkpoint errors."""

        class FaultyCheckpoint(SynthesisCheckpoint):
            @property
            def name(self):
                return "faulty"

            def check(self, history, context):
                raise ValueError("Checkpoint error")

        composite = CompositeSynthesisCheckpoint(
            [FaultyCheckpoint(), ToolCountCheckpoint(max_calls=2)]
        )

        history = [{"tool": "a"}, {"tool": "b"}]
        # Should not raise, should use working checkpoint
        result = composite.check(history, {})
        assert result.should_synthesize is True

    def test_empty_tool_name(self):
        """Test handling of empty tool names."""
        checkpoint = DuplicateToolCheckpoint(threshold=2)
        history = [{"tool": ""}, {"tool": ""}]
        result = checkpoint.check(history, {})
        # Empty tool names shouldn't falsely trigger
        assert result.should_synthesize is False


class TestUnifiedTaskTrackerCheckpoint:
    """Tests for UnifiedTaskTrackerCheckpoint (Phase 3 enhancement)."""

    def test_checkpoint_without_tracker(self):
        """Test checkpoint returns no-synthesis when UnifiedTaskTracker not available."""
        from unittest.mock import MagicMock

        checkpoint = UnifiedTaskTrackerCheckpoint()
        history = [{"tool": "read"}, {"tool": "grep"}]
        context = {}  # No unified_task_tracker in context

        result = checkpoint.check(history, context)

        assert result.should_synthesize is False
        assert "not available" in result.reason

    def test_checkpoint_with_no_loop_warning(self):
        """Test checkpoint when UnifiedTaskTracker detects no loop."""
        from unittest.mock import MagicMock

        mock_tracker = MagicMock()
        mock_tracker.check_loop_warning.return_value = None  # No loop

        checkpoint = UnifiedTaskTrackerCheckpoint(tracker=mock_tracker)
        history = [{"tool": "read"}, {"tool": "grep"}]
        context = {}

        result = checkpoint.check(history, context)

        assert result.should_synthesize is False
        assert "No loop detected" in result.reason
        mock_tracker.check_loop_warning.assert_called_once()

    def test_checkpoint_with_loop_warning(self):
        """Test checkpoint triggers synthesis when UnifiedTaskTracker detects loop."""
        from unittest.mock import MagicMock

        mock_tracker = MagicMock()
        mock_tracker.check_loop_warning.return_value = "Approaching loop (2/3): read|file.py|0|100"

        checkpoint = UnifiedTaskTrackerCheckpoint(tracker=mock_tracker)
        history = [{"tool": "read"}, {"tool": "grep"}]
        context = {}

        result = checkpoint.check(history, context)

        assert result.should_synthesize is True
        assert "UnifiedTaskTracker loop detection" in result.reason
        assert "Approaching loop" in result.reason
        assert result.priority == 9  # High priority for loops
        assert result.suggested_prompt is not None
        assert "synthesize" in result.suggested_prompt.lower()

    def test_checkpoint_gets_tracker_from_context(self):
        """Test checkpoint can get UnifiedTaskTracker from task_context."""
        from unittest.mock import MagicMock

        mock_tracker = MagicMock()
        mock_tracker.check_loop_warning.return_value = None

        checkpoint = UnifiedTaskTrackerCheckpoint()  # No tracker passed to __init__
        history = [{"tool": "read"}]
        context = {"unified_task_tracker": mock_tracker}

        result = checkpoint.check(history, context)

        assert result.should_synthesize is False
        mock_tracker.check_loop_warning.assert_called_once()

    def test_checkpoint_handles_exception_gracefully(self):
        """Test checkpoint handles UnifiedTaskTracker exceptions gracefully."""
        from unittest.mock import MagicMock

        mock_tracker = MagicMock()
        mock_tracker.check_loop_warning.side_effect = Exception("Tracker error")

        checkpoint = UnifiedTaskTrackerCheckpoint(tracker=mock_tracker)
        history = [{"tool": "read"}]
        context = {}

        result = checkpoint.check(history, context)

        # Should not raise, should return error result
        assert result.should_synthesize is False
        assert "error" in result.reason.lower()

    def test_checkpoint_name(self):
        """Test checkpoint has correct name."""
        checkpoint = UnifiedTaskTrackerCheckpoint()
        assert checkpoint.name == "unified_task_tracker"


class TestPhase3CheckpointFactoryUpdates:
    """Tests for Phase 3 updates to factory functions.

    Verifies that create_default_checkpoint, create_aggressive_checkpoint, and
    create_relaxed_checkpoint now use UnifiedTaskTrackerCheckpoint instead of
    primitive heuristic checkpoints (DuplicateTool, SimilarArgs, NoProgress).
    """

    def test_default_checkpoint_uses_unified_tracker(self):
        """Test default checkpoint includes UnifiedTaskTrackerCheckpoint."""
        checkpoint = create_default_checkpoint()

        # Should include UnifiedTaskTrackerCheckpoint
        has_unified = any(
            isinstance(c, UnifiedTaskTrackerCheckpoint) for c in checkpoint._checkpoints
        )
        assert has_unified, "Default checkpoint should include UnifiedTaskTrackerCheckpoint"

        # Should NOT include primitive checkpoints
        has_duplicate = any(isinstance(c, DuplicateToolCheckpoint) for c in checkpoint._checkpoints)
        has_similar = any(isinstance(c, SimilarArgsCheckpoint) for c in checkpoint._checkpoints)
        has_no_progress = any(isinstance(c, NoProgressCheckpoint) for c in checkpoint._checkpoints)

        assert (
            not has_duplicate
        ), "Should not include DuplicateToolCheckpoint (replaced by UnifiedTaskTracker)"
        assert (
            not has_similar
        ), "Should not include SimilarArgsCheckpoint (replaced by UnifiedTaskTracker)"
        assert (
            not has_no_progress
        ), "Should not include NoProgressCheckpoint (replaced by UnifiedTaskTracker)"

        # Should still include unique-value checkpoints
        has_tool_count = any(isinstance(c, ToolCountCheckpoint) for c in checkpoint._checkpoints)
        has_timeout = any(
            isinstance(c, TimeoutApproachingCheckpoint) for c in checkpoint._checkpoints
        )
        has_error_rate = any(isinstance(c, ErrorRateCheckpoint) for c in checkpoint._checkpoints)

        assert has_tool_count, "Should include ToolCountCheckpoint (unique value)"
        assert has_timeout, "Should include TimeoutApproachingCheckpoint (unique value)"
        assert has_error_rate, "Should include ErrorRateCheckpoint (unique value)"

    def test_aggressive_checkpoint_uses_unified_tracker(self):
        """Test aggressive checkpoint includes UnifiedTaskTrackerCheckpoint."""
        checkpoint = create_aggressive_checkpoint()

        # Should include UnifiedTaskTrackerCheckpoint
        has_unified = any(
            isinstance(c, UnifiedTaskTrackerCheckpoint) for c in checkpoint._checkpoints
        )
        assert has_unified

        # Should NOT include primitive checkpoints
        has_duplicate = any(isinstance(c, DuplicateToolCheckpoint) for c in checkpoint._checkpoints)
        assert not has_duplicate

    def test_relaxed_checkpoint_uses_unified_tracker(self):
        """Test relaxed checkpoint includes UnifiedTaskTrackerCheckpoint."""
        checkpoint = create_relaxed_checkpoint()

        # Should include UnifiedTaskTrackerCheckpoint
        has_unified = any(
            isinstance(c, UnifiedTaskTrackerCheckpoint) for c in checkpoint._checkpoints
        )
        assert has_unified

        # Should NOT include primitive checkpoints
        has_duplicate = any(isinstance(c, DuplicateToolCheckpoint) for c in checkpoint._checkpoints)
        assert not has_duplicate

    def test_checkpoint_count_reduced(self):
        """Test that replacing 3 primitive checkpoints with 1 UnifiedTaskTracker reduces count."""
        default = create_default_checkpoint()

        # Phase 3: 4 checkpoints (was 6 before Phase 3)
        # Removed: DuplicateTool, SimilarArgs, NoProgress (3 checkpoints)
        # Added: UnifiedTaskTrackerCheckpoint (1 checkpoint)
        # Kept: ToolCount, TimeoutApproaching, ErrorRate (3 checkpoints)
        # Total: 4 checkpoints
        assert (
            len(default._checkpoints) == 4
        ), f"Expected 4 checkpoints, got {len(default._checkpoints)}"
