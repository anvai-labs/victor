"""Tests for ToolReputationTracker — online mid-turn credit feedback."""

from victor.framework.rl.tool_reputation import ToolReputationTracker


class TestToolReputationTracker:
    """Core EMA reputation tracking."""

    def test_initial_state(self):
        tracker = ToolReputationTracker()
        assert tracker.tracked_tools == 0
        assert tracker.get_reputation("read") == 0.0

    def test_successful_tool_increases_reputation(self):
        tracker = ToolReputationTracker(alpha=0.5)
        score = tracker.record("read", success=True, duration_ms=50.0)
        assert score > 0

    def test_failed_tool_decreases_reputation(self):
        tracker = ToolReputationTracker(alpha=0.5)
        score = tracker.record("shell", success=False, duration_ms=100.0)
        assert score < 0

    def test_aliases_merge_into_canonical_tool_reputation(self):
        tracker = ToolReputationTracker(alpha=0.5)
        tracker.record("read_file", success=True, duration_ms=50.0)
        score = tracker.record("read", success=True, duration_ms=60.0)

        assert score > 0
        assert tracker.get_reputation("read") == tracker.get_reputation("read_file")
        assert tracker.tracked_tools == 1

    def test_ema_smoothing(self):
        """Reputation converges via exponential moving average."""
        tracker = ToolReputationTracker(alpha=0.3)

        # Several successes build up reputation
        for _ in range(5):
            tracker.record("read", True, 50.0)

        high_rep = tracker.get_reputation("read")

        # One failure shouldn't tank it completely
        tracker.record("read", False, 100.0)
        after_fail = tracker.get_reputation("read")

        assert after_fail < high_rep
        assert after_fail > 0  # Still positive overall

    def test_slow_success_gets_penalty(self):
        """Very slow tools get slightly lower reward even on success."""
        tracker = ToolReputationTracker(alpha=1.0)  # No smoothing

        fast = tracker.record("fast_tool", True, 100.0)
        tracker.reset_tool("fast_tool")

        slow = tracker.record("slow_tool", True, 15000.0)

        assert fast > slow

    def test_tracked_tools_count(self):
        tracker = ToolReputationTracker()
        tracker.record("tool_a", True, 50.0)
        tracker.record("tool_b", False, 50.0)
        tracker.record("tool_c", True, 50.0)
        assert tracker.tracked_tools == 3

    def test_reset_clears_all(self):
        tracker = ToolReputationTracker()
        tracker.record("tool_a", True, 50.0)
        tracker.record("tool_b", True, 50.0)
        tracker.reset()
        assert tracker.tracked_tools == 0
        assert tracker.get_reputation("tool_a") == 0.0

    def test_reset_single_tool(self):
        tracker = ToolReputationTracker()
        tracker.record("tool_a", True, 50.0)
        tracker.record("tool_b", True, 50.0)
        tracker.reset_tool("tool_a")
        assert tracker.tracked_tools == 1
        assert tracker.get_reputation("tool_a") == 0.0
        assert tracker.get_reputation("tool_b") > 0


class TestToolHints:
    """Hint generation from reputation scores."""

    def test_no_hints_with_insufficient_calls(self):
        tracker = ToolReputationTracker(min_calls_for_guidance=3)
        tracker.record("read", True, 50.0)  # Only 1 call
        hints = tracker.get_tool_hints()
        assert "read" not in hints

    def test_hints_after_sufficient_calls(self):
        tracker = ToolReputationTracker(min_calls_for_guidance=2)
        tracker.record("read", True, 50.0)
        tracker.record("read", True, 60.0)
        hints = tracker.get_tool_hints()
        assert "read" in hints
        assert hints["read"] > 0

    def test_hints_use_canonical_name_for_alias_inputs(self):
        tracker = ToolReputationTracker(min_calls_for_guidance=2)
        tracker.record("read_file", True, 50.0)
        tracker.record("read", True, 60.0)

        hints = tracker.get_tool_hints()
        assert "read" in hints
        assert "read_file" not in hints


class TestSelectionGuidance:
    """Guidance string generation for prompt injection."""

    def test_no_guidance_with_no_data(self):
        tracker = ToolReputationTracker()
        assert tracker.get_selection_guidance() is None

    def test_no_guidance_with_neutral_tools(self):
        """Tools near zero reputation don't generate guidance."""
        tracker = ToolReputationTracker(
            min_calls_for_guidance=2,
            positive_threshold=0.8,
            negative_threshold=-0.8,
        )
        # One success, one failure → near zero
        tracker.record("tool_a", True, 50.0)
        tracker.record("tool_a", False, 50.0)
        assert tracker.get_selection_guidance() is None

    def test_guidance_flags_reliable_tools(self):
        tracker = ToolReputationTracker(min_calls_for_guidance=2)
        tracker.record("read", True, 50.0)
        tracker.record("read", True, 60.0)
        tracker.record("read", True, 55.0)

        guidance = tracker.get_selection_guidance()
        assert guidance is not None
        assert "reliable" in guidance
        assert "read" in guidance

    def test_guidance_flags_unreliable_tools(self):
        tracker = ToolReputationTracker(min_calls_for_guidance=2)
        tracker.record("shell", False, 5000.0)
        tracker.record("shell", False, 3000.0)
        tracker.record("shell", False, 4000.0)

        guidance = tracker.get_selection_guidance()
        assert guidance is not None
        assert "unreliable" in guidance
        assert "shell" in guidance

    def test_guidance_contains_both_prefer_and_avoid(self):
        tracker = ToolReputationTracker(min_calls_for_guidance=2)
        # Good tool
        for _ in range(3):
            tracker.record("read", True, 50.0)
        # Bad tool
        for _ in range(3):
            tracker.record("shell", False, 5000.0)

        guidance = tracker.get_selection_guidance()
        assert guidance is not None
        assert "reliable" in guidance
        assert "unreliable" in guidance

    def test_guidance_respects_max_hints(self):
        tracker = ToolReputationTracker(min_calls_for_guidance=2)
        # Create many good tools
        for i in range(10):
            name = f"tool_{i}"
            tracker.record(name, True, 50.0)
            tracker.record(name, True, 50.0)

        guidance = tracker.get_selection_guidance(max_hints=2)
        # Should limit output
        lines = [l for l in guidance.split("\n") if l.startswith("- ")]
        assert len(lines) <= 4  # max 2 prefer + max 2 avoid
