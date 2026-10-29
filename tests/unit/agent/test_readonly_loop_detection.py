"""Tests for read-only loop detection in UnifiedTaskTracker."""

from victor.agent.unified_task_tracker import (
    UnifiedTaskTracker,
    TrackerTaskType,
    StopReason,
)


class TestReadOnlyLoopDetection:
    """Test read-only loop detection that catches analysis/research loops."""

    def test_no_loop_on_initial_iterations(self):
        """Should not detect loop before minimum iterations."""
        tracker = UnifiedTaskTracker()
        tracker.set_task_type(TrackerTaskType.ANALYZE)

        # Record reading 5 files
        for i in range(5):
            tracker._progress.files_read.add(f"file_{i}.py")

        decision = tracker.should_stop()
        assert not decision.should_stop

    def test_readonly_loop_detected_at_20_files(self):
        """Should detect read-only loop after 20 files without modifications."""
        tracker = UnifiedTaskTracker()
        tracker.set_task_type(TrackerTaskType.ANALYZE)

        # Record reading 21 files without any modifications
        for i in range(21):
            tracker._progress.files_read.add(f"file_{i}.py")
            tracker._progress.iteration_count += 1

        decision = tracker.should_stop()
        assert decision.should_stop
        assert decision.reason == StopReason.LOOP_DETECTED
        assert "Read-only loop" in decision.hint
        assert "21 files read" in decision.hint

    def test_readonly_loop_detected_at_50_files_with_few_writes(self):
        """Should detect read-heavy loop after 50 files with <5 modifications."""
        tracker = UnifiedTaskTracker()
        tracker.set_task_type(TrackerTaskType.SEARCH)

        # Record reading 51 files with only 3 modifications
        for i in range(51):
            tracker._progress.files_read.add(f"file_{i}.py")
            tracker._progress.iteration_count += 1

        for i in range(3):
            tracker._progress.files_modified.add(f"file_{i}.py")

        decision = tracker.should_stop()
        assert decision.should_stop
        assert decision.reason == StopReason.LOOP_DETECTED
        assert "Read-heavy loop" in decision.hint
        assert "51 files read" in decision.hint
        assert "only 3 files modified" in decision.hint

    def test_no_loop_with_sufficient_writes(self):
        """Should not detect loop when modifications are balanced."""
        tracker = UnifiedTaskTracker()
        tracker.set_task_type(TrackerTaskType.RESEARCH)

        # Record reading 25 files with 10 modifications (below iteration limit)
        for i in range(25):
            tracker._progress.files_read.add(f"file_{i}.py")
            tracker._progress.iteration_count += 1

        for i in range(10):
            tracker._progress.files_modified.add(f"file_{i}.py")

        decision = tracker.should_stop()
        # Should not stop due to read-only loop (may stop for other reasons)
        assert decision.reason != StopReason.LOOP_DETECTED or not decision.should_stop

    def test_readonly_loop_only_for_analysis_tasks(self):
        """Should only check read-only loop for ANALYZE/SEARCH/RESEARCH tasks."""
        tracker = UnifiedTaskTracker()
        tracker.set_task_type(TrackerTaskType.EDIT)

        # Record reading 30 files without modifications (EDIT task)
        for i in range(30):
            tracker._progress.files_read.add(f"file_{i}.py")
            tracker._progress.iteration_count += 1

        decision = tracker.should_stop()
        assert not decision.should_stop  # EDIT tasks don't trigger read-only check

    def test_no_readonly_loop_below_threshold(self):
        """Should not detect loop at exactly 20 files (threshold is >20)."""
        tracker = UnifiedTaskTracker()
        tracker.set_task_type(TrackerTaskType.ANALYZE)

        # Record reading exactly 20 files
        for i in range(20):
            tracker._progress.files_read.add(f"file_{i}.py")
            tracker._progress.iteration_count += 1

        decision = tracker.should_stop()
        assert not decision.should_stop  # Threshold is >20, not >=20
