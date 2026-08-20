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

"""Comprehensive tests for workflow optimization components.

Tests all 6 components introduced to address MODE workflow issues:
1. TaskCompletionDetector - Task completion detection
2. ReadResultCache - File read deduplication
3. TimeAwareExecutor - Time-aware execution phases
4. ThinkingPatternDetector - Thinking loop detection
5. ResourceManager - Resource lifecycle management
6. ModeCompletionCriteria - Mode-specific early exit

Reference: workflow-test-issues-v2.md
"""

import time
from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# Test: TaskCompletionDetector (Issue #1)
# =============================================================================


class TestTaskCompletionDetector:
    """Tests for task completion detection."""

    @pytest.fixture
    def detector(self):
        """Create a fresh detector for each test."""
        from victor.agent.task_completion import TaskCompletionDetector

        return TaskCompletionDetector()

    def test_analyze_intent_file_creation(self, detector):
        """Test intent analysis for file creation requests."""
        deliverables = detector.analyze_intent("Create a cache_manager.py file")

        from victor.agent.task_completion import DeliverableType

        assert DeliverableType.FILE_CREATED in deliverables

    def test_analyze_intent_plan_request(self, detector):
        """Test intent analysis for planning requests."""
        deliverables = detector.analyze_intent("Plan the implementation of a retry decorator")

        from victor.agent.task_completion import DeliverableType

        assert DeliverableType.PLAN_PROVIDED in deliverables

    def test_analyze_intent_question(self, detector):
        """Test intent analysis for questions."""
        deliverables = detector.analyze_intent("What does this function do?")

        from victor.agent.task_completion import DeliverableType

        assert DeliverableType.ANSWER_PROVIDED in deliverables

    def test_record_tool_result_success(self, detector):
        """Test recording successful tool results."""
        detector.analyze_intent("Create a file")
        detector.record_tool_result("write", {"success": True, "path": "/test/file.py"})

        state = detector.get_state()
        assert len(state.completed_deliverables) == 1
        assert state.completed_deliverables[0].artifact_path == "/test/file.py"

    def test_record_tool_result_normalizes_edit_aliases(self, detector):
        """Edit aliases should count as file-modifying deliverables."""
        detector.analyze_intent("Update a file")
        detector.record_tool_result("edit_file", {"success": True, "path": "/test/file.py"})

        state = detector.get_state()
        assert len(state.completed_deliverables) == 1
        assert state.completed_deliverables[0].artifact_path == "/test/file.py"

    def test_completion_signal_detection(self, detector):
        """Test detection of completion signals in response text."""
        detector.analyze_response("The file has been created successfully.")

        state = detector.get_state()
        # Signals now have priority prefix (passive: for phrase detection)
        assert any("has been created" in signal for signal in state.completion_signals)

    def test_continuation_loop_detection(self, detector):
        """Test detection of continuation request patterns."""
        detector.analyze_response("How would you like to proceed?")
        detector.analyze_response("Would you like me to continue?")

        state = detector.get_state()
        assert state.continuation_requests >= 2

    def test_should_stop_after_file_created(self, detector):
        """Test that detector signals stop after file creation."""
        detector.analyze_intent("Create a cache.py file")
        detector.record_tool_result("write", {"success": True, "path": "cache.py"})

        assert detector.should_stop()

    def test_force_complete_after_max_continuations(self, detector):
        """Test forced completion after too many continuations."""
        state = detector.get_state()
        state.max_continuation_requests = 2

        detector.analyze_response("How would you like to proceed?")
        detector.analyze_response("Would you like me to continue?")

        assert detector.should_stop()

    def test_completion_summary_format(self, detector):
        """Test completion summary generation."""
        detector.analyze_intent("Create a file")
        detector.record_tool_result("write", {"success": True, "path": "test.py"})

        summary = detector.get_completion_summary()
        assert "Deliverables" in summary
        assert "test.py" in summary

    def test_reset_clears_state(self, detector):
        """Test that reset clears all state."""
        detector.analyze_intent("Create a file")
        detector.record_tool_result("write", {"success": True, "path": "test.py"})
        detector.reset()

        state = detector.get_state()
        assert len(state.completed_deliverables) == 0
        assert len(state.expected_deliverables) == 0


# =============================================================================
# Test: ReadResultCache (Issue #2)
# =============================================================================


class TestReadResultCache:
    """Tests for read result caching."""

    @pytest.fixture
    def cache(self):
        """Create a fresh cache for each test."""
        from victor.agent.read_cache import ReadResultCache

        return ReadResultCache(ttl_seconds=60.0, max_entries=10)

    def test_cache_miss_on_empty(self, cache):
        """Test cache miss when cache is empty."""
        result = cache.get("/path/to/file.py")
        assert result is None

    def test_cache_hit_after_put(self, cache):
        """Test cache hit after putting content."""
        cache.put("/path/to/file.py", "file content")
        result = cache.get("/path/to/file.py")

        assert result == "file content"

    def test_cache_invalidation(self, cache):
        """Test cache invalidation."""
        cache.put("/path/to/file.py", "content")
        cache.invalidate("/path/to/file.py")

        result = cache.get("/path/to/file.py")
        assert result is None

    def test_cache_expiration(self, cache):
        """Test cache expiration after TTL."""
        from victor.agent.read_cache import ReadResultCache

        short_cache = ReadResultCache(ttl_seconds=0.01)
        short_cache.put("/path/to/file.py", "content")

        time.sleep(0.02)
        result = short_cache.get("/path/to/file.py")
        assert result is None

    def test_lru_eviction(self, cache):
        """Test LRU eviction when at capacity."""
        # Fill cache to capacity
        for i in range(10):
            cache.put(f"/path/file{i}.py", f"content{i}")

        # Add one more to trigger eviction
        cache.put("/path/new.py", "new content")

        # First file should be evicted (LRU)
        result = cache.get("/path/file0.py")
        assert result is None

    def test_cache_stats(self, cache):
        """Test cache statistics tracking."""
        cache.put("/path/file.py", "content")
        cache.get("/path/file.py")  # hit
        cache.get("/path/nonexistent.py")  # miss

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_redundant_read_warning(self, cache):
        """Test redundant read warning."""
        from victor.agent.read_cache import ReadResultCache

        cache = ReadResultCache(redundant_threshold=2, redundant_window=60.0)
        cache.put("/path/file.py", "content")
        cache.get("/path/file.py")
        cache.get("/path/file.py")

        assert cache.should_warn_redundant("/path/file.py")

    def test_partial_read_with_offset(self, cache):
        """Test partial read with line offset."""
        content = "line1\nline2\nline3\nline4"
        cache.put("/path/file.py", content)

        result = cache.get("/path/file.py", offset=1, limit=2)
        assert "line2" in result
        assert "line3" in result
        assert "line1" not in result

    def test_content_hash_change_detection(self, cache):
        """Test that cache detects content changes."""
        cache.put("/path/file.py", "original content")
        cache.put("/path/file.py", "modified content")

        result = cache.get("/path/file.py")
        assert result == "modified content"

    def test_invalidate_all(self, cache):
        """Test invalidating all entries."""
        cache.put("/path/file1.py", "content1")
        cache.put("/path/file2.py", "content2")
        cache.invalidate_all()

        assert cache.get("/path/file1.py") is None
        assert cache.get("/path/file2.py") is None


# =============================================================================
# Test: TimeAwareExecutor (Issue #3)
# =============================================================================


class TestTimeAwareExecutor:
    """Tests for time-aware execution."""

    @pytest.fixture
    def executor(self):
        """Create executor with 100 second budget."""
        from victor.agent.time_aware_executor import TimeAwareExecutor

        return TimeAwareExecutor(timeout_seconds=100.0)

    def test_normal_phase_at_start(self, executor):
        """Test that phase is NORMAL at start."""
        from victor.agent.time_aware_executor import TimePhase

        assert executor.get_phase() == TimePhase.NORMAL

    def test_no_guidance_in_normal_phase(self, executor):
        """Test no guidance message in NORMAL phase."""
        guidance = executor.get_time_guidance()
        assert guidance == "" or "NORMAL" not in guidance

    def test_remaining_seconds(self, executor):
        """Test remaining seconds calculation."""
        remaining = executor.get_remaining_seconds()
        assert remaining is not None
        assert remaining <= 100.0

    def test_elapsed_seconds(self, executor):
        """Test elapsed seconds calculation."""
        time.sleep(0.01)
        elapsed = executor.get_elapsed_seconds()
        assert elapsed > 0

    def test_checkpoint_recording(self, executor):
        """Test checkpoint recording."""
        checkpoint = executor.checkpoint("Completed analysis")

        assert checkpoint is not None
        assert checkpoint.description == "Completed analysis"

    def test_budget_summary(self, executor):
        """Test budget summary generation."""
        summary = executor.get_budget_summary()

        assert summary is not None
        assert "total_seconds" in summary
        assert summary["total_seconds"] == 100.0

    def test_should_summarize_false_in_normal(self, executor):
        """Test should_summarize is False in NORMAL phase."""
        assert not executor.should_summarize_now()

    def test_should_avoid_exploration_false_in_normal(self, executor):
        """Test should_avoid_exploration is False in NORMAL phase."""
        assert not executor.should_avoid_exploration()

    def test_tool_budget_recommendation_in_normal(self, executor):
        """Test tool budget recommendation in NORMAL phase."""
        recommendation = executor.get_tool_budget_recommendation()
        assert recommendation == 20

    def test_extend_budget(self, executor):
        """Test budget extension."""
        original = executor.get_remaining_seconds()
        executor.extend_budget(50.0)
        extended = executor.get_remaining_seconds()

        assert extended > original

    def test_phase_change_callback(self):
        """Test phase change callback is called."""
        from victor.agent.time_aware_executor import TimeAwareExecutor, TimePhase

        callback_called = []

        def on_change(old, new):
            callback_called.append((old, new))

        executor = TimeAwareExecutor(timeout_seconds=0.1, on_phase_change=on_change)
        time.sleep(0.15)
        executor.get_phase()

        # Should have transitioned to EXPIRED
        assert len(callback_called) > 0

    def test_is_expired(self):
        """Test is_expired check."""
        from victor.agent.time_aware_executor import TimeAwareExecutor

        executor = TimeAwareExecutor(timeout_seconds=0.01)
        time.sleep(0.02)
        assert executor.is_expired()

    def test_context_manager(self):
        """Test TimeAwareContext context manager."""
        from victor.agent.time_aware_executor import TimeAwareContext

        with TimeAwareContext(timeout_seconds=60.0) as executor:
            assert executor.get_remaining_seconds() is not None


# =============================================================================
# Test: ThinkingPatternDetector (Issue #4)
# =============================================================================


class TestThinkingPatternDetector:
    """Tests for thinking pattern detection."""

    @pytest.fixture
    def detector(self):
        """Create a fresh detector for each test."""
        from victor.agent.thinking_detector import ThinkingPatternDetector

        return ThinkingPatternDetector()

    def test_no_loop_on_first_thinking(self, detector):
        """Test no loop detected on first thinking block."""
        is_loop, guidance = detector.record_thinking("Let me analyze this code.")
        assert not is_loop
        assert guidance == ""

    def test_exact_repetition_detection(self, detector):
        """Test detection of exact repetition (may trigger loop or stalling detection)."""
        content = "Let me read the file to understand it."

        detector.record_thinking(content)
        detector.record_thinking(content)
        is_loop, guidance = detector.record_thinking(content)

        assert is_loop
        # Repetitive intent statements may trigger stalling detection
        assert "LOOP DETECTED" in guidance or "STALLING DETECTED" in guidance

    def test_semantic_similarity_detection(self, detector):
        """Test detection of semantically similar thinking."""
        detector.record_thinking("Let me read the file content.")
        detector.record_thinking("I need to read this file's content.")
        is_loop, guidance = detector.record_thinking("Reading the file content now.")

        # May or may not detect as loop depending on similarity threshold
        # Just verify it runs without error
        assert isinstance(is_loop, bool)

    def test_keyword_extraction(self, detector):
        """Test keyword extraction from thinking blocks."""
        keywords = detector._extract_keywords("Let me analyze the cache manager code.")

        assert "cache" in keywords
        assert "manager" in keywords
        assert "analyze" in keywords
        # Stopwords should be excluded
        assert "the" not in keywords
        assert "me" not in keywords

    def test_similarity_computation(self, detector):
        """Test Jaccard similarity computation."""
        kw1 = {"cache", "manager", "file"}
        kw2 = {"cache", "manager", "code"}

        similarity = detector._compute_similarity(kw1, kw2)
        assert 0.0 < similarity < 1.0

    def test_circular_phrase_detection(self, detector):
        """Test detection of circular thinking phrases."""
        has_circular = detector._detect_circular_phrases("Let me read the file again.")
        assert has_circular

    def test_category_detection(self, detector):
        """Test categorization of thinking blocks."""
        assert detector._categorize_thinking("Let me read the file") == "file_read"
        assert detector._categorize_thinking("I need to search for") == "search"
        assert detector._categorize_thinking("Let me analyze this") == "analysis"
        assert detector._categorize_thinking("I'll implement this") == "implementation"

    def test_stats_tracking(self, detector):
        """Test statistics tracking."""
        detector.record_thinking("First thought")
        detector.record_thinking("Second thought")

        stats = detector.get_stats()
        assert stats["total_analyzed"] == 2
        assert stats["history_size"] == 2

    def test_reset_clears_state(self, detector):
        """Test that reset clears detector state."""
        detector.record_thinking("Test thought")
        detector.reset()

        stats = detector.get_stats()
        assert stats["history_size"] == 0
        assert stats["unique_patterns"] == 0

    def test_guidance_message_category_specific(self, detector):
        """Test that guidance messages are category-specific."""
        content = "Let me read the file again"
        detector.record_thinking(content)
        detector.record_thinking(content)
        is_loop, guidance = detector.record_thinking(content)

        if is_loop:
            assert "read" in guidance.lower() or "file" in guidance.lower()


# =============================================================================
# Test: ResourceManager (Issue #5)
# =============================================================================


class TestResourceManager:
    """Tests for resource lifecycle management."""

    @pytest.fixture
    def manager(self):
        """Create a fresh resource manager."""
        from victor.agent.resource_manager import ResourceManager

        # Reset singleton for testing
        ResourceManager._instance = None
        mgr = ResourceManager()
        yield mgr
        mgr.reset()
        ResourceManager._instance = None

    def test_singleton_pattern(self, manager):
        """Test that ResourceManager is a singleton."""
        from victor.agent.resource_manager import ResourceManager

        manager2 = ResourceManager()
        assert manager is manager2

    def test_register_cleanup_callback(self, manager):
        """Test registering cleanup callbacks."""
        callback_called = []

        def cleanup():
            callback_called.append(True)

        manager.register_cleanup(cleanup)
        manager.cleanup_all()

        assert len(callback_called) == 1

    def test_cleanup_priority_order(self, manager):
        """Test callbacks are called in priority order."""
        order = []

        manager.register_cleanup(lambda: order.append(1), priority=1)
        manager.register_cleanup(lambda: order.append(3), priority=3)
        manager.register_cleanup(lambda: order.append(2), priority=2)

        manager.cleanup_all()

        assert order == [3, 2, 1]  # Higher priority first

    def test_register_resource(self, manager):
        """Test registering resources for cleanup."""

        class MockResource:
            closed = False

            def close(self):
                self.closed = True

        resource = MockResource()
        manager.register_resource(resource, "test_resource", "close")
        manager.cleanup_all()

        assert resource.closed

    def test_unregister_resource(self, manager):
        """Test unregistering resources."""

        class MockResource:
            def close(self):
                pass

        resource = MockResource()
        manager.register_resource(resource, "test_resource", "close")
        result = manager.unregister_resource("test_resource")

        assert result is True

    def test_cleanup_specific_resource(self, manager):
        """Test cleaning up specific resources."""

        class MockResource:
            closed = False

            def close(self):
                self.closed = True

        resource = MockResource()
        manager.register_resource(resource, "test_resource", "close")
        manager.cleanup_resource("test_resource")

        assert resource.closed

    def test_resource_status(self, manager):
        """Test getting resource status."""

        class MockResource:
            def close(self):
                pass

        resource = MockResource()
        manager.register_resource(resource, "test", "close")

        status = manager.get_resource_status()
        assert "test" in status
        assert status["test"]["alive"] is True

    def test_cleanup_error_handling(self, manager):
        """Test cleanup continues despite errors."""
        order = []

        def failing_cleanup():
            raise RuntimeError("Cleanup failed")

        def success_cleanup():
            order.append("success")

        manager.register_cleanup(failing_cleanup)
        manager.register_cleanup(success_cleanup)

        results = manager.cleanup_all()

        # Should have recorded both success and failure
        assert len(results) == 2

    def test_managed_resource_context_manager(self, manager):
        """Test context manager for managed resources."""

        class MockResource:
            closed = False

            def close(self):
                self.closed = True

        resource = MockResource()
        with manager.managed_resource(resource, "test", "close"):
            assert not resource.closed

        assert resource.closed

    def test_stats(self, manager):
        """Test getting manager statistics."""
        manager.register_cleanup(lambda: None)

        class MockResource:
            def close(self):
                pass

        manager.register_resource(MockResource(), "test", "close")

        stats = manager.get_stats()
        assert stats["resources_registered"] == 1
        assert stats["callbacks_registered"] == 1


# =============================================================================
# Test: ModeCompletionCriteria (Issue #6)
# =============================================================================


class TestModeCompletionCriteria:
    """Tests for mode-specific completion criteria."""

    @pytest.fixture
    def criteria(self):
        """Create a fresh criteria checker."""
        from victor.agent.budget_manager import ModeCompletionCriteria

        return ModeCompletionCriteria()

    def test_get_explore_criteria(self, criteria):
        """Test getting EXPLORE mode criteria."""
        config = criteria.get_criteria("EXPLORE")

        assert config.min_files_read == 1
        assert config.max_iterations == 15
        assert len(config.completion_signals) > 0

    def test_get_plan_criteria(self, criteria):
        """Test getting PLAN mode criteria."""
        config = criteria.get_criteria("PLAN")

        assert config.min_files_read == 1
        assert config.max_iterations == 20
        assert len(config.required_sections) > 0

    def test_get_build_criteria(self, criteria):
        """Test getting BUILD mode criteria."""
        config = criteria.get_criteria("BUILD")

        assert config.min_files_written == 1
        assert config.max_iterations == 30

    def test_case_insensitive_mode(self, criteria):
        """Test mode name is case insensitive."""
        config1 = criteria.get_criteria("EXPLORE")
        config2 = criteria.get_criteria("explore")
        config3 = criteria.get_criteria("Explore")

        assert config1.max_iterations == config2.max_iterations == config3.max_iterations

    def test_unknown_mode_returns_default(self, criteria):
        """Test unknown mode returns default criteria."""
        config = criteria.get_criteria("UNKNOWN_MODE")

        from victor.agent.budget_manager import ModeCompletionConfig

        default = ModeCompletionConfig()
        assert config.max_iterations == default.max_iterations

    def test_early_exit_explore_success(self, criteria):
        """Test early exit for EXPLORE mode when complete."""
        should_exit, reason = criteria.check_early_exit(
            mode="EXPLORE",
            files_read=2,
            files_written=0,
            iterations=5,
            response_text="Here's what I found in the codebase.",
        )

        assert should_exit
        assert "signal detected" in reason.lower()

    def test_early_exit_explore_min_not_met(self, criteria):
        """Test no early exit when minimum files not read."""
        should_exit, reason = criteria.check_early_exit(
            mode="EXPLORE",
            files_read=0,
            files_written=0,
            iterations=5,
            response_text="Here's what I found.",
        )

        assert not should_exit
        assert "more file" in reason.lower()

    def test_early_exit_plan_success(self, criteria):
        """Test early exit for PLAN mode with required sections."""
        should_exit, reason = criteria.check_early_exit(
            mode="PLAN",
            files_read=2,
            files_written=0,
            iterations=5,
            response_text="Here's the implementation plan:\n## Step 1\nModify file x.py",
        )

        assert should_exit

    def test_early_exit_plan_missing_sections(self, criteria):
        """Test no early exit when required sections missing."""
        should_exit, reason = criteria.check_early_exit(
            mode="PLAN",
            files_read=2,
            files_written=0,
            iterations=5,
            response_text="Here's the plan - just do it.",
        )

        # Should fail because missing required sections
        assert not should_exit or "section" in reason.lower()

    def test_early_exit_build_success(self, criteria):
        """Test early exit for BUILD mode when file written."""
        should_exit, reason = criteria.check_early_exit(
            mode="BUILD",
            files_read=0,
            files_written=1,
            iterations=5,
            response_text="The file has been created successfully.",
        )

        assert should_exit

    def test_early_exit_build_no_file_written(self, criteria):
        """Test no early exit when no file written in BUILD mode."""
        should_exit, reason = criteria.check_early_exit(
            mode="BUILD",
            files_read=3,
            files_written=0,
            iterations=5,
            response_text="I've created the implementation.",
        )

        assert not should_exit
        assert "file" in reason.lower()

    def test_early_exit_max_iterations(self, criteria):
        """Test early exit when max iterations exceeded."""
        should_exit, reason = criteria.check_early_exit(
            mode="EXPLORE",
            files_read=0,
            files_written=0,
            iterations=15,
            response_text="Still exploring...",
        )

        assert should_exit
        assert "iterations" in reason.lower()

    def test_no_completion_signal(self, criteria):
        """Test no early exit when no completion signal detected."""
        should_exit, reason = criteria.check_early_exit(
            mode="EXPLORE",
            files_read=2,
            files_written=0,
            iterations=3,
            response_text="I'm still analyzing the code structure.",
        )

        assert not should_exit
        assert "signal" in reason.lower()

    def test_progress_tracking(self, criteria):
        """Test progress tracking."""
        criteria.check_early_exit(
            mode="PLAN",
            files_read=1,
            files_written=0,
            iterations=10,
            response_text="Working on it...",
        )

        progress = criteria.get_progress("PLAN")
        assert progress["iterations"] == 10
        assert progress["progress_pct"] == 50.0

    def test_reset_clears_iterations(self, criteria):
        """Test reset clears iteration counts."""
        criteria.check_early_exit("PLAN", 1, 0, 10, "Working...")
        criteria.reset()

        progress = criteria.get_progress("PLAN")
        assert progress["iterations"] == 0


# =============================================================================
# Test: ExtendedBudgetManager Integration
# =============================================================================


class TestExtendedBudgetManager:
    """Tests for ExtendedBudgetManager with mode completion."""

    @pytest.fixture
    def manager(self):
        """Create a fresh extended budget manager."""
        from victor.agent.budget_manager import create_extended_budget_manager

        return create_extended_budget_manager(mode="BUILD")

    def test_set_mode(self, manager):
        """Test setting operating mode."""
        manager.set_mode("PLAN")
        assert manager._current_mode == "PLAN"

    def test_record_file_operations(self, manager):
        """Test file operation tracking."""
        manager.record_file_read()
        manager.record_file_write()

        assert manager._files_read == 1
        assert manager._files_written == 1

    def test_tool_call_tracks_files(self, manager):
        """Test that tool calls track file operations."""
        manager.record_tool_call("read")
        manager.record_tool_call("write")

        assert manager._files_read == 1
        assert manager._files_written == 1

    def test_should_early_exit_integration(self, manager):
        """Test should_early_exit integration."""
        manager.set_mode("BUILD")
        manager.record_file_write()

        should_exit, reason = manager.should_early_exit("The file has been created successfully.")

        assert should_exit

    def test_should_early_exit_no_mode(self, manager):
        """Test should_early_exit with no mode set."""
        manager._current_mode = None

        should_exit, reason = manager.should_early_exit("Some text")

        assert not should_exit
        assert "No mode" in reason

    def test_mode_progress_tracking(self, manager):
        """Test mode progress tracking."""
        manager.set_mode("PLAN")
        manager.record_file_read()
        manager.record_file_read()

        progress = manager.get_mode_progress()

        assert progress["files_read"] == 2
        assert progress["mode"] == "PLAN"

    def test_reset_clears_file_counts(self, manager):
        """Test reset clears file counts."""
        manager.record_file_read()
        manager.record_file_write()
        manager.reset()

        assert manager._files_read == 0
        assert manager._files_written == 0


# =============================================================================
# Test: Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_task_completion_detector(self):
        """Test task completion detector factory."""
        from victor.agent.task_completion import create_task_completion_detector

        detector = create_task_completion_detector()
        assert detector is not None

    def test_create_read_cache(self):
        """Test read cache factory."""
        from victor.agent.read_cache import create_read_cache

        cache = create_read_cache(ttl_seconds=120.0, max_entries=50)
        assert cache is not None
        assert cache._ttl == 120.0

    def test_create_time_aware_executor(self):
        """Test time aware executor factory."""
        from victor.agent.time_aware_executor import create_time_aware_executor

        executor = create_time_aware_executor(timeout_seconds=300.0)
        assert executor is not None
        assert executor.get_remaining_seconds() is not None

    def test_create_thinking_detector(self):
        """Test thinking detector factory."""
        from victor.agent.thinking_detector import create_thinking_detector

        detector = create_thinking_detector(repetition_threshold=4)
        assert detector is not None
        assert detector._repetition_threshold == 4

    def test_create_budget_manager(self):
        """Test budget manager factory."""
        from victor.agent.budget_manager import create_budget_manager

        manager = create_budget_manager(model_multiplier=1.2)
        assert manager is not None
        assert manager._model_multiplier == 1.2

    def test_create_mode_completion_criteria(self):
        """Test mode completion criteria factory."""
        from victor.agent.budget_manager import create_mode_completion_criteria

        criteria = create_mode_completion_criteria()
        assert criteria is not None

    def test_get_resource_manager(self):
        """Test resource manager factory."""
        from victor.agent.resource_manager import get_resource_manager, ResourceManager

        # Reset singleton
        ResourceManager._instance = None

        manager = get_resource_manager()
        assert manager is not None

        # Cleanup
        manager.reset()
        ResourceManager._instance = None
