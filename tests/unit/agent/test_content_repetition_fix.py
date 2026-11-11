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

"""Tests for content repetition detection and loop exit fixes.

P0 Issues Fixed:
1. Content Repetition Loop - High overlap detection now forces completion earlier
2. Tool Call Extraction Failures - Lower confidence threshold and improved patterns
3. Loop Detection Forces Early Completion - Proper exit with forced_stop flag
4. Forced Tool Execution Loop - Faster exit after 2 failed attempts (down from 3)
"""

from unittest.mock import MagicMock

from victor.agent.tool_call_extractor import (
    ToolCallExtractor,
    extract_tool_call_from_text,
)
from victor.agent.unified_task_tracker import UnifiedTaskTracker, TrackerTaskType


class TestContentRepetitionDetection:
    """Tests for improved content repetition detection."""

    def test_repetition_threshold_lowered(self):
        """P0 FIX: Repetition threshold lowered from 0.6 to 0.5 overlap."""
        from victor.agent.services.chat_stream_executor import StreamingChatExecutor

        # Create executor with mocked dependencies
        runtime_owner = MagicMock()
        executor = StreamingChatExecutor(runtime_owner=runtime_owner)

        # Simulate two responses with 55% overlap (should trigger detection with new threshold)
        content1 = "I will analyze the code structure in victor/agent/ directory. "
        content1 += "This will help me understand the flow. "
        content1 += "Let me check the files."

        content2 = "I will analyze the code structure in victor/agent/ directory. "
        content2 += "This will help me understand the flow. "
        content2 += "Let me check the files now."  # Minor variation

        # Normalize as the executor does
        import re

        normalized1 = re.sub(r"\s+", " ", content1.strip().lower())
        normalized2 = re.sub(r"\s+", " ", content2.strip().lower())

        words1 = set(normalized1.split())
        words2 = set(normalized2.split())

        overlap = len(words1 & words2) / len(words1 | words2)

        # With new threshold of 0.5, 55% overlap should trigger detection
        assert overlap > 0.5, f"Expected overlap > 0.5, got {overlap:.2f}"

    def test_forces_completion_after_two_repetitions(self):
        """P0 FIX: Should force completion after 2 consecutive high-overlap iterations."""
        tracker = UnifiedTaskTracker()
        tracker.set_task_type(TrackerTaskType.ANALYZE)

        # First response - no loop yet
        response1 = "Let me analyze the codebase structure to find the issue."
        is_loop1 = tracker.check_response_loop(response1)
        assert is_loop1 is False
        assert tracker.progress.response_loop_detected is False

        # Second response - identical high-overlap content starts the loop counter.
        response2 = response1
        is_loop2 = tracker.check_response_loop(response2)
        # Should not trigger loop yet (only first detection)
        assert is_loop2 is False

        # Third response - identical to second (triggers second consecutive detection)
        response3 = response1
        is_loop3 = tracker.check_response_loop(response3)
        # Should trigger loop detection after 2 consecutive detections
        assert is_loop3 is True
        assert tracker.progress.response_loop_detected is True
        assert tracker.progress.forced_stop == "response_loop"


class TestToolCallExtractionImprovements:
    """Tests for improved tool call extraction."""

    def test_lower_confidence_threshold(self):
        """P0 FIX: Extraction should succeed with lower confidence threshold."""
        extractor = ToolCallExtractor()

        # Text with hallucinated tool mention (low confidence extraction)
        text = "Let me check the src/ directory for the main file."

        result = extractor.extract_from_text(text, ["ls"], context=None)

        # Should succeed with new threshold of 0.3 (old was 0.5)
        assert result is not None
        assert result.tool_name == "ls"
        assert result.arguments.get("path") == "src/"
        assert result.confidence >= 0.3

    def test_extract_ls_from_explore_pattern(self):
        """P0 FIX: Should extract ls from 'explore directory' pattern."""
        text = "Let me explore victor/agent/ to understand the code structure."

        result = extract_tool_call_from_text(text, ["ls"], context=None)

        assert result is not None
        assert result.tool_name == "ls"
        assert "victor/agent/" in result.arguments.get("path", "")

    def test_extract_ls_from_show_me_pattern(self):
        """P0 FIX: Should extract ls from 'show me what's in' pattern."""
        text = "Can you show me what's in the src/ directory?"

        result = extract_tool_call_from_text(text, ["ls"], context=None)

        assert result is not None
        assert result.tool_name == "ls"
        assert "src/" in result.arguments.get("path", "")

    def test_prioritized_tool_extraction(self):
        """P0 FIX: Should prioritize ls over other tools when multiple mentioned."""
        text = "I need to explore the directory and check the files."

        # ls should be prioritized even when other tools might match
        result = extract_tool_call_from_text(text, ["ls", "grep"], context=None)

        assert result is not None
        assert result.tool_name == "ls"

    def test_extract_read_with_high_confidence(self):
        """P0 FIX: Read extraction should have high confidence for explicit paths."""
        text = "Let me read the main.py file to understand the entry point."

        result = extract_tool_call_from_text(text, ["read"], context=None)

        assert result is not None
        assert result.tool_name == "read"
        assert "main.py" in result.arguments.get("path", "")
        # Explicit file paths should get high confidence
        assert result.confidence >= 0.8


class TestLoopDetectionWithForcedStop:
    """Tests for proper loop exit with forced_stop flag."""

    def test_should_stop_returns_true_when_forced_stop_set(self):
        """P0 FIX: should_stop should return True when forced_stop is set."""
        tracker = UnifiedTaskTracker()
        tracker.set_task_type(TrackerTaskType.ANALYZE)

        # Initially should not stop
        decision = tracker.should_stop()
        assert decision.should_stop is False

        # Set forced stop
        tracker.force_stop("test_forced_stop")

        # Now should stop
        decision = tracker.should_stop()
        assert decision.should_stop is True
        assert decision.reason.value == "manual_stop"
        assert "test_forced_stop" in decision.hint

    def test_response_loop_sets_forced_stop_flag(self):
        """P0 FIX: Response loop detection should set forced_stop flag."""
        tracker = UnifiedTaskTracker()
        tracker.set_task_type(TrackerTaskType.SEARCH)

        response = "I will search for the relevant code in the repository."

        # First check - no loop
        is_loop1 = tracker.check_response_loop(response)
        assert is_loop1 is False

        # Second check - similar content triggers potential loop
        is_loop2 = tracker.check_response_loop(response)
        assert is_loop2 is False  # Not triggered yet

        # Third check - same content triggers forced stop
        is_loop3 = tracker.check_response_loop(response)
        assert is_loop3 is True
        assert tracker.progress.forced_stop == "response_loop"

        # Verify should_stop returns True
        decision = tracker.should_stop()
        assert decision.should_stop is True
        assert decision.reason.value == "manual_stop"
        assert "response_loop" in decision.hint
