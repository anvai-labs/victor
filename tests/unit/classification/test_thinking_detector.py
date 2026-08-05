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

"""Tests for thinking pattern detection."""

from unittest.mock import MagicMock

import pytest

from victor.agent.decisions.schemas import DecisionType, LoopDetection, LoopType
from victor.agent.services.protocols.decision_service import DecisionResult
from victor.agent.thinking_detector import (
    CIRCULAR_PATTERNS,
    STALLING_PATTERNS,
    ThinkingPatternDetector,
    create_thinking_detector,
)


class TestThinkingPatternDetector:
    """Tests for ThinkingPatternDetector class."""

    def test_initialization(self):
        """Test detector initializes with default values."""
        detector = ThinkingPatternDetector()

        assert detector._repetition_threshold == 3
        assert detector._similarity_threshold == 0.65
        assert detector._stalling_threshold == 2
        assert detector._iteration == 0
        assert detector._consecutive_stalls == 0

    def test_custom_initialization(self):
        """Test detector initializes with custom values."""
        detector = ThinkingPatternDetector(
            repetition_threshold=5,
            similarity_threshold=0.8,
            stalling_threshold=3,
        )

        assert detector._repetition_threshold == 5
        assert detector._similarity_threshold == 0.8
        assert detector._stalling_threshold == 3


class TestKeywordExtraction:
    """Tests for keyword extraction."""

    def test_extract_keywords_filters_stopwords(self):
        """Test that stopwords are filtered out."""
        detector = ThinkingPatternDetector()

        keywords = detector._extract_keywords("The file contains important data")

        assert "the" not in keywords
        assert "file" in keywords
        assert "contains" in keywords
        assert "important" in keywords
        assert "data" in keywords

    def test_extract_keywords_filters_short_words(self):
        """Test that short words are filtered out."""
        detector = ThinkingPatternDetector()

        keywords = detector._extract_keywords("I am at the top of a file")

        assert "top" not in keywords  # Less than 4 chars
        assert "file" in keywords  # 4 chars, included


class TestSimilarityComputation:
    """Tests for Jaccard similarity."""

    def test_compute_similarity_identical(self):
        """Test similarity of identical sets."""
        detector = ThinkingPatternDetector()

        kw = {"read", "file", "content"}
        similarity = detector._compute_similarity(kw, kw)

        assert similarity == 1.0

    def test_compute_similarity_disjoint(self):
        """Test similarity of disjoint sets."""
        detector = ThinkingPatternDetector()

        kw1 = {"read", "file"}
        kw2 = {"write", "content"}
        similarity = detector._compute_similarity(kw1, kw2)

        assert similarity == 0.0

    def test_compute_similarity_partial(self):
        """Test similarity of partially overlapping sets."""
        detector = ThinkingPatternDetector()

        kw1 = {"read", "file", "content"}
        kw2 = {"read", "file", "data"}
        similarity = detector._compute_similarity(kw1, kw2)

        # Intersection: 2, Union: 4, Jaccard: 2/4 = 0.5
        assert similarity == 0.5

    def test_compute_similarity_empty(self):
        """Test similarity with empty sets."""
        detector = ThinkingPatternDetector()

        assert detector._compute_similarity(set(), set()) == 0.0
        assert detector._compute_similarity({"read"}, set()) == 0.0


class TestCircularPhraseDetection:
    """Tests for circular phrase detection."""

    def test_detect_circular_let_me_read(self):
        """Test detection of 'let me read' pattern."""
        detector = ThinkingPatternDetector()

        assert detector._detect_circular_phrases("Let me read the file")
        assert detector._detect_circular_phrases("let me check this code")
        assert detector._detect_circular_phrases("Let me see the file first")

    def test_detect_circular_i_need_to(self):
        """Test detection of 'I need to' pattern."""
        detector = ThinkingPatternDetector()

        assert detector._detect_circular_phrases("I need to read the file")
        assert detector._detect_circular_phrases("I need to check this code")

    def test_detect_circular_first_let_me(self):
        """Test detection of 'first let me' pattern."""
        detector = ThinkingPatternDetector()

        assert detector._detect_circular_phrases("First let me review")
        assert detector._detect_circular_phrases("Now let me check")

    def test_no_circular_phrase(self):
        """Test that non-circular phrases are not detected."""
        detector = ThinkingPatternDetector()

        assert not detector._detect_circular_phrases("The file contains data")
        assert not detector._detect_circular_phrases("Implementing the feature")


class TestStallingDetection:
    """Tests for stalling pattern detection (DeepSeek-specific)."""

    def test_detect_stalling_let_me(self):
        """Test detection of 'let me' stalling pattern."""
        detector = ThinkingPatternDetector()

        assert detector._detect_stalling("Let me read the file first")
        assert detector._detect_stalling("let me check this code")

    def test_detect_stalling_intent(self):
        """Test detection of intent-based stalling patterns."""
        detector = ThinkingPatternDetector()

        assert detector._detect_stalling("I'll read the file now")
        assert detector._detect_stalling("I will check the code")
        assert detector._detect_stalling("I need to examine this")
        assert detector._detect_stalling("I should look at this file")

    def test_detect_stalling_now(self):
        """Test detection of 'now' stalling pattern."""
        detector = ThinkingPatternDetector()

        assert detector._detect_stalling("Now I'll read the file")
        assert detector._detect_stalling("Now let me check")

    def test_detect_stalling_first(self):
        """Test detection of 'first' stalling pattern."""
        detector = ThinkingPatternDetector()

        assert detector._detect_stalling("First I need to read the file")
        assert detector._detect_stalling("First let me understand")

    def test_no_stalling_action_text(self):
        """Test that action text is not detected as stalling."""
        detector = ThinkingPatternDetector()

        assert not detector._detect_stalling("The file contains user data")
        assert not detector._detect_stalling("Implementing authentication logic")
        assert not detector._detect_stalling("Based on the analysis...")


class TestRecordThinking:
    """Tests for the main record_thinking method."""

    def test_record_thinking_no_loop_first_time(self):
        """Test that first thinking block is not a loop."""
        detector = ThinkingPatternDetector()

        is_loop, guidance = detector.record_thinking("Let me analyze the code")

        assert not is_loop
        assert guidance == ""

    def test_record_thinking_exact_repetition_loop(self):
        """Test exact repetition detection."""
        detector = ThinkingPatternDetector()

        # Record same content 3 times (default threshold)
        detector.record_thinking("Analyzing the authentication module")
        detector.record_thinking("Analyzing the authentication module")
        is_loop, guidance = detector.record_thinking("Analyzing the authentication module")

        assert is_loop
        assert "LOOP DETECTED" in guidance
        assert "exact_repetition" in guidance

    def test_record_thinking_stalling_detection(self):
        """Test stalling detection with consecutive stalls."""
        detector = ThinkingPatternDetector(stalling_threshold=2)

        # First stall - no detection yet
        is_loop, guidance = detector.record_thinking("Let me read the file")
        assert not is_loop

        # Second stall - should trigger detection
        is_loop, guidance = detector.record_thinking("Let me check the code")
        assert is_loop
        assert "STALLING DETECTED" in guidance

    def test_record_thinking_stalling_reset_on_action(self):
        """Test that stalling counter resets on non-stalling content."""
        detector = ThinkingPatternDetector(stalling_threshold=2)

        # First stall
        detector.record_thinking("Let me read the file")

        # Action text - should reset counter
        detector.record_thinking("The file contains user authentication logic")

        # Another stall - should not trigger (counter was reset)
        is_loop, guidance = detector.record_thinking("Let me check the database")
        assert not is_loop

    def test_record_thinking_uses_runtime_intelligence_for_ambiguous_loop(self):
        """RuntimeIntelligenceService is the canonical LLM loop-decision path."""
        runtime_intelligence = MagicMock()
        runtime_intelligence.decide_sync.return_value = DecisionResult(
            decision_type=DecisionType.LOOP_DETECTION,
            result=LoopDetection(is_loop=True, loop_type=LoopType.CIRCULAR),
            source="llm",
            confidence=0.8,
        )
        detector = ThinkingPatternDetector(runtime_intelligence=runtime_intelligence)

        detector.record_thinking("Inspecting service boundaries")
        detector.record_thinking("Reviewing remaining loop heuristics")
        is_loop, guidance = detector.record_thinking("Comparing runtime decision branches")

        assert is_loop is True
        assert "circular" in guidance
        runtime_intelligence.decide_sync.assert_called_once()

    def test_record_thinking_semantic_similarity_loop(self):
        """Test semantic similarity detection."""
        detector = ThinkingPatternDetector(similarity_threshold=0.6)

        # Record similar content multiple times
        detector.record_thinking("Reading authentication module source code file")
        detector.record_thinking("Reading authentication module source file content")
        is_loop, guidance = detector.record_thinking(
            "Reading authentication module source code content"
        )

        assert is_loop
        assert "LOOP DETECTED" in guidance


class TestGuidanceGeneration:
    """Tests for guidance message generation."""

    def test_guidance_stalling_file_read(self):
        """Test stalling guidance for file read category."""
        detector = ThinkingPatternDetector(stalling_threshold=2)

        detector.record_thinking("Let me read the configuration file")
        is_loop, guidance = detector.record_thinking("Let me read the settings file")

        assert is_loop
        assert "EXECUTE the read tool NOW" in guidance

    def test_guidance_stalling_search(self):
        """Test stalling guidance for search category."""
        detector = ThinkingPatternDetector(stalling_threshold=2)

        detector.record_thinking("Let me search for the function definition")
        is_loop, guidance = detector.record_thinking("Let me find the class definition")

        assert is_loop
        assert "EXECUTE the search tool NOW" in guidance

    def test_guidance_stalling_implementation(self):
        """Test stalling guidance for implementation category."""
        detector = ThinkingPatternDetector(stalling_threshold=2)

        detector.record_thinking("Let me implement the feature now")
        is_loop, guidance = detector.record_thinking("Let me write the code")

        assert is_loop
        assert "EXECUTE the edit/write tool NOW" in guidance


class TestStatistics:
    """Tests for statistics tracking."""

    def test_get_stats_initial(self):
        """Test initial statistics."""
        detector = ThinkingPatternDetector()
        stats = detector.get_stats()

        assert stats["total_analyzed"] == 0
        assert stats["loops_detected"] == 0
        assert stats["exact_matches"] == 0
        assert stats["similar_matches"] == 0
        assert stats["stalling_detected"] == 0
        assert stats["consecutive_stalls"] == 0

    def test_get_stats_after_stalling(self):
        """Test statistics after stalling detection."""
        detector = ThinkingPatternDetector(stalling_threshold=2)

        detector.record_thinking("Let me read the file")
        detector.record_thinking("Let me check the code")

        stats = detector.get_stats()
        assert stats["total_analyzed"] == 2
        assert stats["loops_detected"] == 1
        assert stats["stalling_detected"] == 1

    def test_clear_stats(self):
        """Test clearing statistics."""
        detector = ThinkingPatternDetector(stalling_threshold=2)

        detector.record_thinking("Let me read the file")
        detector.record_thinking("Let me check the code")

        detector.clear_stats()
        stats = detector.get_stats()

        assert stats["total_analyzed"] == 0
        assert stats["loops_detected"] == 0
        assert stats["stalling_detected"] == 0


class TestReset:
    """Tests for detector reset."""

    def test_reset_clears_state(self):
        """Test that reset clears all state."""
        detector = ThinkingPatternDetector()

        detector.record_thinking("Some thinking")
        detector.record_thinking("Some thinking")

        detector.reset()

        assert detector._iteration == 0
        assert detector._consecutive_stalls == 0
        assert len(detector._history) == 0
        assert len(detector._pattern_counts) == 0


class TestFactoryFunction:
    """Tests for create_thinking_detector factory."""

    def test_create_thinking_detector_defaults(self):
        """Test factory with default values."""
        detector = create_thinking_detector()

        assert isinstance(detector, ThinkingPatternDetector)
        assert detector._repetition_threshold == 3
        assert detector._similarity_threshold == 0.65

    def test_create_thinking_detector_custom(self):
        """Test factory with custom values."""
        detector = create_thinking_detector(
            repetition_threshold=5,
            similarity_threshold=0.8,
        )

        assert detector._repetition_threshold == 5
        assert detector._similarity_threshold == 0.8


class TestPatternCompilation:
    """Tests for pattern regex compilation."""

    def test_circular_patterns_compiled(self):
        """Test that circular patterns are compiled regex patterns."""
        for pattern in CIRCULAR_PATTERNS:
            assert hasattr(pattern, "search")
            assert hasattr(pattern, "match")

    def test_stalling_patterns_compiled(self):
        """Test that stalling patterns are compiled regex patterns."""
        for pattern in STALLING_PATTERNS:
            assert hasattr(pattern, "search")
            assert hasattr(pattern, "match")
