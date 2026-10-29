"""Tests for co-occurrence tracker.

Tests cover:
- Initialization and configuration
- Tool sequence recording
- Co-occurrence matrix updates
- Sequential pattern extraction (bigrams, trigrams)
- Success rate tracking
- Next tool prediction
- Success rate boosting
- Decay and history management
- Statistics and reset
"""

import pytest

from victor.agent.planning.cooccurrence_tracker import (
    CooccurrenceTracker,
    ToolPattern,
    ToolPrediction,
)


class TestInitialization:
    """Test tracker initialization."""

    def test_default_initialization(self):
        """Test tracker with default settings."""
        tracker = CooccurrenceTracker()

        assert tracker.max_history == 10000
        assert tracker.min_observations == 3
        assert tracker.decay_factor == 0.95
        assert tracker.enable_task_type_specificity is True

    def test_custom_initialization(self):
        """Test tracker with custom settings."""
        tracker = CooccurrenceTracker(
            max_history=1000,
            min_observations=5,
            decay_factor=0.9,
            enable_task_type_specificity=False,
        )

        assert tracker.max_history == 1000
        assert tracker.min_observations == 5
        assert tracker.decay_factor == 0.9
        assert tracker.enable_task_type_specificity is False

    def test_initial_statistics(self):
        """Test initial statistics are empty."""
        tracker = CooccurrenceTracker()

        stats = tracker.get_statistics()

        assert stats["total_sequences_recorded"] == 0
        assert stats["history_size"] == 0
        assert stats["unique_tools"] == 0
        assert stats["total_bigrams"] == 0
        assert stats["total_trigrams"] == 0


class TestToolSequenceRecording:
    """Test recording of tool sequences."""

    def test_record_simple_sequence(self):
        """Test recording a simple tool sequence."""
        tracker = CooccurrenceTracker()

        tracker.record_tool_sequence(
            tools=["search", "read", "edit"],
            task_type="bugfix",
            success=True,
        )

        stats = tracker.get_statistics()
        assert stats["total_sequences_recorded"] == 1
        assert stats["history_size"] == 1

    def test_record_empty_sequence(self):
        """Test recording an empty sequence (no-op)."""
        tracker = CooccurrenceTracker()

        tracker.record_tool_sequence(tools=[], task_type="bugfix", success=True)

        stats = tracker.get_statistics()
        assert stats["total_sequences_recorded"] == 0

    def test_record_multiple_sequences(self):
        """Test recording multiple sequences."""
        tracker = CooccurrenceTracker()

        tracker.record_tool_sequence(["search", "read"], "bugfix", True)
        tracker.record_tool_sequence(["edit", "write"], "bugfix", True)
        tracker.record_tool_sequence(["test", "verify"], "testing", True)

        stats = tracker.get_statistics()
        assert stats["total_sequences_recorded"] == 3

    def test_record_sequence_with_duplicates(self):
        """Test recording sequences with repeated tools."""
        tracker = CooccurrenceTracker()

        tracker.record_tool_sequence(["search", "read", "search"], "bugfix", True)

        # Should record both bigrams: search→read and read→search
        stats = tracker.get_statistics()
        assert stats["total_bigrams"] == 2

    def test_task_type_specificity_enabled(self):
        """Test task-specific patterns when enabled."""
        tracker = CooccurrenceTracker(enable_task_type_specificity=True)

        tracker.record_tool_sequence(["search", "read"], "bugfix", True)
        tracker.record_tool_sequence(["search", "read"], "feature", True)

        stats = tracker.get_statistics()
        assert "bugfix" in stats["task_types"]
        assert "feature" in stats["task_types"]

    def test_task_type_specificity_disabled(self):
        """Test merged patterns when task-specificity disabled."""
        tracker = CooccurrenceTracker(enable_task_type_specificity=False)

        tracker.record_tool_sequence(["search", "read"], "bugfix", True)
        tracker.record_tool_sequence(["search", "read"], "feature", True)

        stats = tracker.get_statistics()
        # Should only have "default" task type
        assert stats["task_types"] == ["default"]


class TestCooccurrenceMatrix:
    """Test co-occurrence matrix updates."""

    def test_cooccurrence_matrix_update(self):
        """Test that co-occurrence matrix is updated correctly."""
        tracker = CooccurrenceTracker()

        tracker.record_tool_sequence(["search", "read", "edit"], "bugfix", True)

        # Check that search co-occurs with read and edit
        matrix = tracker._cooccurrence_matrices["bugfix"]
        assert "search" in matrix
        assert "read" in matrix["search"]
        assert "edit" in matrix["search"]

    def test_cooccurrence_counts(self):
        """Test co-occurrence count accumulation."""
        tracker = CooccurrenceTracker()

        # Record search→read twice
        tracker.record_tool_sequence(["search", "read"], "bugfix", True)
        tracker.record_tool_sequence(["search", "read"], "bugfix", True)

        matrix = tracker._cooccurrence_matrices["bugfix"]
        assert matrix["search"]["read"] == 2

    def test_backward_cooccurrence(self):
        """Test that all tools in sequence are added to matrix as keys."""
        tracker = CooccurrenceTracker()

        tracker.record_tool_sequence(["search", "read"], "bugfix", True)

        # Both "search" and "read" should be in the matrix as keys
        matrix = tracker._cooccurrence_matrices["bugfix"]
        assert "read" in matrix
        # But read→search is not tracked (only forward direction: search→read)
        assert "search" not in matrix["read"] or matrix["read"]["search"] == 0


class TestSequentialPatterns:
    """Test sequential pattern extraction."""

    def test_bigram_extraction(self):
        """Test bigram pattern extraction."""
        tracker = CooccurrenceTracker()

        tracker.record_tool_sequence(["search", "read", "edit"], "bugfix", True)

        bigrams = tracker._bigrams["bugfix"]
        assert len(bigrams) == 2

        # Should have: search→read and read→edit
        sequences = [p.sequence for p in bigrams]
        assert ["search", "read"] in sequences
        assert ["read", "edit"] in sequences

    def test_trigram_extraction(self):
        """Test trigram pattern extraction."""
        tracker = CooccurrenceTracker()

        tracker.record_tool_sequence(["search", "read", "edit"], "bugfix", True)

        trigrams = tracker._trigrams["bugfix"]
        assert len(trigrams) == 1

        # Should have: search→read→edit
        assert trigrams[0].sequence == ["search", "read", "edit"]

    def test_pattern_support_update(self):
        """Test that pattern support increases with repeated observations."""
        tracker = CooccurrenceTracker()

        tracker.record_tool_sequence(["search", "read"], "bugfix", True)
        tracker.record_tool_sequence(["search", "read"], "bugfix", True)

        bigrams = tracker._bigrams["bugfix"]
        pattern = [p for p in bigrams if p.sequence == ["search", "read"]][0]

        assert pattern.support == 2

    def test_pattern_success_rate(self):
        """Test pattern success rate tracking."""
        tracker = CooccurrenceTracker()

        # Record successful sequence
        tracker.record_tool_sequence(["search", "read"], "bugfix", success=True)
        # Record failed sequence
        tracker.record_tool_sequence(["search", "read"], "bugfix", success=False)
        # Record successful sequence
        tracker.record_tool_sequence(["search", "read"], "bugfix", success=True)

        bigrams = tracker._bigrams["bugfix"]
        pattern = [p for p in bigrams if p.sequence == ["search", "read"]][0]

        # Success rate = 2/3 = 0.667
        assert pattern.success_rate == pytest.approx(0.667, rel=0.01)


class TestToolSuccessRates:
    """Test tool success rate tracking."""

    def test_tool_success_rate_tracking(self):
        """Test that tool success rates are tracked."""
        tracker = CooccurrenceTracker()

        tracker.record_tool_sequence(["search", "read"], "bugfix", success=True)
        tracker.record_tool_sequence(["search", "edit"], "bugfix", success=False)
        tracker.record_tool_sequence(["search", "read"], "bugfix", success=True)

        # search: 2/3 = 0.667 (used in 3 sequences, 2 successful)
        search_success = tracker._get_tool_success_rate("search", "bugfix")
        assert search_success == pytest.approx(0.667, rel=0.01)

    def test_unobserved_tool_default_rate(self):
        """Test that unobserved tools get default success rate."""
        tracker = CooccurrenceTracker()

        success_rate = tracker._get_tool_success_rate("unknown_tool", "bugfix")

        assert success_rate == 0.5


class TestNextToolPrediction:
    """Test next tool prediction."""

    def test_predict_from_empty_history(self):
        """Test prediction with no current tools."""
        tracker = CooccurrenceTracker()

        predictions = tracker.predict_next_tools(
            current_tools=[],
            task_type="bugfix",
        )

        assert predictions == []

    def test_predict_from_cooccurrence(self):
        """Test prediction based on co-occurrence."""
        tracker = CooccurrenceTracker()

        # Train: search is always followed by read
        tracker.record_tool_sequence(["search", "read"], "bugfix", True)
        tracker.record_tool_sequence(["search", "read"], "bugfix", True)

        predictions = tracker.predict_next_tools(
            current_tools=["search"],
            task_type="bugfix",
        )

        assert len(predictions) > 0
        assert predictions[0].tool_name == "read"
        assert predictions[0].probability > 0.5

    def test_predict_from_bigram(self):
        """Test prediction using bigram patterns."""
        tracker = CooccurrenceTracker()

        # Train bigram: search→read→edit
        tracker.record_tool_sequence(["search", "read", "edit"], "bugfix", True)
        tracker.record_tool_sequence(["search", "read", "edit"], "bugfix", True)

        predictions = tracker.predict_next_tools(
            current_tools=["search", "read"],
            task_type="bugfix",
        )

        # Should predict "edit" with high confidence
        assert predictions[0].tool_name == "edit"

    def test_predict_from_trigram(self):
        """Test prediction using trigram patterns."""
        tracker = CooccurrenceTracker()

        # Train trigram: search→read→edit→write
        tracker.record_tool_sequence(["search", "read", "edit", "write"], "bugfix", True)
        tracker.record_tool_sequence(["search", "read", "edit", "write"], "bugfix", True)

        predictions = tracker.predict_next_tools(
            current_tools=["search", "read", "edit"],
            task_type="bugfix",
        )

        # Should predict "write"
        assert predictions[0].tool_name == "write"

    def test_top_k_limiting(self):
        """Test that top_k limits predictions."""
        tracker = CooccurrenceTracker()

        # Train multiple next tools
        tracker.record_tool_sequence(["search", "read"], "bugfix", True)
        tracker.record_tool_sequence(["search", "edit"], "bugfix", True)
        tracker.record_tool_sequence(["search", "write"], "bugfix", True)
        tracker.record_tool_sequence(["search", "test"], "bugfix", True)

        predictions = tracker.predict_next_tools(
            current_tools=["search"],
            task_type="bugfix",
            top_k=2,
        )

        assert len(predictions) == 2

    def test_predictions_sorted_by_probability(self):
        """Test that predictions are sorted by probability."""
        tracker = CooccurrenceTracker()

        # Train with different frequencies
        tracker.record_tool_sequence(["search", "read"], "bugfix", True)
        tracker.record_tool_sequence(["search", "read"], "bugfix", True)
        tracker.record_tool_sequence(["search", "read"], "bugfix", True)
        tracker.record_tool_sequence(["search", "edit"], "bugfix", True)

        predictions = tracker.predict_next_tools(
            current_tools=["search"],
            task_type="bugfix",
        )

        # "read" should be first (higher probability)
        assert predictions[0].tool_name == "read"
        probabilities = [p.probability for p in predictions]
        assert probabilities == sorted(probabilities, reverse=True)


class TestSuccessRateBoosting:
    """Test success rate boosting in predictions."""

    def test_high_success_rate_boosting(self):
        """Test that high success tools get boosted."""
        tracker = CooccurrenceTracker()

        # Tool "read" has high success rate
        for _ in range(10):
            tracker.record_tool_sequence(["search", "read"], "bugfix", success=True)

        # Tool "edit" has low success rate
        for _ in range(10):
            tracker.record_tool_sequence(["search", "edit"], "bugfix", success=False)

        predictions = tracker.predict_next_tools(
            current_tools=["search"],
            task_type="bugfix",
        )

        # "read" should be ranked higher due to success boosting
        assert predictions[0].tool_name == "read"
        assert predictions[0].success_rate > 0.8


class TestDecayAndHistoryManagement:
    """Test history management and decay."""

    def test_history_limit(self):
        """Test that history respects max_history limit."""
        tracker = CooccurrenceTracker(max_history=5)

        # Add more sequences than max_history
        for i in range(10):
            tracker.record_tool_sequence([f"tool_{i}"], "bugfix", True)

        stats = tracker.get_statistics()
        # History should be limited (may have some decay)
        assert stats["history_size"] <= tracker.max_history

    def test_decay_reduces_old_counts(self):
        """Test that decay reduces old pattern counts."""
        tracker = CooccurrenceTracker(max_history=10, decay_factor=0.5)

        # Fill history
        for i in range(10):
            tracker.record_tool_sequence(["search", "read"], "bugfix", True)

        # Get initial count
        initial_count = tracker._cooccurrence_matrices["bugfix"]["search"]["read"]

        # Add more sequences to trigger decay
        for i in range(10):
            tracker.record_tool_sequence([f"tool_{i}", f"tool_{i+1}"], "bugfix", True)

        # Count should be reduced by decay
        current_count = tracker._cooccurrence_matrices["bugfix"]["search"]["read"]
        assert current_count < initial_count


class TestStatisticsAndReset:
    """Test statistics and reset functionality."""

    def test_statistics_accuracy(self):
        """Test that statistics are accurate."""
        tracker = CooccurrenceTracker()

        tracker.record_tool_sequence(["search", "read", "edit"], "bugfix", True)
        tracker.record_tool_sequence(["test", "verify"], "testing", True)

        stats = tracker.get_statistics()

        assert stats["total_sequences_recorded"] == 2
        assert stats["history_size"] == 2
        assert stats["unique_tools"] == 5  # search, read, edit, test, verify
        assert stats["total_bigrams"] == 3  # search→read, read→edit, test→verify
        assert stats["total_trigrams"] == 1  # search→read→edit

    def test_reset_clears_all_data(self):
        """Test that reset clears all tracking data."""
        tracker = CooccurrenceTracker()

        tracker.record_tool_sequence(["search", "read"], "bugfix", True)
        tracker.record_tool_sequence(["edit", "write"], "feature", True)

        tracker.reset()

        stats = tracker.get_statistics()
        assert stats["total_sequences_recorded"] == 0
        assert stats["history_size"] == 0
        assert stats["unique_tools"] == 0

    def test_task_type_in_statistics(self):
        """Test that task types are tracked in statistics."""
        tracker = CooccurrenceTracker()

        tracker.record_tool_sequence(["search", "read"], "bugfix", True)
        tracker.record_tool_sequence(["edit", "write"], "feature", True)

        stats = tracker.get_statistics()
        assert "bugfix" in stats["task_types"]
        assert "feature" in stats["task_types"]


class TestToolPattern:
    """Test ToolPattern dataclass."""

    def test_pattern_validation(self):
        """Test pattern validation."""
        pattern = ToolPattern(
            sequence=["search", "read"],
            support=5,
            confidence=0.8,
            success_rate=0.9,
        )

        assert pattern.sequence == ["search", "read"]
        assert pattern.support == 5
        assert pattern.confidence == 0.8
        assert pattern.success_rate == 0.9

    def test_invalid_support(self):
        """Test that invalid support raises error."""
        with pytest.raises(ValueError, match="support cannot be negative"):
            ToolPattern(
                sequence=["search", "read"],
                support=-1,
                confidence=0.8,
                success_rate=0.9,
            )

    def test_invalid_confidence(self):
        """Test that invalid confidence raises error."""
        with pytest.raises(ValueError, match="Confidence must be in \\[0, 1\\]"):
            ToolPattern(
                sequence=["search", "read"],
                support=5,
                confidence=1.5,
                success_rate=0.9,
            )

    def test_invalid_success_rate(self):
        """Test that invalid success rate raises error."""
        with pytest.raises(ValueError, match="Success rate must be in \\[0, 1\\]"):
            ToolPattern(
                sequence=["search", "read"],
                support=5,
                confidence=0.8,
                success_rate=1.5,
            )


class TestToolPrediction:
    """Test ToolPrediction dataclass."""

    def test_prediction_creation(self):
        """Test prediction creation."""
        prediction = ToolPrediction(
            tool_name="read",
            probability=0.85,
            pattern_source="bigram",
            success_rate=0.9,
        )

        assert prediction.tool_name == "read"
        assert prediction.probability == 0.85
        assert prediction.pattern_source == "bigram"
        assert prediction.success_rate == 0.9

    def test_invalid_probability(self):
        """Test that invalid probability raises error."""
        with pytest.raises(ValueError, match="Probability must be in \\[0, 1\\]"):
            ToolPrediction(
                tool_name="read",
                probability=1.5,
                pattern_source="bigram",
                success_rate=0.9,
            )


class TestIntegration:
    """Integration tests for co-occurrence tracker."""

    def test_full_prediction_workflow(self):
        """Test complete workflow from recording to prediction."""
        tracker = CooccurrenceTracker()

        # Train with typical bugfix workflow
        tracker.record_tool_sequence(["search", "read", "edit", "test"], "bugfix", True)
        tracker.record_tool_sequence(["search", "read", "edit", "test"], "bugfix", True)
        tracker.record_tool_sequence(["search", "read", "edit", "test"], "bugfix", True)

        # Predict after "search"
        predictions = tracker.predict_next_tools(
            current_tools=["search"],
            task_type="bugfix",
        )

        assert len(predictions) > 0
        assert predictions[0].tool_name == "read"
        assert predictions[0].probability > 0.5

        # Predict after "search", "read"
        predictions = tracker.predict_next_tools(
            current_tools=["search", "read"],
            task_type="bugfix",
        )

        assert predictions[0].tool_name == "edit"

    def test_task_type_specific_predictions(self):
        """Test predictions are task-type specific."""
        tracker = CooccurrenceTracker()

        # Bugfix workflow: search → read → edit
        for _ in range(5):
            tracker.record_tool_sequence(["search", "read", "edit"], "bugfix", True)

        # Feature workflow: plan → design → implement
        for _ in range(5):
            tracker.record_tool_sequence(["plan", "design", "implement"], "feature", True)

        # Predict for bugfix
        bugfix_predictions = tracker.predict_next_tools(
            current_tools=["search"],
            task_type="bugfix",
        )

        # Should predict "read"
        assert bugfix_predictions[0].tool_name == "read"

        # Predict for feature
        feature_predictions = tracker.predict_next_tools(
            current_tools=["plan"],
            task_type="feature",
        )

        # Should predict "design"
        assert feature_predictions[0].tool_name == "design"

    def test_learning_from_outcomes(self):
        """Test that predictions improve with more data."""
        tracker = CooccurrenceTracker()

        # Initial training
        tracker.record_tool_sequence(["search", "read"], "bugfix", True)
        predictions_v1 = tracker.predict_next_tools(
            current_tools=["search"],
            task_type="bugfix",
        )

        # More training with same pattern
        for _ in range(10):
            tracker.record_tool_sequence(["search", "read"], "bugfix", True)

        predictions_v2 = tracker.predict_next_tools(
            current_tools=["search"],
            task_type="bugfix",
        )

        # Confidence should increase with more data
        assert predictions_v2[0].probability >= predictions_v1[0].probability
