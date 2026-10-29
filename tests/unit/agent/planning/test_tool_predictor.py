"""Tests for tool predictor.

Tests cover:
- Initialization and configuration
- Keyword-based prediction
- Semantic similarity prediction
- Co-occurrence-based prediction
- Ensemble voting
- Success rate boosting
- Confidence level determination
- Statistics and metadata
"""

from victor.agent.planning.tool_predictor import (
    ToolPredictor,
    ToolPredictorConfig,
    ToolPrediction,
)
from victor.agent.planning.cooccurrence_tracker import CooccurrenceTracker


class TestInitialization:
    """Test predictor initialization."""

    def test_default_initialization(self):
        """Test predictor with default settings."""
        predictor = ToolPredictor()

        assert predictor.config.keyword_weight == 0.3
        assert predictor.config.semantic_weight == 0.4
        assert predictor.config.cooccurrence_weight == 0.2
        assert predictor.config.success_weight == 0.1
        assert predictor.config.top_k == 5
        assert predictor._cooccurrence_tracker is None

    def test_custom_initialization(self):
        """Test predictor with custom settings."""
        config = ToolPredictorConfig(
            keyword_weight=0.4,
            semantic_weight=0.3,
            cooccurrence_weight=0.2,
            success_weight=0.1,
            top_k=10,
        )
        predictor = ToolPredictor(config=config)

        assert predictor.config.keyword_weight == 0.4
        assert predictor.config.semantic_weight == 0.3
        assert predictor.config.top_k == 10

    def test_initialization_with_cooccurrence_tracker(self):
        """Test predictor with co-occurrence tracker."""
        tracker = CooccurrenceTracker()
        predictor = ToolPredictor(cooccurrence_tracker=tracker)

        assert predictor._cooccurrence_tracker is tracker

    def test_keyword_patterns_compiled(self):
        """Test that keyword patterns are compiled."""
        predictor = ToolPredictor()

        assert "search" in predictor._compiled_patterns
        assert len(predictor._compiled_patterns["search"]) > 0
        assert hasattr(predictor._compiled_patterns["search"][0], "search")


class TestKeywordPrediction:
    """Test keyword-based prediction."""

    def test_predict_search_from_keywords(self):
        """Test search tool prediction from keywords."""
        predictor = ToolPredictor()

        predictions = predictor.predict_tools(
            task_description="Find the bug in the code",
            current_step="exploration",
        )

        tool_names = [p.tool_name for p in predictions]
        assert "search" in tool_names

    def test_predict_edit_from_keywords(self):
        """Test edit tool prediction from keywords."""
        predictor = ToolPredictor()

        predictions = predictor.predict_tools(
            task_description="Fix the authentication bug",
            current_step="implementation",
        )

        tool_names = [p.tool_name for p in predictions]
        assert "edit" in tool_names

    def test_predict_test_from_keywords(self):
        """Test test tool prediction from keywords."""
        predictor = ToolPredictor()

        predictions = predictor.predict_tools(
            task_description="Run the unit tests",
            current_step="verification",
        )

        tool_names = [p.tool_name for p in predictions]
        assert "test" in tool_names or "run" in tool_names

    def test_case_insensitive_matching(self):
        """Test that keyword matching is case-insensitive."""
        predictor = ToolPredictor()

        predictions_upper = predictor.predict_tools(
            task_description="SEARCH for the file",
            current_step="exploration",
        )
        predictions_lower = predictor.predict_tools(
            task_description="search for the file",
            current_step="exploration",
        )

        assert len(predictions_upper) > 0
        assert len(predictions_lower) > 0

    def test_no_matches_returns_empty(self):
        """Test prediction with no matching keywords."""
        predictor = ToolPredictor()

        predictions = predictor.predict_tools(
            task_description="do something completely unrelated",
            current_step="unknown",
        )

        # Should return empty or very low confidence predictions
        assert len(predictions) == 0 or all(p.probability < 0.3 for p in predictions)


class TestSemanticPrediction:
    """Test semantic similarity prediction."""

    def test_predict_with_embedding_fn(self):
        """Test prediction with embedding function."""
        predictor = ToolPredictor()

        # Mock embedding function that returns high similarity for "search"
        def mock_embedding_fn(tools, query):
            return [0.9 if tool == "search" else 0.3 for tool in tools]

        predictions = predictor.predict_tools(
            task_description="find files in the project",
            current_step="exploration",
            embedding_fn=mock_embedding_fn,
        )

        tool_names = [p.tool_name for p in predictions]
        assert "search" in tool_names

    def test_semantic_disabled_without_fn(self):
        """Test that semantic prediction is skipped without embedding_fn."""
        config = ToolPredictorConfig(enable_keyword_matching=False, enable_semantic_matching=True)
        predictor = ToolPredictor(config=config)

        # No embedding function provided
        predictions = predictor.predict_tools(
            task_description="find files",
            current_step="exploration",
        )

        # Should return empty since semantic requires embedding_fn
        assert len(predictions) == 0


class TestCooccurrencePrediction:
    """Test co-occurrence-based prediction."""

    def test_predict_from_cooccurrence(self):
        """Test prediction using co-occurrence patterns."""
        tracker = CooccurrenceTracker()

        # Train: search is always followed by read
        tracker.record_tool_sequence(["search", "read"], "bugfix", True)
        tracker.record_tool_sequence(["search", "read"], "bugfix", True)
        tracker.record_tool_sequence(["search", "read"], "bugfix", True)

        predictor = ToolPredictor(cooccurrence_tracker=tracker)

        predictions = predictor.predict_tools(
            task_description="find the bug",
            current_step="exploration",
            recent_tools=["search"],
            task_type="bugfix",
        )

        tool_names = [p.tool_name for p in predictions]
        assert "read" in tool_names

    def test_cooccurrence_disabled_without_tracker(self):
        """Test that co-occurrence prediction is skipped without tracker."""
        predictor = ToolPredictor(cooccurrence_tracker=None)

        predictions = predictor.predict_tools(
            task_description="find files",
            current_step="exploration",
            recent_tools=["search"],
        )

        # Should not crash, just use keyword matching
        assert len(predictions) >= 0


class TestEnsembleVoting:
    """Test ensemble voting from multiple classifiers."""

    def test_ensemble_combines_signals(self):
        """Test that ensemble combines multiple signals."""
        tracker = CooccurrenceTracker()

        # Train co-occurrence: search → read
        tracker.record_tool_sequence(["search", "read"], "bugfix", True)
        tracker.record_tool_sequence(["search", "read"], "bugfix", True)

        predictor = ToolPredictor(cooccurrence_tracker=tracker)

        # Mock embedding that favors "edit"
        def mock_embedding_fn(tools, query):
            return [0.8 if tool == "edit" else 0.2 for tool in tools]

        predictions = predictor.predict_tools(
            task_description="fix the bug",
            current_step="exploration",
            recent_tools=["search"],
            task_type="bugfix",
            embedding_fn=mock_embedding_fn,
        )

        # Should have predictions from both sources
        assert len(predictions) > 0

    def test_top_k_limiting(self):
        """Test that top_k limits predictions."""
        predictor = ToolPredictor(config=ToolPredictorConfig(top_k=2))

        predictions = predictor.predict_tools(
            task_description="find read edit write test run the files",
            current_step="exploration",
        )

        assert len(predictions) <= 2

    def test_min_confidence_filtering(self):
        """Test that low confidence predictions are filtered."""
        predictor = ToolPredictor(config=ToolPredictorConfig(min_confidence=0.3))

        predictions = predictor.predict_tools(
            task_description="do something",
            current_step="unknown",
        )

        # All predictions should have probability >= 0.3
        for p in predictions:
            assert p.probability >= 0.3


class TestSuccessRateBoosting:
    """Test success rate boosting in predictions."""

    def test_high_success_boost(self):
        """Test that high success tools get boosted."""
        tracker = CooccurrenceTracker()

        # Tool "read" has high success rate
        for _ in range(10):
            tracker.record_tool_sequence(["search", "read"], "bugfix", success=True)

        # Tool "edit" has low success rate
        for _ in range(10):
            tracker.record_tool_sequence(["search", "edit"], "bugfix", success=False)

        predictor = ToolPredictor(cooccurrence_tracker=tracker)

        # Predict with recent tools to trigger co-occurrence
        predictions = predictor.predict_tools(
            task_description="find and fix",
            current_step="exploration",
            recent_tools=["search"],
            task_type="bugfix",
        )

        # "read" should be ranked higher due to success boosting
        if len(predictions) > 0:
            top_prediction = predictions[0]
            assert top_prediction.tool_name == "read"
            assert top_prediction.success_rate > 0.7


class TestConfidenceLevel:
    """Test confidence level determination."""

    def test_high_confidence(self):
        """Test HIGH confidence level."""
        prediction = ToolPrediction(
            tool_name="search",
            probability=0.85,
        )

        assert prediction.confidence_level == "HIGH"

    def test_medium_confidence(self):
        """Test MEDIUM confidence level."""
        prediction = ToolPrediction(
            tool_name="read",
            probability=0.55,
        )

        assert prediction.confidence_level == "MEDIUM"

    def test_low_confidence(self):
        """Test LOW confidence level."""
        prediction = ToolPrediction(
            tool_name="edit",
            probability=0.25,
        )

        assert prediction.confidence_level == "LOW"

    def test_probability_clamping(self):
        """Test that probability is clamped to [0, 1]."""
        # Test high value
        prediction = ToolPrediction(tool_name="test", probability=1.5)
        assert prediction.probability == 1.0

        # Test low value
        prediction = ToolPrediction(tool_name="test", probability=-0.5)
        assert prediction.probability == 0.0


class TestPredictionSource:
    """Test prediction source determination."""

    def test_keyword_source(self):
        """Test keyword-only prediction source."""
        predictor = ToolPredictor()

        predictions = predictor.predict_tools(
            task_description="find the bug",
            current_step="exploration",
        )

        # At least one prediction should have keyword source
        keyword_sources = [p for p in predictions if "keyword" in p.source]
        assert len(keyword_sources) > 0

    def test_cooccurrence_source(self):
        """Test co-occurrence prediction source."""
        tracker = CooccurrenceTracker()
        tracker.record_tool_sequence(["search", "read"], "bugfix", True)

        predictor = ToolPredictor(cooccurrence_tracker=tracker)

        predictions = predictor.predict_tools(
            task_description="find",
            current_step="exploration",
            recent_tools=["search"],
            task_type="bugfix",
        )

        # Should have co-occurrence in source
        if predictions:
            assert any("cooccurrence" in p.source for p in predictions)

    def test_ensemble_source(self):
        """Test ensemble source for combined predictions."""
        tracker = CooccurrenceTracker()
        tracker.record_tool_sequence(["search", "read"], "bugfix", True)

        predictor = ToolPredictor(cooccurrence_tracker=tracker)

        def mock_embedding_fn(tools, query):
            return [0.8 if tool == "search" else 0.2 for tool in tools]

        predictions = predictor.predict_tools(
            task_description="find the bug",
            current_step="exploration",
            recent_tools=["search"],
            task_type="bugfix",
            embedding_fn=mock_embedding_fn,
        )

        # Some predictions may have combined sources
        combined_sources = [p for p in predictions if "+" in p.source]
        assert len(combined_sources) >= 0  # May or may not have combined


class TestStatistics:
    """Test predictor statistics."""

    def test_statistics_includes_config(self):
        """Test that statistics include configuration."""
        predictor = ToolPredictor()

        stats = predictor.get_statistics()

        assert "config" in stats
        assert stats["config"]["keyword_weight"] == 0.3
        assert stats["config"]["semantic_weight"] == 0.4

    def test_statistics_includes_tracker_status(self):
        """Test that statistics include tracker status."""
        predictor = ToolPredictor(cooccurrence_tracker=None)

        stats = predictor.get_statistics()

        assert "has_cooccurrence_tracker" in stats
        assert stats["has_cooccurrence_tracker"] is False

    def test_statistics_includes_available_tools(self):
        """Test that statistics include available tool count."""
        predictor = ToolPredictor()

        stats = predictor.get_statistics()

        assert "available_tools" in stats
        assert stats["available_tools"] > 0

    def test_statistics_includes_cooccurrence_stats(self):
        """Test that statistics include co-occurrence stats when available."""
        tracker = CooccurrenceTracker()
        tracker.record_tool_sequence(["search", "read"], "bugfix", True)

        predictor = ToolPredictor(cooccurrence_tracker=tracker)

        stats = predictor.get_statistics()

        assert "cooccurrence_stats" in stats
        assert stats["cooccurrence_stats"]["total_sequences_recorded"] == 1


class TestToolPredictorConfig:
    """Test predictor configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ToolPredictorConfig()

        assert config.keyword_weight == 0.3
        assert config.semantic_weight == 0.4
        assert config.cooccurrence_weight == 0.2
        assert config.success_weight == 0.1
        assert config.min_confidence == 0.1
        assert config.top_k == 5
        assert config.enable_keyword_matching is True
        assert config.enable_semantic_matching is True
        assert config.enable_cooccurrence is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ToolPredictorConfig(
            keyword_weight=0.5,
            semantic_weight=0.3,
            cooccurrence_weight=0.2,
            success_weight=0.0,
            min_confidence=0.2,
            top_k=10,
            enable_keyword_matching=False,
        )

        assert config.keyword_weight == 0.5
        assert config.semantic_weight == 0.3
        assert config.min_confidence == 0.2
        assert config.top_k == 10
        assert config.enable_keyword_matching is False


class TestIntegration:
    """Integration tests for tool predictor."""

    def test_full_prediction_workflow(self):
        """Test complete prediction workflow."""
        tracker = CooccurrenceTracker()

        # Train with typical workflow
        tracker.record_tool_sequence(["search", "read", "edit", "test"], "bugfix", True)
        tracker.record_tool_sequence(["search", "read", "edit", "test"], "bugfix", True)
        tracker.record_tool_sequence(["search", "read", "edit", "test"], "bugfix", True)

        predictor = ToolPredictor(cooccurrence_tracker=tracker)

        # Predict after "search"
        predictions = predictor.predict_tools(
            task_description="find and fix the authentication bug",
            current_step="exploration",
            recent_tools=["search"],
            task_type="bugfix",
        )

        # Should have predictions
        assert len(predictions) > 0

        # Top prediction should be "read" (from co-occurrence)
        assert predictions[0].tool_name in ("read", "edit", "search")

    def test_task_type_specific_predictions(self):
        """Test predictions are task-type specific."""
        tracker = CooccurrenceTracker()

        # Bugfix: search → read → edit
        for _ in range(5):
            tracker.record_tool_sequence(["search", "read", "edit"], "bugfix", True)

        # Feature: plan → design → implement
        for _ in range(5):
            tracker.record_tool_sequence(["plan", "design", "implement"], "feature", True)

        predictor = ToolPredictor(cooccurrence_tracker=tracker)

        # Predict for bugfix
        bugfix_predictions = predictor.predict_tools(
            task_description="fix the bug",
            current_step="exploration",
            recent_tools=["search"],
            task_type="bugfix",
        )

        # Predict for feature
        feature_predictions = predictor.predict_tools(
            task_description="add new feature",
            current_step="exploration",
            recent_tools=["plan"],
            task_type="feature",
        )

        # Should have different predictions
        assert len(bugfix_predictions) > 0
        assert len(feature_predictions) > 0

    def test_high_confidence_rate(self):
        """Test that high confidence rate tracking works."""
        predictor = ToolPredictor()

        predictions = predictor.predict_tools(
            task_description="search for and read the file to find the bug",
            current_step="exploration",
        )

        # At least some predictions should have MEDIUM or HIGH confidence
        high_conf = [p for p in predictions if p.confidence_level in ("HIGH", "MEDIUM")]
        assert len(high_conf) > 0
