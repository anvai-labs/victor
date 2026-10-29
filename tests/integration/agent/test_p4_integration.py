"""Integration tests for P4 Multi-Provider Excellence features.

This module tests the integration of:
1. Hybrid Search (semantic + keyword with RRF)
2. RL-based threshold learning for semantic search
3. Tool call deduplication tracker
4. Provider-specific tool guidance

These tests verify that all components work together correctly.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from victor.config.settings import Settings
from victor.core.schema import Tables
from victor.framework.search import HybridSearchEngine, create_hybrid_search_engine
from victor.framework.rl.learners.semantic_threshold import SemanticThresholdLearner
from victor.agent.tool_call_tracker import ToolCallTracker as ToolDeduplicationTracker
from victor.agent.tool_pipeline import ToolPipeline, ToolPipelineConfig
from victor.agent.tool_executor import ToolExecutor
from victor.tools.registry import ToolRegistry


class TestHybridSearchIntegration:
    """Test hybrid search combining semantic and keyword results."""

    def test_hybrid_search_combines_results(self):
        """Test that hybrid search combines semantic and keyword results using RRF."""
        # Arrange
        semantic_results = [
            {"file_path": "foo.py", "content": "def foo():", "score": 0.92},
            {"file_path": "bar.py", "content": "def bar():", "score": 0.85},
        ]
        keyword_results = [
            {"file_path": "bar.py", "content": "def bar():", "score": 15.0},
            {"file_path": "baz.py", "content": "def baz():", "score": 10.0},
        ]

        engine = create_hybrid_search_engine(semantic_weight=0.6, keyword_weight=0.4)

        # Act
        results = engine.combine_results(semantic_results, keyword_results, max_results=10)

        # Assert
        assert len(results) > 0
        # bar.py should rank high (in both semantic and keyword)
        file_paths = [r.file_path for r in results]
        assert "bar.py" in file_paths
        # Check that results have combined scores
        for result in results:
            assert result.combined_score > 0
            assert result.semantic_score >= 0
            assert result.keyword_score >= 0

    def test_hybrid_search_weights_normalization(self):
        """Test that hybrid search normalizes weights to sum to 1.0."""
        # Arrange
        engine = create_hybrid_search_engine(semantic_weight=3.0, keyword_weight=2.0)

        # Assert - weights should be normalized to 0.6 and 0.4
        assert abs(engine.semantic_weight - 0.6) < 0.01
        assert abs(engine.keyword_weight - 0.4) < 0.01

    def test_hybrid_search_keyword_only_fallback(self):
        """Test hybrid search with only keyword results (semantic failed)."""
        # Arrange
        semantic_results = []
        keyword_results = [
            {"file_path": "foo.py", "content": "def foo():", "score": 10.0},
        ]

        engine = create_hybrid_search_engine()

        # Act
        results = engine.combine_results(semantic_results, keyword_results, max_results=10)

        # Assert
        assert len(results) == 1
        assert results[0].file_path == "foo.py"
        assert results[0].keyword_score > 0
        assert results[0].semantic_score == 0


class TestThresholdLearnerIntegration:
    """Test RL-based threshold learning integration."""

    def test_threshold_learner_records_outcomes(self):
        """Test that threshold learner records search outcomes."""
        # Arrange - use a clean learner with no persisted state
        import sqlite3
        from victor.framework.rl.base import RLOutcome

        # Create an in-memory database for testing
        db = sqlite3.connect(":memory:")

        # Create learner with db connection
        learner = SemanticThresholdLearner(name="semantic_threshold", db_connection=db)
        embedding_model = "bge-small"
        task_type = "search"
        tool_name = "code_search"

        # Act - record a few outcomes
        learner.record_outcome(
            RLOutcome(
                provider=embedding_model,  # embedding_model is passed as provider for this learner
                model=tool_name,  # tool_name is passed as model for this learner
                task_type=task_type,
                success=True,
                quality_score=0.8,
                metadata={
                    "embedding_model": embedding_model,
                    "tool_name": tool_name,
                    "query": "tool registration",
                    "results_count": 5,
                    "threshold_used": 0.5,
                    "false_negatives": False,
                },
            )
        )

        learner.record_outcome(
            RLOutcome(
                provider=embedding_model,
                model=tool_name,
                task_type=task_type,
                success=False,
                quality_score=0.2,
                metadata={
                    "embedding_model": embedding_model,
                    "tool_name": tool_name,
                    "query": "error handling",
                    "results_count": 0,
                    "threshold_used": 0.5,
                    "false_negatives": True,
                },
            )
        )

        # Assert
        # Check database directly via the canonical unified RL stats table.
        cursor = db.cursor()
        cursor.execute(
            f"SELECT * FROM {Tables.RL_TASK_STAT} WHERE learner_id = ?",
            ("semantic_threshold",),
        )
        rows = cursor.fetchall()
        assert len(rows) > 0

        db.close()

    def test_threshold_learner_recommends_adjustment(self):
        """Test that learner recommends threshold adjustment after sufficient data."""
        # Arrange
        import sqlite3
        from victor.framework.rl.base import RLOutcome

        db = sqlite3.connect(":memory:")
        learner = SemanticThresholdLearner(name="semantic_threshold", db_connection=db)

        embedding_model = "bge-small"
        task_type = "search"
        tool_name = "code_search"

        # Act - record several zero-result searches (high false negative rate)
        for _ in range(10):
            learner.record_outcome(
                RLOutcome(
                    provider=embedding_model,
                    model=tool_name,
                    task_type=task_type,
                    success=False,
                    quality_score=0.1,
                    metadata={
                        "embedding_model": embedding_model,
                        "tool_name": tool_name,
                        "query": "test query",
                        "results_count": 0,
                        "threshold_used": 0.7,
                        "false_negatives": True,
                    },
                )
            )

        # Assert - should recommend lowering threshold
        recommendation = learner.get_recommendation(
            provider=embedding_model, model=tool_name, task_type=task_type
        )
        assert recommendation is not None
        assert recommendation.value < 0.7  # Should recommend lower threshold

        db.close()

    def test_threshold_learner_get_recommendations(self):
        """Test that learner provides recommendations after sufficient data."""
        # Arrange - use a clean learner with no persisted state
        import sqlite3
        from victor.framework.rl.base import RLOutcome

        db = sqlite3.connect(":memory:")
        learner = SemanticThresholdLearner(name="semantic_threshold", db_connection=db)

        embedding_model = "bge-small"
        task_type = "search"
        tool_name = "code_search"

        # Record multiple outcomes with high false negative rate
        for i in range(10):
            learner.record_outcome(
                RLOutcome(
                    provider=embedding_model,
                    model=tool_name,
                    task_type=task_type,
                    success=(i >= 6),
                    quality_score=0.8 if i >= 6 else 0.1,
                    metadata={
                        "embedding_model": embedding_model,
                        "tool_name": tool_name,
                        "query": f"test query {i}",
                        "results_count": 0 if i < 6 else 5,  # 60% false negatives
                        "threshold_used": 0.7,
                        "false_negatives": (i < 6),
                    },
                )
            )

        # Act - get recommendation
        recommendation = learner.get_recommendation(
            provider=embedding_model, model=tool_name, task_type=task_type
        )

        # Assert - should have recommendation for this context
        assert recommendation is not None
        # Should recommend a lower threshold due to high false negatives
        assert recommendation.value < 0.7

        db.close()


class TestToolDeduplicationIntegration:
    """Test tool call deduplication tracker integration."""

    def test_deduplication_tracker_detects_exact_duplicates(self):
        """Test that tracker detects exact duplicate tool calls."""
        # Arrange
        tracker = ToolDeduplicationTracker(window_size=10)

        # Act
        tracker.add_call("code_search", {"query": "tool registration", "path": "."})
        is_redundant = tracker.is_redundant(
            "code_search", {"query": "tool registration", "path": "."}
        )

        # Assert
        assert is_redundant is True

    def test_deduplication_tracker_allows_related_queries(self):
        """Test that tracker allows distinct but related search queries."""
        # Arrange
        tracker = ToolDeduplicationTracker(window_size=10)

        # Act
        tracker.add_call("code_search", {"query": "tool registration"})
        is_redundant = tracker.is_redundant("code_search", {"query": "register tool"})

        # Assert
        assert is_redundant is False

    def test_deduplication_tracker_detects_file_redundancy(self):
        """Test that tracker detects redundant file operations."""
        # Arrange
        tracker = ToolDeduplicationTracker(window_size=10)

        # Act
        tracker.add_call("read_file", {"path": "/tmp/test.py"})
        is_redundant = tracker.is_redundant("read_file", {"path": "/tmp/test.py"})

        # Assert
        assert is_redundant is True

    def test_deduplication_tracker_detects_list_redundancy(self):
        """Test that tracker detects redundant list operations."""
        # Arrange
        tracker = ToolDeduplicationTracker(window_size=10)

        # Act
        tracker.add_call("list_directory", {"path": "/tmp"})
        is_redundant = tracker.is_redundant("list_directory", {"path": "/tmp"})

        # Assert
        assert is_redundant is True

    def test_deduplication_tracker_window_size(self):
        """Test that tracker respects window size limit."""
        # Arrange
        tracker = ToolDeduplicationTracker(window_size=3)

        # Act - record 4 calls (exceeds window)
        tracker.add_call("read_file", {"path": "/tmp/1.py"})
        tracker.add_call("read_file", {"path": "/tmp/2.py"})
        tracker.add_call("read_file", {"path": "/tmp/3.py"})
        tracker.add_call("read_file", {"path": "/tmp/4.py"})

        # Assert - first call should be forgotten
        is_redundant = tracker.is_redundant("read_file", {"path": "/tmp/1.py"})
        assert is_redundant is False  # Should not be considered redundant anymore


@pytest.mark.asyncio
class TestToolPipelineDeduplicationIntegration:
    """Test tool pipeline with deduplication tracker integration."""

    async def test_pipeline_with_deduplication_tracker(self):
        """Test that pipeline integrates with deduplication tracker."""
        # Arrange
        tools = ToolRegistry()
        executor = Mock(spec=ToolExecutor)
        executor.execute = AsyncMock(
            return_value=Mock(success=True, result="test result", error=None)
        )

        tracker = ToolDeduplicationTracker(window_size=10)

        # Create pipeline with tracker
        pipeline = ToolPipeline(
            tool_registry=tools,
            tool_executor=executor,
            config=ToolPipelineConfig(tool_budget=25),
            deduplication_tracker=tracker,
        )

        # Assert - tracker is properly integrated
        assert pipeline.deduplication_tracker is tracker
        assert pipeline.deduplication_tracker.window_size == 10

    async def test_deduplication_tracker_records_successful_calls(self):
        """Test that successful calls are recorded in the tracker."""
        # Arrange
        tracker = ToolDeduplicationTracker(window_size=10)

        # Act
        tracker.add_call("test_tool", {"arg1": "value1"})
        tracker.add_call("test_tool", {"arg1": "value2"})

        # Assert - should have 2 calls tracked
        assert len(tracker.recent_calls) == 2

        # Exact duplicate should be detected
        is_redundant = tracker.is_redundant("test_tool", {"arg1": "value1"})
        assert is_redundant is True

        # Different args should not be redundant
        is_redundant = tracker.is_redundant("test_tool", {"arg1": "value3"})
        assert is_redundant is False

    async def test_write_invalidates_search_freshness_state(self, tmp_path: Path):
        """A successful write should allow a follow-up code search to execute fresh."""
        tools = Mock(spec=ToolRegistry)
        tools.is_tool_enabled.return_value = True
        executor = Mock(spec=ToolExecutor)
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        changed_file = repo_root / "module.py"
        changed_file.write_text("print('before')\n", encoding="utf-8")

        executor.execute = AsyncMock(
            side_effect=[
                Mock(success=True, result="before", error=None, execution_time=0.01),
                Mock(success=True, result="updated", error=None, execution_time=0.01),
                Mock(success=True, result="after", error=None, execution_time=0.01),
            ]
        )

        pipeline = ToolPipeline(
            tool_registry=tools,
            tool_executor=executor,
            config=ToolPipelineConfig(
                tool_budget=25,
                enable_idempotent_caching=False,
                enable_output_aggregation=False,
                enable_synthesis_checkpoints=False,
                enable_semantic_caching=False,
            ),
            deduplication_tracker=ToolDeduplicationTracker(window_size=10),
        )

        search_call = [
            {
                "name": "code_search",
                "arguments": {"query": "node_ids", "path": str(repo_root)},
            }
        ]
        write_call = [
            {
                "name": "write",
                "arguments": {"path": str(changed_file), "content": "print('after')"},
            }
        ]

        # code_search_tool's index cache was replaced by the capability-loader
        # path: ToolPipeline._invalidate_post_edit_freshness_state calls
        # load_code_search_module().mark_index_cache_stale_for_path(...).
        mock_code_search = MagicMock()
        mock_code_search.mark_index_cache_stale_for_path = MagicMock(return_value=1)

        with patch(
            "victor.core.utils.capability_loader.load_code_search_module",
            return_value=mock_code_search,
        ):
            first = await pipeline.execute_tool_calls(search_call, {})
            assert first.results[0].skipped is False

            write_result = await pipeline.execute_tool_calls(write_call, {})
            assert write_result.results[0].success is True

            second = await pipeline.execute_tool_calls(search_call, {})

        # The write must have marked the search index stale for the changed file,
        # letting the follow-up search run fresh rather than deduplicated.
        mock_code_search.mark_index_cache_stale_for_path.assert_called_once()
        assert second.results[0].skipped is False
        assert second.results[0].result == "after"
        assert executor.execute.await_count == 3


class TestEndToEndIntegration:
    """Test all P4 features working together."""

    def test_settings_configuration(self):
        """Test that all P4 settings are properly configured in their nested groups."""
        settings = Settings()

        # Hybrid search — lives in settings.search (SearchSettings)
        assert hasattr(settings.search, "enable_hybrid_search")
        assert hasattr(settings.search, "hybrid_search_semantic_weight")
        assert hasattr(settings.search, "hybrid_search_keyword_weight")
        assert hasattr(settings.search, "enable_semantic_threshold_rl_learning")
        assert hasattr(settings.search, "semantic_threshold_overrides")
        assert hasattr(settings.search, "semantic_similarity_threshold")
        assert hasattr(settings.search, "semantic_query_expansion_enabled")

        # Tool deduplication — lives in settings.tool_selection (ToolSelectionSettings)
        assert hasattr(settings.tool_selection, "enable_tool_deduplication")
        assert hasattr(settings.tool_selection, "tool_deduplication_window_size")

    def test_hybrid_search_factory(self):
        """Test hybrid search engine factory function."""
        # Act
        engine = create_hybrid_search_engine(semantic_weight=0.7, keyword_weight=0.3)

        # Assert
        assert isinstance(engine, HybridSearchEngine)
        assert engine.semantic_weight == 0.7
        assert engine.keyword_weight == 0.3

    def test_components_can_be_disabled(self):
        """Test that P4 features can be disabled via their nested settings groups."""
        settings = Settings()

        settings.search.enable_hybrid_search = False
        settings.search.enable_semantic_threshold_rl_learning = False
        settings.tool_selection.enable_tool_deduplication = False

        assert settings.search.enable_hybrid_search is False
        assert settings.search.enable_semantic_threshold_rl_learning is False
        assert settings.tool_selection.enable_tool_deduplication is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
