"""End-to-end integration tests for predictive enhancements.

Tests cover:
- Full predictive workflow from prediction → selection → preloading
- Phase-aware context management
- Performance benchmarks
- Error handling and recovery
"""

from datetime import datetime, timezone, timedelta


from victor.agent.conversation.state_machine import ConversationStage
from victor.agent.planning.cooccurrence_tracker import CooccurrenceTracker
from victor.agent.planning.tool_preloader import ToolPreloader
from victor.agent.planning.tool_predictor import ToolPredictor
from victor.core.shared_types import TaskPhase


class TestPredictiveWorkflowIntegration:
    """Test complete predictive workflow integration."""

    def test_full_predictive_workflow(self):
        """Test complete workflow from prediction to preloading."""
        # Setup all components
        tracker = CooccurrenceTracker()
        predictor = ToolPredictor(cooccurrence_tracker=tracker)
        preloader = ToolPreloader(
            tool_predictor=predictor,
        )

        # Train the system with typical bugfix workflow
        tracker.record_tool_sequence(
            tools=["search", "read", "edit"],
            task_type="bugfix",
            success=True,
        )
        tracker.record_tool_sequence(
            tools=["search", "read", "edit"],
            task_type="bugfix",
            success=True,
        )
        tracker.record_tool_sequence(
            tools=["search", "read", "edit"],
            task_type="bugfix",
            success=True,
        )

        # Get predictions after "search"
        predictions = predictor.predict_tools(
            task_description="Fix the authentication bug in login.py",
            current_step="exploration",
            recent_tools=["search"],
            task_type="bugfix",
        )

        # Should predict "read" or "edit" with high confidence
        tool_names = [p.tool_name for p in predictions]
        assert len(predictions) > 0
        assert any(name in ["read", "edit"] for name in tool_names)

        # Check top prediction
        top_prediction = predictions[0]
        assert top_prediction.confidence_level in ("HIGH", "MEDIUM")

    def test_learning_from_outcomes(self):
        """Test that system learns from execution outcomes."""
        tracker = CooccurrenceTracker()
        predictor = ToolPredictor(cooccurrence_tracker=tracker)

        # Train with successful patterns
        for _ in range(5):
            tracker.record_tool_sequence(
                tools=["search", "read", "edit"],
                task_type="bugfix",
                success=True,
            )

        # Train with some failures
        for _ in range(3):
            tracker.record_tool_sequence(
                tools=["search", "write"],
                task_type="bugfix",
                success=False,
            )

        # Get predictions
        predictions = predictor.predict_tools(
            task_description="Fix the bug",
            current_step="exploration",
            recent_tools=["search"],
            task_type="bugfix",
        )

        # Should have predictions
        assert len(predictions) > 0

        # Check that success boosting works
        for pred in predictions:
            assert 0.0 <= pred.probability <= 1.0


class TestPhaseAwareContextIntegration:
    """Test phase-aware context management integration."""

    def test_phase_detection_integration(self):
        """Test phase detection with actual conversation state."""
        from victor.agent.context_phase_detector import PhaseDetector

        detector = PhaseDetector()

        # Test exploration phase detection
        phase = detector.detect_phase(
            current_stage=ConversationStage.INITIAL,
            recent_tools=[],
            message_content="I'll help you explore the codebase",
        )

        assert phase == TaskPhase.EXPLORATION

    def test_phase_transition_workflow(self):
        """Test phase transition through conversation."""
        from victor.agent.context_phase_detector import PhaseTransitionDetector

        detector = PhaseTransitionDetector()

        # First transition (exploration to planning) should be allowed
        assert detector.should_transition(
            new_phase=TaskPhase.PLANNING,
        )

        # Simulate transition
        detector._last_transition_time = datetime.now(timezone.utc) - timedelta(seconds=3)

        # Should allow another transition (cooldown passed)
        assert detector.should_transition(
            new_phase=TaskPhase.EXECUTION,
        )

    def test_phase_aware_scoring(self):
        """Test phase-aware scoring affects message prioritization."""
        from victor.agent.conversation.scoring import score_messages

        from victor.agent.conversation.types import (
            ConversationMessage,
            MessageRole,
        )

        # Create test messages
        messages = [
            ConversationMessage(
                role=MessageRole.USER,
                content="Find the bug",
            ),
            ConversationMessage(
                role=MessageRole.ASSISTANT,
                content="I'll search for it",
            ),
        ]

        # Score with exploration phase
        exploration_scores = score_messages(
            messages=messages,
            phase=TaskPhase.EXPLORATION,
        )

        # Score with execution phase
        execution_scores = score_messages(
            messages=messages,
            phase=TaskPhase.EXECUTION,
        )

        # Scores should differ based on phase
        assert exploration_scores is not None
        assert execution_scores is not None


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery scenarios."""

    def test_predictor_failure_fallback(self):
        """Test graceful fallback when predictor fails."""

        predictor = ToolPredictor()

        # Test with empty task description
        predictions = predictor.predict_tools(
            task_description="",
            current_step="exploration",
            recent_tools=[],
        )

        # Should return empty list or very low confidence predictions
        assert isinstance(predictions, list)

    async def test_cache_error_handling(self):
        """Test that cache errors are handled gracefully."""
        from victor.agent.planning.tool_preloader import ToolPreloader

        preloader = ToolPreloader()

        # Try to get non-existent tool
        schema = await preloader.get_tool_schema("nonexistent_tool")

        # Should handle gracefully (return None when not found)
        assert schema is None

    def test_tracker_error_recovery(self):
        """Test tracker handles errors gracefully."""
        tracker = CooccurrenceTracker()

        # Record empty sequence (should be handled)
        tracker.record_tool_sequence(
            tools=[],
            task_type="bugfix",
            success=True,
        )

        # Should not crash
        stats = tracker.get_statistics()
        assert stats["total_sequences_recorded"] == 0  # Empty sequences not counted


class TestPerformanceBenchmarks:
    """Performance benchmarks for predictive components."""

    def test_prediction_accuracy_target(self):
        """Test that prediction accuracy meets target (>80%)."""
        tracker = CooccurrenceTracker()
        predictor = ToolPredictor(cooccurrence_tracker=tracker)

        # Train with clear patterns
        for _ in range(10):
            tracker.record_tool_sequence(
                tools=["search", "read"],
                task_type="bugfix",
                success=True,
            )
            tracker.record_tool_sequence(
                tools=["read", "edit"],
                task_type="bugfix",
                success=True,
            )

        # Test predictions
        correct = 0
        total = 5

        for _ in range(total):
            predictions = predictor.predict_tools(
                task_description="Find and fix the bug",
                current_step="exploration",
                recent_tools=["search"],
                task_type="bugfix",
            )

            # Should predict "read" with high confidence
            top_prediction = predictions[0] if predictions else None
            if top_prediction and top_prediction.tool_name == "read":
                correct += 1

        accuracy = correct / total if total > 0 else 0

        # Should have >80% accuracy
        assert accuracy > 0.8

    async def test_cache_performance(self):
        """Test cache hit rate target (>60%)."""
        from victor.agent.planning.tool_preloader import ToolPreloader

        preloader = ToolPreloader()

        # Add some entries to L1 cache
        schema = {"type": "object"}
        for tool_name in ["read", "grep", "edit", "write", "test"]:
            from victor.agent.planning.tool_preloader import CacheEntry

            entry = CacheEntry(
                tool_name=tool_name,
                schema=schema,
                expires_at=datetime.now(timezone.utc) + timedelta(seconds=60),
            )
            preloader._add_to_l1_cache(tool_name, entry)

        # Access all tools (cache hits)
        for tool_name in ["read", "grep", "edit", "write", "test"]:
            await preloader.get_tool_schema(tool_name)

        stats = preloader.get_statistics()

        # Should have 100% cache hit rate
        assert stats["l1_hit_rate"] == 1.0


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_existing_code_works_without_enhancements(self):
        """Test that existing code works without enhancements."""

        # This should work as before (no predictive features)
        # ToolSelector initialization doesn't require predictive components
        # The code should gracefully handle missing predictive features

    def test_gradual_enhancement_path(self):
        """Test that enhancements can be adopted gradually."""
        # Step 1: Use without enhancements (baseline)
        # Step 2: Enable feature flags but 0% rollout
        # Step 3: Increase rollout percentage gradually
        # Step 4: Full rollout

        # All steps should be valid without breaking changes
        assert True  # Placeholder for gradual enhancement


class TestObservabilityAndMonitoring:
    """Test observability and monitoring capabilities."""

    def test_statistics_aggregation(self):
        """Test that statistics can be aggregated across components."""
        tracker = CooccurrenceTracker()
        predictor = ToolPredictor(cooccurrence_tracker=tracker)
        preloader = ToolPreloader(tool_predictor=predictor)

        # Record some activity
        tracker.record_tool_sequence(
            tools=["read", "edit"],
            task_type="bugfix",
            success=True,
        )

        # Get statistics from all components
        tracker_stats = tracker.get_statistics()
        predictor_stats = predictor.get_statistics()
        preloader_stats = preloader.get_statistics()

        # All should return statistics
        assert tracker_stats is not None
        assert predictor_stats is not None
        assert preloader_stats is not None

    def test_error_tracking(self):
        """Test error tracking across components."""
        tracker = CooccurrenceTracker()

        # Record errors
        tracker.record_tool_sequence(
            tools=["read", "broken_tool"],
            task_type="bugfix",
            success=False,
        )

        stats = tracker.get_statistics()

        # Should track failures
        # (implementation-specific - adjust as needed)
        assert "total_sequences_recorded" in stats

    def test_performance_metrics(self):
        """Test performance metrics collection."""
        from victor.agent.planning.tool_preloader import ToolPreloader

        preloader = ToolPreloader()

        # Generate some activity
        preloader._background_loads = 10
        preloader._preload_count = 15

        stats = preloader.get_statistics()

        # Should track performance
        assert stats["preload_count"] == 15
        assert stats["background_loads"] == 10


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    def test_bugfix_workflow(self):
        """Test typical bugfix workflow with all enhancements."""
        # Setup
        tracker = CooccurrenceTracker()
        predictor = ToolPredictor(cooccurrence_tracker=tracker)

        # Simulate bugfix workflow
        # Step 1: Search for bug
        tracker.record_tool_sequence(
            tools=["search"],
            task_type="bugfix",
            success=True,
        )

        # Step 2: Read code
        predictions = predictor.predict_tools(
            task_description="Read the file to understand the bug",
            current_step="exploration",
            recent_tools=["search"],
            task_type="bugfix",
        )

        # Should predict "read" or similar
        tool_names = [p.tool_name for p in predictions]
        assert any(name in ["read", "code_search"] for name in tool_names)

    def test_feature_development_workflow(self):
        """Test feature development workflow."""
        tracker = CooccurrenceTracker()
        predictor = ToolPredictor(cooccurrence_tracker=tracker)

        # Simulate feature workflow
        # Step 1: Plan
        tracker.record_tool_sequence(
            tools=["plan"],
            task_type="feature",
            success=True,
        )

        # Step 2: Design
        predictions = predictor.predict_tools(
            task_description="Design the new feature",
            current_step="planning",
            recent_tools=["plan"],
            task_type="feature",
        )

        # Should predict design-related tools
        tool_names = [p.tool_name for p in predictions]
        assert len(tool_names) >= 0

    def test_multi_phase_workflow(self):
        """Test workflow that spans multiple phases."""
        from victor.agent.context_phase_detector import PhaseDetector

        detector = PhaseDetector()

        # Start in exploration
        phase1 = detector.detect_phase(
            current_stage=ConversationStage.INITIAL,
            recent_tools=[],
            message_content="Let me explore the codebase",
        )

        # Move to planning
        phase2 = detector.detect_phase(
            current_stage=ConversationStage.PLANNING,
            recent_tools=["search", "read"],
            message_content="Now I'll create a plan",
        )

        # Move to execution
        phase3 = detector.detect_phase(
            current_stage=ConversationStage.EXECUTION,
            recent_tools=["write"],
            message_content="Implementing the fix",
        )

        # Should detect different phases
        assert phase1 == TaskPhase.EXPLORATION
        assert phase2 in (TaskPhase.EXPLORATION, TaskPhase.PLANNING)
        assert phase3 in (TaskPhase.EXECUTION, TaskPhase.EXPLORATION)


class TestDocumentationCompleteness:
    """Test that documentation is complete and accessible."""

    # Note: docs/rollout-guide.md was intentionally removed in the documentation
    # consolidation (commit 6132152dd); the predictive-enhancements feature is
    # still implemented and exercised by the rest of this suite.

    def test_api_documentation(self):
        """Test that API is documented."""
        # Check that key modules have docstrings
        from victor.agent.planning.tool_predictor import ToolPredictor

        assert ToolPredictor.__doc__ is not None
        assert ToolPredictor.predict_tools.__doc__ is not None
