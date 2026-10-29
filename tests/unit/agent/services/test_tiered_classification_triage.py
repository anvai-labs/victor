"""Unit tests for tiered classification triage system."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from victor.agent.services.tiered_decision_service import (
    TieredDecisionService,
    ClassificationTriage,
    TieredClassificationResult,
    EdgeLLMVerificationResult,
)
from victor.agent.decisions.schemas import DecisionType
from victor.config.decision_settings import DecisionServiceSettings
from victor.framework.runtime_evaluation_policy import RuntimeEvaluationPolicy


class TestClassificationTriageDataStructures:
    """Test classification triage data structures."""

    def test_classification_triage_enum(self):
        """Test ClassificationTriage enum values."""
        assert ClassificationTriage.ACCEPT == "accept"
        assert ClassificationTriage.VERIFY == "verify"
        assert ClassificationTriage.REJECT == "reject"
        assert len(ClassificationTriage) == 3

    def test_edge_llm_verification_result(self):
        """Test EdgeLLMVerificationResult dataclass."""
        result = EdgeLLMVerificationResult(
            original_result="debug",
            original_confidence=0.65,
            verified_result="analysis",
            verification_confidence=0.85,
            verification_passed=True,
            latency_ms=150.0,
            tokens_used=25,
        )

        assert result.original_result == "debug"
        assert result.original_confidence == 0.65
        assert result.verified_result == "analysis"
        assert result.verification_confidence == 0.85
        assert result.verification_passed is True
        assert result.latency_ms == 150.0
        assert result.tokens_used == 25

        # Test to_dict conversion
        result_dict = result.to_dict()
        assert result_dict["original_result"] == "debug"
        assert result_dict["verification_passed"] is True

    def test_tiered_classification_result(self):
        """Test TieredClassificationResult dataclass."""
        result = TieredClassificationResult(
            result="analysis",
            confidence=0.85,
            triage_outcome=ClassificationTriage.ACCEPT,
            source="edge_verification",
            latency_ms=50.0,
            metadata={"test": "value"},
        )

        assert result.result == "analysis"
        assert result.confidence == 0.85
        assert result.triage_outcome == ClassificationTriage.ACCEPT
        assert result.source == "edge_verification"
        assert result.latency_ms == 50.0
        assert result.verification_result is None
        assert result.metadata == {"test": "value"}

        # Test to_dict conversion
        result_dict = result.to_dict()
        assert result_dict["triage_outcome"] == "accept"
        assert result_dict["verification_result"] is None


class MockDecisionResult:
    """Mock DecisionResult for testing."""

    def __init__(self, result, confidence, source="heuristic"):
        self.result = result
        self.confidence = confidence
        self.source = source


class TestTieredDecisionServiceTriage:
    """Test TieredDecisionService classify_with_triage method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = DecisionServiceSettings(enabled=True)
        self.service = TieredDecisionService(self.config)
        self.runtime_policy = RuntimeEvaluationPolicy()

    def test_high_confidence_accept_fast_path(self):
        """Test high confidence (≥0.8) is accepted immediately."""
        # Mock the base decision to return high confidence
        with patch.object(
            self.service,
            "decide_sync",
            return_value=MockDecisionResult("debug", 0.85, "keyword"),
        ) as mock_decide:
            result = self.service.classify_with_triage(
                DecisionType.TASK_TYPE_CLASSIFICATION,
                context={"message": "fix the bug"},
                heuristic_result="debug",
                heuristic_confidence=0.85,
                runtime_policy=self.runtime_policy,
            )

            assert result.triage_outcome == ClassificationTriage.ACCEPT
            assert result.confidence == 0.85
            assert result.source == "heuristic"
            assert result.verification_result is None
            assert result.latency_ms < 10  # Fast path
            assert result.metadata.get("fast_path") is True
            assert result.metadata.get("skipped_decision_service") is True
            mock_decide.assert_not_called()

    def test_medium_confidence_triggers_verification(self):
        """Test medium confidence (0.5-0.8) triggers verification."""
        # Mock base decision with medium confidence
        mock_base_result = MockDecisionResult("analysis", 0.65, "keyword")

        # Mock edge service for verification
        mock_edge_service = MagicMock()
        mock_verification = MockDecisionResult("analysis", 0.85, "llm")
        mock_edge_service.decide_sync.return_value = mock_verification

        with patch.object(self.service, "_get_service", return_value=mock_edge_service):
            with patch.object(self.service, "decide_sync", return_value=mock_base_result):
                result = self.service.classify_with_triage(
                    DecisionType.TASK_TYPE_CLASSIFICATION,
                    context={"message": "analyze code"},
                    heuristic_result="analysis",
                    heuristic_confidence=0.65,
                    runtime_policy=self.runtime_policy,
                )

                assert result.triage_outcome == ClassificationTriage.VERIFY
                assert result.verification_result is not None
                assert result.verification_result.verification_passed is True
                assert result.latency_ms >= 0  # Verification adds some latency
                assert result.source == "edge_verification"

    def test_low_confidence_rejects_early(self):
        """Test low confidence (<0.5) rejects early."""
        # Mock base decision with low confidence
        with patch.object(
            self.service,
            "decide_sync",
            return_value=MockDecisionResult("unknown", 0.3, "heuristic"),
        ) as mock_decide:
            result = self.service.classify_with_triage(
                DecisionType.TASK_TYPE_CLASSIFICATION,
                context={"message": "do something"},
                heuristic_result="default",
                heuristic_confidence=0.3,
                runtime_policy=self.runtime_policy,
            )

            assert result.triage_outcome == ClassificationTriage.REJECT
            assert result.source == "heuristic_fallback"
            assert result.result == "default"
            assert result.latency_ms < 10  # Early rejection
            assert result.metadata.get("reason") == "confidence_below_threshold"
            assert result.metadata.get("skipped_decision_service") is True
            mock_decide.assert_not_called()

    def test_verification_passed_uses_verified_result(self):
        """Test successful verification uses verified result."""
        # Mock base decision with medium confidence
        mock_base_result = MockDecisionResult("action", 0.65, "keyword")

        # Mock edge service that verifies to a different result
        mock_edge_service = MagicMock()
        mock_verification = MockDecisionResult("generation", 0.85, "llm")
        mock_edge_service.decide_sync.return_value = mock_verification

        with patch.object(self.service, "_get_service", return_value=mock_edge_service):
            with patch.object(self.service, "decide_sync", return_value=mock_base_result):
                result = self.service.classify_with_triage(
                    DecisionType.TASK_TYPE_CLASSIFICATION,
                    context={"message": "create a file"},
                    heuristic_result="action",
                    heuristic_confidence=0.65,
                    runtime_policy=self.runtime_policy,
                )

                assert result.triage_outcome == ClassificationTriage.VERIFY
                assert result.verification_result.verification_passed is True
                assert result.result == "generation"  # Verified result
                assert result.confidence == 0.85  # Verified confidence

    def test_verification_failed_uses_original_with_penalty(self):
        """Test failed verification uses original result."""
        # Mock base decision with medium confidence
        mock_base_result = MockDecisionResult("analysis", 0.65, "keyword")

        # Mock edge service that fails verification (low confidence)
        mock_edge_service = MagicMock()
        mock_verification = MockDecisionResult("analysis", 0.4, "llm")
        mock_edge_service.decide_sync.return_value = mock_verification

        with patch.object(self.service, "_get_service", return_value=mock_edge_service):
            with patch.object(self.service, "decide_sync", return_value=mock_base_result):
                result = self.service.classify_with_triage(
                    DecisionType.TASK_TYPE_CLASSIFICATION,
                    context={"message": "analyze code"},
                    heuristic_result="analysis",
                    heuristic_confidence=0.65,
                    runtime_policy=self.runtime_policy,
                )

                assert result.triage_outcome == ClassificationTriage.VERIFY
                assert result.verification_result.verification_passed is False
                assert result.result == "analysis"  # Original result
                assert result.confidence == 0.65  # Original confidence
                assert result.source == "keyword"  # Original source

    def test_edge_unavailable_conservative_rejection(self):
        """Test that edge unavailability causes conservative rejection."""
        # Mock base decision with medium confidence
        mock_base_result = MockDecisionResult("analysis", 0.65, "keyword")

        # Mock edge service as unavailable
        with patch.object(self.service, "_get_service", return_value=None):
            with patch.object(self.service, "decide_sync", return_value=mock_base_result):
                result = self.service.classify_with_triage(
                    DecisionType.TASK_TYPE_CLASSIFICATION,
                    context={"message": "analyze code"},
                    heuristic_result="analysis",
                    heuristic_confidence=0.65,
                    runtime_policy=self.runtime_policy,
                )

                assert result.triage_outcome == ClassificationTriage.VERIFY
                assert result.verification_result is not None
                assert result.verification_result.verification_passed is False
                assert result.source == "keyword"  # Falls back to original

    def test_custom_runtime_policy_thresholds(self):
        """Test custom runtime policy thresholds."""
        custom_policy = RuntimeEvaluationPolicy(
            high_confidence_threshold=0.9,
            medium_confidence_threshold=0.6,
        )

        # Mock base decision with 0.85 confidence (would be high with default, medium with custom)
        with patch.object(
            self.service,
            "decide_sync",
            return_value=MockDecisionResult("debug", 0.85, "keyword"),
        ):
            result = self.service.classify_with_triage(
                DecisionType.TASK_TYPE_CLASSIFICATION,
                context={"message": "fix bug"},
                heuristic_result="debug",
                heuristic_confidence=0.85,
                runtime_policy=custom_policy,
            )

            # With custom thresholds, 0.85 is in VERIFY range (0.6-0.9)
            # But since edge is unavailable, it should fall back to original
            assert result.confidence == 0.85

    def test_intent_classification_triage(self):
        """Test triage for INTENT_CLASSIFICATION decision type."""
        with patch.object(
            self.service,
            "decide_sync",
            return_value=MockDecisionResult("continuation", 0.75, "semantic"),
        ):
            result = self.service.classify_with_triage(
                DecisionType.INTENT_CLASSIFICATION,
                context={"text_tail": "Let me read the file"},
                heuristic_result="continuation",
                heuristic_confidence=0.75,
                runtime_policy=self.runtime_policy,
            )

            # Result should have VERIFY triage outcome (medium confidence)
            assert result.triage_outcome == ClassificationTriage.VERIFY
            assert result.result == "continuation"


class TestEdgeLLMVerificationStrategy:
    """Test edge LLM verification strategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = DecisionServiceSettings(enabled=True)
        self.service = TieredDecisionService(self.config)
        self.runtime_policy = RuntimeEvaluationPolicy()

    def test_verification_context_includes_original_result(self):
        """Test that verification context includes original classification."""
        mock_edge_service = MagicMock()
        mock_verification = MockDecisionResult("debug", 0.85, "llm")
        mock_edge_service.decide_sync.return_value = mock_verification

        original_result = MockDecisionResult("analysis", 0.65, "keyword")

        with patch.object(self.service, "_get_service", return_value=mock_edge_service):
            verification = self.service._verify_with_edge_llm(
                DecisionType.TASK_TYPE_CLASSIFICATION,
                context={"message": "fix bug"},
                original_result=original_result,
                runtime_policy=self.runtime_policy,
            )

            # Check that edge service was called with verification context
            mock_edge_service.decide_sync.assert_called_once()
            call_args = mock_edge_service.decide_sync.call_args
            verification_context = call_args[0][1]  # Second positional arg (context)

            assert "original_classification" in verification_context
            assert verification_context["original_classification"] == "analysis"
            assert "original_confidence" in verification_context
            assert verification_context["original_confidence"] == "0.65"
            assert verification_context["verification_task"] == "verify"

    def test_verification_error_handling(self):
        """Test that verification errors are handled gracefully."""
        # Mock edge service that raises exception
        mock_edge_service = MagicMock(side_effect=Exception("Edge service error"))

        original_result = MockDecisionResult("analysis", 0.65, "keyword")

        with patch.object(self.service, "_get_service", return_value=mock_edge_service):
            verification = self.service._verify_with_edge_llm(
                DecisionType.TASK_TYPE_CLASSIFICATION,
                context={"message": "analyze code"},
                original_result=original_result,
                runtime_policy=self.runtime_policy,
            )

            assert verification.verification_passed is False
            assert verification.original_result == "analysis"
            assert verification.verified_result == "analysis"  # Falls back to original


class TestTriageIntegrationScenarios:
    """Test realistic triage integration scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = DecisionServiceSettings(enabled=True)
        self.service = TieredDecisionService(self.config)
        self.runtime_policy = RuntimeEvaluationPolicy()

    def test_clear_task_type_accepts_immediately(self):
        """Test clear task type is accepted without verification."""
        with patch.object(
            self.service,
            "decide_sync",
            return_value=MockDecisionResult(
                MagicMock(task_type="debug", value="debug"), 0.92, "keyword"
            ),
        ):
            result = self.service.classify_with_triage(
                DecisionType.TASK_TYPE_CLASSIFICATION,
                context={"message": "fix the authentication bug"},
                heuristic_result="debug",
                heuristic_confidence=0.92,
                runtime_policy=self.runtime_policy,
            )

            assert result.triage_outcome == ClassificationTriage.ACCEPT
            assert result.verification_result is None

    def test_ambiguous_task_type_triggers_verification(self):
        """Test ambiguous task type triggers verification."""
        # Mock medium confidence base result
        mock_base = MockDecisionResult(
            MagicMock(task_type="analysis", value="analysis"), 0.65, "semantic"
        )

        # Mock edge service that confirms the classification
        mock_edge_service = MagicMock()
        mock_verified = MockDecisionResult(
            MagicMock(task_type="analysis", value="analysis"), 0.88, "llm"
        )
        mock_edge_service.decide_sync.return_value = mock_verified

        with patch.object(self.service, "_get_service", return_value=mock_edge_service):
            with patch.object(self.service, "decide_sync", return_value=mock_base):
                result = self.service.classify_with_triage(
                    DecisionType.TASK_TYPE_CLASSIFICATION,
                    context={"message": "maybe analyze this?"},
                    heuristic_result="analysis",
                    heuristic_confidence=0.65,
                    runtime_policy=self.runtime_policy,
                )

                assert result.triage_outcome == ClassificationTriage.VERIFY
                assert result.verification_result.verification_passed is True

    def test_very_low_confidence_rejects(self):
        """Test very low confidence classifications are rejected."""
        with patch.object(
            self.service,
            "decide_sync",
            return_value=MockDecisionResult(
                MagicMock(task_type="default", value="default"), 0.2, "heuristic"
            ),
        ):
            result = self.service.classify_with_triage(
                DecisionType.TASK_TYPE_CLASSIFICATION,
                context={"message": "do stuff"},
                heuristic_result="default",
                heuristic_confidence=0.2,
                runtime_policy=self.runtime_policy,
            )

            assert result.triage_outcome == ClassificationTriage.REJECT
            assert result.source == "heuristic_fallback"
