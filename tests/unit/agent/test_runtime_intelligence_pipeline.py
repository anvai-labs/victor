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

"""Unit tests for the RuntimeIntelligencePipeline module.

Tests the runtime-intelligence pipeline which integrates:
- IntelligentPromptBuilder: Embedding-based context selection
- AdaptiveModeController: Q-learning for mode transitions
- ResponseQualityScorer: Multi-dimensional quality assessment
- GroundingVerifier: Hallucination detection
- ResilientExecutor: Circuit breaker, retry, rate limiting

Coverage targets:
- Lines 215-223, 242-243, 262-263, 274-275, 285-298, 307-308, 313-336
- Lines 359, 376-377, 381-388, 392-397, 405-406, 423-438
- Lines 475-561, 599, 607-616, 739-743, 777-800
- Lines 822-838, 842, 846-847, 852-860, 864-869, 873-884, 908-918, 923
"""

import pytest
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch


from victor.agent.runtime_intelligence_pipeline import (
    RuntimeIntelligencePipeline,
    PipelineStats,
    RequestContext,
    ResponseResult,
    get_runtime_intelligence_pipeline,
    clear_runtime_intelligence_pipeline_cache,
    PROVIDERS_WITH_REPETITION_ISSUES,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_provider_adapter():
    """Create a mock provider adapter with default capabilities."""
    adapter = MagicMock()
    adapter.capabilities.quality_threshold = 0.8
    adapter.capabilities.grounding_strictness = 0.7
    adapter.capabilities.output_deduplication = False
    return adapter


@pytest.fixture
def mock_provider_adapter_with_dedup():
    """Create a mock provider adapter with deduplication enabled."""
    adapter = MagicMock()
    adapter.capabilities.quality_threshold = 0.8
    adapter.capabilities.grounding_strictness = 0.7
    adapter.capabilities.output_deduplication = True
    return adapter


@pytest.fixture
def pipeline(tmp_path, mock_provider_adapter):
    """Create a RuntimeIntelligencePipeline for testing."""
    with patch(
        "victor.agent.runtime_intelligence_pipeline.get_provider_adapter",
        return_value=mock_provider_adapter,
    ):
        return RuntimeIntelligencePipeline(
            provider_name="anthropic",
            model="claude-3-sonnet",
            profile_name="test-profile",
            project_root=str(tmp_path),
        )


@pytest.fixture
def pipeline_xai(tmp_path, mock_provider_adapter_with_dedup):
    """Create a RuntimeIntelligencePipeline for xAI provider (has dedup)."""
    with patch(
        "victor.agent.runtime_intelligence_pipeline.get_provider_adapter",
        return_value=mock_provider_adapter_with_dedup,
    ):
        return RuntimeIntelligencePipeline(
            provider_name="xai",
            model="grok-2",
            profile_name="test-xai",
            project_root=str(tmp_path),
        )


@pytest.fixture
def mock_prompt_builder():
    """Mock IntelligentPromptBuilder."""
    builder = AsyncMock()
    builder.build = AsyncMock(return_value="System prompt content")
    builder.record_feedback = MagicMock()
    builder.get_profile_stats = MagicMock(return_value={"cache_state": "warm"})
    return builder


@pytest.fixture
def mock_mode_controller():
    """Mock AdaptiveModeController."""
    controller = MagicMock()
    action = MagicMock()
    action.target_mode = MagicMock()
    action.target_mode.value = "build"
    action.confidence = 0.85
    action.should_continue = True
    action.adjust_tool_budget = 2
    controller.get_recommended_action = MagicMock(return_value=action)
    controller.get_optimal_tool_budget = MagicMock(return_value=15)
    controller.record_outcome = MagicMock(return_value=0.5)
    controller.should_continue = MagicMock(return_value=(True, "Continue processing"))
    controller.get_session_stats = MagicMock(return_value={"mode_transitions": 2})
    controller.reset_session = MagicMock()
    controller.get_quality_thresholds = MagicMock(
        return_value={"min_quality": 0.7, "grounding_threshold": 0.65}
    )
    return controller


@pytest.fixture
def mock_quality_scorer():
    """Mock ResponseQualityScorer."""
    scorer = AsyncMock()
    result = MagicMock()
    result.overall_score = 0.85
    result.dimension_scores = [
        MagicMock(dimension=MagicMock(value="relevance"), score=0.9),
        MagicMock(dimension=MagicMock(value="completeness"), score=0.8),
    ]
    result.improvement_suggestions = ["Add more context"]
    scorer.score = AsyncMock(return_value=result)
    return scorer


@pytest.fixture
def mock_grounding_verifier():
    """Mock GroundingVerifier."""
    verifier = AsyncMock()
    result = MagicMock()
    result.is_grounded = True
    result.confidence = 0.9
    result.issues = []
    result.generate_feedback_prompt = MagicMock(return_value="")
    result.verified_references = ["main.py", "main.py:2-4"]
    result.unverified_references = []
    verifier.verify = AsyncMock(return_value=result)
    return verifier


@pytest.fixture
def mock_grounding_verifier_failed():
    """Mock GroundingVerifier that fails grounding."""
    verifier = AsyncMock()
    result = MagicMock()
    result.is_grounded = False
    result.confidence = 0.4
    issue = MagicMock()
    issue.issue_type = MagicMock()
    issue.issue_type.value = "file_not_found"
    issue.description = "Referenced file does not exist"
    result.issues = [issue]
    result.verified_references = ["main.py"]
    result.unverified_references = ["missing.py"]
    result.generate_feedback_prompt = MagicMock(return_value="Please verify the file path exists.")
    verifier.verify = AsyncMock(return_value=result)
    return verifier


@pytest.fixture
def mock_resilient_executor():
    """Mock ResilientExecutor."""
    executor = MagicMock()
    executor.execute = AsyncMock(return_value={"content": "test response"})
    return executor


# =============================================================================
# Test Dataclasses
# =============================================================================


class TestRequestContext:
    """Tests for RequestContext dataclass."""

    def test_request_context_creation(self):
        """Test RequestContext can be created with required fields."""
        context = RequestContext(
            system_prompt="You are a helpful assistant.",
            recommended_tool_budget=20,
            recommended_mode="build",
            should_continue=True,
        )

        assert context.system_prompt == "You are a helpful assistant."
        assert context.recommended_tool_budget == 20
        assert context.recommended_mode == "build"
        assert context.should_continue is True
        assert context.mode_confidence == 0.5  # default
        assert context.profile_stats == {}  # default

    def test_request_context_with_optional_fields(self):
        """Test RequestContext with all optional fields."""
        context = RequestContext(
            system_prompt="Test prompt",
            recommended_tool_budget=15,
            recommended_mode="explore",
            should_continue=True,
            continuation_context="Previous context",
            mode_confidence=0.9,
            profile_stats={"cache_state": "warm"},
        )

        assert context.continuation_context == "Previous context"
        assert context.mode_confidence == 0.9
        assert context.profile_stats == {"cache_state": "warm"}


class TestResponseResult:
    """Tests for ResponseResult dataclass."""

    def test_response_result_creation(self):
        """Test ResponseResult creation with required fields."""
        result = ResponseResult(
            is_valid=True,
            quality_score=0.85,
            grounding_score=0.9,
            is_grounded=True,
        )

        assert result.is_valid is True
        assert result.quality_score == 0.85
        assert result.grounding_score == 0.9
        assert result.is_grounded is True
        assert result.learning_reward == 0.0  # default

    def test_response_result_with_grounding_failure(self):
        """Test ResponseResult with grounding failure flags."""
        result = ResponseResult(
            is_valid=False,
            quality_score=0.6,
            grounding_score=0.4,
            is_grounded=False,
            grounding_issues=["File not found"],
            should_finalize=True,
            finalize_reason="grounding failure limit exceeded",
            grounding_feedback="Check file paths",
        )

        assert result.is_valid is False
        assert result.should_finalize is True
        assert result.finalize_reason == "grounding failure limit exceeded"
        assert result.grounding_feedback == "Check file paths"

    def test_response_result_carries_grounding_evidence_summary(self):
        """ResponseResult should expose verified and unverified grounding evidence."""
        result = ResponseResult(
            is_valid=False,
            quality_score=0.8,
            grounding_score=0.45,
            is_grounded=False,
            grounding_verified_references=["main.py", "main.py:2-4"],
            grounding_unverified_references=["missing.py"],
            grounding_evidence_summary=(
                "Grounding evidence:\n"
                "- Verified: main.py, main.py:2-4\n"
                "- Unverified: missing.py"
            ),
        )

        assert "Verified: main.py" in result.grounding_evidence_summary
        assert result.grounding_unverified_references == ["missing.py"]


class TestPipelineStats:
    """Tests for PipelineStats dataclass."""

    def test_pipeline_stats_defaults(self):
        """Test PipelineStats has correct defaults."""
        stats = PipelineStats()

        assert stats.total_requests == 0
        assert stats.successful_requests == 0
        assert stats.avg_quality_score == 0.0
        assert stats.avg_grounding_score == 0.0
        assert stats.total_learning_reward == 0.0
        assert stats.mode_transitions == 0
        assert stats.circuit_breaker_trips == 0
        assert stats.retry_attempts == 0
        assert stats.cache_state == "cold"
        assert stats.profile_name == ""

    def test_pipeline_stats_with_profile(self):
        """Test PipelineStats with profile name."""
        stats = PipelineStats(profile_name="test-profile")

        assert stats.profile_name == "test-profile"


# =============================================================================
# Test RuntimeIntelligencePipeline Initialization
# =============================================================================


class TestPipelineInit:
    """Tests for RuntimeIntelligencePipeline initialization."""

    def test_init_basic(self, tmp_path, mock_provider_adapter):
        """Test basic pipeline initialization."""
        with patch(
            "victor.agent.runtime_intelligence_pipeline.get_provider_adapter",
            return_value=mock_provider_adapter,
        ):
            pipeline = RuntimeIntelligencePipeline(
                provider_name="anthropic",
                model="claude-3-sonnet",
                profile_name="test",
                project_root=str(tmp_path),
            )

        assert pipeline.provider_name == "anthropic"
        assert pipeline.model == "claude-3-sonnet"
        assert pipeline.profile_name == "test"
        assert pipeline.project_root == str(tmp_path)
        assert pipeline._prompt_builder is None
        assert pipeline._mode_controller is None
        assert pipeline._quality_scorer is None
        assert pipeline._grounding_verifier is None
        assert pipeline._resilient_executor is None

    def test_init_without_project_root(self, mock_provider_adapter):
        """Test pipeline initialization without project root."""
        with patch(
            "victor.agent.runtime_intelligence_pipeline.get_provider_adapter",
            return_value=mock_provider_adapter,
        ):
            pipeline = RuntimeIntelligencePipeline(
                provider_name="openai",
                model="gpt-4",
                profile_name="test-openai",
            )

        assert pipeline.project_root is None

    def test_deduplication_enabled_for_xai(self, tmp_path, mock_provider_adapter_with_dedup):
        """Test deduplication is enabled for xAI provider."""
        with patch(
            "victor.agent.runtime_intelligence_pipeline.get_provider_adapter",
            return_value=mock_provider_adapter_with_dedup,
        ):
            pipeline = RuntimeIntelligencePipeline(
                provider_name="xai",
                model="grok-2",
                profile_name="test-xai",
            )

        assert pipeline._deduplication_enabled is True

    def test_deduplication_disabled_for_anthropic(self, tmp_path, mock_provider_adapter):
        """Test deduplication is disabled for anthropic provider."""
        with patch(
            "victor.agent.runtime_intelligence_pipeline.get_provider_adapter",
            return_value=mock_provider_adapter,
        ):
            pipeline = RuntimeIntelligencePipeline(
                provider_name="anthropic",
                model="claude-3-sonnet",
                profile_name="test",
            )

        assert pipeline._deduplication_enabled is False


class TestPipelineCreateFactory:
    """Tests for the create factory method (lines 215-223)."""

    @pytest.mark.asyncio
    async def test_create_with_profile_name(self, tmp_path, mock_provider_adapter):
        """Test create factory with explicit profile name."""
        with patch(
            "victor.agent.runtime_intelligence_pipeline.get_provider_adapter",
            return_value=mock_provider_adapter,
        ):
            pipeline = await RuntimeIntelligencePipeline.create(
                provider_name="anthropic",
                model="claude-3-sonnet",
                profile_name="custom-profile",
                project_root=str(tmp_path),
            )

        assert pipeline.profile_name == "custom-profile"

    @pytest.mark.asyncio
    async def test_create_without_profile_name(self, mock_provider_adapter):
        """Test create factory generates default profile name (line 215)."""
        with patch(
            "victor.agent.runtime_intelligence_pipeline.get_provider_adapter",
            return_value=mock_provider_adapter,
        ):
            pipeline = await RuntimeIntelligencePipeline.create(
                provider_name="openai",
                model="gpt-4o",
            )

        # Default profile = "{provider}:{model}"
        assert pipeline.profile_name == "openai:gpt-4o"


# =============================================================================
# Test Lazy Component Initialization
# =============================================================================


class TestLazyPromptBuilder:
    """Tests for lazy prompt builder initialization (lines 242-243)."""

    @pytest.mark.asyncio
    async def test_get_prompt_builder_failure(self, pipeline):
        """Test prompt builder failure is handled gracefully (lines 242-243)."""
        # Patch the module that gets imported inside the method
        with patch.dict("sys.modules", {"victor.agent.intelligent_prompt_builder": MagicMock()}):
            import sys

            mock_module = sys.modules["victor.agent.intelligent_prompt_builder"]
            mock_module.IntelligentPromptBuilder.create = AsyncMock(
                side_effect=Exception("Import error")
            )

            result = await pipeline._get_prompt_builder()

        assert result is None
        assert pipeline._prompt_builder is None


class TestLazyModeController:
    """Tests for lazy mode controller initialization (lines 262-263, 274-275)."""

    def test_get_mode_controller_with_rl_coordinator(self, pipeline):
        """Test mode controller with RL coordinator integration (lines 262-263)."""
        mock_learner = MagicMock()
        mock_controller_instance = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "victor.agent.adaptive_mode_controller": MagicMock(),
                "victor.agent.services.rl_runtime": MagicMock(),
            },
        ):
            import sys

            mock_amc_module = sys.modules["victor.agent.adaptive_mode_controller"]
            mock_amc_module.AdaptiveModeController.return_value = mock_controller_instance

            mock_coord_module = sys.modules["victor.agent.services.rl_runtime"]
            mock_coord_module.get_rl_coordinator.return_value.get_learner.return_value = (
                mock_learner
            )

            result = pipeline._get_mode_controller()

        assert result is not None

    def test_get_mode_controller_coordinator_failure(self, pipeline):
        """Test mode controller when coordinator fails (lines 262-263)."""
        mock_controller_instance = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "victor.agent.adaptive_mode_controller": MagicMock(),
                "victor.agent.services.rl_runtime": MagicMock(),
            },
        ):
            import sys

            mock_amc_module = sys.modules["victor.agent.adaptive_mode_controller"]
            mock_amc_module.AdaptiveModeController.return_value = mock_controller_instance

            mock_coord_module = sys.modules["victor.agent.services.rl_runtime"]
            mock_coord_module.get_rl_coordinator.side_effect = Exception("No coordinator")

            result = pipeline._get_mode_controller()

        assert result is not None

    def test_get_mode_controller_failure(self, pipeline):
        """Test mode controller init failure (lines 274-275)."""
        with patch.dict(
            "sys.modules",
            {
                "victor.agent.adaptive_mode_controller": MagicMock(),
            },
        ):
            import sys

            mock_module = sys.modules["victor.agent.adaptive_mode_controller"]
            mock_module.AdaptiveModeController.side_effect = Exception("Controller error")

            result = pipeline._get_mode_controller()

        assert result is None


class TestProviderQualityThresholds:
    """Tests for provider quality thresholds (lines 285-298)."""

    def test_get_provider_quality_thresholds_from_adapter(self, pipeline, mock_provider_adapter):
        """Test thresholds from provider adapter (lines 285-290)."""
        mock_provider_adapter.capabilities.quality_threshold = 0.85
        mock_provider_adapter.capabilities.grounding_strictness = 0.75
        pipeline._provider_adapter = mock_provider_adapter

        thresholds = pipeline.get_provider_quality_thresholds()

        assert thresholds["min_quality"] == 0.85
        assert thresholds["grounding_threshold"] == 0.75

    def test_get_provider_quality_thresholds_from_controller(self, pipeline):
        """Test thresholds fallback to controller (lines 292-295)."""
        pipeline._provider_adapter = None  # No adapter
        mock_controller = MagicMock()
        mock_controller.get_quality_thresholds.return_value = {
            "min_quality": 0.7,
            "grounding_threshold": 0.6,
        }
        pipeline._mode_controller = mock_controller

        thresholds = pipeline.get_provider_quality_thresholds()

        assert thresholds["min_quality"] == 0.7
        assert thresholds["grounding_threshold"] == 0.6

    def test_get_provider_quality_thresholds_default(self, pipeline):
        """Test default thresholds (lines 297-298)."""
        # Ensure no adapter and _get_mode_controller returns None to use fallback
        pipeline._provider_adapter = None
        pipeline._mode_controller = None
        with patch.object(pipeline, "_get_mode_controller", return_value=None):
            thresholds = pipeline.get_provider_quality_thresholds()

        assert thresholds["min_quality"] == 0.70
        assert thresholds["grounding_threshold"] == 0.65


class TestLazyQualityScorer:
    """Tests for lazy quality scorer initialization (lines 307-308)."""

    @pytest.mark.asyncio
    async def test_get_quality_scorer_failure(self, pipeline):
        """Test quality scorer failure is handled (lines 307-308)."""
        with patch.dict("sys.modules", {"victor.agent.response_quality": MagicMock()}):
            import sys

            mock_module = sys.modules["victor.agent.response_quality"]
            mock_module.ResponseQualityScorer.side_effect = Exception("Scorer error")

            result = await pipeline._get_quality_scorer()

        assert result is None


class TestLazyGroundingVerifier:
    """Tests for lazy grounding verifier initialization (lines 313-336)."""

    @pytest.mark.asyncio
    async def test_get_grounding_verifier_with_rl_learner(self, pipeline):
        """Test grounding verifier with RL learner (lines 317-327)."""
        mock_learner = MagicMock()
        mock_verifier_instance = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "victor.agent.grounding_verifier": MagicMock(),
                "victor.agent.services.rl_runtime": MagicMock(),
            },
        ):
            import sys

            mock_gv_module = sys.modules["victor.agent.grounding_verifier"]
            mock_gv_module.GroundingVerifier.return_value = mock_verifier_instance

            mock_coord_module = sys.modules["victor.agent.services.rl_runtime"]
            mock_coord_module.get_rl_coordinator.return_value.get_learner.return_value = (
                mock_learner
            )

            result = await pipeline._get_grounding_verifier()

        assert result is not None

    @pytest.mark.asyncio
    async def test_get_grounding_verifier_no_project_root(self, mock_provider_adapter):
        """Test grounding verifier not created without project root."""
        with patch(
            "victor.agent.runtime_intelligence_pipeline.get_provider_adapter",
            return_value=mock_provider_adapter,
        ):
            pipeline = RuntimeIntelligencePipeline(
                provider_name="anthropic",
                model="claude-3-sonnet",
                profile_name="test",
                project_root=None,  # No project root
            )

        result = await pipeline._get_grounding_verifier()

        assert result is None

    @pytest.mark.asyncio
    async def test_get_grounding_verifier_failure(self, pipeline):
        """Test grounding verifier failure handling (lines 334-335)."""
        with patch.dict("sys.modules", {"victor.agent.grounding_verifier": MagicMock()}):
            import sys

            mock_module = sys.modules["victor.agent.grounding_verifier"]
            mock_module.GroundingVerifier.side_effect = Exception("Verifier error")

            result = await pipeline._get_grounding_verifier()

        assert result is None


class TestLazyResilientExecutor:
    """Tests for lazy resilient executor initialization (lines 381-388)."""

    @pytest.mark.asyncio
    async def test_get_resilient_executor_failure(self, pipeline):
        """Test resilient executor failure handling (lines 386-387)."""
        with patch.dict("sys.modules", {"victor.agent.resilience": MagicMock()}):
            import sys

            mock_module = sys.modules["victor.agent.resilience"]
            mock_module.ResilientExecutor.side_effect = Exception("Executor error")

            result = await pipeline._get_resilient_executor()

        assert result is None


class TestOutputDeduplicator:
    """Tests for output deduplicator (lines 392-397, 405-406, 423-438)."""

    def test_get_output_deduplicator(self, pipeline):
        """Test lazy init of output deduplicator (lines 392-397)."""
        dedup = pipeline._get_output_deduplicator()

        assert dedup is not None
        assert pipeline._output_deduplicator is not None

    def test_should_enable_deduplication(self, pipeline_xai):
        """Test deduplication check for xAI (lines 405-406)."""
        # xai provider should have dedup enabled
        result = pipeline_xai._should_enable_deduplication()

        assert result is True

    def test_should_not_enable_deduplication(self, pipeline):
        """Test deduplication not enabled for anthropic."""
        result = pipeline._should_enable_deduplication()

        assert result is False

    def test_deduplicate_response_disabled(self, pipeline):
        """Test deduplication returns original when disabled."""
        response = "This is a test response."

        result, stats = pipeline.deduplicate_response(response)

        assert result == response
        assert stats["deduplication_applied"] is False

    def test_deduplicate_response_enabled(self, pipeline_xai):
        """Test deduplication is applied (lines 423-438)."""
        # Response with duplicate blocks
        response = """First paragraph with some content.

This is a duplicate paragraph.

Some more unique content here.

This is a duplicate paragraph."""

        result, stats = pipeline_xai.deduplicate_response(response)

        assert stats["deduplication_applied"] is True
        assert stats["provider"] == "xai"

    def test_deduplicate_empty_response(self, pipeline_xai):
        """Test deduplication handles empty response."""
        result, stats = pipeline_xai.deduplicate_response("")

        assert result == ""
        assert stats["deduplication_applied"] is False


# =============================================================================
# Test Grounding Event Emission
# =============================================================================


class TestGroundingEventEmission:
    """Tests for grounding RL event emission (lines 359, 376-377)."""

    def test_emit_grounding_event_success(self, pipeline):
        """Test grounding event emission on success."""
        with patch.dict(
            "sys.modules",
            {
                "victor.framework.rl.hooks": MagicMock(),
            },
        ):
            import sys

            mock_hooks_module = sys.modules["victor.framework.rl.hooks"]
            mock_hooks = MagicMock()
            mock_hooks_module.get_rl_hooks.return_value = mock_hooks
            mock_hooks_module.RLEvent = MagicMock()
            mock_hooks_module.RLEventType = MagicMock()

            pipeline._emit_grounding_event(
                is_grounded=True,
                grounding_score=0.9,
                task_type="analysis",
            )

            mock_hooks.emit.assert_called_once()

    def test_emit_grounding_event_no_hooks(self, pipeline):
        """Test grounding event when hooks not available (line 359)."""
        with patch.dict(
            "sys.modules",
            {
                "victor.framework.rl.hooks": MagicMock(),
            },
        ):
            import sys

            mock_hooks_module = sys.modules["victor.framework.rl.hooks"]
            mock_hooks_module.get_rl_hooks.return_value = None
            mock_hooks_module.RLEvent = MagicMock()
            mock_hooks_module.RLEventType = MagicMock()

            # Should not raise
            pipeline._emit_grounding_event(
                is_grounded=True,
                grounding_score=0.9,
                task_type="analysis",
            )

    def test_emit_grounding_event_failure(self, pipeline):
        """Test grounding event emission failure handling (lines 376-377)."""
        with patch.dict(
            "sys.modules",
            {
                "victor.framework.rl.hooks": MagicMock(),
            },
        ):
            import sys

            mock_hooks_module = sys.modules["victor.framework.rl.hooks"]
            mock_hooks_module.get_rl_hooks.side_effect = Exception("Hook error")

            # Should not raise, just log
            pipeline._emit_grounding_event(
                is_grounded=True,
                grounding_score=0.9,
                task_type="analysis",
            )


# =============================================================================
# Test prepare_request (lines 475-561)
# =============================================================================


class TestPrepareRequest:
    """Tests for prepare_request method."""

    @pytest.mark.asyncio
    async def test_prepare_request_basic(self, pipeline, mock_prompt_builder, mock_mode_controller):
        """Test basic prepare_request functionality (lines 475-561)."""
        pipeline._prompt_builder = mock_prompt_builder
        pipeline._mode_controller = mock_mode_controller

        context = await pipeline.prepare_request(
            task="Analyze the auth module",
            task_type="analysis",
            current_mode="explore",
        )

        assert isinstance(context, RequestContext)
        assert context.system_prompt == "System prompt content"
        assert context.recommended_mode == "build"
        assert context.should_continue is True
        assert pipeline._stats.total_requests == 1

    @pytest.mark.asyncio
    async def test_prepare_request_budget_increase(
        self, pipeline, mock_prompt_builder, mock_mode_controller
    ):
        """Test budget increase when RL suggests higher (lines 524-531)."""
        mock_mode_controller.get_optimal_tool_budget.return_value = 25  # Higher than default
        pipeline._prompt_builder = mock_prompt_builder
        pipeline._mode_controller = mock_mode_controller

        context = await pipeline.prepare_request(
            task="Complex task",
            task_type="design",
            tool_budget=20,
        )

        # Should recommend higher budget based on learning
        assert context.recommended_tool_budget >= 20

    @pytest.mark.asyncio
    async def test_prepare_request_without_components(self, pipeline):
        """Test prepare_request when components fail to init."""
        with (
            patch.object(
                pipeline,
                "_get_prompt_builder",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch.object(pipeline, "_get_mode_controller", return_value=None),
        ):
            context = await pipeline.prepare_request(
                task="Simple task",
                task_type="general",
            )

        assert context.system_prompt == ""
        assert context.recommended_mode == "explore"  # fallback to current_mode


# =============================================================================
# Test process_response (lines 599, 607-616)
# =============================================================================


class TestProcessResponse:
    """Tests for process_response method."""

    @pytest.mark.asyncio
    async def test_process_response_basic(
        self, pipeline, mock_quality_scorer, mock_grounding_verifier
    ):
        """Test basic response processing."""
        pipeline._quality_scorer = mock_quality_scorer
        pipeline._grounding_verifier = mock_grounding_verifier
        pipeline._stats.total_requests = 1  # Simulate prepare_request was called

        result = await pipeline.process_response(
            response="The auth module uses JWT...",
            query="How does auth work?",
            tool_calls=5,
            success=True,
        )

        assert isinstance(result, ResponseResult)
        assert result.quality_score == 0.85
        assert result.is_grounded is True

    @pytest.mark.asyncio
    async def test_process_response_with_deduplication(self, pipeline_xai):
        """Test response processing with deduplication (line 599)."""
        pipeline_xai._stats.total_requests = 1

        result = await pipeline_xai.process_response(
            response="Repeated content.\n\nRepeated content.",
            tool_calls=2,
            success=True,
        )

        assert isinstance(result, ResponseResult)

    @pytest.mark.asyncio
    async def test_process_response_with_quality_scoring(
        self, pipeline, mock_quality_scorer, mock_grounding_verifier
    ):
        """Test quality scoring in response (lines 607-616)."""
        pipeline._quality_scorer = mock_quality_scorer
        pipeline._grounding_verifier = mock_grounding_verifier
        pipeline._stats.total_requests = 1

        result = await pipeline.process_response(
            response="Detailed response",
            query="Test query",
            success=True,
        )

        assert result.quality_score == 0.85
        assert "relevance" in result.quality_details
        assert "Add more context" in result.improvement_suggestions

    @pytest.mark.asyncio
    async def test_process_response_grounding_failure_retry(
        self, pipeline, mock_grounding_verifier_failed
    ):
        """Test grounding failure allows retry."""
        pipeline._grounding_verifier = mock_grounding_verifier_failed
        pipeline._grounding_failure_count = 0
        pipeline._stats.total_requests = 1

        result = await pipeline.process_response(
            response="Reference to nonexistent file",
            success=True,
        )

        assert result.is_grounded is False
        assert result.should_retry is True
        assert result.should_finalize is False
        assert len(result.grounding_feedback) > 0

    @pytest.mark.asyncio
    async def test_process_response_soft_finalizes_substantial_analysis_reference_issues(
        self, pipeline
    ):
        """Read-heavy answers should not trigger hidden retry turns for reference-only issues."""
        issue = MagicMock()
        issue.issue_type = MagicMock(value="file_not_found")
        issue.description = "Referenced file does not exist"

        grounding_result = MagicMock()
        grounding_result.is_grounded = False
        grounding_result.confidence = 0.45
        grounding_result.issues = [issue]
        grounding_result.generate_feedback_prompt = MagicMock(
            return_value="Please verify the referenced file path exists."
        )

        verifier = AsyncMock()
        verifier.verify = AsyncMock(return_value=grounding_result)

        pipeline._grounding_verifier = verifier
        pipeline._grounding_failure_count = 0
        pipeline._stats.total_requests = 1

        result = await pipeline.process_response(
            response="A" * 800,
            success=True,
            tool_calls=4,
            task_type="analysis",
        )

        assert result.is_grounded is False
        assert result.should_finalize is True
        assert result.should_retry is False
        assert "soft grounding finalize" in result.finalize_reason

    @pytest.mark.asyncio
    async def test_process_response_keeps_retry_for_code_task_reference_issues(self, pipeline):
        """Action/code tasks should still retry when grounding fails on references."""
        issue = MagicMock()
        issue.issue_type = MagicMock(value="file_not_found")
        issue.description = "Referenced file does not exist"

        grounding_result = MagicMock()
        grounding_result.is_grounded = False
        grounding_result.confidence = 0.45
        grounding_result.issues = [issue]
        grounding_result.generate_feedback_prompt = MagicMock(
            return_value="Please verify the referenced file path exists."
        )

        verifier = AsyncMock()
        verifier.verify = AsyncMock(return_value=grounding_result)

        pipeline._grounding_verifier = verifier
        pipeline._grounding_failure_count = 0
        pipeline._stats.total_requests = 1

        result = await pipeline.process_response(
            response="A" * 800,
            success=True,
            tool_calls=4,
            task_type="code_generation",
        )

        assert result.is_grounded is False
        assert result.should_retry is True
        assert result.should_finalize is False

    @pytest.mark.asyncio
    async def test_process_response_builds_grounding_evidence_summary(self, pipeline):
        """Grounding verification should return structured evidence references."""
        grounding_result = MagicMock()
        grounding_result.is_grounded = False
        grounding_result.confidence = 0.55
        grounding_result.issues = []
        grounding_result.verified_references = ["src/main.rs", "src/main.rs:10"]
        grounding_result.unverified_references = ["src/missing.rs:99"]
        grounding_result.generate_feedback_prompt = MagicMock(return_value="")

        verifier = AsyncMock()
        verifier.verify = AsyncMock(return_value=grounding_result)
        pipeline._grounding_verifier = verifier
        pipeline._stats.total_requests = 1

        result = await pipeline.process_response(
            response="See src/main.rs:10 and src/missing.rs:99.",
            success=True,
            tool_calls=2,
            task_type="analysis",
        )

        assert result.grounding_verified_references == ["src/main.rs", "src/main.rs:10"]
        assert result.grounding_unverified_references == ["src/missing.rs:99"]
        assert "- Verified: src/main.rs, src/main.rs:10" in result.grounding_evidence_summary
        assert "- Unverified: src/missing.rs:99" in result.grounding_evidence_summary

    @pytest.mark.asyncio
    async def test_process_response_grounding_failure_finalize(
        self, pipeline, mock_grounding_verifier_failed
    ):
        """Test grounding failure forces finalize after max retries."""
        pipeline._grounding_verifier = mock_grounding_verifier_failed
        pipeline._grounding_failure_count = 1  # Already one failure
        pipeline._max_grounding_retries = 1
        pipeline._stats.total_requests = 1

        result = await pipeline.process_response(
            response="Another bad reference",
            success=True,
        )

        assert result.is_grounded is False
        assert result.should_finalize is True
        assert "grounding failure limit exceeded" in result.finalize_reason

    @pytest.mark.asyncio
    async def test_process_response_resets_grounding_on_success(
        self, pipeline, mock_grounding_verifier
    ):
        """Test grounding counter resets on success."""
        pipeline._grounding_verifier = mock_grounding_verifier
        pipeline._grounding_failure_count = 5
        pipeline._stats.total_requests = 1

        await pipeline.process_response(
            response="Good response",
            success=True,
        )

        assert pipeline._grounding_failure_count == 0

    @pytest.mark.asyncio
    async def test_process_response_without_prepare(self, pipeline):
        """Test process_response without prior prepare_request."""
        # total_requests == 0 - should handle gracefully
        result = await pipeline.process_response(
            response="Test response",
            success=True,
        )

        assert isinstance(result, ResponseResult)
        assert pipeline._stats.total_requests == 1  # Gets initialized


# =============================================================================
# Test Observer Pattern (lines 739-743, 842, 846-847)
# =============================================================================


class TestObserverPattern:
    """Tests for observer pattern implementation."""

    @pytest.mark.asyncio
    async def test_notify_observers(self, pipeline):
        """Test observers are notified (lines 739-743)."""
        results_collected = []

        def observer(result: ResponseResult):
            results_collected.append(result)

        pipeline.add_observer(observer)
        pipeline._stats.total_requests = 1

        await pipeline.process_response(
            response="Test",
            success=True,
        )

        assert len(results_collected) == 1
        assert isinstance(results_collected[0], ResponseResult)

    @pytest.mark.asyncio
    async def test_observer_error_handled(self, pipeline):
        """Test observer errors are caught (lines 742-743)."""

        def failing_observer(result: ResponseResult):
            raise ValueError("Observer error")

        pipeline.add_observer(failing_observer)
        pipeline._stats.total_requests = 1

        # Should not raise
        result = await pipeline.process_response(
            response="Test",
            success=True,
        )

        assert isinstance(result, ResponseResult)

    def test_add_observer(self, pipeline):
        """Test add_observer method (line 842)."""
        observer = MagicMock()

        pipeline.add_observer(observer)

        assert observer in pipeline._observers

    def test_remove_observer(self, pipeline):
        """Test remove_observer method (lines 846-847)."""
        observer = MagicMock()
        pipeline.add_observer(observer)

        pipeline.remove_observer(observer)

        assert observer not in pipeline._observers

    def test_remove_nonexistent_observer(self, pipeline):
        """Test removing observer that was never added."""
        observer = MagicMock()

        # Should not raise
        pipeline.remove_observer(observer)


# =============================================================================
# Test execute_with_resilience (lines 777-800)
# =============================================================================


class TestExecuteWithResilience:
    """Tests for execute_with_resilience method."""

    @pytest.mark.asyncio
    async def test_execute_without_resilience(self, pipeline):
        """Test execution when resilient executor unavailable (lines 777-779)."""
        mock_provider = AsyncMock()
        mock_provider.chat = AsyncMock(return_value={"content": "response"})

        with patch.object(
            pipeline,
            "_get_resilient_executor",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await pipeline.execute_with_resilience(
                provider=mock_provider,
                messages=[{"role": "user", "content": "test"}],
            )

        assert result == {"content": "response"}
        mock_provider.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_with_resilience_success(self, pipeline, mock_resilient_executor):
        """Test successful resilient execution (lines 781-797)."""
        pipeline._resilient_executor = mock_resilient_executor
        mock_provider = AsyncMock()

        with patch.object(
            pipeline,
            "_get_resilient_executor",
            new_callable=AsyncMock,
            return_value=mock_resilient_executor,
        ):
            result = await pipeline.execute_with_resilience(
                provider=mock_provider,
                messages=[{"role": "user", "content": "test"}],
            )

        assert result == {"content": "test response"}

    @pytest.mark.asyncio
    async def test_execute_with_resilience_circuit_trip(self, pipeline, mock_resilient_executor):
        """Test circuit breaker trip updates stats (lines 798-800)."""
        mock_resilient_executor.execute = AsyncMock(side_effect=Exception("Circuit open"))
        pipeline._resilient_executor = mock_resilient_executor
        mock_provider = AsyncMock()

        with (
            patch.object(
                pipeline,
                "_get_resilient_executor",
                new_callable=AsyncMock,
                return_value=mock_resilient_executor,
            ),
            pytest.raises(Exception, match="Circuit open"),
        ):
            await pipeline.execute_with_resilience(
                provider=mock_provider,
                messages=[],
            )

        assert pipeline._stats.circuit_breaker_trips == 1


# =============================================================================
# Test should_continue (lines 822-838)
# =============================================================================


class TestShouldContinue:
    """Tests for should_continue method."""

    def test_should_continue_with_controller(self, pipeline, mock_mode_controller):
        """Test should_continue delegates to controller (lines 822-829)."""
        pipeline._mode_controller = mock_mode_controller
        mock_mode_controller.should_continue.return_value = (True, "Keep going")

        result, reason = pipeline.should_continue(
            tool_calls_made=5,
            tool_budget=20,
            quality_score=0.7,
            iteration_count=5,
            iteration_budget=30,
        )

        assert result is True
        assert reason == "Keep going"

    def test_should_continue_tool_budget_exhausted(self, pipeline):
        """Test fallback: tool budget exhausted (lines 832-833)."""
        result, reason = pipeline.should_continue(
            tool_calls_made=20,
            tool_budget=20,
            quality_score=0.7,
            iteration_count=5,
            iteration_budget=30,
        )

        assert result is False
        assert "Tool budget exhausted" in reason

    def test_should_continue_iteration_budget_exhausted(self, pipeline):
        """Test fallback: iteration budget exhausted (lines 834-835)."""
        result, reason = pipeline.should_continue(
            tool_calls_made=5,
            tool_budget=20,
            quality_score=0.7,
            iteration_count=30,
            iteration_budget=30,
        )

        assert result is False
        assert "Iteration budget exhausted" in reason

    def test_should_continue_high_quality(self, pipeline):
        """Test fallback: high quality achieved (lines 836-837)."""
        # Ensure _get_mode_controller returns None to use fallback logic
        pipeline._mode_controller = None
        with patch.object(pipeline, "_get_mode_controller", return_value=None):
            result, reason = pipeline.should_continue(
                tool_calls_made=5,
                tool_budget=20,
                quality_score=0.9,  # > 0.85 threshold
                iteration_count=5,
                iteration_budget=30,
            )

        assert result is False
        assert "High quality achieved" in reason

    def test_should_continue_normal(self, pipeline):
        """Test fallback: continue processing (line 838)."""
        result, reason = pipeline.should_continue(
            tool_calls_made=5,
            tool_budget=20,
            quality_score=0.7,
            iteration_count=5,
            iteration_budget=30,
        )

        assert result is True
        assert reason == "Continue processing"


# =============================================================================
# Test Stats and Session Management (lines 852-860, 864-869, 873-884)
# =============================================================================


class TestStatsAndSession:
    """Tests for get_stats, reset_session, get_learning_summary."""

    def test_get_stats(self, pipeline, mock_prompt_builder, mock_mode_controller):
        """Test get_stats with lazy component updates (lines 852-860)."""
        pipeline._prompt_builder = mock_prompt_builder
        pipeline._mode_controller = mock_mode_controller
        mock_prompt_builder.get_profile_stats.return_value = {"cache_state": "warm"}
        mock_mode_controller.get_session_stats.return_value = {"mode_transitions": 3}

        stats = pipeline.get_stats()

        assert isinstance(stats, PipelineStats)
        assert stats.cache_state == "warm"
        assert stats.mode_transitions == 3

    def test_get_stats_no_components(self, pipeline):
        """Test get_stats without initialized components."""
        stats = pipeline.get_stats()

        assert isinstance(stats, PipelineStats)
        assert stats.profile_name == "test-profile"

    def test_reset_session(self, pipeline, mock_mode_controller):
        """Test reset_session clears state (lines 864-869)."""
        pipeline._mode_controller = mock_mode_controller
        pipeline._current_context = MagicMock()
        pipeline._stats.total_requests = 10

        pipeline.reset_session()

        assert pipeline._current_context is None
        assert pipeline._stats.total_requests == 0
        mock_mode_controller.reset_session.assert_called_once()

    def test_reset_session_without_controller(self, pipeline):
        """Test reset_session without mode controller."""
        pipeline._stats.total_requests = 5

        pipeline.reset_session()

        assert pipeline._stats.total_requests == 0

    def test_get_learning_summary(self, pipeline, mock_prompt_builder, mock_mode_controller):
        """Test get_learning_summary (lines 873-884)."""
        pipeline._prompt_builder = mock_prompt_builder
        pipeline._mode_controller = mock_mode_controller
        pipeline._stats.total_requests = 20
        pipeline._stats.successful_requests = 18

        summary = pipeline.get_learning_summary()

        assert summary["profile_name"] == "test-profile"
        assert summary["total_requests"] == 20
        assert summary["success_rate"] == 0.9
        assert "prompt_profile" in summary
        assert "mode_session" in summary

    def test_get_learning_summary_no_requests(self, pipeline):
        """Test get_learning_summary with zero requests."""
        summary = pipeline.get_learning_summary()

        assert summary["success_rate"] == 0.0  # 0/1 to avoid division by zero


# =============================================================================
# Test Module-Level Functions (lines 908-918, 923)
# =============================================================================


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    @pytest.mark.asyncio
    async def test_get_runtime_intelligence_pipeline(self, mock_provider_adapter):
        """Test get_runtime_intelligence_pipeline creates and caches agents (lines 894-914)."""
        clear_runtime_intelligence_pipeline_cache()  # Start fresh

        with patch(
            "victor.agent.runtime_intelligence_pipeline.get_provider_adapter",
            return_value=mock_provider_adapter,
        ):
            pipeline1 = await get_runtime_intelligence_pipeline(
                provider_name="anthropic",
                model="claude-3-sonnet",
            )
            pipeline2 = await get_runtime_intelligence_pipeline(
                provider_name="anthropic",
                model="claude-3-sonnet",
            )

        # Should return same instance (cached)
        assert pipeline1 is pipeline2

    @pytest.mark.asyncio
    async def test_get_runtime_intelligence_pipeline_different_keys(self, mock_provider_adapter):
        """Test get_runtime_intelligence_pipeline creates different instances for different keys."""
        clear_runtime_intelligence_pipeline_cache()

        with patch(
            "victor.agent.runtime_intelligence_pipeline.get_provider_adapter",
            return_value=mock_provider_adapter,
        ):
            pipeline1 = await get_runtime_intelligence_pipeline(
                provider_name="anthropic",
                model="claude-3-sonnet",
            )
            pipeline2 = await get_runtime_intelligence_pipeline(
                provider_name="openai",
                model="gpt-4",
            )

        assert pipeline1 is not pipeline2

    def test_clear_runtime_intelligence_pipeline_cache(self, mock_provider_adapter):
        """Test clear_runtime_intelligence_pipeline_cache (line 919)."""
        # Add something to cache
        from victor.agent.runtime_intelligence_pipeline import (
            _runtime_intelligence_pipeline_cache,
        )

        with patch(
            "victor.agent.runtime_intelligence_pipeline.get_provider_adapter",
            return_value=mock_provider_adapter,
        ):
            test_pipeline = RuntimeIntelligencePipeline(
                provider_name="test",
                model="test",
                profile_name="test",
            )
        _runtime_intelligence_pipeline_cache[("test", "test", "test", None)] = test_pipeline

        clear_runtime_intelligence_pipeline_cache()

        assert len(_runtime_intelligence_pipeline_cache) == 0

    def test_get_optimal_tool_budget_delegates_to_mode_controller(self, mock_provider_adapter):
        """Test get_optimal_tool_budget delegates to mode controller."""
        pipeline = RuntimeIntelligencePipeline(
            provider_name="test",
            model="test",
            profile_name="test",
        )

        # Mock _get_mode_controller to return None (not initialized)
        with patch.object(pipeline, "_get_mode_controller", return_value=None):
            # Without mode controller, returns 0
            assert pipeline.get_optimal_tool_budget("analysis") == 0

        # With mode controller mocked
        mock_controller = MagicMock()
        mock_controller.get_optimal_tool_budget.return_value = 25

        with patch.object(pipeline, "_get_mode_controller", return_value=mock_controller):
            budget = pipeline.get_optimal_tool_budget("analysis")

            assert budget == 25
            mock_controller.get_optimal_tool_budget.assert_called_once_with("analysis")


# =============================================================================
# Test Constants
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_providers_with_repetition_issues(self):
        """Test PROVIDERS_WITH_REPETITION_ISSUES constant."""
        assert "xai" in PROVIDERS_WITH_REPETITION_ISSUES
        assert "grok" in PROVIDERS_WITH_REPETITION_ISSUES
        assert "x-ai" in PROVIDERS_WITH_REPETITION_ISSUES
        assert "anthropic" not in PROVIDERS_WITH_REPETITION_ISSUES
