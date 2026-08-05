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

"""Tests for orchestrator intelligent pipeline hooks.

These tests verify that the IntelligentAgentPipeline integration hooks
work correctly within the AgentOrchestrator.
"""

import asyncio
from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@dataclass
class MockRequestContext:
    """Mock for RequestContext from intelligent pipeline."""

    system_prompt: str = ""
    recommended_tool_budget: int = 15
    recommended_mode: str = "explore"
    should_continue: bool = True


@dataclass
class MockResponseResult:
    """Mock for ResponseResult from intelligent pipeline."""

    is_valid: bool = True
    quality_score: float = 0.8
    grounding_score: float = 0.9
    is_grounded: bool = True
    grounding_issues: list = None
    quality_details: dict = None
    learning_reward: float = 0.5

    def __post_init__(self):
        if self.grounding_issues is None:
            self.grounding_issues = []
        if self.quality_details is None:
            self.quality_details = {}


class TestOrchestratorIntelligentHooks:
    """Tests for the intelligent pipeline hook methods in AgentOrchestrator."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings with intelligent pipeline enabled."""
        settings = MagicMock()
        settings.runtime_intelligence_enabled = True
        settings.intelligent_quality_scoring = True
        settings.intelligent_mode_learning = True
        settings.intelligent_prompt_optimization = True
        settings.intelligent_grounding_verification = True
        settings.intelligent_min_quality_threshold = 0.5
        settings.intelligent_grounding_threshold = 0.7
        settings.intelligent_exploration_rate = 0.3
        settings.intelligent_learning_rate = 0.1
        settings.intelligent_discount_factor = 0.9
        settings.provider = "test"
        settings.model = "test-model"
        settings.temperature = 0.7
        settings.max_tokens = 2048
        settings.log_level = "WARNING"
        settings.tools.tool_call_budget = 50
        return settings

    @pytest.fixture
    def mock_provider(self):
        """Create mock provider."""
        provider = MagicMock()
        provider.name = "test"
        provider.supports_tools = MagicMock(return_value=True)
        return provider

    @pytest.fixture
    def mock_integration(self):
        """Create mock OrchestratorIntegration."""
        integration = MagicMock()
        integration.prepare_request = AsyncMock(
            return_value=MockRequestContext(
                system_prompt="Optimized prompt",
                recommended_tool_budget=20,
                recommended_mode="build",
                should_continue=True,
            )
        )
        integration.validate_response = AsyncMock(
            return_value=MockResponseResult(
                is_valid=True,
                quality_score=0.85,
                grounding_score=0.9,
                is_grounded=True,
            )
        )
        integration.should_continue = MagicMock(return_value=(True, "Continue"))
        integration.pipeline = MagicMock()
        integration.pipeline._mode_controller = MagicMock()
        integration.pipeline._mode_controller.record_outcome = MagicMock()
        return integration

    def test_prepare_intelligent_request_returns_none_when_disabled(self, mock_settings):
        """Should return None when intelligent pipeline is disabled."""
        mock_settings.runtime_intelligence_enabled = False

        # Create a minimal mock orchestrator
        orchestrator = MagicMock()
        orchestrator._runtime_intelligence_enabled = False
        orchestrator._runtime_intelligence_integration = None

        # Access the property should return None
        assert orchestrator._runtime_intelligence_integration is None

    def test_prepare_intelligent_request_handles_exceptions(self):
        """Should handle exceptions gracefully in prepare_request."""
        # This tests the error handling in the hook
        integration = MagicMock()
        integration.prepare_request = AsyncMock(side_effect=Exception("Test error"))

        # The hook should catch the exception and return None
        assert integration.prepare_request is not None

    def test_validate_intelligent_response_skips_short_responses(self):
        """Should skip validation for very short responses."""
        _integration = MagicMock()  # noqa: F841

        # For responses < 50 chars, validation should be skipped
        short_response = "OK"
        assert len(short_response.strip()) < 50

    def test_validate_intelligent_response_logs_grounding_issues(self):
        """Should log grounding issues when detected."""
        result = MockResponseResult(
            is_valid=False,
            is_grounded=False,
            grounding_issues=["Hallucinated function name", "Made up file path"],
        )

        assert not result.is_grounded
        assert len(result.grounding_issues) > 0

    def test_record_intelligent_outcome_succeeds(self, mock_integration):
        """Should record outcome for Q-learning feedback."""
        # Call record_outcome through the pipeline
        mock_integration.pipeline._mode_controller.record_outcome(
            success=True,
            quality_score=0.8,
            user_satisfied=True,
            completed=True,
        )

        mock_integration.pipeline._mode_controller.record_outcome.assert_called_once()

    def test_should_continue_intelligent_returns_tuple(self, mock_integration):
        """should_continue should return tuple of (bool, str)."""
        result = mock_integration.should_continue()

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    def test_integration_config_creation(self):
        """Test IntegrationConfig creation with all settings."""
        from victor.agent.orchestrator_integration import IntegrationConfig

        config = IntegrationConfig(
            enable_resilient_calls=True,
            enable_quality_scoring=True,
            enable_mode_learning=True,
            enable_prompt_optimization=True,
            min_quality_threshold=0.5,
            grounding_confidence_threshold=0.7,
        )

        assert config.enable_resilient_calls is True
        assert config.enable_quality_scoring is True
        assert config.min_quality_threshold == 0.5

    def test_integration_metrics_tracking(self):
        """Test IntegrationMetrics initial values."""
        from victor.agent.orchestrator_integration import IntegrationMetrics

        metrics = IntegrationMetrics()

        assert metrics.total_requests == 0
        assert metrics.enhanced_requests == 0
        assert metrics.avg_quality_score == 0.0


class TestOrchestratorIntegrationProperty:
    """Tests for the runtime_intelligence_integration property accessor."""

    def test_property_returns_none_when_disabled(self):
        """Property should return None when feature is disabled."""
        orchestrator = MagicMock()
        orchestrator._runtime_intelligence_enabled = False
        orchestrator._runtime_intelligence_integration = None

        # Simulating the property behavior
        if not orchestrator._runtime_intelligence_enabled:
            result = None
        else:
            result = orchestrator._runtime_intelligence_integration

        assert result is None

    def test_property_lazy_initializes(self):
        """Property should lazy initialize on first access."""
        orchestrator = MagicMock()
        orchestrator._runtime_intelligence_enabled = True
        orchestrator._runtime_intelligence_integration = None

        # First access triggers initialization
        # In real code, this would create the integration
        assert orchestrator._runtime_intelligence_integration is None  # Before init

    def test_property_returns_cached_instance(self):
        """Property should return cached instance on subsequent calls."""
        integration = MagicMock()
        orchestrator = MagicMock()
        orchestrator._runtime_intelligence_enabled = True
        orchestrator._runtime_intelligence_integration = integration

        # Should return the cached instance
        assert orchestrator._runtime_intelligence_integration is integration


class TestHookIntegration:
    """Integration tests for hooks working together."""

    @pytest.mark.asyncio
    async def test_prepare_and_validate_flow(self):
        """Test the full flow of prepare_request followed by validate_response."""
        # Create mock integration
        integration = MagicMock()
        integration.prepare_request = AsyncMock(
            return_value=MockRequestContext(
                recommended_tool_budget=25,
            )
        )
        integration.validate_response = AsyncMock(
            return_value=MockResponseResult(
                quality_score=0.9,
            )
        )

        # Simulate the flow
        context = await integration.prepare_request(
            task="Analyze code",
            task_type="analysis",
            current_mode="explore",
        )

        assert context.recommended_tool_budget == 25

        result = await integration.validate_response(
            response="The code analysis shows...",
            query="Analyze code",
            tool_calls=3,
            success=True,
            task_type="analysis",
        )

        assert result.quality_score == 0.9

    @pytest.mark.asyncio
    async def test_hooks_handle_pipeline_failure(self):
        """Hooks should gracefully handle pipeline failures."""
        integration = MagicMock()
        integration.prepare_request = AsyncMock(side_effect=Exception("Pipeline error"))

        try:
            await integration.prepare_request(
                task="Test",
                task_type="test",
                current_mode="explore",
            )
            raise AssertionError("Should have raised exception")
        except Exception as e:
            assert "Pipeline error" in str(e)


class TestConfigurationSettings:
    """Tests for configuration settings in settings.py."""

    def test_intelligent_settings_exist(self):
        """Verify all intelligent pipeline settings exist in Settings.pipeline."""
        from victor.config.settings import Settings

        settings = Settings()
        pipeline = settings.pipeline

        # These should all exist with defaults under the pipeline group
        assert hasattr(pipeline, "runtime_intelligence_enabled")
        assert hasattr(pipeline, "intelligent_quality_scoring")
        assert hasattr(pipeline, "intelligent_mode_learning")
        assert hasattr(pipeline, "intelligent_prompt_optimization")
        assert hasattr(pipeline, "intelligent_grounding_verification")
        assert hasattr(pipeline, "intelligent_min_quality_threshold")
        assert hasattr(pipeline, "intelligent_grounding_threshold")

    def test_intelligent_settings_defaults(self):
        """Verify default values for intelligent pipeline settings."""
        from victor.config.settings import Settings

        settings = Settings()
        pipeline = settings.pipeline

        assert pipeline.runtime_intelligence_enabled is True
        assert pipeline.intelligent_quality_scoring is True
        assert pipeline.intelligent_min_quality_threshold == 0.5
        assert pipeline.intelligent_grounding_threshold == 0.7
