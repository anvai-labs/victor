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

"""
Unit tests for smart routing engine.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from victor.providers.smart_router import (
    RoutingDecision,
    RoutingDecisionEngine,
    SmartRoutingProvider,
)
from victor.providers.routing_config import SmartRoutingConfig, RoutingProfile
from victor.providers.performance_tracker import (
    ProviderPerformanceTracker,
    RequestMetric,
)
from victor.providers.resource_detector import (
    ResourceAvailabilityDetector,
    GPUAvailability,
)
from victor.providers.health import ProviderHealthChecker, ProviderHealthResult
from victor.providers.base import Message


class TestRoutingDecision:
    """Tests for RoutingDecision dataclass."""

    def test_routing_decision_creation(self):
        """Test creating a routing decision."""
        decision = RoutingDecision(
            selected_provider="ollama",
            fallback_chain=["anthropic", "openai"],
            rationale="Test rationale",
            confidence=0.85,
            factors={"health": 1.0, "cost": 0.7},
        )

        assert decision.selected_provider == "ollama"
        assert len(decision.fallback_chain) == 2
        assert decision.confidence == 0.85
        assert decision.factors["health"] == 1.0

    def test_to_dict(self):
        """Test converting decision to dictionary."""
        decision = RoutingDecision(
            selected_provider="anthropic",
            fallback_chain=["openai"],
            rationale="Test",
            confidence=0.9,
        )

        data = decision.to_dict()

        assert data["selected_provider"] == "anthropic"
        assert data["fallback_chain"] == ["openai"]
        assert data["confidence"] == 0.9

    def test_to_topology_hints(self):
        """Topology hints should expose stable provider-routing fields."""
        decision = RoutingDecision(
            selected_provider="ollama",
            fallback_chain=["openai"],
            rationale="Prefer local provider",
            confidence=0.78,
            factors={
                "health": 1.0,
                "resources": 1.0,
                "cost": 0.9,
                "latency": 0.3,
                "performance": 0.5,
            },
        )

        hints = decision.to_topology_hints()

        assert hints["provider_hint"] == "ollama"
        assert hints["provider_locality"] == "local"
        assert hints["fallback_chain"] == ["openai"]
        assert hints["health_score"] == 1.0
        assert hints["latency_score"] == 0.3
        assert hints["provider_degraded"] is False
        assert hints["provider_degradation_reasons"] == []


class TestRoutingDecisionEngine:
    """Tests for RoutingDecisionEngine."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return SmartRoutingConfig(
            enabled=True,
            profile_name="balanced",
            learning_enabled=False,  # Disable for predictable tests
        )

    @pytest.fixture
    def tracker(self):
        """Create performance tracker."""
        return ProviderPerformanceTracker(db=None)

    @pytest.fixture
    def detector(self):
        """Create resource detector."""
        return ResourceAvailabilityDetector()

    @pytest.fixture
    def checker(self):
        """Create health checker."""
        return ProviderHealthChecker()

    @pytest.fixture
    def engine(self, config, tracker, detector, checker):
        """Create routing engine."""
        return RoutingDecisionEngine(
            config=config,
            performance_tracker=tracker,
            resource_detector=detector,
            health_checker=checker,
            available_providers=["ollama", "anthropic", "openai"],
        )

    def test_engine_initialization(self, engine):
        """Test engine initialization."""
        assert engine is not None
        assert len(engine.available_providers) == 3
        assert engine.profile is not None

    @pytest.mark.asyncio
    async def test_decide_basic(self, engine):
        """Test basic routing decision."""
        decision = await engine.decide(task_type="default")

        assert isinstance(decision, RoutingDecision)
        assert decision.selected_provider in ["ollama", "anthropic", "openai"]
        assert 0.0 <= decision.confidence <= 1.0
        assert len(decision.rationale) > 0

    @pytest.mark.asyncio
    async def test_decide_with_preferred_providers(self, engine):
        """Test routing with user-specified providers."""
        decision = await engine.decide(
            task_type="default",
            preferred_providers=["anthropic"],
        )

        assert decision.selected_provider == "anthropic"

    @pytest.mark.asyncio
    async def test_decide_invalid_preferred_providers(self, engine):
        """Test routing with invalid preferred providers."""
        decision = await engine.decide(
            task_type="default",
            preferred_providers=["invalid_provider"],
        )

        # Should fall back to profile candidates
        assert decision.selected_provider in ["ollama", "anthropic", "openai"]

    @pytest.mark.asyncio
    async def test_score_provider_health(self, engine):
        """Test provider health scoring."""
        # Mock health checker to return healthy
        engine.checker._provider_health["ollama"] = ProviderHealthResult(
            healthy=True,
            provider="ollama",
            model="test",
        )

        score = await engine._score_health("ollama")

        assert score == 1.0

    @pytest.mark.asyncio
    async def test_score_provider_health_unhealthy(self, engine):
        """Test unhealthy provider scoring."""
        # Mock health checker to return unhealthy
        engine.checker._provider_health["ollama"] = ProviderHealthResult(
            healthy=False,
            provider="ollama",
            model="test",
            issues=["API key missing"],
        )

        score = await engine._score_health("ollama")

        assert score < 1.0

    @pytest.mark.asyncio
    async def test_score_resources_local_with_gpu(self, engine):
        """Test resource scoring for local provider with GPU."""
        # Mock GPU available with valid cache time
        engine.detector._gpu_cache = GPUAvailability(
            available=True,
            memory_mb=16384,
        )
        engine.detector._gpu_cache_time = datetime.now()

        score = await engine._score_resources("ollama")

        assert score == 1.0

    @pytest.mark.asyncio
    async def test_score_resources_local_without_gpu(self, engine):
        """Test resource scoring for local provider without GPU."""
        # Mock GPU unavailable with valid cache time so the injected cache is
        # honored instead of re-probing the host (which has a GPU on some CI
        # runners / dev machines and would otherwise flip this to 1.0).
        engine.detector._gpu_cache = GPUAvailability(
            available=False,
            reason="No GPU detected",
        )
        engine.detector._gpu_cache_time = datetime.now()

        score = await engine._score_resources("ollama")

        assert score == 0.0

    @pytest.mark.asyncio
    async def test_score_resources_cloud(self, engine):
        """Test resource scoring for cloud provider."""
        score = await engine._score_resources("anthropic")

        assert score == 1.0  # Cloud providers always have resources

    @pytest.mark.asyncio
    async def test_score_cost_local_low_preference(self, engine):
        """Test cost scoring for local provider with low cost preference."""
        engine.profile.cost_preference = "low"

        score = await engine._score_cost("ollama")

        assert score == 1.0  # Local is free, perfect match

    @pytest.mark.asyncio
    async def test_score_cost_cloud_high_preference(self, engine):
        """Test cost scoring for cloud provider with high cost preference."""
        engine.profile.cost_preference = "high"

        score = await engine._score_cost("anthropic")

        assert score == 1.0  # Cloud is OK with high cost

    @pytest.mark.asyncio
    async def test_score_latency_local_high_preference(self, engine):
        """Test latency scoring for local provider with accuracy preference."""
        engine.profile.latency_preference = "high"  # Accuracy over speed

        score = await engine._score_latency("ollama")

        assert score >= 0.7  # Local is OK when accuracy matters

    @pytest.mark.asyncio
    async def test_score_latency_cloud_low_preference(self, engine):
        """Test latency scoring for cloud provider with low latency preference."""
        engine.profile.latency_preference = "low"  # Speed matters

        score = await engine._score_latency("anthropic")

        assert score == 1.0  # Cloud is fast, perfect match

    @pytest.mark.asyncio
    async def test_score_performance_disabled(self, engine):
        """Test performance scoring when learning is disabled."""
        engine.config.learning_enabled = False

        score = await engine._score_performance("ollama")

        assert score == 0.5  # Neutral when disabled

    @pytest.mark.asyncio
    async def test_score_performance_with_history(self, engine):
        """Test performance scoring with request history."""
        engine.config.learning_enabled = True

        # Add some successful requests
        for _ in range(5):
            engine.tracker.record_request(
                RequestMetric(
                    provider="ollama",
                    model="test",
                    success=True,
                    latency_ms=1000.0,
                    timestamp=datetime.now(),
                )
            )

        score = await engine._score_performance("ollama")

        assert score > 0.5  # Should have good score

    @pytest.mark.asyncio
    async def test_score_performance_penalizes_degraded_provider(self, engine):
        """Persistent degradation should lower the effective performance score."""
        engine.config.learning_enabled = True
        now = datetime.now()

        for _ in range(2):
            engine.tracker.record_request(
                RequestMetric(
                    provider="ollama",
                    model="test",
                    success=True,
                    latency_ms=700.0,
                    timestamp=now,
                )
            )
        for _ in range(3):
            engine.tracker.record_request(
                RequestMetric(
                    provider="ollama",
                    model="test",
                    success=False,
                    latency_ms=2200.0,
                    timestamp=now,
                    error_type="ProviderError",
                )
            )

        score = await engine._score_performance("ollama")

        assert score < 0.5

    @pytest.mark.asyncio
    async def test_decide_downranks_recently_degraded_provider(
        self, config, tracker, detector, checker
    ):
        """Degradation-aware scoring should prefer a stable provider when other factors tie."""
        config.learning_enabled = True
        config.custom_fallback_chain = ["ollama", "anthropic"]
        engine = RoutingDecisionEngine(
            config=config,
            performance_tracker=tracker,
            resource_detector=detector,
            health_checker=checker,
            available_providers=["ollama", "anthropic"],
        )
        now = datetime.now()
        for _ in range(3):
            tracker.record_request(
                RequestMetric(
                    provider="ollama",
                    model="test",
                    success=False,
                    latency_ms=2200.0,
                    timestamp=now,
                    error_type="ProviderError",
                )
            )
            tracker.record_request(
                RequestMetric(
                    provider="anthropic",
                    model="test",
                    success=True,
                    latency_ms=800.0,
                    timestamp=now,
                )
            )

        with (
            patch.object(engine, "_score_health", new=AsyncMock(return_value=1.0)),
            patch.object(engine, "_score_resources", new=AsyncMock(return_value=1.0)),
            patch.object(engine, "_score_cost", new=AsyncMock(return_value=1.0)),
            patch.object(engine, "_score_latency", new=AsyncMock(return_value=1.0)),
        ):
            decision = await engine.decide(task_type="default")

        assert decision.selected_provider == "anthropic"
        assert "ollama" in decision.fallback_chain

    @pytest.mark.asyncio
    async def test_get_topology_provider_hints(self, engine):
        """Engine should expose selector-friendly provider-routing hints."""
        hints = await engine.get_topology_provider_hints(task_type="default")

        assert "provider_hint" in hints
        assert "provider_locality" in hints
        assert "fallback_chain" in hints
        assert "health_score" in hints
        assert "cost_score" in hints
        assert "provider_degraded" in hints
        assert "provider_degradation_penalty" in hints


class TestSmartRoutingProvider:
    """Tests for SmartRoutingProvider."""

    @pytest.fixture
    def mock_providers(self):
        """Create mock providers."""
        providers = []
        for name in ["ollama", "anthropic", "openai"]:
            provider = MagicMock()
            provider.name = name
            provider.supports_tools = MagicMock(return_value=True)
            provider.supports_streaming = MagicMock(return_value=True)
            providers.append(provider)
        return providers

    @pytest.fixture
    def config(self):
        """Create test config."""
        return SmartRoutingConfig(
            enabled=True,
            profile_name="balanced",
            learning_enabled=False,
        )

    @pytest.fixture
    def smart_provider(self, mock_providers, config):
        """Create smart routing provider."""
        return SmartRoutingProvider(
            providers=mock_providers,
            config=config,
        )

    def test_smart_provider_initialization(self, smart_provider):
        """Test smart provider initialization."""
        assert smart_provider.name == "smart-router"
        assert len(smart_provider.providers) == 3
        assert smart_provider.engine is not None

    def test_supports_tools(self, smart_provider):
        """Test tools support detection."""
        assert smart_provider.supports_tools() is True

    def test_supports_streaming(self, smart_provider):
        """Test streaming support detection."""
        assert smart_provider.supports_streaming() is True

    def test_get_stats(self, smart_provider):
        """Test getting statistics."""
        stats = smart_provider.get_stats()

        assert "config" in stats
        assert "providers" in stats
        assert "performance" in stats
        assert len(stats["providers"]) == 3

    @pytest.mark.asyncio
    async def test_chat_success(self, smart_provider, mock_providers):
        """Test successful chat execution."""
        # Mock successful response
        mock_response = MagicMock()
        mock_providers[0].chat.return_value = mock_response

        # Mock resilient provider to return our mock
        with patch.object(
            smart_provider.resilient_providers["ollama"],
            "chat",
            new=AsyncMock(return_value=mock_response),
        ):
            messages = [Message(role="user", content="test")]
            response = await smart_provider.chat(messages, model="test-model")

            assert response == mock_response

    @pytest.mark.asyncio
    async def test_chat_failure_recording(self, smart_provider, mock_providers):
        """Test that failed requests are recorded."""
        # Mock failed response
        mock_providers[0].chat.side_effect = Exception("Test error")

        # Mock resilient provider to raise error
        with patch.object(
            smart_provider.resilient_providers["ollama"],
            "chat",
            new=AsyncMock(side_effect=Exception("Test error")),
        ):
            messages = [Message(role="user", content="test")]

            with pytest.raises(Exception):
                await smart_provider.chat(messages, model="test-model")

            # Verify metric was recorded
            metrics = smart_provider.tracker.get_metrics("ollama")
            assert len(metrics) == 1
            assert metrics[0].success is False

    @pytest.mark.asyncio
    async def test_stream_success(self, smart_provider, mock_providers):
        """Test successful stream execution."""

        # Mock stream chunks
        async def mock_stream(*args, **kwargs):
            chunks = [MagicMock(content="Hello"), MagicMock(content=" World")]
            for chunk in chunks:
                yield chunk

        mock_providers[0].stream.return_value = mock_stream()

        # Mock resilient provider to return our mock
        with patch.object(
            smart_provider.resilient_providers["ollama"],
            "stream",
            new=mock_stream,
        ):
            messages = [Message(role="user", content="test")]
            chunks = []
            async for chunk in smart_provider.stream(messages, model="test-model"):
                chunks.append(chunk)

            assert len(chunks) == 2

    @pytest.mark.asyncio
    async def test_stream_metrics_recording(self, smart_provider, mock_providers):
        """Test that stream metrics are recorded."""

        async def mock_stream(*args, **kwargs):
            yield MagicMock(content="test")

        mock_providers[0].stream.return_value = mock_stream()

        with patch.object(
            smart_provider.resilient_providers["ollama"],
            "stream",
            new=mock_stream,
        ):
            messages = [Message(role="user", content="test")]
            async for _ in smart_provider.stream(messages, model="test-model"):
                pass

            # Verify metric was recorded
            metrics = smart_provider.tracker.get_metrics("ollama")
            assert len(metrics) == 1
            assert metrics[0].success is True
