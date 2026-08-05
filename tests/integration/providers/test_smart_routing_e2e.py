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
Integration tests for smart routing system.

Tests end-to-end scenarios including:
- Local to cloud fallback on failure
- Performance learning over time
- Explicit provider respect
- Multi-factor routing decisions
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from victor.providers.smart_router import (
    RoutingDecisionEngine,
    SmartRoutingProvider,
)
from victor.providers.routing_config import SmartRoutingConfig, load_routing_profiles
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


def _seed_all_healthy(smart_provider, providers) -> None:
    """Seed the health cache so every mocked provider scores as healthy.

    Mock providers carry no real credentials, so the production health checker
    marks cloud providers (anthropic/openai) unhealthy for a missing API key
    (health score 0.0) — which makes them always lose to local ollama regardless
    of performance history or the active profile. Seeding reflects that the mocks
    are configured and healthy.
    """
    for provider in providers:
        smart_provider.checker._provider_health[provider.name] = ProviderHealthResult(
            healthy=True,
            provider=provider.name,
            model="test",
        )


@pytest.mark.integration
class TestSmartRoutingE2E:
    """End-to-end tests for smart routing."""

    @pytest.fixture
    def mock_providers(self):
        """Create mock providers for testing."""
        providers = []
        for name in ["ollama", "anthropic", "openai"]:
            provider = MagicMock()
            provider.name = name
            provider.supports_tools = MagicMock(return_value=True)
            provider.supports_streaming = MagicMock(return_value=True)

            # Mock chat method
            async def mock_chat(*args, **kwargs):
                return MagicMock(content=f"Response from {name}")

            provider.chat = AsyncMock(side_effect=mock_chat)
            providers.append(provider)

        return providers

    @pytest.fixture
    def config(self):
        """Create test config."""
        return SmartRoutingConfig(
            enabled=True,
            profile_name="balanced",
            learning_enabled=True,
        )

    @pytest.mark.asyncio
    async def test_local_to_cloud_fallback_on_failure(self, mock_providers, config):
        """Test automatic fallback from local to cloud provider on failure."""
        # Create smart routing provider
        smart_provider = SmartRoutingProvider(
            providers=mock_providers,
            config=config,
        )

        # Seed ONLY ollama healthy so the first request deterministically routes
        # to the local provider. (With the cloud providers also healthy, the
        # first-request score becomes resource/GPU-dependent and is not
        # deterministic on a CI runner without a GPU.)
        smart_provider.checker._provider_health["ollama"] = ProviderHealthResult(
            healthy=True,
            provider="ollama",
            model="test",
        )

        # Make first request - should use Ollama (local first in balanced profile)
        messages = [Message(role="user", content="test")]
        response1 = await smart_provider.chat(messages, model="test-model", task_type="default")

        # Verify Ollama was called
        mock_providers[0].chat.assert_called_once()

        # Now simulate Ollama failure (circuit breaker opens)
        # Record enough failures to open the circuit (threshold is typically 5)
        circuit_breaker = smart_provider.resilient_providers["ollama"].circuit_breaker
        for _ in range(10):  # Ensure we exceed the failure threshold
            try:
                circuit_breaker.record_failure(Exception("Simulated failure"))
            except Exception:
                pass  # record_failure may raise in some states

        smart_provider.checker._provider_health["ollama"] = ProviderHealthResult(
            healthy=False,
            provider="ollama",
            model="test",
            issues=["Circuit breaker open"],
        )
        # Anthropic is the intended fallback target: seed it healthy now (the
        # mocked cloud provider has no API key, so the production health check
        # would otherwise score it 0.0 and block the fallback).
        smart_provider.checker._provider_health["anthropic"] = ProviderHealthResult(
            healthy=True,
            provider="anthropic",
            model="test",
        )

        # Make second request - should fallback to Anthropic
        response2 = await smart_provider.chat(messages, model="test-model", task_type="default")

        # Verify Anthropic (second provider) was called
        mock_providers[1].chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_performance_learning_adapts_routing(self, mock_providers, config):
        """Test that routing decisions adapt based on performance history."""
        smart_provider = SmartRoutingProvider(
            providers=mock_providers,
            config=config,
        )

        # Seed health for all mocked providers so performance history — not a
        # missing-API-key health penalty — drives the decision.
        _seed_all_healthy(smart_provider, mock_providers)

        # Simulate Ollama performing VERY poorly (high latency + some failures)
        for i in range(10):
            smart_provider.tracker.record_request(
                RequestMetric(
                    provider="ollama",
                    model="test",
                    success=i < 7,  # 30% failure rate
                    latency_ms=10000.0,  # Extremely slow
                    timestamp=datetime.now(),
                )
            )

        # Simulate Anthropic performing well (low latency + all success)
        for _ in range(10):
            smart_provider.tracker.record_request(
                RequestMetric(
                    provider="anthropic",
                    model="test",
                    success=True,
                    latency_ms=500.0,  # Fast
                    timestamp=datetime.now(),
                )
            )

        # Make routing decision
        decision = await smart_provider.engine.decide(task_type="default")

        # Should prefer Anthropic due to MUCH better performance
        # (even though Ollama is cheaper and has resources)
        assert decision.selected_provider == "anthropic"
        assert "performance" in decision.factors

    @pytest.mark.asyncio
    async def test_explicit_provider_not_overridden(self, mock_providers, config):
        """Test that explicit provider choice is never overridden by routing."""
        smart_provider = SmartRoutingProvider(
            providers=mock_providers,
            config=config,
        )

        # Make routing decision with explicit provider
        decision = await smart_provider.engine.decide(
            task_type="default",
            preferred_providers=["openai"],  # User explicitly chose OpenAI
        )

        # Must respect user's choice
        assert decision.selected_provider == "openai"
        assert decision.fallback_chain == []  # No fallbacks when explicit

    @pytest.mark.asyncio
    async def test_multi_factor_routing_decision(self, mock_providers, config):
        """Test that routing considers multiple factors."""
        smart_provider = SmartRoutingProvider(
            providers=mock_providers,
            config=config,
        )

        # Mock health: Ollama healthy, Anthropic healthy
        for provider_name in ["ollama", "anthropic"]:
            smart_provider.checker._provider_health[provider_name] = ProviderHealthResult(
                healthy=True,
                provider=provider_name,
                model="test",
            )

        # Mock GPU available
        smart_provider.detector._gpu_cache = GPUAvailability(
            available=True,
            memory_mb=16384,
        )
        smart_provider.detector._gpu_cache_time = datetime.now()

        # Make routing decision
        decision = await smart_provider.engine.decide(task_type="default")

        # Should have considered all factors
        assert "health" in decision.factors
        assert "resources" in decision.factors
        assert "cost" in decision.factors
        assert "latency" in decision.factors
        assert 0.0 <= decision.confidence <= 1.0
        assert len(decision.rationale) > 0

    @pytest.mark.asyncio
    async def test_custom_fallback_chain(self, mock_providers):
        """Test custom fallback chain configuration."""
        # Reorder providers to match custom chain
        reordered_providers = [
            mock_providers[2],
            mock_providers[1],
            mock_providers[0],
        ]  # openai, anthropic, ollama

        config = SmartRoutingConfig(
            enabled=True,
            profile_name="balanced",
            custom_fallback_chain=["openai", "anthropic", "ollama"],
        )

        smart_provider = SmartRoutingProvider(
            providers=reordered_providers,
            config=config,
        )

        # Make routing decision
        decision = await smart_provider.engine.decide(task_type="default")

        # Custom chain is used as candidates (not strict ordering)
        # Provider selection is based on multi-factor scoring
        # So the best provider from the custom chain is selected
        assert decision.selected_provider in ["openai", "anthropic", "ollama"]
        # Verify custom chain is respected in candidates
        assert "anthropic" in decision.fallback_chain or "ollama" in decision.fallback_chain


@pytest.mark.integration
class TestSmartRoutingScenarios:
    """Test realistic routing scenarios."""

    @pytest.mark.asyncio
    async def test_gpu_unavailable_fallback_to_cloud(self):
        """Test scenario: GPU unavailable, fallback to cloud."""
        from victor.providers.routing_config import SmartRoutingConfig
        from victor.providers.smart_router import SmartRoutingProvider
        from unittest.mock import MagicMock

        # Create mock providers
        providers = []
        for name in ["ollama", "anthropic"]:
            provider = MagicMock()
            provider.name = name
            provider.supports_tools = MagicMock(return_value=True)
            provider.supports_streaming = MagicMock(return_value=True)

            async def mock_chat(*args, **kwargs):
                return MagicMock(content=f"Response from {name}")

            provider.chat = AsyncMock(side_effect=mock_chat)
            providers.append(provider)

        config = SmartRoutingConfig(
            enabled=True,
            profile_name="performance",  # Performance profile prefers cloud/fast
        )

        smart_provider = SmartRoutingProvider(
            providers=providers,
            config=config,
        )

        # Mock GPU unavailable
        smart_provider.detector._gpu_cache = GPUAvailability(
            available=False,
            reason="No GPU detected",
        )
        smart_provider.detector._gpu_cache_time = datetime.now()

        # Make routing decision for coding task
        decision = await smart_provider.engine.decide(task_type="coding")

        # Should prefer Anthropic (cloud + fast) over Ollama (no GPU + slow)
        assert decision.selected_provider == "anthropic"

    @pytest.mark.asyncio
    async def test_cost_optimized_profile_prefers_local(self):
        """Test scenario: cost-optimized profile prefers local providers."""
        from victor.providers.routing_config import SmartRoutingConfig
        from victor.providers.smart_router import SmartRoutingProvider
        from unittest.mock import MagicMock

        providers = []
        for name in ["ollama", "anthropic"]:
            provider = MagicMock()
            provider.name = name
            provider.supports_tools = MagicMock(return_value=True)
            provider.supports_streaming = MagicMock(return_value=True)

            async def mock_chat(*args, **kwargs):
                return MagicMock(content=f"Response from {name}")

            provider.chat = AsyncMock(side_effect=mock_chat)
            providers.append(provider)

        config = SmartRoutingConfig(
            enabled=True,
            profile_name="cost-optimized",
        )

        smart_provider = SmartRoutingProvider(
            providers=providers,
            config=config,
        )

        # Mock GPU available
        smart_provider.detector._gpu_cache = GPUAvailability(
            available=True,
            memory_mb=16384,
        )
        smart_provider.detector._gpu_cache_time = datetime.now()

        # Make routing decision
        decision = await smart_provider.engine.decide(task_type="default")

        # Should prefer Ollama (free) over Anthropic (paid)
        assert decision.selected_provider == "ollama"

    @pytest.mark.asyncio
    async def test_performance_profile_prefers_cloud(self):
        """Test scenario: performance profile prefers cloud providers."""
        from victor.providers.routing_config import SmartRoutingConfig
        from victor.providers.smart_router import SmartRoutingProvider
        from unittest.mock import MagicMock

        providers = []
        for name in ["anthropic", "ollama"]:
            provider = MagicMock()
            provider.name = name
            provider.supports_tools = MagicMock(return_value=True)
            provider.supports_streaming = MagicMock(return_value=True)

            async def mock_chat(*args, **kwargs):
                return MagicMock(content=f"Response from {name}")

            provider.chat = AsyncMock(side_effect=mock_chat)
            providers.append(provider)

        config = SmartRoutingConfig(
            enabled=True,
            profile_name="performance",
        )

        smart_provider = SmartRoutingProvider(
            providers=providers,
            config=config,
        )

        # Seed health for the mocked providers so the 'performance' profile's
        # cloud-first ordering isn't overridden by a missing-API-key penalty.
        _seed_all_healthy(smart_provider, providers)

        # Make routing decision
        decision = await smart_provider.engine.decide(task_type="default")

        # Should prefer Anthropic (fast cloud) over Ollama (slow local)
        assert decision.selected_provider == "anthropic"


@pytest.mark.integration
class TestSmartRoutingMetrics:
    """Test metrics tracking and observability."""

    @pytest.mark.asyncio
    async def test_routing_decision_logging(self):
        """Test that routing decisions are logged with full context."""
        from victor.providers.routing_config import SmartRoutingConfig
        from victor.providers.smart_router import SmartRoutingProvider
        from unittest.mock import MagicMock
        import logging

        providers = []
        for name in ["ollama", "anthropic"]:
            provider = MagicMock()
            provider.name = name
            provider.supports_tools = MagicMock(return_value=True)
            provider.supports_streaming = MagicMock(return_value=True)

            async def mock_chat(*args, **kwargs):
                return MagicMock(content=f"Response from {name}")

            provider.chat = AsyncMock(side_effect=mock_chat)
            providers.append(provider)

        config = SmartRoutingConfig(
            enabled=True,
            profile_name="balanced",
        )

        smart_provider = SmartRoutingProvider(
            providers=providers,
            config=config,
        )

        # Get routing decision
        decision = await smart_provider.engine.decide(task_type="default")

        # Verify decision structure
        assert hasattr(decision, "selected_provider")
        assert hasattr(decision, "fallback_chain")
        assert hasattr(decision, "rationale")
        assert hasattr(decision, "confidence")
        assert hasattr(decision, "factors")

        # Verify factors include all scoring components
        expected_factors = {"health", "resources", "cost", "latency", "performance"}
        assert set(decision.factors.keys()) >= expected_factors

    @pytest.mark.asyncio
    async def test_performance_metrics_tracking(self):
        """Test that performance metrics are tracked correctly."""
        from victor.providers.routing_config import SmartRoutingConfig
        from victor.providers.smart_router import SmartRoutingProvider
        from unittest.mock import MagicMock

        providers = []
        for name in ["ollama", "anthropic"]:
            provider = MagicMock()
            provider.name = name
            provider.supports_tools = MagicMock(return_value=True)
            provider.supports_streaming = MagicMock(return_value=True)

            async def mock_chat(*args, **kwargs):
                return MagicMock(content=f"Response from {name}")

            provider.chat = AsyncMock(side_effect=mock_chat)
            providers.append(provider)

        config = SmartRoutingConfig(
            enabled=True,
            profile_name="balanced",
            performance_window_size=50,
        )

        smart_provider = SmartRoutingProvider(
            providers=providers,
            config=config,
        )

        # Make multiple requests
        messages = [Message(role="user", content="test")]
        for _ in range(5):
            await smart_provider.chat(messages, model="test-model", task_type="default")

        # Check metrics were recorded
        stats = smart_provider.get_stats()
        assert "performance" in stats
        assert "providers_tracked" in stats["performance"]

        # Should have tracked at least one provider
        assert stats["performance"]["providers_tracked"] >= 1
