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
Smart Routing Engine for Automatic Provider Selection.

This module provides intelligent routing that automatically switches between
providers based on health, resources, cost, latency, and performance history.

Key components:
- RoutingDecision: Dataclass for routing decisions with rationale
- RoutingDecisionEngine: Multi-factor routing logic
- SmartRoutingProvider: Provider wrapper that implements smart routing

Usage:
    from victor.providers.smart_router import SmartRoutingProvider
    from victor.providers.routing_config import SmartRoutingConfig

    # Create smart routing provider
    config = SmartRoutingConfig(enabled=True, profile_name="balanced")
    smart_provider = SmartRoutingProvider(
        providers=[ollama, anthropic, openai],
        config=config,
    )

    # Use like any other provider
    response = await smart_provider.chat(messages, model=model)
"""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from victor.providers.base import BaseProvider, CompletionResponse, Message
from victor.providers.health import ProviderHealthChecker
from victor.providers.performance_tracker import (
    ProviderPerformanceTracker,
    RequestMetric,
)
from victor.providers.resilience import ResilientProvider
from victor.providers.resource_detector import ResourceAvailabilityDetector
from victor.providers.routing_config import SmartRoutingConfig, load_routing_profiles

logger = logging.getLogger(__name__)

LOCAL_PROVIDER_HINTS = {
    "ollama",
    "lmstudio",
    "vllm",
    "llamacpp",
    "llama.cpp",
    "local",
}

# L4: when the recent per-turn cost trace (C0 TaskExecutionReport) exceeds these, routing biases
# toward cheaper/faster local providers. Only consulted when a cost_trace is passed to decide().
_COST_TRACE_USD_THRESHOLD = 0.05  # a turn costing > 5¢ is "expensive"
_COST_TRACE_LATENCY_THRESHOLD_S = 30.0  # a turn slower than 30s is "slow"


@dataclass
class RoutingDecision:
    """Routing decision with rationale.

    Attributes:
        selected_provider: Chosen provider name
        fallback_chain: Ordered list of fallback providers
        rationale: Human-readable explanation
        confidence: Confidence score (0.0 to 1.0)
        factors: Dict of factors considered and their weights
    """

    selected_provider: str
    fallback_chain: List[str]
    rationale: str
    confidence: float
    factors: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "selected_provider": self.selected_provider,
            "fallback_chain": self.fallback_chain,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "factors": self.factors,
        }

    def to_topology_hints(self) -> Dict[str, Any]:
        """Return stable provider-routing hints for topology selection."""
        provider_lower = self.selected_provider.lower()
        provider_locality = (
            "local" if any(hint in provider_lower for hint in LOCAL_PROVIDER_HINTS) else "remote"
        )
        return {
            "provider_hint": self.selected_provider,
            "provider_locality": provider_locality,
            "fallback_chain": list(self.fallback_chain),
            "confidence": self.confidence,
            "health_score": self.factors.get("health"),
            "resource_score": self.factors.get("resources"),
            "cost_score": self.factors.get("cost"),
            "latency_score": self.factors.get("latency"),
            "performance_score": self.factors.get("performance"),
            "provider_degraded": bool(self.factors.get("degraded", False)),
            "provider_degradation_penalty": self.factors.get("degradation_penalty"),
            "provider_degradation_reasons": list(self.factors.get("degradation_reasons") or []),
            "provider_failure_streak": self.factors.get("failure_streak"),
            "provider_success_streak": self.factors.get("success_streak"),
            "provider_recovered_from_recent_incident": bool(
                self.factors.get("recovered_from_recent_incident", False)
            ),
            "provider_time_to_recover_seconds": self.factors.get("time_to_recover_seconds"),
        }


class RoutingDecisionEngine:
    """Makes intelligent routing decisions based on multiple factors.

    Factors considered (in order of priority):
    1. Provider health (circuit breaker state, recent errors)
    2. Resource availability (GPU, API quota)
    3. Cost preference (local free vs cloud paid)
    4. Latency preference (local slow vs cloud fast)
    5. Performance history (success rate, latency trends)
    """

    def __init__(
        self,
        config: SmartRoutingConfig,
        performance_tracker: ProviderPerformanceTracker,
        resource_detector: ResourceAvailabilityDetector,
        health_checker: ProviderHealthChecker,
        available_providers: List[str],
    ):
        """Initialize routing decision engine.

        Args:
            config: Smart routing configuration
            performance_tracker: Performance tracking instance
            resource_detector: Resource detection instance
            health_checker: Health checking instance
            available_providers: List of available provider names
        """
        self.config = config
        self.tracker = performance_tracker
        self.detector = resource_detector
        self.checker = health_checker
        self.available_providers = available_providers

        # Load routing profile
        self.profile = load_routing_profiles().get(config.profile_name)
        if not self.profile:
            logger.warning(f"Profile '{config.profile_name}' not found, using 'balanced'")
            self.profile = load_routing_profiles().get("balanced")

        logger.debug(
            f"RoutingDecisionEngine initialized with {len(available_providers)} providers, "
            f"profile='{config.profile_name}'"
        )

    async def decide(
        self,
        task_type: str = "default",
        model_hint: Optional[str] = None,
        preferred_providers: Optional[List[str]] = None,
        cost_trace: Optional[Dict[str, Any]] = None,
    ) -> RoutingDecision:
        """Make routing decision based on multiple factors.

        Args:
            task_type: Type of task (default, coding, chat, etc.)
            model_hint: Suggested model (can influence provider choice)
            preferred_providers: User-suggested providers (respected if provided)
            cost_trace: Optional recent per-turn cost record (C0 TaskExecutionReport). When it
                shows recent turns are expensive/slow, routing is biased toward cheaper/faster
                local providers (L4). Omitted -> behavior unchanged.

        Returns:
            RoutingDecision with selected provider and fallback chain
        """
        # Get candidate providers
        candidates = await self._get_candidates(
            task_type=task_type,
            preferred_providers=preferred_providers,
        )

        if not candidates:
            # No candidates available
            logger.error("No providers available for routing")
            return RoutingDecision(
                selected_provider="",
                fallback_chain=[],
                rationale="No providers available",
                confidence=0.0,
                factors={"error": "no_candidates"},
            )

        # Score each provider
        scored_providers = []
        for provider in candidates:
            score, factors = await self._score_provider(
                provider=provider,
                task_type=task_type,
                cost_trace=cost_trace,
            )
            scored_providers.append((provider, score, factors))

        # Sort by score (descending)
        scored_providers.sort(key=lambda x: x[1], reverse=True)

        # Select best provider
        selected_provider, selected_score, selected_factors = scored_providers[0]

        # Build fallback chain (remaining providers in score order)
        fallback_chain = [p for p, _, _ in scored_providers[1:]]

        # Generate rationale
        rationale = self._generate_rationale(
            selected_provider=selected_provider,
            score=selected_score,
            factors=selected_factors,
        )

        decision = RoutingDecision(
            selected_provider=selected_provider,
            fallback_chain=fallback_chain,
            rationale=rationale,
            confidence=selected_score,
            factors=selected_factors,
        )

        logger.info(
            f"Routing decision: {decision.selected_provider} "
            f"(confidence={decision.confidence:.2f}, "
            f"fallbacks={len(decision.fallback_chain)})"
        )

        return decision

    def _generate_rationale(
        self,
        selected_provider: str,
        score: float,
        factors: Dict[str, Any],
    ) -> str:
        """Generate human-readable rationale for routing decision.

        Args:
            selected_provider: Chosen provider
            score: Confidence score
            factors: Factor scores

        Returns:
            Rationale string
        """
        # Find highest-weighted factor
        numeric_factors = {
            name: float(value)
            for name, value in factors.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        max_factor = (
            max(numeric_factors.items(), key=lambda x: x[1])
            if numeric_factors
            else ("score", score)
        )

        rationale = f"Selected {selected_provider} (confidence={score:.2f})"

        if max_factor[1] >= 0.8:
            rationale += f" based on {max_factor[0]} ({max_factor[1]:.2f})"
        if factors.get("degraded"):
            reasons = ",".join(str(reason) for reason in factors.get("degradation_reasons") or [])
            rationale += f"; recent degradation remains active ({reasons or 'unknown'})"
        elif factors.get("recovered_from_recent_incident"):
            rationale += "; provider has recently recovered from instability"

        return rationale

    async def _get_candidates(
        self,
        task_type: str,
        preferred_providers: Optional[List[str]],
    ) -> List[str]:
        """Get list of candidate providers for routing.

        Args:
            task_type: Type of task
            preferred_providers: User-suggested providers

        Returns:
            List of provider names
        """
        # If user specified providers, use those
        if preferred_providers:
            candidates = [
                p.lower() for p in preferred_providers if p.lower() in self.available_providers
            ]
            if candidates:
                logger.debug(f"Using user-specified providers: {candidates}")
                return candidates

        # Use custom fallback chain if specified, otherwise use profile
        if self.config.custom_fallback_chain:
            fallback_chain = self.config.custom_fallback_chain
        else:
            fallback_chain = self.profile.get_fallback_chain(task_type)

        # Filter to available providers
        candidates = [p for p in fallback_chain if p in self.available_providers]

        logger.debug(f"Found {len(candidates)} candidates for task_type={task_type}")
        return candidates

    async def _score_provider(
        self,
        provider: str,
        task_type: str,
        cost_trace: Optional[Dict[str, Any]] = None,
    ) -> tuple[float, Dict[str, Any]]:
        """Score a provider based on multiple factors.

        Args:
            provider: Provider name
            task_type: Type of task
            cost_trace: Optional recent per-turn cost record (L4) used to bias the cost factor.

        Returns:
            Tuple of (score, factors_dict)
        """
        factors = {}
        total_score = 0.0
        weight_sum = 0.0
        degradation_snapshot = self._get_degradation_snapshot(provider)

        # Factor 1: Provider health (weight: 0.3)
        health_score = await self._score_health(provider)
        factors["health"] = health_score
        total_score += 0.3 * health_score
        weight_sum += 0.3

        # Factor 2: Resource availability (weight: 0.25)
        resource_score = await self._score_resources(provider)
        factors["resources"] = resource_score
        total_score += 0.25 * resource_score
        weight_sum += 0.25

        # Factor 3: Cost preference (weight: 0.15)
        cost_score = await self._score_cost(provider, cost_trace=cost_trace)
        factors["cost"] = cost_score
        total_score += 0.15 * cost_score
        weight_sum += 0.15

        # Factor 4: Latency preference (weight: 0.15)
        latency_score = await self._score_latency(provider)
        factors["latency"] = latency_score
        total_score += 0.15 * latency_score
        weight_sum += 0.15

        # Factor 5: Performance history (weight: 0.15)
        perf_score = await self._score_performance(
            provider,
            degradation_snapshot=degradation_snapshot,
        )
        factors["performance"] = perf_score
        total_score += 0.15 * perf_score
        weight_sum += 0.15

        if degradation_snapshot is not None:
            degradation_penalty = self._degradation_penalty(degradation_snapshot)
            factors["degraded"] = bool(getattr(degradation_snapshot, "degraded", False))
            factors["degradation_penalty"] = degradation_penalty
            factors["degradation_reasons"] = list(
                getattr(degradation_snapshot, "degradation_reasons", ()) or ()
            )
            factors["failure_streak"] = int(getattr(degradation_snapshot, "failure_streak", 0) or 0)
            factors["success_streak"] = int(getattr(degradation_snapshot, "success_streak", 0) or 0)
            factors["recovered_from_recent_incident"] = bool(
                getattr(degradation_snapshot, "recovered_from_recent_incident", False)
            )
            factors["time_to_recover_seconds"] = getattr(
                degradation_snapshot,
                "time_to_recover_seconds",
                None,
            )

        # Normalize score
        if weight_sum > 0:
            normalized_score = total_score / weight_sum
        else:
            normalized_score = 0.0

        return normalized_score, factors

    async def get_topology_provider_hints(
        self,
        task_type: str = "default",
        model_hint: Optional[str] = None,
        preferred_providers: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Return provider routing hints in a selector-friendly shape."""
        decision = await self.decide(
            task_type=task_type,
            model_hint=model_hint,
            preferred_providers=preferred_providers,
        )
        return decision.to_topology_hints()

    async def _score_health(self, provider: str) -> float:
        """Score provider health (0.0 to 1.0).

        Checks:
        - Provider registration
        - API key availability
        - Circuit breaker state

        Args:
            provider: Provider name

        Returns:
            Health score
        """
        # Check health cache
        health_result = self.checker.get_provider_health(provider)

        if not health_result:
            # No health info, run quick check
            try:
                health_result = await self.checker.check_provider(
                    provider=provider,
                    model="test",
                    check_connectivity=False,
                )
            except Exception as e:
                logger.debug(f"Health check failed for {provider}: {e}")
                return 0.3  # Low score for unknown providers

        if health_result.healthy:
            return 1.0
        else:
            # Check if issues are critical
            critical_issues = ["api_key", "authentication", "auth"]
            if any(issue in str(health_result.issues).lower() for issue in critical_issues):
                return 0.0  # Critical failure
            else:
                return 0.3  # Non-critical issues

    async def _score_resources(self, provider: str) -> float:
        """Score resource availability (0.0 to 1.0).

        Args:
            provider: Provider name

        Returns:
            Resource score
        """
        # Check GPU availability for local providers
        local_providers = {"ollama", "lmstudio", "vllm"}
        if provider in local_providers:
            gpu = await self.detector.check_gpu_availability()
            if gpu.available:
                return 1.0
            else:
                return 0.0  # Local provider needs GPU

        # Cloud providers don't have resource constraints
        return 1.0

    async def _score_cost(
        self, provider: str, cost_trace: Optional[Dict[str, Any]] = None
    ) -> float:
        """Score cost preference alignment (0.0 to 1.0).

        L4: when a per-turn cost trace (the C0 ``TaskExecutionReport``) is supplied and shows
        recent turns are expensive or slow, the static profile preference is overridden to bias
        routing toward cheaper/faster local providers — closing the cost loop with measured data
        instead of a fixed preference. Without a trace, behavior is unchanged.

        Args:
            provider: Provider name
            cost_trace: Optional recent per-turn cost record (``total_cost_usd``,
                ``duration_seconds``); when over threshold, biases toward local providers.

        Returns:
            Cost score
        """
        local_providers = {"ollama", "lmstudio", "vllm"}

        # L4 trace-driven override: recent turns are over budget / too slow -> prefer local.
        if cost_trace:
            over_budget = (
                float(cost_trace.get("total_cost_usd", 0.0) or 0.0) > _COST_TRACE_USD_THRESHOLD
            )
            over_latency = (
                float(cost_trace.get("duration_seconds", 0.0) or 0.0)
                > _COST_TRACE_LATENCY_THRESHOLD_S
            )
            if over_budget or over_latency:
                return 1.0 if provider in local_providers else 0.2

        # Local providers are free
        if provider in local_providers:
            if self.profile.cost_preference == "low":
                return 1.0  # Perfect match
            elif self.profile.cost_preference == "normal":
                return 0.7  # Good match
            else:  # high
                return 0.3  # Poor match

        # Cloud providers have costs
        # (Simplified - in reality, would check actual pricing)
        if self.profile.cost_preference == "low":
            return 0.3  # Poor match
        elif self.profile.cost_preference == "normal":
            return 0.7  # Good match
        else:  # high
            return 1.0  # Perfect match

    async def _score_latency(self, provider: str) -> float:
        """Score latency preference alignment (0.0 to 1.0).

        Args:
            provider: Provider name

        Returns:
            Latency score
        """
        # Local providers are slower
        local_providers = {"ollama", "lmstudio", "vllm"}
        if provider in local_providers:
            if self.profile.latency_preference == "low":
                return 0.3  # Poor match (local is slow)
            elif self.profile.latency_preference == "normal":
                return 0.7  # Good match
            else:  # high
                return 1.0  # Perfect match (accuracy over speed)

        # Cloud providers are faster
        if self.profile.latency_preference == "low":
            return 1.0  # Perfect match
        elif self.profile.latency_preference == "normal":
            return 0.7  # Good match
        else:  # high
            return 0.5  # Neutral

    def _get_degradation_snapshot(self, provider: str) -> Optional[Any]:
        """Return the latest degradation snapshot for a provider when available."""
        if not hasattr(self.tracker, "get_degradation_snapshot"):
            return None
        try:
            return self.tracker.get_degradation_snapshot(provider)
        except Exception as exc:
            logger.debug("Provider degradation snapshot unavailable for %s: %s", provider, exc)
            return None

    @staticmethod
    def _degradation_penalty(snapshot: Optional[Any]) -> float:
        """Return a bounded penalty for degraded or recently recovered providers."""
        if snapshot is None:
            return 0.0

        degraded = bool(getattr(snapshot, "degraded", False))
        recovered = bool(getattr(snapshot, "recovered_from_recent_incident", False))
        failure_streak = int(getattr(snapshot, "failure_streak", 0) or 0)
        recent_success_rate = float(getattr(snapshot, "recent_success_rate", 0.5) or 0.5)
        reasons = set(getattr(snapshot, "degradation_reasons", ()) or ())

        penalty = 0.0
        if degraded:
            penalty += min(0.30, 0.08 * max(1, failure_streak))
            if "low_recent_success_rate" in reasons:
                penalty += 0.12
            if "latency_trend" in reasons:
                penalty += 0.08
            if recent_success_rate < 0.5:
                penalty += min(0.15, (0.5 - recent_success_rate) * 0.3)
        elif recovered:
            penalty += min(
                0.08,
                0.02 * max(1, int(getattr(snapshot, "recent_incident_failure_count", 0) or 0)),
            )

        return round(min(0.55, penalty), 4)

    async def _score_performance(
        self,
        provider: str,
        *,
        degradation_snapshot: Optional[Any] = None,
    ) -> float:
        """Score based on performance history (0.0 to 1.0).

        Args:
            provider: Provider name

        Returns:
            Performance score
        """
        if not self.config.learning_enabled:
            base_score = 0.5  # Neutral if learning disabled
        else:
            # Get composite score from tracker
            base_score = self.tracker.get_provider_score(provider)

        snapshot = degradation_snapshot or self._get_degradation_snapshot(provider)
        penalty = self._degradation_penalty(snapshot)
        return max(0.0, min(1.0, base_score - penalty))


class SmartRoutingProvider:
    """Provider wrapper that implements smart routing.

    Wraps multiple providers and intelligently routes requests based on:
    - Health status
    - Resource availability
    - Cost and latency preferences
    - Historical performance

    Delegates actual execution to ResilientProvider for retry and circuit breaking.

    Usage:
        smart_provider = SmartRoutingProvider(
            providers=[ollama_provider, anthropic_provider],
            config=SmartRoutingConfig(enabled=True),
        )

        response = await smart_provider.chat(messages, model=model)
    """

    def __init__(
        self,
        providers: List[BaseProvider],
        config: SmartRoutingConfig,
    ):
        """Initialize smart routing provider.

        Args:
            providers: List of available provider instances
            config: Smart routing configuration
        """
        self.providers = {p.name: p for p in providers}
        self.config = config

        # Keep non-learning mode session-local and deterministic. Persisting and
        # hydrating cross-session routing history only helps when learning is
        # enabled, and it makes unit-test behavior depend on ambient DB state.
        if config.learning_enabled:
            self.tracker = ProviderPerformanceTracker(window_size=config.performance_window_size)
        else:
            self.tracker = ProviderPerformanceTracker(
                window_size=config.performance_window_size,
                db=None,
            )
        self.detector = ResourceAvailabilityDetector()
        self.checker = ProviderHealthChecker()

        # Create resilient providers for each base provider
        self.resilient_providers: Dict[str, ResilientProvider] = {}
        for name, provider in self.providers.items():
            self.resilient_providers[name] = ResilientProvider(
                provider=provider,
                request_timeout=120.0,
            )

        # Create routing engine
        self.engine = RoutingDecisionEngine(
            config=config,
            performance_tracker=self.tracker,
            resource_detector=self.detector,
            health_checker=self.checker,
            available_providers=list(self.providers.keys()),
        )

        logger.info(
            f"SmartRoutingProvider initialized with {len(providers)} providers, "
            f"profile='{config.profile_name}'"
        )

    @property
    def name(self) -> str:
        """Provider name."""
        return "smart-router"

    def supports_tools(self) -> bool:
        """Check if any provider supports tools."""
        return any(getattr(p, "supports_tools", lambda: False)() for p in self.providers.values())

    def supports_streaming(self) -> bool:
        """Check if any provider supports streaming."""
        return any(
            getattr(p, "supports_streaming", lambda: False)() for p in self.providers.values()
        )

    async def chat(
        self,
        messages: List[Message],
        *,
        model: str,
        task_type: str = "default",
        **kwargs,
    ) -> CompletionResponse:
        """Execute chat with smart routing.

        Args:
            messages: List of messages
            model: Model identifier
            task_type: Type of task (influences routing)
            **kwargs: Additional arguments for provider

        Returns:
            Chat response

        Raises:
            Exception: If all providers fail
        """
        # Make routing decision
        decision = await self.engine.decide(
            task_type=task_type,
            model_hint=model,
            preferred_providers=None,  # Could be added as parameter
        )

        if not decision.selected_provider:
            raise Exception("No providers available for routing")

        # Get primary provider
        primary_provider = self.resilient_providers.get(decision.selected_provider)
        if not primary_provider:
            raise Exception(f"Provider '{decision.selected_provider}' not found")

        # Build fallback chain
        fallback_providers = [
            self.resilient_providers[name]
            for name in decision.fallback_chain
            if name in self.resilient_providers
        ]

        # Update primary provider's fallbacks
        primary_provider.fallback_providers = [fb.provider for fb in fallback_providers]

        # Execute with metrics tracking
        start_time = time.time()
        try:
            response = await primary_provider.chat(messages, model=model, **kwargs)
            latency_ms = (time.time() - start_time) * 1000

            # Record successful metric
            self.tracker.record_request(
                RequestMetric(
                    provider=decision.selected_provider,
                    model=model,
                    success=True,
                    latency_ms=latency_ms,
                    timestamp=datetime.now(),
                )
            )

            logger.info(
                f"Smart routing success: {decision.selected_provider} "
                f"({latency_ms:.0f}ms, rationale: {decision.rationale})"
            )

            return response

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000

            # Record failed metric
            self.tracker.record_request(
                RequestMetric(
                    provider=decision.selected_provider,
                    model=model,
                    success=False,
                    latency_ms=latency_ms,
                    timestamp=datetime.now(),
                    error_type=type(e).__name__,
                )
            )

            logger.warning(
                f"Smart routing failed: {decision.selected_provider} "
                f"({latency_ms:.0f}ms, error: {e})"
            )

            raise

    async def stream(
        self,
        messages: List[Message],
        *,
        model: str,
        task_type: str = "default",
        **kwargs,
    ):
        """Stream chat with smart routing.

        Note: Streaming has limited retry capability.

        Args:
            messages: List of messages
            model: Model identifier
            task_type: Type of task
            **kwargs: Additional arguments

        Yields:
            Stream chunks
        """
        # Make routing decision
        decision = await self.engine.decide(
            task_type=task_type,
            model_hint=model,
        )

        if not decision.selected_provider:
            raise Exception("No providers available for routing")

        # Get primary provider
        primary_provider = self.resilient_providers.get(decision.selected_provider)
        if not primary_provider:
            raise Exception(f"Provider '{decision.selected_provider}' not found")

        # Stream with metrics tracking
        start_time = time.time()
        success = False
        chunks_received = 0

        try:
            inner = primary_provider.stream(messages, model=model, **kwargs)
            async with contextlib.aclosing(inner) as inner_stream:
                async for chunk in inner_stream:
                    chunks_received += 1
                    yield chunk

            success = True

        finally:
            latency_ms = (time.time() - start_time) * 1000

            # Record metric
            self.tracker.record_request(
                RequestMetric(
                    provider=decision.selected_provider,
                    model=model,
                    success=success,
                    latency_ms=latency_ms,
                    timestamp=datetime.now(),
                    error_type=None if success else "StreamingError",
                )
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get routing statistics.

        Returns:
            Dict with routing stats
        """
        return {
            "config": {
                "profile": self.config.profile_name,
                "learning_enabled": self.config.learning_enabled,
                "window_size": self.config.performance_window_size,
            },
            "providers": list(self.providers.keys()),
            "performance": self.tracker.get_stats(),
        }
