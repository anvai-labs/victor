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

"""Automatic model fallback with circuit breaker pattern.

This module provides intelligent model fallback based on:
- Failure history (circuit breaker)
- Model capabilities matching
- Task type requirements
- Cost considerations

Circuit Breaker States:
- CLOSED: Model is healthy, requests proceed normally
- OPEN: Model has failed too many times, requests are rejected
- HALF_OPEN: Testing if model has recovered
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from victor.core.circuit_breaker import CircuitState

from victor.agent.recovery.protocols import (
    FailureType,
    ModelFallbackPolicy,
    RecoveryContext,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelCircuitBreaker:
    """Circuit breaker for a single model."""

    provider: str
    model: str
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    last_state_change: float = 0.0

    # Configuration
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes before closing from half-open
    timeout_seconds: float = 60.0  # Time before trying half-open
    half_open_max_requests: int = 3  # Max requests in half-open

    def record_failure(self, failure_type: FailureType) -> None:
        """Record a failure."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self._transition_to(CircuitState.OPEN)
                logger.warning(
                    f"Circuit OPEN for {self.provider}/{self.model} "
                    f"after {self.failure_count} failures"
                )
        elif self.state == CircuitState.HALF_OPEN:
            # Any failure in half-open reopens the circuit
            self._transition_to(CircuitState.OPEN)
            logger.warning(
                f"Circuit reopened for {self.provider}/{self.model} "
                f"after failure in half-open state"
            )

    def record_success(self) -> None:
        """Record a success."""
        self.success_count += 1

        if self.state == CircuitState.HALF_OPEN:
            if self.success_count >= self.success_threshold:
                self._transition_to(CircuitState.CLOSED)
                logger.info(
                    f"Circuit CLOSED for {self.provider}/{self.model} "
                    f"after {self.success_count} successes"
                )
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = max(0, self.failure_count - 1)

    def is_available(self) -> bool:
        """Check if the model is available for requests."""
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            # Check if timeout has passed
            if time.time() - self.last_state_change >= self.timeout_seconds:
                self._transition_to(CircuitState.HALF_OPEN)
                return True
            return False
        elif self.state == CircuitState.HALF_OPEN:
            # Allow limited requests in half-open
            return True
        return False

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self.state
        self.state = new_state
        self.last_state_change = time.time()

        if new_state == CircuitState.HALF_OPEN:
            self.success_count = 0
        elif new_state == CircuitState.CLOSED:
            self.failure_count = 0
            self.success_count = 0

        logger.debug(
            f"Circuit state transition for {self.provider}/{self.model}: "
            f"{old_state.name} -> {new_state.name}"
        )


# Backward compatibility alias
CircuitBreaker = ModelCircuitBreaker


@dataclass
class ModelCapability:
    """Capability profile for a model."""

    provider: str
    model: str
    supports_tool_calls: bool = True
    supports_streaming: bool = True
    max_context_tokens: int = 8192
    cost_tier: str = "medium"  # low, medium, high
    strengths: Set[str] = field(default_factory=set)  # e.g., {"code", "analysis", "creative"}
    weaknesses: Set[str] = field(default_factory=set)


class AutomaticModelFallback:
    """Automatic model fallback with circuit breaker pattern.

    Implements ModelFallbackPolicy protocol.

    Features:
    - Circuit breaker per model
    - Capability-based fallback selection
    - Task-type matching
    - Cost-aware fallback ordering

    Follows:
    - Single Responsibility: Only handles model fallback decisions
    - Open/Closed: New models can be added via configuration
    - Dependency Inversion: Works with any provider implementing protocol
    """

    # Default fallback chains by provider
    DEFAULT_FALLBACK_CHAINS: Dict[str, List[Tuple[str, str]]] = {
        # Anthropic fallbacks
        "anthropic": [
            ("anthropic", "claude-3-5-sonnet-latest"),
            ("anthropic", "claude-3-5-haiku-latest"),
            ("openai", "gpt-4o"),
        ],
        # OpenAI fallbacks
        "openai": [
            ("openai", "gpt-4o-mini"),
            ("anthropic", "claude-3-5-sonnet-latest"),
            ("openai", "gpt-3.5-turbo"),
        ],
        # Ollama fallbacks (local)
        "ollama": [
            ("ollama", "qwen2.5-coder:14b"),
            ("ollama", "llama3.1:8b"),
            ("ollama", "mistral:7b"),
            ("lmstudio", "default"),
        ],
        # LMStudio fallbacks (local)
        "lmstudio": [
            ("ollama", "qwen2.5-coder:14b"),
            ("ollama", "llama3.1:8b"),
        ],
        # Groq fallbacks
        "groq": [
            ("groq", "llama-3.1-8b-instant"),
            ("openai", "gpt-4o-mini"),
        ],
    }

    # Default model capabilities
    DEFAULT_CAPABILITIES: Dict[str, ModelCapability] = {
        "claude-3-5-sonnet": ModelCapability(
            provider="anthropic",
            model="claude-3-5-sonnet-latest",
            max_context_tokens=200000,
            cost_tier="high",
            strengths={"code", "analysis", "reasoning"},
        ),
        "claude-3-5-haiku": ModelCapability(
            provider="anthropic",
            model="claude-3-5-haiku-latest",
            max_context_tokens=200000,
            cost_tier="low",
            strengths={"speed", "simple_tasks"},
        ),
        "gpt-4o": ModelCapability(
            provider="openai",
            model="gpt-4o",
            max_context_tokens=128000,
            cost_tier="high",
            strengths={"code", "analysis", "creative"},
        ),
        "gpt-4o-mini": ModelCapability(
            provider="openai",
            model="gpt-4o-mini",
            max_context_tokens=128000,
            cost_tier="medium",
            strengths={"speed", "code"},
        ),
        "qwen2.5-coder": ModelCapability(
            provider="ollama",
            model="qwen2.5-coder:14b",
            max_context_tokens=32768,
            cost_tier="free",
            strengths={"code", "local"},
        ),
        "llama3.1": ModelCapability(
            provider="ollama",
            model="llama3.1:8b",
            max_context_tokens=128000,
            cost_tier="free",
            strengths={"general", "local"},
        ),
    }

    def __init__(
        self,
        fallback_chains: Optional[Dict[str, List[Tuple[str, str]]]] = None,
        capabilities: Optional[Dict[str, ModelCapability]] = None,
        airgapped_mode: bool = False,
    ):
        self._fallback_chains = fallback_chains or dict(self.DEFAULT_FALLBACK_CHAINS)
        self._capabilities = capabilities or dict(self.DEFAULT_CAPABILITIES)
        self._airgapped_mode = airgapped_mode

        # Circuit breakers per model
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}

        # Failure-specific fallback preferences
        self._failure_preferences: Dict[FailureType, List[str]] = {
            # For stuck loops, prefer models with better instruction following
            FailureType.STUCK_LOOP: ["claude", "gpt-4"],
            # For hallucinations, prefer more deterministic models
            FailureType.HALLUCINATED_TOOL: ["claude", "gpt-4o"],
            # For timeouts, prefer faster models
            FailureType.TIMEOUT_APPROACHING: ["haiku", "mini", "instant"],
        }

    def _get_circuit_breaker(self, provider: str, model: str) -> CircuitBreaker:
        """Get or create circuit breaker for a model."""
        key = f"{provider}:{model}"
        if key not in self._circuit_breakers:
            self._circuit_breakers[key] = CircuitBreaker(provider=provider, model=model)
        return self._circuit_breakers[key]

    def get_fallback_model(
        self,
        current_provider: str,
        current_model: str,
        failure_type: FailureType,
    ) -> Optional[Tuple[str, str]]:
        """Get fallback provider and model.

        Args:
            current_provider: Current provider name
            current_model: Current model name
            failure_type: Type of failure that triggered fallback

        Returns:
            Tuple of (provider, model) or None if no fallback available
        """
        # Get fallback chain for current provider
        chain = self._fallback_chains.get(current_provider.lower(), [])

        # Filter out current model
        candidates = [
            (p, m) for p, m in chain if not (p == current_provider and m == current_model)
        ]

        # Filter by airgapped mode
        if self._airgapped_mode:
            candidates = [
                (p, m) for p, m in candidates if p.lower() in ("ollama", "lmstudio", "vllm")
            ]

        # Filter by circuit breaker availability
        available = [
            (p, m) for p, m in candidates if self._get_circuit_breaker(p, m).is_available()
        ]

        if not available:
            logger.warning(f"No available fallback models for {current_provider}/{current_model}")
            return None

        # Score candidates based on failure type preferences
        scored = []
        preferences = self._failure_preferences.get(failure_type, [])

        for provider, model in available:
            score = 0.0

            # Preference matching
            for i, pref in enumerate(preferences):
                if pref.lower() in model.lower() or pref.lower() in provider.lower():
                    score += 1.0 / (i + 1)  # Higher score for earlier preferences

            # Cost consideration (prefer cheaper for non-critical failures)
            cap_key = self._find_capability_key(model)
            if cap_key and cap_key in self._capabilities:
                cap = self._capabilities[cap_key]
                if cap.cost_tier == "free":
                    score += 0.3
                elif cap.cost_tier == "low":
                    score += 0.2
                elif cap.cost_tier == "medium":
                    score += 0.1

            scored.append(((provider, model), score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        best = scored[0][0]
        logger.info(
            f"Fallback selected: {best[0]}/{best[1]} "
            f"for {failure_type.name} (from {current_provider}/{current_model})"
        )
        return best

    def _find_capability_key(self, model: str) -> Optional[str]:
        """Find capability key for a model."""
        model_lower = model.lower()
        for key in self._capabilities:
            if key.lower() in model_lower:
                return key
        return None

    def record_model_failure(
        self,
        provider: str,
        model: str,
        failure_type: FailureType,
    ) -> None:
        """Record a model failure for circuit breaker logic."""
        cb = self._get_circuit_breaker(provider, model)
        cb.record_failure(failure_type)

    def record_model_success(self, provider: str, model: str) -> None:
        """Record a model success."""
        cb = self._get_circuit_breaker(provider, model)
        cb.record_success()

    def is_model_available(self, provider: str, model: str) -> bool:
        """Check if a model is currently available (not circuit-broken)."""
        cb = self._get_circuit_breaker(provider, model)
        return cb.is_available()

    def get_circuit_breaker_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers."""
        return {
            key: {
                "state": cb.state.name,
                "failure_count": cb.failure_count,
                "success_count": cb.success_count,
                "is_available": cb.is_available(),
            }
            for key, cb in self._circuit_breakers.items()
        }

    def add_fallback_chain(
        self,
        provider: str,
        chain: List[Tuple[str, str]],
    ) -> None:
        """Add or update a fallback chain for a provider."""
        self._fallback_chains[provider.lower()] = chain

    def add_model_capability(
        self,
        key: str,
        capability: ModelCapability,
    ) -> None:
        """Add or update model capability."""
        self._capabilities[key] = capability

    def reset_circuit_breakers(self) -> None:
        """Reset all circuit breakers to closed state."""
        for cb in self._circuit_breakers.values():
            cb._transition_to(CircuitState.CLOSED)
        logger.info("All circuit breakers reset to CLOSED")
