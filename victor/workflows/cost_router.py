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

"""Cost-aware routing for provider and model selection.

Routes LLM requests to appropriate providers based on cost tier constraints.
Supports automatic fallback to lower-cost options when budget constraints apply.

Cost Tiers:
- FREE: Local models, cached responses
- LOW: Haiku, small models, batch API
- MEDIUM: Sonnet, standard models
- HIGH: Opus, premium models

Example:
    from victor.workflows.cost_router import CostAwareRouter, CostTier

    router = CostAwareRouter()

    # Get best model for constraints
    model = router.select_model(
        max_cost_tier=CostTier.LOW,
        task_type="simple_extraction",
    )

    # Get provider recommendations
    recommendations = router.get_recommendations(
        max_cost_tier=CostTier.MEDIUM,
        required_capabilities=["tool_calling", "streaming"],
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CostTier(IntEnum):
    """Cost tier for providers and models.

    Lower values = lower cost. Used for comparison operations.
    """

    FREE = 0  # Local models, cached responses
    LOW = 1  # Haiku, small models, batch API
    MEDIUM = 2  # Sonnet, standard models
    HIGH = 3  # Opus, premium models

    @classmethod
    def from_string(cls, value: str) -> "CostTier":
        """Parse cost tier from string."""
        mapping = {
            "free": cls.FREE,
            "low": cls.LOW,
            "medium": cls.MEDIUM,
            "high": cls.HIGH,
        }
        return mapping.get(value.lower(), cls.MEDIUM)


@dataclass
class ModelConfig:
    """Configuration for a model.

    Attributes:
        name: Model identifier (e.g., "claude-3-haiku")
        provider: Provider name (e.g., "anthropic")
        cost_tier: Cost tier for this model
        capabilities: List of capabilities (tool_calling, streaming, vision, etc.)
        max_context: Maximum context window size
        input_cost_per_1k: Cost per 1k input tokens (for reference)
        output_cost_per_1k: Cost per 1k output tokens (for reference)
    """

    name: str
    provider: str
    cost_tier: CostTier
    capabilities: List[str] = field(default_factory=list)
    max_context: int = 100000
    input_cost_per_1k: float = 0.0
    output_cost_per_1k: float = 0.0

    def has_capability(self, capability: str) -> bool:
        """Check if model has a specific capability."""
        return capability in self.capabilities


@dataclass
class RoutingDecision:
    """Result of a routing decision.

    Attributes:
        model: Selected model name
        provider: Selected provider name
        cost_tier: Cost tier of selected model
        reason: Explanation for the decision
        alternatives: List of alternative models considered
    """

    model: str
    provider: str
    cost_tier: CostTier
    reason: str
    alternatives: List[str] = field(default_factory=list)


class CostAwareRouter:
    """Routes requests to appropriate providers based on cost constraints.

    Maintains a registry of models and their cost tiers, and provides
    intelligent routing based on task requirements and budget constraints.

    Example:
        router = CostAwareRouter()

        # Register custom model
        router.register_model(ModelConfig(
            name="my-local-model",
            provider="ollama",
            cost_tier=CostTier.FREE,
            capabilities=["tool_calling"],
        ))

        # Get routing decision
        decision = router.route(
            max_cost_tier=CostTier.LOW,
            required_capabilities=["tool_calling"],
        )
    """

    # Default model registry
    DEFAULT_MODELS: List[ModelConfig] = [
        # Anthropic models
        ModelConfig(
            name="claude-opus-4-20250514",
            provider="anthropic",
            cost_tier=CostTier.HIGH,
            capabilities=["tool_calling", "streaming", "vision", "code", "reasoning"],
            max_context=200000,
        ),
        ModelConfig(
            name="claude-sonnet-4-20250514",
            provider="anthropic",
            cost_tier=CostTier.MEDIUM,
            capabilities=["tool_calling", "streaming", "vision", "code"],
            max_context=200000,
        ),
        ModelConfig(
            name="claude-3-5-haiku-20241022",
            provider="anthropic",
            cost_tier=CostTier.LOW,
            capabilities=["tool_calling", "streaming", "code"],
            max_context=200000,
        ),
        # OpenAI models
        ModelConfig(
            name="gpt-4o",
            provider="openai",
            cost_tier=CostTier.MEDIUM,
            capabilities=["tool_calling", "streaming", "vision", "code"],
            max_context=128000,
        ),
        ModelConfig(
            name="gpt-4o-mini",
            provider="openai",
            cost_tier=CostTier.LOW,
            capabilities=["tool_calling", "streaming", "vision", "code"],
            max_context=128000,
        ),
        # Local/Free models
        ModelConfig(
            name="ollama/llama3.2",
            provider="ollama",
            cost_tier=CostTier.FREE,
            capabilities=["streaming", "code"],
            max_context=8192,
        ),
        ModelConfig(
            name="cached",
            provider="cache",
            cost_tier=CostTier.FREE,
            capabilities=[],
            max_context=0,
        ),
    ]

    def __init__(
        self,
        models: Optional[List[ModelConfig]] = None,
        default_provider: str = "anthropic",
        default_model: str = "claude-sonnet-4-20250514",
    ):
        """Initialize router with model configurations.

        Args:
            models: List of model configurations (uses defaults if not provided)
            default_provider: Default provider when no constraints match
            default_model: Default model when no constraints match
        """
        self._models: Dict[str, ModelConfig] = {}
        self._default_provider = default_provider
        self._default_model = default_model

        # Register default models
        for model in models or self.DEFAULT_MODELS:
            self.register_model(model)

    def register_model(self, config: ModelConfig) -> None:
        """Register a model configuration.

        Args:
            config: Model configuration to register
        """
        self._models[config.name] = config
        logger.debug(f"Registered model: {config.name} (tier={config.cost_tier.name})")

    def get_model(self, name: str) -> Optional[ModelConfig]:
        """Get model configuration by name."""
        return self._models.get(name)

    def get_models_by_tier(self, tier: CostTier) -> List[ModelConfig]:
        """Get all models at a specific cost tier."""
        return [m for m in self._models.values() if m.cost_tier == tier]

    def get_models_up_to_tier(self, max_tier: CostTier) -> List[ModelConfig]:
        """Get all models at or below a cost tier."""
        return [m for m in self._models.values() if m.cost_tier <= max_tier]

    def route(
        self,
        max_cost_tier: Optional[CostTier] = None,
        required_capabilities: Optional[List[str]] = None,
        preferred_provider: Optional[str] = None,
        min_context: Optional[int] = None,
        task_hint: Optional[str] = None,
    ) -> RoutingDecision:
        """Route to best model based on constraints.

        Args:
            max_cost_tier: Maximum allowed cost tier
            required_capabilities: Required model capabilities
            preferred_provider: Preferred provider (used as tiebreaker)
            min_context: Minimum required context window
            task_hint: Hint about task type for optimization

        Returns:
            RoutingDecision with selected model and alternatives
        """
        max_tier = max_cost_tier or CostTier.HIGH
        required_caps = set(required_capabilities or [])

        # Filter models by constraints
        candidates = []
        for model in self._models.values():
            # Check cost tier
            if model.cost_tier > max_tier:
                continue

            # Check capabilities
            if required_caps and not required_caps.issubset(set(model.capabilities)):
                continue

            # Check context window
            if min_context and model.max_context < min_context:
                continue

            candidates.append(model)

        if not candidates:
            # No candidates found, return default with warning
            logger.warning(
                f"No models found for constraints: tier<={max_tier.name}, "
                f"caps={required_caps}, falling back to default"
            )
            return RoutingDecision(
                model=self._default_model,
                provider=self._default_provider,
                cost_tier=CostTier.MEDIUM,
                reason="No matching models found, using default",
                alternatives=[],
            )

        # Sort candidates by preference
        def score_model(m: ModelConfig) -> Tuple[int, int, bool]:
            tier_score = m.cost_tier  # Lower is better
            cap_score = -len(m.capabilities)  # More caps is better (negative)
            preferred = m.provider == preferred_provider
            return (tier_score, cap_score, not preferred)

        candidates.sort(key=score_model)

        best = candidates[0]
        alternatives = [m.name for m in candidates[1:4]]  # Top 3 alternatives

        reason = f"Selected {best.name} (tier={best.cost_tier.name})"
        if preferred_provider and best.provider == preferred_provider:
            reason += ", preferred provider match"
        if task_hint:
            reason += f", task={task_hint}"

        return RoutingDecision(
            model=best.name,
            provider=best.provider,
            cost_tier=best.cost_tier,
            reason=reason,
            alternatives=alternatives,
        )

    def select_for_constraints(
        self,
        constraints_dict: Dict[str, Any],
    ) -> RoutingDecision:
        """Route based on TaskConstraints dictionary.

        Convenience method that extracts routing parameters from
        a constraints dictionary (as returned by constraints.to_dict()).

        Args:
            constraints_dict: Dictionary with constraint settings

        Returns:
            RoutingDecision based on constraints
        """
        max_tier_str = constraints_dict.get("max_cost_tier", "HIGH")
        max_tier = CostTier.from_string(max_tier_str)

        # Check if LLM is allowed
        if not constraints_dict.get("llm_allowed", True):
            # Return cached/free option
            return RoutingDecision(
                model="cached",
                provider="cache",
                cost_tier=CostTier.FREE,
                reason="LLM not allowed by constraints",
                alternatives=[],
            )

        return self.route(max_cost_tier=max_tier)

    def estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Estimate cost for a model invocation.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in dollars
        """
        config = self.get_model(model)
        if not config:
            return 0.0

        # Canonical pricing first (config/provider_metrics.yaml via
        # metrics_capabilities) — the previous inline math multiplied
        # per-MTok reference numbers as if they were per-1k, overestimating
        # by 1000x. Legacy per-1k fields remain only as a fallback for
        # user-registered models with no canonical entry.
        from victor.config.metrics_capabilities import get_metrics_capabilities

        capabilities = get_metrics_capabilities(config.provider, config.name)
        if capabilities.cost_enabled:
            return capabilities.calculate_cost(input_tokens, output_tokens)["total_cost"]

        input_cost = (input_tokens / 1000) * config.input_cost_per_1k
        output_cost = (output_tokens / 1000) * config.output_cost_per_1k
        return input_cost + output_cost


# Global router instance for convenience
_default_router: Optional[CostAwareRouter] = None


def get_default_router() -> CostAwareRouter:
    """Get or create the default global router."""
    global _default_router
    if _default_router is None:
        _default_router = CostAwareRouter()
    return _default_router


def route_for_cost(
    max_cost_tier: str = "HIGH",
    required_capabilities: Optional[List[str]] = None,
) -> RoutingDecision:
    """Convenience function for quick routing.

    Args:
        max_cost_tier: Maximum cost tier as string
        required_capabilities: Required capabilities

    Returns:
        RoutingDecision for best matching model
    """
    router = get_default_router()
    tier = CostTier.from_string(max_cost_tier)
    return router.route(
        max_cost_tier=tier,
        required_capabilities=required_capabilities,
    )


__all__ = [
    "CostTier",
    "ModelConfig",
    "RoutingDecision",
    "CostAwareRouter",
    "get_default_router",
    "route_for_cost",
]
