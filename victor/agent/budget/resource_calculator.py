# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Resource-aware budget calculator for parallel exploration.

Calculates optimal exploration parameters based on:
- Model speed (cloud vs local)
- Available hardware (CPU, memory)
- Task complexity
- Provider context window

Usage:
    from victor.agent.budget.resource_calculator import calculate_exploration_budget

    budget = calculate_exploration_budget(
        complexity=TaskComplexity.COMPLEX,
        provider="ollama",
        model="gemma4:latest",
    )
    print(budget.max_parallel_agents)   # 1 (local GPU)
    print(budget.exploration_timeout)   # 135 (scaled for slow model)
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ResourceBudget:
    """Calculated resource budget for exploration."""

    max_parallel_agents: int = 3
    context_per_agent: int = 50000
    exploration_timeout: int = 90
    tool_budget_per_agent: int = 10


# Provider speed multipliers — higher = slower = needs more time
_PROVIDER_SPEED_MULTIPLIERS = {
    "anthropic": 1.0,
    "openai": 1.0,
    "deepseek": 1.3,
    "google": 1.0,
    "xai": 1.1,
    "ollama": 2.0,  # Local models are much slower
    "lmstudio": 2.0,
    "vllm": 1.5,
}

# Local providers that run on GPU (can only serve 1 model at a time)
_LOCAL_PROVIDERS = {"ollama", "lmstudio", "vllm", "llamacpp"}


def calculate_exploration_budget(
    complexity: str = "action",
    provider: str = "ollama",
    model: Optional[str] = None,
) -> ResourceBudget:
    """Calculate optimal exploration budget based on available resources.

    Args:
        complexity: Task complexity level (simple, medium, complex, action, analysis)
        provider: LLM provider name
        model: Optional model name for specific limits

    Returns:
        ResourceBudget with scaled parameters
    """
    # 1. Determine parallelism based on provider type
    if provider in _LOCAL_PROVIDERS:
        # Local GPU can typically only serve 1 model at a time
        max_agents = 1
    else:
        # Cloud APIs support concurrent requests
        cpu_count = os.cpu_count() or 4
        max_agents = min(3, max(1, cpu_count // 2))

    # 2. Get provider-aware context from existing infrastructure
    try:
        from victor.agent.subagents.orchestrator import get_context_for_role
        from victor.agent.subagents.base import SubAgentRole

        context_per_agent = get_context_for_role(SubAgentRole.RESEARCHER, provider, model)
    except Exception:
        context_per_agent = 50000

    # 3. Scale timeout with provider speed
    speed_multiplier = _PROVIDER_SPEED_MULTIPLIERS.get(provider, 1.5)
    base_timeout = 90  # seconds

    # Simple tasks: no exploration. Complex: full budget
    complexity_factor = {
        "simple": 0.0,
        "medium": 0.5,
        "complex": 1.0,
        "generation": 0.3,
        "action": 1.0,
        "analysis": 1.2,
    }.get(complexity, 1.0)

    if complexity_factor == 0.0:
        return ResourceBudget(
            max_parallel_agents=0,
            context_per_agent=0,
            exploration_timeout=0,
            tool_budget_per_agent=0,
        )

    exploration_timeout = int(base_timeout * speed_multiplier * complexity_factor)

    # 4. Scale tool budget with complexity
    tool_budget = int(10 * complexity_factor)

    budget = ResourceBudget(
        max_parallel_agents=max_agents,
        context_per_agent=context_per_agent,
        exploration_timeout=exploration_timeout,
        tool_budget_per_agent=max(tool_budget, 5),
    )

    logger.debug(
        "Resource budget: agents=%d, context=%d, timeout=%ds, tools=%d "
        "(provider=%s, complexity=%s, speed=%.1fx)",
        budget.max_parallel_agents,
        budget.context_per_agent,
        budget.exploration_timeout,
        budget.tool_budget_per_agent,
        provider,
        complexity,
        speed_multiplier,
    )

    return budget
