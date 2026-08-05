# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for resource-aware budget calculator."""

from victor.agent.budget.resource_calculator import (
    ResourceBudget,
    calculate_exploration_budget,
)


class TestResourceCalculator:

    def test_ollama_gets_1_agent(self):
        """Local provider → max 1 parallel agent (GPU limitation)."""
        budget = calculate_exploration_budget(provider="ollama")
        assert budget.max_parallel_agents == 1

    def test_cloud_gets_multiple_agents(self):
        """Cloud API → multiple parallel agents."""
        budget = calculate_exploration_budget(provider="anthropic")
        assert budget.max_parallel_agents >= 1

    def test_deepseek_gets_multiple_agents(self):
        """DeepSeek cloud → multiple agents."""
        budget = calculate_exploration_budget(provider="deepseek")
        assert budget.max_parallel_agents >= 1

    def test_simple_task_no_exploration(self):
        """Simple tasks → 0 agents (no exploration needed)."""
        budget = calculate_exploration_budget(complexity="simple")
        assert budget.max_parallel_agents == 0
        assert budget.exploration_timeout == 0

    def test_complex_task_full_budget(self):
        """Complex tasks → full exploration budget."""
        budget = calculate_exploration_budget(complexity="complex", provider="anthropic")
        assert budget.exploration_timeout >= 90
        assert budget.tool_budget_per_agent >= 5

    def test_ollama_timeout_longer(self):
        """Local models get longer timeout (speed multiplier)."""
        cloud = calculate_exploration_budget(complexity="action", provider="anthropic")
        local = calculate_exploration_budget(complexity="action", provider="ollama")
        assert local.exploration_timeout > cloud.exploration_timeout

    def test_analysis_task_longer_timeout(self):
        """Analysis tasks get more time than action tasks."""
        action = calculate_exploration_budget(complexity="action")
        analysis = calculate_exploration_budget(complexity="analysis")
        assert analysis.exploration_timeout >= action.exploration_timeout

    def test_default_budget(self):
        """Default budget has sensible values."""
        budget = ResourceBudget()
        assert budget.max_parallel_agents == 3
        assert budget.exploration_timeout == 90
        assert budget.tool_budget_per_agent == 10
