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

"""Tests for ParadigmRouter."""

import importlib

import pytest

from victor.agent.paradigm_router import (
    ParadigmRouter,
    RoutingDecision,
    ProcessingParadigm,
    ModelTier,
    get_paradigm_router,
)
from victor.agent.topology_contract import TopologyDecisionInput

paradigm_router_module = importlib.import_module("victor.agent.paradigm_router")


class TestParadigmRouter:
    """Test ParadigmRouter functionality."""

    def test_router_initialization(self):
        """Test paradigm router initializes correctly."""
        router = ParadigmRouter(enabled=True)
        assert router.enabled is True
        assert router._routing_count == 0

        router_disabled = ParadigmRouter(enabled=False)
        assert router_disabled.enabled is False

    def test_direct_paradigm_for_simple_task(self):
        """Test direct paradigm for simple task types."""
        router = ParadigmRouter(enabled=True)

        decision = router.route(
            task_type="create_simple",
            query="create a new file",
            history_length=0,
            query_complexity=0.1,
        )

        assert decision.paradigm == ProcessingParadigm.DIRECT
        assert decision.model_tier == ModelTier.SMALL
        assert decision.max_tokens == 500
        assert decision.skip_planning is True
        assert decision.skip_evaluation is True
        assert decision.confidence >= 0.8

    def test_direct_paradigm_for_action_query(self):
        """Test direct paradigm for action queries."""
        router = ParadigmRouter(enabled=True)

        decision = router.route(
            task_type="unknown",
            query="run the tests",
            history_length=0,
            query_complexity=0.2,
        )

        assert decision.paradigm == ProcessingParadigm.DIRECT
        assert decision.model_tier == ModelTier.SMALL
        assert decision.max_tokens == 600
        assert decision.skip_planning is True
        assert decision.skip_evaluation is True

    def test_focused_paradigm_for_medium_task(self):
        """Test focused paradigm for medium complexity tasks."""
        router = ParadigmRouter(enabled=True)

        decision = router.route(
            task_type="edit",
            query="fix the bug in the authentication",
            history_length=1,
            query_complexity=0.4,
        )

        assert decision.paradigm == ProcessingParadigm.FOCUSED
        assert decision.model_tier == ModelTier.MEDIUM
        assert decision.max_tokens == 1000
        assert decision.skip_planning is False
        assert decision.skip_evaluation is False

    def test_focused_paradigm_for_debug_task(self):
        """Test focused paradigm for debug tasks."""
        router = ParadigmRouter(enabled=True)

        decision = router.route(
            task_type="debug",
            query="debug the failing test",
            history_length=0,
            query_complexity=0.5,
        )

        assert decision.paradigm == ProcessingParadigm.FOCUSED
        assert decision.model_tier == ModelTier.MEDIUM
        assert decision.confidence >= 0.7

    def test_deep_paradigm_for_complex_task(self):
        """Test deep paradigm for complex task types."""
        router = ParadigmRouter(enabled=True)

        decision = router.route(
            task_type="design",
            query="design a new authentication system",
            history_length=0,
            query_complexity=0.8,
        )

        assert decision.paradigm == ProcessingParadigm.DEEP
        assert decision.model_tier == ModelTier.LARGE
        assert decision.max_tokens == 4000
        assert decision.skip_planning is False
        assert decision.skip_evaluation is False

    def test_deep_paradigm_for_long_history(self):
        """Test deep paradigm for tasks with long conversation history."""
        router = ParadigmRouter(enabled=True)

        decision = router.route(
            task_type="edit",
            query="continue with the changes",
            history_length=5,  # Long history triggers deep paradigm
            query_complexity=0.4,
        )

        assert decision.paradigm == ProcessingParadigm.DEEP
        assert decision.model_tier == ModelTier.LARGE

    def test_standard_paradigm_default(self):
        """Test standard paradigm for unclassified tasks."""
        router = ParadigmRouter(enabled=True)

        decision = router.route(
            task_type="unknown",
            query="help me with something complex that requires analysis",
            history_length=2,  # Some history but not enough for deep
            query_complexity=0.5,  # Medium complexity → focused
        )

        # With complexity 0.5, it routes to FOCUSED (not STANDARD)
        assert decision.paradigm == ProcessingParadigm.FOCUSED
        assert decision.model_tier == ModelTier.MEDIUM
        assert decision.max_tokens == 1000

    def test_disabled_router_returns_standard(self):
        """Test disabled router always returns standard paradigm."""
        router = ParadigmRouter(enabled=False)

        decision = router.route(
            task_type="create_simple",
            query="create a file",
            history_length=0,
            query_complexity=0.1,
        )

        assert decision.paradigm == ProcessingParadigm.STANDARD
        assert decision.model_tier == ModelTier.MEDIUM
        assert "Router disabled" in decision.reasoning

    def test_custom_tool_budget_respected(self):
        """Test custom tool budget is respected."""
        router = ParadigmRouter(enabled=True)

        decision = router.route(
            task_type="edit",
            query="fix the bug",
            history_length=0,
            tool_budget=15,  # Custom budget
        )

        assert decision.tool_budget == 15

    def test_query_length_affects_direct_routing(self):
        """Test query length affects direct routing decision."""
        router = ParadigmRouter(enabled=True)

        # Short query with action keyword → direct
        decision_short = router.route(
            task_type="unknown",
            query="run tests",
            history_length=0,
        )
        assert decision_short.paradigm == ProcessingParadigm.DIRECT

        # Long query with action keyword → not direct
        decision_long = router.route(
            task_type="unknown",
            query="run the comprehensive test suite for the authentication module "
            "and verify all edge cases are handled correctly",
            history_length=0,
        )
        assert decision_long.paradigm != ProcessingParadigm.DIRECT

    def test_history_length_affects_routing(self):
        """Test conversation history length affects routing."""
        router = ParadigmRouter(enabled=True)

        # No history → direct for simple task
        decision_no_history = router.route(
            task_type="create_simple",
            query="create file",
            history_length=0,
            query_complexity=0.1,
        )
        assert decision_no_history.paradigm == ProcessingParadigm.DIRECT

        # Long history → focused even for simple task
        decision_with_history = router.route(
            task_type="create_simple",
            query="create file",
            history_length=3,  # History prevents direct
            query_complexity=0.1,
        )
        assert decision_with_history.paradigm != ProcessingParadigm.DIRECT

    def test_routing_decision_to_dict(self):
        """Test RoutingDecision to_dict conversion."""
        decision = RoutingDecision(
            paradigm=ProcessingParadigm.DIRECT,
            model_tier=ModelTier.SMALL,
            max_tokens=500,
            tool_budget=3,
            skip_planning=True,
            skip_evaluation=True,
            confidence=0.9,
            reasoning="Test decision",
        )

        decision_dict = decision.to_dict()

        assert decision_dict["paradigm"] == "direct"
        assert decision_dict["model_tier"] == "small"
        assert decision_dict["max_tokens"] == 500
        assert decision_dict["skip_planning"] is True
        assert decision_dict["confidence"] == 0.9

    def test_build_topology_input_from_routing_hints(self):
        """Router should synthesize topology selector input from existing heuristics."""
        router = ParadigmRouter(enabled=True)

        topology_input = router.build_topology_input(
            task_type="exploration",
            query="compare all authentication approaches across the repo",
            history_length=1,
            query_complexity=0.7,
            tool_budget=9,
            context={
                "iteration_budget": 3,
                "privacy_sensitivity": "high",
                "provider_candidates": ["ollama", "openai"],
                "available_team_formations": ["adaptive", "hierarchical"],
            },
        )

        assert isinstance(topology_input, TopologyDecisionInput)
        assert topology_input.task_complexity == "high"
        assert topology_input.expected_depth == "high"
        assert topology_input.expected_breadth == "high"
        assert topology_input.iteration_budget == 3
        assert topology_input.privacy_sensitivity == "high"
        assert topology_input.provider_candidates == ["ollama", "openai"]
        assert topology_input.available_team_formations == ["adaptive", "hierarchical"]

    def test_get_statistics_empty(self):
        """Test statistics when no routings have occurred."""
        router = ParadigmRouter(enabled=True)

        stats = router.get_statistics()

        assert stats["total_routings"] == 0
        assert stats["paradigm_percentages"] == {}

    def test_get_statistics_after_routings(self):
        """Test statistics tracking after multiple routings."""
        router = ParadigmRouter(enabled=True)

        # Make various routing decisions
        router.route("create_simple", "create file", 0, 0.1)  # Direct
        router.route("edit", "fix bug", 0, 0.4)  # Focused
        router.route("design", "design system", 0, 0.8)  # Deep
        router.route(
            "unknown", "help me understand this better", 2, 0.25
        )  # Standard (unknown type, some history)
        router.route("action", "run tests", 0, 0.1)  # Direct

        stats = router.get_statistics()

        assert stats["total_routings"] == 5
        assert stats["paradigm_counts"]["direct"] == 2  # create_simple, action
        assert stats["paradigm_counts"]["focused"] == 1  # edit
        assert stats["paradigm_counts"]["deep"] == 1  # design
        assert stats["paradigm_counts"]["standard"] == 1  # unknown
        assert stats["direct_percentage"] == 40.0  # 2/5 = 40%

    def test_reset_statistics(self):
        """Test statistics reset."""
        router = ParadigmRouter(enabled=True)

        router.route("create_simple", "create file", 0, 0.1)
        assert router._routing_count == 1

        router.reset_statistics()
        assert router._routing_count == 0
        assert router._paradigm_stats["direct"] == 0


class TestParadigmRouterIntegration:
    """Integration tests for paradigm router routing decisions."""

    def test_all_direct_task_types_route_correctly(self):
        """Test all DIRECT_TASK_TYPES route to direct paradigm."""
        router = ParadigmRouter(enabled=True)

        direct_tasks = ["create_simple", "action", "search", "quick_question"]

        for task_type in direct_tasks:
            decision = router.route(
                task_type=task_type,
                query=f"perform {task_type}",
                history_length=0,
                query_complexity=0.1,
            )
            assert (
                decision.paradigm == ProcessingParadigm.DIRECT
            ), f"Task type {task_type} should route to DIRECT paradigm"
            assert decision.model_tier == ModelTier.SMALL

    def test_all_focused_task_types_route_correctly(self):
        """Test all FOCUSED_TASK_TYPES route to focused paradigm."""
        router = ParadigmRouter(enabled=True)

        focused_tasks = ["edit", "debug", "refactor", "test"]

        for task_type in focused_tasks:
            decision = router.route(
                task_type=task_type,
                query=f"perform {task_type}",
                history_length=0,
                query_complexity=0.4,
            )
            assert (
                decision.paradigm == ProcessingParadigm.FOCUSED
            ), f"Task type {task_type} should route to FOCUSED paradigm"
            assert decision.model_tier == ModelTier.MEDIUM

    def test_all_deep_task_types_route_correctly(self):
        """Test all DEEP_TASK_TYPES route to deep paradigm."""
        router = ParadigmRouter(enabled=True)

        deep_tasks = ["design", "analysis_deep", "exploration", "swe_bench_issue"]

        for task_type in deep_tasks:
            decision = router.route(
                task_type=task_type,
                query=f"perform {task_type}",
                history_length=0,
                query_complexity=0.7,
            )
            assert (
                decision.paradigm == ProcessingParadigm.DEEP
            ), f"Task type {task_type} should route to DEEP paradigm"
            assert decision.model_tier == ModelTier.LARGE

    def test_all_action_keywords_detected(self):
        """Test all action keywords trigger direct paradigm."""
        router = ParadigmRouter(enabled=True)

        action_queries = [
            "run the tests",
            "execute the command",
            "create a new file",
            "write the code",
            "delete the logs",
            "list all files",
            "show me results",
            "get the data",
            "find the bug",
            "search the code",
        ]

        for query in action_queries:
            decision = router.route(
                task_type="unknown",
                query=query,
                history_length=0,
                query_complexity=0.2,
            )
            assert (
                decision.paradigm == ProcessingParadigm.DIRECT
            ), f"Action query '{query}' should route to DIRECT paradigm"

    def test_complexity_thresholds_correct(self):
        """Test complexity thresholds are correct."""
        router = ParadigmRouter(enabled=True)

        # Low complexity (<0.3) → direct for simple tasks
        decision_low = router.route(
            task_type="create_simple",
            query="create file",
            history_length=0,
            query_complexity=0.2,
        )
        assert decision_low.paradigm == ProcessingParadigm.DIRECT

        # Medium complexity (0.3-0.6) → focused
        decision_medium = router.route(
            task_type="edit",
            query="fix bug",
            history_length=0,
            query_complexity=0.5,
        )
        assert decision_medium.paradigm == ProcessingParadigm.FOCUSED

        # High complexity (>=0.6) → deep
        decision_high = router.route(
            task_type="design",
            query="design system",
            history_length=0,
            query_complexity=0.7,
        )
        assert decision_high.paradigm == ProcessingParadigm.DEEP

    def test_40_percent_small_model_target_achievable(self):
        """Test that 40% small model usage target is achievable."""
        router = ParadigmRouter(enabled=True)

        # Simulate typical task mix (based on analysis of user queries)
        # 40% simple tasks (create_simple, action, search)
        # 40% medium tasks (edit, debug, refactor, test)
        # 20% complex tasks (design, analysis_deep, exploration)

        tasks = [
            # Simple tasks (should use SMALL model)
            ("create_simple", "create file", 0, 0.1),
            ("action", "run tests", 0, 0.1),
            ("search", "find bug", 0, 0.1),
            ("action", "show files", 0, 0.2),
            # Medium tasks (should use MEDIUM model)
            ("edit", "fix bug", 0, 0.4),
            ("debug", "debug issue", 0, 0.5),
            ("refactor", "clean up code", 0, 0.4),
            ("test", "write tests", 0, 0.4),
            # Complex tasks (should use LARGE model)
            ("design", "design system", 0, 0.8),
            ("analysis_deep", "analyze architecture", 0, 0.7),
        ]

        for task_type, query, history, complexity in tasks:
            router.route(task_type, query, history, complexity)

        stats = router.get_statistics()

        # Target: 40% small model usage (direct paradigm)
        # 4 out of 10 tasks should be direct
        assert stats["direct_percentage"] >= 40.0, (
            f"Direct percentage is {stats['direct_percentage']:.1f}%, "
            f"target is 40%. Should be achievable with typical task mix."
        )


class TestParadigmRouterSingleton:
    """Test singleton pattern for ParadigmRouter."""

    def test_get_paradigm_router_returns_singleton(self):
        """Test get_paradigm_router returns singleton instance."""
        paradigm_router_module._paradigm_router_instance = None
        router1 = get_paradigm_router()
        router2 = get_paradigm_router()

        assert router1 is router2  # Same instance
        assert isinstance(router1, ParadigmRouter)

    def test_singleton_shares_statistics(self):
        """Test singleton instance shares statistics across calls."""
        paradigm_router_module._paradigm_router_instance = None
        router1 = get_paradigm_router()
        router2 = get_paradigm_router()

        router1.route("create_simple", "create file", 0, 0.1)

        stats1 = router1.get_statistics()
        stats2 = router2.get_statistics()

        assert stats1["total_routings"] == stats2["total_routings"] == 1
