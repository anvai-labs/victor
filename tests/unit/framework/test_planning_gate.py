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

"""Tests for PlanningGate (Fast-Slow Planning Architecture)."""

import pytest

from victor.framework.agentic_loop import PlanningGate


class TestPlanningGate:
    """Test PlanningGate functionality."""

    def test_gate_initialization(self):
        """Test planning gate initializes correctly."""
        gate = PlanningGate(enabled=True)
        assert gate.enabled is True
        assert gate._fast_path_count == 0
        assert gate._total_decisions == 0

        gate_disabled = PlanningGate(enabled=False)
        assert gate_disabled.enabled is False

    def test_fast_pattern_create_simple_returns_false(self):
        """Test create_simple task with low tool budget returns False (skip planning)."""
        gate = PlanningGate(enabled=True)

        result = gate.should_use_llm_planning(
            task_type="create_simple",
            tool_budget=2,
            query_complexity=None,
            query_length=0,
        )

        assert result is False, "create_simple with tool_budget=2 should skip planning"

    def test_fast_pattern_action_returns_false(self):
        """Test action task returns False (skip planning)."""
        gate = PlanningGate(enabled=True)

        result = gate.should_use_llm_planning(
            task_type="action",
            tool_budget=1,
            query_complexity=None,
            query_length=0,
        )

        assert result is False, "action task should skip planning"

    def test_fast_pattern_search_returns_false(self):
        """Test search task returns False (skip planning)."""
        gate = PlanningGate(enabled=True)

        result = gate.should_use_llm_planning(
            task_type="search",
            tool_budget=3,
            query_complexity=None,
            query_length=0,
        )

        assert result is False, "search task should skip planning"

    def test_low_complexity_returns_false(self):
        """Test low query complexity (<0.3) returns False (skip planning)."""
        gate = PlanningGate(enabled=True)

        result = gate.should_use_llm_planning(
            task_type="edit",
            tool_budget=5,
            query_complexity=0.2,  # Low complexity
            query_length=0,
        )

        assert result is False, "low complexity (0.2) should skip planning"

    def test_short_action_query_returns_false(self):
        """Test short action query returns False (skip planning)."""
        gate = PlanningGate(enabled=True)

        result = gate.should_use_llm_planning(
            task_type="unknown",
            tool_budget=5,
            query_complexity=None,
            query_length=30,
            context={"query": "run the tests"},  # Short query with action keyword
        )

        assert result is False, "short action query should skip planning"

    def test_disabled_gate_always_returns_true(self):
        """Test disabled gate always returns True (use LLM planning)."""
        gate = PlanningGate(enabled=False)

        result = gate.should_use_llm_planning(
            task_type="create_simple",
            tool_budget=2,
            query_complexity=0.1,
            query_length=10,
        )

        assert result is True, "disabled gate should always use LLM planning"

    def test_planning_feedback_can_force_llm_planning_over_fast_path(self):
        """Experiment-memory constraints can veto fast-path planning skips."""
        gate = PlanningGate(enabled=True)

        result = gate.should_use_llm_planning(
            task_type="create_simple",
            tool_budget=2,
            query_complexity=None,
            query_length=20,
            routing_hints={
                "planning_force_llm": True,
                "planning_force_reason": "experiment_constraints: tests_pass",
            },
        )

        assert result is True, "planning constraints should force LLM planning"
        stats = gate.get_statistics()
        assert stats["forced_slow_path_count"] == 1

    def test_planning_feedback_can_prefer_fast_path_on_close_call(self):
        """Negative planning bias can widen fast-path thresholds on safe close calls."""
        gate = PlanningGate(enabled=True)

        result = gate.should_use_llm_planning(
            task_type="unknown",
            tool_budget=4,
            query_complexity=0.34,
            query_length=60,
            context={"query": "run the tests"},
            routing_hints={
                "planning_prefer_fast_path": True,
                "planning_prefer_reason": "experiment_policy_bias: heuristic_fast_path",
                "planning_fast_path_tool_budget_limit": 4,
                "planning_fast_path_query_length_limit": 70,
                "planning_fast_path_complexity_threshold": 0.36,
            },
        )

        assert result is False, "learned fast-path preference should skip planning"
        stats = gate.get_statistics()
        assert stats["learned_fast_path_count"] == 1

    def test_complex_task_returns_true(self):
        """Test complex task returns True (use LLM planning)."""
        gate = PlanningGate(enabled=True)

        result = gate.should_use_llm_planning(
            task_type="design",
            tool_budget=10,
            query_complexity=0.8,  # High complexity
            query_length=100,
        )

        assert result is True, "complex task should use LLM planning"

    def test_high_tool_budget_returns_true(self):
        """Test task with high tool budget returns True (use LLM planning)."""
        gate = PlanningGate(enabled=True)

        result = gate.should_use_llm_planning(
            task_type="create_simple",
            tool_budget=10,  # High budget despite simple task
            query_complexity=None,
            query_length=0,
        )

        assert result is True, "high tool budget should trigger LLM planning"

    def test_long_query_without_action_keywords_returns_true(self):
        """Test long query without action keywords returns True."""
        gate = PlanningGate(enabled=True)

        result = gate.should_use_llm_planning(
            task_type="unknown",
            tool_budget=5,
            query_complexity=None,
            query_length=100,
            context={
                "query": "I need to understand the architecture of this system"
            },  # Long, no action keyword
        )

        assert result is True, "long query without action keywords should use LLM planning"

    def test_get_statistics(self):
        """Test statistics tracking works."""
        gate = PlanningGate(enabled=True)

        # Make some decisions
        gate.should_use_llm_planning("create_simple", 2, None, 0)  # Fast path
        gate.should_use_llm_planning("edit", 5, None, 0)  # Slow path
        gate.should_use_llm_planning("action", 1, 0.1, 0)  # Fast path (matches FAST_PATTERNS)
        gate.should_use_llm_planning("design", 10, 0.8, 100)  # Slow path

        stats = gate.get_statistics()

        assert stats["fast_path_count"] == 2
        assert stats["total_decisions"] == 4
        assert stats["fast_path_percentage"] == 50.0  # 2/4 = 50%

    def test_no_statistics_when_disabled(self):
        """Test statistics when gate is disabled."""
        gate = PlanningGate(enabled=False)

        gate.should_use_llm_planning("create_simple", 2, None, 0)

        stats = gate.get_statistics()

        # Should still track even when disabled
        assert stats["total_decisions"] == 1
        assert stats["fast_path_count"] == 0
        assert stats["fast_path_percentage"] == 0.0


class TestPlanningGateIntegration:
    """Integration tests for planning gate with fast-path detection."""

    def test_fast_path_patterns_match_expected_simple_tasks(self):
        """Test that fast-path patterns align with simple tasks."""
        gate = PlanningGate(enabled=True)

        # All these should be fast-path (skip planning)
        fast_patterns = [
            ("create_simple", 2, None, 0),
            ("action", 1, None, 0),
            ("search", 3, None, 0),
            ("quick_question", 1, None, 0),
        ]

        for task_type, tool_budget, complexity, length in fast_patterns:
            result = gate.should_use_llm_planning(
                task_type=task_type,
                tool_budget=tool_budget,
                query_complexity=complexity,
                query_length=length,
            )
            assert result is False, f"{task_type} should be fast-path"

    def test_slow_path_patterns_require_planning(self):
        """Test that complex tasks require planning."""
        gate = PlanningGate(enabled=True)

        # These should be slow-path (use LLM planning)
        slow_patterns = [
            ("edit", 5, 0.5, 50),  # Medium complexity
            ("design", 10, 0.8, 100),  # High complexity
            ("analysis_deep", 15, 0.9, 200),  # Very high complexity
        ]

        for task_type, tool_budget, complexity, length in slow_patterns:
            result = gate.should_use_llm_planning(
                task_type=task_type,
                tool_budget=tool_budget,
                query_complexity=complexity,
                query_length=length,
            )
            assert result is True, f"{task_type} should require planning"

    def test_action_keywords_detected_in_short_queries(self):
        """Test that action keywords trigger fast-path even for unknown task types."""
        gate = PlanningGate(enabled=True)

        action_queries = [
            "run the tests",
            "create a new file",
            "delete the old logs",
            "list all files",
            "show me the results",
        ]

        for query in action_queries:
            result = gate.should_use_llm_planning(
                task_type="unknown",
                tool_budget=5,
                query_complexity=None,
                query_length=len(query),
                context={"query": query},
            )
            assert result is False, f"Action query '{query}' should be fast-path"

    def test_30_percent_fast_path_target_achievable(self):
        """Test that 30% fast-path target is achievable with typical task mix."""
        gate = PlanningGate(enabled=True)

        # Simulate a typical task mix (based on analysis of user queries)
        # 30% simple tasks (create_simple, action, search)
        # 70% complex tasks (edit, debug, analysis_deep, design)

        # Simple tasks (should be fast-path)
        simple_tasks = [
            ("create_simple", 2, None, 20),
            ("action", 1, None, 15),
            ("search", 3, None, 25),
        ]

        # Complex tasks (should use planning)
        complex_tasks = [
            ("edit", 5, 0.5, 60),
            ("debug", 8, 0.6, 80),
            ("analysis_deep", 12, 0.7, 150),
            ("design", 15, 0.8, 200),
            ("exploration", 10, 0.6, 100),
            ("exploration", 10, 0.6, 100),
            ("edit", 5, 0.5, 60),
        ]

        # Run all tasks through gate
        fast_count = 0
        for task_type, tool_budget, complexity, length in simple_tasks + complex_tasks:
            result = gate.should_use_llm_planning(
                task_type=task_type,
                tool_budget=tool_budget,
                query_complexity=complexity,
                query_length=length,
            )
            if not result:
                fast_count += 1

        fast_path_percentage = (fast_count / len(simple_tasks + complex_tasks)) * 100

        # Target: 30% fast-path
        assert fast_path_percentage >= 30.0, (
            f"Fast-path percentage is {fast_path_percentage:.1f}%, "
            f"target is 30%. Should be achievable with typical task mix."
        )


# =============================================================================
# Wave E: PlanningGateConfig threshold consolidation tests
# =============================================================================


class TestPlanningGateConfig:
    """Verify PlanningGateConfig consolidates the 60 vs 50 inconsistency."""

    def test_config_importable(self):
        from victor.framework.agentic_loop import PlanningGateConfig

        cfg = PlanningGateConfig()
        assert cfg is not None

    def test_default_short_query_threshold_is_50(self):
        from victor.framework.agentic_loop import PlanningGateConfig

        cfg = PlanningGateConfig()
        assert cfg.short_query_threshold == 50

    def test_default_learned_threshold_is_60(self):
        from victor.framework.agentic_loop import PlanningGateConfig

        cfg = PlanningGateConfig()
        assert cfg.learned_threshold_default == 60

    def test_short_query_threshold_configurable(self):
        from victor.framework.agentic_loop import PlanningGate, PlanningGateConfig

        cfg = PlanningGateConfig(short_query_threshold=30)
        gate = PlanningGate(config=cfg)
        # query_length=35 >= short_query_threshold=30 → NOT fast-pathed by short-query rule
        # task_type='action' with tool_budget=5 > 3 → NOT fast-pathed by pattern 1 either
        result = gate.should_use_llm_planning(
            task_type="action",
            tool_budget=5,
            query_length=35,
            context={
                "query": "run execute something meaningful here for the test query length check"
            },
        )
        assert result is True  # slow path (35 >= 30)

    def test_query_below_custom_threshold_takes_fast_path(self):
        from victor.framework.agentic_loop import PlanningGate, PlanningGateConfig

        cfg = PlanningGateConfig(short_query_threshold=40)
        gate = PlanningGate(config=cfg)
        # query_length=25 < 40 → fast-path (context contains action keyword "run ")
        result = gate.should_use_llm_planning(
            task_type="unknown",
            tool_budget=5,
            query_length=25,
            context={"query": "run the build now"},
        )
        assert result is False  # fast-path via short-query rule

    def test_learned_threshold_used_in_routing_hint(self):
        from victor.framework.agentic_loop import PlanningGate, PlanningGateConfig

        cfg = PlanningGateConfig(learned_threshold_default=40)
        gate = PlanningGate(config=cfg)
        # With learned threshold 40, query_length=35 should be fast-pathed via routing hint
        routing_hints = {
            "planning_prefer_fast_path": True,
            "planning_fast_path_query_length_limit": None,  # falls back to config default (40)
            "planning_fast_path_tool_budget_limit": 10,
            "planning_prefer_reason": "test",
        }
        result = gate.should_use_llm_planning(
            task_type="action",
            tool_budget=5,
            query_length=35,
            context={"query": "run execute the build"},
            routing_hints=routing_hints,
        )
        assert result is False  # fast-path: 35 <= 40 learned threshold

    def test_both_thresholds_read_from_same_config_object(self):
        from victor.framework.agentic_loop import PlanningGate, PlanningGateConfig

        cfg = PlanningGateConfig(short_query_threshold=30, learned_threshold_default=45)
        gate = PlanningGate(config=cfg)
        assert gate._config.short_query_threshold == 30
        assert gate._config.learned_threshold_default == 45

    def test_routing_hint_still_overrides_learned_threshold(self):
        """An explicit routing hint value overrides the config default."""
        from victor.framework.agentic_loop import PlanningGate, PlanningGateConfig

        cfg = PlanningGateConfig(learned_threshold_default=20)
        gate = PlanningGate(config=cfg)
        # Routing hint explicitly sets limit to 80 → query of length 70 fast-pathed
        routing_hints = {
            "planning_prefer_fast_path": True,
            "planning_fast_path_query_length_limit": 80,
            "planning_fast_path_tool_budget_limit": 10,
            "planning_prefer_reason": "explicit_override",
        }
        result = gate.should_use_llm_planning(
            task_type="action",
            tool_budget=5,
            query_length=70,
            context={"query": "run execute the command now do it"},
            routing_hints=routing_hints,
        )
        assert result is False  # fast-path via explicit routing hint (80 > 70)

    def test_agentic_loop_config_has_planning_gate_config(self):
        from victor.framework.agentic_loop import AgenticLoopConfig, PlanningGateConfig

        cfg = AgenticLoopConfig()
        # AgenticLoopConfig should expose a planning_gate field
        import dataclasses

        field_names = {f.name for f in dataclasses.fields(AgenticLoopConfig)}
        assert (
            "planning_gate" in field_names
        ), "AgenticLoopConfig must have a 'planning_gate: PlanningGateConfig' field"
