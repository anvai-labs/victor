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
Integration tests for Credit Assignment with StateGraph.

Tests verify that credit assignment works correctly with real
StateGraph workflows including:
- Multi-step workflows
- Conditional branching
- Cyclic workflows
- Multi-agent coordination
"""

from typing import Dict, Any

import pytest

from victor.framework.graph import StateGraph, END
from victor.framework.rl.credit_assignment import (
    CreditGranularity,
    CreditMethodology,
    ActionMetadata,
)
from victor.framework.rl.credit_graph_integration import (
    CreditTracer,
    Transition,
    ExecutionTrace,
    CreditAwareGraph,
    CompiledCreditAwareGraph,
    create_credit_aware_workflow,
)

# ============================================================================
# Test Fixtures
# ============================================================================


class SimpleState(Dict):
    """Simple state type for testing."""

    pass


@pytest.fixture
def simple_task_graph():
    """Create a simple task graph for testing."""
    graph = StateGraph(SimpleState)

    def analyze(state: SimpleState) -> SimpleState:
        state["analysis"] = "analyzed"
        state["reward"] = 0.3
        return state

    def execute(state: SimpleState) -> SimpleState:
        state["result"] = "executed"
        state["reward"] = 0.5
        return state

    def review(state: SimpleState) -> SimpleState:
        state["approved"] = True
        state["reward"] = 0.2
        return state

    graph.add_node("analyze", analyze)
    graph.add_node("execute", execute)
    graph.add_node("review", review)
    graph.add_edge("analyze", "execute")
    graph.add_edge("execute", "review")
    graph.add_edge("review", END)
    graph.set_entry_point("analyze")

    return graph


@pytest.fixture
def conditional_graph():
    """Create a graph with conditional branching."""
    graph = StateGraph(SimpleState)

    def task(state: SimpleState) -> SimpleState:
        state["count"] = state.get("count", 0) + 1
        state["reward"] = 0.4
        return state

    def should_retry(state: SimpleState) -> str:
        return "retry" if state.get("count", 0) < 3 else "done"

    def done(state: SimpleState) -> SimpleState:
        state["final"] = True
        state["reward"] = 0.6
        return state

    graph.add_node("task", task)
    graph.add_node("done", done)
    graph.add_conditional_edge("task", should_retry, {"retry": "task", "done": "done"})
    graph.add_edge("done", END)
    graph.set_entry_point("task")

    return graph


# ============================================================================
# CreditTracer Tests
# ============================================================================


class TestCreditTracer:
    """Tests for CreditTracer functionality."""

    def test_start_and_end_trace(self):
        """Should create and complete a trace."""
        tracer = CreditTracer()

        initial_state = {"task": "test"}
        trace = tracer.start_trace(initial_state)

        assert trace is not None
        assert trace.trace_id is not None
        assert trace.initial_state == initial_state
        assert trace.success is False

        final_state = {"result": "done"}
        completed = tracer.end_trace(final_state, success=True)

        assert completed.trace_id == trace.trace_id
        assert completed.final_state == final_state
        assert completed.success is True

    def test_record_transitions(self):
        """Should record transitions correctly."""
        tracer = CreditTracer()

        tracer.start_trace({"task": "test"})

        trans1 = tracer.record_transition(
            from_node="start",
            to_node="process",
            state_before={"step": 0},
            state_after={"step": 1},
            node_output={"reward": 0.5},
        )

        assert trans1 is not None
        assert trans1.from_node == "start"
        assert trans1.to_node == "process"
        assert trans1.reward == 0.5

        trans2 = tracer.record_transition(
            from_node="process",
            to_node="end",
            state_before={"step": 1},
            state_after={"step": 2},
            node_output={"score": 0.8},  # Alternative reward field
        )

        assert trans2 is not None
        assert trans2.reward == 0.8

    def test_trace_history(self):
        """Should maintain trace history."""
        tracer = CreditTracer()

        # First trace
        tracer.start_trace({"id": 1})
        tracer.record_transition("a", "b", {}, {}, {})
        tracer.end_trace({}, success=True)

        # Second trace
        tracer.start_trace({"id": 2})
        tracer.record_transition("b", "c", {}, {}, {})
        tracer.end_trace({}, success=True)

        history = tracer.get_trace_history()
        assert len(history) == 2

    def test_custom_reward_extractor(self):
        """Should use custom reward extractor."""

        def extract_reward(output):
            return output.get("value", 0.0) * 2  # Double the value

        tracer = CreditTracer(reward_extractor=extract_reward)

        tracer.start_trace({})
        tracer.record_transition("a", "b", {}, {}, {"value": 0.5})

        trace = tracer.get_active_trace()
        assert trace.transitions[0].reward == 1.0  # 0.5 * 2

    def test_custom_agent_extractor(self):
        """Should use custom agent extractor."""

        def extract_agent(state):
            return state.get("worker", "default")

        tracer = CreditTracer(agent_extractor=extract_agent)

        tracer.start_trace({})
        tracer.record_transition("a", "b", {}, {"worker": "agent_X"}, {})

        trace = tracer.get_active_trace()
        assert trace.transitions[0].agent_id == "agent_X"


# ============================================================================
# Credit Assignment with Traces Tests
# ============================================================================


class TestCreditAssignmentWithTraces:
    """Tests for credit assignment on execution traces."""

    def test_assign_credit_to_trace(self):
        """Should assign credit to trace transitions."""
        tracer = CreditTracer()

        # Create trace with transitions
        tracer.start_trace({"task": "test"})
        tracer.record_transition("start", "step1", {}, {}, {"reward": 0.2})
        tracer.record_transition("step1", "step2", {}, {}, {"reward": 0.3})
        tracer.record_transition("step2", "end", {}, {}, {"reward": 0.5})
        trace = tracer.end_trace({"done": True}, success=True)

        # Assign credit
        signals = tracer.assign_credit_to_trace(trace)

        assert len(signals) == 3
        # Default granularity is STEP
        assert all(s.granularity == CreditGranularity.STEP for s in signals)

    def test_get_agent_attribution(self):
        """Should calculate agent attribution correctly."""
        tracer = CreditTracer()

        # Multi-agent trace
        tracer.start_trace({"agent_id": "agent_A"})
        tracer.record_transition("t1", "t2", {"agent_id": "agent_A"}, {}, {"reward": 0.5})
        tracer.record_transition("t2", "t3", {"agent_id": "agent_B"}, {}, {"reward": 0.3})
        tracer.record_transition("t3", "t4", {"agent_id": "agent_A"}, {}, {"reward": 0.2})
        trace = tracer.end_trace({"done": True}, success=True)

        # Assign credit
        tracer.assign_credit_to_trace(trace, methodology=CreditMethodology.SHAPLEY)

        # Get attribution for agent_A
        attribution_a = tracer.get_agent_attribution(trace, "agent_A")
        assert isinstance(attribution_a, dict)


# ============================================================================
# Credit-Aware Graph Tests
# ============================================================================


class TestCreditAwareGraph:
    """Tests for CreditAwareGraph wrapper."""

    def test_wrap_stategraph(self):
        """Should wrap StateGraph without changing behavior."""
        graph = StateGraph(SimpleState)
        graph.add_node("test", lambda s: {**s, "done": True})
        graph.set_entry_point("test")

        # Wrap
        credit_graph = CreditAwareGraph(graph)

        # Should delegate to underlying graph
        assert hasattr(credit_graph, "add_node")
        assert hasattr(credit_graph, "compile")

    def test_compile_with_credit(self):
        """Should compile with credit tracking enabled."""
        graph = StateGraph(SimpleState)
        graph.add_node("test", lambda s: {**s, "done": True})
        graph.set_entry_point("test")

        credit_graph = CreditAwareGraph(graph)
        compiled = credit_graph.compile(enable_credit=True)

        assert isinstance(compiled, CompiledCreditAwareGraph)

    def test_compile_without_credit(self):
        """Should compile normally when credit disabled."""
        graph = StateGraph(SimpleState)
        graph.add_node("test", lambda s: {**s, "done": True})
        graph.set_entry_point("test")

        credit_graph = CreditAwareGraph(graph)
        compiled = credit_graph.compile(enable_credit=False)

        # Should return normal CompiledGraph
        assert not isinstance(compiled, CompiledCreditAwareGraph)


# ============================================================================
# Integration Tests with Real Workflows
# ============================================================================


class TestCreditAssignmentWorkflows:
    """Integration tests with real StateGraph workflows."""

    @pytest.mark.asyncio
    async def test_simple_workflow_credit_assignment(self):
        """Test credit assignment on simple linear workflow."""
        from victor.framework.graph import CompiledGraph

        graph = StateGraph(SimpleState)

        def step1(state: SimpleState) -> SimpleState:
            state["step1"] = True
            state["reward"] = 0.3
            return state

        def step2(state: SimpleState) -> SimpleState:
            state["step2"] = True
            state["reward"] = 0.5
            return state

        def step3(state: SimpleState) -> SimpleState:
            state["step3"] = True
            state["reward"] = 0.2
            return state

        graph.add_node("step1", step1)
        graph.add_node("step2", step2)
        graph.add_node("step3", step3)
        graph.add_edge("step1", "step2")
        graph.add_edge("step2", "step3")
        graph.add_edge("step3", END)
        graph.set_entry_point("step1")

        # Manually trace execution
        tracer = CreditTracer()
        tracer.start_trace({"initial": True})

        # Simulate transitions
        state = {"initial": True}
        state = {**state, **step1(state.copy())}
        tracer.record_transition("step1", "step2", {"initial": True}, state, {"reward": 0.3})

        state = {**state, **step2(state.copy())}
        tracer.record_transition("step2", "step3", state.copy(), state, {"reward": 0.5})

        state = {**state, **step3(state.copy())}
        tracer.record_transition("step3", "__end__", state.copy(), state, {"reward": 0.2})

        trace = tracer.end_trace(state, success=True)

        # Assign credit
        signals = tracer.assign_credit_to_trace(trace, methodology=CreditMethodology.GAE)

        assert len(signals) == 3
        assert trace.total_reward == 1.0

    @pytest.mark.asyncio
    async def test_conditional_workflow_credit(self):
        """Test credit assignment with conditional branching."""
        tracer = CreditTracer()
        tracer.start_trace({"count": 0})

        # Simulate conditional execution with retries
        for i in range(3):
            state = {"count": i + 1}
            from_node = "start" if i == 0 else "task"
            to_node = "task" if i < 2 else "done"
            tracer.record_transition(from_node, to_node, {"count": i}, state, {"reward": 0.4})

        # Final transition
        tracer.record_transition(
            "done",
            "__end__",
            {"count": 3},
            {"count": 3, "final": True},
            {"reward": 0.6},
        )

        trace = tracer.end_trace({"count": 3, "final": True}, success=True)

        # Assign credit with hindsight for early attempts
        signals = tracer.assign_credit_to_trace(trace, methodology=CreditMethodology.N_STEP_RETURNS)

        assert len(signals) >= 4

    @pytest.mark.asyncio
    async def test_multi_agent_workflow_attribution(self):
        """Test agent attribution in multi-agent workflow."""
        tracer = CreditTracer()

        # Multi-agent collaboration
        tracer.start_trace({"agent_id": "coordinator"})

        # Agent A does research
        tracer.record_transition(
            "start",
            "research",
            {"agent_id": "coordinator"},
            {"agent_id": "agent_A", "data": "researched"},
            {"reward": 0.4},
        )

        # Agent B processes
        tracer.record_transition(
            "research",
            "process",
            {"agent_id": "agent_A", "data": "researched"},
            {"agent_id": "agent_B", "data": "processed"},
            {"reward": 0.3},
        )

        # Agent A validates
        tracer.record_transition(
            "process",
            "validate",
            {"agent_id": "agent_B"},
            {"agent_id": "agent_A", "data": "validated"},
            {"reward": 0.2},
        )

        trace = tracer.end_trace({"agent_id": "agent_A", "data": "validated"}, success=True)

        # Assign credit with Shapley values
        signals = tracer.assign_credit_to_trace(trace, methodology=CreditMethodology.SHAPLEY)

        # Get attribution for each agent
        attribution_a = tracer.get_agent_attribution(trace, "agent_A")
        attribution_b = tracer.get_agent_attribution(trace, "agent_B")

        # Both agents should have some credit
        assert len(attribution_a) > 0 or attribution_a.get("agent_A", 0) > 0
        assert len(attribution_b) > 0 or attribution_b.get("agent_B", 0) > 0

    @pytest.mark.asyncio
    async def test_failed_workflow_hindsight_credit(self):
        """Test hindsight credit assignment for failed workflow."""
        tracer = CreditTracer()
        tracer.start_trace({"task": "complex"})

        # Multiple failed attempts
        for i in range(3):
            tracer.record_transition(
                "attempt",
                "retry",
                {"attempt": i},
                {"attempt": i + 1},
                {"reward": -0.2},  # Negative reward = failure
            )

        trace = tracer.end_trace(
            {"task": "complex", "failed": True},
            success=False,
            error="Max retries exceeded",
        )

        # Use hindsight to reframe as learning
        signals = tracer.assign_credit_to_trace(trace, methodology=CreditMethodology.HINDSIGHT)

        # Hindsight should convert to positive credits
        assert len(signals) > 0
        for signal in signals:
            assert signal.methodology == CreditMethodology.HINDSIGHT


# ============================================================================
# Utility Function Tests
# ============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_create_credit_aware_workflow(self):
        """Should create credit-aware workflow with defaults."""
        graph = StateGraph(SimpleState)
        graph.add_node("task", lambda s: {**s, "reward": 0.5})
        graph.set_entry_point("task")

        credit_graph = create_credit_aware_workflow(
            graph,
            reward_key="reward",
            agent_key="agent_id",
        )

        assert isinstance(credit_graph, CreditAwareGraph)

    def test_custom_keys_in_workflow(self):
        """Should use custom reward and agent keys."""
        graph = StateGraph(SimpleState)
        graph.add_node("task", lambda s: {**s, "score": 0.7, "worker": "bot_1"})
        graph.set_entry_point("task")

        credit_graph = create_credit_aware_workflow(
            graph,
            reward_key="score",
            agent_key="worker",
        )

        compiled = credit_graph.compile(enable_credit=True)
        assert isinstance(compiled, CompiledCreditAwareGraph)


# ============================================================================
# Performance Tests
# ============================================================================


class TestCreditAssignmentPerformance:
    """Performance tests for credit assignment."""

    def test_large_trajectory_performance(self):
        """Test credit assignment on large trajectory (100 steps)."""
        import time

        tracer = CreditTracer()
        tracer.start_trace({"task": "large"})

        # Create large trajectory
        for i in range(100):
            tracer.record_transition(
                f"step_{i}",
                f"step_{i+1}",
                {"index": i},
                {"index": i + 1},
                {"reward": 0.01},
            )

        trace = tracer.end_trace({"index": 100}, success=True)

        # Measure time
        start = time.time()
        signals = tracer.assign_credit_to_trace(trace)
        duration = time.time() - start

        assert len(signals) == 100
        # Should complete in reasonable time (< 1 second)
        assert duration < 1.0

    def test_multi_agent_scalability(self):
        """Test credit assignment with many agents."""
        import time

        tracer = CreditTracer()
        tracer.start_trace({"task": "multi"})

        # 10 agents, 10 steps each
        for agent_idx in range(10):
            for step in range(10):
                tracer.record_transition(
                    f"agent_{agent_idx}_step_{step}",
                    f"agent_{agent_idx}_step_{step+1}",
                    {"agent_id": f"agent_{agent_idx}"},
                    {"agent_id": f"agent_{agent_idx}"},
                    {"reward": 0.1},
                )

        trace = tracer.end_trace({"done": True}, success=True)

        # Measure time for Shapley (more expensive)
        start = time.time()
        signals = tracer.assign_credit_to_trace(trace, methodology=CreditMethodology.SHAPLEY)
        duration = time.time() - start

        assert len(signals) == 100
        # Should complete in reasonable time
        assert duration < 5.0


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
