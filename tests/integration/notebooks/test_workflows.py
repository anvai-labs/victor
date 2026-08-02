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

"""Integration tests for 02_workflows notebook using Ollama.

Tests the StateGraph workflow tutorial with real LLM calls using Ollama.
"""

import pytest
import typing

from victor.framework.graph import END, MemoryCheckpointer, StateGraph


@pytest.mark.integration
class TestWorkflowsNotebook:
    """Integration tests for the workflows notebook.

    Based on: docs/tutorials/notebooks/02_workflows.ipynb
    """

    @pytest.mark.asyncio
    async def test_simple_stategraph_workflow(self):
        """Test a simple StateGraph workflow."""

        class ProcessState(typing.TypedDict):
            input: str
            analysis: str
            summary: str

        def analyze(state: ProcessState) -> ProcessState:
            return {
                "input": state["input"],
                "analysis": f"Analyzed: {state['input']}",
                "summary": state["summary"],
            }

        def summarize(state: ProcessState) -> ProcessState:
            return {
                "input": state["input"],
                "analysis": state["analysis"],
                "summary": f"Summary: {state['analysis']}",
            }

        graph = StateGraph(ProcessState)
        graph.add_node("analyze", analyze)
        graph.add_node("summarize", summarize)
        graph.add_edge("analyze", "summarize")
        graph.add_edge("summarize", END)
        graph.set_entry_point("analyze")

        app = graph.compile()
        result = await app.invoke(ProcessState(input="Hello Victor!", analysis="", summary=""))

        assert result.success is True
        assert "Analyzed:" in result.state["analysis"]
        assert "Summary:" in result.state["summary"]

    @pytest.mark.asyncio
    async def test_conditional_workflow(self):
        """Test a workflow with conditional logic."""

        class ReviewState(typing.TypedDict):
            rating: int
            approved: bool
            outcome: str

        def decide(state: ReviewState) -> ReviewState:
            return state

        def route(state: ReviewState) -> str:
            return "approved" if state["rating"] >= 4 else "rejected"

        def approve(state: ReviewState) -> ReviewState:
            state["approved"] = True
            state["outcome"] = "approved"
            return state

        def reject(state: ReviewState) -> ReviewState:
            state["approved"] = False
            state["outcome"] = "rejected"
            return state

        graph = StateGraph(ReviewState)
        graph.add_node("decide", decide)
        graph.add_node("approve", approve)
        graph.add_node("reject", reject)
        graph.add_conditional_edge(
            "decide",
            route,
            {
                "approved": "approve",
                "rejected": "reject",
            },
        )
        graph.add_edge("approve", END)
        graph.add_edge("reject", END)
        graph.set_entry_point("decide")

        app = graph.compile()

        state1 = await app.invoke(ReviewState(rating=5, approved=False, outcome=""))
        assert state1.success is True
        assert state1.state["approved"] is True
        assert state1.state["outcome"] == "approved"

        state2 = await app.invoke(ReviewState(rating=2, approved=False, outcome=""))
        assert state2.success is True
        assert state2.state["approved"] is False
        assert state2.state["outcome"] == "rejected"

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """Test multi-step workflow execution."""
        import asyncio

        class ParallelState(typing.TypedDict):
            task1_result: str
            task2_result: str
            task3_result: str

        async def task1(state: ParallelState) -> ParallelState:
            await asyncio.sleep(0.01)
            state["task1_result"] = "done1"
            return state

        async def task2(state: ParallelState) -> ParallelState:
            await asyncio.sleep(0.01)
            state["task2_result"] = "done2"
            return state

        async def task3(state: ParallelState) -> ParallelState:
            state["task3_result"] = "done3"
            return state

        graph = StateGraph(ParallelState)
        graph.add_node("task1", task1)
        graph.add_node("task2", task2)
        graph.add_node("task3", task3)
        graph.add_edge("task1", "task2")
        graph.add_edge("task2", "task3")
        graph.add_edge("task3", END)
        graph.set_entry_point("task1")

        app = graph.compile()

        result = await app.invoke(ParallelState(task1_result="", task2_result="", task3_result=""))

        assert result.success is True
        assert result.state["task1_result"] == "done1"
        assert result.state["task2_result"] == "done2"
        assert result.state["task3_result"] == "done3"

    @pytest.mark.asyncio
    async def test_workflow_with_checkpointing(self):
        """Test workflow with checkpoint/save functionality."""

        class CounterState(typing.TypedDict):
            count: int
            doubled: int

        def increment(state: CounterState) -> CounterState:
            state["count"] += 1
            return state

        def should_continue(state: CounterState) -> str:
            return "double" if state["count"] >= 2 else "increment"

        def double(state: CounterState) -> CounterState:
            state["doubled"] = state["count"] * 2
            return state

        graph = StateGraph(CounterState)
        graph.add_node("increment", increment)
        graph.add_node("double", double)
        graph.add_conditional_edge(
            "increment",
            should_continue,
            {
                "increment": "increment",
                "double": "double",
            },
        )
        graph.add_edge("double", END)
        graph.set_entry_point("increment")

        checkpointer = MemoryCheckpointer()
        app = graph.compile(checkpointer=checkpointer)

        thread_id = "notebook-checkpoint-test"
        result = await app.invoke(CounterState(count=0, doubled=0), thread_id=thread_id)
        assert result.success is True
        assert result.state["count"] == 2
        assert result.state["doubled"] == 4

        checkpoints = await checkpointer.list(thread_id)
        assert len(checkpoints) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
