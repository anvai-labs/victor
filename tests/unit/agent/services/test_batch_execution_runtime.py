"""Batch execution runtime (FEP-0041 Phase 2)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from victor.agent.services.batch_execution_runtime import BatchExecutionRuntime
from victor.core.batch_plan import BatchPlan, BatchStep


class _MockPipeline:
    """A minimal ToolPipeline that records calls and returns canned results."""

    def __init__(self, outputs: dict[str, Any] | None = None, fail_on: set[str] | None = None):
        self.calls: list[tuple[str, dict]] = []
        self._outputs = outputs or {}
        self._fail_on = fail_on or set()

    async def _execute_single_call(self, tool_call: dict, context: dict) -> Any:
        name = tool_call["name"]
        self.calls.append((name, tool_call["arguments"]))
        if name in self._fail_on:
            return SimpleNamespace(success=False, error=f"{name} failed")
        return SimpleNamespace(success=True, output=self._outputs.get(name, f"output_{name}"))


class TestBatchExecutionRuntime:
    def test_sequential_execution(self):
        pipeline = _MockPipeline({"read": {"file_path": "a.py", "content": "code"}})
        runtime = BatchExecutionRuntime(pipeline)
        plan = BatchPlan(
            steps=[
                BatchStep(tool_name="read", inputs={"file_path": "a.py"}),
                BatchStep(tool_name="edit", input_refs={"file_path": "$step0.file_path"}),
            ]
        )
        result = asyncio.run(runtime.execute(plan))
        assert result.all_succeeded
        assert len(pipeline.calls) == 2
        assert pipeline.calls[0] == ("read", {"file_path": "a.py"})
        assert pipeline.calls[1] == ("edit", {"file_path": "a.py"})  # resolved from step0

    def test_stop_on_error(self):
        pipeline = _MockPipeline(fail_on={"edit"})
        runtime = BatchExecutionRuntime(pipeline)
        plan = BatchPlan(
            steps=[
                BatchStep(tool_name="read"),
                BatchStep(tool_name="edit"),
                BatchStep(tool_name="test"),  # should NOT run
            ],
            stop_on_error=True,
        )
        result = asyncio.run(runtime.execute(plan))
        assert not result.all_succeeded
        assert result.error == "edit: edit failed"
        assert len(pipeline.calls) == 2  # read + edit; test skipped
        assert pipeline.calls[1][0] == "edit"

    def test_continue_on_error(self):
        pipeline = _MockPipeline(fail_on={"edit"})
        runtime = BatchExecutionRuntime(pipeline)
        plan = BatchPlan(
            steps=[
                BatchStep(tool_name="read"),
                BatchStep(tool_name="edit"),
                BatchStep(tool_name="test"),
            ],
            stop_on_error=False,
        )
        result = asyncio.run(runtime.execute(plan))
        assert not result.all_succeeded
        assert len(pipeline.calls) == 3  # all three ran

    def test_post_failure_ref_resolution_passes_none(self):
        """When stop_on_error=False and a step fails, downstream refs get None.

        A failed step appends None to prior_outputs. A downstream step with
        $stepN.key where step N failed receives None as the input value (not
        a clear "dependency failed" error). This pins the current behavior:
        the tool gets a confusing type error, but execution continues.
        """
        pipeline = _MockPipeline(fail_on={"step1"})
        runtime = BatchExecutionRuntime(pipeline)
        plan = BatchPlan(
            steps=[
                BatchStep(tool_name="step0", inputs={"data": "initial"}),
                BatchStep(tool_name="step1"),  # fails
                BatchStep(
                    tool_name="step2",
                    input_refs={"input": "$step1.some_key"},  # refs failed step
                ),
            ],
            stop_on_error=False,
        )
        result = asyncio.run(runtime.execute(plan))
        assert not result.all_succeeded
        assert len(pipeline.calls) == 3  # all three attempted
        # step2 received None (from failed step1) as input
        assert pipeline.calls[2] == ("step2", {"input": None})

    def test_data_flow_chain(self):
        pipeline = _MockPipeline(
            {
                "search": {"file_path": "found.py"},
                "read": {"content": "file content"},
            }
        )
        runtime = BatchExecutionRuntime(pipeline)
        plan = BatchPlan(
            steps=[
                BatchStep(tool_name="search", inputs={"query": "test"}),
                BatchStep(tool_name="read", input_refs={"file_path": "$step0.file_path"}),
            ]
        )
        result = asyncio.run(runtime.execute(plan))
        assert result.all_succeeded
        # read got search's output's file_path
        assert pipeline.calls[1][1] == {"file_path": "found.py"}

    def test_scalar_whole_value_ref(self):
        pipeline = _MockPipeline({"transform": "transformed string"})
        runtime = BatchExecutionRuntime(pipeline)
        plan = BatchPlan(
            steps=[
                BatchStep(tool_name="transform", inputs={"input": "raw"}),
                BatchStep(tool_name="write", input_refs={"content": "$step0"}),
            ]
        )
        result = asyncio.run(runtime.execute(plan))
        assert result.all_succeeded
        # write got transform's whole output
        assert pipeline.calls[1][1] == {"content": "transformed string"}

    def test_empty_plan(self):
        pipeline = _MockPipeline()
        runtime = BatchExecutionRuntime(pipeline)
        result = asyncio.run(runtime.execute(BatchPlan()))
        assert not result.all_succeeded  # empty = not succeeded
        assert len(pipeline.calls) == 0

    def test_pipeline_without_single_call_raises(self):
        runtime = BatchExecutionRuntime(SimpleNamespace())  # no _execute_single_call
        plan = BatchPlan(steps=[BatchStep(tool_name="read")])
        result = asyncio.run(runtime.execute(plan))
        assert not result.all_succeeded
        assert "_execute_single_call" in result.error
