# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Unit tests for HTIR trace normalization (EVR-5, ADR-012)."""

from __future__ import annotations

from victor.evaluation.agentic_harness import AgenticExecutionTrace, EvalToolCall
from victor.evaluation.htir import (
    ArtifactEffect,
    ETCLOVGLayer,
    HTIRTrace,
    Role,
    StepStatus,
    normalize,
)


def _trace(**kwargs) -> AgenticExecutionTrace:
    kwargs.setdefault("start_time", 0.0)
    kwargs.setdefault("task_id", "t1")
    return AgenticExecutionTrace(**kwargs)


def test_write_tool_is_execution_workspace_delta() -> None:
    trace = _trace(
        tool_calls=[EvalToolCall(name="write", arguments={"file_path": "a.py"}, success=True)]
    )
    htir = normalize(trace)
    assert len(htir.steps) == 1
    step = htir.steps[0]
    assert step.role is Role.TOOL
    assert step.status is StepStatus.OK
    assert step.effect is ArtifactEffect.WORKSPACE_DELTA
    assert step.layer is ETCLOVGLayer.EXECUTION


def test_test_tool_is_verification_verified_check() -> None:
    trace = _trace(
        tool_calls=[EvalToolCall(name="test", arguments={"cmd": "pytest"}, success=True)]
    )
    step = normalize(trace).steps[0]
    assert step.effect is ArtifactEffect.VERIFIED_CHECK
    assert step.layer is ETCLOVGLayer.VERIFICATION


def test_read_tool_is_context_memory_grounded_claim() -> None:
    trace = _trace(tool_calls=[EvalToolCall(name="grep", arguments={"pattern": "x"}, success=True)])
    step = normalize(trace).steps[0]
    assert step.effect is ArtifactEffect.GROUNDED_CLAIM
    assert step.layer is ETCLOVGLayer.CONTEXT_MEMORY


def test_failed_write_is_execution_layer_failure_with_no_effect() -> None:
    trace = _trace(
        tool_calls=[EvalToolCall(name="edit", arguments={"path": "a.py"}, success=False)]
    )
    step = normalize(trace).steps[0]
    assert step.status is StepStatus.FAILED
    assert step.effect is ArtifactEffect.NONE
    assert step.layer is ETCLOVGLayer.EXECUTION
    assert step.is_failure


def test_final_assistant_refusal_is_governance_refused() -> None:
    trace = _trace(messages=[{"role": "assistant", "content": "I cannot help with that request."}])
    htir = normalize(trace)
    step = htir.steps[-1]
    assert step.role is Role.ASSISTANT
    assert step.status is StepStatus.REFUSED
    assert step.layer is ETCLOVGLayer.GOVERNANCE


def test_final_assistant_answer_is_orchestration_ok() -> None:
    trace = _trace(messages=[{"role": "assistant", "content": "Here is the answer: 42."}])
    step = normalize(trace).steps[-1]
    assert step.status is StepStatus.OK
    assert step.layer is ETCLOVGLayer.LIFECYCLE_ORCHESTRATION


def test_failures_by_layer_and_dominant() -> None:
    trace = _trace(
        tool_calls=[
            EvalToolCall(name="edit", arguments={"path": "a.py"}, success=False),
            EvalToolCall(name="shell", arguments={"command": "false"}, success=False),
            EvalToolCall(name="grep", arguments={"pattern": "x"}, success=False),
        ]
    )
    htir = normalize(trace)
    by_layer = htir.failures_by_layer()
    assert by_layer[ETCLOVGLayer.EXECUTION] == 2  # edit + shell
    assert by_layer[ETCLOVGLayer.CONTEXT_MEMORY] == 1  # grep
    assert htir.dominant_failure_layer() is ETCLOVGLayer.EXECUTION
    assert len(htir.failures) == 3


def test_empty_trace_normalizes_cleanly() -> None:
    htir = normalize(_trace())
    assert isinstance(htir, HTIRTrace)
    assert htir.steps == ()
    assert htir.failures_by_layer() == {}
    assert htir.dominant_failure_layer() is None


def test_to_dict_is_serializable() -> None:
    trace = _trace(
        session_id="s1",
        tool_calls=[EvalToolCall(name="write", arguments={"file_path": "a.py"}, success=True)],
    )
    payload = normalize(trace).to_dict()
    assert payload["task_id"] == "t1"
    assert payload["session_id"] == "s1"
    assert payload["n_steps"] == 1
    assert payload["steps"][0]["layer"] == "execution"
