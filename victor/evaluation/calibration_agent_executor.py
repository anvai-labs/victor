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

"""Real-agent executor for the judge-calibration harness (EVR-2, ADR-011).

Bridges a live Victor agent into the offline calibration loop: a
:class:`~victor.evaluation.judge_calibration_harness.VerifiableTask` becomes a
:class:`~victor.evaluation.protocol.BenchmarkTask`, execution runs through the existing
:class:`~victor.evaluation.agent_adapter.VictorAgentAdapter` (workspace-scoped tools, framework
completion detection), and the resulting
:class:`~victor.evaluation.agentic_harness.AgenticExecutionTrace` is projected down to the
blinded :class:`~victor.evaluation.judge_calibration_harness.Transcript` the judge sees.

With a local provider this keeps the whole EVR-2 validation offline::

    from victor.evaluation.agent_adapter import VictorAgentAdapter
    from victor.evaluation.calibration_agent_executor import make_agent_executor
    from victor.evaluation.calibration_corpus import default_corpus
    from victor.evaluation.judge_calibration_harness import JudgeCalibrationHarness

    adapter = VictorAgentAdapter.from_profile("default")  # e.g. an Ollama profile
    harness = JudgeCalibrationHarness(default_corpus(variants=8))
    report = harness.run(make_agent_executor(adapter), judge=my_rubric_judge)

The executor returned by :func:`make_agent_executor` is synchronous (it owns its event loop via
``asyncio.run``) — call it from sync code only. From an already-async context use
:func:`execute_verifiable_task` directly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from victor.evaluation.judge_calibration_harness import (
    CalibrationExecutor,
    PersistentLoopRunner,
    Transcript,
    TranscriptStep,
    VerifiableTask,
)
from victor.evaluation.protocol import BenchmarkTask, BenchmarkType

if TYPE_CHECKING:  # heavy import chain (orchestrator); only needed for typing
    from victor.evaluation.agent_adapter import VictorAgentAdapter
    from victor.evaluation.agentic_harness import AgenticExecutionTrace


def verifiable_to_benchmark_task(
    task: VerifiableTask, *, timeout_seconds: int = 300
) -> BenchmarkTask:
    """Project a calibration task onto the BenchmarkTask shape the agent adapter runs."""
    return BenchmarkTask(
        task_id=task.task_id,
        benchmark=BenchmarkType.CUSTOM,
        description=task.prompt,
        prompt=task.prompt,
        category=task.family,
        timeout_seconds=timeout_seconds,
    )


def trace_to_transcript(trace: "AgenticExecutionTrace") -> Transcript:
    """Project an execution trace down to the judge-visible transcript.

    Only observable behavior crosses this boundary: tool invocations (name + argument keys,
    not results) and assistant messages. Validation results, completion signals, and file-edit
    bookkeeping stay on the trace side — the judge must infer effect from the transcript and
    workspace, not read the harness's own instrumentation.
    """
    steps: list[TranscriptStep] = []
    for call in trace.tool_calls:
        rendered_args = ", ".join(f"{k}={call.arguments[k]!r}" for k in sorted(call.arguments))
        steps.append(TranscriptStep(kind="tool", content=f"{call.name}({rendered_args})"))
    assistant_contents = [
        str(message.get("content", ""))
        for message in trace.messages
        if message.get("role") == "assistant"
    ]
    steps.extend(TranscriptStep(kind="message", content=content) for content in assistant_contents)
    final = assistant_contents[-1] if assistant_contents else trace.generated_code
    return Transcript(steps=tuple(steps), final_message=final)


async def execute_verifiable_task(
    adapter: "VictorAgentAdapter",
    task: VerifiableTask,
    workspace: Path,
    *,
    timeout_seconds: int = 300,
) -> Transcript:
    """Run one calibration task through a real agent and return the blinded transcript.

    The adapter chdirs into the workspace for tool-path resolution; the original working
    directory is restored afterwards so harness iteration and callers are unaffected.
    """
    benchmark_task = verifiable_to_benchmark_task(task, timeout_seconds=timeout_seconds)
    original_cwd = Path.cwd()
    try:
        trace = await adapter.execute_task(benchmark_task, workspace)
    finally:
        os.chdir(original_cwd)
    return trace_to_transcript(trace)


def make_agent_executor(
    adapter: "VictorAgentAdapter", *, timeout_seconds: int = 300
) -> CalibrationExecutor:
    """Wrap a VictorAgentAdapter as a synchronous CalibrationExecutor.

    All tasks share ONE persistent event loop: the adapter's provider holds pooled HTTP
    connections bound to their creating loop, so per-task ``asyncio.run`` breaks every task
    after the first (``Event loop is closed``). Raises RuntimeError if invoked from a running
    loop — use :func:`execute_verifiable_task` there instead.
    """
    runner = PersistentLoopRunner()

    def executor(task: VerifiableTask, workspace: Path) -> Transcript:
        return runner.run(
            execute_verifiable_task(adapter, task, workspace, timeout_seconds=timeout_seconds)
        )

    return executor
