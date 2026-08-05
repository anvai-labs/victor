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

"""Tests for the real-agent calibration executor bridge (EVR-2, ADR-011)."""

import os
import time
from pathlib import Path

from victor.evaluation.agentic_harness import AgenticExecutionTrace, EvalToolCall
from victor.evaluation.calibration_agent_executor import (
    make_agent_executor,
    trace_to_transcript,
    verifiable_to_benchmark_task,
)
from victor.evaluation.calibration_corpus import default_corpus
from victor.evaluation.judge_calibration_harness import (
    JudgeCalibrationHarness,
    Transcript,
)
from victor.evaluation.protocol import BenchmarkType


def _make_trace(task_id: str = "t1") -> AgenticExecutionTrace:
    trace = AgenticExecutionTrace(task_id=task_id, start_time=time.time())
    trace.tool_calls.append(
        EvalToolCall(name="write_file", arguments={"path": "settings_0.toml", "content": "x"})
    )
    trace.messages.append({"role": "user", "content": "do the task"})
    trace.messages.append({"role": "assistant", "content": "Reading the workspace."})
    trace.messages.append({"role": "assistant", "content": "Done — file created."})
    trace.validations["hidden"] = True  # must NOT leak into the transcript
    return trace


class _StubAdapter:
    """Mimics VictorAgentAdapter.execute_task: solves file-create tasks, chdirs like the real one."""

    async def execute_task(self, benchmark_task, workspace_dir: Path) -> AgenticExecutionTrace:
        os.chdir(workspace_dir)  # the real adapter does this; the bridge must undo it
        trace = AgenticExecutionTrace(task_id=benchmark_task.task_id, start_time=time.time())
        if benchmark_task.category == "file-create":
            # Solve for real so the verifier's gold label is 1.0.
            for line in benchmark_task.prompt.split("`"):
                if line.startswith("port = "):
                    filename = benchmark_task.prompt.split("named ")[1].split(" ")[0]
                    (workspace_dir / filename).write_text(line + "\n")
            trace.tool_calls.append(EvalToolCall(name="write_file", arguments={}))
        trace.messages.append({"role": "assistant", "content": "Done."})
        return trace


# --- Mapping ---------------------------------------------------------------------------------------


def test_verifiable_to_benchmark_task_mapping() -> None:
    task = default_corpus(variants=1)[0]
    benchmark_task = verifiable_to_benchmark_task(task, timeout_seconds=120)
    assert benchmark_task.task_id == task.task_id
    assert benchmark_task.benchmark is BenchmarkType.CUSTOM
    assert benchmark_task.prompt == task.prompt
    assert benchmark_task.category == task.family
    assert benchmark_task.timeout_seconds == 120


def test_trace_to_transcript_projection() -> None:
    transcript = trace_to_transcript(_make_trace())
    assert isinstance(transcript, Transcript)
    assert len(transcript.tool_steps()) == 1
    assert "write_file" in transcript.tool_steps()[0].content
    assert transcript.final_message == "Done — file created."
    # Instrumentation must not leak into the judge-visible view.
    combined = transcript.final_message + "".join(s.content for s in transcript.steps)
    assert "hidden" not in combined
    assert "validations" not in combined


def test_trace_to_transcript_empty_trace_falls_back_to_generated_code() -> None:
    trace = AgenticExecutionTrace(task_id="empty", start_time=time.time())
    trace.generated_code = "print('x')"
    transcript = trace_to_transcript(trace)
    assert transcript.final_message == "print('x')"
    assert transcript.steps == ()


# --- Executor bridge -------------------------------------------------------------------------------


def test_agent_executor_runs_task_and_restores_cwd(tmp_path: Path) -> None:
    executor = make_agent_executor(_StubAdapter())
    task = next(t for t in default_corpus(variants=1) if t.family == "file-create")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    task.setup(workspace)

    cwd_before = Path.cwd()
    transcript = executor(task, workspace)

    assert Path.cwd() == cwd_before
    assert transcript.final_message == "Done."
    assert task.verify(workspace, transcript) == 1.0


def test_agent_executor_reuses_one_event_loop(tmp_path: Path) -> None:
    """Regression: per-task asyncio.run breaks adapters whose providers hold pooled HTTP
    connections ('Event loop is closed' after the first task)."""
    import asyncio

    loops: list[int] = []

    class _LoopSpyAdapter(_StubAdapter):
        async def execute_task(self, benchmark_task, workspace_dir: Path):
            loops.append(id(asyncio.get_running_loop()))
            return await super().execute_task(benchmark_task, workspace_dir)

    executor = make_agent_executor(_LoopSpyAdapter())
    for i, task in enumerate(default_corpus(variants=1)[:3]):
        workspace = tmp_path / f"loop_ws_{i}"
        workspace.mkdir()
        task.setup(workspace)
        executor(task, workspace)
    assert len(loops) == 3
    assert len(set(loops)) == 1, "each task ran on a different event loop"


def test_agent_executor_composes_with_harness(tmp_path: Path) -> None:
    tasks = [t for t in default_corpus(variants=2) if t.family in {"file-create", "docs"}]
    harness = JudgeCalibrationHarness(tasks)

    def evidence_judge(_p: str, transcript: Transcript, _w: Path) -> float:
        return 1.0 if transcript.tool_steps() else 0.0

    report = harness.run(
        make_agent_executor(_StubAdapter()), evidence_judge, workspace_root=tmp_path
    )
    # Stub solves file-create (gold 1) and fails docs (gold 0); the evidence judge
    # tracks tool activity, which matches — perfect agreement.
    assert {s.gold for s in report.samples} == {0.0, 1.0}
    assert report.gate_decision.trusted
    assert Path.cwd().is_dir()  # cwd restored to something sane after every task
