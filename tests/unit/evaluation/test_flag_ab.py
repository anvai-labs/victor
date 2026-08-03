# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Unit tests for the flag A/B battery generator (orchestration only — no live model)."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from victor.evaluation.agentic_harness import AgenticExecutionTrace, EvalToolCall
from victor.evaluation.flag_ab import FLAG_ENV, _flag_env_var, run_battery_arm, run_flag_ab

FLAG = "effect_gated_completion"


def _trace(task_id: str) -> AgenticExecutionTrace:
    return AgenticExecutionTrace(
        task_id=task_id,
        start_time=0.0,
        tool_calls=[EvalToolCall(name="write", arguments={"file_path": "a.py"}, success=True)],
        messages=[{"role": "assistant", "content": "done"}],
    )


class _FakeAdapter:
    """Records the flag env seen at execute time and returns a synthetic trace."""

    def __init__(self, env_var: str, captured: list) -> None:
        self._env_var = env_var
        self._captured = captured

    async def _run(self, bench) -> AgenticExecutionTrace:
        return _trace(bench.task_id)

    def execute_task(self, bench, workspace):
        self._captured.append(os.environ.get(self._env_var))
        return self._run(bench)


class _Runner:
    def run(self, coro):
        return asyncio.run(coro)


def _tasks(n: int):
    return [(None, SimpleNamespace(task_id=f"t{i}")) for i in range(n)]


def test_arm_toggles_flag_on_and_scores() -> None:
    env_var = _flag_env_var(FLAG)
    previous = os.environ.get(env_var)
    captured: list = []
    battery = run_battery_arm(
        FLAG,
        True,
        adapter_factory=lambda **kw: _FakeAdapter(env_var, captured),
        task_provider=lambda v: _tasks(5),
        runner=_Runner(),
    )
    assert battery.overall is not None
    assert battery.overall.n == 5
    assert captured == ["true"] * 5  # flag ON for every task in the arm
    assert os.environ.get(env_var) == previous  # env restored afterward


def test_arm_off_sets_false() -> None:
    env_var = _flag_env_var(FLAG)
    captured: list = []
    run_battery_arm(
        FLAG,
        False,
        adapter_factory=lambda **kw: _FakeAdapter(env_var, captured),
        task_provider=lambda v: _tasks(3),
        runner=_Runner(),
    )
    assert captured == ["false"] * 3


def test_flag_env_mapping() -> None:
    assert FLAG_ENV["effect_gated_completion"] == "VICTOR_EFFECT_GATED_COMPLETION"
    assert _flag_env_var("per_turn_auditor") == "VICTOR_PER_TURN_AUDITOR"
    assert _flag_env_var("some_new_flag") == "VICTOR_SOME_NEW_FLAG"


def test_run_flag_ab_writes_two_snapshots(tmp_path: Path) -> None:
    env_var = _flag_env_var(FLAG)
    captured: list = []
    result = run_flag_ab(
        FLAG,
        out_dir=str(tmp_path),
        adapter_factory=lambda **kw: _FakeAdapter(env_var, captured),
        task_provider=lambda v: _tasks(4),
        runner=_Runner(),
    )
    assert Path(result["baseline_path"]).exists()
    assert Path(result["candidate_path"]).exists()
    baseline = json.loads(Path(result["baseline_path"]).read_text(encoding="utf-8"))
    assert baseline["n"] == 4
    assert baseline["overall"]["n"] == 4
    # Baseline arm ran flag off, candidate arm ran it on.
    assert captured == ["false"] * 4 + ["true"] * 4
    # Graduation is included when the decider is importable, else None — both are valid.
    assert result["graduation"] is None or isinstance(result["graduation"], dict)


def test_run_flag_ab_emits_graduation_when_decider_available(tmp_path: Path) -> None:
    pytest.importorskip("victor.evaluation.flag_graduation")
    env_var = _flag_env_var(FLAG)
    result = run_flag_ab(
        FLAG,
        out_dir=str(tmp_path),
        adapter_factory=lambda **kw: _FakeAdapter(env_var, []),
        task_provider=lambda v: _tasks(6),
        runner=_Runner(),
    )
    assert result["graduation"] is not None
    assert result["graduation"]["flag"] == FLAG
    assert "verdict" in result["graduation"]
