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


def test_run_flag_ab_isolates_and_restores_cwd(tmp_path: Path) -> None:
    env_var = _flag_env_var(FLAG)
    original = os.getcwd()
    seen_cwd: list = []

    def factory(**kw):
        seen_cwd.append(os.getcwd())
        return _FakeAdapter(env_var, [])

    run_flag_ab(
        FLAG,
        out_dir=str(tmp_path),
        adapter_factory=factory,
        task_provider=lambda v: _tasks(2),
        runner=_Runner(),
    )
    # The agent was constructed from an isolated cwd, not the caller's dir; cwd is restored after.
    assert seen_cwd and all(c != original for c in seen_cwd)
    assert os.getcwd() == original


def test_run_flag_ab_uses_given_workdir(tmp_path: Path) -> None:
    env_var = _flag_env_var(FLAG)
    workdir = tmp_path / "wd"
    seen_cwd: list = []

    def factory(**kw):
        seen_cwd.append(os.getcwd())
        return _FakeAdapter(env_var, [])

    run_flag_ab(
        FLAG,
        out_dir=str(tmp_path),
        workdir=str(workdir),
        adapter_factory=factory,
        task_provider=lambda v: _tasks(1),
        runner=_Runner(),
    )
    assert seen_cwd and all(Path(c).resolve() == workdir.resolve() for c in seen_cwd)


def test_drain_and_close_drains_then_closes() -> None:
    from victor.evaluation.flag_ab import _drain_and_close

    ran: list = []
    closed: list = []

    class _R:
        def run(self, coro):
            ran.append(True)
            return asyncio.run(coro)  # exercises _stop_background_services on a real loop

        def close(self):
            closed.append(True)

    _drain_and_close(_R())
    assert ran == [True]  # the drain coroutine ran
    assert closed == [True]  # the owned loop was closed


def test_injected_runner_is_not_closed() -> None:
    env_var = _flag_env_var(FLAG)
    closed: list = []

    class _R:
        def run(self, coro):
            return asyncio.run(coro)

        def close(self):
            closed.append(True)

    run_battery_arm(
        FLAG,
        True,
        adapter_factory=lambda **kw: _FakeAdapter(env_var, []),
        task_provider=lambda v: _tasks(2),
        runner=_R(),
    )
    assert closed == []  # an injected runner is the caller's to own — never drained/closed


def test_max_turns_threaded_to_adapter_factory(tmp_path: Path) -> None:
    env_var = _flag_env_var(FLAG)
    seen_max_turns: list = []

    def factory(*, base_url, model, max_turns):
        seen_max_turns.append(max_turns)
        return _FakeAdapter(env_var, [])

    run_flag_ab(
        FLAG,
        max_turns=5,
        out_dir=str(tmp_path),
        adapter_factory=factory,
        task_provider=lambda v: _tasks(1),
        runner=_Runner(),
    )
    assert seen_max_turns == [5, 5]  # both arms


class _FakeTask:
    """A verifiable task double: setup is a no-op; verify returns a preset pass/fail."""

    def __init__(self, task_id: str, passes: bool) -> None:
        self.task_id = task_id
        self._passes = passes

    def setup(self, ws) -> None:
        pass

    def verify(self, ws, transcript) -> float:
        return 1.0 if self._passes else 0.0


def test_verify_score_produces_task_pass_rate_battery() -> None:
    env_var = _flag_env_var(FLAG)
    tasks = [(_FakeTask(f"t{i}", i % 2 == 0), SimpleNamespace(task_id=f"t{i}")) for i in range(4)]
    battery = run_battery_arm(
        FLAG,
        True,
        score="verify",
        adapter_factory=lambda **kw: _FakeAdapter(env_var, []),
        task_provider=lambda v: tasks,
        runner=_Runner(),
    )
    assert battery.overall is not None
    assert battery.overall.n == 4
    assert abs(battery.overall.mean - 0.5) < 1e-9  # 2 of 4 tasks verified


def test_corpus_task_provider_selects_corpus() -> None:
    from victor.evaluation.flag_ab import CORPORA, _corpus_task_provider

    assert set(CORPORA) == {"calibration", "effect-gate"}
    effect = _corpus_task_provider("effect-gate")(1)
    assert len(effect) == 6
    assert "record-answer" in {verifiable.family for verifiable, _bench in effect}
    assert len(_corpus_task_provider("calibration")(1)) == 6


def test_default_factory_tool_budget_is_generous(monkeypatch) -> None:
    # Regression lock: a tight tool budget starves the effect gate (gate-ON → 0% pass observed).
    import victor.evaluation.agent_adapter as agent_adapter
    from victor.evaluation.flag_ab import _default_adapter_factory

    captured: dict = {}

    def _fake_from_profile(*, profile, base_url, model_override, config):
        captured["config"] = config
        return object()

    monkeypatch.setattr(
        agent_adapter.VictorAgentAdapter, "from_profile", staticmethod(_fake_from_profile)
    )
    _default_adapter_factory(base_url=None, model=None, max_turns=8)
    assert captured["config"].tool_budget >= 24  # decoupled + generous, not max(6, 8)=8
