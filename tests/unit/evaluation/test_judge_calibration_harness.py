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

"""Tests for the offline judge-calibration harness and verifiable corpus (EVR-2, ADR-011)."""

import json
from pathlib import Path

import pytest

from victor.evaluation.calibration_corpus import default_corpus
from victor.evaluation.judge_calibration_harness import (
    JudgeCalibrationHarness,
    Transcript,
    VerifiableTask,
    alternating_scripted_executor,
    binary_categorize,
    make_scripted_executor,
)

# --- Corpus integrity ------------------------------------------------------------------------------


def test_corpus_task_ids_are_unique_and_families_covered() -> None:
    tasks = default_corpus(variants=3)
    ids = [t.task_id for t in tasks]
    assert len(ids) == len(set(ids)) == 18
    assert {t.family for t in tasks} == {
        "file-create",
        "code-fix",
        "refactor",
        "docs",
        "qa",
    }


def test_corpus_rejects_zero_variants() -> None:
    with pytest.raises(ValueError):
        default_corpus(variants=0)


def test_every_task_verifies_zero_when_unsolved(tmp_path: Path) -> None:
    executor = make_scripted_executor(lambda _t: False)
    for i, task in enumerate(default_corpus(variants=2)):
        workspace = tmp_path / f"unsolved_{i}"
        workspace.mkdir()
        task.setup(workspace)
        transcript = executor(task, workspace)
        assert task.verify(workspace, transcript) == 0.0, task.task_id


def test_every_task_verifies_one_when_solved(tmp_path: Path) -> None:
    executor = make_scripted_executor(lambda _t: True)
    for i, task in enumerate(default_corpus(variants=2)):
        workspace = tmp_path / f"solved_{i}"
        workspace.mkdir()
        task.setup(workspace)
        transcript = executor(task, workspace)
        assert task.verify(workspace, transcript) == 1.0, task.task_id


def test_every_task_has_a_flawed_solver_that_verifies_zero(tmp_path: Path) -> None:
    """Each flawed solver must make the workspace LOOK worked-on (state differs from setup)
    yet fail verification — that contrast is the whole discrimination test."""
    from victor.evaluation.judge_calibration_harness import make_outcome_executor

    def snapshot(ws: Path) -> dict:
        # Skip __pycache__ bytecode that code-fix verification generates on import.
        out = {}
        for p in ws.rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:
                try:
                    out[str(p.relative_to(ws))] = p.read_text()
                except UnicodeDecodeError:
                    pass
        return out

    flaw_executor = make_outcome_executor(lambda _t: "flaw")
    for i, task in enumerate(default_corpus(variants=2)):
        assert task.solve_flawed is not None or task.reference_answer_flawed, task.task_id
        workspace = tmp_path / f"flawed_{i}"
        workspace.mkdir()
        task.setup(workspace)
        pristine = snapshot(workspace)
        transcript = flaw_executor(task, workspace)
        # gold=0: the flawed attempt does not actually satisfy the task.
        assert task.verify(workspace, transcript) == 0.0, task.task_id
        # ...but it is not a no-op: either the workspace changed or a wrong answer was given.
        changed = snapshot(workspace) != pristine or task.reference_answer_flawed is not None
        assert changed, f"{task.task_id} flawed solve left no trace — not a discrimination case"


# --- Harness end-to-end ----------------------------------------------------------------------------


def _evidence_judge(_prompt: str, transcript: Transcript, _workspace: Path) -> float:
    return 1.0 if transcript.tool_steps() else 0.0


def _constant_judge(_prompt: str, _transcript: Transcript, _workspace: Path) -> float:
    return 1.0


def test_perfectly_agreeing_judge_is_trusted(tmp_path: Path) -> None:
    harness = JudgeCalibrationHarness(default_corpus(variants=4))
    report = harness.run(
        alternating_scripted_executor(period=5),
        _evidence_judge,
        workspace_root=tmp_path,
    )
    assert report.gate_decision.trusted
    assert report.overall.krippendorff_alpha == pytest.approx(1.0)
    assert len(report.samples) == 24
    # Both outcomes must actually occur, or agreement is vacuous.
    golds = {s.gold for s in report.samples}
    assert golds == {0.0, 1.0}


def test_constant_judge_fails_gate_on_mixed_gold(tmp_path: Path) -> None:
    harness = JudgeCalibrationHarness(default_corpus(variants=4))
    report = harness.run(
        alternating_scripted_executor(period=5),
        _constant_judge,
        workspace_root=tmp_path,
    )
    assert not report.gate_decision.trusted


def test_multi_judge_scores_all_on_one_executor_pass(tmp_path: Path) -> None:
    """run_multi_judge must run the executor once and score every judge on the SAME
    trajectories — the executor is called exactly once per task, not once per judge."""
    executor_calls: list[str] = []
    scripted = alternating_scripted_executor(period=5)  # one instance: its counter persists

    def counting_executor(task, workspace):
        executor_calls.append(task.task_id)
        return scripted(task, workspace)

    harness = JudgeCalibrationHarness(default_corpus(variants=2))
    reports = harness.run_multi_judge(
        counting_executor,
        {"evidence": _evidence_judge, "constant": _constant_judge},
        workspace_root=tmp_path,
    )
    assert set(reports) == {"evidence", "constant"}
    # 12 tasks, executor called once each — NOT 24 (once per judge).
    assert len(executor_calls) == 12
    # Both judges scored the same 12 trajectories; their gold columns are identical.
    ev, co = reports["evidence"], reports["constant"]
    assert [s.gold for s in ev.samples] == [s.gold for s in co.samples]
    # The evidence judge tracks tool activity → agrees; the constant judge does not.
    assert ev.overall.krippendorff_alpha == pytest.approx(1.0)
    assert not co.gate_decision.trusted


def test_hard_executor_discriminates_verify_from_activity_judges(
    tmp_path: Path,
) -> None:
    """On the HARD corpus, a judge that only checks for tool activity is fooled by flawed
    cases (activity present, gold=0) while a gold-aware oracle stays perfect — restoring the
    discrimination the easy corpus lost when strong judges saturate at α=1.0."""
    from victor.evaluation.judge_calibration_harness import hard_scripted_executor

    harness = JudgeCalibrationHarness(default_corpus(variants=8))

    def activity_judge(_p: str, transcript: Transcript, _w: Path) -> float:
        return 1.0 if transcript.tool_steps() else 0.0

    reports = harness.run_multi_judge(
        hard_scripted_executor(),
        {"activity": activity_judge},
        workspace_root=tmp_path,
    )
    samples = reports["activity"].samples
    golds = {s.gold for s in samples}
    assert golds == {0.0, 1.0}, "hard corpus must produce both gold classes"
    # Flawed cases: gold=0 but tool activity present → the activity judge scores them 1.0,
    # i.e. it is fooled. So its agreement must fall BELOW a perfect score (unlike on the
    # easy corpus, where activity == gold).
    fooled = [s for s in samples if s.gold == 0.0 and s.judged == 1.0]
    assert fooled, "hard corpus should fool an activity-only judge on at least one flawed case"
    assert reports["activity"].overall.krippendorff_alpha < 1.0


def test_run_delegates_to_multi_judge(tmp_path: Path) -> None:
    """The single-judge run() is a thin wrapper and stays behaviourally identical."""
    harness = JudgeCalibrationHarness(default_corpus(variants=2))
    report = harness.run(
        alternating_scripted_executor(period=5),
        _evidence_judge,
        workspace_root=tmp_path,
    )
    assert report.overall.krippendorff_alpha == pytest.approx(1.0)
    assert len(report.samples) == 12


def test_two_phase_matches_interleaved_and_reorders_calls(tmp_path: Path) -> None:
    """two_phase gives identical samples (deterministic judges) but calls the executor for
    ALL tasks before any judge — the property that avoids per-task model swapping."""
    order_i: list[str] = []
    scripted_i = alternating_scripted_executor(period=5)
    inter = JudgeCalibrationHarness(default_corpus(variants=2)).run_multi_judge(
        lambda t, w: (order_i.append(f"exec:{t.task_id}"), scripted_i(t, w))[1],
        {
            "j": lambda _p, tr, _w: (
                order_i.append("judge"),
                1.0 if tr.tool_steps() else 0.0,
            )[1]
        },
        workspace_root=tmp_path / "a",
        two_phase=False,
    )
    order_t: list[str] = []
    scripted_t = alternating_scripted_executor(period=5)
    two = JudgeCalibrationHarness(default_corpus(variants=2)).run_multi_judge(
        lambda t, w: (order_t.append(f"exec:{t.task_id}"), scripted_t(t, w))[1],
        {
            "j": lambda _p, tr, _w: (
                order_t.append("judge"),
                1.0 if tr.tool_steps() else 0.0,
            )[1]
        },
        workspace_root=tmp_path / "b",
        two_phase=True,
    )
    # Identical results regardless of scheduling.
    assert [s.judged for s in inter["j"].samples] == [s.judged for s in two["j"].samples]
    assert [s.gold for s in inter["j"].samples] == [s.gold for s in two["j"].samples]
    # Interleaved alternates exec/judge; two-phase runs all 12 execs before the first judge.
    first_judge = order_t.index("judge")
    assert first_judge == 12
    assert all(entry.startswith("exec:") for entry in order_t[:first_judge])
    assert order_i.index("judge") < 12  # interleaved judges early


def test_report_shape_and_json_roundtrip(tmp_path: Path) -> None:
    harness = JudgeCalibrationHarness(default_corpus(variants=2))
    report = harness.run(
        alternating_scripted_executor(period=5),
        _evidence_judge,
        workspace_root=tmp_path / "ws",
    )
    out = tmp_path / "reports" / "report.json"
    report.save(out)
    data = json.loads(out.read_text())
    assert data["n"] == 12
    assert set(data["per_family"]) == {
        "file-create",
        "code-fix",
        "refactor",
        "docs",
        "qa",
    }
    assert data["gate"]["trusted"] is True
    assert "completion verdicts only" in data["gate"]["scope"]
    assert all({"task_id", "family", "gold", "judged"} <= set(s) for s in data["samples"])


def test_judge_is_blinded_to_gold(tmp_path: Path) -> None:
    """The judge callable only ever receives (prompt, transcript, workspace)."""
    seen: list[tuple] = []

    def spy_judge(prompt: str, transcript: Transcript, workspace: Path) -> float:
        seen.append((prompt, transcript, workspace))
        return 0.0

    harness = JudgeCalibrationHarness(default_corpus(variants=1))
    harness.run(make_scripted_executor(lambda _t: True), spy_judge, workspace_root=tmp_path)
    assert len(seen) == 6
    for prompt, transcript, workspace in seen:
        assert isinstance(prompt, str)
        assert isinstance(transcript, Transcript)
        assert isinstance(workspace, Path) and workspace.is_dir()


def test_scripted_executor_requires_reference_solution(tmp_path: Path) -> None:
    task = VerifiableTask(
        task_id="no-solve",
        family="qa",
        prompt="unanswerable",
        setup=lambda ws: None,
        verify=lambda ws, t: 0.0,
        solve=None,
    )
    executor = make_scripted_executor(lambda _t: True)
    with pytest.raises(ValueError, match="no reference solution"):
        executor(task, tmp_path)


def test_binary_categorize_threshold() -> None:
    assert binary_categorize(0.5) is True
    assert binary_categorize(0.49) is False


def _isolated_tempdir(tmp_path: Path, monkeypatch) -> Path:
    """Redirect tempfile.mkdtemp() into an isolated dir so these assertions never race
    another process's judge_calibration_* dirs in the shared system temp."""
    import tempfile

    root = tmp_path / "tmproot"
    root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    return root


def test_auto_created_workspace_is_cleaned_up(tmp_path: Path, monkeypatch) -> None:
    """Regression: the harness orphaned one judge_calibration_* dir per run when no
    workspace_root was passed (47 leaked into /tmp before this fix)."""
    root = _isolated_tempdir(tmp_path, monkeypatch)
    harness = JudgeCalibrationHarness(default_corpus(variants=1))
    harness.run(make_scripted_executor(lambda _t: True), lambda p, t, w: 1.0)
    assert list(root.glob("judge_calibration_*")) == [], "temp workspace was not removed"


def test_caller_provided_workspace_is_not_deleted(tmp_path: Path) -> None:
    root = tmp_path / "mine"
    root.mkdir()
    harness = JudgeCalibrationHarness(default_corpus(variants=1))
    harness.run(
        make_scripted_executor(lambda _t: True),
        lambda p, t, w: 1.0,
        workspace_root=root,
    )
    assert root.is_dir(), "caller-owned workspace_root must survive the run"
    assert any(root.iterdir()), "caller-owned root should retain the per-task workspaces"


def test_keep_workspaces_retains_auto_created_dir(tmp_path: Path, monkeypatch) -> None:
    root = _isolated_tempdir(tmp_path, monkeypatch)
    harness = JudgeCalibrationHarness(default_corpus(variants=1))
    harness.run(
        make_scripted_executor(lambda _t: True),
        lambda p, t, w: 1.0,
        keep_workspaces=True,
    )
    kept = list(root.glob("judge_calibration_*"))
    assert len(kept) == 1, "keep_workspaces should retain the temp dir"
    assert any(kept[0].iterdir()), "kept workspace should contain the per-task dirs"
