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

"""FOU-A2c unit suite: deterministic memory A/B battery (FEP-0048, ADR-036).

Covers the battery's evidence contract: baseline (memory OFF) vs treatment
(ON) over the same seeded fixtures, retrieval usefulness on seeded-memory
fixtures, prompt-inflation token cost, and the verdict mechanism
(KEEP_DEFAULT_ON / ESCALATE_PROMPT_INFLATION / HOLD_REVIEW) with the env
restored after each arm. Fully offline; tmp-path databases only.
"""

import json
import os
from pathlib import Path

import pytest

from victor.evaluation.memory_ab import (
    MEMORY_AB_ENV,
    MemoryABTask,
    assess_memory_ab,
    memory_ab_corpus,
    memory_ab_flag_env,
    run_memory_ab,
    run_memory_ab_task,
)


def test_flag_env_constant():
    assert MEMORY_AB_ENV == "VICTOR_MEMORY_ENABLED"
    assert memory_ab_flag_env() == "VICTOR_MEMORY_ENABLED"


def test_corpus_covers_evidence_cells():
    tasks = memory_ab_corpus()
    ids = [t.task_id for t in tasks]
    assert any("user_preference" in i for i in ids)
    assert any("project_preference" in i for i in ids)
    assert any("episode" in i for i in ids)
    assert any("cross_scope" in i for i in ids)
    assert any("empty_store" in i for i in ids)


async def test_baseline_arm_suppresses_all_injections(monkeypatch):
    monkeypatch.setenv(MEMORY_AB_ENV, "false")
    for task in memory_ab_corpus():
        report = await run_memory_ab_task(task)
        assert report["injected"] is False, task.task_id
        assert report["token_estimate"] == 0


async def test_treatment_arm_surfaces_seeded_facts(monkeypatch):
    monkeypatch.setenv(MEMORY_AB_ENV, "true")
    for task in memory_ab_corpus():
        report = await run_memory_ab_task(task)
        assert report["usefulness"] == 1.0, (task.task_id, report)
        if task.seed_content is not None and task.expect_surface:
            assert report["seed_surfaced"] is True, task.task_id


async def test_report_artifact_written(tmp_path: Path, monkeypatch):
    monkeypatch.delenv(MEMORY_AB_ENV, raising=False)
    result = run_memory_ab(out_dir=str(tmp_path))

    assert result["flag"] == "memory_enabled"
    off_path = Path(result["baseline_path"])
    on_path = Path(result["candidate_path"])
    assert off_path.exists() and on_path.exists()
    assert off_path.name == "memory_enabled_off.json"
    assert on_path.name == "memory_enabled_on.json"

    off_snapshot = json.loads(off_path.read_text(encoding="utf-8"))
    on_snapshot = json.loads(on_path.read_text(encoding="utf-8"))
    assert off_snapshot["enabled"] is False
    assert on_snapshot["enabled"] is True
    # Baseline never injects; treatment surfaces the seeds.
    assert off_snapshot["injections"] == 0
    assert off_snapshot["usefulness_mean"] < on_snapshot["usefulness_mean"]
    assert on_snapshot["usefulness_mean"] == pytest.approx(1.0)
    assert on_snapshot["false_surfaces"] == 0
    # Prompt-inflation cost is measured per arm.
    assert on_snapshot["prompt_tokens_max"] > 0
    assert off_snapshot["prompt_tokens_max"] == 0

    # The verdict rides in the report dict (flag_ab reporting convention).
    assert result["verdict"]["verdict"] == "KEEP_DEFAULT_ON"
    assert result["verdict"]["should_keep_default_on"] is True


def test_env_restored_after_battery(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(MEMORY_AB_ENV, "true")
    run_memory_ab(out_dir=str(tmp_path))
    assert os.environ[MEMORY_AB_ENV] == "true"


def test_verdict_keep_default_on():
    baseline = _arm(enabled=False, injections=0, usefulness=0.4, tokens=0.0)
    candidate = _arm(enabled=True, injections=3, usefulness=1.0, tokens=45.0)
    verdict = assess_memory_ab(baseline, candidate)
    assert verdict.verdict == "KEEP_DEFAULT_ON"
    assert verdict.should_keep_default_on is True


def test_verdict_escalates_on_prompt_inflation():
    baseline = _arm(enabled=False, injections=0, usefulness=0.4, tokens=0.0)
    candidate = _arm(enabled=True, injections=3, usefulness=1.0, tokens=900.0)
    verdict = assess_memory_ab(baseline, candidate, token_budget=500)
    assert verdict.verdict == "ESCALATE_PROMPT_INFLATION"
    assert verdict.should_keep_default_on is False
    assert any("inflation" in reason.lower() for reason in verdict.reasons)


def test_verdict_holds_on_uselessness():
    baseline = _arm(enabled=False, injections=0, usefulness=0.0, tokens=0.0)
    candidate = _arm(enabled=True, injections=0, usefulness=0.2, tokens=10.0)
    verdict = assess_memory_ab(baseline, candidate)
    assert verdict.verdict == "HOLD_REVIEW"
    assert verdict.should_keep_default_on is False


def test_verdict_holds_on_false_surfaces():
    baseline = _arm(enabled=False, injections=0, usefulness=0.4, tokens=0.0)
    candidate = _arm(enabled=True, injections=5, usefulness=1.0, tokens=30.0, false_surfaces=2)
    verdict = assess_memory_ab(baseline, candidate)
    assert verdict.verdict == "HOLD_REVIEW"


def test_custom_task_provider_is_used(tmp_path: Path):
    tasks = [
        MemoryABTask(
            task_id="custom_hit",
            query="custom probe",
            seed_scope=None,
            session_scope="project",
            expect_surface=False,
        )
    ]
    result = run_memory_ab(out_dir=str(tmp_path), task_provider=lambda variants: tasks)
    assert result["baseline_battery"]["tasks"][0]["task_id"] == "custom_hit"
    assert result["candidate_battery"]["tasks"][0]["task_id"] == "custom_hit"


def _arm(
    *,
    enabled: bool,
    injections: int,
    usefulness: float,
    tokens: float,
    false_surfaces: int = 0,
):
    from victor.evaluation.memory_ab import MemoryABArmResult

    return MemoryABArmResult(
        enabled=enabled,
        usefulness_scores=[usefulness] * max(1, injections) if injections else [usefulness],
        token_costs=[int(tokens)] * max(1, injections) if injections else [int(tokens)],
        false_surfaces=false_surfaces,
        injections=injections,
    )
