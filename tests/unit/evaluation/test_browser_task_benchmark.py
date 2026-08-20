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

"""Tests for browser/web-task benchmark adapters."""

import json

import pytest

from victor.evaluation.benchmarks.browser_tasks import BrowserTaskBenchmarkRunner
from victor.evaluation.protocol import (
    BenchmarkFailureCategory,
    BenchmarkType,
    EvaluationConfig,
    FailureStage,
    TaskStatus,
)


class TestBrowserTaskBenchmarkRunner:
    """Tests for CLAW/GUIDE/VLAA browser-task benchmark evaluation."""

    @pytest.mark.asyncio
    async def test_load_tasks_with_browser_defaults(self, tmp_path):
        """Browser manifests should load task metadata from defaults and records."""
        dataset = tmp_path / "clawbench.json"
        dataset.write_text(
            json.dumps(
                {
                    "metadata": {
                        "version": "2026.04",
                        "source_name": "ClawBench",
                    },
                    "defaults": {
                        "language": "web",
                        "difficulty": "hard",
                    },
                    "tasks": [
                        {
                            "task_id": "claw-1",
                            "prompt": "Find the settings page.",
                            "required_actions": ["open_url", "click"],
                            "expected_answer_contains": ["settings page"],
                        }
                    ],
                }
            )
        )

        runner = BrowserTaskBenchmarkRunner(BenchmarkType.CLAW_BENCH, dataset)
        tasks = await runner.load_tasks(
            EvaluationConfig(benchmark=BenchmarkType.CLAW_BENCH, model="test")
        )

        assert runner.manifest_metadata.version == "2026.04"
        assert runner.manifest_metadata.source_name == "ClawBench"
        assert tasks[0].task_id == "claw-1"
        assert tasks[0].benchmark == BenchmarkType.CLAW_BENCH
        assert tasks[0].language == "web"
        assert tasks[0].difficulty == "hard"

    @pytest.mark.asyncio
    async def test_run_task_passes_for_complete_action_trace(self, tmp_path):
        """Complete action traces and answers should pass."""
        dataset = tmp_path / "guide.json"
        dataset.write_text(
            json.dumps(
                [
                    {
                        "task_id": "guide-1",
                        "prompt": "Locate the settings page.",
                        "required_actions": ["open_url", "click"],
                        "expected_answer_contains": ["settings page"],
                    }
                ]
            )
        )

        runner = BrowserTaskBenchmarkRunner(BenchmarkType.GUIDE, dataset)
        task = (
            await runner.load_tasks(EvaluationConfig(benchmark=BenchmarkType.GUIDE, model="test"))
        )[0]

        output = json.dumps(
            {
                "actions": [
                    {"name": "open_url"},
                    {"name": "click"},
                ],
                "final_answer": "I found the settings page in the account menu.",
            }
        )
        result = await runner.run_task(
            task,
            output,
            EvaluationConfig(benchmark=BenchmarkType.GUIDE, model="test"),
        )

        assert result.status == TaskStatus.PASSED
        assert result.failure_category is None
        assert result.completion_score == pytest.approx(1.0)
        assert result.failure_details["action_coverage"] == pytest.approx(1.0)
        assert result.failure_details["answer_coverage"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_run_task_flags_forbidden_actions(self, tmp_path):
        """Forbidden browser actions should map to tool-usage failures."""
        dataset = tmp_path / "vlaa.json"
        dataset.write_text(
            json.dumps(
                [
                    {
                        "task_id": "vlaa-1",
                        "prompt": "Open the dashboard without deleting anything.",
                        "required_actions": ["open_url"],
                        "forbidden_actions": ["delete"],
                    }
                ]
            )
        )

        runner = BrowserTaskBenchmarkRunner(BenchmarkType.VLAA_GUI, dataset)
        task = (
            await runner.load_tasks(
                EvaluationConfig(benchmark=BenchmarkType.VLAA_GUI, model="test")
            )
        )[0]

        output = json.dumps(
            {
                "actions": [
                    {"name": "open_url"},
                    {"name": "delete"},
                ],
                "final_answer": "Dashboard opened.",
            }
        )
        result = await runner.run_task(
            task,
            output,
            EvaluationConfig(benchmark=BenchmarkType.VLAA_GUI, model="test"),
        )

        diagnosis = result.get_failure_diagnosis()

        assert result.status == TaskStatus.FAILED
        assert result.failure_category == BenchmarkFailureCategory.TOOL_USAGE
        assert result.failure_details["forbidden_action_hits"] == ["delete"]
        assert diagnosis is not None
        assert diagnosis.stage == FailureStage.ACTION
        assert diagnosis.subtype == "forbidden_action"

    @pytest.mark.asyncio
    async def test_run_task_scores_partial_completion_when_action_missing(self, tmp_path):
        """Missing required actions should produce partial completion scoring."""
        dataset = tmp_path / "claw.json"
        dataset.write_text(
            json.dumps(
                [
                    {
                        "task_id": "claw-partial",
                        "prompt": "Locate the settings page.",
                        "required_actions": ["open_url", "click"],
                        "expected_answer_contains": ["settings page"],
                    }
                ]
            )
        )

        runner = BrowserTaskBenchmarkRunner(BenchmarkType.CLAW_BENCH, dataset)
        task = (
            await runner.load_tasks(
                EvaluationConfig(benchmark=BenchmarkType.CLAW_BENCH, model="test")
            )
        )[0]

        output = json.dumps(
            {
                "actions": [{"name": "open_url"}],
                "final_answer": "I reached the account area.",
            }
        )
        result = await runner.run_task(
            task,
            output,
            EvaluationConfig(benchmark=BenchmarkType.CLAW_BENCH, model="test"),
        )

        assert result.status == TaskStatus.FAILED
        assert result.failure_category == BenchmarkFailureCategory.TASK_COMPLETION
        assert 0.0 < result.completion_score < 1.0
        assert result.failure_details["missing_actions"] == ["click"]
