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

"""Tests for the deep-research benchmark adapter."""

import json

import pytest

from victor.evaluation.benchmarks.deep_research import DeepResearchBenchmarkRunner
from victor.evaluation.protocol import (
    BenchmarkFailureCategory,
    BenchmarkType,
    EvaluationConfig,
    FailureStage,
    TaskStatus,
)


class TestDeepResearchBenchmarkRunner:
    """Tests for DR3-style deep-research benchmark evaluation."""

    @pytest.mark.asyncio
    async def test_load_tasks_from_manifest_defaults(self, tmp_path):
        """Manifest defaults and metadata should normalize into benchmark tasks."""
        dataset = tmp_path / "dr3_tasks.json"
        dataset.write_text(
            json.dumps(
                {
                    "metadata": {
                        "version": "2026.04",
                        "source_name": "DR3-Eval",
                    },
                    "defaults": {
                        "difficulty": "hard",
                        "category": "deep_research",
                    },
                    "tasks": [
                        {
                            "task_id": "dr3-1",
                            "prompt": "Summarize the vendor's AI roadmap.",
                            "language": "report",
                            "required_claims": [
                                "The roadmap prioritizes retrieval quality.",
                                "The roadmap includes evaluation automation.",
                            ],
                            "required_citations": ["[1]", "[2]"],
                        }
                    ],
                }
            )
        )

        runner = DeepResearchBenchmarkRunner(dataset)
        tasks = await runner.load_tasks(
            EvaluationConfig(benchmark=BenchmarkType.DR3_EVAL, model="test")
        )

        assert runner.manifest_metadata.version == "2026.04"
        assert runner.manifest_metadata.source_name == "DR3-Eval"
        assert tasks[0].task_id == "dr3-1"
        assert tasks[0].benchmark == BenchmarkType.DR3_EVAL
        assert tasks[0].difficulty == "hard"
        assert tasks[0].category == "deep_research"

    @pytest.mark.asyncio
    async def test_run_task_passes_when_claims_and_citations_are_covered(self, tmp_path):
        """Complete deep-research reports should pass with full completion score."""
        dataset = tmp_path / "dr3_tasks.json"
        dataset.write_text(
            json.dumps(
                [
                    {
                        "task_id": "dr3-pass",
                        "prompt": "Write a research synthesis.",
                        "required_claims": [
                            "Retrieval quality improved after reranking.",
                            "Benchmark automation reduced regression risk.",
                        ],
                        "required_citations": ["[1]", "[2]"],
                    }
                ]
            )
        )

        runner = DeepResearchBenchmarkRunner(dataset)
        task = (
            await runner.load_tasks(
                EvaluationConfig(benchmark=BenchmarkType.DR3_EVAL, model="test")
            )
        )[0]

        report = (
            "Findings\n"
            "Retrieval quality improved after reranking. [1]\n"
            "Benchmark automation reduced regression risk. [2]\n"
        )
        result = await runner.run_task(
            task,
            report,
            EvaluationConfig(benchmark=BenchmarkType.DR3_EVAL, model="test"),
        )

        assert result.status == TaskStatus.PASSED
        assert result.failure_category is None
        assert result.completion_score == pytest.approx(1.0)
        assert result.failure_details["claim_coverage"] == pytest.approx(1.0)
        assert result.failure_details["citation_coverage"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_run_task_classifies_unsupported_claims(self, tmp_path):
        """Unsupported claims should be surfaced with a normalized failure category."""
        dataset = tmp_path / "dr3_tasks.json"
        dataset.write_text(
            json.dumps(
                [
                    {
                        "task_id": "dr3-unsupported",
                        "prompt": "Write a research synthesis.",
                        "required_claims": ["Retrieval quality improved after reranking."],
                        "forbidden_claims": ["The vendor fully solved hallucinations."],
                    }
                ]
            )
        )

        runner = DeepResearchBenchmarkRunner(dataset)
        task = (
            await runner.load_tasks(
                EvaluationConfig(benchmark=BenchmarkType.DR3_EVAL, model="test")
            )
        )[0]

        report = (
            "Retrieval quality improved after reranking.\n"
            "The vendor fully solved hallucinations.\n"
        )
        result = await runner.run_task(
            task,
            report,
            EvaluationConfig(benchmark=BenchmarkType.DR3_EVAL, model="test"),
        )

        diagnosis = result.get_failure_diagnosis()

        assert result.status == TaskStatus.FAILED
        assert result.failure_category == BenchmarkFailureCategory.UNSUPPORTED_CLAIM
        assert result.failure_details["forbidden_claim_hits"] == [
            "The vendor fully solved hallucinations."
        ]
        assert diagnosis is not None
        assert diagnosis.stage == FailureStage.GROUNDING
        assert diagnosis.subtype == "forbidden_claim"

    @pytest.mark.asyncio
    async def test_run_task_tracks_partial_completion_for_missing_citations(self, tmp_path):
        """Reports missing citations should fail with partial completion scoring."""
        dataset = tmp_path / "dr3_tasks.json"
        dataset.write_text(
            json.dumps(
                [
                    {
                        "task_id": "dr3-partial",
                        "prompt": "Write a research synthesis.",
                        "required_claims": ["Retrieval quality improved after reranking."],
                        "required_citations": ["[1]", "[2]"],
                    }
                ]
            )
        )

        runner = DeepResearchBenchmarkRunner(dataset)
        task = (
            await runner.load_tasks(
                EvaluationConfig(benchmark=BenchmarkType.DR3_EVAL, model="test")
            )
        )[0]

        report = "Retrieval quality improved after reranking. [1]\n"
        result = await runner.run_task(
            task,
            report,
            EvaluationConfig(benchmark=BenchmarkType.DR3_EVAL, model="test"),
        )

        assert result.status == TaskStatus.FAILED
        assert result.failure_category == BenchmarkFailureCategory.TASK_COMPLETION
        assert 0.0 < result.completion_score < 1.0
        assert result.failure_details["missing_citations"] == ["[2]"]
