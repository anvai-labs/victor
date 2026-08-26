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
    async def test_load_tasks_from_official_dr3_query_manifest(self, tmp_path):
        """Official query rows should expose every verified user file to the agent."""
        dataset_dir = tmp_path / "datasets_en"
        task_dir = dataset_dir / "001"
        task_dir.mkdir(parents=True)
        report = task_dir / "market report.pdf"
        video = task_dir / "overview.mp4"
        report.write_bytes(b"report")
        video.write_bytes(b"video")
        dataset = dataset_dir / "query.jsonl"
        query = "Synthesize the attached market evidence."
        dataset.write_text(
            json.dumps(
                {
                    "task": "001",
                    "query": query,
                    "user_files": [report.name, video.name],
                }
            )
            + "\n"
        )

        runner = DeepResearchBenchmarkRunner(dataset)
        task = (
            await runner.load_tasks(
                EvaluationConfig(benchmark=BenchmarkType.DR3_EVAL, model="test")
            )
        )[0]

        assert task.task_id == "001"
        assert task.description == query
        assert task.prompt.startswith(query)
        assert "User-provided files" in task.prompt
        assert str(report.resolve()) in task.prompt
        assert str(video.resolve()) in task.prompt

    @pytest.mark.asyncio
    @pytest.mark.parametrize("user_files", ["report.pdf", {}, [], [""]])
    async def test_official_dr3_user_files_fail_closed_on_invalid_shapes(
        self, tmp_path, user_files
    ):
        """Malformed or empty official file lists must not become fileless tasks."""
        dataset_dir = tmp_path / "datasets_en"
        (dataset_dir / "001").mkdir(parents=True)
        dataset = dataset_dir / "query.jsonl"
        dataset.write_text(
            json.dumps(
                {
                    "task": "001",
                    "query": "Research the supplied evidence.",
                    "user_files": user_files,
                }
            )
            + "\n"
        )

        runner = DeepResearchBenchmarkRunner(dataset)
        with pytest.raises(ValueError, match="user_files"):
            await runner.load_tasks(
                EvaluationConfig(benchmark=BenchmarkType.DR3_EVAL, model="test")
            )

    @pytest.mark.asyncio
    async def test_official_dr3_user_files_fail_closed_when_file_is_missing(self, tmp_path):
        """A manifest must not run when an official user file was not downloaded."""
        dataset_dir = tmp_path / "datasets_en"
        (dataset_dir / "001").mkdir(parents=True)
        dataset = dataset_dir / "query.jsonl"
        dataset.write_text(
            json.dumps(
                {
                    "task": "001",
                    "query": "Research the supplied evidence.",
                    "user_files": ["missing.pdf"],
                }
            )
            + "\n"
        )

        runner = DeepResearchBenchmarkRunner(dataset)
        with pytest.raises(ValueError, match="user file not found"):
            await runner.load_tasks(
                EvaluationConfig(benchmark=BenchmarkType.DR3_EVAL, model="test")
            )

    @pytest.mark.asyncio
    async def test_official_dr3_user_files_reject_task_directory_escape(self, tmp_path):
        """User-file paths may not escape the official task directory."""
        dataset_dir = tmp_path / "datasets_en"
        (dataset_dir / "001").mkdir(parents=True)
        (dataset_dir / "outside.pdf").write_bytes(b"outside")
        dataset = dataset_dir / "query.jsonl"
        dataset.write_text(
            json.dumps(
                {
                    "task": "001",
                    "query": "Research the supplied evidence.",
                    "user_files": ["../outside.pdf"],
                }
            )
            + "\n"
        )

        runner = DeepResearchBenchmarkRunner(dataset)
        with pytest.raises(ValueError, match="escapes task directory"):
            await runner.load_tasks(
                EvaluationConfig(benchmark=BenchmarkType.DR3_EVAL, model="test")
            )

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
