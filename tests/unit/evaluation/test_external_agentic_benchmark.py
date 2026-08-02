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

"""Tests for external agentic benchmark adapters."""

import json

import pytest

from victor.evaluation.benchmarks.external_agentic import ExternalAgenticBenchmarkRunner
from victor.evaluation.protocol import (
    BenchmarkFailureCategory,
    BenchmarkType,
    EvaluationConfig,
    TaskStatus,
)


class TestExternalAgenticBenchmarkRunner:
    """Tests for local manifest external benchmark adapters."""

    def test_rejects_non_external_benchmark(self, tmp_path):
        """Only external perception benchmarks should use this runner."""
        dataset = tmp_path / "tasks.jsonl"
        dataset.write_text("")

        with pytest.raises(ValueError):
            ExternalAgenticBenchmarkRunner(BenchmarkType.SWE_BENCH, dataset)

    @pytest.mark.asyncio
    async def test_load_tasks_from_jsonl(self, tmp_path):
        """JSONL manifests should normalize into BenchmarkTask objects."""
        dataset = tmp_path / "tasks.jsonl"
        dataset.write_text(
            json.dumps(
                {
                    "id": "guide-1",
                    "instruction": "Implement add()",
                    "context": "def add(a, b):\n    return 0\n",
                    "tests": "from solution import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
                    "tags": ["gui", "agentic"],
                }
            )
            + "\n"
        )

        runner = ExternalAgenticBenchmarkRunner(BenchmarkType.GUIDE, dataset)
        tasks = await runner.load_tasks(
            EvaluationConfig(benchmark=BenchmarkType.GUIDE, model="test", max_tasks=1)
        )

        assert len(tasks) == 1
        assert tasks[0].task_id == "guide-1"
        assert tasks[0].benchmark == BenchmarkType.GUIDE
        assert tasks[0].prompt == "Implement add()"
        assert tasks[0].context_code.startswith("def add")

    @pytest.mark.asyncio
    async def test_loads_manifest_metadata_defaults_and_seed_files(self, tmp_path):
        """Object manifests should expose metadata, defaults, and seed files."""
        dataset = tmp_path / "tasks.json"
        dataset.write_text(
            json.dumps(
                {
                    "metadata": {
                        "version": "2026.04",
                        "source_name": "GUIDE Consortium",
                        "languages": ["python", "typescript"],
                    },
                    "defaults": {
                        "language": "python",
                        "tags": ["external", "agentic"],
                        "files": {"fixtures/shared.txt": "shared"},
                    },
                    "tasks": [
                        {
                            "task_id": "guide-3",
                            "prompt": "Implement helper",
                            "files": {"app/config.txt": "configured"},
                        }
                    ],
                }
            )
        )

        runner = ExternalAgenticBenchmarkRunner(BenchmarkType.GUIDE, dataset)
        tasks = await runner.load_tasks(
            EvaluationConfig(benchmark=BenchmarkType.GUIDE, model="test")
        )

        assert runner.manifest_metadata.version == "2026.04"
        assert runner.manifest_metadata.source_name == "GUIDE Consortium"
        assert runner.manifest_metadata.languages == ["python", "typescript"]
        assert tasks[0].language == "python"
        assert tasks[0].tags == ["external", "agentic"]
        assert tasks[0].seed_files == {
            "fixtures/shared.txt": "shared",
            "app/config.txt": "configured",
        }

    @pytest.mark.asyncio
    async def test_run_task_with_plain_code_and_tests(self, tmp_path):
        """Non-patch tasks should evaluate against local tests."""
        dataset = tmp_path / "tasks.json"
        dataset.write_text(
            json.dumps(
                [
                    {
                        "task_id": "guide-2",
                        "prompt": "Implement add()",
                        "context_code": "",
                        "test_code": "from solution import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
                    }
                ]
            )
        )
        runner = ExternalAgenticBenchmarkRunner(BenchmarkType.GUIDE, dataset)
        config = EvaluationConfig(
            benchmark=BenchmarkType.GUIDE,
            model="test",
            use_docker=False,
        )
        task = (await runner.load_tasks(config))[0]

        result = await runner.run_task(task, "def add(a, b):\n    return a + b\n", config)

        assert result.status == TaskStatus.PASSED
        assert result.failure_category is None
        assert result.tests_passed == 1
        assert result.tests_total == 1

    @pytest.mark.asyncio
    async def test_run_task_classifies_patch_failure(self, tmp_path):
        """Invalid patches should surface normalized patch failure taxonomy."""
        dataset = tmp_path / "tasks.json"
        dataset.write_text(
            json.dumps(
                [
                    {
                        "task_id": "claw-1",
                        "prompt": "Fix solution",
                        "context_code": "def add(a, b):\n    return 0\n",
                        "test_code": "from solution import add\n\n\ndef test_add():\n    assert add(1, 1) == 2\n",
                    }
                ]
            )
        )
        runner = ExternalAgenticBenchmarkRunner(BenchmarkType.CLAW_BENCH, dataset)
        config = EvaluationConfig(
            benchmark=BenchmarkType.CLAW_BENCH,
            model="test",
            use_docker=False,
        )
        task = (await runner.load_tasks(config))[0]

        result = await runner.run_task(task, "--- a/missing.py\n+++ b/missing.py\n", config)

        assert result.status == TaskStatus.FAILED
        assert result.failure_category == BenchmarkFailureCategory.PATCH_APPLICATION
