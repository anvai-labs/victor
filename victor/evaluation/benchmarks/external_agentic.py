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

"""Local-manifest adapters for external agentic benchmarks."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from victor.evaluation.harness import BaseBenchmarkRunner, TaskEnvironment
from victor.evaluation.protocol import (
    BenchmarkFailureCategory,
    BenchmarkMetadata,
    BenchmarkTask,
    BenchmarkType,
    EvaluationConfig,
    TaskResult,
    TaskStatus,
    get_benchmark_metadata,
    is_external_agentic_benchmark,
)

logger = logging.getLogger(__name__)


class ExternalAgenticBenchmarkRunner(BaseBenchmarkRunner):
    """Runner for local JSON/JSONL external agentic benchmark manifests."""

    def __init__(self, benchmark_type: BenchmarkType, dataset_path: Path):
        if not is_external_agentic_benchmark(benchmark_type):
            raise ValueError(f"{benchmark_type.value} is not an external agentic benchmark")
        self._benchmark_type = benchmark_type
        self._dataset_path = dataset_path
        self._tasks_cache: list[BenchmarkTask] | None = None
        self._manifest_defaults: dict[str, Any] = {}
        self._records, self.manifest_metadata = self._load_manifest(dataset_path)

    @property
    def benchmark_type(self) -> BenchmarkType:
        return self._benchmark_type

    async def load_tasks(self, config: EvaluationConfig) -> list[BenchmarkTask]:
        """Load tasks from a local dataset manifest."""
        if self._tasks_cache is None:
            self._tasks_cache = [
                self._record_to_task(record, index)
                for index, record in enumerate(self._records, start=1)
            ]
        return self._filter_tasks(self._tasks_cache, config)

    async def run_task(
        self,
        task: BenchmarkTask,
        agent_output: str,
        config: EvaluationConfig,
    ) -> TaskResult:
        """Evaluate an external benchmark task against local tests or patches."""
        result = TaskResult(
            task_id=task.task_id,
            status=TaskStatus.RUNNING,
            generated_code=agent_output,
        )
        env = TaskEnvironment(
            task=task,
            workspace_dir=config.workspace_dir,
            use_docker=config.use_docker and bool(task.repo),
            docker_image=config.docker_image,
        )

        try:
            workspace = await env.setup()
            if self._looks_like_patch(agent_output):
                result.generated_patch = agent_output
                if not await env.apply_patch(agent_output):
                    result.status = TaskStatus.FAILED
                    result.error_message = "Failed to apply generated patch"
                    result.failure_category = BenchmarkFailureCategory.PATCH_APPLICATION
                    return result
            elif agent_output.strip():
                target_dir = workspace / "repo" if (workspace / "repo").exists() else workspace
                (target_dir / "solution.py").write_text(agent_output)
            else:
                result.status = TaskStatus.FAILED
                result.error_message = "No generated output"
                result.failure_category = BenchmarkFailureCategory.TASK_COMPLETION
                return result

            if not task.test_code:
                result.status = TaskStatus.PASSED
                return result

            if not task.repo and not (workspace / "pytest.ini").exists():
                (workspace / "pytest.ini").write_text("[pytest]\npython_files = test_*.py\n")

            passed, total, stdout, stderr = await env.run_tests(
                timeout=task.timeout_seconds or config.timeout_per_task
            )
            result.tests_passed = passed
            result.tests_total = total
            result.tests_failed = max(total - passed, 0)
            result.stdout = stdout
            result.stderr = stderr

            if stderr == "Timeout":
                result.status = TaskStatus.TIMEOUT
                result.error_message = "Test execution timed out"
                result.failure_category = BenchmarkFailureCategory.TIMEOUT
            elif total > 0 and passed == total:
                result.status = TaskStatus.PASSED
            elif total > 0:
                result.status = TaskStatus.FAILED
                result.error_message = (
                    f"Partial pass: {passed}/{total}" if passed else "All tests failed"
                )
                result.failure_category = BenchmarkFailureCategory.TEST_FAILURE
            elif stderr:
                result.status = TaskStatus.ERROR
                result.error_message = stderr[:500]
                result.failure_category = BenchmarkFailureCategory.ENVIRONMENT_ERROR
            else:
                result.status = TaskStatus.PASSED

            return result
        except Exception as exc:
            result.status = TaskStatus.ERROR
            result.error_message = str(exc)
            result.failure_category = BenchmarkFailureCategory.EXECUTION_ERROR
            return result
        finally:
            await env.cleanup()

    def _load_manifest(self, dataset_path: Path) -> tuple[list[dict[str, Any]], BenchmarkMetadata]:
        if not dataset_path.exists():
            raise FileNotFoundError(f"Benchmark dataset not found: {dataset_path}")

        if dataset_path.suffix == ".jsonl":
            records = []
            for line in dataset_path.read_text().splitlines():
                if not line.strip():
                    continue
                records.append(json.loads(line))
            return records, self._build_manifest_metadata({}, records)

        payload = json.loads(dataset_path.read_text())
        if isinstance(payload, list):
            return payload, self._build_manifest_metadata({}, payload)

        if not isinstance(payload, dict):
            raise ValueError("Benchmark dataset payload must be a list or object manifest")

        records = payload.get("tasks", [])
        if not isinstance(records, list):
            raise ValueError("Benchmark manifest 'tasks' must be a list")

        defaults = payload.get("defaults", {})
        if defaults is None:
            defaults = {}
        if not isinstance(defaults, dict):
            raise ValueError("Benchmark manifest 'defaults' must be an object")
        self._manifest_defaults = defaults

        metadata = payload.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError("Benchmark manifest 'metadata' must be an object")

        return records, self._build_manifest_metadata(metadata, records)

    def _build_manifest_metadata(
        self,
        metadata: dict[str, Any],
        records: list[dict[str, Any]],
    ) -> BenchmarkMetadata:
        catalog = get_benchmark_metadata(self._benchmark_type)
        if catalog is None:
            raise ValueError(f"Unknown benchmark metadata for {self._benchmark_type.value}")

        languages = self._as_list(metadata.get("languages")) or list(catalog.languages)
        categories = self._as_list(metadata.get("categories")) or list(catalog.categories)

        return BenchmarkMetadata(
            name=str(metadata.get("name") or catalog.name),
            type=self._benchmark_type,
            version=str(metadata.get("version") or catalog.version),
            total_tasks=int(metadata.get("total_tasks") or len(records)),
            languages=languages,
            categories=categories,
            description=str(metadata.get("description") or catalog.description),
            source_name=str(
                metadata.get("source_name") or metadata.get("source") or catalog.source_name
            ),
            source_url=str(metadata.get("source_url") or catalog.source_url),
            paper_url=str(metadata.get("paper_url") or catalog.paper_url),
            aliases=tuple(self._as_list(metadata.get("aliases")) or list(catalog.aliases)),
            evaluation_mode=str(metadata.get("evaluation_mode") or catalog.evaluation_mode),
            runner_status=catalog.runner_status,
        )

    def _record_to_task(self, record: dict[str, Any], index: int) -> BenchmarkTask:
        merged = dict(self._manifest_defaults)
        merged.update(record)

        description = str(
            merged.get("description")
            or merged.get("instruction")
            or merged.get("prompt")
            or merged.get("issue_text")
            or ""
        ).strip()
        prompt = str(merged.get("prompt") or merged.get("instruction") or description).strip()
        if not prompt:
            raise ValueError(f"Task #{index} is missing a prompt or description")

        return BenchmarkTask(
            task_id=str(
                merged.get("task_id") or merged.get("id") or f"{self._benchmark_type.value}-{index}"
            ),
            benchmark=self._benchmark_type,
            description=description or prompt,
            language=str(merged.get("language") or "python"),
            prompt=prompt,
            context_code=str(merged.get("context_code") or merged.get("context") or ""),
            test_code=str(merged.get("test_code") or merged.get("tests") or ""),
            seed_files=self._merge_seed_files(
                self._manifest_defaults.get("seed_files")
                or self._manifest_defaults.get("workspace_files")
                or self._manifest_defaults.get("files"),
                merged.get("seed_files") or merged.get("workspace_files") or merged.get("files"),
            ),
            repo=merged.get("repo"),
            base_commit=merged.get("base_commit"),
            issue_text=merged.get("issue_text"),
            hints=self._merge_list(self._manifest_defaults.get("hints"), merged.get("hints")),
            solution=merged.get("solution"),
            patch=merged.get("patch"),
            difficulty=str(merged.get("difficulty") or "medium"),
            category=str(merged.get("category") or "external_agentic"),
            tags=self._merge_list(self._manifest_defaults.get("tags"), merged.get("tags")),
            timeout_seconds=int(merged.get("timeout_seconds") or 300),
            complexity_override=merged.get("complexity_override"),
            task_type_hint=merged.get("task_type_hint"),
        )

    @staticmethod
    def _as_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value]
        return [str(value)]

    def _merge_list(self, default_value: Any, record_value: Any) -> list[str]:
        merged: list[str] = []
        for item in self._as_list(default_value) + self._as_list(record_value):
            if item not in merged:
                merged.append(item)
        return merged

    @staticmethod
    def _merge_seed_files(default_value: Any, record_value: Any) -> dict[str, str]:
        merged: dict[str, str] = {}
        for candidate in (default_value, record_value):
            if not candidate:
                continue
            if not isinstance(candidate, dict):
                raise ValueError("seed_files/files/workspace_files must be an object")
            for path, content in candidate.items():
                merged[str(path)] = str(content)
        return merged

    @staticmethod
    def _looks_like_patch(output: str) -> bool:
        stripped = output.lstrip()
        return stripped.startswith("diff --git") or stripped.startswith("--- ") or "@@" in output
