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

"""Browser/web-task benchmark runner for CLAW/GUIDE/VLAA-style manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from victor.evaluation.harness import BaseBenchmarkRunner
from victor.evaluation.protocol import (
    BenchmarkFailureCategory,
    BenchmarkMetadata,
    BenchmarkTask,
    BenchmarkType,
    EvaluationConfig,
    TaskResult,
    TaskStatus,
    get_benchmark_metadata,
    is_browser_task_benchmark,
)


class BrowserTaskBenchmarkRunner(BaseBenchmarkRunner):
    """Runner for browser/web-task manifests with action-trace evaluation."""

    def __init__(self, benchmark_type: BenchmarkType, dataset_path: Path):
        if not is_browser_task_benchmark(benchmark_type):
            raise ValueError(f"{benchmark_type.value} is not a browser-task benchmark")
        self._benchmark_type = benchmark_type
        self._dataset_path = dataset_path
        self._defaults: dict[str, Any] = {}
        self._task_specs: dict[str, dict[str, Any]] = {}
        self._tasks_cache: list[BenchmarkTask] | None = None
        self._records, self.manifest_metadata = self._load_manifest(dataset_path)

    @property
    def benchmark_type(self) -> BenchmarkType:
        return self._benchmark_type

    async def load_tasks(self, config: EvaluationConfig) -> list[BenchmarkTask]:
        """Load browser-task manifests into benchmark tasks."""
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
        """Evaluate browser-task outputs using actions and final-answer coverage."""
        del config  # Reserved for future screenshot/environment controls.

        result = TaskResult(
            task_id=task.task_id,
            status=TaskStatus.RUNNING,
            generated_code=agent_output,
        )

        spec = self._task_specs.get(task.task_id, {})
        required_actions = self._normalize_list(spec.get("required_actions"))
        forbidden_actions = self._normalize_list(spec.get("forbidden_actions"))
        expected_answer_contains = self._normalize_list(spec.get("expected_answer_contains"))

        parsed_output = self._parse_agent_output(agent_output)
        action_names = parsed_output["actions"]
        final_answer = parsed_output["final_answer"]
        matched_actions = [action for action in required_actions if action in action_names]
        missing_actions = [action for action in required_actions if action not in matched_actions]
        forbidden_hits = [action for action in forbidden_actions if action in action_names]
        matched_answer_phrases = self._match_text_requirements(
            expected_answer_contains, final_answer
        )
        missing_answer_phrases = [
            phrase for phrase in expected_answer_contains if phrase not in matched_answer_phrases
        ]

        action_coverage = self._coverage(matched_actions, required_actions)
        answer_coverage = self._coverage(matched_answer_phrases, expected_answer_contains)
        result.completion_score = round((action_coverage * 0.65) + (answer_coverage * 0.35), 4)
        result.failure_details = {
            "action_coverage": action_coverage,
            "answer_coverage": answer_coverage,
            "matched_actions": matched_actions,
            "missing_actions": missing_actions,
            "forbidden_action_hits": forbidden_hits,
            "matched_answer_phrases": matched_answer_phrases,
            "missing_answer_phrases": missing_answer_phrases,
        }

        if forbidden_hits:
            result.status = TaskStatus.FAILED
            result.error_message = "Output contains forbidden browser actions"
            result.failure_category = BenchmarkFailureCategory.TOOL_USAGE
            return result

        if action_coverage >= 1.0 and answer_coverage >= 1.0:
            result.status = TaskStatus.PASSED
            return result

        result.status = TaskStatus.FAILED
        result.error_message = "Output did not complete required browser actions or answer coverage"
        result.failure_category = BenchmarkFailureCategory.TASK_COMPLETION
        return result

    def _load_manifest(self, dataset_path: Path) -> tuple[list[dict[str, Any]], BenchmarkMetadata]:
        if not dataset_path.exists():
            raise FileNotFoundError(f"Benchmark dataset not found: {dataset_path}")

        if dataset_path.suffix == ".jsonl":
            records = []
            for line in dataset_path.read_text().splitlines():
                if line.strip():
                    records.append(json.loads(line))
            return records, self._build_manifest_metadata({}, records)

        payload = json.loads(dataset_path.read_text())
        if isinstance(payload, list):
            return payload, self._build_manifest_metadata({}, payload)

        if not isinstance(payload, dict):
            raise ValueError("Browser benchmark dataset payload must be a list or object manifest")

        records = payload.get("tasks", [])
        if not isinstance(records, list):
            raise ValueError("Browser benchmark manifest 'tasks' must be a list")

        defaults = payload.get("defaults", {}) or {}
        if not isinstance(defaults, dict):
            raise ValueError("Browser benchmark manifest 'defaults' must be an object")
        self._defaults = defaults

        metadata = payload.get("metadata", {}) or {}
        if not isinstance(metadata, dict):
            raise ValueError("Browser benchmark manifest 'metadata' must be an object")

        return records, self._build_manifest_metadata(metadata, records)

    def _build_manifest_metadata(
        self,
        metadata: dict[str, Any],
        records: list[dict[str, Any]],
    ) -> BenchmarkMetadata:
        catalog = get_benchmark_metadata(self._benchmark_type)
        if catalog is None:
            raise ValueError(f"Missing benchmark metadata for {self._benchmark_type.value}")

        return BenchmarkMetadata(
            name=str(metadata.get("name") or catalog.name),
            type=self._benchmark_type,
            version=str(metadata.get("version") or catalog.version),
            total_tasks=int(metadata.get("total_tasks") or len(records)),
            languages=self._normalize_list(metadata.get("languages")) or list(catalog.languages),
            categories=self._normalize_list(metadata.get("categories")) or list(catalog.categories),
            description=str(metadata.get("description") or catalog.description),
            source_name=str(metadata.get("source_name") or catalog.source_name),
            source_url=str(metadata.get("source_url") or catalog.source_url),
            paper_url=str(metadata.get("paper_url") or catalog.paper_url),
            aliases=tuple(self._normalize_list(metadata.get("aliases")) or list(catalog.aliases)),
            evaluation_mode=str(metadata.get("evaluation_mode") or catalog.evaluation_mode),
            runner_status="implemented",
        )

    def _record_to_task(self, record: dict[str, Any], index: int) -> BenchmarkTask:
        merged = dict(self._defaults)
        merged.update(record)

        prompt = str(merged.get("prompt") or merged.get("instruction") or "").strip()
        if not prompt:
            raise ValueError(f"Task #{index} is missing a prompt")

        task_id = str(
            merged.get("task_id") or merged.get("id") or f"{self._benchmark_type.value}-{index}"
        )
        self._task_specs[task_id] = {
            "required_actions": self._normalize_list(merged.get("required_actions")),
            "forbidden_actions": self._normalize_list(merged.get("forbidden_actions")),
            "expected_answer_contains": self._normalize_list(
                merged.get("expected_answer_contains")
            ),
        }

        return BenchmarkTask(
            task_id=task_id,
            benchmark=self._benchmark_type,
            description=str(merged.get("description") or prompt),
            language=str(merged.get("language") or "web"),
            prompt=prompt,
            difficulty=str(merged.get("difficulty") or "medium"),
            category=str(merged.get("category") or "browser_task"),
            tags=self._normalize_list(merged.get("tags")),
            timeout_seconds=int(merged.get("timeout_seconds") or 300),
            hints=self._normalize_list(merged.get("hints")),
        )

    @staticmethod
    def _parse_agent_output(agent_output: str) -> dict[str, Any]:
        stripped = agent_output.strip()
        if not stripped:
            return {"actions": [], "final_answer": ""}
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return {"actions": [], "final_answer": stripped}

        if not isinstance(payload, dict):
            return {"actions": [], "final_answer": stripped}

        raw_actions = payload.get("actions") or []
        actions: list[str] = []
        if isinstance(raw_actions, list):
            for item in raw_actions:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("action")
                    if name:
                        actions.append(str(name).strip().lower())
                elif item:
                    actions.append(str(item).strip().lower())

        final_answer = str(payload.get("final_answer") or payload.get("answer") or "").strip()
        return {
            "actions": actions,
            "final_answer": final_answer,
        }

    @staticmethod
    def _normalize_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            items = value
        else:
            items = [value]
        return [str(item).strip() for item in items if str(item).strip()]

    @staticmethod
    def _match_text_requirements(requirements: list[str], text: str) -> list[str]:
        text_lower = text.lower()
        return [item for item in requirements if item.lower() in text_lower]

    @staticmethod
    def _coverage(matched: list[str], required: list[str]) -> float:
        if not required:
            return 1.0
        return len(matched) / len(required)
