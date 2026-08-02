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

"""Deep-research benchmark runner for report-oriented evaluation tasks."""

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
)


class DeepResearchBenchmarkRunner(BaseBenchmarkRunner):
    """Runner for DR3-style deep-research report benchmarks."""

    def __init__(self, dataset_path: Path):
        self._dataset_path = dataset_path
        self._tasks_cache: list[BenchmarkTask] | None = None
        self._defaults: dict[str, Any] = {}
        self._task_specs: dict[str, dict[str, Any]] = {}
        self._records, self.manifest_metadata = self._load_manifest(dataset_path)

    @property
    def benchmark_type(self) -> BenchmarkType:
        return BenchmarkType.DR3_EVAL

    async def load_tasks(self, config: EvaluationConfig) -> list[BenchmarkTask]:
        """Load deep-research tasks from a local JSON or JSONL manifest."""
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
        """Evaluate a deep-research report against claims and citation requirements."""
        del config  # Reserved for future richer runtime controls.

        result = TaskResult(
            task_id=task.task_id,
            status=TaskStatus.RUNNING,
            generated_code=agent_output,
        )
        report = agent_output.strip()
        if not report:
            result.status = TaskStatus.FAILED
            result.error_message = "No generated report"
            result.failure_category = BenchmarkFailureCategory.TASK_COMPLETION
            result.completion_score = 0.0
            return result

        spec = self._task_specs.get(task.task_id, {})
        required_claims = self._normalize_list(spec.get("required_claims"))
        required_citations = self._normalize_list(spec.get("required_citations"))
        forbidden_claims = self._normalize_list(spec.get("forbidden_claims"))

        matched_claims = self._match_requirements(required_claims, report)
        missing_claims = [claim for claim in required_claims if claim not in matched_claims]
        matched_citations = self._match_requirements(required_citations, report)
        missing_citations = [
            citation for citation in required_citations if citation not in matched_citations
        ]
        forbidden_hits = self._match_requirements(forbidden_claims, report)

        claim_coverage = self._coverage(matched_claims, required_claims)
        citation_coverage = self._coverage(matched_citations, required_citations)
        result.completion_score = round((claim_coverage * 0.6) + (citation_coverage * 0.4), 4)
        result.failure_details = {
            "claim_coverage": claim_coverage,
            "citation_coverage": citation_coverage,
            "matched_claims": matched_claims,
            "missing_claims": missing_claims,
            "matched_citations": matched_citations,
            "missing_citations": missing_citations,
            "forbidden_claim_hits": forbidden_hits,
            "report_length_chars": len(report),
        }

        if forbidden_hits:
            result.status = TaskStatus.FAILED
            result.error_message = "Report contains unsupported claims"
            result.failure_category = BenchmarkFailureCategory.UNSUPPORTED_CLAIM
            return result

        if claim_coverage >= 1.0 and citation_coverage >= 1.0:
            result.status = TaskStatus.PASSED
            return result

        result.status = TaskStatus.FAILED
        result.error_message = "Report did not satisfy required claims or citations"
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
            raise ValueError(
                "Deep research benchmark dataset payload must be a list or object manifest"
            )

        records = payload.get("tasks", [])
        if not isinstance(records, list):
            raise ValueError("Deep research manifest 'tasks' must be a list")

        defaults = payload.get("defaults", {}) or {}
        if not isinstance(defaults, dict):
            raise ValueError("Deep research manifest 'defaults' must be an object")
        self._defaults = defaults

        metadata = payload.get("metadata", {}) or {}
        if not isinstance(metadata, dict):
            raise ValueError("Deep research manifest 'metadata' must be an object")

        return records, self._build_manifest_metadata(metadata, records)

    def _build_manifest_metadata(
        self,
        metadata: dict[str, Any],
        records: list[dict[str, Any]],
    ) -> BenchmarkMetadata:
        catalog = get_benchmark_metadata(BenchmarkType.DR3_EVAL)
        if catalog is None:
            raise ValueError("Missing benchmark metadata for dr3-eval")

        return BenchmarkMetadata(
            name=str(metadata.get("name") or catalog.name),
            type=BenchmarkType.DR3_EVAL,
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

        task_id = str(merged.get("task_id") or merged.get("id") or f"dr3-eval-{index}")
        self._task_specs[task_id] = {
            "required_claims": self._normalize_list(merged.get("required_claims")),
            "required_citations": self._normalize_list(merged.get("required_citations")),
            "forbidden_claims": self._normalize_list(merged.get("forbidden_claims")),
        }

        return BenchmarkTask(
            task_id=task_id,
            benchmark=BenchmarkType.DR3_EVAL,
            description=str(merged.get("description") or prompt),
            language=str(merged.get("language") or "report"),
            prompt=prompt,
            difficulty=str(merged.get("difficulty") or "medium"),
            category=str(merged.get("category") or "deep_research"),
            tags=self._normalize_list(merged.get("tags")),
            timeout_seconds=int(merged.get("timeout_seconds") or 300),
            hints=self._normalize_list(merged.get("hints")),
            solution=str(merged.get("solution") or ""),
        )

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
    def _match_requirements(requirements: list[str], report: str) -> list[str]:
        report_lower = report.lower()
        return [item for item in requirements if item.lower() in report_lower]

    @staticmethod
    def _coverage(matched: list[str], required: list[str]) -> float:
        if not required:
            return 1.0
        return len(matched) / len(required)
