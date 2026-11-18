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

"""Deterministic offline runtime intelligence for integration parity harnesses."""

from __future__ import annotations

from typing import Any

from victor.agent.perception_integration import PerceptionIntegration
from victor.agent.services.runtime_intelligence import RuntimeIntelligenceService
from victor.agent.task_analyzer import TaskAnalyzer
from victor.agent.task_complexity import TaskComplexityService
from victor.agent.unified_classifier import UnifiedTaskClassifier
from victor.classification import TaskType


def make_offline_task_analyzer(
    complexity_classifier: TaskComplexityService | None = None,
) -> TaskAnalyzer:
    """Build an analyzer that cannot load sentence-transformers or access the network."""
    deterministic_complexity = complexity_classifier or TaskComplexityService(
        use_semantic=False,
        use_rl=False,
    )
    analyzer = TaskAnalyzer()
    analyzer._complexity_classifier = deterministic_complexity
    analyzer._task_classifier = _DeterministicTaskClassifier()
    analyzer._unified_classifier = UnifiedTaskClassifier(
        task_analyzer=analyzer,
        enable_semantic=False,
    )
    return analyzer


def make_offline_runtime_intelligence() -> RuntimeIntelligenceService:
    """Build the real runtime boundary with deterministic perception and analysis."""
    # Runtime analysis and perception deliberately use separate analyzers. Sharing
    # one analyzer across the service and perception creates a runtime-intelligence
    # recursion when the unified classifier asks for learned routing context.
    service_analyzer = make_offline_task_analyzer()
    perception_analyzer = make_offline_task_analyzer()
    perception = PerceptionIntegration(task_analyzer=perception_analyzer)
    return RuntimeIntelligenceService(
        task_analyzer=service_analyzer,
        perception_integration=perception,
    )


class _DeterministicTaskClassifier:
    def classify_sync(self, _message: str) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(
            task_type=TaskType.GENERAL,
            confidence=1.0,
            top_matches=[],
            has_file_context=False,
        )
