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

"""Enhanced conversation management for victor-dataanalysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from victor.framework.extensions import (
    ConversationCoordinator,
    ConversationStats,
    TurnType,
)

logger = logging.getLogger(__name__)


@dataclass
class DataAnalysisContext:
    """Data analysis context."""

    datasets_loaded: List[str] = field(default_factory=list)
    analyses_performed: List[Dict[str, Any]] = field(default_factory=list)
    visualizations_created: List[str] = field(default_factory=list)
    insights_found: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "datasets_loaded": self.datasets_loaded,
            "analyses_performed": self.analyses_performed,
            "visualizations_created": self.visualizations_created,
            "insights_found": self.insights_found,
        }


class EnhancedDataAnalysisConversationManager:
    """Enhanced conversation manager for DataAnalysis."""

    def __init__(
        self,
        max_history_turns: int = 50,
        summarization_threshold: int = 40,
    ):
        self._conversation_coordinator = ConversationCoordinator(
            max_history_turns=max_history_turns,
            summarization_threshold=summarization_threshold,
        )
        self._context = DataAnalysisContext()

    def add_message(self, role: str, content: str, turn_type: TurnType, **kwargs) -> str:
        return self._conversation_coordinator.add_message(role, content, turn_type)

    def get_history(self, **kwargs) -> List[Dict[str, Any]]:
        return self._conversation_coordinator.get_history()

    def track_dataset(self, dataset: str) -> None:
        if dataset not in self._context.datasets_loaded:
            self._context.datasets_loaded.append(dataset)

    def track_analysis(self, analysis: str, result: str) -> None:
        self._context.analyses_performed.append({"analysis": analysis, "result": result})

    def track_insight(self, insight: str) -> None:
        self._context.insights_found.append(insight)

    def get_dataanalysis_summary(self) -> str:
        parts = []
        if self._context.datasets_loaded:
            parts.append("## Datasets Loaded")
            for ds in self._context.datasets_loaded:
                parts.append(f"- {ds}")
        if self._context.insights_found:
            parts.append("## Key Insights")
            for insight in self._context.insights_found:
                parts.append(f"- {insight}")
        return "\n".join(parts)

    def get_observability_data(self) -> Dict[str, Any]:
        obs = self._conversation_coordinator.get_observability_data()
        return {
            **obs,
            "dataanalysis_context": self._context.to_dict(),
            "vertical": "dataanalysis",
        }

    def get_stats(self) -> ConversationStats:
        return self._conversation_coordinator.get_stats()


__all__ = ["DataAnalysisContext", "EnhancedDataAnalysisConversationManager"]
