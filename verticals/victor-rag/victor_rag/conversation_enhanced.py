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

"""Enhanced conversation management for victor-rag."""

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
class RAGContext:
    """RAG context."""

    documents_indexed: List[str] = field(default_factory=list)
    queries_performed: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_results: List[Dict[str, Any]] = field(default_factory=list)
    index_updates: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "documents_indexed": self.documents_indexed,
            "queries_performed": self.queries_performed,
            "retrieval_results": self.retrieval_results,
            "index_updates": self.index_updates,
        }


class EnhancedRAGConversationManager:
    """Enhanced conversation manager for RAG."""

    def __init__(
        self,
        max_history_turns: int = 50,
        summarization_threshold: int = 40,
    ):
        self._conversation_coordinator = ConversationCoordinator(
            max_history_turns=max_history_turns,
            summarization_threshold=summarization_threshold,
        )
        self._context = RAGContext()

    def add_message(self, role: str, content: str, turn_type: TurnType, **kwargs) -> str:
        return self._conversation_coordinator.add_message(role, content, turn_type)

    def get_history(self, **kwargs) -> List[Dict[str, Any]]:
        return self._conversation_coordinator.get_history()

    def track_document(self, document: str) -> None:
        if document not in self._context.documents_indexed:
            self._context.documents_indexed.append(document)

    def track_query(self, query: str, results_count: int) -> None:
        self._context.queries_performed.append({"query": query, "results": results_count})

    def track_index_update(self, update_type: str, count: int) -> None:
        self._context.index_updates.append({"type": update_type, "count": count})

    def get_rag_summary(self) -> str:
        parts = []
        if self._context.documents_indexed:
            parts.append("## Documents Indexed")
            parts.append(f"Total: {len(self._context.documents_indexed)}")
        if self._context.queries_performed:
            parts.append("## Queries Performed")
            parts.append(f"Total: {len(self._context.queries_performed)}")
        return "\n".join(parts)

    def get_observability_data(self) -> Dict[str, Any]:
        obs = self._conversation_coordinator.get_observability_data()
        return {**obs, "rag_context": self._context.to_dict(), "vertical": "rag"}

    def get_stats(self) -> ConversationStats:
        return self._conversation_coordinator.get_stats()


__all__ = ["RAGContext", "EnhancedRAGConversationManager"]
