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

from victor_contracts import TurnType
from victor_contracts.verticals import VerticalConversationManager

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


class EnhancedRAGConversationManager(VerticalConversationManager):
    """Coordinator-backed conversation manager with the rag context."""

    def __init__(self, **kwargs) -> None:
        super().__init__(
            RAGContext,
            "rag_context",
            "rag",
            **kwargs,
        )

    def add_message(self, role: str, content: str, turn_type: TurnType, **kwargs) -> str:
        """Backward-compatible kwargs passthrough over the kernel signature."""
        return super().add_message(role, content, turn_type, **kwargs)

    def get_history(self, **kwargs) -> List[Dict[str, Any]]:
        return super().get_history(**kwargs)

    def get_rag_context(self) -> RAGContext:
        return self.get_context()

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


__all__ = ["RAGContext", "EnhancedRAGConversationManager"]
