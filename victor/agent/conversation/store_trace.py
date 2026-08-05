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

"""Trace processing and token estimation logic for ConversationStore."""

from __future__ import annotations
import logging
import re
import asyncio
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.agent.conversation.store import ConversationStore

from victor.agent.conversation.types import ConversationMessage, MessageRole
from victor.tools.core_tool_aliases import canonicalize_core_tool_name
from victor.core.async_utils import run_sync

logger = logging.getLogger(__name__)


class ConversationStoreTrace:
    """Manages token estimation, message trace lexical extraction, and overlap-based relevance scoring."""

    def __init__(self, store: ConversationStore):
        """Initialize trace manager.

        Args:
            store: Reference to parent ConversationStore
        """
        self.store = store

    def estimate_tokens(self, content: str) -> int:
        """Estimate token count from content."""
        from victor.processing.native.tokenizer import count_tokens_fast

        return count_tokens_fast(content)

    def build_trace_metadata(
        self,
        role: MessageRole,
        content: str,
        tool_name: Optional[str],
        tool_call_id: Optional[str],
        metadata: Optional[Dict[str, Any]],
        tool_calls: Optional[List],
    ) -> Dict[str, Any]:
        """Build persisted metadata for semantic and execution retrieval traces."""
        existing = dict(metadata or {})
        semantic_text = str(existing.get("memory_semantic_text") or content)
        trace_kind = (
            "execution"
            if self.is_execution_trace_message(role, tool_name, tool_call_id, tool_calls)
            else "semantic"
        )
        trace_metadata: Dict[str, Any] = {
            "memory_trace_kind": trace_kind,
            "memory_semantic_text": semantic_text,
        }
        if trace_kind == "execution":
            trace_metadata["memory_execution_text"] = str(
                existing.get("memory_execution_text")
                or self.build_execution_trace_text(
                    role=role,
                    content=content,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    tool_calls=tool_calls,
                )
            )
        return trace_metadata

    def is_execution_trace_message(
        self,
        role: MessageRole,
        tool_name: Optional[str],
        tool_call_id: Optional[str],
        tool_calls: Optional[List],
    ) -> bool:
        """Return whether a message should participate in execution-trace retrieval."""
        return (
            role in (MessageRole.TOOL, MessageRole.TOOL_CALL)
            or bool(tool_name)
            or bool(tool_call_id)
            or bool(tool_calls)
        )

    def build_execution_trace_text(
        self,
        role: MessageRole,
        content: str,
        tool_name: Optional[str],
        tool_call_id: Optional[str],
        tool_calls: Optional[List],
    ) -> str:
        """Create a compact lexical representation for execution-oriented recall."""
        parts: List[str] = []
        canonical_tool_name = canonicalize_core_tool_name(tool_name) if tool_name else None
        extracted_tool_name = self.extract_trace_attribute(content, "tool")
        effective_tool_name = canonical_tool_name or extracted_tool_name

        if effective_tool_name:
            parts.append(f"tool {effective_tool_name}")
        else:
            role_value = role.value if hasattr(role, "value") else str(role)
            parts.append(role_value.replace("_", " "))

        if tool_call_id:
            parts.append(tool_call_id)

        for attr_name in ("path", "query", "pattern", "symbol", "file", "name"):
            attr_value = self.extract_trace_attribute(content, attr_name)
            if attr_value:
                parts.append(f"{attr_name} {attr_value}")

        if tool_calls:
            parts.append(str(tool_calls))

        body_text = re.sub(r"<[^>]+>", " ", content)
        body_text = re.sub(r"\s+", " ", body_text).strip()
        if body_text:
            parts.append(body_text[:500])

        trace_text = " ".join(part for part in parts if part).strip()
        return trace_text or content[:500]

    @staticmethod
    def extract_trace_attribute(content: str, attribute_name: str) -> Optional[str]:
        """Extract an XML-style attribute value from stored tool-output markup."""
        match = re.search(rf'{attribute_name}="([^"]+)"', content)
        if not match:
            return None
        return match.group(1).strip() or None

    @staticmethod
    def trace_tokens(text: str) -> List[str]:
        """Tokenize trace text for lightweight overlap scoring."""
        return re.findall(r"[a-z0-9_]+", text.lower())

    def get_message_trace_kind(self, message: ConversationMessage) -> str:
        """Return the retrieval trace kind for a message."""
        metadata = getattr(message, "metadata", {}) or {}
        trace_kind = metadata.get("memory_trace_kind")
        if trace_kind in {"semantic", "execution"}:
            return trace_kind
        role = message.role if isinstance(message.role, MessageRole) else MessageRole(message.role)
        if self.is_execution_trace_message(
            role,
            getattr(message, "tool_name", None),
            getattr(message, "tool_call_id", None),
            getattr(message, "tool_calls", None),
        ):
            return "execution"
        return "semantic"

    def get_message_trace_text(
        self,
        message: ConversationMessage,
        trace_kind: Optional[str] = None,
    ) -> str:
        """Return the retrieval text for the requested trace kind."""
        metadata = getattr(message, "metadata", {}) or {}
        resolved_kind = trace_kind or self.get_message_trace_kind(message)
        if resolved_kind == "execution":
            existing = metadata.get("memory_execution_text")
            if existing:
                return str(existing)
            role = (
                message.role if isinstance(message.role, MessageRole) else MessageRole(message.role)
            )
            return self.build_execution_trace_text(
                role=role,
                content=message.content,
                tool_name=getattr(message, "tool_name", None),
                tool_call_id=getattr(message, "tool_call_id", None),
                tool_calls=getattr(message, "tool_calls", None),
            )
        return str(metadata.get("memory_semantic_text") or message.content)

    def score_execution_trace(
        self,
        query: str,
        trace_text: str,
    ) -> float:
        """Score execution traces by lexical overlap with the current request."""
        query_tokens = set(self.trace_tokens(query))
        trace_tokens = set(self.trace_tokens(trace_text))
        if not query_tokens or not trace_tokens:
            return 0.0

        overlap = query_tokens & trace_tokens
        if not overlap:
            return 0.0

        score = len(overlap) / len(query_tokens)
        if query.lower() in trace_text.lower():
            score += 0.15
        return min(score, 1.0)

    def get_dual_trace_relevant_messages(
        self,
        session_id: str,
        query: str,
        semantic_limit: int = 5,
        execution_limit: int = 3,
        min_similarity: float = 0.3,
    ) -> Dict[str, List[Tuple[ConversationMessage, float]]]:
        """Retrieve semantic and execution traces in separate relevance buckets."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "Cannot call get_dual_trace_relevant_messages from async context. "
                "Use await aget_dual_trace_relevant_messages(...) instead."
            )

        return run_sync(
            self.aget_dual_trace_relevant_messages(
                session_id=session_id,
                query=query,
                semantic_limit=semantic_limit,
                execution_limit=execution_limit,
                min_similarity=min_similarity,
            )
        )

    async def aget_dual_trace_relevant_messages(
        self,
        session_id: str,
        query: str,
        semantic_limit: int = 5,
        execution_limit: int = 3,
        min_similarity: float = 0.3,
    ) -> Dict[str, List[Tuple[ConversationMessage, float]]]:
        """Retrieve semantic and execution traces in separate relevance buckets asynchronously."""
        # Find session messages
        session = self.store.get_session(session_id)
        if not session or not session.messages:
            return {"semantic": [], "execution": []}

        # Check if ConversationEmbeddingStore is available
        embedding_store = getattr(self.store, "_embedding_store", None)
        has_embeddings = embedding_store is not None

        semantic_scored: List[Tuple[ConversationMessage, float]] = []
        execution_scored: List[Tuple[ConversationMessage, float]] = []

        # Scored lists
        for message in session.messages:
            trace_kind = self.get_message_trace_kind(message)
            if trace_kind == "execution":
                # Compute lexical score
                trace_text = self.get_message_trace_text(message, trace_kind="execution")
                score = self.score_execution_trace(query, trace_text)
                if score >= min_similarity:
                    execution_scored.append((message, score))
            elif trace_kind == "semantic" and not has_embeddings:
                # Fallback to simple lexical match for semantic trace if embeddings not available
                text = self.get_message_trace_text(message, trace_kind="semantic")
                score = self.score_execution_trace(query, text)
                if score >= min_similarity:
                    semantic_scored.append((message, score))

        # Sort and limit execution traces
        execution_scored.sort(key=lambda item: item[1], reverse=True)
        execution_results = execution_scored[:execution_limit]

        # Semantic embedding search if available
        if has_embeddings:
            semantic_results = []  # Initialize for exception handler
            try:
                hits = await embedding_store.search_similar(
                    query=query,
                    session_id=session_id,
                    limit=semantic_limit,
                    min_similarity=min_similarity,
                )
                # Build message lookup from session
                message_by_id = {msg.id: msg for msg in session.messages}
                # Convert hits (message_id, similarity) to (ConversationMessage, similarity)
                for hit in hits:
                    msg = message_by_id.get(hit.message_id)
                    if msg:
                        semantic_results.append((msg, hit.similarity))
            except Exception as e:
                logger.warning(f"Embedding search failed, using empty results: {e}")
                semantic_results = []
        else:
            semantic_scored.sort(key=lambda item: item[1], reverse=True)
            semantic_results = semantic_scored[:semantic_limit]

        return {
            "semantic": semantic_results,
            "execution": execution_results,
        }
