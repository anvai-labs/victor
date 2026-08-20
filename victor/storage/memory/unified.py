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

"""Unified Memory Coordinator for federated search across memory systems.

Provides a single interface for querying multiple memory backends:
- Entity memory (code entities, relationships)
- Conversation memory (message history)
- Embedding memory (semantic vectors)
- Graph memory (entity relationships)
- RL experience (tool selection patterns)

Design Patterns:
- Strategy Pattern: Pluggable ranking strategies
- Adapter Pattern: Memory providers wrap existing systems
- Repository Pattern: Unified query interface

Example:
    coordinator = UnifiedMemoryCoordinator()
    coordinator.register_provider(ConversationMemoryAdapter(store))
    coordinator.register_provider(EntityMemoryAdapter(entity_memory))

    # Federated search across all providers
    results = await coordinator.search_all(
        query="authentication",
        limit=20,
    )

    # Filter by memory type
    results = await coordinator.search_all(
        query="login function",
        memory_types=[MemoryType.ENTITY, MemoryType.EMBEDDING],
    )
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
import uuid
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    runtime_checkable,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class MemoryType(Enum):
    """Types of memory systems for federated queries.

    Each type represents a distinct memory backend with its own
    storage characteristics and query patterns.
    """

    ENTITY = auto()
    """Entity memory - code entities, symbols, definitions."""

    CONVERSATION = auto()
    """Conversation memory - message history, context."""

    EMBEDDING = auto()
    """Embedding memory - semantic vector storage."""

    GRAPH = auto()
    """Graph memory - entity relationships, traversal."""

    RL_EXPERIENCE = auto()
    """RL experience - tool selection patterns, outcomes."""

    CODE = auto()
    """Code memory - file contents, chunks."""


# =============================================================================
# Data Types
# =============================================================================


@dataclass(frozen=True)
class MemoryProvenance:
    """Immutable record of where a MemoryResult came from.

    Stamped by adapters at search time so dedup/merge logic can trace
    conflicts back to the originating store without inspecting metadata.
    """

    store_class: str
    query_id: str
    lane: "MemoryType"
    store_id: Optional[str] = None
    adapter_version: str = "1"


@dataclass
class MemoryResult:
    """Unified result from any memory system.

    Provides a common format for results regardless of source,
    enabling consistent ranking and deduplication.

    Attributes:
        source: Which memory system produced this result
        content: The actual content (text, entity, etc.)
        relevance: Relevance score 0.0-1.0 (higher is more relevant)
        id: Unique identifier for deduplication
        metadata: Additional source-specific information
        timestamp: When the content was stored (for recency ranking)
    """

    source: MemoryType
    content: Any
    relevance: float
    id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[float] = None
    provenance: Optional[MemoryProvenance] = None

    def __post_init__(self):
        """Generate ID if not provided."""
        if not self.id:
            content_str = str(self.content)[:100]
            self.id = hashlib.md5(f"{self.source.name}:{content_str}".encode()).hexdigest()[:16]

    def __repr__(self) -> str:
        return (
            f"MemoryResult(source={self.source.name}, "
            f"relevance={self.relevance:.3f}, id={self.id[:8]}...)"
        )


@dataclass
class MemoryQuery:
    """Query specification for memory search.

    Encapsulates all query parameters for consistent handling
    across memory providers.

    Attributes:
        query: Search query text
        limit: Maximum results to return
        memory_types: Filter to specific memory types
        filters: Additional provider-specific filters
        min_relevance: Minimum relevance threshold
        session_id: Optional session context
        include_metadata: Whether to include full metadata
    """

    query: str
    limit: int = 20
    memory_types: Optional[List[MemoryType]] = None
    filters: Optional[Dict[str, Any]] = None
    min_relevance: float = 0.0
    session_id: Optional[str] = None
    include_metadata: bool = True
    query_id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class MemoryEvolutionTrace:
    """Stored trace from a successful memory-backed interaction."""

    query: str
    result_ids: List[str]
    source_types: List[str]
    timestamp: float
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryTransferHint:
    """Reusable hint derived from prior successful traces."""

    matched_query: str
    score: float
    preferred_result_ids: List[str] = field(default_factory=list)
    preferred_source_types: List[str] = field(default_factory=list)
    session_id: Optional[str] = None
    cross_session: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProactiveMemoryHint:
    """Next-turn hint derived from successful memory transfer patterns."""

    hint: str
    score: float
    matched_query: str
    preferred_result_ids: List[str] = field(default_factory=list)
    preferred_source_types: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class MemoryProviderProtocol(Protocol):
    """Protocol for memory system adapters.

    Each memory system should implement this protocol to integrate
    with the unified coordinator. The protocol is intentionally
    minimal to allow diverse backend implementations.

    Example:
        class ConversationMemoryAdapter(MemoryProviderProtocol):
            @property
            def memory_type(self) -> MemoryType:
                return MemoryType.CONVERSATION

            async def search(self, query: MemoryQuery) -> List[MemoryResult]:
                # Query conversation store
                messages = await self._store.search(query.query)
                return [self._to_result(m) for m in messages]
    """

    @property
    @abstractmethod
    def memory_type(self) -> MemoryType:
        """The type of memory this provider handles.

        Returns:
            MemoryType enum value
        """
        ...

    @abstractmethod
    async def search(
        self,
        query: MemoryQuery,
    ) -> List[MemoryResult]:
        """Search this memory system.

        Args:
            query: Search query specification

        Returns:
            List of memory results sorted by relevance
        """
        ...

    async def store(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store a value in this memory system.

        Optional - not all memory systems support direct storage.

        Args:
            key: Unique identifier for the value
            value: Content to store
            metadata: Additional metadata
        """
        pass

    async def get(self, key: str) -> Optional[MemoryResult]:
        """Get a specific item by key.

        Optional - not all memory systems support direct access.

        Args:
            key: Unique identifier

        Returns:
            Memory result or None if not found
        """
        return None

    def is_available(self) -> bool:
        """Check if this provider is available and functional.

        Returns:
            True if provider is ready for queries
        """
        return True


@runtime_checkable
class RankingStrategyProtocol(Protocol):
    """Protocol for result ranking strategies.

    Different strategies can be used to rank results from
    multiple memory sources based on application needs.
    """

    @abstractmethod
    def rank(
        self,
        results: List[MemoryResult],
        query: MemoryQuery,
        limit: int,
    ) -> List[MemoryResult]:
        """Rank and deduplicate results.

        Args:
            results: Raw results from all providers
            query: Original query (for context)
            limit: Maximum results to return

        Returns:
            Ranked, deduplicated results
        """
        ...


# =============================================================================
# Ranking Strategies
# =============================================================================


class RelevanceRankingStrategy:
    """Rank results purely by relevance score."""

    def rank(
        self,
        results: List[MemoryResult],
        query: MemoryQuery,
        limit: int,
    ) -> List[MemoryResult]:
        """Rank by relevance score descending."""
        # Deduplicate by ID; log provenance conflicts for observability
        seen: Dict[str, MemoryResult] = {}
        unique_results = []
        for r in results:
            if r.id not in seen:
                seen[r.id] = r
                unique_results.append(r)
            elif r.provenance and seen[r.id].provenance:
                prev = seen[r.id]
                if str(r.content) != str(prev.content):
                    logger.debug(
                        "memory conflict id=%s: %s vs %s",
                        r.id[:8],
                        r.provenance.store_class,
                        prev.provenance.store_class,
                    )

        # Sort by relevance
        sorted_results = sorted(
            unique_results,
            key=lambda r: r.relevance,
            reverse=True,
        )

        # Filter by min relevance
        if query.min_relevance > 0:
            sorted_results = [r for r in sorted_results if r.relevance >= query.min_relevance]

        return sorted_results[:limit]


class RecencyRankingStrategy:
    """Rank results by recency (most recent first)."""

    def rank(
        self,
        results: List[MemoryResult],
        query: MemoryQuery,
        limit: int,
    ) -> List[MemoryResult]:
        """Rank by timestamp descending (most recent first)."""
        # Deduplicate
        seen: Set[str] = set()
        unique_results = []
        for r in results:
            if r.id not in seen:
                seen.add(r.id)
                unique_results.append(r)

        # Sort by timestamp (None goes last)
        sorted_results = sorted(
            unique_results,
            key=lambda r: r.timestamp or 0,
            reverse=True,
        )

        return sorted_results[:limit]


class HybridRankingStrategy:
    """Combine relevance and recency with configurable weights."""

    def __init__(
        self,
        relevance_weight: float = 0.7,
        recency_weight: float = 0.3,
        recency_decay_hours: float = 24.0,
    ):
        """Initialize hybrid strategy.

        Args:
            relevance_weight: Weight for relevance score (0-1)
            recency_weight: Weight for recency score (0-1)
            recency_decay_hours: Hours after which recency score is 0.5
        """
        self.relevance_weight = relevance_weight
        self.recency_weight = recency_weight
        self.recency_decay_hours = recency_decay_hours

    def _recency_score(self, timestamp: Optional[float]) -> float:
        """Calculate recency score with exponential decay."""
        if timestamp is None:
            return 0.5  # Default for missing timestamp

        age_hours = (time.time() - timestamp) / 3600
        # Exponential decay: score = 1.0 at age 0, 0.5 at decay_hours
        decay_rate = 0.693 / self.recency_decay_hours  # ln(2) / decay_hours
        return max(0.0, min(1.0, 2 ** (-decay_rate * age_hours)))

    def rank(
        self,
        results: List[MemoryResult],
        query: MemoryQuery,
        limit: int,
    ) -> List[MemoryResult]:
        """Rank by weighted combination of relevance and recency."""
        # Deduplicate
        seen: Set[str] = set()
        unique_results = []
        for r in results:
            if r.id not in seen:
                seen.add(r.id)
                unique_results.append(r)

        # Calculate combined scores
        scored_results = []
        for r in unique_results:
            recency = self._recency_score(r.timestamp)
            combined = self.relevance_weight * r.relevance + self.recency_weight * recency
            scored_results.append((combined, r))

        # Sort by combined score
        scored_results.sort(key=lambda x: x[0], reverse=True)

        # Extract results
        ranked = [r for _, r in scored_results]

        # Filter by min relevance
        if query.min_relevance > 0:
            ranked = [r for r in ranked if r.relevance >= query.min_relevance]

        return ranked[:limit]


# =============================================================================
# Unified Memory Coordinator
# =============================================================================


class UnifiedMemoryCoordinator:
    """Federated search across multiple memory systems.

    Provides a unified interface for querying diverse memory backends
    with configurable ranking and deduplication.

    Features:
    - Parallel search across providers
    - Configurable ranking strategies
    - Automatic deduplication
    - Memory type filtering
    - Provider health tracking

    Example:
        coordinator = UnifiedMemoryCoordinator()

        # Register providers
        coordinator.register_provider(ConversationMemoryAdapter(store))
        coordinator.register_provider(EntityMemoryAdapter(entity_mem))

        # Search all memory
        results = await coordinator.search_all("authentication")

        # Search specific types
        results = await coordinator.search_all(
            query="login function",
            memory_types=[MemoryType.ENTITY, MemoryType.CODE],
        )
    """

    def __init__(
        self,
        ranking_strategy: Optional[RankingStrategyProtocol] = None,
        enable_memory_evolution: Optional[bool] = None,
        max_evolution_traces: int = 200,
        trace_ttl_hours: float = 24.0 * 7,
        transfer_boost: float = 0.08,
    ):
        """Initialize coordinator.

        Args:
            ranking_strategy: Strategy for ranking results (default: HybridRankingStrategy)
        """
        if enable_memory_evolution is None:
            try:
                from victor.core.feature_flags import (
                    FeatureFlag,
                    get_feature_flag_manager,
                )

                enable_memory_evolution = get_feature_flag_manager().is_enabled(
                    FeatureFlag.USE_PRIME_MEMORY_EVOLUTION
                )
            except Exception:
                enable_memory_evolution = False
        self._providers: Dict[MemoryType, MemoryProviderProtocol] = {}
        self._ranking_strategy = ranking_strategy or HybridRankingStrategy()
        self._query_count = 0
        self._error_count = 0
        self._enable_memory_evolution = enable_memory_evolution
        self._max_evolution_traces = max(1, max_evolution_traces)
        self._trace_ttl_hours = max(0.0, trace_ttl_hours)
        self._transfer_boost = max(0.0, transfer_boost)
        self._evolution_traces: List[MemoryEvolutionTrace] = []
        self._last_transfer_hints: List[MemoryTransferHint] = []
        self._last_proactive_hints: List[ProactiveMemoryHint] = []
        self._memory_reuse_hits = 0

    def register_provider(self, provider: MemoryProviderProtocol) -> None:
        """Register a memory provider.

        Args:
            provider: Provider implementing MemoryProviderProtocol

        Raises:
            ValueError: If provider for this type already registered
        """
        memory_type = provider.memory_type
        if memory_type in self._providers:
            logger.warning(f"Replacing existing provider for {memory_type.name}")
        self._providers[memory_type] = provider
        logger.info(f"Registered memory provider: {memory_type.name}")

    def unregister_provider(self, memory_type: MemoryType) -> bool:
        """Unregister a memory provider.

        Args:
            memory_type: Type of provider to remove

        Returns:
            True if provider was removed, False if not found
        """
        if memory_type in self._providers:
            del self._providers[memory_type]
            logger.info(f"Unregistered memory provider: {memory_type.name}")
            return True
        return False

    def get_provider(self, memory_type: MemoryType) -> Optional[MemoryProviderProtocol]:
        """Get a specific provider.

        Args:
            memory_type: Type of provider to get

        Returns:
            Provider or None if not registered
        """
        return self._providers.get(memory_type)

    def get_registered_types(self) -> List[MemoryType]:
        """Get list of registered memory types.

        Returns:
            List of registered MemoryType values
        """
        return list(self._providers.keys())

    def set_ranking_strategy(self, strategy: RankingStrategyProtocol) -> None:
        """Set the ranking strategy.

        Args:
            strategy: New ranking strategy to use
        """
        self._ranking_strategy = strategy

    @property
    def last_transfer_hints(self) -> List[MemoryTransferHint]:
        """Latest transfer hints applied during search."""
        return list(self._last_transfer_hints)

    @property
    def last_proactive_hints(self) -> List[ProactiveMemoryHint]:
        """Latest proactive next-turn hints derived during search."""
        return list(self._last_proactive_hints)

    async def search_all(
        self,
        query: str,
        limit: int = 20,
        memory_types: Optional[List[MemoryType]] = None,
        filters: Optional[Dict[str, Any]] = None,
        min_relevance: float = 0.0,
        session_id: Optional[str] = None,
    ) -> List[MemoryResult]:
        """Federated search across registered providers.

        Searches all matching providers in parallel and returns
        ranked, deduplicated results.

        Args:
            query: Search query
            limit: Maximum results to return
            memory_types: Filter to specific memory types (default: all)
            filters: Additional provider-specific filters
            min_relevance: Minimum relevance threshold
            session_id: Optional session context

        Returns:
            Ranked, deduplicated results from all sources
        """
        self._query_count += 1

        # Build query object
        memory_query = MemoryQuery(
            query=query,
            limit=limit * 2,  # Get more for deduplication
            memory_types=memory_types,
            filters=filters,
            min_relevance=min_relevance,
            session_id=session_id,
        )

        # Determine which providers to query
        target_types = memory_types or list(self._providers.keys())
        target_providers = [
            self._providers[t]
            for t in target_types
            if t in self._providers and self._providers[t].is_available()
        ]

        if not target_providers:
            logger.warning("No available providers for query")
            return []

        transfer_context = self._extract_transfer_context(filters)
        transfer_hints = self.suggest_transfer(
            query,
            session_id=session_id,
            limit=min(limit, 3),
            project_path=transfer_context["project_path"],
            vertical_name=transfer_context["vertical_name"],
            transfer_group=transfer_context["transfer_group"],
            allow_cross_project=transfer_context["allow_cross_project"],
        )
        self._last_transfer_hints = transfer_hints
        self._last_proactive_hints = self._derive_proactive_hints_from_transfer_hints(
            transfer_hints,
            limit=min(limit, 3),
        )

        # Parallel search
        search_tasks = [
            self._search_provider(provider, memory_query) for provider in target_providers
        ]

        results_lists = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Flatten and filter errors
        all_results: List[MemoryResult] = []
        for i, results in enumerate(results_lists):
            if isinstance(results, Exception):
                self._error_count += 1
                logger.warning(f"Provider {target_providers[i].memory_type.name} failed: {results}")
            else:
                all_results.extend(results)

        if transfer_hints and all_results:
            all_results = self._apply_transfer_hints(all_results, transfer_hints)
            self._memory_reuse_hits += 1

        # Rank and deduplicate
        ranked_results = self._ranking_strategy.rank(all_results, memory_query, limit)

        logger.debug(
            f"Search '{query[:50]}...' returned {len(ranked_results)} results "
            f"from {len(target_providers)} providers"
        )

        return ranked_results

    async def _search_provider(
        self,
        provider: MemoryProviderProtocol,
        query: MemoryQuery,
    ) -> List[MemoryResult]:
        """Search a single provider with error handling."""
        try:
            return await provider.search(query)
        except Exception as e:
            logger.error(f"Error searching {provider.memory_type.name}: {e}")
            raise

    async def search_type(
        self,
        memory_type: MemoryType,
        query: str,
        limit: int = 10,
        **kwargs: Any,
    ) -> List[MemoryResult]:
        """Search a specific memory type.

        Convenience method for single-type searches.

        Args:
            memory_type: Type of memory to search
            query: Search query
            limit: Maximum results
            **kwargs: Additional search parameters

        Returns:
            Results from the specified memory type
        """
        return await self.search_all(
            query=query,
            limit=limit,
            memory_types=[memory_type],
            **kwargs,
        )

    async def get(
        self,
        memory_type: MemoryType,
        key: str,
    ) -> Optional[MemoryResult]:
        """Get a specific item from a memory type.

        Args:
            memory_type: Type of memory
            key: Item identifier

        Returns:
            Memory result or None if not found
        """
        provider = self._providers.get(memory_type)
        if provider is None:
            return None
        return await provider.get(key)

    async def store(
        self,
        memory_type: MemoryType,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Store a value in a specific memory type.

        Args:
            memory_type: Type of memory
            key: Unique identifier
            value: Content to store
            metadata: Additional metadata

        Returns:
            True if stored successfully
        """
        provider = self._providers.get(memory_type)
        if provider is None:
            logger.warning(f"No provider for {memory_type.name}")
            return False

        try:
            await provider.store(key, value, metadata)
            return True
        except Exception as e:
            logger.error(f"Failed to store in {memory_type.name}: {e}")
            return False

    async def record_outcome(
        self,
        query: str,
        results: List[MemoryResult],
        success: bool,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Record a successful query outcome for future transfer hints."""
        if not self._enable_memory_evolution or not success or not query.strip() or not results:
            return False

        trace = MemoryEvolutionTrace(
            query=query.strip(),
            result_ids=[result.id for result in results if result.id][:10],
            source_types=sorted({result.source.name for result in results}),
            timestamp=time.time(),
            session_id=session_id,
            metadata=self._normalize_transfer_metadata(metadata),
        )
        self._evolution_traces.append(trace)
        if len(self._evolution_traces) > self._max_evolution_traces:
            self._evolution_traces = self._evolution_traces[-self._max_evolution_traces :]
        return True

    def suggest_transfer(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 3,
        *,
        project_path: Optional[str] = None,
        vertical_name: Optional[str] = None,
        transfer_group: Optional[str] = None,
        allow_cross_project: bool = False,
    ) -> List[MemoryTransferHint]:
        """Suggest transferable memory hints from prior successful traces."""
        if not self._enable_memory_evolution or not query.strip() or limit <= 0:
            return []

        query_tokens = self._tokenize_query(query)
        if not query_tokens:
            return []

        freshness_cutoff = time.time() - (self._trace_ttl_hours * 3600)
        hints: List[MemoryTransferHint] = []
        for trace in reversed(self._evolution_traces):
            if trace.timestamp < freshness_cutoff:
                continue

            trace_tokens = self._tokenize_query(trace.query)
            if not trace_tokens:
                continue

            overlap = query_tokens & trace_tokens
            if not overlap:
                continue

            similarity = len(overlap) / max(1, len(query_tokens | trace_tokens))
            cross_session = bool(session_id and trace.session_id and session_id != trace.session_id)
            transfer_scope = self._resolve_transfer_scope(
                trace,
                session_id=session_id,
                project_path=project_path,
                vertical_name=vertical_name,
                transfer_group=transfer_group,
                allow_cross_project=allow_cross_project,
            )
            if transfer_scope is None:
                continue
            if similarity < 0.2 and not cross_session:
                continue

            score = similarity + (0.05 if cross_session else 0.0)
            hint_metadata = dict(trace.metadata)
            source_project_path = self._normalize_scope_value(trace.metadata.get("project_path"))
            source_vertical_name = self._normalize_scope_value(trace.metadata.get("vertical_name"))
            normalized_transfer_group = self._normalize_scope_value(
                trace.metadata.get("transfer_group")
            )
            if source_project_path:
                hint_metadata["source_project_path"] = source_project_path
            if source_vertical_name:
                hint_metadata["source_vertical_name"] = source_vertical_name
            if normalized_transfer_group:
                hint_metadata["transfer_group"] = normalized_transfer_group
            hint_metadata["transfer_scope"] = transfer_scope
            hints.append(
                MemoryTransferHint(
                    matched_query=trace.query,
                    score=score,
                    preferred_result_ids=list(trace.result_ids),
                    preferred_source_types=list(trace.source_types),
                    session_id=trace.session_id,
                    cross_session=cross_session,
                    metadata=hint_metadata,
                )
            )

        hints.sort(key=lambda hint: hint.score, reverse=True)
        return hints[:limit]

    def suggest_proactive_hints(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 3,
        *,
        project_path: Optional[str] = None,
        vertical_name: Optional[str] = None,
        transfer_group: Optional[str] = None,
        allow_cross_project: bool = False,
    ) -> List[ProactiveMemoryHint]:
        """Generate bounded next-turn hints from successful memory traces."""
        transfer_hints = self.suggest_transfer(
            query,
            session_id=session_id,
            limit=limit,
            project_path=project_path,
            vertical_name=vertical_name,
            transfer_group=transfer_group,
            allow_cross_project=allow_cross_project,
        )
        return self._derive_proactive_hints_from_transfer_hints(transfer_hints, limit=limit)

    def _derive_proactive_hints_from_transfer_hints(
        self,
        transfer_hints: List[MemoryTransferHint],
        limit: int,
    ) -> List[ProactiveMemoryHint]:
        """Convert transfer hints into bounded next-turn guidance."""
        proactive_hints: List[ProactiveMemoryHint] = []
        seen_hints: Set[str] = set()

        for transfer_hint in transfer_hints:
            hint_text = self._build_proactive_hint_text(transfer_hint)
            if not hint_text or hint_text in seen_hints:
                continue

            proactive_hints.append(
                ProactiveMemoryHint(
                    hint=hint_text,
                    score=transfer_hint.score,
                    matched_query=transfer_hint.matched_query,
                    preferred_result_ids=list(transfer_hint.preferred_result_ids),
                    preferred_source_types=list(transfer_hint.preferred_source_types),
                    metadata=dict(transfer_hint.metadata),
                )
            )
            seen_hints.add(hint_text)

        proactive_hints.sort(key=lambda hint: hint.score, reverse=True)
        return proactive_hints[:limit]

    def _apply_transfer_hints(
        self,
        results: List[MemoryResult],
        transfer_hints: List[MemoryTransferHint],
    ) -> List[MemoryResult]:
        """Boost matching results using prior successful traces."""
        result_boosts: Dict[str, float] = {}
        source_boosts: Dict[str, float] = {}
        query_map: Dict[str, List[str]] = {}
        scope_map: Dict[str, str] = {}
        source_scope_map: Dict[str, str] = {}

        for hint in transfer_hints:
            transfer_scope = str(hint.metadata.get("transfer_scope") or "")
            for result_id in hint.preferred_result_ids:
                result_boosts[result_id] = max(
                    result_boosts.get(result_id, 0.0), self._transfer_boost
                )
                query_map.setdefault(result_id, []).append(hint.matched_query)
                if transfer_scope and result_id not in scope_map:
                    scope_map[result_id] = transfer_scope
            for source_type in hint.preferred_source_types:
                source_boosts[source_type] = max(
                    source_boosts.get(source_type, 0.0), self._transfer_boost * 0.5
                )
                if transfer_scope and source_type not in source_scope_map:
                    source_scope_map[source_type] = transfer_scope

        boosted_results: List[MemoryResult] = []
        for result in results:
            boost = result_boosts.get(result.id, 0.0)
            boost += source_boosts.get(result.source.name, 0.0)
            if boost <= 0:
                boosted_results.append(result)
                continue

            metadata = dict(result.metadata)
            metadata["memory_transfer_boost"] = round(boost, 4)
            metadata["memory_transfer_queries"] = query_map.get(result.id, [])
            transfer_scope = scope_map.get(result.id) or source_scope_map.get(result.source.name)
            if transfer_scope:
                metadata["memory_transfer_scope"] = transfer_scope
            boosted_results.append(
                MemoryResult(
                    source=result.source,
                    content=result.content,
                    relevance=min(1.0, result.relevance + boost),
                    id=result.id,
                    metadata=metadata,
                    timestamp=result.timestamp,
                )
            )

        return boosted_results

    def _build_proactive_hint_text(
        self,
        transfer_hint: MemoryTransferHint,
    ) -> str:
        """Create a short reusable hint from transfer metadata."""
        metadata = transfer_hint.metadata
        fragments: List[str] = []

        gaps = self._normalize_hint_list(metadata.get("gaps"))
        if gaps:
            fragments.append(f"Carry forward open gaps: {', '.join(gaps[:2])}.")

        next_steps = self._normalize_hint_list(metadata.get("next_steps"))
        if next_steps:
            fragments.append(f"Reuse next steps: {'; '.join(next_steps[:2])}.")

        if not fragments and transfer_hint.preferred_source_types:
            fragments.append(
                "Revisit memory sources: "
                + ", ".join(transfer_hint.preferred_source_types[:2])
                + "."
            )

        if not fragments and transfer_hint.preferred_result_ids:
            fragments.append("Revisit prior successful memory results.")

        if not fragments:
            return ""

        return f"Matched prior query '{transfer_hint.matched_query}'. " + " ".join(fragments)

    @staticmethod
    def _normalize_hint_list(value: Any) -> List[str]:
        """Normalize metadata hint fields to a compact string list."""
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            items = value
        else:
            items = [value]
        return [str(item).strip() for item in items if str(item).strip()]

    @staticmethod
    def _normalize_scope_value(value: Any) -> Optional[str]:
        """Normalize transfer-scope values for comparisons."""
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _normalize_transfer_metadata(
        self,
        metadata: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Normalize transfer metadata persisted with evolution traces."""
        normalized = dict(metadata or {})
        for key in ("project_path", "vertical_name", "transfer_group"):
            value = self._normalize_scope_value(normalized.get(key))
            if value is None:
                normalized.pop(key, None)
            else:
                normalized[key] = value
        if "allow_cross_project" in normalized:
            normalized["allow_cross_project"] = bool(normalized["allow_cross_project"])
        return normalized

    def _extract_transfer_context(
        self,
        filters: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Extract transfer-policy context from federated-search filters."""
        filter_dict = dict(filters or {})
        return {
            "project_path": self._normalize_scope_value(filter_dict.get("project_path")),
            "vertical_name": self._normalize_scope_value(filter_dict.get("vertical_name")),
            "transfer_group": self._normalize_scope_value(filter_dict.get("transfer_group")),
            "allow_cross_project": bool(filter_dict.get("allow_cross_project")),
        }

    def _resolve_transfer_scope(
        self,
        trace: MemoryEvolutionTrace,
        *,
        session_id: Optional[str],
        project_path: Optional[str],
        vertical_name: Optional[str],
        transfer_group: Optional[str],
        allow_cross_project: bool,
    ) -> Optional[str]:
        """Decide whether a trace may transfer into the current request scope."""
        trace_project_path = self._normalize_scope_value(trace.metadata.get("project_path"))
        trace_vertical_name = self._normalize_scope_value(trace.metadata.get("vertical_name"))
        trace_transfer_group = self._normalize_scope_value(trace.metadata.get("transfer_group"))
        current_project_path = self._normalize_scope_value(project_path)
        current_vertical_name = self._normalize_scope_value(vertical_name)
        current_transfer_group = self._normalize_scope_value(transfer_group)
        cross_session = bool(session_id and trace.session_id and session_id != trace.session_id)

        if (
            current_project_path
            and trace_project_path
            and current_project_path != trace_project_path
        ):
            if not allow_cross_project:
                return None
            if (
                current_transfer_group
                and trace_transfer_group
                and current_transfer_group == trace_transfer_group
            ):
                return "cross_project_transfer_group"
            if (
                current_vertical_name
                and trace_vertical_name
                and current_vertical_name == trace_vertical_name
            ):
                return "cross_project_same_vertical"
            if bool(trace.metadata.get("allow_cross_project")):
                return "cross_project_explicit"
            return None

        if (
            current_project_path
            and trace_project_path
            and current_project_path == trace_project_path
        ):
            return "same_project" if cross_session else "same_session"

        if cross_session:
            return "cross_session"

        return "same_session"

    def _tokenize_query(self, query: str) -> Set[str]:
        """Extract stable lexical tokens for simple transfer matching."""
        stopwords = {
            "the",
            "and",
            "for",
            "with",
            "this",
            "that",
            "from",
            "into",
            "your",
            "about",
            "have",
            "has",
        }
        return {
            token
            for token in re.findall(r"[a-z0-9_]+", query.lower())
            if len(token) > 2 and token not in stopwords
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics.

        Returns:
            Dict with query counts, error rates, provider status
        """
        return {
            "query_count": self._query_count,
            "error_count": self._error_count,
            "error_rate": (self._error_count / max(1, self._query_count)),
            "memory_evolution_enabled": self._enable_memory_evolution,
            "memory_reuse_hits": self._memory_reuse_hits,
            "evolution_trace_count": len(self._evolution_traces),
            "last_transfer_hint_count": len(self._last_transfer_hints),
            "last_proactive_hint_count": len(self._last_proactive_hints),
            "providers": {
                t.name: {
                    "available": p.is_available(),
                }
                for t, p in self._providers.items()
            },
        }


# =============================================================================
# Factory Functions
# =============================================================================


_coordinator_instance: Optional[UnifiedMemoryCoordinator] = None


def create_memory_coordinator(
    ranking_strategy: Optional[RankingStrategyProtocol] = None,
    enable_memory_evolution: Optional[bool] = None,
    max_evolution_traces: int = 200,
    trace_ttl_hours: float = 24.0 * 7,
    transfer_boost: float = 0.08,
) -> UnifiedMemoryCoordinator:
    """Create a new memory coordinator.

    Args:
        ranking_strategy: Optional ranking strategy

    Returns:
        New UnifiedMemoryCoordinator instance
    """
    return UnifiedMemoryCoordinator(
        ranking_strategy=ranking_strategy,
        enable_memory_evolution=enable_memory_evolution,
        max_evolution_traces=max_evolution_traces,
        trace_ttl_hours=trace_ttl_hours,
        transfer_boost=transfer_boost,
    )


def get_memory_coordinator() -> UnifiedMemoryCoordinator:
    """Get or create the singleton memory coordinator.

    Returns:
        Global UnifiedMemoryCoordinator instance
    """
    global _coordinator_instance
    if _coordinator_instance is None:
        _coordinator_instance = create_memory_coordinator()
    return _coordinator_instance


def reset_memory_coordinator() -> None:
    """Reset the singleton coordinator (for testing)."""
    global _coordinator_instance
    _coordinator_instance = None


__all__ = [
    # Enums
    "MemoryType",
    # Data types
    "MemoryResult",
    "MemoryQuery",
    "MemoryEvolutionTrace",
    "MemoryTransferHint",
    "ProactiveMemoryHint",
    # Protocols
    "MemoryProviderProtocol",
    "RankingStrategyProtocol",
    # Strategies
    "RelevanceRankingStrategy",
    "RecencyRankingStrategy",
    "HybridRankingStrategy",
    # Coordinator
    "UnifiedMemoryCoordinator",
    # Factory
    "create_memory_coordinator",
    "get_memory_coordinator",
    "reset_memory_coordinator",
]
