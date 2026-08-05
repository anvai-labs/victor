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

"""Prompt Enrichment Framework for auto prompt optimization.

This module provides infrastructure for automatically enriching prompts with
relevant context from various sources. It's designed to be vertical-agnostic,
with specific enrichment strategies injected by each vertical.

Enrichment Types by Vertical:
- Coding: Knowledge graph symbols, related code snippets, imports
- Research: Web search results, source citations, prior research
- DevOps: Infrastructure context, command patterns, env hints
- Data Analysis: Schema context, query patterns, data profiles
- RAG: Retrieved documents, knowledge base excerpts

Features:
- Token budget enforcement (default: 2000 tokens max)
- Synchronous enrichment with configurable timeout
- Caching for repeated enrichments
- RL tracking for learning enrichment effectiveness

This module is part of the framework layer and should NOT have dependencies
on specific verticals. Verticals inject their strategies via protocols.

Example:
    from victor.framework.enrichment import (
        PromptEnrichmentService,
        EnrichmentContext,
        EnrichmentType,
    )

    # Create service
    service = PromptEnrichmentService(
        max_tokens=2000,
        timeout_ms=500,
    )

    # Register vertical strategies
    service.register_strategy("coding", coding_enrichment_strategy)

    # Enrich a prompt
    context = EnrichmentContext(
        session_id="session_123",
        task_type="edit",
        file_mentions=["src/auth/login.py"],
        symbol_mentions=["authenticate_user"],
    )

    enriched = await service.enrich(
        prompt="Add rate limiting to the authentication function",
        vertical="coding",
        context=context,
    )

    print(enriched.enriched_prompt)  # Original + context enrichments
    print(enriched.enrichments)  # List of applied enrichments
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class EnrichmentType(str, Enum):
    """Types of prompt enrichment.

    Each type corresponds to a different source of contextual information
    that can enhance prompt quality.

    Attributes:
        KNOWLEDGE_GRAPH: Code symbols, relationships, and dependencies
        CODE_SNIPPET: Related code examples from the codebase
        CONVERSATION: Relevant snippets from conversation history
        WEB_SEARCH: Pre-fetched web search results for grounding
        SCHEMA: Database/API schema information
        TOOL_HISTORY: Patterns from successful tool usage
        PROJECT_CONTEXT: Project-specific context (.victor.md, README)
    """

    KNOWLEDGE_GRAPH = "knowledge_graph"
    CODE_SNIPPET = "code_snippet"
    CONVERSATION = "conversation"
    WEB_SEARCH = "web_search"
    SCHEMA = "schema"
    TOOL_HISTORY = "tool_history"
    PROJECT_CONTEXT = "project_context"


class EnrichmentPriority(int, Enum):
    """Priority levels for enrichment ordering.

    Higher priority enrichments are applied first and get more
    of the token budget.

    Attributes:
        CRITICAL: Must-have context (e.g., file being edited)
        HIGH: Important supporting context
        NORMAL: Standard enrichment
        LOW: Nice-to-have context
    """

    CRITICAL = 100
    HIGH = 75
    NORMAL = 50
    LOW = 25


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class EnrichmentContext:
    """Context for prompt enrichment.

    Carries metadata about the current task to inform enrichment
    strategies about what context would be most valuable.

    Attributes:
        session_id: Current session identifier
        task_type: Detected task type (e.g., "edit", "create", "analyze")
        file_mentions: Files mentioned in the conversation
        symbol_mentions: Code symbols (functions, classes) mentioned
        tool_history: Recent tool calls in this session
        metadata: Additional vertical-specific metadata
    """

    session_id: Optional[str] = None
    task_type: Optional[str] = None
    file_mentions: List[str] = field(default_factory=list)
    symbol_mentions: List[str] = field(default_factory=list)
    tool_history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextEnrichment:
    """A single enrichment to be added to a prompt.

    Represents one piece of contextual information that can enhance
    the prompt. Multiple enrichments are combined respecting token budgets.

    Attributes:
        type: Type of enrichment (knowledge_graph, code_snippet, etc.)
        content: The actual enrichment text to inject
        priority: Priority level for ordering and budget allocation
        source: Where this enrichment came from (e.g., file path, URL)
        token_estimate: Estimated token count for budget tracking
        metadata: Additional metadata about the enrichment
    """

    type: EnrichmentType
    content: str
    priority: EnrichmentPriority = EnrichmentPriority.NORMAL
    source: Optional[str] = None
    token_estimate: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Estimate token count if not provided."""
        if self.token_estimate == 0 and self.content:
            # Rough estimate: ~4 chars per token
            self.token_estimate = len(self.content) // 4


@dataclass
class EnrichedPrompt:
    """Result of prompt enrichment.

    Contains the enriched prompt along with metadata about
    what enrichments were applied.

    Attributes:
        original_prompt: The original prompt before enrichment
        enriched_prompt: The prompt with enrichments applied
        enrichments: List of enrichments that were applied
        total_tokens_added: Total tokens added through enrichments
        enrichment_time_ms: Time taken for enrichment in milliseconds
        from_cache: Whether the result came from cache
    """

    original_prompt: str
    enriched_prompt: str
    enrichments: List[ContextEnrichment] = field(default_factory=list)
    total_tokens_added: int = 0
    enrichment_time_ms: float = 0.0
    from_cache: bool = False

    @property
    def enrichment_count(self) -> int:
        """Get the number of enrichments applied."""
        return len(self.enrichments)

    @property
    def enrichment_types(self) -> List[str]:
        """Get list of enrichment types applied."""
        return [e.type.value for e in self.enrichments]


@dataclass
class EnrichmentOutcome:
    """Outcome of an enrichment for RL learning.

    Records whether an enrichment helped or hurt task completion
    for learning which enrichments work best.

    Attributes:
        enrichment_type: Type of enrichment applied
        enrichment_count: Number of enrichments of this type
        task_success: Whether the task completed successfully
        quality_improvement: Estimated quality improvement (-1.0 to 1.0)
        task_type: The task type that was enriched
        vertical: The vertical that provided the enrichment
    """

    enrichment_type: str
    enrichment_count: int
    task_success: bool
    quality_improvement: float
    task_type: Optional[str] = None
    vertical: Optional[str] = None


# =============================================================================
# Protocol
# =============================================================================


@runtime_checkable
class EnrichmentStrategyProtocol(Protocol):
    """Protocol for vertical-specific enrichment strategies.

    Each vertical implements this protocol to provide domain-specific
    prompt enrichments. Strategies are registered with the enrichment
    service and called based on the active vertical.

    Example:
        class CodingEnrichmentStrategy:
            async def get_enrichments(
                self,
                prompt: str,
                context: EnrichmentContext,
            ) -> List[ContextEnrichment]:
                # Query knowledge graph for relevant symbols
                symbols = await self.knowledge_graph.search(prompt)
                return [
                    ContextEnrichment(
                        type=EnrichmentType.KNOWLEDGE_GRAPH,
                        content=format_symbols(symbols),
                        priority=EnrichmentPriority.HIGH,
                    )
                ]

            def get_priority(self) -> int:
                return 50

            def get_token_allocation(self) -> float:
                return 0.4  # Use up to 40% of token budget
    """

    async def get_enrichments(
        self,
        prompt: str,
        context: EnrichmentContext,
    ) -> List[ContextEnrichment]:
        """Get enrichments for a prompt.

        Args:
            prompt: The prompt to enrich
            context: Context about the current task

        Returns:
            List of enrichments to apply to the prompt
        """
        ...

    def get_priority(self) -> int:
        """Get priority for this strategy.

        Lower values are processed first.

        Returns:
            Priority value (default 50)
        """
        ...

    def get_token_allocation(self) -> float:
        """Get fraction of token budget this strategy can use.

        Returns:
            Float between 0.0 and 1.0 (e.g., 0.4 for 40%)
        """
        ...


# =============================================================================
# Cache
# =============================================================================


@dataclass
class _CacheEntry:
    """Internal cache entry for enrichment results."""

    result: EnrichedPrompt
    timestamp: float
    cache_key: str


class EnrichmentCache:
    """Simple in-memory cache for enrichment results.

    Caches enriched prompts to avoid re-computing enrichments for
    the same prompt/context combination within a short time window.

    Attributes:
        ttl_seconds: Time-to-live for cache entries
        max_entries: Maximum number of entries to cache
    """

    def __init__(self, ttl_seconds: int = 300, max_entries: int = 100):
        """Initialize the cache.

        Args:
            ttl_seconds: Cache entry lifetime in seconds (default: 5 minutes)
            max_entries: Maximum cache size (default: 100 entries)
        """
        self._cache: Dict[str, _CacheEntry] = {}
        self._ttl = ttl_seconds
        self._max_entries = max_entries

    def _compute_key(
        self,
        prompt: str,
        vertical: str,
        context: EnrichmentContext,
    ) -> str:
        """Compute cache key from prompt and context."""
        key_parts = [
            prompt[:500],  # First 500 chars of prompt
            vertical,
            context.task_type or "",
            ",".join(context.file_mentions[:5]),
            ",".join(context.symbol_mentions[:5]),
        ]
        key_str = "|".join(key_parts)
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]

    def get(
        self,
        prompt: str,
        vertical: str,
        context: EnrichmentContext,
    ) -> Optional[EnrichedPrompt]:
        """Get cached enrichment result.

        Args:
            prompt: The prompt that was enriched
            vertical: The vertical used for enrichment
            context: The enrichment context

        Returns:
            Cached EnrichedPrompt if found and valid, None otherwise
        """
        key = self._compute_key(prompt, vertical, context)
        entry = self._cache.get(key)

        if entry is None:
            return None

        # Check TTL
        if time.time() - entry.timestamp > self._ttl:
            del self._cache[key]
            return None

        # Mark as cache hit
        result = entry.result
        return EnrichedPrompt(
            original_prompt=result.original_prompt,
            enriched_prompt=result.enriched_prompt,
            enrichments=result.enrichments,
            total_tokens_added=result.total_tokens_added,
            enrichment_time_ms=result.enrichment_time_ms,
            from_cache=True,
        )

    def set(
        self,
        prompt: str,
        vertical: str,
        context: EnrichmentContext,
        result: EnrichedPrompt,
    ) -> None:
        """Cache an enrichment result.

        Args:
            prompt: The prompt that was enriched
            vertical: The vertical used for enrichment
            context: The enrichment context
            result: The enrichment result to cache
        """
        # Evict old entries if at capacity
        if len(self._cache) >= self._max_entries:
            self._evict_oldest()

        key = self._compute_key(prompt, vertical, context)
        self._cache[key] = _CacheEntry(
            result=result,
            timestamp=time.time(),
            cache_key=key,
        )

    def _evict_oldest(self) -> None:
        """Evict oldest cache entries."""
        if not self._cache:
            return

        # Sort by timestamp and remove oldest 20%
        sorted_entries = sorted(self._cache.items(), key=lambda x: x[1].timestamp)
        to_remove = max(1, len(sorted_entries) // 5)

        for key, _ in sorted_entries[:to_remove]:
            del self._cache[key]

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()

    def __len__(self) -> int:
        """Get number of cache entries."""
        return len(self._cache)

    def __bool__(self) -> bool:
        """Cache is always truthy when it exists (even if empty)."""
        return True


# =============================================================================
# Main Service
# =============================================================================


class PromptEnrichmentService:
    """Auto prompt optimization service.

    Coordinates prompt enrichment across verticals, enforcing token budgets,
    timeouts, and caching. Supports learning from enrichment outcomes.

    This is a framework-level service that is vertical-agnostic. Each vertical
    registers its own enrichment strategy that knows how to find relevant
    context for that domain.

    Example:
        service = PromptEnrichmentService(max_tokens=2000)
        service.register_strategy("coding", CodingEnrichmentStrategy())

        enriched = await service.enrich(
            prompt="Fix the authentication bug",
            vertical="coding",
            context=EnrichmentContext(task_type="debug"),
        )
    """

    def __init__(
        self,
        max_tokens: int = 2000,
        timeout_ms: float = 500.0,
        cache_ttl_seconds: int = 300,
        cache_enabled: bool = True,
    ):
        """Initialize the enrichment service.

        Args:
            max_tokens: Maximum tokens to add through enrichment (default: 2000)
            timeout_ms: Timeout for enrichment in milliseconds (default: 500ms)
            cache_ttl_seconds: Cache TTL in seconds (default: 5 minutes)
            cache_enabled: Whether to enable caching (default: True)
        """
        self._max_tokens = max_tokens
        self._timeout_ms = timeout_ms
        self._strategies: Dict[str, EnrichmentStrategyProtocol] = {}
        self._cache = EnrichmentCache(ttl_seconds=cache_ttl_seconds) if cache_enabled else None
        self._outcome_callbacks: List[Callable[[EnrichmentOutcome], None]] = []

        logger.debug(
            "PromptEnrichmentService initialized: max_tokens=%d, " "timeout_ms=%.1f, cache=%s",
            max_tokens,
            timeout_ms,
            "enabled" if cache_enabled else "disabled",
        )

    def register_strategy(
        self,
        vertical: str,
        strategy: EnrichmentStrategyProtocol,
    ) -> None:
        """Register an enrichment strategy for a vertical.

        Args:
            vertical: Vertical name (e.g., "coding", "research")
            strategy: Strategy instance implementing EnrichmentStrategyProtocol
        """
        self._strategies[vertical] = strategy
        logger.info(
            "Registered enrichment strategy for vertical=%s, priority=%d",
            vertical,
            strategy.get_priority(),
        )

    def unregister_strategy(self, vertical: str) -> None:
        """Unregister an enrichment strategy.

        Args:
            vertical: Vertical name to unregister
        """
        self._strategies.pop(vertical, None)

    def on_outcome(
        self,
        callback: Callable[[EnrichmentOutcome], None],
    ) -> None:
        """Register callback for enrichment outcomes.

        Use this to track enrichment effectiveness for RL learning.

        Args:
            callback: Function called with EnrichmentOutcome after task completion
        """
        self._outcome_callbacks.append(callback)

    def record_outcome(self, outcome: EnrichmentOutcome) -> None:
        """Record an enrichment outcome for learning.

        Args:
            outcome: The outcome to record
        """
        for callback in self._outcome_callbacks:
            try:
                callback(outcome)
            except Exception as e:
                logger.warning("Error in outcome callback: %s", e)

    async def enrich(
        self,
        prompt: str,
        vertical: str,
        context: EnrichmentContext,
    ) -> EnrichedPrompt:
        """Enrich a prompt with relevant context.

        This is the main entry point for prompt enrichment. It:
        1. Checks cache for existing enrichment
        2. Gets enrichments from the vertical's strategy
        3. Applies token budget constraints
        4. Merges enrichments into the prompt
        5. Caches the result

        Args:
            prompt: The prompt to enrich
            vertical: The active vertical (e.g., "coding")
            context: Context about the current task

        Returns:
            EnrichedPrompt with the enriched prompt and metadata
        """
        start_time = time.time()

        # Check cache first
        if self._cache:
            cached = self._cache.get(prompt, vertical, context)
            if cached:
                logger.debug(
                    "Enrichment cache hit for vertical=%s, task_type=%s",
                    vertical,
                    context.task_type,
                )
                return cached

        # Get strategy for vertical
        strategy = self._strategies.get(vertical)
        if not strategy:
            logger.debug(
                "No enrichment strategy for vertical=%s, returning original prompt",
                vertical,
            )
            return EnrichedPrompt(
                original_prompt=prompt,
                enriched_prompt=prompt,
                enrichment_time_ms=(time.time() - start_time) * 1000,
            )

        # Get enrichments with timeout
        try:
            enrichments = await asyncio.wait_for(
                strategy.get_enrichments(prompt, context),
                timeout=self._timeout_ms / 1000.0,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Enrichment timeout for vertical=%s after %.1fms",
                vertical,
                self._timeout_ms,
            )
            return EnrichedPrompt(
                original_prompt=prompt,
                enriched_prompt=prompt,
                enrichment_time_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            logger.error("Enrichment error for vertical=%s: %s", vertical, e)
            return EnrichedPrompt(
                original_prompt=prompt,
                enriched_prompt=prompt,
                enrichment_time_ms=(time.time() - start_time) * 1000,
            )

        # Apply token budget constraints
        constrained = self._apply_token_budget(enrichments)

        # Merge enrichments into prompt
        enriched_prompt = self._merge_enrichments(prompt, constrained)

        # Calculate totals
        total_tokens = sum(e.token_estimate for e in constrained)
        elapsed_ms = (time.time() - start_time) * 1000

        result = EnrichedPrompt(
            original_prompt=prompt,
            enriched_prompt=enriched_prompt,
            enrichments=constrained,
            total_tokens_added=total_tokens,
            enrichment_time_ms=elapsed_ms,
        )

        # Cache result
        if self._cache:
            self._cache.set(prompt, vertical, context, result)

        logger.info(
            "Enriched prompt: vertical=%s, task_type=%s, enrichments=%d, "
            "tokens_added=%d, time_ms=%.1f",
            vertical,
            context.task_type,
            len(constrained),
            total_tokens,
            elapsed_ms,
        )

        return result

    def _apply_token_budget(
        self,
        enrichments: List[ContextEnrichment],
    ) -> List[ContextEnrichment]:
        """Apply token budget constraints to enrichments.

        Sorts by priority and includes enrichments until budget is exhausted.

        Args:
            enrichments: List of potential enrichments

        Returns:
            List of enrichments that fit within token budget
        """
        if not enrichments:
            return []

        # Sort by priority (highest first)
        sorted_enrichments = sorted(
            enrichments,
            key=lambda e: (e.priority.value, -e.token_estimate),
            reverse=True,
        )

        # Include enrichments until budget exhausted
        result = []
        remaining_budget = self._max_tokens

        for enrichment in sorted_enrichments:
            if enrichment.token_estimate <= remaining_budget:
                result.append(enrichment)
                remaining_budget -= enrichment.token_estimate
            elif remaining_budget <= 0:
                break

        logger.debug(
            "Token budget: %d/%d used, %d/%d enrichments included",
            self._max_tokens - remaining_budget,
            self._max_tokens,
            len(result),
            len(enrichments),
        )

        return result

    def _merge_enrichments(
        self,
        prompt: str,
        enrichments: List[ContextEnrichment],
    ) -> str:
        """Merge enrichments into the prompt.

        Formats enrichments with clear section headers and injects
        them before the main prompt.

        Args:
            prompt: The original prompt
            enrichments: Enrichments to merge

        Returns:
            Enriched prompt string
        """
        if not enrichments:
            return prompt

        # Group enrichments by type for cleaner formatting
        sections: Dict[EnrichmentType, List[str]] = {}
        for enrichment in enrichments:
            if enrichment.type not in sections:
                sections[enrichment.type] = []
            sections[enrichment.type].append(enrichment.content)

        # Build enrichment section
        enrichment_parts = []

        # Define section headers
        headers = {
            EnrichmentType.KNOWLEDGE_GRAPH: "Relevant Code Context",
            EnrichmentType.CODE_SNIPPET: "Related Code Examples",
            EnrichmentType.CONVERSATION: "Relevant Conversation History",
            EnrichmentType.WEB_SEARCH: "Web Search Context",
            EnrichmentType.SCHEMA: "Schema Information",
            EnrichmentType.TOOL_HISTORY: "Successful Tool Patterns",
            EnrichmentType.PROJECT_CONTEXT: "Project Context",
        }

        for etype, contents in sections.items():
            header = headers.get(etype, etype.value.replace("_", " ").title())
            enrichment_parts.append(f"## {header}")
            enrichment_parts.extend(contents)
            enrichment_parts.append("")

        if not enrichment_parts:
            return prompt

        # Combine enrichments with original prompt
        enrichment_text = "\n".join(enrichment_parts)
        return f"{enrichment_text}\n---\n\n{prompt}"

    @property
    def max_tokens(self) -> int:
        """Get the maximum token budget."""
        return self._max_tokens

    @max_tokens.setter
    def max_tokens(self, value: int) -> None:
        """Set the maximum token budget."""
        self._max_tokens = max(0, value)

    @property
    def timeout_ms(self) -> float:
        """Get the timeout in milliseconds."""
        return self._timeout_ms

    @timeout_ms.setter
    def timeout_ms(self, value: float) -> None:
        """Set the timeout in milliseconds."""
        self._timeout_ms = max(0.0, value)

    @property
    def registered_verticals(self) -> List[str]:
        """Get list of verticals with registered strategies."""
        return list(self._strategies.keys())

    def clear_cache(self) -> None:
        """Clear the enrichment cache."""
        if self._cache:
            self._cache.clear()


# =============================================================================
# Module Exports
# =============================================================================


__all__ = [
    # Enums
    "EnrichmentType",
    "EnrichmentPriority",
    # Data Classes
    "EnrichmentContext",
    "ContextEnrichment",
    "EnrichedPrompt",
    "EnrichmentOutcome",
    # Protocols
    "EnrichmentStrategyProtocol",
    # Services
    "PromptEnrichmentService",
    "EnrichmentCache",
]
