from __future__ import annotations

# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Intelligent system prompt builder with embedding-based context selection.

Architecture (Strategy + Observer + Builder patterns):
- Uses conversation embeddings to select relevant historical context
- Learns from user feedback via reinforcement learning signals
- Adapts prompts per-profile based on model capabilities
- Supports cold start (lazy) and warm cache (background) modes

Key Features:
1. Embedding-Based Context Selection:
   - Retrieves semantically similar past interactions
   - Filters by task type, success rate, and recency
   - Weighs context by relevance score

2. Profile-Specific Learning:
   - Tracks per-model performance metrics
   - Adjusts prompt style based on model strengths/weaknesses
   - Learns optimal tool budgets and mode transitions

3. Adaptive Prompt Generation:
   - Task-type-specific prompt templates
   - Dynamic grounding rules based on model reliability
   - Success-weighted example selection

4. Cold/Warm Cache Management:
   - Lazy evaluation on first use (cold start)
   - Background embedding refresh (warm cache)
   - Automatic cache invalidation on model switch

Usage:
    builder = await IntelligentPromptBuilder.create(
        provider_name="ollama",
        model="qwen2.5:32b",
        profile_name="local-qwen",
    )

    prompt = await builder.build(
        task="analyze the authentication module",
        task_type="analysis",
        conversation_history=messages,
    )
"""

import asyncio
import hashlib
from victor.core.json_utils import json_dumps, json_loads
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from victor.core.grounding_texts import (
    GROUNDING_RULES as CANONICAL_GROUNDING_RULES,
    GROUNDING_RULES_EXTENDED as CANONICAL_GROUNDING_RULES_EXTENDED,
)
from victor.tools.core_tool_aliases import canonicalize_core_tool_name

if TYPE_CHECKING:
    from victor.agent.conversation_embedding_store import ConversationEmbeddingStore
    from victor.agent.tool_calling.base import ToolCallingCapabilities
    from victor.storage.embeddings.service import EmbeddingService

logger = logging.getLogger(__name__)


class PromptStrategy(Enum):
    """Prompt generation strategies."""

    MINIMAL = "minimal"  # Cloud models - minimal guidance
    STRUCTURED = "structured"  # Capable local models - structured guidance
    STRICT = "strict"  # Less capable models - strict rules
    ADAPTIVE = "adaptive"  # Dynamic based on learned performance


class CacheState(Enum):
    """Embedding cache states."""

    COLD = "cold"  # No embeddings loaded, lazy on demand
    WARMING = "warming"  # Background loading in progress
    WARM = "warm"  # Embeddings ready for fast retrieval
    STALE = "stale"  # Cache needs refresh


@dataclass
class ProfileMetrics:
    """Performance metrics for a model profile.

    Used for reinforcement learning to optimize prompts.
    """

    profile_name: str
    provider: str
    model: str

    # Success metrics
    total_requests: int = 0
    successful_completions: int = 0
    tool_call_success_rate: float = 0.0
    grounding_accuracy: float = 0.0

    # Response quality
    avg_quality_score: float = 0.5
    avg_response_time_ms: float = 0.0
    avg_token_usage: int = 0

    # Tool usage patterns
    avg_tool_calls_per_request: float = 0.0
    tool_budget_adherence: float = 1.0  # 1.0 = stays within budget

    # Mode transition efficiency
    mode_transition_success: float = 0.0
    optimal_tool_budget: int = 10

    # Learned preferences
    prefers_structured_prompts: bool = False
    needs_strict_grounding: bool = True
    supports_parallel_tools: bool = False

    # Temporal tracking
    last_updated: datetime = field(default_factory=datetime.now)

    def update_from_interaction(
        self,
        success: bool,
        quality_score: float,
        response_time_ms: float,
        tool_calls: int,
        tool_budget: int,
        grounded: bool,
    ) -> None:
        """Update metrics from an interaction using exponential moving average."""
        alpha = 0.1
        beta = 1 - alpha  # Pre-calculate for efficiency

        self.total_requests += 1
        if success:
            self.successful_completions += 1

        # Batch EMA updates
        current_rate = self.successful_completions / self.total_requests
        adherence = 1.0 if tool_calls <= tool_budget else tool_budget / tool_calls
        grounding_score = float(grounded)

        self.tool_call_success_rate = beta * self.tool_call_success_rate + alpha * current_rate
        self.avg_quality_score = beta * self.avg_quality_score + alpha * quality_score
        self.avg_response_time_ms = beta * self.avg_response_time_ms + alpha * response_time_ms
        self.avg_tool_calls_per_request = (
            beta * self.avg_tool_calls_per_request + alpha * tool_calls
        )
        self.tool_budget_adherence = beta * self.tool_budget_adherence + alpha * adherence
        self.grounding_accuracy = beta * self.grounding_accuracy + alpha * grounding_score

        # Update derived metrics
        if success and tool_calls > 0:
            self.optimal_tool_budget = int(beta * self.optimal_tool_budget + alpha * tool_calls)
        if success and quality_score > 0.7:
            self.prefers_structured_prompts = tool_calls > 3
        self.needs_strict_grounding = self.grounding_accuracy < 0.8
        self.last_updated = datetime.now()

    def get_recommended_strategy(self) -> PromptStrategy:
        """Get recommended prompt strategy based on learned metrics."""
        if self.grounding_accuracy > 0.9 and self.tool_call_success_rate > 0.9:
            return PromptStrategy.MINIMAL
        elif self.grounding_accuracy > 0.7 and self.tool_call_success_rate > 0.7:
            return PromptStrategy.STRUCTURED
        elif self.total_requests < 10:
            return PromptStrategy.ADAPTIVE  # Not enough data
        else:
            return PromptStrategy.STRICT


@dataclass
class ContextFragment:
    """A fragment of relevant context from conversation history."""

    content: str
    similarity: float
    task_type: str
    was_successful: bool
    timestamp: datetime
    source: str  # "conversation" | "example" | "documentation"

    @property
    def relevance_score(self) -> float:
        """Calculate weighted relevance score."""
        # Combine similarity, success, and recency
        recency_weight = 1.0 / (1.0 + (datetime.now() - self.timestamp).days / 7)
        success_weight = 1.2 if self.was_successful else 0.8
        return self.similarity * success_weight * recency_weight


@dataclass
class PromptContext:
    """Context for prompt generation."""

    task: str
    task_type: str
    profile_name: str
    provider: str
    model: str

    # Historical context
    relevant_fragments: List[ContextFragment] = field(default_factory=list)

    # Tool context
    available_tools: List[str] = field(default_factory=list)
    # The budget the runtime actually enforces this turn, or None if unknown.
    # This is quoted verbatim to the model, so it must never be a guess or a
    # learned average — see _get_tool_guidance().
    recommended_tool_budget: Optional[int] = None

    # Mode context
    current_mode: str = "explore"
    iteration_budget: int = 20
    continuation_context: Optional[str] = None

    # Profile metrics
    profile_metrics: Optional[ProfileMetrics] = None


class ProfileLearningStore:
    """SQLite-backed store for profile learning metrics.

    Persists learned profile behaviors for reinforcement learning.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize the learning store."""
        if db_path is None:
            from victor.config.settings import get_project_paths

            paths = get_project_paths()
            db_path = paths.project_victor_dir / "profile_learning.db"

        self.db_path = db_path
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Ensure database tables exist."""
        if self._initialized:
            return

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS profile_metrics (
                    profile_name TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS interaction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    profile_name TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    quality_score REAL NOT NULL,
                    response_time_ms REAL NOT NULL,
                    tool_calls INTEGER NOT NULL,
                    tool_budget INTEGER NOT NULL,
                    grounded INTEGER NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_interaction_profile
                ON interaction_history(profile_name, timestamp)
            """)

        self._initialized = True

    def save_metrics(self, metrics: ProfileMetrics) -> None:
        """Save profile metrics to database."""
        self._ensure_initialized()

        # Use __dict__ and filter out non-serializable fields for efficiency
        metrics_dict = {
            k: v
            for k, v in metrics.__dict__.items()
            if k not in ("profile_name", "provider", "model", "last_updated")
        }

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO profile_metrics "
                "(profile_name, provider, model, metrics_json, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    metrics.profile_name,
                    metrics.provider,
                    metrics.model,
                    json_dumps(metrics_dict, default=str),
                    datetime.now().isoformat(),
                ),
            )

    def load_metrics(self, profile_name: str, provider: str, model: str) -> ProfileMetrics:
        """Load profile metrics from database."""
        self._ensure_initialized()

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT metrics_json, updated_at FROM profile_metrics
                WHERE profile_name = ?
            """,
                (profile_name,),
            ).fetchone()

            if row:
                metrics_dict = json_loads(row["metrics_json"])
                return ProfileMetrics(
                    profile_name=profile_name,
                    provider=provider,
                    model=model,
                    total_requests=metrics_dict.get("total_requests", 0),
                    successful_completions=metrics_dict.get("successful_completions", 0),
                    tool_call_success_rate=metrics_dict.get("tool_call_success_rate", 0.0),
                    grounding_accuracy=metrics_dict.get("grounding_accuracy", 0.0),
                    avg_quality_score=metrics_dict.get("avg_quality_score", 0.5),
                    avg_response_time_ms=metrics_dict.get("avg_response_time_ms", 0.0),
                    avg_token_usage=metrics_dict.get("avg_token_usage", 0),
                    avg_tool_calls_per_request=metrics_dict.get("avg_tool_calls_per_request", 0.0),
                    tool_budget_adherence=metrics_dict.get("tool_budget_adherence", 1.0),
                    mode_transition_success=metrics_dict.get("mode_transition_success", 0.0),
                    optimal_tool_budget=metrics_dict.get("optimal_tool_budget", 10),
                    prefers_structured_prompts=metrics_dict.get(
                        "prefers_structured_prompts", False
                    ),
                    needs_strict_grounding=metrics_dict.get("needs_strict_grounding", True),
                    supports_parallel_tools=metrics_dict.get("supports_parallel_tools", False),
                    last_updated=datetime.fromisoformat(row["updated_at"]),
                )

        # Return new metrics if not found
        return ProfileMetrics(
            profile_name=profile_name,
            provider=provider,
            model=model,
        )

    def record_interaction(
        self,
        profile_name: str,
        task_type: str,
        success: bool,
        quality_score: float,
        response_time_ms: float,
        tool_calls: int,
        tool_budget: int,
        grounded: bool,
    ) -> None:
        """Record an interaction for learning."""
        self._ensure_initialized()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO interaction_history
                (profile_name, task_type, success, quality_score, response_time_ms,
                 tool_calls, tool_budget, grounded, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    profile_name,
                    task_type,
                    1 if success else 0,
                    quality_score,
                    response_time_ms,
                    tool_calls,
                    tool_budget,
                    1 if grounded else 0,
                    datetime.now().isoformat(),
                ),
            )

    def get_recent_interactions(
        self,
        profile_name: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get recent interactions for a profile."""
        self._ensure_initialized()

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM interaction_history
                WHERE profile_name = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (profile_name, limit),
            ).fetchall()

            return [dict(row) for row in rows]


class EmbeddingScheduler:
    """Manages cold/warm embedding cache for intelligent prompting.

    Strategies:
    - Cold: Lazy load on first access (good for quick tasks)
    - Warm: Background pre-load (good for interactive sessions)
    - On-demand: Targeted loading for specific task types
    """

    def __init__(
        self,
        embedding_store: Optional["ConversationEmbeddingStore"] = None,
        embedding_service: Optional["EmbeddingService"] = None,
    ):
        """Initialize the scheduler."""
        self._store = embedding_store
        self._service = embedding_service
        self._state = CacheState.COLD
        self._last_refresh: Optional[datetime] = None
        self._background_task: Optional[asyncio.Task] = None
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_ttl = timedelta(minutes=30)

    @property
    def state(self) -> CacheState:
        """Get current cache state."""
        if (
            self._state == CacheState.WARM
            and self._last_refresh
            and datetime.now() - self._last_refresh > self._cache_ttl
        ):
            return CacheState.STALE
        return self._state

    async def ensure_warm(self, session_id: Optional[str] = None) -> None:
        """Ensure cache is warm for the session."""
        if self._state == CacheState.WARM:
            return

        if self._state == CacheState.WARMING:
            # Wait for background task
            if self._background_task:
                await self._background_task
            return

        await self._warm_cache(session_id)

    async def _warm_cache(self, session_id: Optional[str] = None) -> None:
        """Load embeddings into warm cache."""
        self._state = CacheState.WARMING

        try:
            if self._store:
                # Trigger lazy embedding if needed
                await self._store._ensure_embeddings(session_id)

            self._state = CacheState.WARM
            self._last_refresh = datetime.now()
            logger.info("[EmbeddingScheduler] Cache warmed successfully")

        except Exception as e:
            logger.warning(f"[EmbeddingScheduler] Cache warming failed: {e}")
            self._state = CacheState.COLD

    def start_background_refresh(self, session_id: Optional[str] = None) -> None:
        """Start background cache refresh."""
        if self._background_task and not self._background_task.done():
            return

        self._background_task = asyncio.create_task(self._background_refresh_loop(session_id))

    async def _background_refresh_loop(self, session_id: Optional[str] = None) -> None:
        """Background loop to keep cache fresh."""
        sleep_duration = self._cache_ttl.total_seconds() / 2
        while True:
            try:
                await asyncio.sleep(sleep_duration)
                await self._warm_cache(session_id)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"[EmbeddingScheduler] Background refresh failed: {e}")

    def stop_background_refresh(self) -> None:
        """Stop background refresh."""
        if self._background_task:
            self._background_task.cancel()
            self._background_task = None

    def invalidate(self) -> None:
        """Invalidate the cache."""
        self._cache.clear()
        self._state = CacheState.COLD
        self._last_refresh = None


class IntelligentPromptBuilder:
    """Intelligent system prompt builder with learning capabilities.

    Uses:
    - Conversation embeddings for relevant context retrieval
    - Profile-specific learning for adaptive prompts
    - Cold/warm cache management for performance
    - Reinforcement signals from user feedback
    """

    # NOTE: this class deliberately keeps no provider-name sets. Prompt strategy is
    # derived from declared capabilities (config/model_capabilities.yaml via
    # ModelCapabilityLoader, plus the provider spec's prompt_caching flag) — see
    # _determine_strategy(). Hand-maintained name lists silently misclassified every
    # provider added after they were written.

    # Grounding rules (thrifty & performant)
    GROUNDING_RULES_MINIMAL = CANONICAL_GROUNDING_RULES

    GROUNDING_RULES_STRICT = CANONICAL_GROUNDING_RULES_EXTENDED

    def __init__(
        self,
        provider_name: str,
        model: str,
        profile_name: str,
        embedding_store: Optional["ConversationEmbeddingStore"] = None,
        embedding_service: Optional["EmbeddingService"] = None,
        learning_store: Optional[ProfileLearningStore] = None,
        retrieval_gateway: Optional[Any] = None,
    ):
        """Initialize the intelligent prompt builder.

        Args:
            provider_name: Provider name (e.g., "ollama", "anthropic")
            model: Model name/identifier
            profile_name: Profile name for learning tracking
            embedding_store: Optional conversation embedding store
            embedding_service: Optional embedding service
            learning_store: Optional profile learning store
            retrieval_gateway: Optional RetrievalGateway for semantic search
        """
        self.provider_name = (str(provider_name) if provider_name else "").lower()
        self.model = str(model) if model else ""
        self.model_lower = self.model.lower()
        self.profile_name = profile_name or f"{self.provider_name}:{self.model}"

        self._embedding_store = embedding_store
        self._embedding_service = embedding_service
        self._retrieval_gateway = retrieval_gateway
        self._learning_store = learning_store or ProfileLearningStore()
        self._scheduler = EmbeddingScheduler(embedding_store, embedding_service)

        # Load profile metrics
        self._metrics = self._learning_store.load_metrics(
            self.profile_name, self.provider_name, self.model
        )

        # Declared tool-calling capabilities, resolved lazily and cached for the
        # lifetime of the builder (a builder is bound to one provider/model pair).
        self._capabilities: Optional["ToolCallingCapabilities"] = None

        # Observers for feedback
        self._observers: List[Callable[[str, float, bool], None]] = []

    @classmethod
    async def create(
        cls,
        provider_name: str,
        model: str,
        profile_name: Optional[str] = None,
    ) -> "IntelligentPromptBuilder":
        """Factory method to create an initialized builder.

        Args:
            provider_name: Provider name
            model: Model name
            profile_name: Optional profile name

        Returns:
            Initialized IntelligentPromptBuilder
        """
        # Get embedding services if available
        embedding_store = None
        embedding_service = None

        try:
            from victor.agent.conversation_embedding_store import (
                get_conversation_embedding_store,
            )
            from victor.storage.embeddings.service import get_embedding_service

            embedding_service = get_embedding_service()
            embedding_store = await get_conversation_embedding_store(embedding_service)
        except ImportError:
            logger.debug("[IntelligentPromptBuilder] Embedding services not available")
        except Exception as e:
            logger.warning(f"[IntelligentPromptBuilder] Failed to init embeddings: {e}")

        return cls(
            provider_name=provider_name,
            model=model,
            profile_name=profile_name or f"{provider_name}:{model}",
            embedding_store=embedding_store,
            embedding_service=embedding_service,
        )

    def add_observer(self, observer: Callable[[str, float, bool], None]) -> None:
        """Add observer for feedback notifications."""
        self._observers.append(observer)

    def remove_observer(self, observer: Callable[[str, float, bool], None]) -> None:
        """Remove observer."""
        if observer in self._observers:
            self._observers.remove(observer)

    async def build(
        self,
        task: str,
        task_type: str = "general",
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        available_tools: Optional[List[str]] = None,
        current_mode: str = "explore",
        tool_budget: Optional[int] = None,
        iteration_budget: int = 20,
        session_id: Optional[str] = None,
        continuation_context: Optional[str] = None,
    ) -> str:
        """Build an intelligent system prompt.

        Args:
            task: Current task/query
            task_type: Detected task type
            conversation_history: Recent conversation messages
            available_tools: List of available tool names
            current_mode: Current agent mode (explore/build/plan)
            tool_budget: Tool call budget the runtime enforces this turn, or None
                if unknown (the prompt then omits any budget claim)
            iteration_budget: Remaining iteration budget
            session_id: Session ID for context retrieval
            continuation_context: Context from previous continuation

        Returns:
            Optimized system prompt
        """
        # Ensure embeddings are ready if in interactive mode
        if self._scheduler.state == CacheState.COLD and session_id:
            # Don't block - warm in background
            self._scheduler.start_background_refresh(session_id)

        # Build context
        context = await self._build_context(
            task=task,
            task_type=task_type,
            conversation_history=conversation_history,
            available_tools=available_tools or [],
            current_mode=current_mode,
            tool_budget=tool_budget,
            iteration_budget=iteration_budget,
            session_id=session_id,
            continuation_context=continuation_context,
        )

        # Determine strategy
        strategy = self._determine_strategy(context)

        # Generate prompt
        prompt = self._generate_prompt(context, strategy)

        logger.debug(
            f"[IntelligentPromptBuilder] Generated {strategy.value} prompt "
            f"for {self.profile_name} ({len(prompt)} chars)"
        )

        return prompt

    async def _build_context(
        self,
        task: str,
        task_type: str,
        conversation_history: Optional[List[Dict[str, Any]]],
        available_tools: List[str],
        current_mode: str,
        tool_budget: Optional[int],
        iteration_budget: int,
        session_id: Optional[str],
        continuation_context: Optional[str],
    ) -> PromptContext:
        """Build context for prompt generation."""
        context = PromptContext(
            task=task,
            task_type=task_type,
            profile_name=self.profile_name,
            provider=self.provider_name,
            model=self.model,
            available_tools=available_tools,
            # Quote the budget the runtime enforces, not the EWMA-learned one.
            # `optimal_tool_budget` remains a learning signal for budget
            # recommendations; it is not a fact about this turn, and stating it as
            # one produced a prompt that claimed "Budget: 10 calls max" while the
            # enforcer allowed 20.
            recommended_tool_budget=tool_budget,
            current_mode=current_mode,
            iteration_budget=iteration_budget,
            continuation_context=continuation_context,
            profile_metrics=self._metrics,
        )

        # Retrieve relevant context fragments
        if self._embedding_store and session_id:
            context.relevant_fragments = await self._retrieve_relevant_context(
                task, task_type, session_id
            )

        return context

    async def _retrieve_relevant_context(
        self,
        task: str,
        task_type: str,
        session_id: str,
        limit: int = 5,
    ) -> List[ContextFragment]:
        """Retrieve relevant context fragments from conversation history."""
        try:
            gateway = self._get_retrieval_gateway()
            if gateway is not None:
                from victor.storage.retrieval.gateway import RetrievalRequest

                request = RetrievalRequest(
                    query=task,
                    session_id=session_id,
                    limit=limit * 2,
                    min_similarity=0.4,
                )
                items = await gateway.search(request)
                fragments = [
                    ContextFragment(
                        content=f"[Previous: {item.message_id}]",
                        similarity=item.score,
                        task_type=task_type,
                        was_successful=True,
                        timestamp=datetime.now(),
                        source="conversation",
                    )
                    for item in items[:limit]
                ]
                return sorted(fragments, key=lambda f: f.relevance_score, reverse=True)

            # Fallback: direct embedding store access when gateway unavailable
            if not self._embedding_store:
                return []
            results = await self._embedding_store.search_similar(
                query=task, session_id=session_id, limit=limit * 2, min_similarity=0.4
            )
            fragments = [
                ContextFragment(
                    content=f"[Previous: {r.message_id}]",
                    similarity=r.similarity,
                    task_type=task_type,
                    was_successful=True,
                    timestamp=r.timestamp or datetime.now(),
                    source="conversation",
                )
                for r in results[:limit]
            ]
            return sorted(fragments, key=lambda f: f.relevance_score, reverse=True)

        except Exception as e:
            logger.warning(f"[IntelligentPromptBuilder] Context retrieval failed: {e}")
            return []

    def _get_retrieval_gateway(self):
        """Get RetrievalGateway for semantic context retrieval (injected via constructor)."""
        return self._retrieval_gateway

    def _determine_strategy(self, context: PromptContext) -> PromptStrategy:
        """Determine the best prompt strategy from declared capabilities.

        Capabilities come from ``config/model_capabilities.yaml`` via
        ``ModelCapabilityLoader`` — the same source the tool-calling adapters use.
        This replaces two hand-maintained name lists (``CLOUD_PROVIDERS`` and a
        17-substring model-name match) that no dual-dialect provider ever appeared
        in: a glm-5.2 session whose every tool call parsed as
        ``native_passthrough`` was classified as a weak local model and handed the
        STRICT "you are a code analyst / plain English only" prompt.
        """
        caps = self._tool_calling_capabilities()

        # Models the catalog flags as needing strict prompting always get it.
        if caps.requires_strict_prompting:
            return PromptStrategy.STRICT

        # Use learned strategy once we have enough evidence for this profile.
        if self._metrics.total_requests >= 10:
            return self._metrics.get_recommended_strategy()

        if caps.native_tool_calls:
            # Hosted providers cache the prefix and need less scaffolding; local
            # runtimes get the fuller structured guidance.
            return (
                PromptStrategy.MINIMAL if self._is_remotely_hosted() else PromptStrategy.STRUCTURED
            )

        return PromptStrategy.STRICT

    def _tool_calling_capabilities(self) -> "ToolCallingCapabilities":
        """Resolve declared tool-calling capabilities for this provider/model."""
        if self._capabilities is None:
            from victor.agent.tool_calling.capabilities import get_model_capabilities

            self._capabilities = get_model_capabilities(self.provider_name, self.model)
        return self._capabilities

    def _is_remotely_hosted(self) -> bool:
        """Whether the provider is a hosted API rather than a local runtime.

        This is the MINIMAL-vs-STRUCTURED axis: hosted providers cache prompt
        prefixes and tolerate the shorter, less scaffolded prompt, while a local
        runtime benefits from the fuller structured guidance.

        Resolution order, most authoritative first:

        1. The OpenAI-compat provider spec's ``prompt_caching`` capability, which
           covers every dual-dialect profile (zai, deepseek, moonshot, xai, ...).
        2. The canonical ``LOCAL_PROVIDERS`` set from ``victor.config.api_keys`` —
           the same one the auth surface uses to decide a provider needs no API
           key. Absence from the OpenAI-compat catalog carries no signal on its
           own (both ``anthropic`` and ``ollama`` are absent), so it must not be
           read as one.
        """
        from victor.config.api_keys import LOCAL_PROVIDERS
        from victor.providers.openai_compat_model_policy import (
            get_openai_compat_provider_spec,
        )

        try:
            spec = get_openai_compat_provider_spec(self.provider_name)
        except Exception:
            # Not in the OpenAI-compat catalog (anthropic, google, ollama, ...).
            # Expected for those providers, so debug rather than warning — but
            # logged, because a silently swallowed lookup is how a provider gets
            # misclassified without anyone noticing.
            logger.debug(
                "[IntelligentPromptBuilder] No OpenAI-compat spec for %r; "
                "falling back to the local-provider set",
                self.provider_name,
                exc_info=True,
            )
        else:
            return bool(spec.capabilities.prompt_caching)

        return self.provider_name not in LOCAL_PROVIDERS

    def _generate_prompt(self, context: PromptContext, strategy: PromptStrategy) -> str:
        """Generate the system prompt based on strategy."""
        # Build parts list efficiently
        parts = [self._get_base_identity(strategy, context.current_mode)]

        # Add optional parts only if they exist
        optional_parts = [
            self._get_task_hint(context.task_type),
            self._get_mode_hint(context.current_mode, context.iteration_budget),
            self._get_tool_guidance(context, strategy),
            (
                f"\nCONTINUATION CONTEXT:\n{context.continuation_context}"
                if context.continuation_context
                else None
            ),
            (
                self._format_context_fragments(context.relevant_fragments)
                if context.relevant_fragments
                else None
            ),
        ]

        parts.extend(part for part in optional_parts if part)
        parts.append(self._get_grounding_rules(strategy, context.profile_metrics))

        return "\n\n".join(parts)

    # Modes in which the agent is expected to change the workspace. In these modes
    # no strategy may present a read-only identity: doing so contradicts the
    # operating mode the user selected, and a well-behaved model resolves that
    # contradiction by refusing to edit.
    _WRITE_CAPABLE_MODES = frozenset({"build", "delegate"})

    def _get_base_identity(self, strategy: PromptStrategy, current_mode: str = "explore") -> str:
        """Get base identity prompt based on strategy and operating mode."""
        writes_allowed = str(current_mode or "").lower() in self._WRITE_CAPABLE_MODES

        if strategy == PromptStrategy.MINIMAL:
            return (
                "You are an expert code analyst with access to tools for exploring "
                "and modifying code. Use them effectively."
            )
        elif strategy == PromptStrategy.STRUCTURED:
            return (
                "You are an expert coding assistant. You can analyze, explain, and generate code.\n"
                "When asked to write or complete code, provide working implementations directly.\n"
                "When asked to explore or analyze code, use the available tools."
            )
        elif writes_allowed:  # STRICT or ADAPTIVE, in a write-capable mode
            return (
                "You are a coding assistant. Follow the rules below EXACTLY.\n"
                "Your job is to help the user understand and modify code — when the "
                "task calls for a change, make it rather than only describing it."
            )
        else:  # STRICT or ADAPTIVE, read-oriented mode
            return (
                "You are a code analyst. Follow the rules below EXACTLY.\n"
                "Your primary job is to help the user understand and modify code."
            )

    def _get_task_hint(self, task_type: str) -> str:
        """Get task-specific hint (concise)."""
        hints = {
            "code_generation": "[GENERATE] Write code directly. Full implementation.",
            "create_simple": "[CREATE] Write file immediately. One tool call max.",
            "create": "[CREATE+CONTEXT] Read relevant files, then create.",
            "edit": "[EDIT] Read target file first, then modify.",
            "search": "[SEARCH] Use code_search/ls. Summarize after 2-4 calls.",
            "action": "[ACTION] Execute git/test/build. Continue until complete.",
            "analysis_deep": "[ANALYSIS] Thorough exploration. Read all modules.",
            "analyze": "[ANALYZE] Examine code carefully. Structured findings.",
            "general_query": (
                "[QUERY] Answer directly. Use tools only if the prompt explicitly "
                "requires external lookup or workspace inspection."
            ),
            "general": "[GENERAL] Moderate exploration. Answer concisely.",
        }
        task_type_str = str(task_type).lower() if task_type else "general"
        return hints.get(task_type_str, "")

    def _get_mode_hint(self, mode: str, iteration_budget: int) -> str:
        """Get mode-specific hint (concise).

        Covers every ``AgentMode`` member. An unrecognised mode is logged rather
        than silently dropped: the previous version knew only explore/build/plan,
        so REVIEW and DELEGATE — and every ``ConversationStage`` name that used to
        be passed in here — produced an empty hint with no trace of why.
        """
        mode_hints = {
            "explore": f"MODE: Explore - Understand code. Budget: {iteration_budget} turns.",
            "build": f"MODE: Build - Implement. Budget: {iteration_budget} turns.",
            "plan": f"MODE: Plan - Draft plan first. Budget: {iteration_budget} turns.",
            "review": f"MODE: Review - Assess and critique. Budget: {iteration_budget} turns.",
            "delegate": f"MODE: Delegate - Dispatch subtasks. Budget: {iteration_budget} turns.",
        }
        mode_str = str(mode).lower() if mode else "explore"
        hint = mode_hints.get(mode_str)
        if hint is None:
            logger.warning(
                "[IntelligentPromptBuilder] Unknown operating mode %r — omitting mode hint. "
                "Expected one of %s (a ConversationStage name here means the mode and the "
                "stage have been conflated again).",
                mode,
                sorted(mode_hints),
            )
            return ""
        return hint

    def _get_tool_guidance(
        self,
        context: PromptContext,
        strategy: PromptStrategy,
    ) -> str:
        """Get tool usage guidance based on strategy (concise)."""
        available_tools = sorted(
            {
                canonicalize_core_tool_name(tool)
                for tool in context.available_tools
                if isinstance(tool, str) and tool
            }
        )
        browse_guidance = ""
        if available_tools:
            browse_guidance = f"\n- Available tools: {', '.join(available_tools[:6])}"

        # Only state a budget the runtime actually enforces. When it is unknown, say
        # nothing rather than quoting a default — a countdown the model can measure as
        # false reads exactly like an injected instruction, and it will (correctly)
        # refuse to act on it.
        budget = context.recommended_tool_budget

        if strategy == PromptStrategy.MINIMAL:
            rules = ["Use for information gathering"]
            if budget:
                rules.append(f"Budget: {budget} calls")
            body = "\n".join(f"- {rule}" for rule in rules)
            return f"TOOLS:\n{body}{browse_guidance}"

        elif strategy == PromptStrategy.STRUCTURED:
            rules = [
                "Use ls/read to inspect code",
                "Call tools sequentially, waiting for results",
            ]
            if budget:
                rules.append(f"Budget: {budget} calls")
            rules.append("Ensure each call provides NEW information")
            body = "\n".join(f"- {rule}" for rule in rules)
            return f"TOOL RULES:\n{body}{browse_guidance}"

        else:  # STRICT
            rules = ["Call tools sequentially; wait for each result."]
            if budget:
                rules.append(f"Budget: {budget} calls max.")
            rules.append("Gather sufficient info before answering.")
            # "Plain English only" is a formatting rule meant to stop weak models
            # emitting raw JSON/XML tool syntax as prose. In a write-capable mode it
            # reads as "do not edit, just describe" and directly contradicts the
            # operating mode, so state the formatting intent without the prohibition.
            if str(context.current_mode or "").lower() in self._WRITE_CAPABLE_MODES:
                rules.append(
                    "Write prose as plain text — never emit raw tool-call syntax as "
                    "your answer. Use the tools to make changes."
                )
            else:
                rules.append("Provide plain English text responses only.")
            rules.append("Ensure calls are unique and purposeful.")
            body = "\n".join(f"{i}. {rule}" for i, rule in enumerate(rules, start=1))
            return f"TOOL RULES:\n{body}{browse_guidance}"

    def _format_context_fragments(self, fragments: List[ContextFragment]) -> str:
        """Format relevant context fragments."""
        if not fragments:
            return ""

        lines = ["RELEVANT CONTEXT (from previous interactions):"]
        for i, frag in enumerate(fragments[:3], 1):
            lines.append(f"{i}. [{frag.source}] (relevance: {frag.relevance_score:.2f})")

        return "\n".join(lines)

    def _get_grounding_rules(
        self,
        strategy: PromptStrategy,
        metrics: Optional[ProfileMetrics],
    ) -> str:
        """Get grounding rules based on strategy and learned needs."""
        minimal_rules, strict_rules = self._get_canonical_grounding_rule_variants()

        # If we've learned this model needs strict grounding
        if metrics and metrics.needs_strict_grounding:
            return strict_rules

        # Otherwise based on strategy
        if strategy in (PromptStrategy.STRICT, PromptStrategy.ADAPTIVE):
            return strict_rules

        return minimal_rules

    @classmethod
    def _get_canonical_grounding_rule_variants(cls) -> Tuple[str, str]:
        """Resolve minimal/strict grounding text from the shared section registry."""
        minimal_rules = cls.GROUNDING_RULES_MINIMAL
        strict_rules = cls.GROUNDING_RULES_STRICT

        try:
            from victor.agent.prompt_section_registry import build_fallback_map

            fallback_map = build_fallback_map(["GROUNDING_RULES", "GROUNDING_RULES_EXTENDED"])
            minimal_rules = fallback_map.get("GROUNDING_RULES") or minimal_rules
            strict_rules = fallback_map.get("GROUNDING_RULES_EXTENDED") or strict_rules
        except Exception:
            logger.debug(
                "[IntelligentPromptBuilder] Falling back to legacy grounding rules",
                exc_info=True,
            )

        return minimal_rules, strict_rules

    def record_feedback(
        self,
        task_type: str,
        success: bool,
        quality_score: float,
        response_time_ms: float,
        tool_calls: int,
        tool_budget: int,
        grounded: bool,
    ) -> None:
        """Record feedback for reinforcement learning.

        Call this after each interaction to improve future prompts.
        """
        # Update metrics
        self._metrics.update_from_interaction(
            success=success,
            quality_score=quality_score,
            response_time_ms=response_time_ms,
            tool_calls=tool_calls,
            tool_budget=tool_budget,
            grounded=grounded,
        )

        # Persist learning
        self._learning_store.save_metrics(self._metrics)
        self._learning_store.record_interaction(
            profile_name=self.profile_name,
            task_type=task_type,
            success=success,
            quality_score=quality_score,
            response_time_ms=response_time_ms,
            tool_calls=tool_calls,
            tool_budget=tool_budget,
            grounded=grounded,
        )

        # Notify observers
        for observer in self._observers:
            try:
                observer(task_type, quality_score, success)
            except Exception as e:
                logger.warning(f"[IntelligentPromptBuilder] Observer error: {e}")

        logger.debug(
            f"[IntelligentPromptBuilder] Recorded feedback: "
            f"success={success}, quality={quality_score:.2f}, "
            f"tools={tool_calls}/{tool_budget}"
        )

    def get_profile_stats(self) -> Dict[str, Any]:
        """Get profile statistics."""
        return {
            "profile_name": self.profile_name,
            "provider": self.provider_name,
            "model": self.model,
            "total_requests": self._metrics.total_requests,
            "success_rate": self._metrics.tool_call_success_rate,
            "avg_quality": self._metrics.avg_quality_score,
            "grounding_accuracy": self._metrics.grounding_accuracy,
            "optimal_tool_budget": self._metrics.optimal_tool_budget,
            "recommended_strategy": self._metrics.get_recommended_strategy().value,
            "cache_state": self._scheduler.state.value,
        }

    def reset_learning(self) -> None:
        """Reset learned profile metrics."""
        self._metrics = ProfileMetrics(
            profile_name=self.profile_name,
            provider=self.provider_name,
            model=self.model,
        )
        self._learning_store.save_metrics(self._metrics)
        logger.info(f"[IntelligentPromptBuilder] Reset learning for {self.profile_name}")


# Convenience function for backward compatibility
async def build_intelligent_prompt(
    provider_name: str,
    model: str,
    task: str,
    task_type: str = "general",
    profile_name: Optional[str] = None,
    **kwargs,
) -> str:
    """Build an intelligent system prompt (convenience function).

    Args:
        provider_name: Provider name
        model: Model name
        task: Current task
        task_type: Task type
        profile_name: Optional profile name
        **kwargs: Additional arguments for build()

    Returns:
        System prompt string
    """
    builder = await IntelligentPromptBuilder.create(
        provider_name=provider_name,
        model=model,
        profile_name=profile_name,
    )
    return await builder.build(task=task, task_type=task_type, **kwargs)
