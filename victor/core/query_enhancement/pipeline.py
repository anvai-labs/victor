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

"""Query Enhancement Pipeline.

Orchestrates multi-step query enhancement by chaining strategies
and enforcing configuration constraints.

Features:
- Configurable strategy chain
- Timeout enforcement with progress indication
- Caching integration
- Graceful degradation to non-LLM strategies
- Metrics collection for RL learning
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import sys
import time
from typing import Dict, Optional, TYPE_CHECKING

from victor.protocols.query_enhancement import (
    EnhancedQuery,
    EnhancementContext,
    EnhancementTechnique,
    IQueryEnhancementStrategy,
    QueryEnhancementConfig,
)
from victor.core.query_enhancement.registry import (
    QueryEnhancementRegistry,
    get_default_registry,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class EnhancementSpinner:
    """Simple console spinner for enhancement progress indication."""

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str = "Enhancing query"):
        self._message = message
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def _spin(self):
        """Animate spinner."""
        frame_idx = 0
        try:
            while self._running:
                frame = self.FRAMES[frame_idx % len(self.FRAMES)]
                # Write to stderr to avoid polluting stdout
                sys.stderr.write(f"\r{frame} {self._message}...")
                sys.stderr.flush()
                frame_idx += 1
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass
        finally:
            # Clear the line
            sys.stderr.write("\r" + " " * (len(self._message) + 10) + "\r")
            sys.stderr.flush()

    def start(self):
        """Start the spinner animation."""
        self._running = True
        self._task = asyncio.create_task(self._spin())

    def stop(self):
        """Stop the spinner animation."""
        self._running = False
        if self._task:
            self._task.cancel()


class QueryEnhancementCache:
    """Simple cache for enhanced queries.

    Uses in-memory dict with TTL-based expiration.
    """

    def __init__(self, ttl_seconds: int = 300, max_size: int = 1000):
        """Initialize cache.

        Args:
            ttl_seconds: Time-to-live in seconds
            max_size: Maximum cache entries
        """
        self._cache: Dict[str, tuple[EnhancedQuery, float]] = {}
        self._ttl = ttl_seconds
        self._max_size = max_size

    def _make_key(self, query: str, context: EnhancementContext) -> str:
        """Create cache key from query and context.

        Args:
            query: Query string
            context: Enhancement context

        Returns:
            Cache key string
        """
        key_data = f"{query}|{context.domain}|{','.join(context.get_entity_names())}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, query: str, context: EnhancementContext) -> Optional[EnhancedQuery]:
        """Get cached enhanced query.

        Args:
            query: Query string
            context: Enhancement context

        Returns:
            Cached EnhancedQuery or None
        """
        key = self._make_key(query, context)
        if key in self._cache:
            result, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl:
                return result
            # Expired
            del self._cache[key]
        return None

    def set(self, query: str, context: EnhancementContext, result: EnhancedQuery) -> None:
        """Cache enhanced query result.

        Args:
            query: Query string
            context: Enhancement context
            result: Enhanced query to cache
        """
        # Evict old entries if cache is full
        if len(self._cache) >= self._max_size:
            self._evict_oldest()

        key = self._make_key(query, context)
        self._cache[key] = (result, time.time())

    def _evict_oldest(self) -> None:
        """Evict oldest 10% of cache entries."""
        if not self._cache:
            return

        entries = sorted(self._cache.items(), key=lambda x: x[1][1])
        evict_count = max(1, len(entries) // 10)

        for key, _ in entries[:evict_count]:
            del self._cache[key]


class QueryEnhancementPipeline:
    """Orchestrates multi-step query enhancement.

    Applies configured techniques in order, merging results.
    Falls back to entity expansion if LLM unavailable and
    fallback_to_expansion is enabled.

    Example:
        pipeline = QueryEnhancementPipeline(config=QueryEnhancementConfig(
            techniques=[EnhancementTechnique.REWRITE, EnhancementTechnique.ENTITY_EXPAND],
            enable_llm=True,
        ))

        result = await pipeline.enhance(
            query="What is AAPL revenue?",
            context=EnhancementContext(domain="financial"),
        )
    """

    def __init__(
        self,
        config: Optional[QueryEnhancementConfig] = None,
        registry: Optional[QueryEnhancementRegistry] = None,
    ):
        """Initialize pipeline.

        Args:
            config: Enhancement configuration
            registry: Strategy registry (uses default if not provided)
        """
        self._config = config or QueryEnhancementConfig()
        self._registry = registry or get_default_registry()
        self._cache = QueryEnhancementCache(ttl_seconds=self._config.cache_ttl_seconds)
        self._llm_available: Optional[bool] = None

    async def enhance(
        self,
        query: str,
        context: EnhancementContext,
        show_progress: bool = False,
    ) -> EnhancedQuery:
        """Apply enhancement pipeline to query.

        Applies configured techniques in order, merging results.
        Falls back to entity expansion if LLM unavailable.

        Args:
            query: Original query
            context: Enhancement context
            show_progress: Show spinner animation during LLM enhancement

        Returns:
            Enhanced query with all transformations applied
        """
        # Check cache first
        cached = self._cache.get(query, context)
        if cached:
            logger.debug(f"Cache hit for query: {query[:50]}...")
            return cached

        # Check LLM availability
        llm_available = await self._check_llm_availability()

        # If LLM not available and fallback enabled, use entity expansion
        if not llm_available and self._config.fallback_to_expansion:
            logger.debug("LLM unavailable, using entity expansion fallback")
            result = await self._apply_entity_expansion(query, context)
            self._cache.set(query, context, result)
            return result

        # Start progress spinner if enabled and LLM will be used
        spinner = None
        if show_progress and llm_available:
            spinner = EnhancementSpinner("Enhancing query with LLM")
            spinner.start()

        try:
            return await self._apply_techniques(query, context, llm_available)
        finally:
            if spinner:
                spinner.stop()

    async def _apply_techniques(
        self,
        query: str,
        context: EnhancementContext,
        llm_available: bool,
    ) -> EnhancedQuery:
        """Apply enhancement techniques in order.

        Args:
            query: Original query
            context: Enhancement context
            llm_available: Whether LLM is available

        Returns:
            Enhanced query result
        """
        # Apply each technique in order
        current_query = query
        all_variants: list[str] = []
        all_sub_queries: list[str] = []
        primary_technique = (
            self._config.techniques[0]
            if self._config.techniques
            else EnhancementTechnique.ENTITY_EXPAND
        )

        for technique in self._config.techniques:
            strategy = self._registry.get(
                technique,
                provider=self._config.provider,
                model=self._config.model,
            )

            if not strategy:
                logger.warning(f"No strategy for {technique.value}")
                continue

            # Skip LLM strategies if LLM unavailable
            if strategy.requires_llm and not llm_available:
                logger.debug(f"Skipping {technique.value} - LLM unavailable")
                continue

            try:
                timeout = self._config.max_enhancement_time_ms / 1000.0
                result = await asyncio.wait_for(
                    strategy.enhance(current_query, context),
                    timeout=timeout,
                )

                # Accumulate results
                current_query = result.enhanced
                all_variants.extend(result.variants)
                all_sub_queries.extend(result.sub_queries)

                logger.debug(
                    f"Applied {technique.value}: {query[:30]}... -> {current_query[:30]}..."
                )

            except asyncio.TimeoutError:
                logger.warning(f"Enhancement timeout for {technique.value}")
                break
            except Exception as e:
                logger.warning(f"Enhancement error for {technique.value}: {e}")
                continue

        # Build final result
        enhanced = EnhancedQuery(
            original=query,
            enhanced=current_query,
            technique=primary_technique,
            variants=list(dict.fromkeys(all_variants)),  # Remove duplicates
            sub_queries=list(dict.fromkeys(all_sub_queries)),
            confidence=0.9 if llm_available else 0.7,
            metadata={
                "domain": context.domain,
                "techniques_applied": [t.value for t in self._config.techniques],
                "llm_available": llm_available,
            },
        )

        # Cache result
        self._cache.set(query, context, enhanced)

        return enhanced

    async def _check_llm_availability(self) -> bool:
        """Check if LLM is available for enhancement.

        Uses cached result after first check.

        Returns:
            True if LLM is available
        """
        if not self._config.enable_llm:
            return False

        if self._llm_available is not None:
            return self._llm_available

        try:
            from victor.config.settings import load_settings
            from victor.providers.registry import ProviderRegistry

            settings = load_settings()

            # Check if in air-gapped mode
            if getattr(settings, "airgapped_mode", False):
                self._llm_available = False
                return False

            # Try to get provider
            provider_name = self._config.provider or settings.provider.default_provider
            if provider_name:
                ProviderRegistry.get(provider_name)
                self._llm_available = True
                return True

            self._llm_available = False
            return False

        except Exception as e:
            logger.debug(f"LLM availability check failed: {e}")
            self._llm_available = False
            return False

    async def _apply_entity_expansion(
        self, query: str, context: EnhancementContext
    ) -> EnhancedQuery:
        """Apply entity expansion fallback.

        Args:
            query: Original query
            context: Enhancement context

        Returns:
            Enhanced query with entity expansion
        """
        strategy = self._registry.get(EnhancementTechnique.ENTITY_EXPAND)

        if strategy:
            return await strategy.enhance(query, context)

        # No expansion strategy available
        return EnhancedQuery(
            original=query,
            enhanced=query,
            technique=EnhancementTechnique.ENTITY_EXPAND,
            confidence=0.5,
            metadata={"domain": context.domain, "no_enhancement": True},
        )

    def is_llm_available(self) -> bool:
        """Check if LLM is available (cached check).

        Returns:
            True if LLM available, False otherwise
        """
        return self._llm_available or False

    def __repr__(self) -> str:
        techniques = [t.value for t in self._config.techniques]
        return f"QueryEnhancementPipeline(techniques={techniques}, enable_llm={self._config.enable_llm})"
