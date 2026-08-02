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

"""Tool selector strategy factory.

This module provides the factory for creating tool selector strategies
as part of HIGH-002: Unified Tool Selection Architecture - Release 2, Phase 4.

Supports three strategies:
- keyword: Fast registry-based keyword matching (<1ms)
- semantic: ML-based embedding similarity (10-50ms)
- hybrid: Blends both approaches (Release 3)
- auto: Automatic selection based on environment
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Literal, Optional, Set

from victor.agent.protocols import IToolSelector
from victor.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from victor.agent.conversation.state_machine import ConversationStateMachine
    from victor.config.settings import Settings
    from victor.storage.embeddings.service import EmbeddingService

logger = logging.getLogger(__name__)


# =============================================================================
# Enums for Tool Selection Strategies
# =============================================================================


class AutoSelectStrategy(str, Enum):
    """Automatic tool selection strategy.

    Strategies for automatically selecting the best tool selector
    based on environment capabilities.

    Selection logic:
    1. If airgapped_mode → KEYWORD (no embeddings available)
    2. If embedding service available → SEMANTIC (best quality)
    3. Fallback → KEYWORD (always works)
    """

    KEYWORD = "keyword"  # Fast registry-based keyword matching (<1ms)
    SEMANTIC = "semantic"  # ML-based embedding similarity (10-50ms)


# Backward compatibility
_AutoSelectStrategyLiteral = Literal["keyword", "semantic"]


def create_tool_selector_strategy(
    strategy: str,
    tools: ToolRegistry,
    conversation_state: Optional["ConversationStateMachine"] = None,
    model: str = "",
    provider_name: str = "",
    enabled_tools: Optional[Set[str]] = None,
    embedding_service: Optional["EmbeddingService"] = None,
    settings: Optional["Settings"] = None,
) -> IToolSelector:
    """Create tool selector based on strategy.

    Args:
        strategy: Strategy name: "auto", "keyword", "semantic", or "hybrid"
        tools: Tool registry with all available tools
        conversation_state: Optional conversation state machine
        model: Model name for capability detection
        provider_name: Provider name for capability detection
        enabled_tools: Optional vertical-specific tool filter
        embedding_service: Optional embedding service for semantic selection
        settings: Optional settings for auto-selection

    Returns:
        IToolSelector implementation

    Raises:
        ValueError: If strategy is unknown or requirements not met

    Examples:
        >>> # Keyword selection (fast, no embeddings)
        >>> selector = create_tool_selector_strategy(
        ...     strategy="keyword",
        ...     tools=tool_registry,
        ...     model="gpt-4",
        ...     provider_name="openai",
        ... )

        >>> # Semantic selection (requires embedding service)
        >>> selector = create_tool_selector_strategy(
        ...     strategy="semantic",
        ...     tools=tool_registry,
        ...     embedding_service=embedding_service,
        ... )

        >>> # Auto-selection (based on environment)
        >>> selector = create_tool_selector_strategy(
        ...     strategy="auto",
        ...     tools=tool_registry,
        ...     settings=settings,
        ...     embedding_service=embedding_service,
        ... )
    """
    # Auto-selection logic
    if strategy == "auto":
        strategy = _auto_select_strategy(
            settings=settings,
            provider_name=provider_name,
            embedding_service=embedding_service,
        )
        logger.info(f"Auto-selected tool selection strategy: {strategy}")

    # Create strategy implementation
    if strategy == "semantic":
        return _create_semantic_selector(
            tools=tools,
            conversation_state=conversation_state,
            model=model,
            provider_name=provider_name,
            enabled_tools=enabled_tools,
            embedding_service=embedding_service,
        )

    elif strategy == "keyword":
        return _create_keyword_selector(
            tools=tools,
            conversation_state=conversation_state,
            model=model,
            provider_name=provider_name,
            enabled_tools=enabled_tools,
        )

    elif strategy == "hybrid":
        return _create_hybrid_selector(
            tools=tools,
            conversation_state=conversation_state,
            model=model,
            provider_name=provider_name,
            enabled_tools=enabled_tools,
            embedding_service=embedding_service,
            settings=settings,
        )

    else:
        raise ValueError(
            f"Unknown tool selection strategy: {strategy}. "
            f"Must be one of: 'auto', 'keyword', 'semantic', 'hybrid'"
        )


def _auto_select_strategy(
    settings: Optional["Settings"],
    provider_name: str,
    embedding_service: Optional["EmbeddingService"],
) -> AutoSelectStrategy:
    """Automatically select best strategy based on environment.

    Selection logic:
    1. If airgapped_mode → KEYWORD (no embeddings available)
    2. If embedding service available → SEMANTIC (best quality)
    3. Fallback → KEYWORD (always works)

    Args:
        settings: Optional settings for configuration
        provider_name: Provider name
        embedding_service: Optional embedding service

    Returns:
        AutoSelectStrategy enum value
    """
    # Check air-gapped mode
    if settings and settings.security.airgapped_mode:
        logger.info("Air-gapped mode detected: using keyword strategy")
        return AutoSelectStrategy.KEYWORD

    # Prefer semantic if embedding service available
    if embedding_service is not None:
        logger.info("Embedding service available: using semantic strategy")
        return AutoSelectStrategy.SEMANTIC

    # Fallback to keyword (always works, no dependencies)
    logger.info("No embedding service: using keyword strategy")
    return AutoSelectStrategy.KEYWORD


def _create_semantic_selector(
    tools: ToolRegistry,
    conversation_state: Optional["ConversationStateMachine"],
    model: str,
    provider_name: str,
    enabled_tools: Optional[Set[str]],
    embedding_service: Optional["EmbeddingService"],
) -> IToolSelector:
    """Create semantic tool selector.

    Args:
        tools: Tool registry
        conversation_state: Optional conversation state machine
        model: Model name
        provider_name: Provider name
        enabled_tools: Optional vertical filter
        embedding_service: Embedding service (required)

    Returns:
        SemanticToolSelector instance

    Raises:
        ValueError: If embedding_service is None
    """
    if embedding_service is None:
        raise ValueError(
            "Semantic tool selector requires embedding_service. "
            "Either provide embedding_service or use 'keyword' strategy."
        )

    from victor.tools.semantic_selector import SemanticToolSelector

    # Note: SemanticToolSelector doesn't fully implement IToolSelector's signature yet
    # It uses its own select_relevant_tools_with_context() method
    # This will be addressed in Release 2 Phase 5
    return SemanticToolSelector(
        embedding_service=embedding_service,
        cache_embeddings=True,  # Enable caching for performance
    )


def _create_keyword_selector(
    tools: ToolRegistry,
    conversation_state: Optional["ConversationStateMachine"],
    model: str,
    provider_name: str,
    enabled_tools: Optional[Set[str]],
) -> IToolSelector:
    """Create keyword tool selector.

    Args:
        tools: Tool registry
        conversation_state: Optional conversation state machine
        model: Model name
        provider_name: Provider name
        enabled_tools: Optional vertical filter

    Returns:
        KeywordToolSelector instance
    """
    from victor.tools.keyword_tool_selector import KeywordToolSelector

    return KeywordToolSelector(
        tools=tools,
        conversation_state=conversation_state,
        model=model,
        provider_name=provider_name,
        enabled_tools=enabled_tools,
    )


def _create_hybrid_selector(
    tools: ToolRegistry,
    conversation_state: Optional["ConversationStateMachine"],
    model: str,
    provider_name: str,
    enabled_tools: Optional[Set[str]],
    embedding_service: Optional["EmbeddingService"],
    settings: Optional["Settings"],
) -> IToolSelector:
    """Create hybrid tool selector (blends semantic + keyword).

    Args:
        tools: Tool registry
        conversation_state: Optional conversation state machine
        model: Model name
        provider_name: Provider name
        enabled_tools: Optional vertical filter
        embedding_service: Embedding service (required)
        settings: Optional settings for configuration

    Returns:
        HybridToolSelector instance

    Raises:
        ValueError: If embedding_service is None
    """
    if embedding_service is None:
        raise ValueError(
            "Hybrid tool selector requires embedding_service. "
            "Either provide embedding_service or use 'keyword' strategy."
        )

    from victor.tools.hybrid_tool_selector import (
        HybridSelectorConfig,
        HybridToolSelector,
    )

    # Create both semantic and keyword selectors
    semantic_selector = _create_semantic_selector(
        tools=tools,
        conversation_state=conversation_state,
        model=model,
        provider_name=provider_name,
        enabled_tools=enabled_tools,
        embedding_service=embedding_service,
    )

    keyword_selector = _create_keyword_selector(
        tools=tools,
        conversation_state=conversation_state,
        model=model,
        provider_name=provider_name,
        enabled_tools=enabled_tools,
    )

    # Create hybrid config with RL settings from global Settings
    enable_rl = True  # Default: enabled for all users
    rl_boost_weight = 0.15  # Default: conservative 15% influence

    if settings:
        enable_rl = getattr(settings, "enable_tool_selection_rl", True)
        rl_boost_weight = getattr(settings, "tool_selection_rl_boost_weight", 0.15)

    config = HybridSelectorConfig(
        semantic_weight=0.7,
        keyword_weight=0.3,
        min_semantic_tools=3,
        min_keyword_tools=2,
        max_total_tools=15,
        enable_rl=enable_rl,
        rl_boost_weight=rl_boost_weight,
    )

    return HybridToolSelector(
        semantic_selector=semantic_selector,
        keyword_selector=keyword_selector,
        config=config,
    )
