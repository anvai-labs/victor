from __future__ import annotations

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

"""Centralized default configuration values.

This module consolidates hardcoded default values that were previously
scattered throughout the codebase. Having them in one place makes them
easier to find, modify, and document.

Design Principles:
    - Single source of truth for default values
    - Easy to override via Settings
    - Well-documented with context for each value
    - Grouped by functional area

Usage:
    from victor.config.defaults import (
        ContextDefaults,
        TokenDefaults,
        ToolDefaults,
        RetryDefaults,
    )

    # Use in code
    max_chars = ContextDefaults.MAX_CONTEXT_CHARS
    budget = ToolDefaults.TOOL_BUDGET
"""

from dataclasses import dataclass
from typing import Final

# =============================================================================
# Context and Memory Defaults
# =============================================================================


@dataclass(frozen=True)
class ContextDefaults:
    """Default values for context window and memory management.

    These values control how much context can be passed to the LLM and
    how memory is managed across conversations.
    """

    # Maximum characters for context overflow detection
    # Based on typical context windows: 200k chars ~ 50k tokens
    MAX_CONTEXT_CHARS: Final[int] = 200_000

    # Context window size threshold for "large context" models
    # Models with >= 32768 tokens get different handling
    LARGE_CONTEXT_THRESHOLD: Final[int] = 32_768

    # Reserve tokens for response generation
    # Ensures model has room to generate output
    RESPONSE_TOKEN_RESERVE: Final[int] = 4_096

    # Maximum conversation turns to keep in memory
    MAX_CONVERSATION_TURNS: Final[int] = 100

    # Conversation compaction threshold (% of context used)
    COMPACTION_THRESHOLD: Final[float] = 0.85


# =============================================================================
# Token Budget Defaults
# =============================================================================


@dataclass(frozen=True)
class TokenDefaults:
    """Default values for token budgets and limits.

    These values control how many tokens can be used for various operations.
    """

    # Default max tokens for chat completions
    MAX_TOKENS: Final[int] = 4_096

    # Extended thinking budget for complex reasoning
    THINKING_BUDGET_TOKENS: Final[int] = 10_000

    # Maximum iterations for chat loop
    CHAT_MAX_ITERATIONS: Final[int] = 10

    # Minimum tokens to reserve for tool responses
    TOOL_RESPONSE_RESERVE: Final[int] = 1_000

    # Default streaming chunk size
    STREAM_CHUNK_SIZE: Final[int] = 256


# =============================================================================
# Tool Execution Defaults
# =============================================================================


@dataclass(frozen=True)
class ToolDefaults:
    """Default values for tool execution and management.

    These values control tool budgets, concurrency, and caching.
    """

    # Maximum tool calls per session
    TOOL_BUDGET: Final[int] = 25

    # Maximum concurrent tool executions
    MAX_CONCURRENT_TOOLS: Final[int] = 5

    # Batch size for parallel tool execution
    PARALLEL_BATCH_SIZE: Final[int] = 10

    # Timeout for individual tool execution (seconds)
    PARALLEL_TIMEOUT_PER_TOOL: Final[float] = 60.0

    # Maximum entries in idempotent tool result cache
    IDEMPOTENT_CACHE_MAX_SIZE: Final[int] = 50  # Reduced from 100

    # TTL for cached tool results (seconds)
    IDEMPOTENT_CACHE_TTL: Final[float] = 300.0  # 5 minutes

    # Maximum tools to return from selection
    TOOL_SELECTION_LIMIT: Final[int] = 10

    # Minimum similarity score for tool selection
    TOOL_SELECTION_MIN_SCORE: Final[float] = 0.3

    # Default tool cost tier budget (LOW tier)
    DEFAULT_COST_TIER_BUDGET: Final[int] = 100


# =============================================================================
# Retry and Resilience Defaults
# =============================================================================


@dataclass(frozen=True)
class RetryDefaults:
    """Default values for retry logic and resilience.

    These values control how failures are handled and retried.
    """

    # Maximum retries for rate-limited requests
    MAX_RATE_LIMIT_RETRIES: Final[int] = 3

    # Maximum attempts for tool execution
    TOOL_RETRY_MAX_ATTEMPTS: Final[int] = 3

    # Base delay for exponential backoff (seconds)
    TOOL_RETRY_BASE_DELAY: Final[float] = 1.0

    # Maximum delay cap for exponential backoff (seconds)
    TOOL_RETRY_MAX_DELAY: Final[float] = 30.0

    # Jitter factor for backoff randomization
    RETRY_JITTER_FACTOR: Final[float] = 0.1

    # Circuit breaker failure threshold
    CIRCUIT_BREAKER_THRESHOLD: Final[int] = 5

    # Circuit breaker reset timeout (seconds)
    CIRCUIT_BREAKER_TIMEOUT: Final[float] = 60.0


# =============================================================================
# Cache Defaults
# =============================================================================


@dataclass(frozen=True)
class CacheDefaults:
    """Default values for various caches throughout the system.

    These values control cache sizes and TTLs.
    """

    # Embedding cache memory limit (bytes)
    EMBEDDING_CACHE_MAX_MEMORY: Final[int] = 50 * 1024 * 1024  # 50MB

    # Entry point cache TTL (seconds)
    ENTRY_POINT_CACHE_TTL: Final[float] = 3600.0  # 1 hour

    # Workflow definition cache TTL (seconds)
    WORKFLOW_CACHE_TTL: Final[int] = 3600  # 1 hour

    # Maximum workflow cache entries
    WORKFLOW_CACHE_MAX_ENTRIES: Final[int] = 500

    # Tool metadata cache TTL (seconds)
    TOOL_METADATA_CACHE_TTL: Final[float] = 600.0  # 10 minutes


# =============================================================================
# Provider Defaults
# =============================================================================


@dataclass(frozen=True)
class ProviderDefaults:
    """Default values for LLM provider configuration.

    These values control provider behavior and health checks.
    """

    # Enable provider health checks
    HEALTH_CHECKS_ENABLED: Final[bool] = True

    # Health check interval (seconds)
    HEALTH_CHECK_INTERVAL: Final[float] = 60.0

    # Provider request timeout (seconds)
    REQUEST_TIMEOUT: Final[float] = 120.0

    # Maximum concurrent requests per provider
    MAX_CONCURRENT_REQUESTS: Final[int] = 10

    # Default temperature for sampling
    DEFAULT_TEMPERATURE: Final[float] = 0.7

    # Default top_p for nucleus sampling
    DEFAULT_TOP_P: Final[float] = 0.95


# =============================================================================
# Workflow Defaults
# =============================================================================


@dataclass(frozen=True)
class WorkflowDefaults:
    """Default values for workflow execution.

    These values control workflow behavior and limits.
    """

    # Maximum iterations to prevent infinite loops
    MAX_ITERATIONS: Final[int] = 25

    # Default execution timeout (seconds, None = no timeout)
    EXECUTION_TIMEOUT: Final[float | None] = None

    # Enable checkpointing for workflow state
    ENABLE_CHECKPOINTING: Final[bool] = True

    # Maximum parallel branches
    MAX_PARALLEL_BRANCHES: Final[int] = 10

    # Default join strategy for parallel nodes
    DEFAULT_JOIN_STRATEGY: Final[str] = "all"


# =============================================================================
# Convenience Functions
# =============================================================================


def get_default(settings: object, attr: str, default_class: type, attr_name: str) -> any:
    """Get a configuration value with fallback to defaults.

    This helper function provides a clean way to access settings with
    fallback to centralized defaults.

    Args:
        settings: Settings object to check first
        attr: Attribute name on settings object
        default_class: Default class (e.g., TokenDefaults)
        attr_name: Attribute name on default class

    Returns:
        Value from settings if present, otherwise default

    Example:
        max_tokens = get_default(
            settings, "max_tokens",
            TokenDefaults, "MAX_TOKENS"
        )
    """
    return getattr(settings, attr, getattr(default_class, attr_name))


__all__ = [
    "ContextDefaults",
    "TokenDefaults",
    "ToolDefaults",
    "RetryDefaults",
    "CacheDefaults",
    "ProviderDefaults",
    "WorkflowDefaults",
    "get_default",
]
