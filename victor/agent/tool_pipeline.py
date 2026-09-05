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

"""Tool Execution Pipeline - Coordinates tool execution flow.

This module extracts tool execution coordination from AgentOrchestrator:
- Tool call validation
- Argument normalization
- Execution via ToolExecutor
- Result tracking and analytics
- Failed signature tracking to avoid loops

Design Principles:
- Single Responsibility: Coordinates tool execution only
- Composable: Works with existing ToolExecutor and ToolRegistry
- Observable: Emits events for monitoring
- Configurable: Budget, retry, and caching settings
"""

from __future__ import annotations

import asyncio
import ast
from victor.core.json_utils import json_dumps
import logging
import os
from pathlib import Path
import re
import threading
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, TYPE_CHECKING

from victor.agent.argument_normalizer import ArgumentNormalizer, NormalizationStrategy
from victor.agent.tool_executor import ToolExecutor, ToolExecutionResult
from victor.agent.parameter_enforcer import (
    get_enforcer_for_tool,
    ParameterInferenceError,
    ParameterValidationError,
)
from victor.agent.output_aggregator import (
    OutputAggregator,
    AggregationState,
)
from victor.core.errors import ErrorInfo, ErrorCategory, ErrorSeverity
from victor.agent.synthesis_checkpoint import (
    CompositeSynthesisCheckpoint,
    get_checkpoint_for_complexity,
)
from victor.agent.file_state import capture_file_state, normalize_file_path
from victor.config.tool_selection_defaults import (
    ToolPipelineDefaults,
    IdempotentTools,
)
from victor.tools.core_tool_aliases import (
    canonicalize_core_tool_name,
    normalize_model_tool_name,
)
from victor.agent.safety import get_write_tool_names

# Import native compute_signature for 10-20x faster signature generation
try:
    from victor.processing.native import compute_signature as native_compute_signature

    _NATIVE_SIGNATURE_AVAILABLE = True
except ImportError:
    _NATIVE_SIGNATURE_AVAILABLE = False
    native_compute_signature = None  # type: ignore

if TYPE_CHECKING:
    from victor.agent.search_router import SearchRouter
    from victor.tools.registry import ToolRegistry
    from victor.storage.cache.tool_cache import ToolCache
    from victor.agent.signature_store import SignatureStore
    from victor.agent.middleware_chain import MiddlewareChain
    from victor.agent.tool_result_cache import ToolResultCache
    from victor.security.permissions import PermissionPolicy

logger = logging.getLogger(__name__)


@dataclass
class ToolPipelineConfig:
    """Configuration for tool pipeline.

    Uses centralized defaults from ToolPipelineDefaults.
    """

    tool_budget: int = ToolPipelineDefaults.TOOL_BUDGET
    enable_caching: bool = True
    enable_analytics: bool = True
    enable_failed_signature_tracking: bool = True
    max_tool_name_length: int = 64

    # Per-tool-call timeout for serial execution path
    # Increased from 30 to 60 seconds for semantic search (code_search, etc.)
    per_tool_timeout_seconds: float = 60.0

    # Parallel execution settings (co-design review item 16).
    # Default OFF: the read-only allowlist gate (see _is_parallel_eligible)
    # makes this sound to enable, but it still changes execution
    # ordering/timing for a broad swath of tool calls, so it ships opt-in.
    parallel_tool_execution: bool = False
    max_concurrent_tools: int = ToolPipelineDefaults.MAX_CONCURRENT_TOOLS

    # Session-level result caching for idempotent tools
    # This prevents re-reading same files during a session (helps DeepSeek, Ollama)
    enable_idempotent_caching: bool = True
    idempotent_cache_max_size: int = ToolPipelineDefaults.IDEMPOTENT_CACHE_MAX_SIZE

    # Batch-level deduplication for providers that send duplicate tool calls
    # (helps DeepSeek, Ollama which sometimes issue identical calls in same batch)
    enable_batch_deduplication: bool = True

    # Output aggregation and synthesis checkpoint settings
    enable_output_aggregation: bool = True
    enable_synthesis_checkpoints: bool = True
    synthesis_checkpoint_complexity: str = "medium"  # simple, medium, complex

    # Semantic result caching with FAISS (mtime-based invalidation)
    # Enables intelligent caching based on semantic similarity, reducing redundant tool calls
    enable_semantic_caching: bool = True

    # Cross-turn dedup: cache results for "effectively idempotent" tools
    # (web_search, grep_search, http_request, git) across turns within a session.
    enable_cross_turn_dedup: bool = True
    cross_turn_dedup_ttl: float = 300.0  # seconds

    # Navigation hint loop detection settings
    # Prevents infinite loops when navigation hints keep rewriting to same region
    navigation_hint_max_uses: int = 3
    navigation_hint_ttl_seconds: float = 600.0
    navigation_hint_loop_threshold: int = 3


# Tools that are safe to cache (read-only, deterministic for same arguments)
# Include both lowercase and capitalized variants for tool name normalization
# Extended from centralized IdempotentTools with additional pipeline-specific variants
IDEMPOTENT_TOOLS = IdempotentTools.IDEMPOTENT_TOOLS | frozenset(
    {
        # Additional capitalized variants and aliases for tool name normalization
        "Read",
        "Grep",
        "Graph",
        "Glob",
        "code_search",  # Semantic code search alias
        "semantic_code_search",
        "refs",  # Reference lookup
    }
)

# Cross-turn dedup: tools that are "effectively idempotent" within a session
# window even though they may be classified as non-idempotent. These produce
# the same result for identical args within a short time window (e.g., 5 min),
# making re-execution wasteful. Separate from IDEMPOTENT_TOOLS to allow
# a shorter TTL and explicit opt-in.
CROSS_TURN_DEDUP_TOOLS = frozenset(
    {
        "web_search",
        "web_fetch",
        "http_request",
        "grep_search",
        "plan_files",
        "git",  # git status/log/diff are read-only
    }
)

# Candidate names for parallel-eligible dispatch (co-design review item 16):
# the filesystem-read/search/grep/ls/web-search class. This is a CANDIDATE
# set only — _compute_parallel_eligible_tools() below intersects it against
# the write-tools complement before it becomes the actual allowlist, so a
# name here that's also independently classified as a write tool (e.g.
# "git", present in both this set's spirit and _STATIC_WRITE_TOOL_NAMES) is
# still excluded.
#
# Eligibility is checked against the LOWERCASE-normalized name (see
# _is_parallel_eligible / _normalize_valid_tool_name), so a capitalized-only
# entry can never match — "graph" (lowercase) is added explicitly here
# rather than relying on IDEMPOTENT_TOOLS's "Graph" alias, which is
# normalized away before the lookup and would otherwise silently make the
# graph code-query tool never parallel-eligible.
_PARALLEL_CANDIDATE_TOOLS = IDEMPOTENT_TOOLS | frozenset({"web_search", "web_fetch", "graph"})

# Tool names NEVER eligible for parallel dispatch regardless of classification.
# Matches action_authorizer.SHELL_TOOL_ALIASES: an arbitrary shell command's
# side effects can't be statically verified, so no read-only classification
# (including should_allow_shell_for_read_only_intent's message-based one) is
# trusted here.
_PARALLEL_INELIGIBLE_TOOLS = frozenset({"shell", "execute_bash", "bash"})


def _compute_parallel_eligible_tools() -> frozenset:
    """Read-only allowlist for parallel tool dispatch.

    Deliberately the INVERSE of the authorizer's blocked-set check (unsound
    for this purpose: unknown tools there classify READ_ONLY, which is fine
    for a permission gate but wrong for a parallelism gate). Built as an
    INTERSECTION of two independently-maintained sources so an unknown or
    ambiguously-classified tool defaults to sequential, safer execution:
      - _PARALLEL_CANDIDATE_TOOLS: the existing read-only/deterministic
        classification already used for idempotent-result caching, plus
        the web-search class (network reads, safe to run concurrently but
        not idempotent-cacheable across turns the same way).
      - The complement of get_write_tool_names() (registry-driven + static
        fallback), recomputed on every call rather than cached at module
        import time since the registry can still be growing as verticals
        finish loading.
    Computed fresh (not memoized) for the same reason — correctness over a
    cheap set-membership call's negligible cost.
    """
    write_tools = get_write_tool_names()
    return frozenset(
        name
        for name in _PARALLEL_CANDIDATE_TOOLS
        if name not in write_tools and name.lower() not in _PARALLEL_INELIGIBLE_TOOLS
    )


GRAPH_TOOL_ARGUMENTS = frozenset(
    {
        "mode",
        "node",
        "source",
        "target",
        "query",
        "file",
        "node_type",
        "edge_types",
        "depth",
        "top_k",
        "direction",
        "expand",
        "exclude_paths",
        "only_runtime",
        "runtime_weighted",
        "modules_only",
        "include_callsites",
        "max_callsites",
        "structured",
        "include_symbols",
        "include_modules",
        "include_calls",
        "include_refs",
    }
)

CODE_INTELLIGENCE_FIRST_MODES = frozenset({"plan", "build", "review", "delegate", "explore"})
CODE_FILE_EXTENSIONS = frozenset(
    {
        ".py",
        ".pyi",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".java",
        ".go",
        ".rs",
        ".c",
        ".cc",
        ".cpp",
        ".cxx",
        ".h",
        ".hpp",
        ".cs",
        ".rb",
        ".php",
        ".swift",
        ".kt",
        ".scala",
    }
)
TARGETED_READ_LINE_CONTEXT = 40
TARGETED_READ_LINE_LIMIT = 120


class ToolRateLimiter:
    """Token bucket rate limiter for tool execution.

    Implements a token bucket algorithm to limit the rate of tool executions.
    This prevents overwhelming external services or local resources.

    Attributes:
        rate: Tokens per second to add (refill rate)
        burst: Maximum tokens that can accumulate (bucket size)
        tokens: Current available tokens
        last_update: Timestamp of last token refill

    Example:
        limiter = ToolRateLimiter(rate=10.0, burst=5)

        # Synchronous check
        if limiter.acquire():
            execute_tool()

        # Asynchronous wait
        await limiter.wait()
        execute_tool()
    """

    def __init__(self, rate: float = 10.0, burst: int = 5):
        """Initialize the rate limiter.

        Args:
            rate: Tokens per second (refill rate)
            burst: Maximum tokens (bucket capacity)
        """
        self.rate = rate
        self.burst = burst
        self.tokens = float(burst)  # Start with full bucket
        self.last_update = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> bool:
        """Try to acquire a token without blocking.

        Returns:
            True if token acquired, False if no tokens available.
        """
        with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

    async def wait(self) -> None:
        """Wait until a token is available.

        This method blocks asynchronously until a token can be acquired.
        """
        while not self.acquire():
            # Sleep for time needed to get one token
            await asyncio.sleep(1.0 / self.rate)


class LRUToolCache:
    """LRU (Least Recently Used) cache with TTL for idempotent tool results.

    Provides efficient caching with LRU eviction policy and time-based expiry.
    This prevents stale results from being returned after files change and
    bounds memory growth for long-running sessions.

    Attributes:
        _cache: OrderedDict maintaining insertion/access order
        _timestamps: Dict mapping keys to creation timestamps
        _max_size: Maximum number of entries to cache (default: 50)
        _ttl_seconds: Time-to-live in seconds (default: 300s = 5 minutes)

    Example:
        cache = LRUToolCache(max_size=50, ttl_seconds=300)
        cache.set("key", result)
        result = cache.get("key")  # Returns cached result if not expired
    """

    def __init__(
        self,
        max_size: int = ToolPipelineDefaults.IDEMPOTENT_CACHE_MAX_SIZE,
        ttl_seconds: float = ToolPipelineDefaults.IDEMPOTENT_CACHE_TTL,
    ):
        """Initialize the LRU cache with TTL.

        Args:
            max_size: Maximum number of entries to cache
            ttl_seconds: Time-to-live in seconds
        """
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        # Co-design review item 16: concurrent tool dispatch means multiple
        # coroutines can now call get()/set() around the SAME cache instance
        # without an intervening await inside either method, so CPython's
        # GIL already makes each individual call atomic — no torn
        # OrderedDict state is possible. This lock exists for a caller that
        # needs an atomic read-check-then-write span ACROSS an await
        # boundary (e.g. "miss the cache, run the tool, then populate it"
        # without a duplicate concurrent run for the same key).
        #
        # NOTE this is NOT guaranteed today by upstream batch deduplication:
        # deduplicate_tool_calls() signatures raw, pre-normalization
        # arguments, while get_cached_result()/cache_result() signature
        # fully-normalized arguments (post parameter-enforcement defaults,
        # post search-routing rewrite) — two calls with different raw args
        # can still converge on the same cache signature after
        # normalization and land in the same parallel gather batch. Without
        # this lock actually being used by those two methods, that race is
        # merely wasteful (both miss, both execute the same idempotent
        # operation, one cache write clobbers the other) rather than
        # corrupting, because both sides of the race represent the exact
        # same operation by construction — but it is a real race, not one
        # dedup rules out.
        self.lock = asyncio.Lock()

    def _is_expired(self, key: str) -> bool:
        """Check if a cache entry has expired.

        Args:
            key: Cache key to check

        Returns:
            True if entry is expired or timestamp missing
        """
        if key not in self._timestamps:
            return True
        age = time.time() - self._timestamps[key]
        return age > self._ttl_seconds

    def _evict_expired(self) -> int:
        """Remove all expired entries from the cache.

        Returns:
            Number of entries evicted
        """
        expired_keys = [k for k in self._cache if self._is_expired(k)]
        for key in expired_keys:
            del self._cache[key]
            self._timestamps.pop(key, None)
        return len(expired_keys)

    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.

        If found and not expired, the entry is moved to the end (most recently used).
        Expired entries are automatically removed.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found or expired
        """
        if key in self._cache:
            # Check TTL expiration
            if self._is_expired(key):
                # Remove expired entry
                del self._cache[key]
                self._timestamps.pop(key, None)
                return None
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        """Set a value in the cache.

        If the key exists, it's updated and moved to the end with refreshed TTL.
        If cache is full, expired entries are evicted first, then LRU eviction.

        Args:
            key: Cache key
            value: Value to cache
        """
        current_time = time.time()

        if key in self._cache:
            # Update existing and move to end
            self._cache.move_to_end(key)
        self._cache[key] = value
        self._timestamps[key] = current_time

        # Evict expired entries first (cheaper than LRU for memory)
        if len(self._cache) > self._max_size:
            self._evict_expired()

        # Still over capacity? LRU eviction
        while len(self._cache) > self._max_size:
            oldest_key, _ = self._cache.popitem(last=False)
            self._timestamps.pop(oldest_key, None)

    def clear(self) -> None:
        """Clear all entries from the cache."""
        self._cache.clear()
        self._timestamps.clear()

    def __len__(self) -> int:
        """Return the number of entries in the cache."""
        return len(self._cache)

    def items(self):
        """Return all non-expired items in the cache."""
        # Filter out expired entries
        return [(k, v) for k, v in self._cache.items() if not self._is_expired(k)]

    def remove(self, key: str) -> bool:
        """Remove a specific key from the cache.

        Args:
            key: Cache key to remove

        Returns:
            True if key was found and removed, False otherwise
        """
        if key in self._cache:
            del self._cache[key]
            self._timestamps.pop(key, None)
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring.

        Returns:
            Dict with size, max_size, ttl_seconds, expired_count
        """
        expired_count = sum(1 for k in self._cache if self._is_expired(k))
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "ttl_seconds": self._ttl_seconds,
            "expired_count": expired_count,
            "active_count": len(self._cache) - expired_count,
        }


@dataclass
class ToolCallResult:
    """Result of a single tool call through the pipeline."""

    tool_name: str
    arguments: Dict[str, Any]
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    cached: bool = False
    normalization_applied: Optional[str] = None
    skipped: bool = False
    skip_reason: Optional[str] = None
    outcome_kind: Optional[str] = None
    block_source: Optional[str] = None
    retryable: Optional[bool] = None
    user_message: Optional[str] = None

    # OpenAI-compatible tool_call_id — links response to assistant's tool_calls[].id
    tool_call_id: Optional[str] = None

    # Structured error information with traceback and metadata
    error_info: Optional["ErrorInfo"] = None


def _build_skip_result(
    *,
    tool_name: str,
    arguments: Dict[str, Any],
    success: bool,
    skip_reason: str,
    outcome_kind: str,
    block_source: str,
    retryable: bool,
    user_message: Optional[str] = None,
    result: Any = None,
    error: Optional[str] = None,
    execution_time_ms: float = 0.0,
    cached: bool = False,
    normalization_applied: Optional[str] = None,
    tool_call_id: Optional[str] = None,
) -> ToolCallResult:
    """Create a skipped tool-call result with structured block metadata."""
    return ToolCallResult(
        tool_name=tool_name,
        arguments=arguments,
        success=success,
        result=result,
        error=error,
        execution_time_ms=execution_time_ms,
        cached=cached,
        normalization_applied=normalization_applied,
        skipped=True,
        skip_reason=skip_reason,
        outcome_kind=outcome_kind,
        block_source=block_source,
        retryable=retryable,
        user_message=user_message or error or skip_reason,
        tool_call_id=tool_call_id,
    )


def _format_tool_command_value(value: Any) -> str:
    """Serialize a tool argument value for compact follow-up guidance."""
    if isinstance(value, str):
        return json_dumps(value)
    return repr(value)


def _build_tool_follow_up_command(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Build a compact tool command example for recovery suggestions."""
    serialized_args = ", ".join(
        f"{key}={_format_tool_command_value(value)}"
        for key, value in arguments.items()
        if value is not None
    )
    return f"{tool_name}({serialized_args})"


def _build_follow_up_suggestion(
    tool_name: str,
    arguments: Dict[str, Any],
    description: str,
) -> Dict[str, Any]:
    """Create a normalized follow-up suggestion payload."""
    return {
        "tool": tool_name,
        "command": _build_tool_follow_up_command(tool_name, arguments),
        "arguments": arguments,
        "description": description,
        "reason": description,
    }


def _build_redundant_call_recovery(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Build actionable recovery guidance for redundant tool calls."""
    canonical_name = canonicalize_core_tool_name(tool_name.lower())
    scope_path = str(arguments.get("path") or arguments.get("search_path") or ".")
    query = arguments.get("query") or arguments.get("pattern")
    symbol_name = arguments.get("symbol_name")
    suggestions: List[Dict[str, Any]] = []

    if canonical_name in {"code_search", "semantic_code_search", "grep"}:
        suggestions.append(
            _build_follow_up_suggestion(
                "overview",
                {"path": scope_path, "max_depth": 2},
                "Inspect the surrounding project structure instead of repeating the same search.",
            )
        )
        if query:
            suggestions.append(
                _build_follow_up_suggestion(
                    "graph",
                    {
                        "mode": "search",
                        "query": str(query),
                        "path": scope_path,
                        "top_k": 5,
                    },
                    f'Search the graph index for "{query}" instead of repeating the same text search.',
                )
            )
        else:
            suggestions.append(
                _build_follow_up_suggestion(
                    "graph",
                    {"mode": "overview", "path": scope_path, "top_k": 5},
                    "Use the graph overview to find the next module or symbol to inspect.",
                )
            )
        message = (
            f"[The same {_build_tool_follow_up_command(tool_name, arguments)} was already tried "
            "recently and is unlikely to return new information. Do not repeat the same search "
            "with identical arguments. Change the query or mode, narrow or broaden the path, or "
            "switch to structure-aware tools such as overview(...) or graph(...).]"
        )
    elif canonical_name == "refs":
        if symbol_name:
            suggestions.append(
                _build_follow_up_suggestion(
                    "graph",
                    {"mode": "callers|callees", "node": str(symbol_name), "depth": 2},
                    f'Explore call relationships for "{symbol_name}" instead of repeating refs().',
                )
            )
            suggestions.append(
                _build_follow_up_suggestion(
                    "code_search",
                    {
                        "query": str(symbol_name),
                        "path": scope_path,
                        "mode": "text",
                        "k": 5,
                    },
                    f'Search textually for "{symbol_name}" to find concrete files to read next.',
                )
            )
        else:
            suggestions.append(
                _build_follow_up_suggestion(
                    "overview",
                    {"path": scope_path, "max_depth": 2},
                    "Inspect the project structure to choose a more specific symbol target.",
                )
            )
        message = (
            f"[The same {_build_tool_follow_up_command(tool_name, arguments)} was already tried "
            "recently and is unlikely to reveal new references. Do not repeat the same refs() "
            "scan with identical arguments. Inspect the symbol through graph relationships or "
            "switch to a more targeted search to identify a concrete file to read.]"
        )
    else:
        suggestions.append(
            _build_follow_up_suggestion(
                "overview",
                {"path": scope_path, "max_depth": 2},
                "Inspect the nearby project structure before choosing a different tool or scope.",
            )
        )
        suggestions.append(
            _build_follow_up_suggestion(
                "graph",
                {"mode": "overview", "path": scope_path, "top_k": 5},
                "Use the graph overview to identify a higher-value next step.",
            )
        )
        message = (
            f"[{_build_tool_follow_up_command(tool_name, arguments)} overlaps with a recent call "
            "and was skipped. Do not repeat the same tool with identical arguments. Change the "
            "scope or switch to a different discovery tool.]"
        )

    return {
        "success": False,
        "error": message,
        "metadata": {"follow_up_suggestions": suggestions[:2]},
    }


def _build_repeated_failure_recovery(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Build recovery guidance for a call that already failed with the same arguments."""
    recovery = _build_redundant_call_recovery(tool_name, arguments)
    original_error = recovery.get("error", "")
    repeated_message = (
        f"[{_build_tool_follow_up_command(tool_name, arguments)} already failed earlier in this "
        "session with the same arguments. Do not repeat the same failing call. Change the path, "
        "query, or tool choice before retrying.]"
    )
    if original_error:
        repeated_message = f"{repeated_message}\n{original_error}"
    return {
        "success": False,
        "error": repeated_message,
        "metadata": recovery.get("metadata", {}),
    }


def _has_informative_skip_payload(call_result: "ToolCallResult") -> bool:
    """Return True when a skipped call still gave the model actionable information."""
    if not getattr(call_result, "skipped", False):
        return False

    output = getattr(call_result, "result", None)
    if isinstance(output, str):
        return bool(output.strip())
    if isinstance(output, dict):
        return bool(
            output
            and (
                output.get("error")
                or output.get("result") is not None
                or output.get("metadata")
                or output.get("success") is False
            )
        )
    return output is not None


@dataclass
class PipelineExecutionResult:
    """Result of executing multiple tool calls."""

    results: List[ToolCallResult] = field(default_factory=list)
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    skipped_calls: int = 0
    total_time_ms: float = 0.0
    budget_exhausted: bool = False
    # Parallel execution metrics
    parallel_execution_used: bool = False
    parallel_speedup: float = 1.0
    # Output aggregation and synthesis
    synthesis_recommended: bool = False
    synthesis_reason: Optional[str] = None
    synthesis_prompt: Optional[str] = None
    aggregation_state: Optional[str] = None


class ToolPipeline:
    """Coordinates tool execution flow.

    This class handles the complete lifecycle of tool calls:
    1. Validation (name format, tool exists, budget check)
    2. Argument normalization
    3. Deduplication (skip repeated failures)
    4. Execution via ToolExecutor
    5. Result tracking and analytics

    Example:
        pipeline = ToolPipeline(
            tool_registry=registry,
            tool_executor=executor,
            config=ToolPipelineConfig(tool_budget=20),
        )

        result = await pipeline.execute_tool_calls(tool_calls, context)
        for call_result in result.results:
            if call_result.success:
                print(f"{call_result.tool_name}: {call_result.result}")
    """

    # Regex pattern for valid tool names
    VALID_TOOL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")

    def __init__(
        self,
        tool_registry: "ToolRegistry",
        tool_executor: ToolExecutor,
        config: Optional[ToolPipelineConfig] = None,
        tool_cache: Optional["ToolCache"] = None,
        argument_normalizer: Optional[ArgumentNormalizer] = None,
        signature_store: Optional["SignatureStore"] = None,
        on_tool_start: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        on_tool_complete: Optional[Callable[[ToolCallResult], None]] = None,
        on_tool_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        deduplication_tracker: Optional[Any] = None,
        middleware_chain: Optional["MiddlewareChain"] = None,
        semantic_cache: Optional["ToolResultCache"] = None,
        search_router: Optional["SearchRouter"] = None,
        permission_policy: Optional["PermissionPolicy"] = None,
    ):
        """Initialize tool pipeline.

        Args:
            tool_registry: Registry of available tools
            tool_executor: Executor for running tools
            config: Pipeline configuration
            tool_cache: Optional cache for tool results
            argument_normalizer: Optional argument normalizer
            signature_store: Optional persistent storage for failed
                signatures (cross-session learning)
            on_tool_start: Callback when tool execution starts
            on_tool_complete: Callback when tool execution completes
            deduplication_tracker: Optional tracker for detecting redundant tool calls
            middleware_chain: Optional middleware chain for processing tool calls
            semantic_cache: Optional FAISS-based semantic cache for tool results
            search_router: Optional search router for enriching search tool calls
        """
        self.tools = tool_registry
        self.executor = tool_executor
        self.config = config or ToolPipelineConfig()
        self.tool_cache = tool_cache
        self.normalizer = argument_normalizer or ArgumentNormalizer()
        self.signature_store = signature_store
        self.deduplication_tracker = deduplication_tracker
        self.middleware_chain = middleware_chain
        self.semantic_cache = semantic_cache
        self.search_router = search_router
        self._permission_policy = permission_policy

        # Callbacks
        self.on_tool_start = on_tool_start
        self.on_tool_complete = on_tool_complete
        self.on_tool_event = on_tool_event

        # Spin detection: track whether last batch had all calls blocked
        self.last_batch_all_skipped: bool = False
        self.last_batch_effectively_blocked: bool = False

        # Output aggregation and synthesis checkpoints
        self._output_aggregator: Optional[OutputAggregator] = None
        self._synthesis_checkpoint: Optional[CompositeSynthesisCheckpoint] = None
        if self.config.enable_output_aggregation:
            self._output_aggregator = OutputAggregator(
                max_results=self.config.tool_budget,
                stale_threshold_seconds=60.0,
            )
        if self.config.enable_synthesis_checkpoints:
            self._synthesis_checkpoint = get_checkpoint_for_complexity(
                self.config.synthesis_checkpoint_complexity
            )

        # State tracking
        self._calls_used = 0
        # In-memory failed signatures for session-only tracking (backward compatible)
        # When signature_store is provided, it's used for persistent cross-session tracking
        self._failed_signatures: Set[tuple] = set()
        self._executed_tools: List[str] = []
        self._recent_code_navigation_hints: Dict[str, Dict[str, Any]] = {}
        self._last_code_navigation_tool: Optional[str] = None
        self._failed_path_redirects: Dict[str, str] = {}

        # Analytics
        self._tool_stats: Dict[str, Dict[str, Any]] = {}

        # Parallel-dispatch observability (co-design review item 16),
        # refreshed by every _dispatch_unique_calls() call.
        self._last_dispatch_used_parallelism: bool = False
        self._last_dispatch_speedup: float = 1.0

        # Session-level idempotent tool result cache with LRU eviction (Workstream E fix)
        # Prevents DeepSeek/Ollama from re-reading same files multiple times
        # Uses LRU instead of FIFO for more optimal cache behavior
        self._idempotent_cache: LRUToolCache = LRUToolCache(
            max_size=self.config.idempotent_cache_max_size
        )
        self._cache_hits = 0
        self._cache_misses = 0

        # Batch-level deduplication tracking
        self._batch_dedup_count = 0  # Total duplicates skipped

        # Cross-turn dedup cache for "effectively idempotent" tools
        # Uses the same LRU structure but covers tools outside IDEMPOTENT_TOOLS
        # (web_search, http_request, grep_search, etc.) with configurable TTL.
        self._cross_turn_enabled = self.config.enable_cross_turn_dedup
        self._cross_turn_cache: LRUToolCache = LRUToolCache(
            max_size=30,
            ttl_seconds=self.config.cross_turn_dedup_ttl,
        )
        self._cross_turn_hits = 0

        # File read timestamp tracking for deduplication (prompting loop fix)
        # Prevents re-reading identical files within TTL window
        self._read_file_timestamps: Dict[str, float] = {}
        self._read_file_paths: Dict[str, str] = {}
        self._read_file_snapshots: Dict[str, Any] = {}

        # Observability: pre-execution intent logging (LogAct-inspired)
        self._observability_bus: Optional[Any] = None
        self._trace_enricher: Optional[Any] = None

        # Credit assignment: automatic tool-level credit tracking (FEP-0001 Phase 3)
        self._credit_tracking_service: Optional[Any] = None

        # Online tool reputation: mid-turn credit feedback (updates per tool call)
        self._tool_reputation: Optional[Any] = None
        try:
            from victor.framework.rl.tool_reputation import ToolReputationTracker

            self._tool_reputation = ToolReputationTracker()
        except ImportError:
            pass

    @property
    def credit_tracking_service(self) -> Any:
        """Credit tracking service (if enabled). None otherwise."""
        return self._credit_tracking_service

    @property
    def calls_used(self) -> int:
        """Number of tool calls used."""
        return self._calls_used

    @property
    def calls_remaining(self) -> int:
        """Number of tool calls remaining in budget."""
        return max(0, self.config.tool_budget - self._calls_used)

    @property
    def tool_budget(self) -> int:
        """Configured maximum tool budget for the current turn."""
        return self.config.tool_budget

    @property
    def executed_tools(self) -> List[str]:
        """List of executed tool names."""
        return list(self._executed_tools)

    def set_tool_budget(self, budget: int) -> None:
        """Update the configured tool budget limit."""
        if budget < 0:
            raise ValueError(f"Tool budget must be non-negative: {budget}")
        self.config.tool_budget = budget
        if self._output_aggregator is not None and hasattr(self._output_aggregator, "_max_results"):
            self._output_aggregator._max_results = budget

    def start_new_turn(self) -> None:
        """Reset per-turn budget counters while preserving caches and learned state."""
        self._calls_used = 0
        self.last_batch_all_skipped = False
        self.last_batch_effectively_blocked = False

    def consume_budget(self, amount: int = 1) -> None:
        """Record externally executed tool usage against the shared turn budget."""
        if amount < 0:
            raise ValueError(f"Cannot consume negative tool budget: {amount}")
        self._calls_used += amount

    def reset(self) -> None:
        """Reset pipeline state for new conversation."""
        self._calls_used = 0
        self._failed_signatures.clear()
        self._executed_tools.clear()
        self._recent_code_navigation_hints.clear()
        self._last_code_navigation_tool = None
        # Clear idempotent cache (new session may have different file contents)
        self._idempotent_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self._batch_dedup_count = 0
        # Reset output aggregator
        if self._output_aggregator:
            self._output_aggregator.reset()
        # Clear semantic cache
        if self.semantic_cache is not None:
            self.semantic_cache.clear()

    def set_semantic_cache(self, cache: "ToolResultCache") -> None:
        """Set the semantic cache after initialization.

        Useful when embedding service isn't available at construction time.

        Args:
            cache: ToolResultCache instance to use for semantic caching
        """
        self.semantic_cache = cache
        self.config.enable_semantic_caching = cache is not None
        logger.debug("Semantic cache set on tool pipeline")

    def is_valid_tool_name(self, name: str) -> bool:
        """Check if tool name is valid format.

        Args:
            name: Tool name to validate

        Returns:
            True if valid format
        """
        if not name or not isinstance(name, str):
            return False
        normalized = normalize_model_tool_name(name)
        if not normalized or name != normalized:
            return False
        if len(normalized) > self.config.max_tool_name_length:
            return False
        return bool(self.VALID_TOOL_NAME_PATTERN.match(normalized))

    def _normalize_valid_tool_name(self, name: str) -> Optional[str]:
        """Normalize provider-emitted names and validate the canonical format."""
        raw_name = str(name or "").strip()
        if "-" in raw_name or "." in raw_name:
            return None
        normalized = normalize_model_tool_name(raw_name)
        if not normalized:
            return None
        if len(normalized) > self.config.max_tool_name_length:
            return None
        if not self.VALID_TOOL_NAME_PATTERN.match(normalized):
            return None
        return normalized

    def _normalize_arguments(
        self, tool_name: str, arguments: Any, context: Optional[Dict[str, Any]] = None
    ) -> tuple[Dict[str, Any], NormalizationStrategy]:
        """Normalize tool arguments with parameter enforcement.

        Args:
            tool_name: Name of the tool
            arguments: Raw arguments (may be string, dict, or None)
            context: Optional context for parameter inference

        Returns:
            Tuple of (normalized_args, strategy_used)
        """
        # Single authority: coercion + value-envelope recovery + normalization
        # (replaces the local json/ast/{value:...} ladder + normalize_arguments).
        normalized_args, strategy = self.normalizer.parse_tool_arguments(arguments, tool_name)

        # Apply parameter enforcement if available for this tool
        # This handles missing required parameters and type coercion (GAP-9)
        enforcer = get_enforcer_for_tool(tool_name)
        if enforcer is not None:
            try:
                enforced_args = enforcer.enforce(normalized_args, context=context or {})
                if enforced_args != normalized_args:
                    logger.debug(
                        f"Parameter enforcer modified args for {tool_name}: "
                        f"{set(enforced_args.keys()) - set(normalized_args.keys())} added"
                    )
                    normalized_args = enforced_args
                    # Update strategy to indicate enforcement was applied
                    if strategy == NormalizationStrategy.DIRECT:
                        strategy = NormalizationStrategy.MANUAL_REPAIR
            except ParameterInferenceError as e:
                # Could not infer required parameter - log but don't fail here
                # The tool execution will fail with a more meaningful error
                logger.warning(f"Could not infer required parameter for {tool_name}: {e}")
            except ParameterValidationError as e:
                # Invalid parameter value - log warning
                logger.warning(f"Parameter validation failed for {tool_name}: {e}")

        return normalized_args, strategy

    def _apply_search_routing(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> tuple[str, Dict[str, Any], bool]:
        """Apply search router hints to search-tool calls when beneficial."""
        if self.search_router is None:
            return tool_name, arguments, False

        if tool_name not in ("code_search", "semantic_code_search"):
            return tool_name, arguments, False

        if arguments.get("mode"):
            return tool_name, arguments, False

        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            return tool_name, arguments, False

        try:
            route = self.search_router.route(query)
        except Exception as exc:
            logger.debug("Search router enrichment failed for %s: %s", tool_name, exc)
            return tool_name, arguments, False

        if not route.tool_name or not route.tool_arguments:
            return tool_name, arguments, False

        routed_tool_name = route.tool_name
        if routed_tool_name == "graph" and routed_tool_name != tool_name:
            routed_args = {
                key: value for key, value in arguments.items() if key in GRAPH_TOOL_ARGUMENTS
            }
            routed_args.update(route.tool_arguments)
        else:
            routed_args = dict(route.tool_arguments)
            routed_args.update(arguments)
        changed = routed_tool_name != tool_name or routed_args != arguments

        if changed:
            logger.debug(
                "Applied search route to %s query: tool=%s extra_args=%s",
                tool_name,
                routed_tool_name,
                route.tool_arguments,
            )

        return routed_tool_name, routed_args, changed

    @staticmethod
    def _coerce_bool_like(value: Any) -> Any:
        """Coerce common boolean-like strings for tool parameters."""
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "on"}:
                return True
            if lowered in {"false", "0", "no", "off"}:
                return False
        return value

    def _apply_shell_policy(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> tuple[Dict[str, Any], bool]:
        """Apply the single-shell readonly policy.

        The shell tool defaults to ``readonly=False`` — the dangerous-command
        check and ShellSafetyPolicy are the primary safety floor. On read-only
        or display-only turns, ``readonly`` is pinned to True (opt into the
        allowlist for purely exploratory turns).
        """
        if tool_name != "shell":
            return arguments, False

        from victor.agent.action_authorizer import ActionIntent

        adjusted = dict(arguments)
        changed = False

        readonly = adjusted.get("readonly")
        if readonly is None:
            adjusted["readonly"] = False
            changed = True
        else:
            coerced = self._coerce_bool_like(readonly)
            if coerced is not readonly:
                adjusted["readonly"] = coerced
                changed = True

        raw_intent = (context or {}).get("current_intent")
        intent: Optional[ActionIntent] = None
        if isinstance(raw_intent, ActionIntent):
            intent = raw_intent
        elif isinstance(raw_intent, str):
            try:
                intent = ActionIntent(raw_intent)
            except ValueError:
                intent = None

        if (
            intent in {ActionIntent.DISPLAY_ONLY, ActionIntent.READ_ONLY}
            and adjusted.get("readonly") is not True
        ):
            adjusted["readonly"] = True
            changed = True
            logger.info("Pinned shell readonly=True due to %s intent", intent.value)

        # For WRITE_ALLOWED intent, detect obvious write patterns and default to False
        # This reduces the LLM's burden for common file write operations
        if intent == ActionIntent.WRITE_ALLOWED and readonly is None:
            cmd = str(adjusted.get("cmd", ""))
            # Detect write operations: redirection (excluding /dev/null), tee, sed -i
            has_write_pattern = (
                ">" in cmd
                and ">/dev/null" not in cmd
                and "2>/dev/null" not in cmd
                or ">>" in cmd
                or "tee" in cmd
                or (cmd.startswith("sed") and "-i" in cmd)
            )
            if has_write_pattern:
                adjusted["readonly"] = False
                changed = True
                logger.info("Set shell readonly=False for WRITE_ALLOWED intent with write pattern")

        return adjusted, changed

    def _normalize_tool_call(
        self, tool_name: str, arguments: Any, context: Optional[Dict[str, Any]] = None
    ) -> tuple[str, Dict[str, Any], NormalizationStrategy]:
        """Normalize a full tool call, including search-routing adjustments."""
        normalized_args, strategy = self._normalize_arguments(tool_name, arguments, context=context)
        normalized_args, readonly_changed = self._apply_shell_policy(
            tool_name,
            normalized_args,
            context=context,
        )
        normalized_tool_name, normalized_args, search_changed = self._apply_search_routing(
            tool_name, normalized_args
        )
        changed = readonly_changed or search_changed
        if changed and strategy == NormalizationStrategy.DIRECT:
            strategy = NormalizationStrategy.MANUAL_REPAIR
        return normalized_tool_name, normalized_args, strategy

    def _get_call_signature(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Generate signature for deduplication.

        Uses native xxHash3-based signature computation when available for
        10-20x speedup over JSON serialization + MD5.

        Args:
            tool_name: Tool name
            args: Tool arguments

        Returns:
            Hashable signature string (16-char hex with native, or tool:args with fallback)
        """
        # Use native signature computation if available (10-20x faster)
        if _NATIVE_SIGNATURE_AVAILABLE and native_compute_signature is not None:
            try:
                return native_compute_signature(tool_name, args)
            except Exception:
                pass  # Fall through to Python implementation

        # Pure Python fallback
        try:
            args_str = json_dumps(args, sort_keys=True, default=str)
        except Exception:
            args_str = str(args)
        return f"{tool_name}:{args_str}"

    @staticmethod
    def _path_arg_key(args: Dict[str, Any]) -> Optional[str]:
        """Return the first recognized path-like argument key."""
        for key in ("path", "file_path", "file", "filename"):
            if key in args and args.get(key):
                return key
        return None

    @staticmethod
    def _extract_suggested_paths(error: Optional[str]) -> List[str]:
        """Extract Did-you-mean path suggestions from filesystem errors."""
        if not error:
            return []
        suggestions = re.findall(r"^\s*-\s+(.+?)\s*$", error, re.MULTILINE)
        if suggestions:
            return [suggestion.strip() for suggestion in suggestions if suggestion.strip()]
        inline = re.search(r"Did you mean:\s*(.+)", error)
        if not inline:
            return []
        return [candidate.strip() for candidate in inline.group(1).split(",") if candidate.strip()]

    @classmethod
    def _choose_path_redirect(
        cls,
        original_path: str,
        suggestions: List[str],
        tool_name: str,
    ) -> Optional[str]:
        """Choose the best safe read redirect from a suggestion list."""
        if not original_path or not suggestions:
            return None
        canonical_tool = canonicalize_core_tool_name(tool_name.lower())
        if canonical_tool != "read":
            return None

        normalized_original = original_path.rstrip("/").replace("\\", "/")
        base_path, _ = os.path.splitext(normalized_original)
        file_suggestions = [candidate for candidate in suggestions if not candidate.endswith("/")]
        package_file_matches = [
            candidate
            for candidate in file_suggestions
            if candidate.replace("\\", "/").startswith(f"{base_path}/")
        ]
        if package_file_matches:
            return package_file_matches[0]
        if file_suggestions:
            return file_suggestions[0]
        return suggestions[0]

    def _record_path_redirect_from_error(
        self,
        tool_name: str,
        args: Dict[str, Any],
        error: Optional[str],
    ) -> None:
        """Remember read(path=bad) -> read(path=suggested) after filesystem errors."""
        path_key = self._path_arg_key(args)
        if path_key is None:
            return
        original_path = str(args.get(path_key) or "")
        redirect = self._choose_path_redirect(
            original_path,
            self._extract_suggested_paths(error),
            tool_name,
        )
        if not redirect or redirect == original_path:
            return
        self._failed_path_redirects[original_path] = redirect
        logger.info(
            "[Pipeline] Recorded path redirect for %s: %s -> %s",
            tool_name,
            original_path,
            redirect,
        )

    def _apply_path_redirect(
        self,
        tool_name: str,
        args: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Return rewritten args when a prior read failure supplied a better path."""
        if canonicalize_core_tool_name(tool_name.lower()) != "read":
            return None
        path_key = self._path_arg_key(args)
        if path_key is None:
            return None
        original_path = str(args.get(path_key) or "")
        redirect = self._failed_path_redirects.get(original_path)
        if not redirect:
            return None
        rewritten = dict(args)
        rewritten[path_key] = redirect
        return rewritten

    def _generate_tool_call_id(self, tool_name: str) -> str:
        """Generate a deterministic tool_call_id if one isn't provided.

        Uses tool name + timestamp to create a unique ID that's consistent
        for the same tool call within a session.

        Args:
            tool_name: Name of the tool being called

        Returns:
            A 16-character hex string tool_call_id
        """
        import hashlib

        timestamp = int(time.time() * 1000)
        hash_input = f"{tool_name}_{timestamp}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]

    def is_known_failure(self, tool_name: str, args: Dict[str, Any]) -> bool:
        """Check if a tool call is known to fail.

        Uses persistent SignatureStore when available for cross-session learning,
        falls back to in-memory set for session-only tracking.

        Args:
            tool_name: Name of the tool
            args: Tool arguments

        Returns:
            True if this call has failed before
        """
        if not self.config.enable_failed_signature_tracking:
            return False

        # Check persistent store first (cross-session)
        if self.signature_store is not None:
            try:
                if self.signature_store.is_known_failure(tool_name, args):
                    logger.debug(f"Tool call is known failure (persistent): {tool_name}")
                    return True
            except (OSError, IOError) as e:
                logger.warning(f"Signature store check failed (I/O error): {e}")
            except (KeyError, ValueError) as e:
                logger.debug(f"Signature store check failed (data error): {e}")

        # Fall back to in-memory check (session-only)
        signature = self._get_call_signature(tool_name, args)
        return signature in self._failed_signatures

    def record_failure(self, tool_name: str, args: Dict[str, Any], error_message: str) -> None:
        """Record a failed tool call.

        Uses persistent SignatureStore when available for cross-session learning,
        also records in-memory for current session.

        Args:
            tool_name: Name of the tool
            args: Tool arguments
            error_message: Error message from the failure
        """
        if not self.config.enable_failed_signature_tracking:
            return

        # Record in persistent store (cross-session)
        if self.signature_store is not None:
            try:
                self.signature_store.record_failure(tool_name, args, error_message)
                logger.debug(f"Recorded failure to persistent store: {tool_name}")
            except (OSError, IOError) as e:
                logger.warning(f"Failed to record to signature store (I/O error): {e}")
            except (ValueError, TypeError) as e:
                logger.debug(f"Failed to record to signature store (data error): {e}")

        # Also record in-memory for current session
        signature = self._get_call_signature(tool_name, args)
        self._failed_signatures.add(signature)

    def clear_failure(self, tool_name: str, args: Dict[str, Any]) -> bool:
        """Clear a specific failure signature (e.g., after a fix).

        Args:
            tool_name: Name of the tool
            args: Tool arguments

        Returns:
            True if signature was found and cleared
        """
        cleared = False

        # Clear from persistent store
        if self.signature_store is not None:
            try:
                if self.signature_store.clear_signature(tool_name, args):
                    cleared = True
                    logger.debug(f"Cleared failure from persistent store: {tool_name}")
            except (OSError, IOError) as e:
                logger.warning(f"Failed to clear from signature store (I/O error): {e}")
            except (KeyError, ValueError) as e:
                logger.debug(f"Failed to clear from signature store (data error): {e}")

        # Clear from in-memory
        signature = self._get_call_signature(tool_name, args)
        if signature in self._failed_signatures:
            self._failed_signatures.discard(signature)
            cleared = True

        return cleared

    def is_idempotent_tool(self, tool_name: str) -> bool:
        """Check if a tool is idempotent (safe to cache results).

        Idempotent tools are read-only and produce the same output for
        the same input arguments. Results can be safely cached within a session.

        Args:
            tool_name: Name of the tool

        Returns:
            True if tool is idempotent
        """
        return tool_name in IDEMPOTENT_TOOLS

    def get_cached_result(self, tool_name: str, args: Dict[str, Any]) -> Optional[ToolCallResult]:
        """Get cached result for an idempotent tool call.

        Args:
            tool_name: Name of the tool
            args: Tool arguments

        Returns:
            Cached ToolCallResult if found, None otherwise
        """
        if not self.config.enable_idempotent_caching:
            return None

        if not self.is_idempotent_tool(tool_name):
            return None

        signature = self._get_call_signature(tool_name, args)
        cached = self._idempotent_cache.get(signature)
        if cached is not None:
            self._cache_hits += 1
            logger.debug(f"Cache hit for {tool_name}: {signature[:100]}...")
        return cached

    def cache_result(self, tool_name: str, args: Dict[str, Any], result: ToolCallResult) -> None:
        """Cache result for an idempotent tool call.

        Only successful results from idempotent tools are cached.

        Args:
            tool_name: Name of the tool
            args: Tool arguments
            result: Tool call result to cache
        """
        if not self.config.enable_idempotent_caching:
            return

        if not self.is_idempotent_tool(tool_name):
            return

        if not result.success:
            return  # Don't cache failures

        signature = self._get_call_signature(tool_name, args)
        # Mark as cached in the result
        cached_result = ToolCallResult(
            tool_name=result.tool_name,
            arguments=result.arguments,
            success=result.success,
            result=result.result,
            error=result.error,
            execution_time_ms=0.0,  # Cached results have ~0 execution time
            cached=True,  # Mark as from cache
            normalization_applied=result.normalization_applied,
        )
        # LRU cache handles eviction automatically (Workstream E fix)
        self._idempotent_cache.set(signature, cached_result)
        self._cache_misses += 1
        logger.debug(f"Cached result for {tool_name} (cache size: {len(self._idempotent_cache)})")

    def invalidate_file_cache(self, file_path: str) -> int:
        """Invalidate cached results that involve a specific file path.

        Call this after a file is modified to ensure subsequent reads
        get fresh content.

        Args:
            file_path: Path of the modified file

        Returns:
            Number of cache entries invalidated
        """
        if not self.config.enable_idempotent_caching:
            return 0

        # Find and remove all cache entries involving this file
        to_remove = []
        for sig, result in self._idempotent_cache.items():
            # Check if file path is in arguments
            if file_path in str(result.arguments):
                to_remove.append(sig)

        for sig in to_remove:
            self._idempotent_cache.remove(sig)

        if to_remove:
            logger.debug(f"Invalidated {len(to_remove)} cache entries for {file_path}")

        return len(to_remove)

    def _invalidate_post_edit_freshness_state(
        self,
        file_path: str,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, int]:
        """Invalidate freshness-sensitive state after a successful edit/write."""
        summary = {
            "idempotent_cache_entries": self.invalidate_file_cache(file_path),
            "search_index_roots": 0,
            "dedup_tracker_calls": 0,
        }

        if self.semantic_cache is not None:
            self.semantic_cache.invalidate(file_path)

        try:
            from victor.core.utils.capability_loader import load_code_search_module

            module = load_code_search_module()
            summary["search_index_roots"] = module.mark_index_cache_stale_for_path(
                file_path,
                exec_ctx=context,
            )
        except ImportError:
            logger.debug("victor-coding not available, skipping index cache invalidation.")
        except Exception as exc:
            logger.debug("Failed to mark code_search index stale for %s: %s", file_path, exc)

        tracker = self.deduplication_tracker
        invalidate_tracker = getattr(tracker, "invalidate_for_modified_path", None)
        if callable(invalidate_tracker):
            try:
                summary["dedup_tracker_calls"] = int(invalidate_tracker(file_path) or 0)
            except Exception as exc:
                logger.debug(
                    "Failed to invalidate deduplication tracker for %s: %s",
                    file_path,
                    exc,
                )

        logger.debug(
            "Post-edit freshness invalidation for %s: cache=%d index_roots=%d dedup_calls=%d",
            file_path,
            summary["idempotent_cache_entries"],
            summary["search_index_roots"],
            summary["dedup_tracker_calls"],
        )
        return summary

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with idempotent cache stats and semantic cache stats
        """
        stats = {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_size": len(self._idempotent_cache),
            "hit_rate": (
                self._cache_hits / (self._cache_hits + self._cache_misses)
                if (self._cache_hits + self._cache_misses) > 0
                else 0.0
            ),
            "batch_dedup_count": self._batch_dedup_count,
            "cross_turn_hits": self._cross_turn_hits,
            "cross_turn_cache_size": len(self._cross_turn_cache),
        }
        # Include semantic cache stats if available
        if self.semantic_cache is not None:
            stats["semantic_cache"] = self.semantic_cache.get_stats()
        return stats

    def _is_duplicate_read(self, file_path: str, max_age_seconds: float = 300.0) -> bool:
        """Check if file was recently read (within TTL window).

        Used to prevent re-reading identical files during prompting loops.

        Args:
            file_path: Path of the file
            max_age_seconds: TTL for file read cache (default 5 minutes)

        Returns:
            True if file was read recently and should not be re-read
        """
        if file_path not in self._read_file_timestamps:
            return False
        age = time.monotonic() - self._read_file_timestamps[file_path]
        if age >= max_age_seconds:
            return False

        current_path = self._read_file_paths.get(file_path, file_path)
        previous_snapshot = self._read_file_snapshots.get(file_path)
        current_snapshot = capture_file_state(current_path)
        if previous_snapshot is None and current_snapshot is None:
            return True
        if previous_snapshot is None or current_snapshot is None:
            return False
        return previous_snapshot == current_snapshot

    @staticmethod
    def _normalize_mode_name(context: Optional[Dict[str, Any]]) -> Optional[str]:
        """Return the normalized current mode name from execution context."""
        if not isinstance(context, dict):
            return None
        for key in ("mode", "current_mode", "active_mode"):
            value = context.get(key)
            if value is None:
                continue
            text = str(value).strip().lower()
            if text:
                return text
        return None

    @staticmethod
    def _looks_like_code_file(file_path: Optional[str]) -> bool:
        """Return True when the path looks like a source file."""
        if not isinstance(file_path, str) or not file_path.strip():
            return False
        return Path(file_path).suffix.lower() in CODE_FILE_EXTENSIONS

    @staticmethod
    def _coerce_optional_int(value: Any) -> Optional[int]:
        """Best-effort integer coercion for tool arguments."""
        if value in (None, ""):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _tool_is_enabled(self, tool_name: str) -> bool:
        """Best-effort tool availability check for guidance generation."""
        try:
            return bool(self.tools.is_tool_enabled(tool_name))
        except Exception:
            return False

    def _has_structure_aware_navigation(self) -> bool:
        """Return True once the session has already narrowed via structure-aware tools."""
        return any(
            canonicalize_core_tool_name(tool_name.lower())
            in {
                "symbol",
                "refs",
                "lsp",
                "code_search",
                "semantic_code_search",
                "graph",
                "project_overview",
                "dependency_graph",
            }
            for tool_name in self._executed_tools
        )

    def _should_steer_broad_code_read(
        self,
        *,
        file_path: Optional[str],
        offset: Any,
        limit: Any,
        context: Optional[Dict[str, Any]],
    ) -> bool:
        """Return True when a broad code read should be steered to code intelligence first."""
        if not self._looks_like_code_file(file_path):
            return False
        mode_name = self._normalize_mode_name(context)
        if mode_name not in CODE_INTELLIGENCE_FIRST_MODES:
            return False
        if self._has_structure_aware_navigation():
            return False
        start_offset = self._coerce_optional_int(offset)
        max_lines = self._coerce_optional_int(limit)
        if start_offset not in (None, 0):
            return False
        if max_lines is not None and max_lines < 400:
            return False
        return any(
            self._tool_is_enabled(tool_name)
            for tool_name in ("project_overview", "lsp", "symbol", "refs")
        )

    def _build_broad_code_read_recovery(self, file_path: str) -> Dict[str, Any]:
        """Build actionable next steps for whole-file code reads in analysis modes."""
        parent_path = str(Path(file_path).parent or ".")
        suggestions: List[Dict[str, Any]] = []
        if self._tool_is_enabled("project_overview"):
            suggestions.append(
                _build_follow_up_suggestion(
                    "project_overview",
                    {"path": parent_path, "max_depth": 2},
                    "Inspect the nearby project structure before opening the whole file.",
                )
            )
        if self._tool_is_enabled("lsp"):
            suggestions.append(
                _build_follow_up_suggestion(
                    "lsp",
                    {"action": "diagnostics", "file_path": file_path},
                    "Use diagnostics to identify the exact hotspot before reading the full file.",
                )
            )
        if self._tool_is_enabled("symbol"):
            suggestions.append(
                _build_follow_up_suggestion(
                    "symbol",
                    {"file_path": file_path, "symbol_name": "<target_symbol>"},
                    "Open the specific function or class definition once you know the target symbol.",
                )
            )
        message = (
            f"[You are attempting a broad full-file read of '{file_path}' before narrowing the "
            "target. Do not start with a broad read(path=...) on a whole code file in default "
            "coding modes. First use structure-aware navigation such as "
            "project_overview(...), lsp(action='diagnostics', ...), symbol(...), or refs(...) "
            "to identify the exact symbol or range, then read only the relevant section.]"
        )
        return {
            "success": False,
            "error": message,
            "metadata": {"follow_up_suggestions": suggestions[:3]},
        }

    @classmethod
    def _extract_target_symbol_hint(cls, context: Optional[Dict[str, Any]]) -> Optional[str]:
        """Best-effort extraction of a target symbol from execution context."""
        if not isinstance(context, dict):
            return None
        for key in ("target_symbol", "symbol_name", "focus_symbol", "current_symbol"):
            raw_value = context.get(key)
            if raw_value is None:
                continue
            value = str(raw_value).strip()
            if value:
                return value
        return None

    def _select_broad_code_read_rewrite(
        self,
        file_path: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        exclude_tools: Optional[set[str]] = None,
    ) -> Optional[tuple[str, Dict[str, Any]]]:
        """Choose a safe structure-aware replacement for a broad code-file read."""
        excluded = {canonicalize_core_tool_name(name.lower()) for name in (exclude_tools or set())}
        symbol_hint = self._extract_target_symbol_hint(context)
        if symbol_hint is not None and "symbol" not in excluded and self._tool_is_enabled("symbol"):
            return (
                "symbol",
                {
                    "file_path": file_path,
                    "symbol_name": symbol_hint,
                },
            )
        if symbol_hint is not None and "refs" not in excluded and self._tool_is_enabled("refs"):
            return (
                "refs",
                {
                    "symbol_name": symbol_hint,
                    "search_path": str(Path(file_path).parent or "."),
                },
            )
        if "lsp" not in excluded and self._tool_is_enabled("lsp"):
            return (
                "lsp",
                {
                    "action": "diagnostics",
                    "file_path": file_path,
                },
            )
        if "project_overview" not in excluded and self._tool_is_enabled("project_overview"):
            return (
                "project_overview",
                {
                    "path": str(Path(file_path).parent or "."),
                    "max_depth": 2,
                },
            )
        return None

    @staticmethod
    def _extract_tool_result_items(result: Any, *keys: str) -> List[Any]:
        """Return a flattened list of candidate items from a tool result payload."""
        if isinstance(result, Mapping):
            items: List[Any] = []
            for key in keys:
                value = result.get(key)
                if isinstance(value, list):
                    items.extend(value)
                elif value not in (None, ""):
                    items.append(value)
            return items
        if isinstance(result, list):
            return list(result)
        if result not in (None, ""):
            return [result]
        return []

    @staticmethod
    def _coerce_line_reference(value: Any) -> Optional[Dict[str, Any]]:
        """Parse a best-effort file/line hint from code-intelligence output."""
        if isinstance(value, str):
            match = re.match(r"^(?P<path>.+?):(?P<line>\d+)(?::\d+)?$", value.strip())
            if not match:
                return None
            file_path = match.group("path").strip()
            line_number = ToolPipeline._coerce_optional_int(match.group("line"))
            if not file_path or line_number is None:
                return None
            return {"file_path": file_path, "line": line_number}
        if isinstance(value, Mapping):
            file_path = (
                value.get("file_path") or value.get("path") or value.get("file") or value.get("uri")
            )
            if not isinstance(file_path, str) or not file_path.strip():
                return None
            line_number = ToolPipeline._coerce_optional_int(
                value.get("line")
                or value.get("line_number")
                or value.get("start_line")
                or value.get("row")
            )
            if line_number is None:
                line_number = ToolPipeline._coerce_lsp_range_line(value)
            return {
                "file_path": file_path.strip(),
                "line": line_number,
            }
        return None

    @staticmethod
    def _coerce_lsp_range_line(value: Any) -> Optional[int]:
        """Extract a 1-indexed line number from standard LSP position/range payloads."""
        if not isinstance(value, Mapping):
            return None

        def _position_line(position: Any) -> Optional[int]:
            if not isinstance(position, Mapping):
                return None
            raw_line = position.get("line")
            line_number = ToolPipeline._coerce_optional_int(raw_line)
            if line_number is None:
                return None
            return line_number + 1

        for range_key in ("range", "selectionRange", "selection_range"):
            range_value = value.get(range_key)
            if not isinstance(range_value, Mapping):
                continue
            line_number = _position_line(range_value.get("start"))
            if line_number is not None:
                return line_number

        for position_key in ("position", "start"):
            line_number = _position_line(value.get(position_key))
            if line_number is not None:
                return line_number

        return None

    def _record_code_navigation_hints(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        result: Any,
    ) -> None:
        """Persist recent structure-aware hints for later broad-read narrowing."""
        canonical_tool = canonicalize_core_tool_name(tool_name.lower())
        if canonical_tool not in {"project_overview", "refs", "lsp", "symbol"}:
            return

        hints: List[Dict[str, Any]] = []
        if canonical_tool == "project_overview":
            for item in self._extract_tool_result_items(result, "entries", "files", "results"):
                if isinstance(item, Mapping):
                    file_path = item.get("path") or item.get("file_path") or item.get("file")
                    if isinstance(file_path, str) and file_path.strip():
                        hints.append({"file_path": file_path.strip(), "line": None})
        elif canonical_tool == "refs":
            for item in self._extract_tool_result_items(
                result,
                "references",
                "refs",
                "matches",
                "locations",
                "results",
            ):
                hint = self._coerce_line_reference(item)
                if hint is not None:
                    hints.append(hint)
        elif canonical_tool == "lsp":
            file_path = arguments.get("file_path") or arguments.get("path")
            if isinstance(file_path, str) and file_path.strip():
                for item in self._extract_tool_result_items(result, "diagnostics", "results"):
                    line_hint = (
                        self._coerce_optional_int(item.get("line"))
                        if isinstance(item, Mapping)
                        else None
                    )
                    if line_hint is None:
                        line_hint = self._coerce_lsp_range_line(item)
                    hint = self._coerce_line_reference(item) or {
                        "file_path": file_path.strip(),
                        "line": line_hint,
                    }
                    hints.append(hint)
        elif canonical_tool == "symbol":
            file_path = arguments.get("file_path") or arguments.get("path")
            if isinstance(file_path, str) and file_path.strip():
                line_hint: Optional[int] = None
                symbol_items = self._extract_tool_result_items(
                    result,
                    "matches",
                    "definitions",
                    "locations",
                    "results",
                    "symbols",
                )
                for item in symbol_items:
                    parsed_hint = self._coerce_line_reference(item)
                    if parsed_hint is not None and parsed_hint.get("line") is not None:
                        line_hint = parsed_hint["line"]
                        break
                hints.append({"file_path": file_path.strip(), "line": line_hint})

        if not hints:
            return

        self._last_code_navigation_tool = canonical_tool
        for hint in hints:
            file_path = hint.get("file_path")
            if not isinstance(file_path, str) or not file_path.strip():
                continue
            normalized_file_path = normalize_file_path(file_path) or file_path
            self._recent_code_navigation_hints[normalized_file_path] = {
                "source_tool": canonical_tool,
                "file_path": file_path.strip(),
                "line": self._coerce_optional_int(hint.get("line")),
                "use_count": 0,
                "created_at": time.monotonic(),
                "last_result_hash": None,
            }

    def _select_follow_up_code_read_rewrite(
        self,
        file_path: str,
        *,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[tuple[str, Dict[str, Any], str]]:
        """Reuse recent navigation hints to keep narrowing later broad reads.

        Args:
            file_path: The file path to read
            offset: Explicit offset from the agent (if provided, hints are ignored)
            limit: Explicit limit from the agent (if provided, hints are ignored)
            context: Optional context dictionary

        Returns:
            None if no rewrite should be applied, or a tuple of (tool_name, arguments, steering_message)
        """
        # If agent provided explicit offset/limit, respect them and don't rewrite
        if offset is not None or limit is not None:
            return None

        normalized_file_path = normalize_file_path(file_path) or file_path
        recent_hint = self._recent_code_navigation_hints.get(normalized_file_path)
        if recent_hint is not None:
            # Check if hint has expired (TTL)
            hint_age = time.monotonic() - recent_hint.get("created_at", 0)
            if hint_age > self.config.navigation_hint_ttl_seconds:
                logger.debug(
                    "[Pipeline] Navigation hint for %s expired (age=%.1fs > ttl=%.1fs)",
                    normalized_file_path,
                    hint_age,
                    self.config.navigation_hint_ttl_seconds,
                )
                del self._recent_code_navigation_hints[normalized_file_path]
                return None

            # Check if hint has been used too many times
            use_count = recent_hint.get("use_count", 0)
            if use_count >= self.config.navigation_hint_max_uses:
                logger.debug(
                    "[Pipeline] Navigation hint for %s exhausted (use_count=%d >= max=%d)",
                    normalized_file_path,
                    use_count,
                    self.config.navigation_hint_max_uses,
                )
                del self._recent_code_navigation_hints[normalized_file_path]
                return None

            line_number = self._coerce_optional_int(recent_hint.get("line"))
            if line_number is not None:
                # Increment use count
                recent_hint["use_count"] = use_count + 1

                calculated_offset = max(line_number - TARGETED_READ_LINE_CONTEXT, 0)
                return (
                    "read",
                    {
                        "path": file_path,
                        "offset": calculated_offset,
                        "limit": TARGETED_READ_LINE_LIMIT,
                    },
                    (
                        f"Narrowed broad read(path=...) on '{file_path}' to the region around "
                        f"line {line_number} from recent {recent_hint.get('source_tool')} output "
                        f"(use #{recent_hint['use_count']}/{self.config.navigation_hint_max_uses})."
                    ),
                )

        symbol_hint = self._extract_target_symbol_hint(context)
        if (
            symbol_hint is not None
            and self._last_code_navigation_tool == "project_overview"
            and self._tool_is_enabled("symbol")
        ):
            return (
                "symbol",
                {
                    "file_path": file_path,
                    "symbol_name": symbol_hint,
                },
                (
                    f"Continued narrowing broad read(path=...) on '{file_path}' after recent "
                    f"project_overview(...) by routing to symbol(symbol_name={symbol_hint!r})."
                ),
            )
        return None

    @staticmethod
    def _mapping_field_is_empty(result: Mapping[str, Any], *keys: str) -> bool:
        """Return True when none of the candidate keys contain meaningful data."""
        present = False
        for key in keys:
            if key not in result:
                continue
            present = True
            value = result.get(key)
            if value not in (None, [], (), {}, ""):
                return False
        return present or not bool(result)

    def _get_low_signal_broad_read_rewrite_reason(
        self,
        *,
        tool_name: str,
        result: Any,
    ) -> Optional[str]:
        """Return a human-readable reason when a rewrite result is too weak to stop."""
        canonical_tool = canonicalize_core_tool_name(tool_name.lower())
        if not isinstance(result, Mapping):
            return None
        if canonical_tool == "lsp":
            diagnostics = result.get("diagnostics")
            if diagnostics in (None, [], ()) and self._tool_is_enabled("project_overview"):
                return "LSP diagnostics were empty"
            return None
        if canonical_tool == "symbol":
            if self._mapping_field_is_empty(
                result,
                "matches",
                "definitions",
                "locations",
                "results",
                "symbols",
            ) and self._tool_is_enabled("refs"):
                return "Symbol lookup was empty"
            return None
        if canonical_tool == "refs":
            if self._mapping_field_is_empty(
                result,
                "references",
                "refs",
                "matches",
                "locations",
                "results",
            ) and self._tool_is_enabled("project_overview"):
                return "Reference lookup was empty"
            return None
        return None

    def record_file_read(self, read_key: str, file_path: Optional[str] = None) -> None:
        """Record that a file was read.

        Updates the timestamp for deduplication tracking.

        Args:
            read_key: Deduplication key for the file read, including offset/limit
            file_path: Actual file path associated with the read key
        """
        # Evict oldest entries if at capacity (prevent unbounded growth)
        if len(self._read_file_timestamps) >= 2000:
            oldest_key = next(iter(self._read_file_timestamps))
            del self._read_file_timestamps[oldest_key]
            self._read_file_paths.pop(oldest_key, None)
            self._read_file_snapshots.pop(oldest_key, None)
        self._read_file_timestamps[read_key] = time.monotonic()
        actual_path = file_path or read_key
        self._read_file_paths[read_key] = actual_path
        self._read_file_snapshots[read_key] = capture_file_state(actual_path)
        logger.debug(f"Recorded file read: {read_key}")

    def _clear_read_tracking_for_file(self, file_path: str) -> None:
        """Clear read-tracking entries for a file after it changes."""
        normalized_target = normalize_file_path(file_path)
        if not normalized_target:
            return

        stale_keys = [
            read_key
            for read_key, tracked_path in self._read_file_paths.items()
            if normalize_file_path(tracked_path) == normalized_target
        ]
        for read_key in stale_keys:
            self._read_file_timestamps.pop(read_key, None)
            self._read_file_paths.pop(read_key, None)
            self._read_file_snapshots.pop(read_key, None)

    def deduplicate_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], List[tuple[int, tuple]]]:
        """Deduplicate identical tool calls within a batch.

        Some providers (DeepSeek, Ollama) sometimes issue duplicate tool calls
        in the same response. This method detects and removes duplicates,
        returning deduplicated results for the duplicates that use the first
        occurrence's result.

        Args:
            tool_calls: List of tool call requests

        Returns:
            Tuple of (unique_calls, duplicate_info) where duplicate_info
            contains (original_index, signature) tuples for duplicates
        """
        if not self.config.enable_batch_deduplication:
            return tool_calls, []

        seen_signatures: Dict[tuple, int] = {}  # signature -> index in unique_calls
        unique_calls: List[Dict[str, Any]] = []
        duplicate_indices: List[tuple[int, tuple]] = []  # (original_index, signature)

        for i, tc in enumerate(tool_calls):
            # Skip invalid structures - let execute_tool_calls handle them
            if not isinstance(tc, dict):
                unique_calls.append(tc)
                continue

            tool_name = tc.get("name", "")
            raw_args = tc.get("arguments", {})

            # Coerce arguments for signature comparison (single coercion ladder).
            if isinstance(raw_args, str):
                raw_args = ArgumentNormalizer.coerce_arg_string(raw_args)
            elif raw_args is None:
                raw_args = {}

            signature = self._get_call_signature(tool_name, raw_args)

            if signature in seen_signatures:
                # Duplicate found
                duplicate_indices.append((i, signature))
                self._batch_dedup_count += 1
                logger.info(
                    f"[Pipeline] Deduplicating {tool_name} call "
                    f"(duplicate #{self._batch_dedup_count})"
                )
            else:
                # First occurrence
                seen_signatures[signature] = len(unique_calls)
                unique_calls.append(tc)

        return unique_calls, duplicate_indices

    async def _compute_one_call_result(
        self, tool_call: Any, context: Dict[str, Any]
    ) -> ToolCallResult:
        """Compute the ToolCallResult for one already-deduplicated call.

        Matches the pre-parallel-dispatch inline logic exactly: a
        coordinator-marked ``_invalid`` (hallucinated name) call short-
        circuits to a skip result without reaching ``_execute_single_call``,
        with a tool_call_id generated up front when the provider gave none
        — unlike other skip paths, this one always gets an id even though
        it's marked skipped, so response-coverage matching still finds it.
        """
        if isinstance(tool_call, dict) and tool_call.get("_invalid"):
            tool_name = tool_call.get("name", "unknown")
            tc_id = tool_call.get("id")
            if tc_id is None:
                tc_id = self._generate_tool_call_id(tool_name)
            return _build_skip_result(
                tool_name=tool_name,
                arguments={},
                success=False,
                error=tool_call.get("_error", "Unknown tool"),
                skip_reason=tool_call.get("_error"),
                outcome_kind="invalid_tool_name",
                block_source="tool_validation",
                retryable=False,
                tool_call_id=tc_id,
            )
        return await self._execute_single_call(tool_call, context)

    def _is_parallel_eligible(self, tool_call: Any, eligible_tools: frozenset) -> bool:
        """True iff ``tool_call`` may run concurrently with other eligible
        calls in the same batch (co-design review item 16)."""
        if not isinstance(tool_call, dict) or tool_call.get("_invalid"):
            return False
        name = self._normalize_valid_tool_name(tool_call.get("name", ""))
        return bool(name) and name in eligible_tools

    def _budget_exhausted_skip(self, tool_call: Any) -> ToolCallResult:
        """Build the tail-skip result for a call never dispatched because
        the batch's admitted-call count already met the remaining budget.

        Matches the pre-parallelism inline tail-skip exactly (retryable,
        with an explicit user-facing message) rather than the internal
        check inside ``_execute_single_call`` (used only when a call is
        actually dispatched and discovers budget was exhausted by a
        same-batch predecessor it couldn't have known about in advance —
        see the docstring on ``_dispatch_unique_calls``).
        """
        tool_name = (
            self._normalize_valid_tool_name(tool_call.get("name", ""))
            if isinstance(tool_call, dict)
            else ""
        ) or "unknown"
        return _build_skip_result(
            tool_name=tool_name,
            arguments={},
            success=False,
            skip_reason="Tool budget exhausted — call not executed",
            outcome_kind="budget_exhausted",
            block_source="tool_budget",
            retryable=True,
            user_message="The tool budget was exhausted before this call could be executed.",
        )

    async def _dispatch_unique_calls(
        self, unique_calls_iter: List[Any], context: Dict[str, Any]
    ) -> List[ToolCallResult]:
        """Compute call results for every entry in ``unique_calls_iter``, in
        order (co-design review item 16).

        Both the sequential (default) and parallel paths compute the
        budget-skip decision BEFORE dispatching a call, via the same
        ``admitted`` vs. ``remaining_budget`` check, rather than relying on
        ``_execute_single_call``'s own internal check-and-skip — so a call
        that will certainly be skipped never pays for normalization,
        path-redirect, or permission-policy work, and always gets the
        explicit, retryable tail-skip result
        (``_budget_exhausted_skip``) instead of whatever
        ``_execute_single_call`` happens to produce internally. This keeps
        the sequential default path byte-identical to the pre-parallelism
        behavior (the old inline loop's tail-skip-and-break), which relying
        on ``_execute_single_call``'s internal check alone would not: that
        internal check exists as a defensive fallback for budget already
        exhausted by a PRIOR turn, not as the primary decision point within
        a batch.

        When ``config.parallel_tool_execution`` is on, contiguous runs of
        parallel-eligible calls (read-only allowlist; shell, write tools,
        and unknown/unclassified tools always excluded — see
        ``_compute_parallel_eligible_tools``) dispatch concurrently under a
        ``max_concurrent_tools`` semaphore, reassembled back into original
        call order. Computing the budget-skip decision before dispatch
        matters most here: several concurrent calls could otherwise all
        pass a stale "budget not yet exhausted" check before any of them
        increments the shared counter, overshooting the budget by up to
        ``max_concurrent_tools - 1``.

        The ``admitted`` count treats every dispatched call as consuming one
        budget unit even though a handful of edge cases (a cache hit, a
        call skipped for an unrelated reason after admission) don't
        actually increment ``self._calls_used`` — an intentionally
        conservative approximation: it can only under-admit relative to the
        true remaining budget, never overshoot it.
        """
        n = len(unique_calls_iter)
        results: List[Optional[ToolCallResult]] = [None] * n
        self._last_dispatch_used_parallelism = False
        self._last_dispatch_speedup = 1.0

        remaining_budget = max(0, self.config.tool_budget - self._calls_used)

        def _may_dispatch(admitted_count: int) -> bool:
            # The very first call this dispatch attempts ALWAYS gets to
            # actually run, even if remaining_budget is already 0 — matching
            # the pre-parallelism behavior, where the outer bookkeeping loop
            # only ever tail-skipped calls AFTER at least one call had
            # already gone through _execute_single_call and its own
            # internal budget check (which fires immediately, before any
            # real tool invocation, when budget was already exhausted by a
            # PRIOR turn). Skipping the first call here too — without ever
            # giving _execute_single_call a chance to run — would silently
            # change its result from a defensive-fallback skip
            # (retryable=False) to this method's tail-skip (retryable=True),
            # which is read by callers deciding whether to retry.
            return admitted_count == 0 or admitted_count < remaining_budget

        if not self.config.parallel_tool_execution:
            admitted = 0
            for idx, tool_call in enumerate(unique_calls_iter):
                if not _may_dispatch(admitted):
                    results[idx] = self._budget_exhausted_skip(tool_call)
                    continue
                results[idx] = await self._compute_one_call_result(tool_call, context)
                admitted += 1
            return results  # type: ignore[return-value]

        eligible_tools = _compute_parallel_eligible_tools()
        semaphore = asyncio.Semaphore(max(1, self.config.max_concurrent_tools))
        admitted = 0

        async def _run(idx: int, tool_call: Any) -> None:
            async with semaphore:
                results[idx] = await self._compute_one_call_result(tool_call, context)

        i = 0
        parallel_calls_run = 0
        parallel_rounds = 0
        while i < n:
            if not _may_dispatch(admitted):
                results[i] = self._budget_exhausted_skip(unique_calls_iter[i])
                i += 1
                continue

            if self._is_parallel_eligible(unique_calls_iter[i], eligible_tools):
                batch_indices = []
                while (
                    i < n
                    and _may_dispatch(admitted)
                    and self._is_parallel_eligible(unique_calls_iter[i], eligible_tools)
                ):
                    batch_indices.append(i)
                    admitted += 1
                    i += 1
                # return_exceptions=True: a raise from one gathered call (e.g.
                # an unguarded ParameterInferenceError from the enforcer, or a
                # permission-policy failure — _execute_single_call's own
                # try/except only wraps the actual tool invocation) must not
                # propagate and orphan its still-running siblings in this
                # same gather group as abandoned, unawaited background tasks.
                gather_outcomes = await asyncio.gather(
                    *(_run(idx, unique_calls_iter[idx]) for idx in batch_indices),
                    return_exceptions=True,
                )
                for idx, outcome in zip(batch_indices, gather_outcomes):
                    if isinstance(outcome, BaseException):
                        tool_call = unique_calls_iter[idx]
                        tool_name = (
                            self._normalize_valid_tool_name(tool_call.get("name", ""))
                            if isinstance(tool_call, dict)
                            else ""
                        ) or "unknown"
                        logger.error(
                            "[pipeline] Parallel dispatch of %s raised unexpectedly: %s",
                            tool_name,
                            outcome,
                        )
                        results[idx] = ToolCallResult(
                            tool_name=tool_name,
                            arguments={},
                            success=False,
                            error=str(outcome),
                        )
                if len(batch_indices) > 1:
                    parallel_calls_run += len(batch_indices)
                    parallel_rounds += 1
            else:
                results[i] = await self._compute_one_call_result(unique_calls_iter[i], context)
                admitted += 1
                i += 1

        if parallel_rounds > 0:
            self._last_dispatch_used_parallelism = True
            self._last_dispatch_speedup = parallel_calls_run / parallel_rounds

        return results  # type: ignore[return-value]

    async def execute_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> PipelineExecutionResult:
        """Execute multiple tool calls.

        Args:
            tool_calls: List of tool call requests
            context: Execution context passed to tools

        Returns:
            PipelineExecutionResult with all results
        """
        context = context or {}
        result = PipelineExecutionResult(total_calls=len(tool_calls))
        start_time = time.monotonic()

        # Batch-level deduplication to handle providers that send duplicate calls
        unique_calls, duplicate_info = self.deduplicate_tool_calls(tool_calls)

        # Track results by signature for duplicate resolution
        results_by_signature: Dict[tuple, ToolCallResult] = {}

        # Build tool history for synthesis checkpoint
        tool_history: List[Dict[str, Any]] = []

        unique_calls_iter = list(unique_calls)

        # Compute every call's result up front (co-design review item 16):
        # sequential and byte-identical to the prior inline per-iteration
        # logic when parallel_tool_execution is off (the default); when on,
        # contiguous runs of read-only-allowlisted calls dispatch
        # concurrently. See _dispatch_unique_calls for the full contract.
        call_results = await self._dispatch_unique_calls(unique_calls_iter, context)
        result.parallel_execution_used = self._last_dispatch_used_parallelism
        result.parallel_speedup = self._last_dispatch_speedup

        for unique_call_idx, tool_call in enumerate(unique_calls_iter):
            # Capture tool_call_id BEFORE execution so it's set even if the call fails
            tc_id = tool_call.get("id") if isinstance(tool_call, dict) else None

            call_result = call_results[unique_call_idx]
            # Propagate tool_call_id from provider's tool_calls[].id per OpenAI spec.
            # Only auto-generate an ID for executed calls — internally-skipped calls
            # without a provider ID have no corresponding tool_call entry, so they
            # must NOT get an ID (process_tool_results would otherwise emit a spurious
            # tool response for them, double-counting executed_tools).
            if tc_id is not None:
                call_result.tool_call_id = tc_id
            elif not call_result.skipped:
                tool_name = (
                    tool_call.get("name", "unknown") if isinstance(tool_call, dict) else "unknown"
                )
                tc_id = self._generate_tool_call_id(tool_name)
                call_result.tool_call_id = tc_id
            result.results.append(call_result)

            # Store result by signature for duplicate resolution (only for dict items)
            tool_name = ""
            raw_args: Dict[str, Any] = {}
            if isinstance(tool_call, dict):
                tool_name = tool_call.get("name", "")
                raw_args = tool_call.get("arguments", {})
                if isinstance(raw_args, str):
                    raw_args = ArgumentNormalizer.coerce_arg_string(raw_args)
                elif raw_args is None:
                    raw_args = {}
                signature = self._get_call_signature(tool_name, raw_args)
                results_by_signature[signature] = call_result

            # Add to output aggregator
            if self._output_aggregator and tool_name:
                self._output_aggregator.add_result(
                    tool_name=tool_name,
                    result=call_result.result,
                    success=call_result.success,
                    metadata={"args": raw_args},
                )

            # Track for synthesis checkpoint
            tool_history.append(
                {
                    "tool": tool_name,
                    "args": raw_args,
                    "success": call_result.success,
                    "result": (str(call_result.result)[:500] if call_result.result else ""),
                    "error": call_result.error,
                }
            )

            if call_result.skipped:
                result.skipped_calls += 1
            elif call_result.success:
                result.successful_calls += 1
            else:
                result.failed_calls += 1

        # Budget exhaustion is now determined by _dispatch_unique_calls having
        # already produced a "budget exhausted" skip result (co-design review
        # item 16) for every call from the exhaustion point onward — computed
        # BEFORE dispatch in both the sequential and parallel paths (see
        # _dispatch_unique_calls's docstring). A separate post-hoc tail-skip
        # pass here would be not just redundant but WRONG: since the whole
        # batch is now dispatched before this bookkeeping loop runs at all,
        # self._calls_used already reflects its final, post-batch value from
        # the very first iteration — checking it per-iteration would
        # spuriously flag budget exhaustion starting at index 0 and overwrite
        # already-successful results with duplicate skips.
        #
        # Only set the flag when this batch actually had calls to run: an
        # empty (or fully pre-deduplicated-away) batch on an
        # already-exhausted session budget didn't itself exhaust anything,
        # so it must not report budget_exhausted=True.
        if unique_calls_iter and self._calls_used >= self.config.tool_budget:
            result.budget_exhausted = True

        # Add skipped results for duplicates (reuse first occurrence's result)
        for original_idx, signature in duplicate_info:
            if signature in results_by_signature:
                original_result = results_by_signature[signature]
                # Use the provider-given ID for the duplicate call if available.
                # If the provider gave no ID, leave it None — process_tool_results
                # will then skip the entry (no tool_call to respond to per OpenAI spec).
                dup_call = tool_calls[original_idx] if original_idx < len(tool_calls) else {}
                dup_tc_id = dup_call.get("id") if isinstance(dup_call, dict) else None
                dup_result = _build_skip_result(
                    tool_name=original_result.tool_name,
                    arguments=original_result.arguments,
                    success=original_result.success,
                    result=original_result.result,
                    error=original_result.error,
                    execution_time_ms=0.0,  # Deduplicated, no execution time
                    cached=True,  # Mark as effectively cached
                    skip_reason="Deduplicated (duplicate call in batch)",
                    outcome_kind="duplicate_in_batch",
                    block_source="batch_deduplication",
                    retryable=True,
                    user_message=(
                        "This tool call duplicated another call in the same batch, so the "
                        "existing result was reused."
                    ),
                    tool_call_id=dup_tc_id,
                )
                result.results.append(dup_result)
                result.skipped_calls += 1

        # Track whether ALL tool calls in this batch were skipped.
        # Some skipped batches still return actionable recovery content and should
        # not count as a hard blocked turn for spin detection.
        self.last_batch_all_skipped = len(result.results) > 0 and all(
            getattr(r, "skipped", False) for r in result.results
        )
        self.last_batch_effectively_blocked = self.last_batch_all_skipped and not any(
            _has_informative_skip_payload(r) for r in result.results
        )
        if self.last_batch_all_skipped:
            # Collect actual skip reasons for accurate logging
            skip_reasons = []
            block_sources = []
            for r in result.results:
                reason = getattr(r, "skip_reason", None)
                source = getattr(r, "block_source", None)
                if reason:
                    skip_reasons.append(reason)
                if source:
                    block_sources.append(source)

            # Build informative log message
            if skip_reasons:
                # Count unique reasons
                from collections import Counter

                reason_counts = Counter(skip_reasons)
                reason_summary = ", ".join(f"{r} ({c})" for r, c in reason_counts.items())
                logger.info(
                    f"[pipeline] All {len(result.results)} tool calls in batch were "
                    f"skipped: {reason_summary}"
                )
            else:
                logger.info(
                    f"[pipeline] All {len(result.results)} tool calls in batch were "
                    f"skipped/blocked"
                )

        # Check synthesis checkpoint
        if self._synthesis_checkpoint and tool_history:
            task_context = {
                "elapsed_time": context.get("elapsed_time", 0),
                "timeout": context.get("timeout", 180),
                "task_type": context.get("task_type", "unknown"),
                "unified_task_tracker": context.get(
                    "unified_task_tracker"
                ),  # Phase 3: Pass UnifiedTaskTracker
            }
            checkpoint_result = self._synthesis_checkpoint.check(tool_history, task_context)
            if checkpoint_result.should_synthesize:
                result.synthesis_recommended = True
                result.synthesis_reason = checkpoint_result.reason
                result.synthesis_prompt = checkpoint_result.suggested_prompt
                logger.info(f"Synthesis checkpoint triggered: {checkpoint_result.reason}")

        # Set aggregation state
        if self._output_aggregator:
            result.aggregation_state = self._output_aggregator.state.value
            # If aggregator is in a synthesis-ready state, also recommend synthesis
            if self._output_aggregator.state in (
                AggregationState.READY_TO_SYNTHESIZE,
                AggregationState.LOOPING,
                AggregationState.STUCK,
            ):
                if not result.synthesis_recommended:
                    result.synthesis_recommended = True
                    result.synthesis_reason = (
                        f"Aggregation state: {self._output_aggregator.state.value}"
                    )
                    result.synthesis_prompt = self._output_aggregator.get_synthesis_prompt()

        result.total_time_ms = (time.monotonic() - start_time) * 1000
        return result

    def _emit_tool_intent(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Emit pre-execution intent event (LogAct-inspired).

        Records tool intent BEFORE execution, enabling future voting/gating.
        Uses ObservabilityBus for event emission.
        """
        bus = getattr(self, "_observability_bus", None)
        if bus is None:
            return

        import hashlib

        args_str = str(sorted(arguments.items())) if arguments else ""
        args_hash = hashlib.md5(args_str.encode()).hexdigest()[:12]

        reasoning = ""
        enricher = getattr(self, "_trace_enricher", None)
        if enricher:
            reasoning = getattr(enricher, "_pending_reasoning", "") or ""

        # Correlation spine (R1): stamp session/turn/request so this invoked record
        # joins the turn's offered (tool.supply) and resulted (rl_outcome) records.
        try:
            from victor.core.context import get_request_id, get_session_id, get_turn_id

            correlation = {
                "session_id": get_session_id() or "",
                "turn_id": get_turn_id() or "",
                "request_id": get_request_id() or "",
            }
        except Exception:
            correlation = {}

        try:
            bus.emit_sync(
                "tool.intent",
                {
                    "tool_name": tool_name,
                    "arguments_hash": args_hash,
                    "reasoning_before": reasoning[:500],
                    "timestamp": time.time(),
                    **correlation,
                },
            )
        except Exception:
            pass  # Intent logging is non-critical

    async def _execute_single_call(
        self,
        tool_call: Dict[str, Any],
        context: Dict[str, Any],
    ) -> ToolCallResult:
        """Execute a single tool call.

        Args:
            tool_call: Tool call request
            context: Execution context

        Returns:
            ToolCallResult
        """
        # Validate structure
        if not isinstance(tool_call, dict):
            return _build_skip_result(
                tool_name="unknown",
                arguments={},
                success=False,
                skip_reason="Invalid tool call structure (not a dict)",
                outcome_kind="invalid_tool_call",
                block_source="validation",
                retryable=False,
            )

        raw_tool_name = tool_call.get("name", "")
        tool_name = self._normalize_valid_tool_name(raw_tool_name)
        raw_args = tool_call.get("arguments", {})

        # Validate tool name
        if not raw_tool_name:
            return _build_skip_result(
                tool_name="unknown",
                arguments={},
                success=False,
                skip_reason="Tool call missing name",
                outcome_kind="missing_tool_name",
                block_source="validation",
                retryable=False,
            )

        if not tool_name:
            return _build_skip_result(
                tool_name=str(raw_tool_name),
                arguments={},
                success=False,
                skip_reason=f"Invalid tool name format: {raw_tool_name}",
                outcome_kind="invalid_tool_name",
                block_source="validation",
                retryable=False,
            )

        # Check if tool exists
        tool_name, normalized_args, strategy = self._normalize_tool_call(
            tool_name, raw_args, context=context
        )

        if not self.tools.is_tool_enabled(tool_name):
            return _build_skip_result(
                tool_name=tool_name,
                arguments=normalized_args,
                success=False,
                skip_reason=f"Unknown or disabled tool: {tool_name}",
                outcome_kind="tool_unavailable",
                block_source="tool_registry",
                retryable=False,
            )

        redirected_args = self._apply_path_redirect(tool_name, normalized_args)
        if redirected_args is not None:
            original_path = normalized_args.get(self._path_arg_key(normalized_args) or "path")
            redirected_path = redirected_args.get(self._path_arg_key(redirected_args) or "path")
            logger.info(
                "[Pipeline] Rewriting %s path from previous suggestion: %s -> %s",
                tool_name,
                original_path,
                redirected_path,
            )
            normalized_args = redirected_args
            if strategy == NormalizationStrategy.DIRECT:
                strategy = NormalizationStrategy.MANUAL_REPAIR

        # Check permission policy (if configured)
        if self._permission_policy is not None:
            decision = self._permission_policy.authorize(tool_name, normalized_args)
            if not decision.allowed and not decision.needs_prompt:
                return _build_skip_result(
                    tool_name=tool_name,
                    arguments=normalized_args,
                    success=False,
                    skip_reason=f"Permission denied: {decision.reason}",
                    outcome_kind="permission_denied",
                    block_source="permission_policy",
                    retryable=False,
                )
            # needs_prompt tools are allowed through — the UI layer
            # handles escalation prompting before reaching the pipeline.

        # Check budget
        if self._calls_used >= self.config.tool_budget:
            return _build_skip_result(
                tool_name=tool_name,
                arguments=normalized_args,
                success=False,
                skip_reason="Tool budget exhausted",
                outcome_kind="budget_exhausted",
                block_source="tool_budget",
                retryable=False,
            )

        normalization_applied = None if strategy == NormalizationStrategy.DIRECT else strategy.value
        steering_user_message: Optional[str] = None

        # Check idempotent cache for read-only tools
        # This prevents DeepSeek/Ollama from re-reading the same file multiple times
        cached_result = self.get_cached_result(tool_name, normalized_args)
        if cached_result is not None:
            logger.info(f"[Pipeline] Returning cached result for {tool_name}")
            # Still count as a tool call for tracking, but don't execute
            self._executed_tools.append(tool_name)
            # Notify callbacks about cached result
            if self.on_tool_start:
                try:
                    self.on_tool_start(tool_name, normalized_args)
                except Exception as e:
                    logger.warning(f"on_tool_start callback failed: {e}")
            if self.on_tool_complete:
                try:
                    self.on_tool_complete(cached_result)
                except Exception as e:
                    logger.warning(f"on_tool_complete callback failed: {e}")
            return cached_result

        # Cross-turn dedup: cache "effectively idempotent" tools (web_search, etc.)
        # These produce identical results for identical args within a session window.
        if self._cross_turn_enabled and tool_name in CROSS_TURN_DEDUP_TOOLS:
            signature = self._get_call_signature(tool_name, normalized_args)
            cross_cached = self._cross_turn_cache.get(signature)
            if cross_cached is not None:
                self._cross_turn_hits += 1
                logger.info(
                    "[Pipeline] Cross-turn dedup hit for %s (hits=%d)",
                    tool_name,
                    self._cross_turn_hits,
                )
                self._executed_tools.append(tool_name)
                if self.on_tool_start:
                    try:
                        self.on_tool_start(tool_name, normalized_args)
                    except Exception as e:
                        logger.warning(f"on_tool_start callback failed: {e}")
                if self.on_tool_complete:
                    try:
                        self.on_tool_complete(cross_cached)
                    except Exception as e:
                        logger.warning(f"on_tool_complete callback failed: {e}")
                return cross_cached

        # Check semantic cache (FAISS-based with mtime invalidation)
        # This catches similar-but-not-identical queries that would return same results
        if self.config.enable_semantic_caching and self.semantic_cache is not None:
            try:
                semantic_result = await self.semantic_cache.get(tool_name, normalized_args)
                if semantic_result is not None:
                    logger.info(f"[Pipeline] Semantic cache hit for {tool_name}")
                    self._executed_tools.append(tool_name)
                    # Build result from cached data
                    sem_cached_result = ToolCallResult(
                        tool_name=tool_name,
                        arguments=normalized_args,
                        success=True,
                        result=semantic_result,
                        cached=True,
                        normalization_applied=normalization_applied,
                    )
                    if self.on_tool_start:
                        try:
                            self.on_tool_start(tool_name, normalized_args)
                        except Exception as e:
                            logger.warning(f"on_tool_start callback failed: {e}")
                    if self.on_tool_complete:
                        try:
                            self.on_tool_complete(sem_cached_result)
                        except Exception as e:
                            logger.warning(f"on_tool_complete callback failed: {e}")
                    return sem_cached_result
            except (TimeoutError, asyncio.TimeoutError):
                logger.debug("Semantic cache timeout, skipping")
            except (OSError, ValueError) as e:
                logger.warning(f"Semantic cache error (may indicate corruption): {e}")

        # Session-level file read dedup - prevents re-reading files even if cache was cleared
        # This is a fallback for the prompting loop fix when idempotent cache doesn't match
        # NOTE: Only dedup reads of the EXACT same file+args within a short window.
        # Do NOT block re-reads after the file has been edited (agent needs to verify).
        # Allow re-reads with different offset/limit (reading different sections).
        canonical_core_tool = canonicalize_core_tool_name(tool_name.lower())

        if canonical_core_tool == "read":
            file_path = normalized_args.get("path") or normalized_args.get("file_path")
            offset = normalized_args.get("offset")
            limit = normalized_args.get("limit")
            if self._should_steer_broad_code_read(
                file_path=file_path,
                offset=offset,
                limit=limit,
                context=context,
            ):
                rewrite = self._select_broad_code_read_rewrite(
                    str(file_path),
                    context=context,
                )
                if rewrite is None:
                    logger.info(
                        "[Pipeline] Steering broad code read of %s toward structure-aware guidance",
                        file_path,
                    )
                    return _build_skip_result(
                        tool_name=tool_name,
                        arguments=normalized_args,
                        success=True,
                        result=self._build_broad_code_read_recovery(str(file_path)),
                        skip_reason="Broad code-file read before structure-aware navigation",
                        outcome_kind="broad_code_read",
                        block_source="code_intelligence_first",
                        retryable=True,
                        user_message=(
                            f"Use symbol, refs, diagnostics, or project overview before reading "
                            f"all of '{file_path}'."
                        ),
                        normalization_applied=normalization_applied,
                    )
                tool_name, normalized_args = rewrite
                canonical_core_tool = canonicalize_core_tool_name(tool_name.lower())
                steering_user_message = (
                    f"Steered broad read(path=...) on '{file_path}' to "
                    f"{tool_name}({', '.join(f'{key}={_format_tool_command_value(value)}' for key, value in normalized_args.items())})."
                )
                logger.info(
                    "[Pipeline] Rewriting broad code read of %s to %s",
                    file_path,
                    tool_name,
                )
            else:
                # Check if agent explicitly provided offset/limit in the original call
                # (not just defaults from argument normalization). ``raw_args`` may be
                # None when a tool call arrives without an arguments object; treat that
                # as "no explicit offset/limit" rather than crashing.
                raw_args_map = raw_args if isinstance(raw_args, dict) else {}
                explicit_offset = "offset" in raw_args_map
                explicit_limit = "limit" in raw_args_map
                follow_up_rewrite = (
                    self._select_follow_up_code_read_rewrite(
                        str(file_path),
                        offset=offset if explicit_offset else None,
                        limit=limit if explicit_limit else None,
                        context=context,
                    )
                    if isinstance(file_path, str) and file_path.strip()
                    else None
                )
                if follow_up_rewrite is not None:
                    tool_name, normalized_args, steering_user_message = follow_up_rewrite
                    canonical_core_tool = canonicalize_core_tool_name(tool_name.lower())
                    logger.info(
                        "[Pipeline] Rewriting broad code read of %s using recent navigation %s",
                        file_path,
                        tool_name,
                    )
                # Only dedup exact same file+offset+limit reads
                dedup_key = f"{file_path}:{offset}:{limit}" if file_path else None
                if follow_up_rewrite is None and dedup_key and self._is_duplicate_read(dedup_key):
                    logger.info(
                        f"[Pipeline] Skipping duplicate read of {file_path} "
                        "(file was read recently in this session)"
                    )
                    return _build_skip_result(
                        tool_name=tool_name,
                        arguments=normalized_args,
                        success=True,  # Mark as success but indicate it was cached
                        result=(
                            f"[File '{file_path}' was already read in this session and "
                            "content is unchanged. Do not repeat the same read(path, offset, limit). "
                            "Use a different offset/limit, read a different file, or switch tools "
                            "such as symbol, refs, lsp, code_search, graph, or project_overview.]"
                        ),
                        skip_reason="Duplicate file read within session",
                        outcome_kind="duplicate_read",
                        block_source="session_read_dedup",
                        retryable=True,
                        user_message=(
                            f"File '{file_path}' was already read with the same offset/limit. "
                            "Choose a different range, file, or tool."
                        ),
                        normalization_applied=normalization_applied,
                    )

        # Invalidate read cache for files that were just edited/written
        if canonical_core_tool in {"edit", "write"}:
            edited_path = normalized_args.get("path") or normalized_args.get("file_path")
            if edited_path:
                self._clear_read_tracking_for_file(edited_path)

        # NOTE: code correction runs in the ToolExecutor (the choke point covering all
        # call paths), not here. The pipeline-level correction path was dead code
        # (enable_code_correction defaulted False, code_correction_middleware never wired)
        # and is removed; see FEP-0024.

        # Check for repeated failures
        if self.config.enable_failed_signature_tracking:
            signature = self._get_call_signature(tool_name, normalized_args)
            if signature in self._failed_signatures:
                logger.info(
                    "[Pipeline] Skipping repeated failing call: %s(%s)",
                    tool_name,
                    normalized_args,
                )
                recovery_result = _build_repeated_failure_recovery(tool_name, normalized_args)
                return _build_skip_result(
                    tool_name=tool_name,
                    arguments=normalized_args,
                    success=False,
                    result=recovery_result,
                    error=recovery_result.get("error"),
                    skip_reason="Repeated failing call with same arguments",
                    outcome_kind="repeated_failure",
                    block_source="failed_signature_tracker",
                    retryable=True,
                    user_message=(
                        "This exact tool call already failed recently. Use the recovery "
                        "guidance or try different arguments."
                    ),
                    normalization_applied=normalization_applied,
                )

        # Check for redundant tool calls using deduplication tracker
        if self.deduplication_tracker is not None:
            try:
                if self.deduplication_tracker.is_redundant(tool_name, normalized_args):
                    logger.info(
                        f"[Pipeline] Skipping redundant tool call: {tool_name} "
                        f"(semantic overlap with recent calls)"
                    )
                    recovery_result = _build_redundant_call_recovery(tool_name, normalized_args)
                    return _build_skip_result(
                        tool_name=tool_name,
                        arguments=normalized_args,
                        success=True,
                        result=recovery_result,
                        error=recovery_result.get("error"),
                        skip_reason="Redundant call (semantic overlap with recent operations)",
                        outcome_kind="redundant_call",
                        block_source="semantic_deduplication",
                        retryable=True,
                        user_message=(
                            "This tool call overlaps with recent work. Use the suggested next "
                            "step instead of repeating it unchanged."
                        ),
                        normalization_applied=normalization_applied,
                    )
            except (ValueError, TypeError) as e:
                logger.warning(f"Deduplication tracker check failed (data error): {e}")
            except AttributeError as e:
                logger.debug(f"Deduplication tracker not properly initialized: {e}")

        # Process through middleware chain (before execution)
        if self.middleware_chain is not None:
            try:
                before_result = await self.middleware_chain.process_before(
                    tool_name, normalized_args
                )
                if not before_result.proceed:
                    logger.info(
                        f"[Pipeline] Tool '{tool_name}' blocked by middleware: "
                        f"{before_result.error_message}"
                    )
                    return _build_skip_result(
                        tool_name=tool_name,
                        arguments=normalized_args,
                        success=False,
                        skip_reason=f"Blocked by middleware: {before_result.error_message}",
                        outcome_kind="middleware_blocked",
                        block_source="middleware_chain",
                        retryable=True,
                        user_message=before_result.error_message,
                        normalization_applied=normalization_applied,
                    )
                # Apply any argument modifications from middleware
                if before_result.modified_arguments:
                    normalized_args = before_result.modified_arguments
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"Middleware chain process_before failed (data error): {e}")
            except AttributeError as e:
                logger.debug(f"Middleware chain not properly configured: {e}")

        # Emit pre-execution intent event (LogAct-inspired)
        self._emit_tool_intent(tool_name, normalized_args)

        # Notify start
        if self.on_tool_start:
            try:
                self.on_tool_start(tool_name, normalized_args)
            except Exception as e:
                logger.warning(f"on_tool_start callback failed: {e}")

        # Execute with per-tool timeout
        # Log tool call sent to LLM
        logger.debug(
            "[ToolCall→LLM] Executing tool=%s args=%s",
            tool_name,
            json_dumps(normalized_args, default=str)[:500],
        )
        start_time = time.monotonic()
        try:
            exec_result = await asyncio.wait_for(
                self.executor.execute(
                    tool_name=tool_name,
                    arguments=normalized_args,
                    context=context,
                ),
                timeout=self._get_tool_timeout(tool_name),
            )
        except asyncio.TimeoutError:
            effective_timeout = self._get_tool_timeout(tool_name)
            tb_str = traceback.format_exc()

            # Build user-friendly error message with command details
            if tool_name == "shell" and "cmd" in normalized_args:
                cmd = normalized_args.get("cmd", "unknown command")
                error_msg = f"Command timed out after {effective_timeout}s: {cmd}"
                suggestion = (
                    "The command may be hung or waiting for input.\n"
                    "Try: Run the command manually to check if it's interactive, "
                    "increase the per-tool timeout setting, or use a non-interactive alternative."
                )
            else:
                error_msg = f"Tool '{tool_name}' timed out after {effective_timeout}s"
                suggestion = (
                    f"The tool execution exceeded the time limit of {effective_timeout}s.\n"
                    "Try: Increase the per-tool timeout setting or simplify the operation."
                )

            # Log technical details for debugging (hidden from user)
            logger.warning(
                f"[Pipeline] Tool '{tool_name}' timed out after {effective_timeout}s",
                exc_info=False,  # Don't log full traceback to reduce noise
            )

            exec_result = ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=error_msg,
                error_info=ErrorInfo(
                    message=error_msg,
                    category=ErrorCategory.TOOL_TIMEOUT,
                    severity=ErrorSeverity.WARNING,
                    correlation_id=f"timeout_{tool_name}_{int(time.time())}",
                    traceback=tb_str[-2000:],  # Last 2000 chars
                    details={
                        "tool_name": tool_name,
                        "timeout_seconds": effective_timeout,
                        "arguments": normalized_args,
                        "suggestion": suggestion,
                    },
                ),
            )
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.error(f"[Pipeline] Tool '{tool_name}' execution failed: {e}", exc_info=True)
            exec_result = ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=str(e),
                error_info=ErrorInfo(
                    message=f"Tool '{tool_name}' execution failed: {str(e)}",
                    category=ErrorCategory.TOOL_EXECUTION,
                    severity=ErrorSeverity.ERROR,
                    correlation_id=f"error_{tool_name}_{int(time.time())}",
                    traceback=tb_str[-2000:],  # Last 2000 chars
                    details={
                        "tool_name": tool_name,
                        "exception_type": type(e).__name__,
                        "arguments": normalized_args,
                    },
                    original_exception=str(e),
                ),
            )
        execution_time_ms = (time.monotonic() - start_time) * 1000

        # Update state
        self._calls_used += 1
        self._executed_tools.append(tool_name)

        # Build result
        call_result = ToolCallResult(
            tool_name=tool_name,
            arguments=normalized_args,
            success=exec_result.success,
            result=exec_result.result,
            error=exec_result.error,
            execution_time_ms=execution_time_ms,
            normalization_applied=normalization_applied,
            user_message=steering_user_message,
            error_info=exec_result.error_info,  # Preserve structured error info
        )

        low_signal_reason = None
        if steering_user_message is not None and exec_result.success:
            low_signal_reason = self._get_low_signal_broad_read_rewrite_reason(
                tool_name=tool_name,
                result=exec_result.result,
            )
        if low_signal_reason is not None:
            raw_file_path = normalized_args.get("file_path") or normalized_args.get("path")
            file_path = str(raw_file_path).strip() if raw_file_path is not None else ""
            if file_path:
                fallback_rewrite = self._select_broad_code_read_rewrite(
                    file_path,
                    context=context,
                    exclude_tools={tool_name},
                )
                if fallback_rewrite is not None and fallback_rewrite[0] != tool_name:
                    fallback_tool_name, fallback_args = fallback_rewrite
                    logger.info(
                        "[Pipeline] Retrying low-signal broad read rewrite from %s to %s",
                        tool_name,
                        fallback_tool_name,
                    )
                    recovered_exec_result = await asyncio.wait_for(
                        self.executor.execute(
                            tool_name=fallback_tool_name,
                            arguments=fallback_args,
                            context=context,
                        ),
                        timeout=self._get_tool_timeout(fallback_tool_name),
                    )
                    if recovered_exec_result.success:
                        tool_name = fallback_tool_name
                        normalized_args = fallback_args
                        exec_result = recovered_exec_result
                        call_result = ToolCallResult(
                            tool_name=fallback_tool_name,
                            arguments=fallback_args,
                            success=recovered_exec_result.success,
                            result=recovered_exec_result.result,
                            error=recovered_exec_result.error,
                            execution_time_ms=((time.monotonic() - start_time) * 1000),
                            normalization_applied=normalization_applied,
                            user_message=(
                                f"{steering_user_message} {low_signal_reason}, so the "
                                f"pipeline narrowed further with {fallback_tool_name}."
                            ),
                            error_info=recovered_exec_result.error_info,
                        )

        # Log tool result returned to LLM
        result_preview = str(exec_result.result)[:500] if exec_result.result else "(empty)"
        logger.debug(
            "[ToolResult→LLM] tool=%s success=%s time=%.0fms result_preview=%s",
            tool_name,
            exec_result.success,
            execution_time_ms,
            result_preview,
        )

        # Attempt error recovery fallback on failure
        if not exec_result.success and exec_result.error:
            self._record_path_redirect_from_error(tool_name, normalized_args, exec_result.error)
            try:
                from victor.agent.error_recovery import (
                    recover_from_error,
                    RecoveryAction,
                )

                recovery = recover_from_error(
                    Exception(exec_result.error), tool_name, normalized_args
                )
                recovery_tool_name = tool_name
                recovery_args = normalized_args

                if (
                    recovery.action
                    in {
                        RecoveryAction.RETRY_WITH_INFERRED,
                        RecoveryAction.RETRY_WITH_DEFAULTS,
                    }
                    and recovery.modified_args
                ):
                    recovery_args = recovery.modified_args
                    logger.info(
                        "[Pipeline] Retrying %s with recovered arguments: %s",
                        tool_name,
                        recovery_args,
                    )
                elif recovery.action == RecoveryAction.FALLBACK_TOOL and recovery.fallback_tool:
                    fallback_name = recovery.fallback_tool
                    if self.tools.is_tool_enabled(fallback_name):
                        recovery_tool_name = fallback_name
                        logger.info(
                            "[Pipeline] Falling back from %s to %s",
                            tool_name,
                            fallback_name,
                        )

                if recovery_tool_name != tool_name or recovery_args is not normalized_args:
                    try:
                        recovered_exec_result = await asyncio.wait_for(
                            self.executor.execute(
                                tool_name=recovery_tool_name,
                                arguments=recovery_args,
                                context=context,
                            ),
                            timeout=self._get_tool_timeout(recovery_tool_name),
                        )
                    except asyncio.TimeoutError:
                        recovered_exec_result = None

                    if recovered_exec_result is not None and recovered_exec_result.success:
                        tool_name = recovery_tool_name
                        normalized_args = recovery_args
                        exec_result = recovered_exec_result
                        call_result = ToolCallResult(
                            tool_name=recovery_tool_name,
                            arguments=recovery_args,
                            success=recovered_exec_result.success,
                            result=recovered_exec_result.result,
                            error=recovered_exec_result.error,
                            execution_time_ms=((time.monotonic() - start_time) * 1000),
                            normalization_applied=normalization_applied,
                            user_message=steering_user_message,
                            error_info=recovered_exec_result.error_info,
                        )
            except Exception as e:
                logger.debug(f"[Pipeline] Error recovery fallback failed: {e}")

        if self.on_tool_event:
            try:
                payload = {
                    "tool_name": tool_name,
                    "success": exec_result.success,
                    "execution_time_ms": execution_time_ms,
                    "arguments": normalized_args,
                }
                if exec_result.error:
                    payload["error"] = exec_result.error
                if exec_result.result is not None:
                    payload["result"] = exec_result.result
                self.on_tool_event("tool.raw_result", payload)
            except Exception as e:
                logger.debug(f"on_tool_event callback failed for raw result: {e}")

        # Process through middleware chain (after execution)
        if self.middleware_chain is not None:
            try:
                modified_result = await self.middleware_chain.process_after(
                    tool_name, normalized_args, call_result.result, call_result.success
                )
                if modified_result is not None and modified_result != call_result.result:
                    call_result = ToolCallResult(
                        tool_name=call_result.tool_name,
                        arguments=call_result.arguments,
                        success=call_result.success,
                        result=modified_result,
                        error=call_result.error,
                        execution_time_ms=call_result.execution_time_ms,
                        normalization_applied=call_result.normalization_applied,
                    )
                    if self.on_tool_event:
                        try:
                            self.on_tool_event(
                                "tool.middleware_adjusted",
                                {
                                    "tool_name": call_result.tool_name,
                                    "description": "Result modified by middleware chain",
                                },
                            )
                        except Exception as e:
                            logger.debug(f"on_tool_event middleware notification failed: {e}")
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"Middleware chain process_after failed (data error): {e}")
            except AttributeError as e:
                logger.debug(f"Middleware chain not properly configured: {e}")

        # Record file read for deduplication (prompting loop fix)
        # Include capitalized variants for tool name normalization
        if call_result.success:
            self._record_code_navigation_hints(
                call_result.tool_name,
                call_result.arguments,
                call_result.result,
            )
        if call_result.success and canonical_core_tool == "read":
            file_path = normalized_args.get("path") or normalized_args.get("file_path")
            if file_path:
                offset = normalized_args.get("offset")
                limit = normalized_args.get("limit")
                self.record_file_read(f"{file_path}:{offset}:{limit}", file_path=file_path)

        # Track failed signatures
        if not exec_result.success and self.config.enable_failed_signature_tracking:
            signature = self._get_call_signature(tool_name, normalized_args)
            self._failed_signatures.add(signature)

        # Cache successful results for idempotent tools
        if call_result.success:
            self.cache_result(tool_name, normalized_args, call_result)

            # Cross-turn dedup: cache results for effectively-idempotent tools
            if self._cross_turn_enabled and tool_name in CROSS_TURN_DEDUP_TOOLS:
                signature = self._get_call_signature(tool_name, normalized_args)
                dedup_result = ToolCallResult(
                    tool_name=call_result.tool_name,
                    arguments=call_result.arguments,
                    success=call_result.success,
                    result=call_result.result,
                    error=call_result.error,
                    execution_time_ms=0.0,
                    cached=True,
                    normalization_applied=call_result.normalization_applied,
                )
                self._cross_turn_cache.set(signature, dedup_result)

            # Store in semantic cache (FAISS-based with mtime tracking)
            if self.config.enable_semantic_caching and self.semantic_cache is not None:
                try:
                    # Get file path for mtime tracking
                    file_path = normalized_args.get("path") or normalized_args.get("file_path")
                    await self.semantic_cache.put(
                        tool_name,
                        normalized_args,
                        call_result.result,
                        file_path=file_path,
                    )
                except (OSError, IOError) as e:
                    logger.debug(f"Semantic cache store failed (I/O error): {e}")
                except (ValueError, TypeError) as e:
                    logger.debug(f"Semantic cache store failed (data error): {e}")

        # Invalidate caches when files are modified (write/edit tools)
        if call_result.success and canonicalize_core_tool_name(tool_name.lower()) in (
            "write",
            "edit",
        ):
            file_path = normalized_args.get("path") or normalized_args.get("file_path")
            if file_path:
                self._invalidate_post_edit_freshness_state(file_path, context=context)
                logger.debug(f"Invalidated caches for modified file: {file_path}")

        # Record successful tool call in deduplication tracker
        if call_result.success and self.deduplication_tracker is not None:
            try:
                self.deduplication_tracker.add_call(tool_name, normalized_args)
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to record call in deduplication tracker (data error): {e}")
            except AttributeError as e:
                logger.debug(f"Deduplication tracker not properly initialized: {e}")

        # Update analytics
        if self.config.enable_analytics:
            self._update_analytics(call_result)

        # Notify complete
        if self.on_tool_complete:
            try:
                self.on_tool_complete(call_result)
            except Exception as e:
                logger.warning(f"on_tool_complete callback failed: {e}")
        elif self.on_tool_event:
            # Emit a simple fallback event so UI can still react
            try:
                self.on_tool_event(
                    "tool.complete",
                    {
                        "tool_name": call_result.tool_name,
                        "success": call_result.success,
                        "execution_time_ms": call_result.execution_time_ms,
                    },
                )
            except Exception:
                logger.debug("on_tool_event fallback emission failed", exc_info=True)

        # Record tool result for credit assignment (FEP-0001 Phase 3)
        if self._credit_tracking_service is not None:
            try:
                self._credit_tracking_service.record_tool_result(
                    tool_name=call_result.tool_name,
                    success=call_result.success,
                    execution_time_ms=call_result.execution_time_ms,
                    error=call_result.error,
                    arguments=call_result.arguments,
                    agent_id=context.get("agent_id", "default"),
                    team_id=context.get("team_id"),
                    session_id=context.get("session_id"),
                )
            except Exception:
                pass  # Credit tracking is non-critical

        # Update online tool reputation (mid-turn feedback)
        if self._tool_reputation is not None:
            try:
                self._tool_reputation.record(
                    tool_name=call_result.tool_name,
                    success=call_result.success,
                    duration_ms=call_result.execution_time_ms,
                )
            except Exception:
                pass  # Reputation tracking is non-critical

        return call_result

    def _update_analytics(self, result: ToolCallResult) -> None:
        """Update tool analytics.

        Args:
            result: Tool call result
        """
        tool_name = result.tool_name
        if tool_name not in self._tool_stats:
            self._tool_stats[tool_name] = {
                "calls": 0,
                "successes": 0,
                "failures": 0,
                "total_time_ms": 0.0,
            }

        stats = self._tool_stats[tool_name]
        stats["calls"] += 1
        if result.success:
            stats["successes"] += 1
        else:
            stats["failures"] += 1
        stats["total_time_ms"] += result.execution_time_ms

    def get_analytics(self) -> Dict[str, Any]:
        """Get tool execution analytics.

        Returns:
            Dictionary with analytics data
        """
        return {
            "total_calls": self._calls_used,
            "budget": self.config.tool_budget,
            "remaining": self.calls_remaining,
            "tools": dict(self._tool_stats),
        }

    def _get_tool_timeout(self, tool_name: str) -> float:
        """Get effective timeout for a tool, preferring per-tool override.

        Resolution order:
        1. Per-tool timeout from @tool(timeout=N) decorator (_tool_timeout attribute)
        2. Per-tool timeout from ToolDefinition.timeout / timeout_seconds
        3. Pipeline default (config.per_tool_timeout_seconds)
        """
        try:
            tool_func = self.executor.get_tool_function(tool_name)
            if tool_func:
                # Check raw function attribute (set by @tool decorator)
                if hasattr(tool_func, "_tool_timeout"):
                    per_tool = tool_func._tool_timeout
                    if isinstance(per_tool, (int, float)) and per_tool > 0:
                        return float(per_tool)
                # Check ToolDefinition object (wrapped by registry)
                if hasattr(tool_func, "timeout"):
                    per_tool = tool_func.timeout
                    if isinstance(per_tool, (int, float)) and per_tool > 0:
                        return float(per_tool)
                if hasattr(tool_func, "timeout_seconds"):
                    per_tool = tool_func.timeout_seconds
                    if isinstance(per_tool, (int, float)) and per_tool > 0:
                        return float(per_tool)
        except Exception:
            pass
        return self.config.per_tool_timeout_seconds

    def clear_failed_signatures(self) -> None:
        """Clear the failed signature cache."""
        self._failed_signatures.clear()
