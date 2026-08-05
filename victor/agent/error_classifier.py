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

"""Error classification for intelligent tool retry decisions.

Classifies errors as permanent, transient, or retryable to prevent
wasting agent turns on operations that will never succeed.

SOLID Principles:
- SRP: Single responsibility of error classification
- OCP: New error patterns can be added without modifying classification logic
"""

import logging
import re
from functools import lru_cache

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, FrozenSet, Set

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Classification of tool execution errors."""

    PERMANENT = "permanent"
    """Errors that will never succeed on retry (e.g., file not found)."""

    TRANSIENT = "transient"
    """Errors that might succeed later (e.g., network timeout, rate limit)."""

    RETRYABLE = "retryable"
    """Errors where fixing the input might help (e.g., syntax error)."""


@dataclass
class ToolCallSignature:
    """Unique signature for a tool call to detect duplicates."""

    tool_name: str
    arguments_hash: int

    @classmethod
    def from_call(cls, tool_name: str, arguments: dict[str, Any]) -> "ToolCallSignature":
        """Create signature from tool call parameters."""
        # Create a hashable representation of arguments
        # Sort keys for consistent hashing
        try:
            args_tuple = tuple(sorted((k, str(v)) for k, v in arguments.items()))
            args_hash = hash(args_tuple)
        except (TypeError, ValueError):
            # Fallback for unhashable values
            args_hash = hash(str(arguments))

        return cls(tool_name=tool_name, arguments_hash=args_hash)

    def __hash__(self) -> int:
        return hash((self.tool_name, self.arguments_hash))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToolCallSignature):
            return False
        return self.tool_name == other.tool_name and self.arguments_hash == other.arguments_hash


class ToolErrorClassifier:
    """Classifies tool execution errors for retry decisions.

    Tracks permanently failed tool calls to prevent retrying
    operations that will never succeed.

    Uses compiled regex patterns for precise matching, avoiding false positives
    from substring matching (e.g., "connection refused" matching unrelated text).
    """

    # Compiled regex patterns for permanent errors
    # These errors will never succeed on retry
    PERMANENT_PATTERNS: tuple[re.Pattern, ...] = (
        # File system errors
        re.compile(r"No such file or directory:?\s*[\'\"].*?[\'\"]", re.IGNORECASE),
        re.compile(r"FileNotFoundError:?\s*.+?", re.IGNORECASE),
        re.compile(r"\[Errno 2\]\s*No such file or directory", re.IGNORECASE),
        re.compile(r"IsADirectoryError:?\s*.+?", re.IGNORECASE),
        re.compile(r"NotADirectoryError:?\s*.+?", re.IGNORECASE),
        re.compile(r"directory not empty\b", re.IGNORECASE),
        re.compile(r"File exists:?\s*.+?", re.IGNORECASE),
        re.compile(r"read-only file system", re.IGNORECASE),
        # Permission errors
        re.compile(r"Permission denied:?\s*.+?", re.IGNORECASE),
        re.compile(r"PermissionError:?\s*.+?", re.IGNORECASE),
        re.compile(r"\[Errno 13\]\s*Permission denied", re.IGNORECASE),
        re.compile(r"Access denied\b", re.IGNORECASE),
        re.compile(r"403\b.*?Forbidden", re.IGNORECASE),  # HTTP 403 Forbidden
        # Module/import errors
        re.compile(r"ModuleNotFoundError:?\s*No module named\s+[\'\"].*?[\'\"]", re.IGNORECASE),
        re.compile(r"ImportError:?\s*No module named\s+[\'\"].*?[\'\"]", re.IGNORECASE),
        re.compile(
            r"cannot import name\s+[\'\"].*?[\'\"]\s+from\s+[\'\"].*?[\'\"]",
            re.IGNORECASE,
        ),
        # Command errors
        re.compile(r"command not found:?\s*\S+", re.IGNORECASE),
        re.compile(r"executable not found:?\s*\S+", re.IGNORECASE),
        re.compile(r"\[Errno 8\]\s*Exec format error", re.IGNORECASE),
        # Syntax errors (permanent without code changes)
        re.compile(r"SyntaxError:?\s*.+?", re.IGNORECASE),
        re.compile(r"IndentationError:?\s*.+?", re.IGNORECASE),
        # Type/conversion errors
        re.compile(r"TypeError:?\s*.+?", re.IGNORECASE),
        re.compile(r"ValueError:?\s*.+?", re.IGNORECASE),
        # Configuration errors
        re.compile(r"ConfigurationError:?\s*.+?", re.IGNORECASE),
        re.compile(r"ValidationError:?\s*.+?", re.IGNORECASE),
        # Docker/container errors
        re.compile(r"No such container\b", re.IGNORECASE),
        re.compile(r"No such image\b", re.IGNORECASE),
        re.compile(r"Container .*? not found\b", re.IGNORECASE),
        # Git errors
        re.compile(r"pathspec .*? did not match any file", re.IGNORECASE),
        # Edge case: errno names not found
        re.compile(r"\[Errno -2\]\s*No such file or directory", re.IGNORECASE),
        re.compile(r"getaddrinfo failed:?\s*Name or service not known", re.IGNORECASE),
        re.compile(r"nodename nor servname provided", re.IGNORECASE),
        # Edge case: SSL/TLS certificate errors
        re.compile(r"SSL: CERTIFICATE_VERIFY_FAILED", re.IGNORECASE),
        re.compile(r"certificate verify failed", re.IGNORECASE),
        re.compile(r"SSL: WRONG_VERSION_NUMBER", re.IGNORECASE),
        re.compile(r"TLS: wrong version number", re.IGNORECASE),
        re.compile(r"handshake failure", re.IGNORECASE),
        # Edge case: Windows-specific errors
        re.compile(r"\[WinError \d+\]", re.IGNORECASE),
        re.compile(r"\[Error \d+\]", re.IGNORECASE),
    )
    # Edge case: ECONNREFUSED (errno name not found) - TRANSIENT not permanent
    # These are in TRANSIENT_PATTERNS below
    # Edge case: SSL/TLS certificate errors

    # Compiled regex patterns for transient errors
    # These errors might succeed later (network issues, rate limits, etc.)
    TRANSIENT_PATTERNS: tuple[re.Pattern, ...] = (
        # Network connectivity errors - more precise patterns
        re.compile(r"^(?!.*refused to die).*Connection refused(?:\s|$)", re.IGNORECASE),
        re.compile(r"Connection timed out\b", re.IGNORECASE),
        re.compile(r"Connection reset by peer\b", re.IGNORECASE),
        re.compile(r"Network is unreachable\b", re.IGNORECASE),
        re.compile(r"Host unreachable\b", re.IGNORECASE),
        re.compile(r"No route to host\b", re.IGNORECASE),
        # HTTP errors
        re.compile(r"rate limit(?:ed)?\b", re.IGNORECASE),
        re.compile(r"too many requests\b", re.IGNORECASE),
        re.compile(r"429\b"),  # HTTP 429 Too Many Requests
        re.compile(r"500\b"),  # HTTP 500 Internal Server Error
        re.compile(r"502\b"),  # HTTP 502 Bad Gateway
        re.compile(r"503\b"),  # HTTP 503 Service Unavailable
        re.compile(r"504\b"),  # HTTP 504 Gateway Timeout
        re.compile(r"Service Unavailable\b", re.IGNORECASE),
        re.compile(r"Gateway Timeout\b", re.IGNORECASE),
        re.compile(r"Bad Gateway\b", re.IGNORECASE),
        re.compile(r"Internal Server Error\b", re.IGNORECASE),
        # Timeout errors
        re.compile(r"Request timed out\b", re.IGNORECASE),
        re.compile(r"Timeout exceeded\b", re.IGNORECASE),
        re.compile(r"timed out\b", re.IGNORECASE),
        re.compile(r"Network timeout\b", re.IGNORECASE),
        re.compile(r"timeout while connecting\b", re.IGNORECASE),
        # Temporary errors
        re.compile(r"temporary fail(?:ure|ed)?\b", re.IGNORECASE),
        re.compile(r"try again later\b", re.IGNORECASE),
        re.compile(r"temporarily unavailable\b", re.IGNORECASE),
        # SSL/TLS errors
        re.compile(r"SSL:?\s*.*?certificate verify failed", re.IGNORECASE),
        re.compile(r"TLS:?\s*.*?handshake failure", re.IGNORECASE),
        # Edge case: errno names not found (but still TRANSIENT)
        re.compile(r"\[Errno 111\]\s*Connection refused", re.IGNORECASE),
        re.compile(r"\[Errno 61\]\s*Connection refused", re.IGNORECASE),
        re.compile(r"ECONNREFUSED\b", re.IGNORECASE),
    )

    def __init__(self, decision_service=None, runtime_intelligence: Any = None) -> None:
        """Initialize the classifier with empty failure tracking.

        Args:
            decision_service: Optional LLMDecisionService for ambiguous error classification
            runtime_intelligence: Optional canonical runtime-intelligence service
        """
        self._failed_calls: Set[ToolCallSignature] = set()
        self._decision_service = decision_service
        self._runtime_intelligence = runtime_intelligence

    def _has_decision_support(self) -> bool:
        """Return whether low-confidence LLM decisions are available."""
        if self._runtime_intelligence is not None:
            return True
        return self._decision_service is not None

    def _decide_sync(
        self,
        decision_type: Any,
        context: dict[str, Any],
        *,
        heuristic_result: Any = None,
        heuristic_confidence: float = 0.0,
    ) -> Any:
        """Delegate to the canonical runtime-intelligence service when present."""
        if self._runtime_intelligence is not None:
            return self._runtime_intelligence.decide_sync(
                decision_type,
                context,
                heuristic_result=heuristic_result,
                heuristic_confidence=heuristic_confidence,
            )
        if self._decision_service is not None:
            return self._decision_service.decide_sync(
                decision_type,
                context,
                heuristic_result=heuristic_result,
                heuristic_confidence=heuristic_confidence,
            )
        return None

    @lru_cache(maxsize=512)
    def classify(self, error_message: str) -> ErrorType:
        """Classify an error message using regex patterns.

        Uses compiled regex patterns for precise matching, avoiding false positives.
        Results are cached to avoid repeated classification of the same error.

        Args:
            error_message: The error message from tool execution

        Returns:
            ErrorType indicating whether the error is permanent, transient, or retryable
        """
        # Check for permanent patterns first (highest priority)
        for pattern in self.PERMANENT_PATTERNS:
            if pattern.search(error_message):
                logger.debug(f"Error classified as PERMANENT: matched pattern '{pattern.pattern}'")
                return ErrorType.PERMANENT

        # Check for transient patterns
        for pattern in self.TRANSIENT_PATTERNS:
            if pattern.search(error_message):
                logger.debug(f"Error classified as TRANSIENT: matched pattern '{pattern.pattern}'")
                return ErrorType.TRANSIENT

        # LLM augmentation: when no pattern matches, consult LLM if available
        if self._has_decision_support():
            try:
                from victor.agent.decisions.chain import should_use_llm

                if not should_use_llm("error_classification"):
                    return ErrorType.RETRYABLE

                from victor.agent.decisions.schemas import DecisionType

                decision = self._decide_sync(
                    DecisionType.ERROR_CLASSIFICATION,
                    context={"error_message": error_message[:300]},
                    heuristic_result=ErrorType.RETRYABLE,
                    heuristic_confidence=0.4,
                )
                if (
                    decision is not None
                    and decision.result is not None
                    and decision.source == "llm"
                    and hasattr(decision.result, "error_type")
                ):
                    error_type = getattr(
                        decision.result.error_type, "value", decision.result.error_type
                    )
                    type_map = {
                        "permanent": ErrorType.PERMANENT,
                        "transient": ErrorType.TRANSIENT,
                        "retryable": ErrorType.RETRYABLE,
                    }
                    mapped = type_map.get(error_type)
                    if mapped is not None:
                        logger.debug(
                            "LLM classified error as %s (conf=%.2f)",
                            mapped.value,
                            decision.confidence,
                        )
                        return mapped
            except Exception:
                logger.debug("LLM error classification failed", exc_info=True)

        # Default to retryable (e.g., syntax errors can be fixed by agent)
        logger.debug("Error classified as RETRYABLE (no pattern matched)")
        return ErrorType.RETRYABLE

    def record_failure(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        error_message: str,
    ) -> ErrorType:
        """Record a tool failure and return its classification.

        If the error is permanent, the call signature is stored to
        prevent future retry attempts.

        Args:
            tool_name: Name of the failed tool
            arguments: Arguments passed to the tool
            error_message: The error message

        Returns:
            ErrorType classification of the error
        """
        error_type = self.classify(error_message)

        if error_type == ErrorType.PERMANENT:
            signature = ToolCallSignature.from_call(tool_name, arguments)
            self._failed_calls.add(signature)

        return error_type

    def should_skip(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> bool:
        """Check if a tool call should be skipped due to prior permanent failure.

        Args:
            tool_name: Name of the tool
            arguments: Arguments for the tool call

        Returns:
            True if this exact call previously failed permanently
        """
        signature = ToolCallSignature.from_call(tool_name, arguments)
        return signature in self._failed_calls

    def get_skip_reason(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str | None:
        """Get reason for skipping a tool call.

        Args:
            tool_name: Name of the tool
            arguments: Arguments for the tool call

        Returns:
            Skip reason message if call should be skipped, None otherwise
        """
        if self.should_skip(tool_name, arguments):
            return (
                f"Skipping {tool_name}: This exact call previously failed with a "
                "permanent error and will not be retried."
            )
        return None

    def reset(self) -> None:
        """Clear all recorded failures and classification cache."""
        self._failed_calls.clear()
        # Clear the LRU cache for classify method
        self.classify.cache_clear()

    @property
    def failed_call_count(self) -> int:
        """Number of unique permanently failed calls recorded."""
        return len(self._failed_calls)


# Singleton instance for global error tracking across orchestrator runs
_global_classifier: ToolErrorClassifier | None = None


def get_error_classifier(runtime_intelligence: Any = None) -> ToolErrorClassifier:
    """Get or create the global error classifier instance."""
    global _global_classifier
    if _global_classifier is None or (
        runtime_intelligence is not None
        and getattr(_global_classifier, "_runtime_intelligence", None) is not runtime_intelligence
    ):
        _global_classifier = ToolErrorClassifier(runtime_intelligence=runtime_intelligence)
    return _global_classifier


def reset_error_classifier() -> None:
    """Reset the global error classifier."""
    global _global_classifier
    if _global_classifier is not None:
        _global_classifier.reset()
