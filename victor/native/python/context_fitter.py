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

"""Pure Python context fitter implementation.

Provides context window management for fitting messages into token budgets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from victor.native.observability import InstrumentedAccelerator


@dataclass
class FitResult:
    """Result of context fitting operation.

    Attributes:
        kept_indices: Indices of messages that fit within the budget
        total_tokens: Total token count of kept messages
        dropped_count: Number of messages dropped
        freed_tokens: Number of tokens freed by dropping messages
    """

    kept_indices: List[int]
    total_tokens: int
    dropped_count: int
    freed_tokens: int


class PythonContextFitter(InstrumentedAccelerator):
    """Pure Python implementation of ContextFitterProtocol.

    Provides context fitting using Python sorting and selection.
    """

    def __init__(self) -> None:
        super().__init__(backend="python")
        self._version = "1.0.0"

    def get_version(self) -> Optional[str]:
        return self._version

    def fit_context(
        self,
        messages: List[Dict[str, Any]],
        budget: int,
        strategy: str = "smart",
        preserve_system: bool = True,
    ) -> FitResult:
        """Fit messages into a token budget.

        Delegates to the canonical contract in
        ``victor.processing.native.context_fitter`` — this protocol-class
        implementation previously spoke the OLD strategy vocabulary
        (recency/priority/balanced), silently mapping "smart" and "fifo" to
        its balanced branch while the processing wrapper and Rust accepted
        smart/priority/fifo (adversarial-review finding).

        Args:
            messages: List of message dicts with 'role', 'content', and
                      optionally 'token_count' and 'priority' fields
            budget: Maximum token budget
            strategy: One of "smart", "priority", "fifo" (legacy aliases
                      accepted, unknown names raise ValueError)
            preserve_system: Whether to always preserve system messages

        Returns:
            FitResult with indices of kept messages and statistics
        """
        from victor.processing.native.context_fitter import fit_context as _canonical

        with self._timed_call("context_fitting"):
            result = _canonical(
                messages, budget, strategy=strategy, preserve_system=preserve_system
            )
            # Re-wrap in this module's FitResult type for protocol consumers.
            return FitResult(
                kept_indices=result.kept_indices,
                total_tokens=result.total_tokens,
                dropped_count=result.dropped_count,
                freed_tokens=result.freed_tokens,
            )

    def truncate_message(
        self,
        content: str,
        max_tokens: int,
        preserve_lines: bool = True,
    ) -> str:
        """Truncate a message to fit within a token limit.

        Uses line-based or word-based truncation.

        Args:
            content: Message content to truncate
            max_tokens: Maximum number of tokens allowed
            preserve_lines: Whether to truncate at line boundaries

        Returns:
            Truncated content string
        """
        with self._timed_call("message_truncation"):
            if not content:
                return content

            if preserve_lines:
                lines = content.split("\n")
                kept_lines = []
                current_tokens = 0

                for line in lines:
                    line_tokens = len(line.split()) * 13 // 10
                    if line_tokens == 0:
                        line_tokens = 1  # Empty lines still cost a token
                    if current_tokens + line_tokens > max_tokens:
                        break
                    kept_lines.append(line)
                    current_tokens += line_tokens

                return "\n".join(kept_lines)
            else:
                words = content.split()
                # Approximate: ~1.3 tokens per word
                max_words = max(1, max_tokens * 10 // 13)
                return " ".join(words[:max_words])
