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

"""Rust context fitter wrapper.

Provides a protocol-compliant wrapper around the Rust context fitting
functions. The wrapper delegates to victor_native functions while
maintaining the ContextFitterProtocol interface.

Performance characteristics:
- fit_context: 2-4x faster (Rust sorting and selection)
- truncate_message: 2-5x faster (BPE-aware truncation)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import victor_native

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


class RustContextFitter(InstrumentedAccelerator):
    """Rust implementation of ContextFitterProtocol.

    Wraps the high-performance Rust context fitting functions
    with protocol-compliant interface.

    Performance characteristics:
    - fit_context: 2-4x faster (Rust sorting and selection)
    - truncate_message: 2-5x faster (BPE-aware truncation)
    """

    def __init__(self) -> None:
        super().__init__(backend="rust")
        self._version = victor_native.__version__

    def get_version(self) -> Optional[str]:
        return self._version

    def fit_context(
        self,
        messages: List[Dict[str, Any]],
        budget: int,
        strategy: str = "recency",
        preserve_system: bool = True,
    ) -> FitResult:
        """Fit messages into a token budget.

        Delegates to Rust implementation for high-performance fitting.

        Args:
            messages: List of message dicts with 'role', 'content', and
                      optionally 'token_count' and 'priority' fields
            budget: Maximum token budget
            strategy: Fitting strategy ("recency", "priority", "balanced")
            preserve_system: Whether to always preserve system messages

        Returns:
            FitResult with indices of kept messages and statistics
        """
        with self._timed_call("context_fitting"):
            # Build MessageSlot objects for Rust
            slots = []
            for i, msg in enumerate(messages):
                token_count = msg.get("token_count", len(msg.get("content", "").split()) * 13 // 10)
                priority = msg.get("priority", 1.0)
                role = msg.get("role", "user")
                recency = float(i) / max(len(messages), 1)
                slot = victor_native.MessageSlot(
                    index=i,
                    token_count=token_count,
                    priority=priority,
                    role=role,
                    recency=recency,
                )
                slots.append(slot)

            result = victor_native.fit_context(slots, budget, strategy, preserve_system)
            return FitResult(
                kept_indices=list(result.kept_indices),
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

        Delegates to Rust BPE-aware truncation.

        Args:
            content: Message content to truncate
            max_tokens: Maximum number of tokens allowed
            preserve_lines: Whether to truncate at line boundaries

        Returns:
            Truncated content string
        """
        with self._timed_call("message_truncation"):
            return victor_native.truncate_message(content, max_tokens, preserve_lines)
