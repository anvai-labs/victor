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

from typing import Any, Dict, List, Optional

from victor.native.observability import InstrumentedAccelerator
from victor.processing.native.context_fitter import (
    FitResult,
    _truncate_message_python,
)
from victor.processing.native.context_fitter import fit_context as _canonical_fit_context

__all__ = ["FitResult", "PythonContextFitter"]


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
        with self._timed_call("context_fitting"):
            return _canonical_fit_context(
                messages, budget, strategy=strategy, preserve_system=preserve_system
            )

    def truncate_message(
        self,
        content: str,
        max_tokens: int,
        preserve_lines: bool = True,
    ) -> str:
        """Truncate a message to fit within a token limit.

        Delegates to the canonical pure-Python truncation in
        ``victor.processing.native.context_fitter``.

        Args:
            content: Message content to truncate
            max_tokens: Maximum number of tokens allowed
            preserve_lines: Whether to truncate at line boundaries

        Returns:
            Truncated content string
        """
        with self._timed_call("message_truncation"):
            return _truncate_message_python(content, max_tokens, preserve_lines)
