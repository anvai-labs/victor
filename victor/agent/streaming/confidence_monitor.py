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

"""Streaming confidence monitor for adaptive generation termination.

Detects when the LLM has committed to an answer using content heuristics
and token budget enforcement, enabling early stop without logprob access.

Based on: ATCC (arXiv 2603.13906) — adaptive generation termination.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import AsyncIterator, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceConfig:
    """Configuration for streaming confidence monitoring."""

    min_tokens: int = 20
    token_budget: Optional[int] = None
    completion_markers: Tuple[str, ...] = (
        "\n\n",
        "Therefore,",
        "In conclusion",
        "The answer is",
        "To summarize",
        "```\n\n",
        "Hope this helps",
        "Let me know if",
    )
    marker_window: int = 3


class StreamingConfidenceMonitor:
    """Detects when the LLM has committed to an answer and signals early stop.

    Uses two complementary heuristics when logprobs are unavailable:
    - Token budget enforcement: hard stop when completion_tokens >= token_budget
    - Natural completion markers: sentence-final patterns in recent chunks

    The monitor is stateful per-session and should be created fresh per turn.
    Call record() before should_stop() to update internal state from the chunk.

    Designed to be used as a transparent stream wrapper via wrap_stream().
    """

    def __init__(self, config: Optional[ConfidenceConfig] = None) -> None:
        self._config = config or ConfidenceConfig()
        self._tokens_generated: int = 0
        self._recent_content: List[str] = []

    def reset(self) -> None:
        """Reset state for a new turn."""
        self._tokens_generated = 0
        self._recent_content.clear()

    def record(self, content: str, completion_tokens: int = 0) -> None:
        """Update state from the current chunk or turn output.

        Args:
            content: Text content generated so far (or new chunk content)
            completion_tokens: Accurate completion token count from provider usage
        """
        if content:
            # Estimate tokens if not provided accurately
            self._tokens_generated = max(
                completion_tokens or 0,
                self._tokens_generated + max(1, len(content) // 4),
            )
            self._recent_content.append(content)
            if len(self._recent_content) > self._config.marker_window + 2:
                self._recent_content.pop(0)

    def should_stop(self) -> bool:
        """Return True if generation should stop now.

        Can be called after record() to decide whether to break the stream.
        Pure function — does not modify state.
        """
        # Hard stop: token budget from TaskTypeHint
        if self._config.token_budget and self._tokens_generated >= self._config.token_budget:
            logger.debug(
                "[ConfidenceMonitor] Token budget reached (%d >= %d)",
                self._tokens_generated,
                self._config.token_budget,
            )
            return True

        # Gate: don't stop before min_tokens
        if self._tokens_generated < self._config.min_tokens:
            return False

        # Heuristic: natural completion markers in recent window
        window = "".join(self._recent_content[-self._config.marker_window :])
        if any(m in window for m in self._config.completion_markers):
            logger.debug("[ConfidenceMonitor] Completion marker detected in window")
            return True

        return False

    async def wrap_stream(
        self,
        stream: AsyncIterator,
    ) -> AsyncIterator:
        """Transparent stream wrapper — yields chunks through, stops early when confident.

        Usage:
            monitored = monitor.wrap_stream(provider.stream_chat(...))
            async for chunk in monitored:
                yield chunk
        """
        async for chunk in stream:
            content = getattr(chunk, "content", "") or ""
            usage = getattr(chunk, "usage", {}) or {}
            completion_tokens = usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0

            self.record(content, completion_tokens)
            yield chunk

            is_final = getattr(chunk, "is_final", False)
            if not is_final and self.should_stop():
                logger.info("[ConfidenceMonitor] Early stop triggered — breaking stream")
                break
