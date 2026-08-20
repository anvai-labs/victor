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

"""LLM-powered compaction summarizer.

Uses a fast LLM call to produce rich abstractive summaries of compacted
messages, preserving intent, decisions, and pending tasks rather than
just keyword extraction.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, List, Optional, TYPE_CHECKING

from victor.providers.base import Message

if TYPE_CHECKING:
    from victor.providers.base import BaseProvider
    from victor.agent.compaction_summarizer import CompactionSummaryStrategy

logger = logging.getLogger(__name__)


class LLMCompactionSummarizer:
    """CompactionSummaryStrategy using a fast LLM call for rich summaries.

    Falls back to a provided fallback strategy (typically LedgerAwareCompactionSummarizer)
    on any failure, timeout, or when no provider is available.
    """

    def __init__(
        self,
        provider: "BaseProvider",
        model: str = "",
        max_input_chars: int = 8000,
        max_summary_tokens: int = 300,
        timeout_seconds: float = 10.0,
        fallback: Optional["CompactionSummaryStrategy"] = None,
    ):
        self._provider = provider
        self._model = model
        self._max_input_chars = max_input_chars
        self._max_summary_tokens = max_summary_tokens
        self._timeout_seconds = timeout_seconds
        self._fallback = fallback

    def summarize(self, removed_messages: List[Message], ledger: Optional[object] = None) -> str:
        """Implements CompactionSummaryStrategy protocol.

        Builds prompt from messages + ledger, calls provider.chat() with short timeout.
        On any failure, delegates to fallback.
        """
        if not removed_messages:
            return ""

        try:
            prompt = self._build_summary_prompt(removed_messages, ledger)
            messages = [Message(role="user", content=prompt)]

            # Try sync call with timeout
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # We're in an async context — use asyncio.wait_for
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(self._call_provider_sync, messages)
                    result = future.result(timeout=self._timeout_seconds)
            else:
                result = self._call_provider_sync(messages)

            if result:
                return f"[Compacted context: {result.strip()}]"

        except Exception as e:
            logger.debug(f"LLM summarization failed, using fallback: {e}")

        # Fallback
        if self._fallback:
            return self._fallback.summarize(removed_messages, ledger)

        return ""

    def _call_provider_sync(self, messages: List[Message]) -> str:
        """Call provider.chat() synchronously with optional model tiering."""
        target_model = self._model

        # Heuristic: use performance model for complex summaries (> 8 messages)
        # to ensure intent and decisions are preserved accurately.
        if not target_model:
            try:
                from victor.agent.model_switcher import get_model_switcher

                switcher = get_model_switcher()
                if len(messages) > 8:
                    target_model = switcher.get_thorough_model()
            except Exception:
                pass

        kwargs: dict = {
            "messages": messages,
            "max_tokens": self._max_summary_tokens,
            "temperature": 0.3,  # Lower temperature for factual summary
        }
        if target_model:
            kwargs["model"] = target_model

        response = self._provider.chat(**kwargs)

        # Handle both string and Message responses
        if isinstance(response, str):
            return response
        if hasattr(response, "content"):
            return response.content
        return str(response)

    def _build_summary_prompt(
        self, removed_messages: List[Message], ledger: Optional[object] = None
    ) -> str:
        """Build structured prompt for abstractive technical summarization."""
        parts = [
            "Summarize the following conversation segment concisely. "
            "Focus on: technical decisions made, files modified, specific method changes, and current task status. "
            "Preserve exact file paths and method names. "
            "The summary will be used as context for the next stage of implementation.\n"
        ]

        # Add conversation content (truncated)
        conv_parts = []
        total_chars = 0
        for msg in removed_messages:
            content = msg.content
            if total_chars + len(content) > self._max_input_chars:
                remaining = self._max_input_chars - total_chars
                if remaining > 100:
                    content = content[:remaining] + "..."
                else:
                    break
            conv_parts.append(f"{msg.role}: {content}")
            total_chars += len(content)

        parts.append("Conversation:\n" + "\n".join(conv_parts))

        # Add ledger entries if available
        if ledger is not None:
            entries = getattr(ledger, "entries", [])
            if entries:
                ledger_parts = []
                for entry in entries[:20]:
                    ledger_parts.append(f"- [{entry.category}] {entry.key}: {entry.summary}")
                parts.append("\nLedger entries:\n" + "\n".join(ledger_parts))

        return "\n".join(parts)
