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

"""Strategy-based compaction summary generation.

Provides pluggable strategies for generating summaries when messages are
compacted from the conversation history.
"""

from __future__ import annotations

import re
import logging
from typing import List, Optional, Protocol

from victor.providers.base import Message

logger = logging.getLogger(__name__)


class CompactionSummaryStrategy(Protocol):
    """Protocol for compaction summary strategies."""

    def summarize(
        self, removed_messages: List[Message], ledger: Optional[object] = None
    ) -> str: ...


class KeywordCompactionSummarizer:
    """Extracted from ConversationController._generate_compaction_summary().

    Uses keyword extraction to produce summaries like
    "3 user messages, 5 tool results, topics: yaml, config".
    """

    def summarize(self, removed_messages: List[Message], ledger: Optional[object] = None) -> str:
        if not removed_messages:
            return ""

        user_msgs = [m for m in removed_messages if m.role == "user"]
        tool_msgs = [m for m in removed_messages if m.role == "tool"]

        all_content = " ".join(m.content[:200] for m in removed_messages[:10])
        topics = self._extract_key_topics(all_content)

        parts = []
        if user_msgs:
            parts.append(f"{len(user_msgs)} user messages")
        if tool_msgs:
            parts.append(f"{len(tool_msgs)} tool results")
        if topics:
            parts.append(f"topics: {', '.join(topics[:3])}")

        if parts:
            return f"[Earlier conversation: {'; '.join(parts)}]"
        return ""

    def _extract_key_topics(self, text: str) -> List[str]:
        words = re.findall(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b", text)  # CamelCase
        words += re.findall(r"\b[a-z]+_[a-z_]+\b", text)  # snake_case
        words += re.findall(r"\b(?:def|class|function|file|error|test|api)\s+\w+", text.lower())

        seen: set = set()
        topics: List[str] = []
        for w in words:
            w_lower = w.lower()
            if w_lower not in seen and len(w) > 3:
                seen.add(w_lower)
                topics.append(w)
                if len(topics) >= 5:
                    break
        return topics


class LedgerAwareCompactionSummarizer:
    """Uses SessionLedger to produce structured summaries of compacted segments.

    Produces output like:
    "Accomplished: [X]. Decided: [Y]. Pending: [Z]."
    Falls back to KeywordCompactionSummarizer if no ledger available.
    """

    def __init__(self) -> None:
        self._fallback = KeywordCompactionSummarizer()

    def summarize(self, removed_messages: List[Message], ledger: Optional[object] = None) -> str:
        if ledger is None:
            return self._fallback.summarize(removed_messages)

        max_turn = len(removed_messages)

        # Access ledger entries
        entries = getattr(ledger, "entries", [])
        if not entries:
            return self._fallback.summarize(removed_messages)

        # Filter entries within the compacted range
        relevant = [e for e in entries if e.turn_index <= max_turn]
        if not relevant:
            return self._fallback.summarize(removed_messages)

        files_read = set()
        files_modified = set()
        decisions = []
        recommendations = []
        pending = []

        for entry in relevant:
            if entry.category == "file_read":
                files_read.add(entry.key)
            elif entry.category == "file_modified":
                files_modified.add(entry.key)
            elif entry.category == "decision":
                decisions.append(entry.summary)
            elif entry.category == "recommendation":
                recommendations.append(entry.summary)
            elif entry.category == "pending_action" and not entry.resolved:
                pending.append(entry.summary)

        parts = []
        if files_read:
            parts.append(f"Read {len(files_read)} files: {', '.join(sorted(files_read)[:5])}")
        if files_modified:
            parts.append(
                f"Modified {len(files_modified)} files: {', '.join(sorted(files_modified)[:5])}"
            )
        if decisions:
            parts.append(f"Decided: {'; '.join(decisions[:3])}")
        if recommendations:
            parts.append(f"Recommended: {'; '.join(recommendations[:3])}")
        if pending:
            parts.append(f"Pending: {'; '.join(pending[:3])}")

        if parts:
            return f"[Compacted context: {'. '.join(parts)}.]"

        return self._fallback.summarize(removed_messages)
