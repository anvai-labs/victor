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

"""Tool result deduplication for repeated file reads.

Replaces older duplicate file read results with compact stubs to reduce
context consumption without losing awareness that the file was read.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from victor.config.orchestrator_constants import (
    DeduplicationConfig,
    DEDUPLICATION_CONFIG,
)
from victor.providers.base import Message

logger = logging.getLogger(__name__)

# Pattern to extract path from TOOL_OUTPUT markers
_TOOL_OUTPUT_PATH_RE = re.compile(
    r'<TOOL_OUTPUT\s+tool="(\w+)"(?:\s+path="([^"]*)")?', re.IGNORECASE
)


class ToolResultDeduplicator:
    """Deduplicates repeated file reads in conversation history.

    Only replaces .content of identified messages — never removes messages
    or changes message order. Minimum content threshold prevents stubbing
    small results.
    """

    def __init__(self, config: Optional[DeduplicationConfig] = None):
        self._config = config or DEDUPLICATION_CONFIG

    @property
    def config(self) -> DeduplicationConfig:
        return self._config

    def should_deduplicate(self, tool_name: str, args: Dict[str, Any]) -> bool:
        if not self._config.enabled:
            return False
        return tool_name.lower() in self._config.dedup_tool_names

    def deduplicate_in_place(
        self, messages: List[Message], new_tool_name: str, new_args: Dict[str, Any]
    ) -> int:
        """Replace older duplicate tool outputs with compact stubs.

        Two dedup strategies:
        1. Path-based: exact path match for read-like tools (existing)
        2. Content-hash: identical content from any tool (new)

        Args:
            messages: Conversation message list (modified in place)
            new_tool_name: Name of the tool that just executed
            new_args: Arguments of the tool that just executed

        Returns:
            Number of messages stubbed
        """
        if not self._config.enabled:
            return 0

        stubbed = self._dedup_by_path(messages, new_tool_name, new_args)
        stubbed += self._dedup_by_content_hash(messages)
        return stubbed

    def deduplicate_history_view(self, messages: List[Message]) -> Tuple[List[Message], int]:
        """Non-mutating view-stage dedup (FEP-0023 P2).

        Returns a NEW message list in which older duplicate tool outputs
        (identical re-reads) are replaced with compact stubs, together with the
        number stubbed. The input ``messages`` list and its ``Message`` objects
        are never modified — stubs are freshly-constructed ``Message`` instances
        that replace entries in the *copy* only. This is the safe variant of
        :meth:`deduplicate_in_place` for running over an assembled context view
        without touching source-of-truth history.

        Uses the content-hash strategy (identical content, any path); the
        path-based strategy is not applicable here because a view has no notion
        of a single "just-executed" tool call.
        """
        if not self._config.enabled:
            return list(messages), 0
        working = list(messages)  # shallow copy; entries are replaced, not mutated
        stubbed = self._dedup_by_content_hash(working)
        return working, stubbed

    def _dedup_by_path(
        self, messages: List[Message], new_tool_name: str, new_args: Dict[str, Any]
    ) -> int:
        """Stub older tool outputs with matching path."""
        target_path = new_args.get("path", new_args.get("file_path", ""))
        if not target_path:
            return 0

        stubbed = 0
        for i in range(len(messages) - 1):
            msg = messages[i]
            if msg.role != "user":
                continue
            if len(msg.content) < self._config.min_content_chars_to_dedup:
                continue

            match = _TOOL_OUTPUT_PATH_RE.search(msg.content)
            if not match:
                continue

            matched_tool = match.group(1)
            matched_path = match.group(2) or ""

            if matched_tool.lower() not in self._config.dedup_tool_names:
                continue
            if matched_path != target_path:
                continue

            lines = msg.content.count("\n") + 1
            stub = self._config.stub_template.format(path=target_path, lines=lines)
            messages[i] = Message(role=msg.role, content=stub)
            stubbed += 1
            logger.debug("Deduplicated path match for %s at message %d", target_path, i)

        return stubbed

    def _dedup_by_content_hash(self, messages: List[Message]) -> int:
        """Stub older messages with identical content (regardless of path).

        Uses a hash of first 2000 + last 500 chars as a fingerprint.
        When duplicates found, the OLDER message is replaced with a stub,
        keeping the most recent version in the conversation.
        """
        seen: Dict[int, int] = {}  # content fingerprint → index of newest occurrence
        stubbed = 0
        min_chars = self._config.min_content_chars_to_dedup

        for i, msg in enumerate(messages):
            if msg.role != "user" or len(msg.content) < min_chars:
                continue
            # Already a stub? Skip
            if msg.content.startswith("[Previously") or msg.content.startswith("[Duplicate"):
                continue

            fingerprint = hash(msg.content[:2000] + msg.content[-500:])
            if fingerprint in seen:
                prev_idx = seen[fingerprint]
                # Only stub if the previous message is still full content
                prev = messages[prev_idx]
                if not prev.content.startswith("["):
                    lines = prev.content.count("\n") + 1
                    messages[prev_idx] = Message(
                        role=prev.role,
                        content=f"[Duplicate tool output — {lines} lines, see later message]",
                    )
                    stubbed += 1
                    logger.debug(
                        "Content-hash dedup: stubbed message %d (dup of %d)",
                        prev_idx,
                        i,
                    )
            seen[fingerprint] = i

        return stubbed
