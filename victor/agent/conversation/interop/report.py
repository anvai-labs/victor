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

"""Conversion report: every loss in an import/export is accounted, never silent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from victor.agent.conversation.interop.formats import TranscriptFormat


@dataclass
class ConversionReport:
    """What happened during a transcript conversion."""

    source_format: str = ""
    target_format: str = ""
    entries_read: int = 0
    role_counts: Dict[str, int] = field(default_factory=dict)
    tool_calls_preserved: int = 0
    thinking_blocks_dropped: int = 0
    encrypted_items_skipped: int = 0
    truncations: int = 0
    skipped: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    model: Optional[str] = None
    time_span: Optional[str] = None
    session_id: Optional[str] = None
    context_bytes: Optional[int] = None
    context_byte_limit: Optional[int] = None

    @property
    def context_over_budget(self) -> bool:
        return (
            self.context_bytes is not None
            and self.context_byte_limit is not None
            and self.context_bytes > self.context_byte_limit
        )

    def count_role(self, role: str) -> None:
        self.role_counts[role] = self.role_counts.get(role, 0) + 1

    def note_skip(self, index: int, kind: str, reason: str) -> None:
        self.skipped.append({"index": index, "type": kind, "reason": reason})

    def note_warning(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    @property
    def message_count(self) -> int:
        return sum(self.role_counts.values())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_format": self.source_format,
            "target_format": self.target_format,
            "entries_read": self.entries_read,
            "message_count": self.message_count,
            "role_counts": dict(self.role_counts),
            "tool_calls_preserved": self.tool_calls_preserved,
            "thinking_blocks_dropped": self.thinking_blocks_dropped,
            "encrypted_items_skipped": self.encrypted_items_skipped,
            "truncations": self.truncations,
            "skipped": list(self.skipped),
            "warnings": list(self.warnings),
            "model": self.model,
            "time_span": self.time_span,
            "session_id": self.session_id,
            "context_bytes": self.context_bytes,
            "context_byte_limit": self.context_byte_limit,
        }

    def render_summary(self) -> str:
        """Compact multi-line summary for console panels."""
        roles = " · ".join(f"{v} {k}" for k, v in sorted(self.role_counts.items())) or "none"
        lines = [
            f"Format: {self.source_format or '?'} → {self.target_format or '?'}",
            f"Messages: {self.message_count} ({roles})",
        ]
        if self.tool_calls_preserved:
            lines.append(f"Tool calls preserved: {self.tool_calls_preserved}")
        losses = []
        if self.thinking_blocks_dropped:
            losses.append(f"{self.thinking_blocks_dropped} thinking blocks dropped")
        if self.encrypted_items_skipped:
            losses.append(f"{self.encrypted_items_skipped} encrypted items skipped")
        if self.truncations:
            losses.append(f"{self.truncations} truncations")
        if self.skipped:
            losses.append(f"{len(self.skipped)} entries skipped")
        if losses:
            lines.append("Lossy: " + "; ".join(losses))
        if self.time_span:
            lines.append(f"Span: {self.time_span}")
        lines.extend(f"Warning: {warning}" for warning in self.warnings)
        return "\n".join(lines)


def format_label(fmt: TranscriptFormat) -> str:
    return fmt.value
