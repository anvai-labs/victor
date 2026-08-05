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

"""Hierarchical compaction compression.

Groups old compaction summaries into epoch-level summaries to prevent
linear accumulation of compaction context over long sessions.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.agent.compaction_summarizer import CompactionSummaryStrategy

logger = logging.getLogger(__name__)


@dataclass
class CompactionEpoch:
    """A compressed epoch combining multiple compaction summaries."""

    epoch_id: str
    summary: str
    source_count: int
    created_at: float
    turn_range: Tuple[int, int]


class HierarchicalCompactionManager:
    """Manages compaction summaries with hierarchical compression.

    When individual summaries exceed epoch_threshold, older summaries
    are compressed into epoch-level summaries. Only the most recent
    max_individual summaries are kept as individual entries.
    """

    def __init__(
        self,
        summarizer: Optional["CompactionSummaryStrategy"] = None,
        max_individual: int = 3,
        epoch_threshold: int = 6,
    ):
        self._summarizer = summarizer
        self._max_individual = max_individual
        self._epoch_threshold = epoch_threshold
        self._individual_summaries: List[Tuple[str, int]] = []  # (summary, turn_index)
        self._epochs: List[CompactionEpoch] = []

    def add_summary(self, summary: str, turn_index: int) -> None:
        """Add a compaction summary; triggers epoch creation when threshold exceeded."""
        self._individual_summaries.append((summary, turn_index))

        if len(self._individual_summaries) >= self._epoch_threshold:
            self._create_epoch()

    def _create_epoch(self) -> None:
        """Compress older summaries into an epoch."""
        # Keep max_individual most recent, compress the rest
        to_compress = self._individual_summaries[: -self._max_individual]
        self._individual_summaries = self._individual_summaries[-self._max_individual :]

        if not to_compress:
            return

        summaries = [s for s, _ in to_compress]
        turns = [t for _, t in to_compress]
        turn_range = (min(turns), max(turns))

        # Epochs are summaries-of-summaries; simple concatenation is correct.
        # The CompactionSummaryStrategy protocol is designed for raw messages,
        # not for re-summarizing existing summaries.
        epoch_summary = " | ".join(summaries)

        epoch = CompactionEpoch(
            epoch_id=str(uuid.uuid4())[:8],
            summary=epoch_summary,
            source_count=len(to_compress),
            created_at=time.time(),
            turn_range=turn_range,
        )
        self._epochs.append(epoch)
        logger.debug(
            f"Created compaction epoch {epoch.epoch_id} from {len(to_compress)} summaries (turns {turn_range[0]}-{turn_range[1]})"
        )

    def get_active_context(self, max_chars: int = 2000) -> str:
        """Returns epoch summaries + recent individual summaries within budget."""
        parts = []
        chars_used = 0

        # Add epoch summaries first (older context)
        for epoch in self._epochs:
            entry = f"[Epoch {epoch.epoch_id} (turns {epoch.turn_range[0]}-{epoch.turn_range[1]}, {epoch.source_count} compactions): {epoch.summary}]"
            if chars_used + len(entry) > max_chars:
                break
            parts.append(entry)
            chars_used += len(entry)

        # Add recent individual summaries
        for summary, _ in self._individual_summaries:
            if chars_used + len(summary) > max_chars:
                break
            parts.append(summary)
            chars_used += len(summary)

        return " | ".join(parts) if parts else ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistence."""
        return {
            "individual_summaries": [
                {"summary": s, "turn_index": t} for s, t in self._individual_summaries
            ],
            "epochs": [
                {
                    "epoch_id": e.epoch_id,
                    "summary": e.summary,
                    "source_count": e.source_count,
                    "created_at": e.created_at,
                    "turn_range": list(e.turn_range),
                }
                for e in self._epochs
            ],
            "max_individual": self._max_individual,
            "epoch_threshold": self._epoch_threshold,
        }

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], summarizer: Optional[Any] = None
    ) -> "HierarchicalCompactionManager":
        """Deserialize from dict."""
        manager = cls(
            summarizer=summarizer,
            max_individual=data.get("max_individual", 3),
            epoch_threshold=data.get("epoch_threshold", 6),
        )
        manager._individual_summaries = [
            (item["summary"], item["turn_index"]) for item in data.get("individual_summaries", [])
        ]
        manager._epochs = [
            CompactionEpoch(
                epoch_id=e["epoch_id"],
                summary=e["summary"],
                source_count=e["source_count"],
                created_at=e["created_at"],
                turn_range=tuple(e["turn_range"]),
            )
            for e in data.get("epochs", [])
        ]
        return manager
