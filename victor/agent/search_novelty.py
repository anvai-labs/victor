# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Search-novelty / diminishing-returns detection for the agentic loop.

Detects when an agent keeps searching the codebase but is finding mostly the *same* results —
the failure mode behind a "summarize the architecture" task that ran 31 distinct ``code_search``
calls without converging. The existing guards miss it: each query differs (so the
``SpinDetector`` signature differs) and the assistant content varies (so content-repetition
doesn't fire). The discriminator here is the per-turn *hit set*: a genuine multi-file task keeps
surfacing NEW ``(path, symbol)`` hits (novelty stays high), while a thrash re-surfaces the same
files (novelty collapses).

Import-clean: consumes only ``list[dict]`` tool results (the ``result_items`` attached by the
tool service), so both the headless and streaming loops can share one tracker via the
``TurnEvaluationController``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

_SEARCH_TOOLS = frozenset({"code_search", "semantic_code_search", "grep", "search"})


@dataclass(frozen=True)
class NoveltyConfig:
    """Thresholds for diminishing-returns detection (all tunable via settings)."""

    novelty_ratio_threshold: float = 0.34
    """Below this ratio of NEW files, a search turn counts as 'low novelty'. At 0.34, a search
    that surfaces only already-seen files (ratio 0) or mostly-seen files counts; a search with
    a healthy fraction of new files does not."""
    consecutive_low_novelty_limit: int = 3
    """Consecutive low-novelty search turns before force-completing (synthesize)."""
    nudge_after_low_novelty: int = 2
    """Consecutive low-novelty search turns before emitting a 'synthesize now' nudge."""
    min_search_turns: int = 2
    """Warm-up: ignore the first N search turns before counting low novelty."""

    @classmethod
    def from_exploration(cls, exploration: Any = None) -> "NoveltyConfig":
        """Build from an ``ExplorationSettings`` (or ``None`` -> defaults), read defensively."""
        if exploration is None:
            return cls()
        g = exploration
        return cls(
            novelty_ratio_threshold=getattr(g, "novelty_ratio_threshold", 0.34),
            consecutive_low_novelty_limit=getattr(g, "novelty_consecutive_low_limit", 3),
            nudge_after_low_novelty=getattr(g, "novelty_nudge_after_low", 2),
            min_search_turns=getattr(g, "novelty_min_search_turns", 2),
        )


@dataclass
class NoveltyResult:
    """Outcome of one turn's novelty check."""

    novelty_ratio: float = 1.0
    consecutive_low_novelty: int = 0
    should_nudge: bool = False
    should_force_complete: bool = False
    total_distinct_hits: int = 0
    had_search: bool = False


_SYNTHESIZE_NUDGE = (
    "You have searched the codebase several times and are now finding mostly the same results "
    "— you have enough context. Stop searching and write your answer/summary now."
)


def synthesize_nudge_message() -> str:
    """The shared 'you have enough — synthesize' nudge text."""
    return _SYNTHESIZE_NUDGE


class SearchNoveltyTracker:
    """Track the novelty of successive search results to detect diminishing returns.

    Feed each turn's ``tool_results`` to :meth:`record_turn`. Non-search turns (read/edit/no
    search) are neutral and never accumulate. O(hits) set operations per turn; no LLM calls.
    """

    def __init__(self, config: Optional[NoveltyConfig] = None) -> None:
        self._config = config or NoveltyConfig()
        self._seen_files: Set[str] = set()
        self._consecutive_low = 0
        self._search_turns = 0

    def record_turn(self, tool_results: Optional[List[Dict[str, Any]]]) -> NoveltyResult:
        """Ingest one turn's tool results; novelty is measured at FILE granularity.

        A search that surfaces only files already seen this conversation is 'low novelty'
        regardless of how many line/symbol matches it returns — that is the diminishing-returns
        signal behind a narrow-search thrash (many queries re-covering the same handful of
        files). A genuine multi-file task keeps surfacing NEW files, so it never saturates.
        """
        files = self._extract_files(tool_results)
        if not files:
            # No search this turn — neutral, don't disturb the counter.
            return NoveltyResult(
                consecutive_low_novelty=self._consecutive_low,
                total_distinct_hits=len(self._seen_files),
                had_search=False,
            )

        self._search_turns += 1
        new_files = files - self._seen_files
        ratio = len(new_files) / len(files)
        self._seen_files |= files

        cfg = self._config
        if self._search_turns <= cfg.min_search_turns:  # warm-up
            return NoveltyResult(
                novelty_ratio=ratio,
                consecutive_low_novelty=self._consecutive_low,
                total_distinct_hits=len(self._seen_files),
                had_search=True,
            )

        if ratio < cfg.novelty_ratio_threshold:
            self._consecutive_low += 1
        else:
            self._consecutive_low = 0

        return NoveltyResult(
            novelty_ratio=ratio,
            consecutive_low_novelty=self._consecutive_low,
            should_nudge=self._consecutive_low >= cfg.nudge_after_low_novelty,
            should_force_complete=self._consecutive_low >= cfg.consecutive_low_novelty_limit,
            total_distinct_hits=len(self._seen_files),
            had_search=True,
        )

    @staticmethod
    def _extract_files(tool_results: Optional[List[Dict[str, Any]]]) -> Set[str]:
        """Pull the set of file paths a search turn touched, from ``result_items``."""
        if not tool_results:
            return set()
        files: Set[str] = set()
        for result in tool_results:
            if not isinstance(result, dict) or not result.get("success", True):
                continue
            name = result.get("tool_name") or result.get("name") or ""
            if name and name not in _SEARCH_TOOLS:
                continue
            items = result.get("result_items")
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict):
                    path = item.get("path") or item.get("file_path")
                    if path:
                        files.add(str(path))
        return files

    def reset(self) -> None:
        self._seen_files.clear()
        self._consecutive_low = 0
        self._search_turns = 0
