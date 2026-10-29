"""Skill analytics — tracks selection events, hit rates, and usage patterns.

Provides in-memory analytics for skill auto-selection. Can be queried
via ``victor skill stats`` or exported for observability.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class _SkillRecord:
    """Accumulated stats for one skill."""

    count: int = 0
    total_score: float = 0.0
    last_used: float = 0.0

    @property
    def avg_score(self) -> float:
        return self.total_score / self.count if self.count > 0 else 0.0


class SkillAnalytics:
    """In-memory skill selection analytics.

    Thread-safe for single-process use. Tracks per-skill selection
    counts, average scores, and global hit/miss rates.
    """

    def __init__(self) -> None:
        self._skills: Dict[str, _SkillRecord] = defaultdict(_SkillRecord)
        self._total_matches: int = 0
        self._total_misses: int = 0
        self._multi_skill_count: int = 0

    def record_selection(self, skill_name: str, score: float) -> None:
        """Record a single skill selection event."""
        record = self._skills[skill_name]
        record.count += 1
        record.total_score += score
        record.last_used = time.time()
        self._total_matches += 1

    def record_miss(self) -> None:
        """Record a turn where no skill matched."""
        self._total_misses += 1

    def record_multi_selection(self, selections: List[Tuple[str, float]]) -> None:
        """Record a multi-skill selection event."""
        for name, score in selections:
            self.record_selection(name, score)
        if len(selections) > 1:
            self._multi_skill_count += 1

    def get_skill_stats(self, skill_name: str) -> Dict[str, Any]:
        """Get stats for a specific skill."""
        record = self._skills.get(skill_name)
        if not record or record.count == 0:
            return {"name": skill_name, "count": 0, "avg_score": 0.0}
        return {
            "name": skill_name,
            "count": record.count,
            "avg_score": record.avg_score,
            "last_used": record.last_used,
        }

    def get_all_stats(self) -> List[Dict[str, Any]]:
        """Get stats for all skills, sorted by count descending."""
        results = []
        for name, record in self._skills.items():
            if record.count > 0:
                results.append(
                    {
                        "name": name,
                        "count": record.count,
                        "avg_score": record.avg_score,
                        "last_used": record.last_used,
                    }
                )
        return sorted(results, key=lambda x: -x["count"])

    def get_global_stats(self) -> Dict[str, Any]:
        """Get global analytics summary."""
        total = self._total_matches + self._total_misses
        return {
            "total_matches": self._total_matches,
            "total_misses": self._total_misses,
            "miss_rate": self._total_misses / total if total > 0 else 0.0,
            "multi_skill_count": self._multi_skill_count,
            "unique_skills_used": sum(1 for r in self._skills.values() if r.count > 0),
        }

    def reset(self) -> None:
        """Reset all analytics."""
        self._skills.clear()
        self._total_matches = 0
        self._total_misses = 0
        self._multi_skill_count = 0
