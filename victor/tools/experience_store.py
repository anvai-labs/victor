"""Tool Experience Store — stores and retrieves tool usage experiences.

Implements the experience storage layer for E3-TIR (Enhanced Experience
Exploitation for Tool-Integrated Reasoning). Stores three experience types:

1. **Demonstration**: Expert-curated tool usage trajectories
2. **Self-Play**: Agent-generated tool interaction traces
3. **Targeted Exploration**: Deliberate probes of under-utilized tools

Related research:
- Experience replay for RL (arXiv:2507.07451 RLEP)
- Mode collapse prevention in LLM agents
- Warm-up paradigms for tool-integrated reasoning
"""

from __future__ import annotations

from victor.core.json_utils import json_dumps, json_loads
import logging
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


class ExperienceType(Enum):
    """Type of tool experience."""

    DEMONSTRATION = "demonstration"
    SELF_PLAY = "self_play"
    TARGETED_EXPLORATION = "targeted_exploration"


@dataclass
class ToolExperience:
    """A single tool usage experience record."""

    tool_name: str
    task_type: str
    experience_type: ExperienceType
    success: bool
    reward: float  # 0.0 to 1.0
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    # Trajectory: sequence of (tool, args_summary, outcome)
    trajectory: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ToolStats:
    """Aggregated statistics for a single tool."""

    total_uses: int = 0
    successes: int = 0
    failures: int = 0
    avg_reward: float = 0.0
    last_used: float = 0.0
    experience_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def success_rate(self) -> float:
        return self.successes / max(self.total_uses, 1)

    @property
    def staleness(self) -> float:
        """Seconds since last use. Higher = more stale."""
        if self.last_used == 0:
            return float("inf")
        return time.time() - self.last_used


class ToolExperienceStore:
    """Persistent store for tool usage experiences.

    Supports three operations central to E3-TIR:
    - Recording experiences of all three types
    - Sampling experiences for warm-up (by type, recency, diversity)
    - Computing per-tool statistics for exploration scheduling

    Storage is in-memory with optional JSON persistence for cross-session
    continuity.
    """

    def __init__(
        self,
        max_experiences: int = 5000,
        persist_path: Optional[Path] = None,
    ) -> None:
        self._experiences: List[ToolExperience] = []
        self._max_experiences = max_experiences
        self._persist_path = persist_path
        self._tool_stats: Dict[str, ToolStats] = defaultdict(ToolStats)
        self._all_known_tools: set = set()
        # R3: this in-memory store is a *projection* of the durable RL_OUTCOME
        # stream. It is empty on restart, so it is warm-started once from the DB
        # (see warm_start_from_outcomes) to avoid drifting from the durable truth.
        self._warm_started: bool = False

        if persist_path and persist_path.exists():
            self._load()

    def record(self, experience: ToolExperience) -> None:
        """Record a tool experience."""
        self._experiences.append(experience)
        self._all_known_tools.add(experience.tool_name)

        # Update stats
        stats = self._tool_stats[experience.tool_name]
        stats.total_uses += 1
        if experience.success:
            stats.successes += 1
        else:
            stats.failures += 1
        # Running average reward
        n = stats.total_uses
        stats.avg_reward = stats.avg_reward * ((n - 1) / n) + experience.reward / n
        stats.last_used = experience.timestamp
        stats.experience_counts[experience.experience_type.value] += 1

        # Evict oldest if over capacity
        if len(self._experiences) > self._max_experiences:
            self._experiences = self._experiences[-self._max_experiences :]

    def record_outcome(
        self,
        tool_name: str,
        task_type: str,
        success: bool,
        reward: float,
        experience_type: ExperienceType = ExperienceType.SELF_PLAY,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Convenience: record a tool execution outcome."""
        self.record(
            ToolExperience(
                tool_name=tool_name,
                task_type=task_type,
                experience_type=experience_type,
                success=success,
                reward=reward,
                context=context or {},
            )
        )

    def warm_start_from_outcomes(
        self,
        outcomes: Iterable[tuple],
        *,
        experience_type: ExperienceType = ExperienceType.DEMONSTRATION,
    ) -> int:
        """Rebuild the projection from durable tool-execution outcomes (R3).

        The store is an in-memory *projection* of the durable RL_OUTCOME stream;
        it is otherwise empty on restart, drifting from the durable truth. This
        replays historical outcomes (chronological order expected) so the
        projection reflects what was already recorded.

        Idempotent: runs at most once per store instance (guarded by
        ``_warm_started``) so it cannot double-count alongside the live feed.

        Args:
            outcomes: iterable of ``(tool_name, task_type, success, reward)``.
            experience_type: classification for replayed history.

        Returns:
            Number of outcomes replayed.
        """
        if self._warm_started:
            return 0
        self._warm_started = True
        count = 0
        for tool_name, task_type, success, reward in outcomes:
            if not tool_name:
                continue
            self.record_outcome(
                tool_name=tool_name,
                task_type=task_type or "general",
                success=bool(success),
                reward=float(reward),
                experience_type=experience_type,
            )
            count += 1
        return count

    def register_tools(self, tool_names: List[str]) -> None:
        """Register known tools for exploration tracking."""
        self._all_known_tools.update(tool_names)

    def get_stats(self, tool_name: str) -> ToolStats:
        """Get aggregated stats for a tool."""
        return self._tool_stats[tool_name]

    def get_all_stats(self) -> Dict[str, ToolStats]:
        """Get stats for all known tools."""
        return dict(self._tool_stats)

    def get_underutilized_tools(self, threshold: int = 5) -> List[str]:
        """Get tools with fewer than `threshold` total uses.

        These are candidates for targeted exploration.
        """
        underutilized = []
        for name in self._all_known_tools:
            stats = self._tool_stats.get(name)
            if stats is None or stats.total_uses < threshold:
                underutilized.append(name)
        return sorted(underutilized)

    def get_stale_tools(self, staleness_seconds: float = 3600) -> List[str]:
        """Get tools not used recently. Candidates for re-exploration."""
        stale = []
        for name in self._all_known_tools:
            stats = self._tool_stats.get(name)
            if stats is None or stats.staleness > staleness_seconds:
                stale.append(name)
        return sorted(stale)

    def sample_experiences(
        self,
        experience_type: Optional[ExperienceType] = None,
        task_type: Optional[str] = None,
        limit: int = 10,
        recent_first: bool = True,
    ) -> List[ToolExperience]:
        """Sample experiences for warm-up or replay.

        Args:
            experience_type: Filter by type (None = all)
            task_type: Filter by task (None = all)
            limit: Max experiences to return
            recent_first: If True, newest first

        Returns:
            List of matching experiences
        """
        filtered = self._experiences
        if experience_type is not None:
            filtered = [e for e in filtered if e.experience_type == experience_type]
        if task_type is not None:
            filtered = [e for e in filtered if e.task_type == task_type]

        if recent_first:
            filtered = sorted(filtered, key=lambda e: e.timestamp, reverse=True)

        return filtered[:limit]

    def get_diversity_score(self) -> float:
        """Compute tool usage diversity (0 to 1).

        1.0 = all tools used equally. 0.0 = only one tool ever used.
        Uses normalized entropy of tool usage distribution.
        """
        import math

        if not self._tool_stats:
            return 0.0

        total = sum(s.total_uses for s in self._tool_stats.values())
        if total == 0:
            return 0.0

        n_tools = len(self._all_known_tools)
        if n_tools <= 1:
            return 1.0

        entropy = 0.0
        for stats in self._tool_stats.values():
            if stats.total_uses > 0:
                p = stats.total_uses / total
                entropy -= p * math.log2(p)

        max_entropy = math.log2(n_tools)
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _save(self) -> None:
        """Persist experiences to JSON file."""
        if not self._persist_path:
            return
        try:
            data = {
                "experiences": [
                    {
                        **asdict(e),
                        "experience_type": e.experience_type.value,
                    }
                    for e in self._experiences[-self._max_experiences :]
                ],
                "known_tools": sorted(self._all_known_tools),
            }
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._persist_path.write_text(json_dumps(data, indent=2))
        except Exception as e:
            logger.warning("Failed to persist experience store: %s", e)

    def _load(self) -> None:
        """Load experiences from JSON file."""
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            data = json_loads(self._persist_path.read_text())
            for raw in data.get("experiences", []):
                exp_type = ExperienceType(raw.pop("experience_type"))
                exp = ToolExperience(**raw, experience_type=exp_type)
                self.record(exp)
            self._all_known_tools.update(data.get("known_tools", []))
        except Exception as e:
            logger.warning("Failed to load experience store: %s", e)

    def __len__(self) -> int:
        return len(self._experiences)
