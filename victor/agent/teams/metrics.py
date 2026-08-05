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

"""Team performance metrics for tracking and learning.

This module provides data structures and utilities for measuring
team execution performance, which feeds into the Q-learning system.
"""

from __future__ import annotations

from victor.core.json_utils import json_dumps, json_loads
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

# Import canonical types from victor.teams.types
from victor.teams.types import TeamConfig, TeamFormation, TeamResult

logger = logging.getLogger(__name__)


class TaskCategory(Enum):
    """High-level task categories for team optimization.

    Teams are optimized per category since different tasks
    benefit from different team compositions.
    """

    EXPLORATION = "exploration"  # Code search, research
    IMPLEMENTATION = "implementation"  # Writing/editing code
    REVIEW = "review"  # Code review, quality checks
    TESTING = "testing"  # Test writing, verification
    REFACTORING = "refactoring"  # Code restructuring
    DOCUMENTATION = "documentation"  # Docs, comments
    DEBUGGING = "debugging"  # Bug finding, fixing
    PLANNING = "planning"  # Architecture, design
    MIXED = "mixed"  # Multi-category tasks


@dataclass
class TeamMetrics:
    """Performance metrics for a single team execution.

    Captures everything needed for learning optimal compositions.

    Attributes:
        team_id: Unique execution ID
        task_category: Type of task performed
        formation: Team formation used
        member_count: Number of team members
        role_distribution: Count of each role type
        total_tool_budget: Combined tool budget
        tools_used: Actual tools used
        success: Whether team completed successfully
        quality_score: Quality rating 0.0-1.0
        duration_seconds: Total execution time
        member_results: Per-member success rates
        discoveries_count: Number of discoveries made
        timestamp: When execution occurred
    """

    team_id: str
    task_category: TaskCategory
    formation: TeamFormation
    member_count: int
    role_distribution: Dict[str, int]
    total_tool_budget: int
    tools_used: int
    success: bool
    quality_score: float
    duration_seconds: float
    member_results: Dict[str, bool] = field(default_factory=dict)
    discoveries_count: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @classmethod
    def from_result(
        cls,
        config: TeamConfig,
        result: TeamResult,
        task_category: TaskCategory,
        quality_score: float = 0.8,
    ) -> "TeamMetrics":
        """Create metrics from team config and result.

        Args:
            config: Team configuration used
            result: Execution result
            task_category: Category of task
            quality_score: Quality score (default 0.8 for success, 0.2 for failure)

        Returns:
            TeamMetrics instance
        """
        import uuid

        # Count roles
        role_distribution: Dict[str, int] = {}
        for member in config.members:
            role = member.role.value
            role_distribution[role] = role_distribution.get(role, 0) + 1

        # Per-member success - result.member_results is Dict[str, MemberResult]
        member_success = {}
        discoveries_count = 0
        for member_id, mr in result.member_results.items():
            member_success[member_id] = mr.success
            discoveries_count += len(mr.discoveries)

        # Generate team_id if needed
        team_id = f"team_{uuid.uuid4().hex[:8]}"

        # Determine quality score based on success if not provided
        if not result.success:
            quality_score = 0.2

        return cls(
            team_id=team_id,
            task_category=task_category,
            formation=result.formation,
            member_count=len(config.members),
            role_distribution=role_distribution,
            total_tool_budget=config.total_tool_budget,
            tools_used=result.total_tool_calls,
            success=result.success,
            quality_score=quality_score,
            duration_seconds=result.total_duration,
            member_results=member_success,
            discoveries_count=discoveries_count,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "team_id": self.team_id,
            "task_category": self.task_category.value,
            "formation": self.formation.value,
            "member_count": self.member_count,
            "role_distribution": json_dumps(self.role_distribution),
            "total_tool_budget": self.total_tool_budget,
            "tools_used": self.tools_used,
            "success": self.success,
            "quality_score": self.quality_score,
            "duration_seconds": self.duration_seconds,
            "member_results": json_dumps(self.member_results),
            "discoveries_count": self.discoveries_count,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TeamMetrics":
        """Create from dictionary."""
        role_dist = data.get("role_distribution", "{}")
        if isinstance(role_dist, str):
            role_dist = json_loads(role_dist)

        member_results = data.get("member_results", "{}")
        if isinstance(member_results, str):
            member_results = json_loads(member_results)

        return cls(
            team_id=data["team_id"],
            task_category=TaskCategory(data["task_category"]),
            formation=TeamFormation(data["formation"]),
            member_count=data["member_count"],
            role_distribution=role_dist,
            total_tool_budget=data["total_tool_budget"],
            tools_used=data["tools_used"],
            success=data["success"],
            quality_score=data["quality_score"],
            duration_seconds=data["duration_seconds"],
            member_results=member_results,
            discoveries_count=data.get("discoveries_count", 0),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
        )

    def compute_efficiency(self) -> float:
        """Compute efficiency score (quality per tool call).

        Returns:
            Efficiency score 0.0-1.0
        """
        if self.tools_used == 0:
            return 0.0

        # Efficiency = quality / (tools_used / budget)
        budget_usage = self.tools_used / max(self.total_tool_budget, 1)
        if budget_usage > 1.0:
            budget_usage = 1.0

        # Higher quality with lower budget usage = better efficiency
        efficiency = self.quality_score * (1.0 - 0.5 * budget_usage)
        return max(0.0, min(1.0, efficiency))

    def compute_speed_score(self, target_seconds: float = 60.0) -> float:
        """Compute speed score relative to target.

        Args:
            target_seconds: Target duration for full score

        Returns:
            Speed score 0.0-1.0 (1.0 = at or under target)
        """
        if self.duration_seconds <= target_seconds:
            return 1.0
        # Decay exponentially over target
        ratio = target_seconds / self.duration_seconds
        return max(0.0, ratio)


@dataclass
class CompositionStats:
    """Aggregated statistics for a team composition.

    Tracks performance over multiple executions to learn
    optimal configurations.

    Attributes:
        formation: Team formation
        role_counts: Role distribution (e.g., {"researcher": 2, "executor": 1})
        task_category: Task category these stats apply to
        total_executions: Number of times this composition was used
        successes: Number of successful executions
        total_quality: Sum of quality scores
        total_duration: Sum of durations
        total_tools_used: Sum of tools used
        total_budget: Sum of budgets allocated
        last_updated: Last update timestamp
    """

    formation: TeamFormation
    role_counts: Dict[str, int]
    task_category: TaskCategory
    total_executions: int = 0
    successes: int = 0
    total_quality: float = 0.0
    total_duration: float = 0.0
    total_tools_used: int = 0
    total_budget: int = 0
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_executions == 0:
            return 0.0
        return self.successes / self.total_executions

    @property
    def avg_quality(self) -> float:
        """Calculate average quality score."""
        if self.total_executions == 0:
            return 0.0
        return self.total_quality / self.total_executions

    @property
    def avg_duration(self) -> float:
        """Calculate average duration."""
        if self.total_executions == 0:
            return 0.0
        return self.total_duration / self.total_executions

    @property
    def avg_efficiency(self) -> float:
        """Calculate average tool efficiency."""
        if self.total_budget == 0:
            return 0.0
        return 1.0 - (self.total_tools_used / self.total_budget)

    def get_composition_key(self) -> str:
        """Get unique key for this composition.

        Returns:
            Key like "sequential:researcher=2,executor=1"
        """
        role_str = ",".join(f"{role}={count}" for role, count in sorted(self.role_counts.items()))
        return f"{self.formation.value}:{role_str}"

    def update(self, metrics: TeamMetrics) -> None:
        """Update stats with new metrics.

        Args:
            metrics: Metrics from a team execution
        """
        self.total_executions += 1
        if metrics.success:
            self.successes += 1
        self.total_quality += metrics.quality_score
        self.total_duration += metrics.duration_seconds
        self.total_tools_used += metrics.tools_used
        self.total_budget += metrics.total_tool_budget
        self.last_updated = datetime.now().isoformat()

    def compute_q_value(
        self,
        alpha_success: float = 0.4,
        alpha_quality: float = 0.3,
        alpha_efficiency: float = 0.2,
        alpha_speed: float = 0.1,
    ) -> float:
        """Compute Q-value for this composition.

        Uses weighted combination of success, quality, efficiency, and speed.

        Args:
            alpha_success: Weight for success rate
            alpha_quality: Weight for quality score
            alpha_efficiency: Weight for tool efficiency
            alpha_speed: Weight for execution speed

        Returns:
            Q-value between 0.0 and 1.0
        """
        # Normalize duration to speed score (inverse, capped)
        speed_score = 1.0 / (1.0 + self.avg_duration / 60.0)

        q_value = (
            alpha_success * self.success_rate
            + alpha_quality * self.avg_quality
            + alpha_efficiency * max(0.0, self.avg_efficiency)
            + alpha_speed * speed_score
        )

        return max(0.0, min(1.0, q_value))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "formation": self.formation.value,
            "role_counts": json_dumps(self.role_counts),
            "task_category": self.task_category.value,
            "total_executions": self.total_executions,
            "successes": self.successes,
            "total_quality": self.total_quality,
            "total_duration": self.total_duration,
            "total_tools_used": self.total_tools_used,
            "total_budget": self.total_budget,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompositionStats":
        """Create from dictionary."""
        role_counts = data.get("role_counts", "{}")
        if isinstance(role_counts, str):
            role_counts = json_loads(role_counts)

        return cls(
            formation=TeamFormation(data["formation"]),
            role_counts=role_counts,
            task_category=TaskCategory(data["task_category"]),
            total_executions=data["total_executions"],
            successes=data["successes"],
            total_quality=data["total_quality"],
            total_duration=data["total_duration"],
            total_tools_used=data["total_tools_used"],
            total_budget=data["total_budget"],
            last_updated=data.get("last_updated", datetime.now().isoformat()),
        )


def categorize_task(task_description: str) -> TaskCategory:
    """Categorize a task based on its description.

    Args:
        task_description: Task description text

    Returns:
        TaskCategory for the task
    """
    task_lower = task_description.lower()

    # Check more specific keywords first, then general ones
    # Order matters - check specific categories before overlapping ones

    # Testing - check before "write" catches it
    if any(kw in task_lower for kw in ["test", "spec", "assert", "mock", "unit test"]):
        return TaskCategory.TESTING

    # Review/validation - check before implementation
    if any(kw in task_lower for kw in ["review", "check", "validate", "audit", "quality"]):
        return TaskCategory.REVIEW

    # Debugging - check before generic "fix"
    if any(kw in task_lower for kw in ["debug", "bug", "error", "issue", "troubleshoot"]):
        return TaskCategory.DEBUGGING

    # Refactoring
    if any(kw in task_lower for kw in ["refactor", "restructure", "reorganize", "clean up"]):
        return TaskCategory.REFACTORING

    # Documentation
    if any(kw in task_lower for kw in ["document", "readme", "comment", "explain"]):
        return TaskCategory.DOCUMENTATION

    # Planning
    if any(kw in task_lower for kw in ["plan", "design", "architect"]):
        return TaskCategory.PLANNING

    # Exploration - broad search terms
    if any(kw in task_lower for kw in ["search", "find", "explore", "research", "discover"]):
        return TaskCategory.EXPLORATION

    # Implementation - most general, checked last
    if any(kw in task_lower for kw in ["implement", "create", "build", "write", "add", "fix"]):
        return TaskCategory.IMPLEMENTATION

    return TaskCategory.MIXED


__all__ = [
    "TaskCategory",
    "TeamMetrics",
    "CompositionStats",
    "categorize_task",
]
