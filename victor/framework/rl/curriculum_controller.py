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

"""Curriculum learning controller for progressive agent training.

This module implements curriculum learning to progressively increase
task complexity as the agent learns, improving sample efficiency
and final performance.

Curriculum Strategy:
- Stage 1 (Warm-up): Simple tasks with limited tools
- Stage 2 (Basic): Medium complexity with common tools
- Stage 3 (Intermediate): Full tool access, moderate iterations
- Stage 4 (Advanced): Complex tasks, extended iterations
- Stage 5 (Expert): No restrictions, full capability

Auto-Progression:
- Advance when success_rate > threshold AND sample_size > minimum
- Regress when success_rate drops significantly
- Track per-context progression

Sprint 5: Advanced RL Patterns
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class CurriculumStage(IntEnum):
    """Curriculum learning stages."""

    WARM_UP = 1  # Simple tasks, limited tools
    BASIC = 2  # Medium tasks, common tools
    INTERMEDIATE = 3  # Full tools, moderate iterations
    ADVANCED = 4  # Complex tasks, extended iterations
    EXPERT = 5  # No restrictions


@dataclass
class StageConfig:
    """Configuration for a curriculum stage.

    Attributes:
        stage: Stage enum value
        name: Human-readable name
        max_tools: Maximum number of tools per session
        max_iterations: Maximum iterations allowed
        allowed_task_types: Task types allowed at this stage
        tool_budget: Tool execution budget
        success_threshold: Required success rate to advance
        min_samples: Minimum samples before advancement
        description: Stage description
    """

    stage: CurriculumStage
    name: str
    max_tools: int
    max_iterations: int
    allowed_task_types: Set[str]
    tool_budget: int = 10
    success_threshold: float = 0.75
    min_samples: int = 20
    description: str = ""


@dataclass
class StageMetrics:
    """Metrics for a curriculum stage.

    Attributes:
        stage: Current stage
        sample_count: Number of samples at this stage
        success_count: Number of successful samples
        avg_quality: Average quality score
        avg_iterations: Average iterations used
        avg_tools_used: Average tools used per session
        last_updated: Last update timestamp
    """

    stage: CurriculumStage
    sample_count: int = 0
    success_count: int = 0
    avg_quality: float = 0.5
    avg_iterations: float = 0.0
    avg_tools_used: float = 0.0
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def success_rate(self) -> float:
        """Compute success rate."""
        if self.sample_count == 0:
            return 0.0
        return self.success_count / self.sample_count

    def update(
        self,
        success: bool,
        quality: float,
        iterations: int,
        tools_used: int,
    ) -> None:
        """Update metrics with new sample.

        Args:
            success: Whether task succeeded
            quality: Quality score
            iterations: Iterations used
            tools_used: Tools used
        """
        n = self.sample_count
        self.sample_count += 1
        self.success_count += 1 if success else 0

        # Running averages
        self.avg_quality = (self.avg_quality * n + quality) / (n + 1)
        self.avg_iterations = (self.avg_iterations * n + iterations) / (n + 1)
        self.avg_tools_used = (self.avg_tools_used * n + tools_used) / (n + 1)
        self.last_updated = datetime.now().isoformat()


class CurriculumController:
    """Controller for curriculum-based learning progression.

    Manages the progression through curriculum stages based on
    agent performance, enabling efficient learning through
    gradually increasing complexity.

    Features:
    1. Per-context stage tracking
    2. Auto-advancement based on success rate
    3. Regression protection (gradual, not instant)
    4. Stage-specific constraints (tools, iterations, task types)

    Usage:
        controller = CurriculumController(db_connection)

        # Get constraints for current stage
        config = controller.get_stage_config(context_key)
        max_tools = config.max_tools

        # Record outcome
        controller.record_outcome(context_key, success, quality, iterations, tools)

        # Check for advancement
        if controller.should_advance(context_key):
            controller.advance(context_key)
    """

    # Default stage configurations
    DEFAULT_STAGES = {
        CurriculumStage.WARM_UP: StageConfig(
            stage=CurriculumStage.WARM_UP,
            name="Warm-up",
            max_tools=3,
            max_iterations=5,
            allowed_task_types={"search", "read", "analysis"},
            tool_budget=5,
            success_threshold=0.7,
            min_samples=10,
            description="Simple read/search tasks with limited tools",
        ),
        CurriculumStage.BASIC: StageConfig(
            stage=CurriculumStage.BASIC,
            name="Basic",
            max_tools=8,
            max_iterations=10,
            allowed_task_types={"search", "read", "analysis", "explain"},
            tool_budget=15,
            success_threshold=0.75,
            min_samples=20,
            description="Basic tasks with common tool set",
        ),
        CurriculumStage.INTERMEDIATE: StageConfig(
            stage=CurriculumStage.INTERMEDIATE,
            name="Intermediate",
            max_tools=15,
            max_iterations=20,
            allowed_task_types={
                "search",
                "read",
                "analysis",
                "explain",
                "edit",
                "create",
                "refactor",
            },
            tool_budget=25,
            success_threshold=0.78,
            min_samples=30,
            description="Full tool access with moderate iteration budget",
        ),
        CurriculumStage.ADVANCED: StageConfig(
            stage=CurriculumStage.ADVANCED,
            name="Advanced",
            max_tools=25,
            max_iterations=30,
            allowed_task_types={
                "search",
                "read",
                "analysis",
                "explain",
                "edit",
                "create",
                "refactor",
                "action",
                "debug",
            },
            tool_budget=40,
            success_threshold=0.80,
            min_samples=40,
            description="Complex multi-step tasks",
        ),
        CurriculumStage.EXPERT: StageConfig(
            stage=CurriculumStage.EXPERT,
            name="Expert",
            max_tools=50,
            max_iterations=50,
            allowed_task_types={"all"},  # No restrictions
            tool_budget=60,
            success_threshold=0.85,
            min_samples=50,
            description="Full capability, no restrictions",
        ),
    }

    # Regression parameters
    REGRESSION_THRESHOLD = 0.5  # Success rate below this triggers regression
    REGRESSION_WINDOW = 10  # Recent samples to consider

    def __init__(
        self,
        db_connection: Optional[Any] = None,
        stages: Optional[Dict[CurriculumStage, StageConfig]] = None,
    ):
        """Initialize curriculum controller.

        Args:
            db_connection: Optional SQLite connection for persistence
            stages: Optional custom stage configurations
        """
        self.db = db_connection
        self.stages = stages or self.DEFAULT_STAGES

        # Per-context stage tracking
        self._context_stages: Dict[str, CurriculumStage] = {}

        # Per-context, per-stage metrics
        self._metrics: Dict[str, Dict[CurriculumStage, StageMetrics]] = {}

        # Recent outcomes for regression detection
        self._recent_outcomes: Dict[str, List[bool]] = {}

        if db_connection:
            self._ensure_tables()
            self._load_state()

    def _ensure_tables(self) -> None:
        """Create tables for curriculum state."""
        if not self.db:
            return

        cursor = self.db.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS curriculum_stages (
                context_key TEXT PRIMARY KEY,
                current_stage INTEGER NOT NULL,
                last_updated TEXT NOT NULL
            )
            """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS curriculum_metrics (
                context_key TEXT NOT NULL,
                stage INTEGER NOT NULL,
                sample_count INTEGER NOT NULL DEFAULT 0,
                success_count INTEGER NOT NULL DEFAULT 0,
                avg_quality REAL NOT NULL DEFAULT 0.5,
                avg_iterations REAL NOT NULL DEFAULT 0.0,
                avg_tools_used REAL NOT NULL DEFAULT 0.0,
                last_updated TEXT NOT NULL,
                PRIMARY KEY (context_key, stage)
            )
            """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS curriculum_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                context_key TEXT NOT NULL,
                from_stage INTEGER NOT NULL,
                to_stage INTEGER NOT NULL,
                reason TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """)

        self.db.commit()

    def _load_state(self) -> None:
        """Load state from database."""
        if not self.db:
            return

        cursor = self.db.cursor()

        try:
            # Load current stages
            cursor.execute("SELECT context_key, current_stage FROM curriculum_stages")
            for row in cursor.fetchall():
                self._context_stages[row[0]] = CurriculumStage(row[1])

            # Load metrics
            cursor.execute("SELECT * FROM curriculum_metrics")
            for row in cursor.fetchall():
                row_dict = dict(row)
                context_key = row_dict["context_key"]
                stage = CurriculumStage(row_dict["stage"])

                if context_key not in self._metrics:
                    self._metrics[context_key] = {}

                self._metrics[context_key][stage] = StageMetrics(
                    stage=stage,
                    sample_count=row_dict["sample_count"],
                    success_count=row_dict["success_count"],
                    avg_quality=row_dict["avg_quality"],
                    avg_iterations=row_dict["avg_iterations"],
                    avg_tools_used=row_dict["avg_tools_used"],
                    last_updated=row_dict["last_updated"],
                )

            if self._context_stages:
                logger.info(f"CurriculumController: Loaded {len(self._context_stages)} contexts")

        except Exception as e:
            logger.debug(f"CurriculumController: Could not load state: {e}")

    def get_stage(self, context_key: str) -> CurriculumStage:
        """Get current stage for a context.

        Args:
            context_key: Context identifier (provider:model:task_type)

        Returns:
            Current curriculum stage
        """
        return self._context_stages.get(context_key, CurriculumStage.WARM_UP)

    def get_stage_config(self, context_key: str) -> StageConfig:
        """Get stage configuration for a context.

        Args:
            context_key: Context identifier

        Returns:
            StageConfig for current stage
        """
        stage = self.get_stage(context_key)
        return self.stages[stage]

    def get_constraints(self, context_key: str) -> Dict[str, Any]:
        """Get constraints for a context based on current stage.

        Args:
            context_key: Context identifier

        Returns:
            Dictionary with constraint values
        """
        config = self.get_stage_config(context_key)

        return {
            "max_tools": config.max_tools,
            "max_iterations": config.max_iterations,
            "tool_budget": config.tool_budget,
            "allowed_task_types": list(config.allowed_task_types),
            "stage": config.stage.value,
            "stage_name": config.name,
        }

    def is_task_allowed(self, context_key: str, task_type: str) -> bool:
        """Check if a task type is allowed at current stage.

        Args:
            context_key: Context identifier
            task_type: Task type to check

        Returns:
            True if allowed
        """
        config = self.get_stage_config(context_key)

        if "all" in config.allowed_task_types:
            return True

        return task_type.lower() in config.allowed_task_types

    def record_outcome(
        self,
        context_key: str,
        success: bool,
        quality: float,
        iterations: int,
        tools_used: int,
    ) -> None:
        """Record a task outcome.

        Args:
            context_key: Context identifier
            success: Whether task succeeded
            quality: Quality score
            iterations: Iterations used
            tools_used: Tools used
        """
        # Ensure context is tracked (defaults to WARM_UP for new contexts)
        if context_key not in self._context_stages:
            self._context_stages[context_key] = CurriculumStage.WARM_UP

        stage = self.get_stage(context_key)

        # Ensure metrics exist
        if context_key not in self._metrics:
            self._metrics[context_key] = {}

        if stage not in self._metrics[context_key]:
            self._metrics[context_key][stage] = StageMetrics(stage=stage)

        # Update metrics
        metrics = self._metrics[context_key][stage]
        metrics.update(success, quality, iterations, tools_used)

        # Track recent outcomes for regression
        if context_key not in self._recent_outcomes:
            self._recent_outcomes[context_key] = []

        self._recent_outcomes[context_key].append(success)
        self._recent_outcomes[context_key] = self._recent_outcomes[context_key][
            -self.REGRESSION_WINDOW :
        ]

        # Check for advancement or regression
        self._check_progression(context_key)

        # Save to database
        self._save_state(context_key, stage, metrics)

    def _check_progression(self, context_key: str) -> None:
        """Check and handle stage progression.

        Args:
            context_key: Context identifier
        """
        if self.should_advance(context_key):
            self.advance(context_key)
        elif self.should_regress(context_key):
            self.regress(context_key)

    def should_advance(self, context_key: str) -> bool:
        """Check if context should advance to next stage.

        Args:
            context_key: Context identifier

        Returns:
            True if should advance
        """
        stage = self.get_stage(context_key)

        # Can't advance past expert
        if stage == CurriculumStage.EXPERT:
            return False

        config = self.stages[stage]
        metrics = self._metrics.get(context_key, {}).get(stage)

        if not metrics:
            return False

        # Check requirements
        if metrics.sample_count < config.min_samples:
            return False

        if metrics.success_rate < config.success_threshold:
            return False

        return True

    def should_regress(self, context_key: str) -> bool:
        """Check if context should regress to previous stage.

        Args:
            context_key: Context identifier

        Returns:
            True if should regress
        """
        stage = self.get_stage(context_key)

        # Can't regress below warm-up
        if stage == CurriculumStage.WARM_UP:
            return False

        # Check recent outcomes
        recent = self._recent_outcomes.get(context_key, [])
        if len(recent) < self.REGRESSION_WINDOW:
            return False

        recent_success_rate = sum(recent) / len(recent)
        return recent_success_rate < self.REGRESSION_THRESHOLD

    def advance(self, context_key: str) -> bool:
        """Advance context to next stage.

        Args:
            context_key: Context identifier

        Returns:
            True if advanced
        """
        current_stage = self.get_stage(context_key)

        if current_stage == CurriculumStage.EXPERT:
            return False

        new_stage = CurriculumStage(current_stage.value + 1)
        self._context_stages[context_key] = new_stage

        # Initialize metrics for new stage
        if context_key not in self._metrics:
            self._metrics[context_key] = {}
        self._metrics[context_key][new_stage] = StageMetrics(stage=new_stage)

        # Clear recent outcomes
        self._recent_outcomes[context_key] = []

        # Record history
        self._record_transition(
            context_key,
            current_stage,
            new_stage,
            "Advancement: success threshold met",
        )

        logger.info(
            f"CurriculumController: {context_key} advanced from "
            f"{current_stage.name} to {new_stage.name}"
        )

        return True

    def regress(self, context_key: str) -> bool:
        """Regress context to previous stage.

        Args:
            context_key: Context identifier

        Returns:
            True if regressed
        """
        current_stage = self.get_stage(context_key)

        if current_stage == CurriculumStage.WARM_UP:
            return False

        new_stage = CurriculumStage(current_stage.value - 1)
        self._context_stages[context_key] = new_stage

        # Clear recent outcomes
        self._recent_outcomes[context_key] = []

        # Record history
        self._record_transition(
            context_key,
            current_stage,
            new_stage,
            "Regression: success rate dropped",
        )

        logger.warning(
            f"CurriculumController: {context_key} regressed from "
            f"{current_stage.name} to {new_stage.name}"
        )

        return True

    def _record_transition(
        self,
        context_key: str,
        from_stage: CurriculumStage,
        to_stage: CurriculumStage,
        reason: str,
    ) -> None:
        """Record stage transition in database.

        Args:
            context_key: Context identifier
            from_stage: Previous stage
            to_stage: New stage
            reason: Transition reason
        """
        if not self.db:
            return

        cursor = self.db.cursor()
        timestamp = datetime.now().isoformat()

        cursor.execute(
            """
            INSERT INTO curriculum_history
            (context_key, from_stage, to_stage, reason, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (context_key, from_stage.value, to_stage.value, reason, timestamp),
        )

        cursor.execute(
            """
            INSERT OR REPLACE INTO curriculum_stages
            (context_key, current_stage, last_updated)
            VALUES (?, ?, ?)
            """,
            (context_key, to_stage.value, timestamp),
        )

        self.db.commit()

    def _save_state(
        self,
        context_key: str,
        stage: CurriculumStage,
        metrics: StageMetrics,
    ) -> None:
        """Save state to database.

        Args:
            context_key: Context identifier
            stage: Current stage
            metrics: Stage metrics
        """
        if not self.db:
            return

        cursor = self.db.cursor()
        timestamp = datetime.now().isoformat()

        # Save current stage
        cursor.execute(
            """
            INSERT OR REPLACE INTO curriculum_stages
            (context_key, current_stage, last_updated)
            VALUES (?, ?, ?)
            """,
            (context_key, stage.value, timestamp),
        )

        # Save metrics
        cursor.execute(
            """
            INSERT OR REPLACE INTO curriculum_metrics
            (context_key, stage, sample_count, success_count,
             avg_quality, avg_iterations, avg_tools_used, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                context_key,
                stage.value,
                metrics.sample_count,
                metrics.success_count,
                metrics.avg_quality,
                metrics.avg_iterations,
                metrics.avg_tools_used,
                timestamp,
            ),
        )

        self.db.commit()

    def get_progress_summary(self, context_key: str) -> Dict[str, Any]:
        """Get progress summary for a context.

        Args:
            context_key: Context identifier

        Returns:
            Dictionary with progress information
        """
        stage = self.get_stage(context_key)
        config = self.stages[stage]
        metrics = self._metrics.get(context_key, {}).get(stage)

        if not metrics:
            return {
                "context_key": context_key,
                "current_stage": stage.value,
                "stage_name": config.name,
                "progress_to_next": 0.0,
                "can_advance": False,
            }

        # Compute progress to next stage
        sample_progress = min(1.0, metrics.sample_count / config.min_samples)
        success_progress = min(1.0, metrics.success_rate / config.success_threshold)
        overall_progress = (sample_progress + success_progress) / 2

        return {
            "context_key": context_key,
            "current_stage": stage.value,
            "stage_name": config.name,
            "sample_count": metrics.sample_count,
            "success_rate": metrics.success_rate,
            "avg_quality": metrics.avg_quality,
            "progress_to_next": overall_progress,
            "samples_needed": max(0, config.min_samples - metrics.sample_count),
            "success_rate_needed": config.success_threshold,
            "can_advance": self.should_advance(context_key),
        }

    def export_metrics(self) -> Dict[str, Any]:
        """Export curriculum metrics.

        Returns:
            Dictionary with metrics
        """
        stage_distribution = {stage.name: 0 for stage in CurriculumStage}

        for stage in self._context_stages.values():
            stage_distribution[stage.name] += 1

        total_samples = sum(
            metrics.sample_count
            for context_metrics in self._metrics.values()
            for metrics in context_metrics.values()
        )

        return {
            "total_contexts": len(self._context_stages),
            "stage_distribution": stage_distribution,
            "total_samples": total_samples,
            "stages": {
                stage.name: {
                    "max_tools": config.max_tools,
                    "max_iterations": config.max_iterations,
                    "success_threshold": config.success_threshold,
                }
                for stage, config in self.stages.items()
            },
        }


# Global singleton
_curriculum_controller: Optional[CurriculumController] = None


def get_curriculum_controller(
    db_connection: Optional[Any] = None,
) -> CurriculumController:
    """Get global curriculum controller (lazy init).

    Args:
        db_connection: Optional database connection

    Returns:
        CurriculumController singleton
    """
    global _curriculum_controller
    if _curriculum_controller is None:
        _curriculum_controller = CurriculumController(db_connection)
    return _curriculum_controller
