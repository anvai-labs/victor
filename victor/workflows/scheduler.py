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

"""Workflow Scheduler with Cron Support.

Provides Airflow-like scheduling capabilities for Victor workflows:
- Cron expression parsing and scheduling
- Interval-based scheduling (every N minutes/hours)
- One-time scheduled execution
- Timezone support
- Catchup/backfill support

Usage:
    from victor.workflows.scheduler import (
        WorkflowScheduler,
        CronSchedule,
        ScheduledWorkflow,
        get_scheduler,
    )

    # Create a schedule
    schedule = CronSchedule.from_cron("0 9 * * *")  # Daily at 9 AM

    # Register a scheduled workflow
    scheduler = get_scheduler()
    scheduler.register(ScheduledWorkflow(
        workflow_name="daily_report",
        workflow_path="workflows/reporting.yaml",
        schedule=schedule,
        initial_state={"report_type": "daily"},
    ))

    # Start the scheduler (runs in background)
    await scheduler.start()

YAML Configuration:
    workflows:
      daily_report:
        description: "Generate daily reports"
        schedule:
          cron: "0 9 * * *"      # Daily at 9 AM UTC
          timezone: "UTC"
          catchup: false
          start_date: "2025-01-01"
        nodes:
          - id: generate
            type: agent
            ...
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)

# Singleton instance
_scheduler_instance: Optional["WorkflowScheduler"] = None
_scheduler_lock = threading.Lock()


class ScheduleType(Enum):
    """Type of schedule."""

    CRON = "cron"
    INTERVAL = "interval"
    ONCE = "once"


@dataclass
class CronSchedule:
    """Cron-based schedule configuration.

    Supports standard cron expressions:
    - minute (0-59)
    - hour (0-23)
    - day of month (1-31)
    - month (1-12)
    - day of week (0-6, Sunday=0)

    Also supports common aliases:
    - @hourly: 0 * * * *
    - @daily: 0 0 * * *
    - @weekly: 0 0 * * 0
    - @monthly: 0 0 1 * *
    - @yearly: 0 0 1 1 *

    Attributes:
        expression: Cron expression string
        timezone: Timezone name (default UTC)
        minute: Parsed minute field
        hour: Parsed hour field
        day_of_month: Parsed day of month field
        month: Parsed month field
        day_of_week: Parsed day of week field
    """

    expression: str
    timezone: str = "UTC"
    minute: Set[int] = field(default_factory=set)
    hour: Set[int] = field(default_factory=set)
    day_of_month: Set[int] = field(default_factory=set)
    month: Set[int] = field(default_factory=set)
    day_of_week: Set[int] = field(default_factory=set)

    # Aliases for common schedules
    ALIASES: Dict[str, str] = field(
        default_factory=lambda: {
            "@hourly": "0 * * * *",
            "@daily": "0 0 * * *",
            "@midnight": "0 0 * * *",
            "@weekly": "0 0 * * 0",
            "@monthly": "0 0 1 * *",
            "@yearly": "0 0 1 1 *",
            "@annually": "0 0 1 1 *",
        },
        repr=False,
    )

    def __post_init__(self):
        """Parse the cron expression."""
        expr = self.ALIASES.get(self.expression, self.expression)
        self._parse_expression(expr)

    def _parse_expression(self, expression: str) -> None:
        """Parse a cron expression into field sets."""
        parts = expression.strip().split()
        if len(parts) != 5:
            raise ValueError(
                f"Invalid cron expression: '{expression}'. "
                "Expected 5 fields: minute hour day_of_month month day_of_week"
            )

        self.minute = self._parse_field(parts[0], 0, 59)
        self.hour = self._parse_field(parts[1], 0, 23)
        self.day_of_month = self._parse_field(parts[2], 1, 31)
        self.month = self._parse_field(parts[3], 1, 12)
        self.day_of_week = self._parse_field(parts[4], 0, 6)

    def _parse_field(self, field: str, min_val: int, max_val: int) -> Set[int]:
        """Parse a single cron field.

        Supports:
        - * (all values)
        - Single values: 5
        - Ranges: 1-5
        - Steps: */5, 1-10/2
        - Lists: 1,3,5
        """
        result: Set[int] = set()

        for part in field.split(","):
            if part == "*":
                result.update(range(min_val, max_val + 1))
            elif "/" in part:
                # Step value
                range_part, step = part.split("/")
                step_val = int(step)
                if range_part == "*":
                    result.update(range(min_val, max_val + 1, step_val))
                elif "-" in range_part:
                    start, end = map(int, range_part.split("-"))
                    result.update(range(start, end + 1, step_val))
                else:
                    start = int(range_part)
                    result.update(range(start, max_val + 1, step_val))
            elif "-" in part:
                # Range
                start, end = map(int, part.split("-"))
                result.update(range(start, end + 1))
            else:
                # Single value
                result.add(int(part))

        return result

    @classmethod
    def from_cron(cls, expression: str, tz: str = "UTC") -> "CronSchedule":
        """Create a CronSchedule from a cron expression.

        Args:
            expression: Cron expression or alias
            tz: Timezone name

        Returns:
            CronSchedule instance
        """
        return cls(expression=expression, timezone=tz)

    @classmethod
    def from_interval(
        cls,
        minutes: int = 0,
        hours: int = 0,
        tz: str = "UTC",
    ) -> "CronSchedule":
        """Create a schedule from an interval.

        Args:
            minutes: Run every N minutes (if hours=0)
            hours: Run every N hours

        Returns:
            CronSchedule instance
        """
        if hours > 0:
            # Every N hours
            hour_vals = ",".join(str(h) for h in range(0, 24, hours))
            expr = f"0 {hour_vals} * * *"
        elif minutes > 0:
            # Every N minutes
            minute_vals = ",".join(str(m) for m in range(0, 60, minutes))
            expr = f"{minute_vals} * * * *"
        else:
            raise ValueError("Either minutes or hours must be > 0")

        return cls(expression=expr, timezone=tz)

    def matches(self, dt: datetime) -> bool:
        """Check if a datetime matches this schedule.

        Args:
            dt: Datetime to check (should be in schedule's timezone)

        Returns:
            True if the datetime matches
        """
        return (
            dt.minute in self.minute
            and dt.hour in self.hour
            and dt.day in self.day_of_month
            and dt.month in self.month
            and dt.weekday() in self._convert_weekday(self.day_of_week)
        )

    def _convert_weekday(self, cron_days: Set[int]) -> Set[int]:
        """Convert cron weekday (0=Sunday) to Python weekday (0=Monday)."""
        # Cron: 0=Sunday, 1=Monday, ..., 6=Saturday
        # Python: 0=Monday, 1=Tuesday, ..., 6=Sunday
        python_days = set()
        for d in cron_days:
            if d == 0:
                python_days.add(6)  # Sunday
            else:
                python_days.add(d - 1)
        return python_days

    def next_run(self, after: Optional[datetime] = None) -> datetime:
        """Calculate the next run time after a given datetime.

        Args:
            after: Calculate next run after this time (default: now)

        Returns:
            Next scheduled run time
        """
        if after is None:
            after = datetime.now(timezone.utc)

        # Start from the next minute
        dt = after.replace(second=0, microsecond=0) + timedelta(minutes=1)

        # Find the next matching time (limit to 1 year search)
        max_iterations = 525600  # Minutes in a year
        for _ in range(max_iterations):
            if self.matches(dt):
                return dt
            dt += timedelta(minutes=1)

        raise ValueError("Could not find next run time within 1 year")


@dataclass
class ScheduledWorkflow:
    """A workflow with scheduling configuration.

    Attributes:
        workflow_name: Name of the workflow
        workflow_path: Path to workflow YAML (optional)
        schedule: Cron schedule
        initial_state: Initial state for workflow execution
        enabled: Whether the schedule is active
        catchup: Whether to run missed executions
        start_date: Earliest date to start scheduling
        end_date: Latest date for scheduling (optional)
        max_active_runs: Maximum concurrent runs
        tags: Metadata tags
    """

    workflow_name: str
    schedule: CronSchedule
    workflow_path: Optional[str] = None
    initial_state: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    catchup: bool = False
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    max_active_runs: int = 1
    tags: List[str] = field(default_factory=list)

    # Runtime tracking
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    active_runs: int = 0
    run_count: int = 0

    def __post_init__(self):
        """Calculate initial next_run."""
        if self.next_run is None and self.enabled:
            start = self.start_date or datetime.now(timezone.utc)
            self.next_run = self.schedule.next_run(start)

    @property
    def schedule_id(self) -> str:
        """Generate a unique ID for this scheduled workflow."""
        return hashlib.sha256(
            f"{self.workflow_name}:{self.schedule.expression}".encode()
        ).hexdigest()[:12]

    def should_run(self, now: Optional[datetime] = None) -> bool:
        """Check if the workflow should run now.

        Args:
            now: Current time (default: now)

        Returns:
            True if the workflow should be triggered
        """
        if not self.enabled:
            return False

        if now is None:
            now = datetime.now(timezone.utc)

        if self.end_date and now > self.end_date:
            return False

        if self.active_runs >= self.max_active_runs:
            return False

        if self.next_run and now >= self.next_run:
            return True

        return False

    def mark_started(self) -> None:
        """Mark that a run has started."""
        self.active_runs += 1
        self.last_run = datetime.now(timezone.utc)
        # Calculate next run
        self.next_run = self.schedule.next_run(self.last_run)

    def mark_completed(self) -> None:
        """Mark that a run has completed."""
        self.active_runs = max(0, self.active_runs - 1)
        self.run_count += 1


class WorkflowScheduler:
    """Background scheduler for workflow execution.

    Manages scheduled workflows and triggers them at the appropriate times.
    Runs as a background asyncio task.

    Example:
        scheduler = WorkflowScheduler()

        # Register a scheduled workflow
        scheduler.register(ScheduledWorkflow(
            workflow_name="daily_backup",
            schedule=CronSchedule.from_cron("0 2 * * *"),
        ))

        # Start the scheduler
        await scheduler.start()

        # Stop later
        await scheduler.stop()
    """

    def __init__(
        self,
        check_interval: float = 60.0,  # Check every minute
        executor: Optional[Callable] = None,
    ):
        """Initialize the scheduler.

        Args:
            check_interval: How often to check for due workflows (seconds)
            executor: Optional custom workflow executor function
        """
        self._schedules: Dict[str, ScheduledWorkflow] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._check_interval = check_interval
        self._executor = executor
        self._lock = threading.RLock()
        self._execution_history: List[Dict[str, Any]] = []

    def register(self, scheduled_workflow: ScheduledWorkflow) -> str:
        """Register a scheduled workflow.

        Args:
            scheduled_workflow: The workflow to schedule

        Returns:
            Schedule ID
        """
        with self._lock:
            schedule_id = scheduled_workflow.schedule_id
            self._schedules[schedule_id] = scheduled_workflow
            logger.info(
                f"Registered scheduled workflow: {scheduled_workflow.workflow_name} "
                f"(id={schedule_id}, cron={scheduled_workflow.schedule.expression})"
            )
            return schedule_id

    def unregister(self, schedule_id: str) -> bool:
        """Unregister a scheduled workflow.

        Args:
            schedule_id: ID of the schedule to remove

        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if schedule_id in self._schedules:
                workflow = self._schedules.pop(schedule_id)
                logger.info(f"Unregistered scheduled workflow: {workflow.workflow_name}")
                return True
            return False

    def get_schedule(self, schedule_id: str) -> Optional[ScheduledWorkflow]:
        """Get a scheduled workflow by ID.

        Args:
            schedule_id: Schedule ID

        Returns:
            ScheduledWorkflow or None
        """
        with self._lock:
            return self._schedules.get(schedule_id)

    def list_schedules(self) -> List[ScheduledWorkflow]:
        """List all registered schedules.

        Returns:
            List of scheduled workflows
        """
        with self._lock:
            return list(self._schedules.values())

    def enable(self, schedule_id: str) -> bool:
        """Enable a schedule.

        Args:
            schedule_id: Schedule ID

        Returns:
            True if enabled, False if not found
        """
        with self._lock:
            if schedule_id in self._schedules:
                self._schedules[schedule_id].enabled = True
                return True
            return False

    def disable(self, schedule_id: str) -> bool:
        """Disable a schedule.

        Args:
            schedule_id: Schedule ID

        Returns:
            True if disabled, False if not found
        """
        with self._lock:
            if schedule_id in self._schedules:
                self._schedules[schedule_id].enabled = False
                return True
            return False

    async def start(self) -> None:
        """Start the scheduler background task."""
        if self._running:
            logger.warning("Scheduler is already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Workflow scheduler started")

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Workflow scheduler stopped")

    async def _run_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                await self._check_and_execute()
                await asyncio.sleep(self._check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}", exc_info=True)
                await asyncio.sleep(self._check_interval)

    async def _check_and_execute(self) -> None:
        """Check for due workflows and execute them."""
        now = datetime.now(timezone.utc)

        with self._lock:
            due_schedules = [s for s in self._schedules.values() if s.should_run(now)]

        for schedule in due_schedules:
            asyncio.create_task(self._execute_workflow(schedule))

    async def _execute_workflow(self, schedule: ScheduledWorkflow) -> None:
        """Execute a scheduled workflow.

        Args:
            schedule: The scheduled workflow to execute
        """
        schedule.mark_started()
        execution_id = f"{schedule.schedule_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        logger.info(
            f"Executing scheduled workflow: {schedule.workflow_name} "
            f"(execution_id={execution_id})"
        )

        start_time = datetime.now(timezone.utc)
        success = False
        error_msg = None

        try:
            if self._executor:
                # Use custom executor
                await self._executor(
                    workflow_name=schedule.workflow_name,
                    workflow_path=schedule.workflow_path,
                    initial_state=schedule.initial_state,
                    execution_id=execution_id,
                )
            else:
                # Default: use UnifiedWorkflowCompiler
                await self._default_execute(schedule, execution_id)

            success = True
            logger.info(f"Completed scheduled workflow: {schedule.workflow_name}")

        except Exception as e:
            error_msg = str(e)
            logger.error(
                f"Scheduled workflow failed: {schedule.workflow_name}: {e}",
                exc_info=True,
            )

        finally:
            schedule.mark_completed()

            # Record execution history
            self._execution_history.append(
                {
                    "execution_id": execution_id,
                    "workflow_name": schedule.workflow_name,
                    "schedule_id": schedule.schedule_id,
                    "start_time": start_time,
                    "end_time": datetime.now(timezone.utc),
                    "success": success,
                    "error": error_msg,
                }
            )

            # Keep last 1000 executions
            if len(self._execution_history) > 1000:
                self._execution_history = self._execution_history[-1000:]

    async def _default_execute(
        self,
        schedule: ScheduledWorkflow,
        execution_id: str,
    ) -> None:
        """Default workflow execution using UnifiedWorkflowCompiler.

        **Architecture Note**: UnifiedWorkflowCompiler is the canonical compiler API.
        Scheduled workflows benefit from its two-level caching, which avoids
        re-parsing YAML on recurring executions of the same workflow.

        Args:
            schedule: Scheduled workflow
            execution_id: Unique execution ID
        """
        from pathlib import Path
        from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

        if not schedule.workflow_path:
            raise ValueError(
                f"No workflow_path specified for scheduled workflow: " f"{schedule.workflow_name}"
            )

        compiler = UnifiedWorkflowCompiler(enable_caching=True)
        compiled = compiler.compile_yaml(
            Path(schedule.workflow_path),
            workflow_name=schedule.workflow_name,
        )

        # Add execution metadata to state
        state = dict(schedule.initial_state)
        state["_scheduled_execution_id"] = execution_id
        state["_scheduled_at"] = datetime.now(timezone.utc).isoformat()

        await compiled.invoke(state)

    def get_execution_history(
        self,
        schedule_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get execution history.

        Args:
            schedule_id: Filter by schedule ID
            limit: Maximum entries to return

        Returns:
            List of execution records
        """
        history = self._execution_history
        if schedule_id:
            history = [h for h in history if h["schedule_id"] == schedule_id]
        return history[-limit:]

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running


def get_scheduler() -> WorkflowScheduler:
    """Get the global workflow scheduler.

    Thread-safe singleton access.

    Returns:
        Global WorkflowScheduler instance
    """
    global _scheduler_instance

    if _scheduler_instance is None:
        with _scheduler_lock:
            if _scheduler_instance is None:
                _scheduler_instance = WorkflowScheduler()

    return _scheduler_instance


def schedule_workflow(
    workflow_name: str,
    cron: str,
    workflow_path: Optional[str] = None,
    initial_state: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> str:
    """Schedule a workflow for recurring execution.

    Convenience function for quick scheduling.

    Args:
        workflow_name: Name of the workflow
        cron: Cron expression
        workflow_path: Path to workflow YAML
        initial_state: Initial workflow state
        **kwargs: Additional ScheduledWorkflow options

    Returns:
        Schedule ID
    """
    scheduler = get_scheduler()
    schedule = CronSchedule.from_cron(cron)
    scheduled_workflow = ScheduledWorkflow(
        workflow_name=workflow_name,
        workflow_path=workflow_path,
        schedule=schedule,
        initial_state=initial_state or {},
        **kwargs,
    )
    return scheduler.register(scheduled_workflow)


__all__ = [
    "ScheduleType",
    "CronSchedule",
    "ScheduledWorkflow",
    "WorkflowScheduler",
    "get_scheduler",
    "schedule_workflow",
]
