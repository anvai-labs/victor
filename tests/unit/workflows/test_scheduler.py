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

"""Tests for workflow scheduler with cron support."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestCronSchedule:
    """Tests for CronSchedule parsing and matching."""

    def test_parse_simple_cron_expression(self):
        """Test parsing a simple cron expression."""
        from victor.workflows.scheduler import CronSchedule

        # Every day at 9 AM
        schedule = CronSchedule.from_cron("0 9 * * *")

        assert 0 in schedule.minute
        assert 9 in schedule.hour
        assert len(schedule.day_of_month) == 31
        assert len(schedule.month) == 12
        assert len(schedule.day_of_week) == 7

    def test_parse_cron_with_ranges(self):
        """Test parsing cron with range syntax."""
        from victor.workflows.scheduler import CronSchedule

        # Mon-Fri at 9 AM
        schedule = CronSchedule.from_cron("0 9 * * 1-5")

        assert schedule.day_of_week == {1, 2, 3, 4, 5}

    def test_parse_cron_with_steps(self):
        """Test parsing cron with step syntax."""
        from victor.workflows.scheduler import CronSchedule

        # Every 15 minutes
        schedule = CronSchedule.from_cron("*/15 * * * *")

        assert schedule.minute == {0, 15, 30, 45}

    def test_parse_cron_with_lists(self):
        """Test parsing cron with list syntax."""
        from victor.workflows.scheduler import CronSchedule

        # At 9 AM and 5 PM
        schedule = CronSchedule.from_cron("0 9,17 * * *")

        assert schedule.hour == {9, 17}

    def test_parse_cron_aliases(self):
        """Test parsing cron aliases."""
        from victor.workflows.scheduler import CronSchedule

        # @daily = 0 0 * * *
        schedule = CronSchedule.from_cron("@daily")

        assert schedule.minute == {0}
        assert schedule.hour == {0}

    def test_invalid_cron_expression_raises_error(self):
        """Test that invalid cron raises ValueError."""
        from victor.workflows.scheduler import CronSchedule

        with pytest.raises(ValueError, match="Invalid cron expression"):
            CronSchedule.from_cron("invalid")

    def test_cron_matches_datetime(self):
        """Test that cron matches correct datetime."""
        from victor.workflows.scheduler import CronSchedule

        # Every day at 9 AM
        schedule = CronSchedule.from_cron("0 9 * * *")

        # 9:00 AM on a Monday should match
        dt = datetime(2025, 1, 6, 9, 0, tzinfo=timezone.utc)  # Monday
        assert schedule.matches(dt)

        # 10:00 AM should not match
        dt = datetime(2025, 1, 6, 10, 0, tzinfo=timezone.utc)
        assert not schedule.matches(dt)

    def test_next_run_calculation(self):
        """Test calculating next run time."""
        from victor.workflows.scheduler import CronSchedule

        # Every hour at minute 0
        schedule = CronSchedule.from_cron("0 * * * *")

        # After 9:30, next run should be 10:00
        after = datetime(2025, 1, 6, 9, 30, tzinfo=timezone.utc)
        next_run = schedule.next_run(after)

        assert next_run.hour == 10
        assert next_run.minute == 0

    def test_from_interval_hours(self):
        """Test creating schedule from hourly interval."""
        from victor.workflows.scheduler import CronSchedule

        # Every 4 hours
        schedule = CronSchedule.from_interval(hours=4)

        assert schedule.hour == {0, 4, 8, 12, 16, 20}
        assert schedule.minute == {0}

    def test_from_interval_minutes(self):
        """Test creating schedule from minute interval."""
        from victor.workflows.scheduler import CronSchedule

        # Every 30 minutes
        schedule = CronSchedule.from_interval(minutes=30)

        assert schedule.minute == {0, 30}


class TestScheduledWorkflow:
    """Tests for ScheduledWorkflow configuration."""

    def test_scheduled_workflow_creation(self):
        """Test creating a scheduled workflow."""
        from victor.workflows.scheduler import CronSchedule, ScheduledWorkflow

        schedule = CronSchedule.from_cron("0 9 * * *")
        workflow = ScheduledWorkflow(
            workflow_name="daily_report",
            schedule=schedule,
            initial_state={"report_type": "daily"},
        )

        assert workflow.workflow_name == "daily_report"
        assert workflow.enabled
        assert workflow.next_run is not None

    def test_schedule_id_generation(self):
        """Test that schedule ID is generated."""
        from victor.workflows.scheduler import CronSchedule, ScheduledWorkflow

        schedule = CronSchedule.from_cron("0 9 * * *")
        workflow = ScheduledWorkflow(
            workflow_name="test_workflow",
            schedule=schedule,
        )

        assert workflow.schedule_id
        assert len(workflow.schedule_id) == 12

    def test_should_run_when_due(self):
        """Test should_run returns True when workflow is due."""
        from victor.workflows.scheduler import CronSchedule, ScheduledWorkflow

        schedule = CronSchedule.from_cron("0 9 * * *")
        workflow = ScheduledWorkflow(
            workflow_name="test",
            schedule=schedule,
        )

        # Set next_run to past
        workflow.next_run = datetime.now(timezone.utc) - timedelta(minutes=1)

        assert workflow.should_run()

    def test_should_not_run_when_disabled(self):
        """Test should_run returns False when disabled."""
        from victor.workflows.scheduler import CronSchedule, ScheduledWorkflow

        schedule = CronSchedule.from_cron("0 9 * * *")
        workflow = ScheduledWorkflow(
            workflow_name="test",
            schedule=schedule,
            enabled=False,
        )

        workflow.next_run = datetime.now(timezone.utc) - timedelta(minutes=1)

        assert not workflow.should_run()

    def test_should_not_run_when_max_active_reached(self):
        """Test should_run returns False when max active runs reached."""
        from victor.workflows.scheduler import CronSchedule, ScheduledWorkflow

        schedule = CronSchedule.from_cron("0 9 * * *")
        workflow = ScheduledWorkflow(
            workflow_name="test",
            schedule=schedule,
            max_active_runs=1,
        )

        workflow.next_run = datetime.now(timezone.utc) - timedelta(minutes=1)
        workflow.active_runs = 1

        assert not workflow.should_run()

    def test_mark_started_updates_state(self):
        """Test mark_started updates workflow state."""
        from victor.workflows.scheduler import CronSchedule, ScheduledWorkflow

        # Use every minute schedule to ensure next_run changes
        schedule = CronSchedule.from_cron("* * * * *")  # Every minute
        workflow = ScheduledWorkflow(
            workflow_name="test",
            schedule=schedule,
        )

        workflow.mark_started()

        assert workflow.active_runs == 1
        assert workflow.last_run is not None
        # next_run should be calculated and set
        assert workflow.next_run is not None

    def test_mark_completed_updates_state(self):
        """Test mark_completed updates workflow state."""
        from victor.workflows.scheduler import CronSchedule, ScheduledWorkflow

        schedule = CronSchedule.from_cron("0 9 * * *")
        workflow = ScheduledWorkflow(
            workflow_name="test",
            schedule=schedule,
        )

        workflow.mark_started()
        workflow.mark_completed()

        assert workflow.active_runs == 0
        assert workflow.run_count == 1


class TestWorkflowScheduler:
    """Tests for WorkflowScheduler."""

    def test_register_workflow(self):
        """Test registering a scheduled workflow."""
        from victor.workflows.scheduler import (
            CronSchedule,
            ScheduledWorkflow,
            WorkflowScheduler,
        )

        scheduler = WorkflowScheduler()
        schedule = CronSchedule.from_cron("0 9 * * *")
        workflow = ScheduledWorkflow(
            workflow_name="test",
            schedule=schedule,
        )

        schedule_id = scheduler.register(workflow)

        assert schedule_id
        assert scheduler.get_schedule(schedule_id) is workflow

    def test_unregister_workflow(self):
        """Test unregistering a scheduled workflow."""
        from victor.workflows.scheduler import (
            CronSchedule,
            ScheduledWorkflow,
            WorkflowScheduler,
        )

        scheduler = WorkflowScheduler()
        schedule = CronSchedule.from_cron("0 9 * * *")
        workflow = ScheduledWorkflow(
            workflow_name="test",
            schedule=schedule,
        )

        schedule_id = scheduler.register(workflow)
        result = scheduler.unregister(schedule_id)

        assert result
        assert scheduler.get_schedule(schedule_id) is None

    def test_list_schedules(self):
        """Test listing all schedules."""
        from victor.workflows.scheduler import (
            CronSchedule,
            ScheduledWorkflow,
            WorkflowScheduler,
        )

        scheduler = WorkflowScheduler()

        for i in range(3):
            schedule = CronSchedule.from_cron(f"0 {i} * * *")
            workflow = ScheduledWorkflow(
                workflow_name=f"test_{i}",
                schedule=schedule,
            )
            scheduler.register(workflow)

        schedules = scheduler.list_schedules()
        assert len(schedules) == 3

    def test_enable_disable_schedule(self):
        """Test enabling and disabling schedules."""
        from victor.workflows.scheduler import (
            CronSchedule,
            ScheduledWorkflow,
            WorkflowScheduler,
        )

        scheduler = WorkflowScheduler()
        schedule = CronSchedule.from_cron("0 9 * * *")
        workflow = ScheduledWorkflow(
            workflow_name="test",
            schedule=schedule,
        )

        schedule_id = scheduler.register(workflow)

        scheduler.disable(schedule_id)
        assert not scheduler.get_schedule(schedule_id).enabled

        scheduler.enable(schedule_id)
        assert scheduler.get_schedule(schedule_id).enabled


class TestScheduleYAMLParsing:
    """Tests for parsing schedule configuration from YAML."""

    @pytest.mark.asyncio
    async def test_parse_schedule_from_yaml(self):
        """Test parsing schedule configuration from YAML."""
        from victor.workflows.yaml_loader import (
            load_workflow_from_yaml,
            YAMLWorkflowConfig,
        )

        yaml_content = """
workflows:
  scheduled_workflow:
    description: "A scheduled workflow"
    schedule:
      cron: "0 9 * * *"
      timezone: "UTC"
      catchup: false
      max_active_runs: 2
    nodes:
      - id: task
        type: transform
        transform: "result = 'done'"
        next: []
"""

        config = YAMLWorkflowConfig()
        workflows = load_workflow_from_yaml(yaml_content, config=config)

        workflow_def = workflows.get("scheduled_workflow")
        assert workflow_def is not None

        # Check schedule is in metadata
        schedule = workflow_def.metadata.get("schedule")
        assert schedule is not None
        assert schedule["cron"] == "0 9 * * *"
        assert schedule["timezone"] == "UTC"
        assert schedule["catchup"] is False
        assert schedule["max_active_runs"] == 2


class TestScheduleWorkflowFunction:
    """Tests for the schedule_workflow convenience function."""

    def test_schedule_workflow_function(self):
        """Test the schedule_workflow convenience function."""
        from victor.workflows.scheduler import (
            schedule_workflow,
            get_scheduler,
        )

        # Clear any existing schedules
        scheduler = get_scheduler()
        for s in scheduler.list_schedules():
            scheduler.unregister(s.schedule_id)

        schedule_id = schedule_workflow(
            workflow_name="test_workflow",
            cron="0 12 * * *",
            initial_state={"key": "value"},
        )

        assert schedule_id
        workflow = scheduler.get_schedule(schedule_id)
        assert workflow is not None
        assert workflow.workflow_name == "test_workflow"
        assert workflow.initial_state == {"key": "value"}
