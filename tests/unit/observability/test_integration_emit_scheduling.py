# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for observability event scheduling helpers."""

from unittest.mock import AsyncMock, MagicMock, patch

from victor.observability.integration import ObservabilityIntegration


class TestObservabilityIntegrationEmitScheduling:
    """Validate async event scheduling behavior."""

    def test_schedule_emit_uses_running_loop(self):
        """Scheduling should enqueue bus emits on the active event loop."""
        integration = ObservabilityIntegration(event_bus=MagicMock())
        integration._bus.emit = AsyncMock(return_value=True)
        scheduled = []
        loop = MagicMock()

        def capture_task(coro):
            scheduled.append(coro)
            coro.close()
            return MagicMock()

        loop.create_task.side_effect = capture_task

        with patch(
            "victor.observability.integration.asyncio.get_running_loop",
            return_value=loop,
        ):
            integration._schedule_emit(topic="tool.start", data={"category": "tool"})

        assert len(scheduled) == 1
        integration._bus.emit.assert_called_once_with(
            topic="tool.start",
            data={"category": "tool"},
            correlation_id=None,
        )

    def test_schedule_emit_skips_without_loop(self):
        """Scheduling should no-op cleanly when no event loop is active."""
        integration = ObservabilityIntegration(event_bus=MagicMock())
        integration._bus.emit = AsyncMock(return_value=True)

        with patch(
            "victor.observability.integration.asyncio.get_running_loop",
            side_effect=RuntimeError,
        ):
            integration._schedule_emit(topic="tool.start", data={"category": "tool"})

        integration._bus.emit.assert_not_called()

    def test_on_tool_start_uses_scheduler_helper(self):
        """Public tool-start hook should route through the shared scheduler."""
        integration = ObservabilityIntegration(event_bus=MagicMock(), session_id="session-123")

        with patch.object(integration, "_schedule_emit") as schedule_emit:
            integration.on_tool_start("read_file", {"path": "/tmp/example.py"}, tool_id="tool-1")

        schedule_emit.assert_called_once()
        kwargs = schedule_emit.call_args.kwargs
        assert kwargs["topic"] == "tool.start"
        assert kwargs["correlation_id"] == "session-123"
