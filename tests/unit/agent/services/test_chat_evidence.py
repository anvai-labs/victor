"""Focused contracts for task evidence at the ChatTurnRuntime boundary."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from victor.agent.services.chat_evidence import ChatEvidenceMixin


class _Evidence(ChatEvidenceMixin):
    def __init__(self, turn_runtime):
        self._turn_runtime = turn_runtime
        self._last_task_report = {"stale": True}
        self._logger = MagicMock()


@pytest.mark.asyncio
async def test_task_report_lifecycle_uses_one_turn_runtime_capability() -> None:
    runtime = SimpleNamespace(
        start_task_report=MagicMock(return_value="task-1"),
        finish_task_report=MagicMock(return_value={"task_id": "task-1"}),
    )
    evidence = _Evidence(runtime)

    await evidence._start_task_report("inspect app.py", stream=True, metadata={"source": "test"})
    await evidence._finish_task_report(
        True,
        user_message="inspect app.py",
        stream=True,
        metadata={"source": "test"},
    )

    runtime.start_task_report.assert_called_once_with(
        "inspect app.py", stream=True, metadata={"source": "test"}
    )
    runtime.finish_task_report.assert_called_once_with(
        True,
        user_message="inspect app.py",
        stream=True,
        response=None,
        error=None,
        metadata={"source": "test"},
    )
    assert evidence.get_last_task_report() == {"task_id": "task-1"}


@pytest.mark.asyncio
async def test_missing_turn_runtime_keeps_reporting_optional() -> None:
    evidence = _Evidence(None)

    await evidence._start_task_report("hello", stream=False)
    await evidence._finish_task_report(True, user_message="hello", stream=False)

    assert evidence.get_last_task_report() is None
