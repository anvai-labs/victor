# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""Regression tests for decision session_id propagation (FEP-0012 spine).

Root cause these guard against: decisions logged during a task got an EMPTY
session_id (~90% of stage_detection/task_completion rows) because the
session_id contextvar set by the caller didn't reach the agentic loop's
decision-logging context (lost across async/thread boundaries). The fix:
the benchmark adapter stamps ``orchestrator.active_session_id`` and
``orchestrator.chat()`` re-stamps the contextvar from it, so every decision
the loop logs carries the task's session_id — and ``record_session_outcome``
can join them.
"""

from unittest.mock import MagicMock

from victor.core.context import get_session_id, set_session_id


async def test_orchestrator_chat_stamps_contextvar_from_active_session_id():
    """orchestrator.chat must set the session_id contextvar from active_session_id
    so the agentic loop's decision logs carry the task id."""
    from victor.agent.orchestrator import AgentOrchestrator

    # Bypass __init__ — chat() only needs active_session_id + _chat_service.
    orch = AgentOrchestrator.__new__(AgentOrchestrator)
    orch.active_session_id = "task-ABC-123"

    captured = {}

    async def fake_chat(msg, **kw):
        # Inside the delegated chat (same context as orchestrator.chat), the
        # contextvar must reflect active_session_id.
        captured["sid"] = get_session_id()
        return MagicMock(content="ok")

    orch._chat_service = MagicMock()
    orch._chat_service.chat = fake_chat

    set_session_id("")  # start clean — prove chat() is what stamps it
    await orch.chat("hello")

    assert captured["sid"] == "task-ABC-123", (
        f"decisions logged during chat would get session_id={captured['sid']!r} "
        "(must equal active_session_id for the outcome junction to join)"
    )


async def test_orchestrator_chat_no_active_session_keeps_contextvar_unchanged():
    """No active_session_id ⇒ chat() must not clobber an existing contextvar."""
    from victor.agent.orchestrator import AgentOrchestrator

    orch = AgentOrchestrator.__new__(AgentOrchestrator)
    orch.active_session_id = None

    async def fake_chat(msg, **kw):
        return MagicMock(content="ok")

    orch._chat_service = MagicMock()
    orch._chat_service.chat = fake_chat

    set_session_id("pre-existing-sid")
    await orch.chat("hello")
    assert get_session_id() == "pre-existing-sid"
