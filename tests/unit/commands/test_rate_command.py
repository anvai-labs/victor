# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""Tests for the /rate slash command (FEP-0012 manual RL feedback)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rich.console import Console

from victor.ui.slash import CommandContext
from victor.ui.slash.commands.rate import RateCommand


def _make_ctx(session_id: str = "sess-abc123") -> tuple:
    orch = MagicMock()
    orch.active_session_id = session_id
    agent = MagicMock()
    agent.get_orchestrator.return_value = orch
    agent.active_session_id = session_id
    ctx = CommandContext(
        console=Console(force_terminal=False, width=120),
        settings=SimpleNamespace(),
        agent=agent,
    )
    return ctx, agent, orch


async def test_rate_yes_logs_fulfilled_and_records_success():
    ctx, _agent, _orch = _make_ctx()
    with (
        patch("victor.ui.slash.commands.rate.Confirm.ask", return_value=True),
        patch("victor.agent.decisions.chain.log_decision") as log_dec,
        patch("victor.agent.decisions.outcome.record_session_outcome", return_value=3) as rec,
    ):
        await RateCommand().execute(ctx)

    log_dec.assert_called_once()
    assert log_dec.call_args.args[0] == "task_completion"
    assert log_dec.call_args.kwargs["result"] == "fulfilled"
    assert log_dec.call_args.kwargs["source"] == "manual_rate"
    assert log_dec.call_args.kwargs["session_id_override"] == "sess-abc123"
    rec.assert_called_once_with("sess-abc123", success=True, quality_score=1.0)


async def test_rate_no_logs_not_fulfilled_and_records_failure():
    ctx, _agent, _orch = _make_ctx()
    with (
        patch("victor.ui.slash.commands.rate.Confirm.ask", return_value=False),
        patch("victor.agent.decisions.chain.log_decision") as log_dec,
        patch("victor.agent.decisions.outcome.record_session_outcome", return_value=1) as rec,
    ):
        await RateCommand().execute(ctx)

    assert log_dec.call_args.kwargs["result"] == "not_fulfilled"
    rec.assert_called_once_with("sess-abc123", success=False, quality_score=0.0)


async def test_rate_without_session_is_noop():
    ctx, agent, orch = _make_ctx(session_id="")
    orch.active_session_id = ""
    agent.active_session_id = ""
    with patch("victor.agent.decisions.outcome.record_session_outcome") as rec:
        await RateCommand().execute(ctx)
    rec.assert_not_called()


async def test_rate_command_is_registered():
    import victor.ui.slash.commands  # noqa: F401  (triggers registration)

    from victor.ui.slash.registry import get_command_registry

    cmd = get_command_registry().get("rate")
    assert cmd is not None
    assert cmd.metadata.name == "rate"
    assert cmd.metadata.requires_agent is True
