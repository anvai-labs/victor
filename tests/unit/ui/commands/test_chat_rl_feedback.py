# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""Tests for the chat RL auto-prompt (FEP-0012 reward loop).

Covers ``_maybe_prompt_outcome``: it should fire only on a fresh completion
transition (detector ``should_stop`` flips False→True), log the missing
``task_completion`` spine, then record the human reward.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rich.console import Console


def _make_agent(should_stop: bool, prev_completed: bool = False, session_id: str = "sess-xyz"):
    detector = MagicMock()
    detector.should_stop.return_value = should_stop
    detector._state = SimpleNamespace(completion_percentage=80.0)
    orch = MagicMock()
    orch._task_completion_detector = detector
    orch.active_session_id = session_id
    orch._rl_prev_completed = prev_completed
    agent = MagicMock()
    agent.get_orchestrator.return_value = orch
    return agent, orch


def _run(agent, settings):
    from victor.ui.commands.chat import _maybe_prompt_outcome

    _maybe_prompt_outcome(agent, settings, Console(force_terminal=False, width=120))


def test_transition_to_complete_logs_spine_and_records_success():
    agent, orch = _make_agent(True, prev_completed=False)
    with (
        patch("victor.agent.decisions.chain.log_decision") as log_dec,
        patch("victor.agent.decisions.outcome.record_session_outcome", return_value=5) as rec,
        patch("rich.prompt.Prompt.ask", return_value="y"),
    ):
        _run(agent, SimpleNamespace(enable_rl_feedback_prompt=True))

    log_dec.assert_called_once()
    assert log_dec.call_args.args[0] == "task_completion"
    assert log_dec.call_args.kwargs["result"] == "fulfilled"
    assert log_dec.call_args.kwargs["session_id_override"] == "sess-xyz"
    rec.assert_called_once_with("sess-xyz", success=True, quality_score=1.0)
    assert orch._rl_prev_completed is True


def test_stale_complete_does_not_reprompt():
    # Detector already said "complete" last turn → not a fresh transition.
    agent, _orch = _make_agent(True, prev_completed=True)
    with (
        patch("victor.agent.decisions.chain.log_decision") as log_dec,
        patch("victor.agent.decisions.outcome.record_session_outcome") as rec,
    ):
        _run(agent, SimpleNamespace(enable_rl_feedback_prompt=True))
    log_dec.assert_not_called()
    rec.assert_not_called()


def test_not_complete_does_not_prompt():
    agent, _orch = _make_agent(False, prev_completed=False)
    with patch("victor.agent.decisions.chain.log_decision") as log_dec:
        _run(agent, SimpleNamespace(enable_rl_feedback_prompt=True))
    log_dec.assert_not_called()


def test_passive_mode_logs_spine_without_prompting():
    """Default (passive): the task_completion spine is logged, no blocking
    prompt fires, and no explicit outcome is recorded — the reward loop and
    /rate own outcome recording now."""
    agent, orch = _make_agent(True, prev_completed=False)
    orch._rate_hint_shown = False
    with (
        patch("victor.agent.decisions.chain.log_decision") as log_dec,
        patch("victor.agent.decisions.outcome.record_session_outcome") as rec,
        patch("rich.prompt.Prompt.ask") as ask,
    ):
        _run(agent, SimpleNamespace(enable_rl_feedback_prompt=False))
    log_dec.assert_called_once()
    assert log_dec.call_args.args[0] == "task_completion"
    ask.assert_not_called()
    rec.assert_not_called()
    assert orch._rate_hint_shown is True


def test_rate_hint_shown_once_per_session():
    agent, orch = _make_agent(True, prev_completed=False)
    orch._rate_hint_shown = True
    console = MagicMock()
    with patch("victor.agent.decisions.chain.log_decision"):
        from victor.ui.commands.chat import _maybe_prompt_outcome

        _maybe_prompt_outcome(agent, SimpleNamespace(enable_rl_feedback_prompt=False), console)
    console.print.assert_not_called()


def test_skip_answer_logs_spine_but_records_nothing():
    agent, _orch = _make_agent(True, prev_completed=False)
    with (
        patch("victor.agent.decisions.chain.log_decision") as log_dec,
        patch("victor.agent.decisions.outcome.record_session_outcome") as rec,
        patch("rich.prompt.Prompt.ask", return_value="skip"),
    ):
        _run(agent, SimpleNamespace(enable_rl_feedback_prompt=True))
    # The task_completion spine is still logged; the reward is just skipped.
    log_dec.assert_called_once()
    rec.assert_not_called()


def test_no_answer_records_failure():
    agent, _orch = _make_agent(True, prev_completed=False)
    with (
        patch("victor.agent.decisions.chain.log_decision"),
        patch("victor.agent.decisions.outcome.record_session_outcome", return_value=2) as rec,
        patch("rich.prompt.Prompt.ask", return_value="n"),
    ):
        _run(agent, SimpleNamespace(enable_rl_feedback_prompt=True))
    rec.assert_called_once_with("sess-xyz", success=False, quality_score=0.0)
