"""ADR-023 pillar 2 (slice 2a): the terminal approval modal surfaces the member tag."""

from __future__ import annotations

import asyncio

import pytest
from textual.app import App

from victor.framework.hitl import ApprovalRequest
from victor.ui.tui.approval_modal import ApprovalScreen, make_tui_approval_handler


def _title_text(request: ApprovalRequest) -> str:
    """Build the approval-title markup (pure helper — no running app needed)."""
    return ApprovalScreen(request)._title_markup()


def test_title_shows_member_tag_when_present() -> None:
    request = ApprovalRequest(
        id="r1",
        title="run_command",
        description="",
        context={"member_id": "m1", "tool_name": "run_command"},
    )
    text = _title_text(request)
    assert "member m1" in text
    assert "run_command" in text


def test_title_has_no_member_tag_without_member_id() -> None:
    request = ApprovalRequest(
        id="r2",
        title="run_command",
        description="",
        context={"tool_name": "run_command"},
    )
    text = _title_text(request)
    assert "member" not in text
    assert "run_command" in text


# ── cancellation: a modal must not outlive its cancelled turn (audit B6) ──


class _ModalApp(App[None]):
    def compose(self):
        from textual.containers import Vertical

        yield Vertical()


def _request(timeout: int = 30) -> ApprovalRequest:
    return ApprovalRequest(
        id="cancel-1",
        title="run_command",
        description="",
        context={"tool_name": "run_command"},
        timeout_seconds=timeout,
    )


async def test_force_dismiss_pops_modal_without_decision() -> None:
    """force_dismiss pops immediately with no decision and stops the timer;
    a later timer firing (or keypress path) must not double-dismiss."""
    app = _ModalApp()
    results = []
    async with app.run_test() as pilot:
        screen = ApprovalScreen(_request())
        app.push_screen(screen, results.append)
        await pilot.pause()
        assert isinstance(app.screen, ApprovalScreen)
        screen.force_dismiss()
        await pilot.pause()
        assert results == [None]
        assert not isinstance(app.screen, ApprovalScreen)
        assert screen._decided  # a racing timeout can no longer decide


async def test_handler_cancellation_dismisses_the_modal() -> None:
    """Cancelling the task awaiting the approval handler (ESC interrupt kills
    the turn worker, which is blocked in the handler) must pop the modal right
    away instead of leaving the terminal modal-locked until the timeout."""
    app = _ModalApp()
    async with app.run_test() as pilot:
        handler = make_tui_approval_handler(app)
        task = asyncio.ensure_future(handler(_request(timeout=120)))
        await pilot.pause()
        assert isinstance(app.screen, ApprovalScreen)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await pilot.pause()
        assert not isinstance(app.screen, ApprovalScreen)
