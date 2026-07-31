"""ADR-023 pillar 2 (slice 2a): the terminal approval modal surfaces the member tag."""

from __future__ import annotations

from victor.framework.hitl import ApprovalRequest
from victor.ui.tui.approval_modal import ApprovalScreen


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
