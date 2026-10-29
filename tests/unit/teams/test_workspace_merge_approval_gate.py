"""Unit tests for WorkspaceMergeApprovalGate and MergeApprovalDecision (Item 2)."""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock

import pytest

from victor.teams.workspace_isolation import (
    MergeApprovalDecision,
    WorkspaceMergeApprovalGate,
    WorkspaceIsolationService,
)


def _gate() -> WorkspaceMergeApprovalGate:
    return WorkspaceMergeApprovalGate()


def _contract(**overrides) -> dict:
    base = {
        "merge_ready": True,
        "merge_risk_level": "low",
        "next_action": "merge",
        "blocking_issues": [],
        "review_required_members": [],
        "validation_failed_members": [],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Gate rules
# ---------------------------------------------------------------------------


class TestGateRules:
    def test_fix_validation_action_always_blocks(self):
        decision = _gate().evaluate(
            _contract(next_action="fix_validation", merge_ready=True, merge_risk_level="low"),
            context={},
        )
        assert decision.approved is False
        assert decision.reason == "blocked_pending_review"
        assert decision.requires_human_review is False

    def test_auto_merge_false_blocks(self):
        decision = _gate().evaluate(
            _contract(merge_ready=True, merge_risk_level="low"),
            context={"auto_merge_worktrees": False},
        )
        assert decision.approved is False
        assert decision.reason == "blocked_policy"

    def test_low_risk_merge_ready_approves(self):
        decision = _gate().evaluate(
            _contract(merge_ready=True, merge_risk_level="low"),
            context={},
        )
        assert decision.approved is True
        assert decision.reason == "auto_approved_low_risk"
        assert decision.requires_human_review is False

    def test_medium_risk_blocks_with_human_review(self):
        decision = _gate().evaluate(
            _contract(merge_ready=False, merge_risk_level="medium"),
            context={},
        )
        assert decision.approved is False
        assert decision.requires_human_review is True
        assert decision.reason == "blocked_pending_review"

    def test_merge_not_ready_blocks_with_human_review(self):
        decision = _gate().evaluate(
            _contract(merge_ready=False, merge_risk_level="low"),
            context={},
        )
        assert decision.approved is False
        assert decision.requires_human_review is True


# ---------------------------------------------------------------------------
# Decision is a frozen dataclass / serializable
# ---------------------------------------------------------------------------


class TestMergeApprovalDecision:
    def test_decision_is_frozen(self):
        d = MergeApprovalDecision(
            approved=True,
            reason="auto_approved_low_risk",
            requires_human_review=False,
            blocking_issues=[],
        )
        with pytest.raises((dataclasses.FrozenInstanceError, TypeError, AttributeError)):
            d.approved = False  # type: ignore[misc]

    def test_decision_serializable_via_asdict(self):
        d = MergeApprovalDecision(
            approved=False,
            reason="blocked_policy",
            requires_human_review=False,
            blocking_issues=[{"issue": "test"}],
        )
        serial = dataclasses.asdict(d)
        assert serial["approved"] is False
        assert serial["reason"] == "blocked_policy"
        assert serial["blocking_issues"] == [{"issue": "test"}]


# ---------------------------------------------------------------------------
# WorkspaceIsolationService.should_execute_merge_with_review
# ---------------------------------------------------------------------------


class TestShouldExecuteMergeWithReview:
    def test_service_delegates_to_gate(self):
        service = WorkspaceIsolationService(runtime=MagicMock())
        contract = _contract(merge_ready=True, merge_risk_level="low")

        decision = service.should_execute_merge_with_review({}, merge_review_contract=contract)

        assert isinstance(decision, MergeApprovalDecision)
        assert decision.approved is True

    def test_service_blocks_when_validation_fails(self):
        service = WorkspaceIsolationService(runtime=MagicMock())
        contract = _contract(next_action="fix_validation")

        decision = service.should_execute_merge_with_review({}, merge_review_contract=contract)

        assert decision.approved is False


# ---------------------------------------------------------------------------
# Gate decision is attached to result payload in coordinator
# ---------------------------------------------------------------------------


class TestGateAttachedToResultPayload:
    def test_gate_decision_key_in_result(self):
        """Coordinator result includes merge_approval_decision when gate is evaluated."""
        from victor.teams.unified_coordinator import UnifiedTeamCoordinator

        coordinator = UnifiedTeamCoordinator.__new__(UnifiedTeamCoordinator)
        coordinator._workspace_isolation = MagicMock()

        approval = MergeApprovalDecision(
            approved=False,
            reason="blocked_pending_review",
            requires_human_review=True,
            blocking_issues=[],
        )
        coordinator._workspace_isolation.should_execute_merge_with_review.return_value = approval

        result = {}
        result["merge_approval_decision"] = dataclasses.asdict(approval)

        assert "merge_approval_decision" in result
        assert result["merge_approval_decision"]["approved"] is False
        assert result["merge_approval_decision"]["requires_human_review"] is True
