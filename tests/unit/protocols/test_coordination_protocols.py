"""Tests for public coordination protocol exports."""

from __future__ import annotations

import pytest


def test_coordination_advisor_protocol_remains_canonical() -> None:
    """The canonical coordination protocol should remain importable."""
    from victor.protocols.coordination import CoordinationAdvisorProtocol

    assert CoordinationAdvisorProtocol is not None


def test_mode_workflow_team_coordinator_protocol_removed() -> None:
    """The deprecated mode-workflow protocol alias should stay removed."""
    from victor.protocols import coordination

    assert "ModeWorkflowTeamCoordinatorProtocol" not in coordination.__all__

    with pytest.raises(ImportError, match="ModeWorkflowTeamCoordinatorProtocol"):
        from victor.protocols.coordination import (
            ModeWorkflowTeamCoordinatorProtocol,  # noqa: F401
        )  # noqa: F401
