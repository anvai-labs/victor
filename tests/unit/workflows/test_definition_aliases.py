"""victor.workflows.definition keeps its TeamNodeWorkflow compatibility alias."""

from __future__ import annotations

import pytest

import victor.workflows.definition as definition


def test_team_node_workflow_alias_warns_and_resolves() -> None:
    with pytest.warns(DeprecationWarning, match=r"TeamNodeWorkflow is deprecated.*v0\.9\.0"):
        alias = definition.TeamNodeWorkflow
    assert alias is definition.TeamStepWorkflow
    assert "TeamNodeWorkflow" in dir(definition)


def test_unknown_attribute_still_raises() -> None:
    with pytest.raises(AttributeError):
        _ = definition.TeamNodeWorkflowX  # type: ignore[attr-defined]
