"""victor.framework.workflows.nodes keeps its TeamNode* compatibility aliases."""

from __future__ import annotations

import pytest

import victor.framework.workflows.nodes as nodes


@pytest.mark.parametrize(
    "alias,target", [("TeamNode", "TeamStep"), ("TeamNodeConfig", "TeamStepConfig")]
)
def test_team_node_aliases_warn_and_resolve(alias: str, target: str) -> None:
    with pytest.warns(DeprecationWarning, match=rf"^{alias} is deprecated; use {target} instead"):
        resolved = getattr(nodes, alias)
    assert resolved is getattr(nodes, target)
    assert alias in dir(nodes)


def test_unknown_attribute_still_raises() -> None:
    with pytest.raises(AttributeError, match="has no attribute 'TeamNodeX'"):
        _ = nodes.TeamNodeX  # type: ignore[attr-defined]
