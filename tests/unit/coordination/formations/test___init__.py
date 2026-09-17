"""Public dispatch regressions for WS-A: real members, outcomes and run isolation."""

from victor.coordination.formations import create_formation_registry
from victor.teams.types import TeamFormation


def test_registry_covers_every_enum_and_declares_durability():
    registry = create_formation_registry()
    assert set(registry) == set(TeamFormation)
    for formation in (
        TeamFormation.ADAPTIVE,
        TeamFormation.DYNAMIC_ROUTER,
        TeamFormation.MULTI_LEVEL_HIERARCHY,
    ):
        assert registry[formation].supports_durable_pause() is False
