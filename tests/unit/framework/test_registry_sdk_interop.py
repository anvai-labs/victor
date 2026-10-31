"""Interop tests for SDK registry adapters."""

from victor.framework.team_registry import get_team_registry
from victor_contracts.registries import get_default_team_registry


def test_team_registry_sets_sdk_default() -> None:
    registry = get_team_registry()

    assert get_default_team_registry() is registry
