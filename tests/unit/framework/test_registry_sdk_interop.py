"""Interop tests for SDK registry adapters."""

import pytest


from victor.framework.multi_agent.persona_provider import FrameworkPersonaProvider
from victor.framework.team_registry import get_team_registry
from victor_contracts.registries import (
    get_default_persona_registry,
    get_default_team_registry,
)


def test_team_registry_sets_sdk_default() -> None:
    registry = get_team_registry()

    assert get_default_team_registry() is registry


def test_persona_provider_sets_sdk_default() -> None:
    provider = FrameworkPersonaProvider()

    # The SDK getter is deprecated since victor-contracts 0.14.0.
    with pytest.warns(DeprecationWarning, match="get_default_persona_registry is deprecated"):
        assert get_default_persona_registry() is provider
