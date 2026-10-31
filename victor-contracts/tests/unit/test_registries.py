"""Tests for SDK registry adapters."""

import warnings

from victor_contracts.registries import (
    get_default_team_registry,
    set_default_team_registry,
)


def test_team_registry_setter_round_trips_default() -> None:
    team_registry = object()
    set_default_team_registry(team_registry)  # type: ignore[arg-type]
    try:
        assert get_default_team_registry() is team_registry
    finally:
        set_default_team_registry(None)


def test_default_team_registry_hooks_do_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        set_default_team_registry(None)
        assert get_default_team_registry() is None


def test_private_install_helper_is_a_no_op_for_published_victor_ai_022() -> None:
    """victor-ai 0.22.0's deprecated FrameworkPersonaProvider imports this helper."""
    from victor_contracts import registries

    assert registries._install_default_persona_registry(object()) is None
    assert "_install_default_persona_registry" not in registries.__all__
    assert not hasattr(registries, "get_default_persona_registry")
