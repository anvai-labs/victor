"""Tests for SDK registry adapters."""

import warnings

import pytest

from victor_contracts.registries import (
    get_default_persona_registry,
    get_default_team_registry,
    set_default_persona_registry,
    set_default_team_registry,
)


def test_registry_setters_round_trip_defaults() -> None:
    team_registry = object()
    persona_registry = object()

    set_default_team_registry(team_registry)  # type: ignore[arg-type]
    with pytest.warns(DeprecationWarning):
        set_default_persona_registry(persona_registry)  # type: ignore[arg-type]

    assert get_default_team_registry() is team_registry
    with pytest.warns(DeprecationWarning):
        assert get_default_persona_registry() is persona_registry

    set_default_team_registry(None)
    with pytest.warns(DeprecationWarning):
        set_default_persona_registry(None)


@pytest.mark.parametrize(
    ("call", "name"),
    [
        (lambda: set_default_persona_registry(None), "set_default_persona_registry"),
        (get_default_persona_registry, "get_default_persona_registry"),
    ],
)
def test_default_persona_registry_hooks_are_deprecated(call, name) -> None:  # type: ignore[no-untyped-def]
    with pytest.warns(
        DeprecationWarning, match=rf"{name} is deprecated.*0\.14\.0.*0\.15\.0"
    ) as rec:
        call()
    assert len(rec) == 1
    # Attributed to the caller, not to registries.py.
    assert rec[0].filename == __file__


def test_default_team_registry_hooks_do_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        set_default_team_registry(None)
        assert get_default_team_registry() is None
