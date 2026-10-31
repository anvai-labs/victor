"""FEP-0042 Phase 3: the persona kernel requires an explicit provider (0.15.0)."""

from __future__ import annotations

import warnings
from enum import Enum
from typing import Any, Dict, List

import pytest

from victor_contracts import registries
from victor_contracts.verticals import (
    RegistrationPolicy,
    TraitSpec,
    VerticalPersona,
    VerticalTraits,
    register_vertical_personas,
)


class _Comm(Enum):
    TECHNICAL = "technical"


class _Decision(Enum):
    ANALYTICAL = "analytical"


class _Expertise(Enum):
    TESTING = "testing"


class _FrameworkTraits:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _Traits(VerticalTraits):
    spec = TraitSpec(
        communication_enum=_Comm,
        decision_enum=_Decision,
        style_hints={},
        decision_hints={},
        defaults={"communication_style": "TECHNICAL", "decision_style": "ANALYTICAL"},
        framework_mapping=lambda traits: ({}, {}),
    )

    def to_framework_traits(self, **kwargs: Any) -> Any:  # type: ignore[override]
        return _FrameworkTraits(**kwargs)


def _personas() -> Dict[str, VerticalPersona]:
    return {
        "tester": VerticalPersona(
            name="Tess", role="tester", expertise=[_Expertise.TESTING], traits=_Traits()
        )
    }


class _Provider:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def register_persona(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


# --- provider is required ---------------------------------------------------


def test_provider_is_a_required_argument() -> None:
    with pytest.raises(TypeError, match="provider"):
        register_vertical_personas(_personas(), RegistrationPolicy(vertical="demo"))  # type: ignore[call-arg]


def test_none_provider_is_rejected_with_migration_hint() -> None:
    with pytest.raises(TypeError, match=r"requires a provider.*get_personas\(\)"):
        register_vertical_personas(
            _personas(), RegistrationPolicy(vertical="demo"), None  # type: ignore[arg-type]
        )


def test_explicit_provider_registers_without_warning() -> None:
    provider = _Provider()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        count = register_vertical_personas(
            _personas(), RegistrationPolicy(vertical="demo"), provider=provider
        )
    assert count == 1
    assert provider.calls[0]["name"] == "tester"
    assert provider.calls[0]["vertical"] == "demo"
    assert provider.calls[0]["tags"] == ["testing", "tester"]


# --- removed surfaces ---------------------------------------------------------


@pytest.mark.parametrize("field", ["provider_getter", "auto_register"])
def test_removed_policy_fields_are_rejected(field: str) -> None:
    with pytest.raises(TypeError, match=field):
        RegistrationPolicy(vertical="demo", **{field: "x"})


@pytest.mark.parametrize(
    "name",
    [
        "set_default_persona_registry",
        "get_default_persona_registry",
        "_current_default_persona_registry",
        "_default_persona_registry",
    ],
)
def test_default_persona_registry_hooks_are_removed(name: str) -> None:
    import victor_contracts

    assert not hasattr(registries, name)
    assert not hasattr(victor_contracts, name)
    assert name not in victor_contracts.__all__


def test_policy_construction_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        RegistrationPolicy(vertical="demo", tag_policy="expertise")
