"""FEP-0042 Phase 2a: deprecations on the persona kernel's default-registry path."""

from __future__ import annotations

import warnings
from enum import Enum
from typing import Any, Dict, List, Optional

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


@pytest.fixture(autouse=True)
def _reset_default_registry() -> Any:
    saved: Optional[Any] = registries._default_persona_registry
    registries._default_persona_registry = None
    yield
    registries._default_persona_registry = saved


# --- register_vertical_personas(provider=None) ------------------------------


def test_default_provider_path_warns_and_is_attributed_to_caller() -> None:
    policy = RegistrationPolicy(vertical="demo")
    with pytest.warns(
        DeprecationWarning, match=r"without provider= is deprecated.*0\.14\.0.*0\.15\.0"
    ) as rec:
        assert register_vertical_personas(_personas(), policy) is None
    assert len(rec) == 1  # no second warning from the internal default-registry lookup
    assert rec[0].filename == __file__
    assert "get_personas()" in str(rec[0].message)


def test_default_provider_path_still_uses_installed_default_registry() -> None:
    provider = _Provider()
    registries._default_persona_registry = provider  # type: ignore[assignment]
    with pytest.warns(DeprecationWarning):
        count = register_vertical_personas(_personas(), RegistrationPolicy(vertical="demo"))
    assert count == 1
    assert provider.calls[0]["name"] == "tester"
    assert provider.calls[0]["vertical"] == "demo"


def test_explicit_provider_does_not_warn() -> None:
    provider = _Provider()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        count = register_vertical_personas(
            _personas(), RegistrationPolicy(vertical="demo"), provider=provider
        )
    assert count == 1
    assert provider.calls[0]["tags"] == ["testing", "tester"]


# --- RegistrationPolicy.provider_getter / auto_register ---------------------


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"provider_getter": "persona-runtime"}, "provider_getter"),
        ({"auto_register": False}, "auto_register"),
    ],
)
def test_non_default_inert_policy_fields_warn(kwargs: Dict[str, Any], field: str) -> None:
    with pytest.warns(
        DeprecationWarning, match=rf"RegistrationPolicy\.{field} is deprecated.*0\.14\.0.*0\.15\.0"
    ) as rec:
        policy = RegistrationPolicy(vertical="demo", **kwargs)
    assert len(rec) == 1
    assert rec[0].filename == __file__
    assert getattr(policy, field) == kwargs[field]  # still accepted


def test_both_inert_fields_warn_once_each() -> None:
    with pytest.warns(DeprecationWarning) as rec:
        RegistrationPolicy(vertical="demo", provider_getter="x", auto_register=False)
    assert sorted(str(w.message).split(" ")[0] for w in rec) == [
        "RegistrationPolicy.auto_register",
        "RegistrationPolicy.provider_getter",
    ]


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"provider_getter": "default", "auto_register": True}, {"tag_policy": "expertise"}],
)
def test_default_valued_policy_does_not_warn(kwargs: Dict[str, Any]) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        RegistrationPolicy(vertical="demo", **kwargs)
