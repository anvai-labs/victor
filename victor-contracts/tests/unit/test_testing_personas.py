"""Tests for the persona-registration helpers in ``victor_contracts.testing.personas``."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import Any

import pytest

from victor_contracts.testing.personas import (
    RecordingPersonaRegistry,
    assert_explicit_persona_registration,
    assert_import_registers_no_personas,
)

_KERNEL_SETUP = """
from enum import Enum
from victor_contracts.verticals import (
    RegistrationPolicy, TraitSpec, VerticalPersona, VerticalTraits, register_vertical_personas,
)

class _E(Enum):
    X = "x"

class _T(VerticalTraits):
    spec = TraitSpec(
        communication_enum=_E, decision_enum=_E, style_hints={}, decision_hints={},
        defaults={"communication_style": "X", "decision_style": "X"},
    )
    def to_framework_traits(self, **kwargs):
        return kwargs

PERSONAS = {"p1": VerticalPersona(name="P", role="r", expertise=[_E.X], traits=_T())}
POLICY = RegistrationPolicy(vertical="fake")
"""


def _write_module(root: Path, name: str, body: str) -> None:
    (root / f"{name}.py").write_text(textwrap.dedent(_KERNEL_SETUP) + textwrap.dedent(body))


@pytest.fixture
def module_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(p for p in (str(tmp_path), path) if p))
    return tmp_path


def test_quiet_module_passes_import_probe(module_root: Path) -> None:
    _write_module(module_root, "quiet_personas", "")
    assert_import_registers_no_personas("quiet_personas")


def test_import_time_default_registration_fails_probe(module_root: Path) -> None:
    _write_module(module_root, "loud_personas", "register_vertical_personas(PERSONAS, POLICY)\n")
    with pytest.raises(
        AssertionError, match=r"(?s)loud_personas.*DeprecationWarning: register_vertical_personas"
    ):
        assert_import_registers_no_personas("loud_personas")


def test_import_time_explicit_default_registry_write_fails_probe(module_root: Path) -> None:
    _write_module(
        module_root,
        "sneaky_personas",
        """
        from victor_contracts import registries
        register_vertical_personas(PERSONAS, POLICY, registries._current_default_persona_registry())
        """,
    )
    with pytest.raises(AssertionError, match=r"import-time persona registration: \['p1'\]"):
        assert_import_registers_no_personas("sneaky_personas")


def test_import_probe_requires_a_module() -> None:
    with pytest.raises(ValueError):
        assert_import_registers_no_personas()


def _register_all(personas: dict[str, Any], vertical: str) -> Any:
    def register(provider: Any) -> None:
        for name in personas:
            provider.register_persona(name=name, vertical=vertical)

    return register


def test_explicit_registration_returns_calls() -> None:
    personas = {"a": object(), "b": object()}
    calls = assert_explicit_persona_registration(_register_all(personas, "v"), personas, "v")
    assert [c["name"] for c in calls] == ["a", "b"]


@pytest.mark.parametrize(
    ("registered", "vertical", "match"),
    [({"a": 1}, "v", "registered"), ({"a": 1, "b": 2}, "other", "verticals")],
)
def test_explicit_registration_mismatch_fails(
    registered: dict[str, Any], vertical: str, match: str
) -> None:
    with pytest.raises(AssertionError, match=match):
        assert_explicit_persona_registration(
            _register_all(registered, vertical), {"a": 1, "b": 2}, "v"
        )


def test_explicit_registration_rejects_sdk_deprecation_warning() -> None:
    import warnings

    def register(provider: Any) -> None:
        warnings.warn(
            "register_vertical_personas() without provider= is deprecated",
            DeprecationWarning,
            stacklevel=2,
        )

    with pytest.raises(DeprecationWarning):
        assert_explicit_persona_registration(register, {}, "v")


def test_explicit_registration_ignores_third_party_deprecations() -> None:
    import warnings

    def register(provider: Any) -> None:
        warnings.warn("some library API is deprecated", DeprecationWarning, stacklevel=2)
        provider.register_persona(name="a", vertical="v")

    assert_explicit_persona_registration(register, {"a": object()}, "v")


def test_recording_registry_records_kwargs() -> None:
    recorder = RecordingPersonaRegistry()
    recorder.register_persona(name="a", vertical="v")
    assert recorder.calls == [{"name": "a", "vertical": "v"}]
