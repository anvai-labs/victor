"""Protocol helpers that live in core and are re-exported by the framework."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pytest

import victor.core.protocol_support as core
import victor.framework.protocols as framework


def test_framework_reexports_are_the_core_objects():
    assert framework.IncompatibleVersionError is core.IncompatibleVersionError
    assert framework.verify_protocol_conformance is core.verify_protocol_conformance


def test_runtime_raises_the_class_framework_callers_catch():
    from victor.agent import capability_registry

    assert capability_registry.IncompatibleVersionError is framework.IncompatibleVersionError


def test_incompatible_version_error_carries_its_versions():
    err = core.IncompatibleVersionError("tools", "2.0", "1.3")
    assert (err.capability_name, err.required_version, err.actual_version) == (
        "tools",
        "2.0",
        "1.3",
    )
    assert "tools" in str(err) and "1.3" in str(err) and "2.0" in str(err)
    with pytest.raises(framework.IncompatibleVersionError):
        raise err


@runtime_checkable
class _Greeter(Protocol):
    def greet(self) -> str: ...

    def wave(self) -> None: ...


class _Annotated:
    name: str
    size: int


def test_conformance_passes_for_a_complete_object():
    class Full:
        def greet(self) -> str:
            return "hi"

        def wave(self) -> None:
            return None

    assert core.verify_protocol_conformance(Full(), _Greeter) == (True, [])


def test_conformance_reports_each_missing_member():
    class Partial:
        def greet(self) -> str:
            return "hi"

    assert core.verify_protocol_conformance(Partial(), _Greeter) == (False, ["wave"])


def test_conformance_falls_back_to_annotations():
    class HasName:
        name = "x"

    assert core.verify_protocol_conformance(HasName(), _Annotated) == (False, ["size"])
