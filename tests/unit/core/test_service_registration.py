"""Tests for declarative service registration helper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


from victor.core.container import ServiceContainer, ServiceLifetime
from victor.core.service_registration import (
    ServiceRegistrationSpec,
    register_services_from_specs,
)


class FooProtocol(Protocol):
    def do(self) -> str: ...


@dataclass
class FooImpl:
    value: str = "foo"

    def do(self) -> str:
        return self.value


class ScopedProtocol(Protocol):
    def ident(self) -> int: ...


@dataclass
class ScopedImpl:
    counter: int

    def ident(self) -> int:
        return self.counter


def test_register_services_from_specs_singleton():
    container = ServiceContainer()
    specs = [
        ServiceRegistrationSpec(
            protocol=FooProtocol,
            factory=lambda c: FooImpl(),
            lifetime=ServiceLifetime.SINGLETON,
        )
    ]

    register_services_from_specs(container, specs)

    inst1 = container.get(FooProtocol)
    inst2 = container.get(FooProtocol)

    assert isinstance(inst1, FooImpl)
    assert inst1 is inst2
    assert inst1.do() == "foo"


def test_register_services_from_specs_scoped_instances():
    container = ServiceContainer()
    counter = 0

    def scoped_factory(_):
        nonlocal counter
        counter += 1
        return ScopedImpl(counter)

    specs = [
        ServiceRegistrationSpec(
            protocol=ScopedProtocol,
            factory=scoped_factory,
            lifetime=ServiceLifetime.SCOPED,
        )
    ]

    register_services_from_specs(container, specs)

    with container.create_scope() as scope_a:
        a1 = scope_a.get(ScopedProtocol)
        a2 = scope_a.get(ScopedProtocol)
        assert a1 is a2
        assert a1.ident() == 1

    with container.create_scope() as scope_b:
        b = scope_b.get(ScopedProtocol)
        assert b.ident() == 2
        assert b is not a1
