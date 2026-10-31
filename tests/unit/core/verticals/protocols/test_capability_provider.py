"""victor.core.verticals.protocols.capability_provider re-exports the SDK protocols.

The module used to re-declare the protocols, so its classes were distinct from
``victor.core.verticals.protocols.*`` (already SDK-backed). One object now.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional

import pytest

MODULE = "victor.core.verticals.protocols.capability_provider"
SDK = "victor_contracts.verticals.protocols.promoted"
NAMES = [
    "CapabilityProviderProtocol",
    "ChainProviderProtocol",
    "PersonaProviderProtocol",
    "VerticalPersonaProviderProtocol",
]


@pytest.mark.parametrize("name", NAMES)
def test_core_name_is_the_sdk_object(name: str) -> None:
    core = getattr(importlib.import_module(MODULE), name)
    assert core is getattr(importlib.import_module(SDK), name)
    assert core is getattr(importlib.import_module("victor.core.verticals.protocols"), name)


def test_public_names_unchanged() -> None:
    assert sorted(importlib.import_module(MODULE).__all__) == sorted(NAMES)


def test_runtime_checkable_matching_preserved() -> None:
    from victor.core.verticals.protocols.capability_provider import (
        CapabilityProviderProtocol,
        VerticalPersonaProviderProtocol,
    )

    class _Caps:
        def get_capabilities(self) -> Dict[str, Any]:
            return {}

    class _Personas:
        @classmethod
        def get_persona_specs(cls) -> Dict[str, Any]:
            return {}

        @classmethod
        def get_default_persona(cls) -> Optional[str]:
            return None

        @classmethod
        def get_persona_tags(cls) -> List[str]:
            return []

    assert isinstance(_Caps(), CapabilityProviderProtocol)
    assert not isinstance(object(), CapabilityProviderProtocol)
    assert isinstance(_Personas(), VerticalPersonaProviderProtocol)
    assert not isinstance(_Caps(), VerticalPersonaProviderProtocol)
