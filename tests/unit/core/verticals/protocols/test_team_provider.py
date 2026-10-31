"""victor.core.verticals.protocols.team_provider re-exports the SDK protocols.

``victor.framework.step_handlers`` checks ``isinstance`` against the
package-level (SDK) protocols, while ``focused.framework_extensions`` imported
the module-level copies; both paths now resolve to one object.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Optional

import pytest

MODULE = "victor.core.verticals.protocols.team_provider"
SDK = "victor_contracts.verticals.protocols.promoted"
NAMES = ["TeamSpecProviderProtocol", "VerticalTeamProviderProtocol"]


@pytest.mark.parametrize("name", NAMES)
def test_core_name_is_the_sdk_object(name: str) -> None:
    core = getattr(importlib.import_module(MODULE), name)
    assert core is getattr(importlib.import_module(SDK), name)
    assert core is getattr(importlib.import_module("victor.core.verticals.protocols"), name)


def test_focused_extensions_use_the_same_object() -> None:
    from victor.core.verticals.protocols.focused.framework_extensions import (
        TeamSpecProviderProtocol as Focused,
    )
    from victor.core.verticals.protocols.team_provider import TeamSpecProviderProtocol

    assert Focused is TeamSpecProviderProtocol


def test_public_names_unchanged() -> None:
    assert sorted(importlib.import_module(MODULE).__all__) == sorted(NAMES)


def test_runtime_checkable_matching_preserved() -> None:
    from victor.core.verticals.protocols.team_provider import (
        TeamSpecProviderProtocol,
        VerticalTeamProviderProtocol,
    )

    class _Teams:
        def get_team_specs(self) -> Dict[str, Any]:
            return {}

        def get_default_team(self) -> Optional[str]:
            return None

    class _Vertical:
        @classmethod
        def get_team_spec_provider(cls) -> Optional[Any]:
            return _Teams()

    assert isinstance(_Teams(), TeamSpecProviderProtocol)
    assert not isinstance(object(), TeamSpecProviderProtocol)
    assert isinstance(_Vertical(), VerticalTeamProviderProtocol)
    assert not isinstance(_Teams(), VerticalTeamProviderProtocol)
