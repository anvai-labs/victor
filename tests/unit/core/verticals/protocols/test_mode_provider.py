"""victor.core.verticals.protocols.mode_provider re-exports the SDK definitions.

The module used to re-declare ``ModeConfig`` (a dataclass) and the provider
protocol, so ``isinstance(cfg, mode_provider.ModeConfig)`` rejected configs
built from the package-level (SDK) ``ModeConfig``. One object now.
"""

from __future__ import annotations

import dataclasses
import importlib
from typing import Dict

import pytest

MODULE = "victor.core.verticals.protocols.mode_provider"
PAIRS = [
    ("ModeConfig", "victor_contracts.verticals.protocols.promoted_types"),
    ("ModeConfigProviderProtocol", "victor_contracts.verticals.protocols.promoted"),
]


@pytest.mark.parametrize("name, sdk_mod", PAIRS)
def test_core_name_is_the_sdk_object(name: str, sdk_mod: str) -> None:
    core = getattr(importlib.import_module(MODULE), name)
    assert core is getattr(importlib.import_module(sdk_mod), name)
    assert core is getattr(importlib.import_module("victor.core.verticals.protocols"), name)


def test_public_names_unchanged() -> None:
    assert sorted(importlib.import_module(MODULE).__all__) == sorted(n for n, _ in PAIRS)


def test_mode_config_shape_and_defaults_preserved() -> None:
    from victor.core.verticals.protocols import ModeConfig as PackageModeConfig
    from victor.core.verticals.protocols.mode_provider import ModeConfig

    cfg = PackageModeConfig(name="fast", tool_budget=5, max_iterations=3)
    assert isinstance(cfg, ModeConfig)
    assert [(f.name, f.default) for f in dataclasses.fields(ModeConfig)] == [
        ("name", dataclasses.MISSING),
        ("tool_budget", dataclasses.MISSING),
        ("max_iterations", dataclasses.MISSING),
        ("temperature", 0.7),
        ("description", ""),
    ]


def test_runtime_checkable_matching_preserved() -> None:
    from victor.core.verticals.protocols.mode_provider import (
        ModeConfig,
        ModeConfigProviderProtocol,
    )

    class _Modes:
        def get_mode_configs(self) -> Dict[str, ModeConfig]:
            return {}

        def get_default_mode(self) -> str:
            return "fast"

        def get_default_tool_budget(self) -> int:
            return 5

    assert isinstance(_Modes(), ModeConfigProviderProtocol)
    assert not isinstance(object(), ModeConfigProviderProtocol)
