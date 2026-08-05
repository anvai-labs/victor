# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for mixed-mode runtime module resolution during migration."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from victor.core.verticals.extension_loader import VerticalExtensionLoader
from victor.core.verticals.metadata import VerticalMetadataProvider
from victor.framework.escape_hatch_registry import EscapeHatchRegistry


class ResearchAssistant(VerticalExtensionLoader, VerticalMetadataProvider):
    """Minimal vertical class used to test shared mixed-mode resolution."""

    name = "research"
    description = "Research assistant"
    version = "1.0.0"


def _register_package(monkeypatch: pytest.MonkeyPatch, module_name: str) -> ModuleType:
    """Register a package module with a package-style ``__path__``."""
    module = ModuleType(module_name)
    module.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module_name, module)
    return module


def _register_module(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    **attributes: object,
) -> ModuleType:
    """Register a plain module with the provided attributes."""
    module = ModuleType(module_name)
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, module_name, module)
    return module


@pytest.fixture(autouse=True)
def _clear_shared_caches() -> None:
    """Reset caches shared across extension/escape-hatch resolution tests."""
    ResearchAssistant.clear_extension_cache(clear_all=True)
    EscapeHatchRegistry.reset_instance()
    yield
    ResearchAssistant.clear_extension_cache(clear_all=True)
    EscapeHatchRegistry.reset_instance()


def test_capability_configs_prefer_runtime_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capability configs should resolve from ``runtime.capabilities`` first."""
    _register_package(monkeypatch, "victor_research")
    _register_package(monkeypatch, "victor_research.runtime")
    _register_module(
        monkeypatch,
        "victor_research.runtime.capabilities",
        get_capability_configs=lambda: {"source": "runtime"},
    )
    _register_module(
        monkeypatch,
        "victor_research.capabilities",
        get_capability_configs=lambda: {"source": "root"},
    )

    assert ResearchAssistant.get_capability_configs() == {"source": "runtime"}


def test_capability_provider_prefers_runtime_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capability providers should resolve from ``runtime.capabilities`` first."""

    class RuntimeResearchCapabilityProvider:
        pass

    class RootResearchCapabilityProvider:
        pass

    _register_package(monkeypatch, "victor_research")
    _register_package(monkeypatch, "victor_research.runtime")
    _register_module(
        monkeypatch,
        "victor_research.runtime.capabilities",
        ResearchCapabilityProvider=RuntimeResearchCapabilityProvider,
    )
    _register_module(
        monkeypatch,
        "victor_research.capabilities",
        ResearchCapabilityProvider=RootResearchCapabilityProvider,
    )
    monkeypatch.setattr(
        "victor.framework.entry_point_loader.load_runtime_extension_from_entry_points",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        ResearchAssistant,
        "_extension_module_available",
        classmethod(lambda cls, module_path: module_path.startswith("victor_research")),
    )

    provider = ResearchAssistant.get_capability_provider()

    assert isinstance(provider, RuntimeResearchCapabilityProvider)


def test_safety_extension_prefers_runtime_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Safety extensions should resolve from ``runtime.safety`` before root shims."""

    class ResearchSafetyExtension:
        pass

    class RootResearchSafetyExtension:
        pass

    _register_package(monkeypatch, "victor_research")
    _register_package(monkeypatch, "victor_research.runtime")
    _register_module(
        monkeypatch,
        "victor_research.runtime.safety",
        ResearchSafetyExtension=ResearchSafetyExtension,
    )
    _register_module(
        monkeypatch,
        "victor_research.safety",
        ResearchSafetyExtension=RootResearchSafetyExtension,
    )
    monkeypatch.setattr(
        ResearchAssistant,
        "_extension_module_available",
        classmethod(lambda cls, module_path: module_path.startswith("victor_research")),
    )

    safety = ResearchAssistant.get_safety_extension()

    assert isinstance(safety, ResearchSafetyExtension)


def test_escape_hatch_registry_prefers_runtime_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Escape hatch discovery should prefer runtime modules during migration."""
    _register_package(monkeypatch, "victor_research")
    _register_package(monkeypatch, "victor_research.runtime")
    _register_module(
        monkeypatch,
        "victor_research.runtime.escape_hatches",
        CONDITIONS={"runtime_condition": lambda ctx: "runtime"},
        TRANSFORMS={"runtime_transform": lambda ctx: ctx},
    )
    _register_module(
        monkeypatch,
        "victor_research.escape_hatches",
        CONDITIONS={"root_condition": lambda ctx: "root"},
        TRANSFORMS={"root_transform": lambda ctx: ctx},
    )

    registry = EscapeHatchRegistry()
    registered = registry.discover_from_vertical("research")

    assert registered == (1, 1)
    assert registry.get_condition("runtime_condition", vertical="research") is not None
    assert registry.get_condition("root_condition", vertical="research") is None
