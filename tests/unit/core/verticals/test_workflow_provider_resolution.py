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

"""Tests for vertical workflow provider import resolution behavior."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import patch

from victor.core.verticals.workflow_provider import VerticalWorkflowProvider


class _ResearchVertical(VerticalWorkflowProvider):
    name = "research"


def test_get_handlers_loads_external_module_first(monkeypatch) -> None:
    """Handlers should be loaded from external package when available."""
    handlers_module = ModuleType("victor_research.handlers")
    handlers_module.HANDLERS = {"external_handler": object()}
    monkeypatch.setitem(sys.modules, "victor_research.handlers", handlers_module)

    with patch(
        "victor.core.verticals.workflow_provider.vertical_runtime_module_candidates",
        return_value=["victor_research.handlers", "victor.research.handlers"],
    ):
        handlers = _ResearchVertical.get_handlers()

    assert "external_handler" in handlers


def test_get_handlers_falls_back_to_legacy_module(monkeypatch) -> None:
    """Handlers should fall back to legacy module when external import is missing."""
    legacy_pkg = ModuleType("victor.research")
    legacy_pkg.__path__ = []  # type: ignore[attr-defined]
    legacy_handlers = ModuleType("victor.research.handlers")
    legacy_handlers.HANDLERS = {"legacy_handler": object()}
    monkeypatch.setitem(sys.modules, "victor.research", legacy_pkg)
    monkeypatch.setitem(sys.modules, "victor.research.handlers", legacy_handlers)

    with patch(
        "victor.core.verticals.workflow_provider.vertical_runtime_module_candidates",
        return_value=[
            "victor_research_nonexistent.handlers",  # fails import → fallback
            "victor.research.handlers",  # legacy fallback (monkeypatched)
        ],
    ):
        handlers = _ResearchVertical.get_handlers()

    assert "legacy_handler" in handlers


def test_get_handlers_prefers_runtime_module_when_available(monkeypatch) -> None:
    """Handlers should resolve the runtime package before the root shim."""
    external_pkg = ModuleType("victor_research")
    external_pkg.__path__ = []  # type: ignore[attr-defined]
    runtime_pkg = ModuleType("victor_research.runtime")
    runtime_pkg.__path__ = []  # type: ignore[attr-defined]
    runtime_handlers = ModuleType("victor_research.runtime.handlers")
    runtime_handlers.HANDLERS = {"runtime_handler": object()}
    root_handlers = ModuleType("victor_research.handlers")
    root_handlers.HANDLERS = {"root_handler": object()}

    monkeypatch.setitem(sys.modules, "victor_research", external_pkg)
    monkeypatch.setitem(sys.modules, "victor_research.runtime", runtime_pkg)
    monkeypatch.setitem(sys.modules, "victor_research.runtime.handlers", runtime_handlers)
    monkeypatch.setitem(sys.modules, "victor_research.handlers", root_handlers)

    with patch(
        "victor.core.verticals.workflow_provider.vertical_runtime_module_candidates",
        return_value=["victor_research.runtime.handlers", "victor_research.handlers"],
    ):
        handlers = _ResearchVertical.get_handlers()

    assert "runtime_handler" in handlers
    assert "root_handler" not in handlers


def test_get_workflow_provider_loads_from_external_module(monkeypatch) -> None:
    """Workflow provider should instantiate from resolved external module."""

    class ResearchWorkflowProvider:
        pass

    workflows_module = ModuleType("victor_research.workflows")
    workflows_module.ResearchWorkflowProvider = ResearchWorkflowProvider
    monkeypatch.setitem(sys.modules, "victor_research.workflows", workflows_module)

    with patch(
        "victor.core.verticals.workflow_provider.vertical_runtime_module_candidates",
        side_effect=[
            ["victor_research.workflows"],
            ["victor_research.workflows.provider"],
        ],
    ):
        provider = _ResearchVertical.get_workflow_provider()

    assert isinstance(provider, ResearchWorkflowProvider)


def test_get_workflow_provider_prefers_runtime_module(monkeypatch) -> None:
    """Workflow providers should resolve runtime modules before root shims."""

    class ResearchWorkflowProvider:
        pass

    external_pkg = ModuleType("victor_research")
    external_pkg.__path__ = []  # type: ignore[attr-defined]
    runtime_pkg = ModuleType("victor_research.runtime")
    runtime_pkg.__path__ = []  # type: ignore[attr-defined]
    runtime_workflows = ModuleType("victor_research.runtime.workflows")
    runtime_workflows.ResearchWorkflowProvider = ResearchWorkflowProvider

    monkeypatch.setitem(sys.modules, "victor_research", external_pkg)
    monkeypatch.setitem(sys.modules, "victor_research.runtime", runtime_pkg)
    monkeypatch.setitem(sys.modules, "victor_research.runtime.workflows", runtime_workflows)

    with patch(
        "victor.core.verticals.workflow_provider.vertical_runtime_module_candidates",
        side_effect=[
            ["victor_research.runtime.workflows", "victor_research.workflows"],
            [
                "victor_research.runtime.workflows.provider",
                "victor_research.workflows.provider",
            ],
        ],
    ):
        provider = _ResearchVertical.get_workflow_provider()

    assert isinstance(provider, ResearchWorkflowProvider)


def test_get_workflow_provider_returns_none_when_no_candidate_imports() -> None:
    """Workflow provider should return None when no import candidate resolves."""
    with patch(
        "victor.core.verticals.workflow_provider.vertical_runtime_module_candidates",
        return_value=["nonexistent.module.path"],
    ):
        provider = _ResearchVertical.get_workflow_provider()

    assert provider is None
