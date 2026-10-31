"""Compatibility tests for SDK-backed vertical protocol re-exports."""

from __future__ import annotations

import importlib

import pytest


def test_load_sdk_vertical_extensions_falls_back_to_package_export(monkeypatch) -> None:
    """Core should tolerate SDK layouts without the extensions submodule."""

    from victor.core.verticals import protocols as protocols_module

    class PackageLevelVerticalExtensions:
        pass

    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name == "victor_contracts.verticals.extensions":
            raise ModuleNotFoundError(name)
        if name == "victor_contracts.verticals":
            return type(
                "FakeVerticalPackage",
                (),
                {"VerticalExtensions": PackageLevelVerticalExtensions},
            )
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    resolved = protocols_module._load_sdk_vertical_extensions()

    assert resolved is PackageLevelVerticalExtensions


def test_load_sdk_vertical_extensions_returns_none_when_sdk_layout_is_unavailable(
    monkeypatch,
) -> None:
    """Core should fall back locally when the SDK does not expose the type yet."""

    from victor.core.verticals import protocols as protocols_module

    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name.startswith("victor_contracts.verticals"):
            raise ModuleNotFoundError(name)
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    resolved = protocols_module._load_sdk_vertical_extensions()

    assert resolved is None


def test_middleware_protocol_reexport_is_identity():
    """jscpd consolidation: focused.tool_extensions re-exports the canonical
    MiddlewareProtocol instead of re-declaring it — the two names must be the
    SAME class object, now and forever."""
    from victor.core.verticals.protocols.focused.tool_extensions import (
        MiddlewareProtocol as Reexported,
    )
    from victor.core.verticals.protocols.middleware import MiddlewareProtocol

    assert Reexported is MiddlewareProtocol


_PROMOTED = "victor_contracts.verticals.protocols.promoted"


@pytest.mark.parametrize(
    "submodule, name",
    [
        ("enrichment", "EnrichmentStrategyProtocol"),
        ("enrichment", "VerticalEnrichmentProviderProtocol"),
        ("middleware", "MiddlewareProtocol"),
        ("rl_provider", "RLConfigProviderProtocol"),
        ("rl_provider", "VerticalRLProviderProtocol"),
        ("workflow_provider", "WorkflowProviderProtocol"),
        ("workflow_provider", "VerticalWorkflowProviderProtocol"),
    ],
)
def test_core_protocol_submodule_reexports_sdk_object(submodule: str, name: str) -> None:
    """Round-4 dedup: the per-concern submodules no longer re-declare the SDK
    protocols, so the submodule, package and SDK names are one class object."""
    core = getattr(importlib.import_module(f"victor.core.verticals.protocols.{submodule}"), name)
    assert core is getattr(importlib.import_module(_PROMOTED), name)
    assert core is getattr(importlib.import_module("victor.core.verticals.protocols"), name)


def test_middleware_submodule_reexports_sdk_types() -> None:
    from victor.core.verticals.protocols import middleware
    from victor_contracts.verticals.protocols import promoted_types

    assert middleware.MiddlewarePriority is promoted_types.MiddlewarePriority
    assert middleware.MiddlewareResult is promoted_types.MiddlewareResult
