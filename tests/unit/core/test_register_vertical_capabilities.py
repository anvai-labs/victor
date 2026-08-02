# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for register_vertical() auto-extracting capabilities — TDD."""

from __future__ import annotations

from typing import Any, List, Protocol, Tuple, Type, runtime_checkable

import pytest


@runtime_checkable
class MockCapProtocol(Protocol):
    def mock_method(self) -> str: ...


class MockCapProvider:
    def mock_method(self) -> str:
        return "mock"


class TestRegisterVerticalCapabilities:
    """Phase 2: register_vertical() auto-extracts capabilities."""

    def setup_method(self):
        from victor.core.capability_registry import CapabilityRegistry

        CapabilityRegistry._instance = None

    def teardown_method(self):
        from victor.core.capability_registry import CapabilityRegistry

        CapabilityRegistry._instance = None

    def test_register_vertical_uses_get_capability_registrations(self):
        """Vertical with get_capability_registrations() has caps auto-registered."""
        from victor.core.plugins.context import HostPluginContext
        from victor.core.capability_registry import CapabilityRegistry

        class FakeVertical:
            name = "test"

            @classmethod
            def get_name(cls):
                return "test"

            @classmethod
            def get_capability_registrations(cls) -> List[Tuple[Type, Any]]:
                return [(MockCapProtocol, MockCapProvider())]

        ctx = HostPluginContext(container=None)
        ctx.register_vertical(FakeVertical)

        registry = CapabilityRegistry.get_instance()
        result = registry.get(MockCapProtocol)
        assert result is not None
        assert result.mock_method() == "mock"

    def test_register_vertical_backward_compat(self):
        """Legacy vertical without get_capability_registrations() still registers."""
        from victor.core.plugins.context import HostPluginContext
        from victor.core.capability_registry import CapabilityRegistry

        class LegacyVertical:
            name = "legacy"

            @classmethod
            def get_name(cls):
                return "legacy"

        ctx = HostPluginContext(container=None)
        # Should not raise — no capability extraction crash
        ctx.register_vertical(LegacyVertical)
        # No capabilities should have been registered (no get_capability_registrations)
        registry = CapabilityRegistry.get_instance()
        assert registry.get(MockCapProtocol) is None

    def test_register_vertical_no_duplicate_capabilities(self):
        """Calling register_vertical twice doesn't crash or double-register."""
        from victor.core.plugins.context import HostPluginContext
        from victor.core.capability_registry import CapabilityRegistry

        class DupVertical:
            name = "dup"

            @classmethod
            def get_name(cls):
                return "dup"

            @classmethod
            def get_capability_registrations(cls):
                return [(MockCapProtocol, MockCapProvider())]

        ctx = HostPluginContext(container=None)
        ctx.register_vertical(DupVertical)
        ctx.register_vertical(DupVertical)  # Second call

        registry = CapabilityRegistry.get_instance()
        assert registry.get(MockCapProtocol) is not None

    def test_detect_enhanced_index_factory_is_pure_detector(self):
        """detect_enhanced_index_factory returns the registered factory directly,
        never wrapping it in the EnhancedCodebaseIndexFactory shell.

        Regression: the coding vertical used to register detect()'s return value
        (the shell) as the provider, which self-referenced (the shell's create()
        found only itself in the registry -> ImportError). detect() must be a
        pure detector so this misuse can't recur.
        """
        from victor.core.capability_registry import CapabilityRegistry
        from victor.core.search.indexer import (
            EnhancedCodebaseIndexFactory,
            detect_enhanced_index_factory,
        )
        from victor.framework.vertical_protocols import CodebaseIndexFactoryProtocol

        class _FakeRealFactory:
            def create(self, root_path, **kwargs):
                return "real-index"

        registry = CapabilityRegistry.get_instance()
        registry.register(CodebaseIndexFactoryProtocol, _FakeRealFactory())

        detected = detect_enhanced_index_factory()
        assert detected is not None
        # Whatever real factory is registered (the fake here, or the coding
        # vertical's real one if already registered at import), detect() must
        # NEVER return the shell — a vertical re-registering the shell
        # self-references and raises ImportError.
        assert not isinstance(
            detected, EnhancedCodebaseIndexFactory
        ), "detect() returned the shell — would self-reference if re-registered"
