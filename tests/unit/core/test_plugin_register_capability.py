# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for PluginContext.register_capability() — TDD RED phase."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pytest


@runtime_checkable
class MockProtocol(Protocol):
    def do_something(self) -> str: ...


class MockEnhancedProvider:
    def do_something(self) -> str:
        return "enhanced"


class MockStubProvider:
    def do_something(self) -> str:
        return "stub"


class TestRegisterCapability:
    """Phase 1: register_capability() flows through PluginContext to CapabilityRegistry."""

    def setup_method(self):
        from victor.core.capability_registry import CapabilityRegistry

        CapabilityRegistry._instance = None

    def teardown_method(self):
        from victor.core.capability_registry import CapabilityRegistry

        CapabilityRegistry._instance = None

    def test_register_capability_appears_in_registry(self):
        from victor.core.plugins.context import HostPluginContext
        from victor.core.capability_registry import CapabilityRegistry

        ctx = HostPluginContext(container=None)
        provider = MockEnhancedProvider()
        ctx.register_capability(MockProtocol, provider)

        registry = CapabilityRegistry.get_instance()
        assert registry.get(MockProtocol) is provider
        assert registry.is_enhanced(MockProtocol)

    def test_register_capability_lazy_defers(self):
        from victor.core.plugins.context import HostPluginContext
        from victor.core.capability_registry import CapabilityRegistry

        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return MockEnhancedProvider()

        ctx = HostPluginContext(container=None)
        ctx.register_capability(MockProtocol, factory, lazy=True)
        assert call_count == 0  # Not called yet

        registry = CapabilityRegistry.get_instance()
        result = registry.get(MockProtocol)
        # Access an attribute to trigger resolution
        assert result.do_something() == "enhanced"
        assert call_count == 1

    def test_stub_does_not_overwrite_enhanced(self):
        from victor.core.plugins.context import HostPluginContext
        from victor.core.capability_registry import CapabilityRegistry, CapabilityStatus

        ctx = HostPluginContext(container=None)
        enhanced = MockEnhancedProvider()
        ctx.register_capability(MockProtocol, enhanced)

        # Simulate bootstrap registering a STUB after plugin
        registry = CapabilityRegistry.get_instance()
        registry.register(MockProtocol, MockStubProvider(), CapabilityStatus.STUB)

        # Enhanced should survive
        assert registry.get(MockProtocol) is enhanced

    def test_sdk_protocol_has_register_capability(self):
        from victor_contracts.core.plugins import PluginContext

        assert hasattr(PluginContext, "register_capability")

    def test_mock_plugin_context_has_register_capability(self):
        from victor_contracts.testing import MockPluginContext

        ctx = MockPluginContext()
        assert hasattr(ctx, "register_capability")
        # Should be callable without error
        ctx.register_capability(MockProtocol, MockEnhancedProvider())
