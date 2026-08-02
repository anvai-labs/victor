# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for _AUTO_DETECT_SPECS deprecation — capabilities via plugins now."""

from __future__ import annotations

import pytest


class TestAutoDetectDeprecated:
    """Phase 3: _AUTO_DETECT_SPECS should be empty."""

    def test_auto_detect_specs_empty(self):
        """_AUTO_DETECT_SPECS list should be empty — capabilities flow through plugins."""
        from victor.core.bootstrap import _AUTO_DETECT_SPECS

        assert len(_AUTO_DETECT_SPECS) == 0, (
            f"_AUTO_DETECT_SPECS should be empty. Capabilities should be "
            f"registered via PluginContext.register_capability() instead. "
            f"Found {len(_AUTO_DETECT_SPECS)} specs: "
            f"{[s.get('label', s.get('protocol_attr')) for s in _AUTO_DETECT_SPECS]}"
        )

    def test_plugin_enhanced_survives_bootstrap_stubs(self):
        """ENHANCED capabilities from plugins survive STUB registration in bootstrap."""
        from victor.core.capability_registry import CapabilityRegistry, CapabilityStatus

        # Reset
        CapabilityRegistry._instance = None
        registry = CapabilityRegistry.get_instance()

        # Simulate plugin registering ENHANCED (happens in _phase_plugins)
        from typing import Protocol, runtime_checkable

        @runtime_checkable
        class TestProto(Protocol):
            def test(self) -> str: ...

        class Enhanced:
            def test(self) -> str:
                return "enhanced"

        class Stub:
            def test(self) -> str:
                return "stub"

        registry.register(TestProto, Enhanced(), CapabilityStatus.ENHANCED)

        # Simulate bootstrap registering STUB (happens in _phase_capabilities)
        registry.register(TestProto, Stub(), CapabilityStatus.STUB)

        # Enhanced should survive
        result = registry.get(TestProto)
        assert result.test() == "enhanced"
        assert registry.is_enhanced(TestProto)

        # Cleanup
        CapabilityRegistry._instance = None
