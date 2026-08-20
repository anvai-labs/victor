# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0

"""Tests verifying legacy tool tier coupling has been removed from core.

These tests enforce that TieredToolTemplate no longer contains hardcoded
vertical names. All verticals must register via register_vertical_tools()
at activation time (OCP compliance).
"""

from __future__ import annotations

import pytest

from victor.core.vertical_types import TieredToolTemplate


class TestLegacyDictsRemoved:
    """Legacy hardcoded dicts should be empty or removed."""

    def setup_method(self):
        """Clean up any registered verticals before each test."""
        TieredToolTemplate._registered_verticals.clear()

    def teardown_method(self):
        """Clean up after each test."""
        TieredToolTemplate._registered_verticals.clear()

    def test_legacy_cores_is_empty(self):
        """_LEGACY_CORES should be empty — no hardcoded vertical names."""
        assert TieredToolTemplate._LEGACY_CORES == {}

    def test_legacy_readonly_is_empty(self):
        """_LEGACY_READONLY should be empty — no hardcoded vertical names."""
        assert TieredToolTemplate._LEGACY_READONLY == {}

    def test_backward_compat_aliases_are_empty(self):
        """VERTICAL_CORES and VERTICAL_READONLY_DEFAULTS should be empty."""
        assert TieredToolTemplate.VERTICAL_CORES == {}
        assert TieredToolTemplate.VERTICAL_READONLY_DEFAULTS == {}


class TestConvenienceMethodsRemoved:
    """Vertical-specific convenience methods should be removed."""

    def test_for_coding_removed(self):
        assert not hasattr(TieredToolTemplate, "for_coding")

    def test_for_research_removed(self):
        assert not hasattr(TieredToolTemplate, "for_research")

    def test_for_devops_removed(self):
        assert not hasattr(TieredToolTemplate, "for_devops")

    def test_for_data_analysis_removed(self):
        assert not hasattr(TieredToolTemplate, "for_data_analysis")

    def test_for_rag_removed(self):
        assert not hasattr(TieredToolTemplate, "for_rag")

    def test_register_vertical_legacy_removed(self):
        """Old register_vertical() that wrote to VERTICAL_CORES is removed."""
        assert not hasattr(TieredToolTemplate, "register_vertical")


class TestDynamicRegistryOnly:
    """for_vertical() should only use the dynamic registry."""

    def setup_method(self):
        TieredToolTemplate._registered_verticals.clear()

    def teardown_method(self):
        TieredToolTemplate._registered_verticals.clear()

    def test_unregistered_vertical_returns_none(self):
        """Without registration, for_vertical() returns None."""
        result = TieredToolTemplate.for_vertical("coding")
        assert result is None

    def test_register_vertical_tools_roundtrip(self):
        """register_vertical_tools() then for_vertical() works."""
        TieredToolTemplate.register_vertical_tools(
            "coding",
            core_tools={"edit", "write", "shell", "git"},
            readonly_for_analysis=False,
        )
        config = TieredToolTemplate.for_vertical("coding")
        assert config is not None
        assert "edit" in config.vertical_core
        assert "write" in config.vertical_core
        assert config.readonly_only_for_analysis is False

    def test_register_multiple_verticals(self):
        """Multiple verticals can register independently."""
        TieredToolTemplate.register_vertical_tools(
            "alpha", core_tools={"tool_a"}, readonly_for_analysis=True
        )
        TieredToolTemplate.register_vertical_tools(
            "beta", core_tools={"tool_b"}, readonly_for_analysis=False
        )
        alpha = TieredToolTemplate.for_vertical("alpha")
        beta = TieredToolTemplate.for_vertical("beta")
        assert alpha is not None and "tool_a" in alpha.vertical_core
        assert beta is not None and "tool_b" in beta.vertical_core

    def test_unregister_vertical_tools(self):
        """unregister_vertical_tools() removes a vertical's config."""
        TieredToolTemplate.register_vertical_tools("temp", core_tools={"tool_x"})
        assert TieredToolTemplate.for_vertical("temp") is not None

        TieredToolTemplate.unregister_vertical_tools("temp")
        assert TieredToolTemplate.for_vertical("temp") is None

    def test_unregister_nonexistent_is_safe(self):
        """Unregistering a non-existent vertical does not raise."""
        TieredToolTemplate.unregister_vertical_tools("nonexistent")

    def test_list_verticals_from_registry(self):
        """list_verticals() returns only dynamically registered names."""
        TieredToolTemplate.register_vertical_tools("v1", core_tools={"a"})
        TieredToolTemplate.register_vertical_tools("v2", core_tools={"b"})
        names = TieredToolTemplate.list_verticals()
        assert set(names) == {"v1", "v2"}


class TestToolTierRegistryNoLegacy:
    """ToolTierRegistry._register_defaults() should only create base tier."""

    def test_only_base_tier_registered(self):
        from victor.core.tool_tier_registry import ToolTierRegistry

        ToolTierRegistry.reset_instance()
        registry = ToolTierRegistry.get_instance()

        # Base tier should exist
        assert registry.get("base") is not None

        # Vertical-specific tiers should NOT be auto-created
        assert registry.get("coding") is None
        assert registry.get("research") is None
        assert registry.get("devops") is None

        ToolTierRegistry.reset_instance()
