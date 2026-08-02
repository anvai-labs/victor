# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0

"""Integration tests verifying wiring between new components and core agent loop.

These tests ensure that:
1. register_vertical_tools() propagates to ToolTierRegistry
2. EventTaxonomyRegistry has domain registrations from real modules
3. VerticalLoader calls manifest class validation during negotiation
"""

from __future__ import annotations

import logging

import pytest


class TestToolTierRegistryPropagation:
    """Verify register_vertical_tools() propagates to ToolTierRegistry."""

    def setup_method(self):
        from victor.core.vertical_types import TieredToolTemplate

        TieredToolTemplate._registered_verticals.clear()

    def teardown_method(self):
        from victor.core.tool_tier_registry import ToolTierRegistry
        from victor.core.vertical_types import TieredToolTemplate

        TieredToolTemplate._registered_verticals.clear()
        ToolTierRegistry.reset_instance()

    def test_register_vertical_tools_propagates_to_tier_registry(self):
        """Calling register_vertical_tools() also registers in ToolTierRegistry."""
        from victor.core.tool_tier_registry import ToolTierRegistry
        from victor.core.vertical_types import TieredToolTemplate

        ToolTierRegistry.reset_instance()

        TieredToolTemplate.register_vertical_tools(
            "test_v",
            core_tools={"edit", "write"},
            readonly_for_analysis=False,
        )

        registry = ToolTierRegistry.get_instance()
        config = registry.get("test_v")
        assert config is not None, "Tier should be registered in ToolTierRegistry"
        assert "edit" in config.vertical_core

    def test_unregister_vertical_tools_removes_from_tier_registry(self):
        """Calling unregister_vertical_tools() also unregisters from ToolTierRegistry."""
        from victor.core.tool_tier_registry import ToolTierRegistry
        from victor.core.vertical_types import TieredToolTemplate

        ToolTierRegistry.reset_instance()

        TieredToolTemplate.register_vertical_tools("temp_v", core_tools={"shell"})
        assert ToolTierRegistry.get_instance().get("temp_v") is not None

        TieredToolTemplate.unregister_vertical_tools("temp_v")
        assert ToolTierRegistry.get_instance().get("temp_v") is None


class TestEventTaxonomyDomainRegistrations:
    """Verify domain modules register with EventTaxonomyRegistry."""

    def _ensure_registrations(self):
        """Re-trigger domain registrations after any clear()."""
        from victor.workflows.streaming import _register_workflow_event_taxonomy
        from victor.framework.teams import _register_team_event_taxonomy
        from victor.framework.rl.hooks import _register_rl_event_taxonomy

        _register_workflow_event_taxonomy()
        _register_team_event_taxonomy()
        _register_rl_event_taxonomy()

    def test_workflow_domain_registered(self):
        """WorkflowEventType is registered via module-level call."""
        from victor.core.events.taxonomy import EventTaxonomyRegistry

        self._ensure_registrations()
        assert "workflow" in EventTaxonomyRegistry.list_domains()

    def test_team_domain_registered(self):
        """TeamEventType is registered via module-level call."""
        from victor.core.events.taxonomy import EventTaxonomyRegistry

        self._ensure_registrations()
        assert "team" in EventTaxonomyRegistry.list_domains()

    def test_rl_domain_registered(self):
        """RLEventType is registered via module-level call."""
        from victor.core.events.taxonomy import EventTaxonomyRegistry

        self._ensure_registrations()
        assert "rl" in EventTaxonomyRegistry.list_domains()

    def test_workflow_content_maps_to_canonical(self):
        """WorkflowEventType.AGENT_CONTENT maps to EventType.CONTENT."""
        from victor.core.events.taxonomy import EventTaxonomyRegistry
        from victor.framework.events import EventType
        from victor.workflows.streaming import WorkflowEventType

        self._ensure_registrations()
        canonical = EventTaxonomyRegistry.to_canonical("workflow", WorkflowEventType.AGENT_CONTENT)
        assert canonical == EventType.CONTENT

    def test_team_error_maps_to_canonical(self):
        """TeamEventType.TEAM_ERROR maps to EventType.ERROR."""
        from victor.core.events.taxonomy import EventTaxonomyRegistry
        from victor.framework.events import EventType
        from victor.framework.teams import TeamEventType

        self._ensure_registrations()
        canonical = EventTaxonomyRegistry.to_canonical("team", TeamEventType.TEAM_ERROR)
        assert canonical == EventType.ERROR


class TestManifestValidationWiring:
    """Verify manifest class validation is called during vertical negotiation."""

    def test_negotiate_calls_class_validation(self, caplog):
        """CapabilityNegotiator.negotiate() validates class references from manifest."""
        from victor.core.verticals.capability_negotiator import CapabilityNegotiator
        from victor_contracts.verticals.manifest import ExtensionManifest

        negotiator = CapabilityNegotiator()
        manifest = ExtensionManifest(
            api_version=2,
            name="test_vertical",
            version="1.0.0",
        )
        # Attach stale class reference metadata
        manifest.module_path = "victor.nonexistent.module"
        manifest.class_name = "FakeClass"

        with caplog.at_level(logging.WARNING):
            result = negotiator.negotiate(manifest)

        # Should have warning about stale class reference
        class_warnings = [
            r
            for r in caplog.records
            if "module" in r.message.lower() or "class" in r.message.lower()
        ]
        assert len(class_warnings) >= 1 or any("module" in w for w in result.warnings)
