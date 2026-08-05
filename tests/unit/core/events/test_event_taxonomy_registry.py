# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0

"""Tests for EventTaxonomyRegistry — unified event type correlation.

The registry maps domain-specific event enums to the canonical
EventType without merging them, enabling discovery and correlation
across the 8+ domain event systems.
"""

from __future__ import annotations

from enum import Enum

import pytest

from victor.framework.events import EventType


class MockWorkflowEventType(str, Enum):
    AGENT_CONTENT = "agent_content"
    AGENT_ERROR = "agent_error"
    STEP_COMPLETE = "step_complete"


class MockTeamEventType(str, Enum):
    MEMBER_JOINED = "member_joined"
    TASK_ASSIGNED = "task_assigned"


class TestEventTaxonomyRegistry:
    """Test the event taxonomy registry."""

    def setup_method(self):
        from victor.core.events.taxonomy import EventTaxonomyRegistry

        EventTaxonomyRegistry.clear()

    def teardown_method(self):
        from victor.core.events.taxonomy import EventTaxonomyRegistry

        EventTaxonomyRegistry.clear()

    def test_register_domain(self):
        """Can register a domain with its event type enum and mapping."""
        from victor.core.events.taxonomy import EventTaxonomyRegistry

        mapping = {
            MockWorkflowEventType.AGENT_CONTENT: EventType.CONTENT,
            MockWorkflowEventType.AGENT_ERROR: EventType.ERROR,
        }
        EventTaxonomyRegistry.register_domain("workflow", MockWorkflowEventType, mapping)
        assert "workflow" in EventTaxonomyRegistry.list_domains()

    def test_to_canonical(self):
        """Domain event maps to canonical EventType."""
        from victor.core.events.taxonomy import EventTaxonomyRegistry

        mapping = {
            MockWorkflowEventType.AGENT_CONTENT: EventType.CONTENT,
            MockWorkflowEventType.AGENT_ERROR: EventType.ERROR,
        }
        EventTaxonomyRegistry.register_domain("workflow", MockWorkflowEventType, mapping)
        result = EventTaxonomyRegistry.to_canonical("workflow", MockWorkflowEventType.AGENT_CONTENT)
        assert result == EventType.CONTENT

    def test_unmapped_returns_custom(self):
        """Unmapped domain event returns EventType.CUSTOM."""
        from victor.core.events.taxonomy import EventTaxonomyRegistry

        mapping = {
            MockWorkflowEventType.AGENT_CONTENT: EventType.CONTENT,
        }
        EventTaxonomyRegistry.register_domain("workflow", MockWorkflowEventType, mapping)
        result = EventTaxonomyRegistry.to_canonical("workflow", MockWorkflowEventType.STEP_COMPLETE)
        assert result == EventType.CUSTOM

    def test_from_canonical(self):
        """Reverse lookup: canonical → list of (domain, event) tuples."""
        from victor.core.events.taxonomy import EventTaxonomyRegistry

        EventTaxonomyRegistry.register_domain(
            "workflow",
            MockWorkflowEventType,
            {MockWorkflowEventType.AGENT_CONTENT: EventType.CONTENT},
        )
        EventTaxonomyRegistry.register_domain(
            "team",
            MockTeamEventType,
            {MockTeamEventType.MEMBER_JOINED: EventType.CONTENT},
        )
        results = EventTaxonomyRegistry.from_canonical(EventType.CONTENT)
        assert len(results) == 2
        domains = {r[0] for r in results}
        assert domains == {"workflow", "team"}

    def test_list_domains(self):
        """Returns all registered domain names."""
        from victor.core.events.taxonomy import EventTaxonomyRegistry

        EventTaxonomyRegistry.register_domain("workflow", MockWorkflowEventType, {})
        EventTaxonomyRegistry.register_domain("team", MockTeamEventType, {})
        domains = EventTaxonomyRegistry.list_domains()
        assert set(domains) == {"workflow", "team"}

    def test_duplicate_domain_overwrites(self):
        """Re-registering a domain replaces the previous mapping."""
        from victor.core.events.taxonomy import EventTaxonomyRegistry

        EventTaxonomyRegistry.register_domain(
            "workflow",
            MockWorkflowEventType,
            {MockWorkflowEventType.AGENT_CONTENT: EventType.CONTENT},
        )
        EventTaxonomyRegistry.register_domain(
            "workflow",
            MockWorkflowEventType,
            {MockWorkflowEventType.AGENT_CONTENT: EventType.THINKING},
        )
        result = EventTaxonomyRegistry.to_canonical("workflow", MockWorkflowEventType.AGENT_CONTENT)
        assert result == EventType.THINKING

    def test_clear_registry(self):
        """clear() removes all registered domains."""
        from victor.core.events.taxonomy import EventTaxonomyRegistry

        EventTaxonomyRegistry.register_domain("workflow", MockWorkflowEventType, {})
        assert len(EventTaxonomyRegistry.list_domains()) == 1

        EventTaxonomyRegistry.clear()
        assert len(EventTaxonomyRegistry.list_domains()) == 0

    def test_unknown_domain_returns_custom(self):
        """Querying unregistered domain returns CUSTOM."""
        from victor.core.events.taxonomy import EventTaxonomyRegistry

        result = EventTaxonomyRegistry.to_canonical("nonexistent", "some_event")
        assert result == EventType.CUSTOM
