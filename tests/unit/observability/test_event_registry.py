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

"""Tests for EventCategoryRegistry and related event system (Workstream C).

Updated to use canonical ObservabilityBus from victor.core.events instead of
deprecated EventBus from victor.observability.event_bus.
"""

from unittest.mock import MagicMock, patch
import asyncio

import pytest

from victor.core.events import (
    MessagingEvent,
    ObservabilityBus,
    InMemoryEventBackend,
    BackendType,
    DeliveryGuarantee,
    get_observability_bus,
)
from victor.core.container import reset_container
from victor.core.events.emit_helper import stop_emit_sync_metrics_reporter
from victor.observability.event_registry import (
    CustomEventCategory,
    EventCategoryRegistry,
    resolve_subscription_topic_pattern,
)


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singletons before and after each test."""
    # Reset registry and container before tests
    EventCategoryRegistry.reset_instance()
    reset_container()
    stop_emit_sync_metrics_reporter()

    yield

    # Reset registry and container after tests
    EventCategoryRegistry.reset_instance()
    reset_container()
    stop_emit_sync_metrics_reporter()


# =============================================================================
# C.1: get_observability_bus() Factory Function Tests
# =============================================================================


class TestGetObservabilityBusFactory:
    """Tests for get_observability_bus() factory function (C.1)."""

    @pytest.mark.asyncio
    async def test_get_observability_bus_returns_singleton(self):
        """get_observability_bus should return the ObservabilityBus singleton."""
        bus = get_observability_bus()
        assert isinstance(bus, ObservabilityBus)

    @pytest.mark.asyncio
    async def test_get_observability_bus_returns_same_instance(self):
        """Multiple calls should return the same instance."""
        bus1 = get_observability_bus()
        bus2 = get_observability_bus()
        assert bus1 is bus2

    @pytest.mark.asyncio
    async def test_get_observability_bus_is_functional(self):
        """The returned ObservabilityBus should be fully functional."""
        bus = get_observability_bus()
        await bus.connect()

        events_received = []

        async def handler(event: MessagingEvent):
            events_received.append(event)

        await bus.subscribe("tool.*", handler)
        await bus.emit("tool.test_event", {"data": "test"})

        # Give async events time to process
        await asyncio.sleep(0.5)

        assert len(events_received) > 0
        assert events_received[0].topic == "tool.test_event"

        # Cleanup
        await bus.disconnect()

    def test_get_observability_bus_starts_emit_sync_metrics_reporter_when_enabled(self):
        """Factory should start sync emit metrics reporter when setting is enabled."""

        class _Settings:
            event_backend_type = "in_memory"
            event_emit_sync_metrics_enabled = True
            event_emit_sync_metrics_interval_seconds = 15.0
            event_emit_sync_metrics_reset_after_emit = True
            event_emit_sync_metrics_topic = "custom.emit.metrics"

        with patch("victor.config.settings.get_settings", return_value=_Settings()):
            with patch(
                "victor.core.events.emit_helper.start_emit_sync_metrics_reporter"
            ) as mock_start:
                bus = get_observability_bus()

        mock_start.assert_called_once()
        kwargs = mock_start.call_args.kwargs
        assert kwargs["interval_seconds"] == 15.0
        assert kwargs["topic"] == "custom.emit.metrics"
        assert kwargs["reset_after_emit"] is True
        assert callable(kwargs["event_bus_provider"])
        assert kwargs["event_bus_provider"]() is bus

    def test_get_observability_bus_does_not_start_reporter_when_disabled(self):
        """Factory should not start metrics reporter when disabled."""

        class _Settings:
            event_backend_type = "in_memory"
            event_emit_sync_metrics_enabled = False
            event_emit_sync_metrics_interval_seconds = 30.0
            event_emit_sync_metrics_reset_after_emit = False
            event_emit_sync_metrics_topic = "core.events.emit_sync.metrics"

        with patch("victor.config.settings.get_settings", return_value=_Settings()):
            with patch(
                "victor.core.events.emit_helper.start_emit_sync_metrics_reporter"
            ) as mock_start:
                get_observability_bus()

        mock_start.assert_not_called()

    def test_get_observability_bus_passes_backend_lazy_init_setting(self):
        """Factory should pass event_backend_lazy_init through backend creation."""

        class _Settings:
            event_backend_type = "redis"
            event_delivery_guarantee = "at_least_once"
            event_max_batch_size = 25
            event_flush_interval_ms = 400.0
            event_queue_maxsize = 42
            event_queue_overflow_policy = "drop_oldest"
            event_queue_overflow_block_timeout_ms = 33.0
            event_queue_overflow_topic_policies = {"critical.*": "block_with_timeout"}
            event_queue_overflow_topic_block_timeout_ms = {"critical.*": 77.0}
            event_backend_lazy_init = True
            event_emit_sync_metrics_enabled = False
            event_emit_sync_metrics_interval_seconds = 30.0
            event_emit_sync_metrics_reset_after_emit = False
            event_emit_sync_metrics_topic = "core.events.emit_sync.metrics"

        reset_container()
        try:
            with patch("victor.config.settings.get_settings", return_value=_Settings()):
                with patch(
                    "victor.core.events.backends.create_event_backend",
                    return_value=InMemoryEventBackend(),
                ) as mock_create:
                    get_observability_bus()
        finally:
            reset_container()

        mock_create.assert_called_once()
        assert mock_create.call_args.kwargs["lazy_init"] is True
        config = mock_create.call_args.kwargs["config"]
        assert config.backend_type == BackendType.REDIS
        assert config.delivery_guarantee == DeliveryGuarantee.AT_LEAST_ONCE
        assert config.max_batch_size == 25
        assert config.flush_interval_ms == 400.0
        assert config.extra["queue_maxsize"] == 42
        assert config.extra["queue_overflow_policy"] == "drop_oldest"
        assert config.extra["queue_overflow_block_timeout_ms"] == 33.0
        assert config.extra["queue_overflow_topic_policies"] == {"critical.*": "block_with_timeout"}
        assert config.extra["queue_overflow_topic_block_timeout_ms"] == {"critical.*": 77.0}


# =============================================================================
# C.3: CustomEventCategory Tests
# =============================================================================


class TestCustomEventCategory:
    """Tests for CustomEventCategory dataclass."""

    def test_custom_category_creation(self):
        """Should create CustomEventCategory with valid attributes."""
        category = CustomEventCategory(
            name="security_audit",
            description="Security audit events",
            registered_by="victor.security",
        )
        assert category.name == "security_audit"
        assert category.description == "Security audit events"
        assert category.registered_by == "victor.security"
        assert category.registered_at is not None

    def test_custom_category_immutable(self):
        """CustomEventCategory should be immutable (frozen dataclass)."""
        category = CustomEventCategory(
            name="security_audit",
            description="Security audit events",
            registered_by="victor.security",
        )
        with pytest.raises(AttributeError):
            category.name = "other_name"

    def test_custom_category_empty_name_raises(self):
        """Empty name should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            CustomEventCategory(
                name="",
                description="Test",
                registered_by="test",
            )

    def test_custom_category_invalid_name_uppercase_raises(self):
        """Uppercase names should raise ValueError."""
        with pytest.raises(ValueError, match="must be a valid lowercase identifier"):
            CustomEventCategory(
                name="SecurityAudit",
                description="Test",
                registered_by="test",
            )

    def test_custom_category_invalid_name_special_chars_raises(self):
        """Names with special characters should raise ValueError."""
        with pytest.raises(ValueError, match="must be a valid lowercase identifier"):
            CustomEventCategory(
                name="security-audit",
                description="Test",
                registered_by="test",
            )

    def test_custom_category_valid_names(self):
        """Valid lowercase identifier names should work."""
        valid_names = [
            "security_audit",
            "ml_pipeline",
            "custom123",
            "a",
            "test_category",
        ]
        for name in valid_names:
            category = CustomEventCategory(
                name=name,
                description="Test",
                registered_by="test",
            )
            assert category.name == name


# =============================================================================
# C.3: EventCategoryRegistry Tests
# =============================================================================


class TestEventCategoryRegistry:
    """Tests for EventCategoryRegistry singleton."""

    def test_get_instance_returns_singleton(self):
        """get_instance should return singleton."""
        registry1 = EventCategoryRegistry.get_instance()
        registry2 = EventCategoryRegistry.get_instance()
        assert registry1 is registry2

    def test_register_custom_category(self):
        """Should register custom category."""
        registry = EventCategoryRegistry.get_instance()
        category = registry.register(
            name="security_audit",
            description="Security audit events",
            registered_by="victor.security",
        )

        assert category.name == "security_audit"
        assert registry.has_category("security_audit")

    def test_register_duplicate_same_source_idempotent(self):
        """Registering same category from same source should be idempotent."""
        registry = EventCategoryRegistry.get_instance()

        cat1 = registry.register(
            name="ml_pipeline",
            description="ML pipeline events",
            registered_by="victor.ml",
        )
        cat2 = registry.register(
            name="ml_pipeline",
            description="ML pipeline events v2",  # Different description
            registered_by="victor.ml",  # Same source
        )

        # Should return existing category
        assert cat1 is cat2
        assert registry.count() == 1

    def test_register_duplicate_different_source_raises(self):
        """Registering same category from different source should raise."""
        registry = EventCategoryRegistry.get_instance()

        registry.register(
            name="ml_pipeline",
            description="ML pipeline events",
            registered_by="victor.ml",
        )

        with pytest.raises(ValueError, match="already registered by"):
            registry.register(
                name="ml_pipeline",
                description="Other events",
                registered_by="victor.other",
            )

    def test_register_builtin_name_raises(self):
        """Registering name that conflicts with built-in should raise."""
        registry = EventCategoryRegistry.get_instance()

        # "tool" is a built-in topic prefix
        with pytest.raises(ValueError, match="conflicts with built-in"):
            registry.register(
                name="tool",
                description="Custom tool events",
                registered_by="test",
            )

    def test_has_category_builtin(self):
        """has_category should return True for built-in categories."""
        registry = EventCategoryRegistry.get_instance()

        assert registry.has_category("tool") is True
        assert registry.has_category("state") is True
        assert registry.has_category("lifecycle") is True

    def test_has_category_custom(self):
        """has_category should return True for registered custom categories."""
        registry = EventCategoryRegistry.get_instance()

        assert registry.has_category("security_audit") is False

        registry.register(
            name="security_audit",
            description="Security audit events",
            registered_by="test",
        )

        assert registry.has_category("security_audit") is True

    def test_has_category_unknown(self):
        """has_category should return False for unknown categories."""
        registry = EventCategoryRegistry.get_instance()
        assert registry.has_category("nonexistent_category") is False

    def test_get_category(self):
        """get_category should return CustomEventCategory for custom categories."""
        registry = EventCategoryRegistry.get_instance()

        registry.register(
            name="security_audit",
            description="Security audit events",
            registered_by="victor.security",
        )

        category = registry.get_category("security_audit")
        assert category is not None
        assert category.name == "security_audit"

    def test_get_category_returns_none_for_builtin(self):
        """get_category should return None for built-in categories."""
        registry = EventCategoryRegistry.get_instance()
        assert registry.get_category("tool") is None

    def test_get_category_returns_none_for_unknown(self):
        """get_category should return None for unknown categories."""
        registry = EventCategoryRegistry.get_instance()
        assert registry.get_category("nonexistent") is None

    def test_list_custom(self):
        """list_custom should return only custom category names."""
        registry = EventCategoryRegistry.get_instance()

        registry.register(
            name="security_audit",
            description="Security",
            registered_by="test",
        )
        registry.register(
            name="ml_pipeline",
            description="ML",
            registered_by="test",
        )

        custom = registry.list_custom()
        assert custom == {"security_audit", "ml_pipeline"}

    def test_list_all(self):
        """list_all should return both built-in and custom categories."""
        registry = EventCategoryRegistry.get_instance()

        registry.register(
            name="security_audit",
            description="Security",
            registered_by="test",
        )

        all_cats = registry.list_all()

        # Should include built-in
        assert "tool" in all_cats
        assert "state" in all_cats
        assert "lifecycle" in all_cats

        # Should include custom
        assert "security_audit" in all_cats

    def test_get_all_custom(self):
        """get_all_custom should return dict of all custom categories."""
        registry = EventCategoryRegistry.get_instance()

        registry.register(
            name="security_audit",
            description="Security",
            registered_by="test",
        )

        all_custom = registry.get_all_custom()
        assert "security_audit" in all_custom
        assert isinstance(all_custom["security_audit"], CustomEventCategory)

    def test_count(self):
        """count should return number of custom categories."""
        registry = EventCategoryRegistry.get_instance()

        assert registry.count() == 0

        registry.register(name="cat1", description="Test 1", registered_by="test")
        assert registry.count() == 1

        registry.register(name="cat2", description="Test 2", registered_by="test")
        assert registry.count() == 2

    def test_unregister_success(self):
        """unregister should remove category when called by original registrant."""
        registry = EventCategoryRegistry.get_instance()

        registry.register(
            name="temp_category",
            description="Temporary",
            registered_by="test_module",
        )
        assert registry.has_category("temp_category") is True

        result = registry.unregister("temp_category", registered_by="test_module")

        assert result is True
        assert registry.has_category("temp_category") is False

    def test_unregister_wrong_source_raises(self):
        """unregister should raise when called by different registrant."""
        registry = EventCategoryRegistry.get_instance()

        registry.register(
            name="protected_category",
            description="Protected",
            registered_by="original_module",
        )

        with pytest.raises(ValueError, match="was registered by"):
            registry.unregister("protected_category", registered_by="other_module")

    def test_unregister_unknown_returns_false(self):
        """unregister should return False for unknown category."""
        registry = EventCategoryRegistry.get_instance()
        result = registry.unregister("nonexistent", registered_by="test")
        assert result is False

    def test_reset_instance(self):
        """reset_instance should clear all custom categories."""
        registry = EventCategoryRegistry.get_instance()

        registry.register(name="cat1", description="Test", registered_by="test")
        registry.register(name="cat2", description="Test", registered_by="test")
        assert registry.count() == 2

        EventCategoryRegistry.reset_instance()

        # Same singleton instance but cleared
        assert registry.count() == 0


class TestSubscriptionPatternResolution:
    """Tests for category/pattern -> topic pattern resolution helper."""

    def test_resolve_builtin_category(self):
        """Built-in categories should resolve to '<category>.*'."""
        assert resolve_subscription_topic_pattern("TOOL") == "tool.*"
        assert resolve_subscription_topic_pattern("lifecycle") == "lifecycle.*"

    def test_resolve_custom_registered_category(self):
        """Registered custom categories should resolve to '<category>.*'."""
        registry = EventCategoryRegistry.get_instance()
        registry.register(
            name="security_scan",
            description="Security scanning events",
            registered_by="victor.security",
        )

        assert resolve_subscription_topic_pattern("security_scan") == "security_scan.*"

    def test_resolve_all_wildcard_aliases(self):
        """'*' and 'ALL' aliases should resolve to wildcard subscription."""
        assert resolve_subscription_topic_pattern("*") == "*"
        assert resolve_subscription_topic_pattern("ALL") == "*"
        assert resolve_subscription_topic_pattern("all") == "*"

    def test_resolve_direct_pattern_passthrough(self):
        """Direct topic patterns should be passed through unchanged."""
        assert resolve_subscription_topic_pattern("tool.*") == "tool.*"
        assert resolve_subscription_topic_pattern("security_scan.vuln.*") == "security_scan.vuln.*"

    def test_resolve_unknown_category_raises(self):
        """Unknown category names should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown event category or topic pattern"):
            resolve_subscription_topic_pattern("does_not_exist")


# =============================================================================
# Integration Tests
# =============================================================================


class TestEventSystemIntegration:
    """Integration tests for the event system with canonical ObservabilityBus."""

    @pytest.mark.asyncio
    async def test_observability_bus_with_custom_topics(self):
        """ObservabilityBus should work with custom topic categories."""
        bus = get_observability_bus()
        await bus.connect()

        registry = EventCategoryRegistry.get_instance()

        # Register custom category
        registry.register(
            name="security_scan",
            description="Security scanning events",
            registered_by="victor.security",
        )

        # Use custom topic prefix
        events_received = []

        async def handler(event: MessagingEvent):
            events_received.append(event)

        await bus.subscribe("security_scan.*", handler)
        await bus.emit(
            "security_scan.vulnerability_found",
            {"severity": "high", "file": "config.py"},
        )

        # Give async events time to process
        await asyncio.sleep(0.5)

        assert len(events_received) > 0
        assert events_received[0].topic == "security_scan.vulnerability_found"
        assert events_received[0].data["severity"] == "high"
        assert registry.has_category("security_scan")

        # Cleanup
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_lifecycle_events(self):
        """Test lifecycle-related events with canonical event system."""
        bus = get_observability_bus()
        await bus.connect()

        events_received = []

        async def handler(event: MessagingEvent):
            events_received.append(event)

        await bus.subscribe("lifecycle.*", handler)

        # Emit lifecycle events
        await bus.emit("lifecycle.graph_started", {"graph_id": "main"})
        await bus.emit("lifecycle.workflow_completed", {"status": "success"})
        await bus.emit("lifecycle.session_started", {"session_id": "abc123"})

        # Give async events time to process
        await asyncio.sleep(0.5)

        assert len(events_received) == 3
        topics = [e.topic for e in events_received]
        assert "lifecycle.graph_started" in topics
        assert "lifecycle.workflow_completed" in topics
        assert "lifecycle.session_started" in topics

        # Cleanup
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_custom_category_integration_with_bus(self):
        """Custom categories should integrate with ObservabilityBus topics."""
        bus = get_observability_bus()
        await bus.connect()

        registry = EventCategoryRegistry.get_instance()

        # Register multiple custom categories
        registry.register(
            name="ml_pipeline",
            description="ML pipeline events",
            registered_by="victor.ml",
        )
        registry.register(
            name="data_processing",
            description="Data processing events",
            registered_by="victor.data",
        )

        # Subscribe to all custom categories
        events_received = []

        async def handler(event: MessagingEvent):
            events_received.append(event)

        await bus.subscribe("ml_pipeline.*", handler)
        await bus.subscribe("data_processing.*", handler)

        # Emit events to both custom categories
        await bus.emit("ml_pipeline.training_started", {"model": "classifier"})
        await bus.emit("data_processing.batch_complete", {"batch_id": 123})

        # Give async events time to process
        await asyncio.sleep(0.5)

        assert len(events_received) == 2
        assert registry.has_category("ml_pipeline")
        assert registry.has_category("data_processing")

        # Cleanup
        await bus.disconnect()
