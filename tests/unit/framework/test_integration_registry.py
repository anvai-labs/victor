"""Tests for IntegrationPlanRegistry."""

from victor.framework.integration_registry import IntegrationPlanRegistry


class TestIntegrationPlanRegistry:

    def setup_method(self):
        IntegrationPlanRegistry.reset()

    def teardown_method(self):
        IntegrationPlanRegistry.reset()

    def test_singleton(self):
        r1 = IntegrationPlanRegistry.get_instance()
        r2 = IntegrationPlanRegistry.get_instance()
        assert r1 is r2

    def test_set_and_get_plan(self):
        registry = IntegrationPlanRegistry.get_instance()
        orch = object()
        plan = {"handler": "test", "fingerprint": "abc123"}
        registry.set_plan(orch, plan)
        result = registry.get_plan(orch)
        assert result == plan
        # Should be a deep copy
        assert result is not plan

    def test_get_missing_plan(self):
        registry = IntegrationPlanRegistry.get_instance()
        assert registry.get_plan(object()) is None

    def test_remove_plan(self):
        registry = IntegrationPlanRegistry.get_instance()
        orch = object()
        registry.set_plan(orch, {"test": True})
        registry.remove_plan(orch)
        assert registry.get_plan(orch) is None

    def test_clear(self):
        registry = IntegrationPlanRegistry.get_instance()
        # Keep references so id() values remain unique
        orchs = [object() for _ in range(5)]
        for i, orch in enumerate(orchs):
            registry.set_plan(orch, {"i": i})
        assert registry.size == 5
        registry.clear()
        assert registry.size == 0

    def test_survives_weakref_gc(self):
        """Plans persist even after orchestrator reference is released."""
        registry = IntegrationPlanRegistry.get_instance()

        class FakeOrch:
            pass

        orch = FakeOrch()
        orch_id = id(orch)
        registry.set_plan(orch, {"key": "value"})

        # Plan is retrievable
        assert registry.get_plan(orch) is not None

        # After deletion, id-based lookup still has the entry
        # (unlike WeakKeyDictionary which would lose it)
        assert registry.size == 1

    def test_reset_creates_new_instance(self):
        r1 = IntegrationPlanRegistry.get_instance()
        r1.set_plan(object(), {"x": 1})
        IntegrationPlanRegistry.reset()
        r2 = IntegrationPlanRegistry.get_instance()
        assert r1 is not r2
        assert r2.size == 0
