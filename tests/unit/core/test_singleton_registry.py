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

"""Unit tests for SingletonRegistry and ItemRegistry base classes.

Tests thread-safety, singleton isolation, and common registry patterns.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

import pytest

from victor.core.registry import ItemRegistry, SingletonRegistry


class TestSingletonRegistry:
    """Tests for SingletonRegistry base class."""

    def setup_method(self) -> None:
        """Reset test registries before each test."""
        # Create fresh subclasses for each test to ensure isolation
        pass

    def test_compatibility_facade_exports_canonical_classes(self) -> None:
        """registry_base should re-export the canonical registry package classes."""
        from victor.core.registry_base import ItemRegistry as CompatItemRegistry
        from victor.core.registry_base import (
            SingletonRegistry as CompatSingletonRegistry,
        )

        assert CompatItemRegistry is ItemRegistry
        assert CompatSingletonRegistry is SingletonRegistry

    def test_get_instance_returns_singleton(self) -> None:
        """get_instance() should return the same instance."""

        class TestRegistry(SingletonRegistry["TestRegistry"]):
            pass

        try:
            instance1 = TestRegistry.get_instance()
            instance2 = TestRegistry.get_instance()
            assert instance1 is instance2
        finally:
            TestRegistry.reset_instance()

    def test_reset_instance_clears_singleton(self) -> None:
        """reset_instance() should clear the singleton."""

        class TestRegistry(SingletonRegistry["TestRegistry"]):
            pass

        try:
            instance1 = TestRegistry.get_instance()
            TestRegistry.reset_instance()
            instance2 = TestRegistry.get_instance()
            assert instance1 is not instance2
        finally:
            TestRegistry.reset_instance()

    def test_subclass_isolation(self) -> None:
        """Each subclass should have its own singleton instance."""

        class Registry1(SingletonRegistry["Registry1"]):
            pass

        class Registry2(SingletonRegistry["Registry2"]):
            pass

        try:
            r1 = Registry1.get_instance()
            r2 = Registry2.get_instance()
            assert r1 is not r2
            assert type(r1) is Registry1
            assert type(r2) is Registry2
        finally:
            Registry1.reset_instance()
            Registry2.reset_instance()

    def test_is_initialized(self) -> None:
        """is_initialized() should correctly report state."""

        class TestRegistry(SingletonRegistry["TestRegistry"]):
            pass

        try:
            assert not TestRegistry.is_initialized()
            TestRegistry.get_instance()
            assert TestRegistry.is_initialized()
            TestRegistry.reset_instance()
            assert not TestRegistry.is_initialized()
        finally:
            TestRegistry.reset_instance()

    def test_thread_safety_concurrent_get_instance(self) -> None:
        """Concurrent get_instance() calls should return same instance."""

        class TestRegistry(SingletonRegistry["TestRegistry"]):
            def __init__(self) -> None:
                super().__init__()
                self.init_count = 0
                self.init_count += 1

        try:
            instances = []
            errors = []

            def get_instance():
                try:
                    inst = TestRegistry.get_instance()
                    instances.append(inst)
                except Exception as e:
                    errors.append(e)

            # Run many concurrent get_instance calls
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(get_instance) for _ in range(100)]
                for f in futures:
                    f.result()

            assert len(errors) == 0, f"Errors during concurrent access: {errors}"
            assert len(instances) == 100
            # All instances should be the same
            first = instances[0]
            assert all(inst is first for inst in instances)
            # Should only be initialized once
            assert first.init_count == 1
        finally:
            TestRegistry.reset_instance()

    def test_init_subclass_creates_new_state(self) -> None:
        """Each subclass should get its own _instance and _lock."""

        class Base(SingletonRegistry["Base"]):
            pass

        class Child1(Base):
            pass

        class Child2(Base):
            pass

        # Each class should have separate _instance attributes
        assert Base._instance is None
        assert Child1._instance is None
        assert Child2._instance is None

        # They should also have separate locks
        assert Base._lock is not Child1._lock
        assert Child1._lock is not Child2._lock


class TestItemRegistry:
    """Tests for ItemRegistry base class."""

    def test_register_and_get(self) -> None:
        """Items can be registered and retrieved."""

        class TestRegistry(ItemRegistry["TestRegistry"]):
            pass

        try:
            registry = TestRegistry.get_instance()
            registry.register("foo", {"value": 42})

            result = registry.get("foo")
            assert result == {"value": 42}
        finally:
            TestRegistry.reset_instance()

    def test_unregister(self) -> None:
        """Items can be unregistered."""

        class TestRegistry(ItemRegistry["TestRegistry"]):
            pass

        try:
            registry = TestRegistry.get_instance()
            registry.register("foo", "bar")

            assert registry.unregister("foo") is True
            assert registry.get("foo") is None
            assert registry.unregister("foo") is False
        finally:
            TestRegistry.reset_instance()

    def test_contains(self) -> None:
        """contains() correctly reports item presence."""

        class TestRegistry(ItemRegistry["TestRegistry"]):
            pass

        try:
            registry = TestRegistry.get_instance()

            assert not registry.contains("foo")
            registry.register("foo", "bar")
            assert registry.contains("foo")
        finally:
            TestRegistry.reset_instance()

    def test_list_names(self) -> None:
        """list_names() returns all registered names."""

        class TestRegistry(ItemRegistry["TestRegistry"]):
            pass

        try:
            registry = TestRegistry.get_instance()
            registry.register("a", 1)
            registry.register("b", 2)
            registry.register("c", 3)

            names = registry.list_names()
            assert set(names) == {"a", "b", "c"}
        finally:
            TestRegistry.reset_instance()

    def test_list_items(self) -> None:
        """list_items() returns all registered items."""

        class TestRegistry(ItemRegistry["TestRegistry"]):
            pass

        try:
            registry = TestRegistry.get_instance()
            registry.register("a", 1)
            registry.register("b", 2)

            items = registry.list_items()
            assert set(items) == {1, 2}
        finally:
            TestRegistry.reset_instance()

    def test_count(self) -> None:
        """count() returns correct item count."""

        class TestRegistry(ItemRegistry["TestRegistry"]):
            pass

        try:
            registry = TestRegistry.get_instance()

            assert registry.count() == 0
            registry.register("a", 1)
            assert registry.count() == 1
            registry.register("b", 2)
            assert registry.count() == 2
        finally:
            TestRegistry.reset_instance()

    def test_clear(self) -> None:
        """clear() removes all items and returns count."""

        class TestRegistry(ItemRegistry["TestRegistry"]):
            pass

        try:
            registry = TestRegistry.get_instance()
            registry.register("a", 1)
            registry.register("b", 2)
            registry.register("c", 3)

            cleared = registry.clear()
            assert cleared == 3
            assert registry.count() == 0
        finally:
            TestRegistry.reset_instance()

    def test_thread_safe_operations(self) -> None:
        """Registry operations are thread-safe."""

        class TestRegistry(ItemRegistry["TestRegistry"]):
            pass

        try:
            registry = TestRegistry.get_instance()
            errors = []

            def worker(thread_id: int):
                try:
                    for i in range(50):
                        key = f"thread{thread_id}_item{i}"
                        registry.register(key, i)
                        assert registry.contains(key)
                        assert registry.get(key) == i
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0, f"Errors: {errors}"
            assert registry.count() == 250  # 5 threads x 50 items
        finally:
            TestRegistry.reset_instance()


class TestProgressiveToolsRegistryMigration:
    """Tests that ProgressiveToolsRegistry works correctly after migration."""

    def test_progressive_registry_inherits_singleton(self) -> None:
        """ProgressiveToolsRegistry should inherit from SingletonRegistry."""
        from victor.tools.progressive_registry import ProgressiveToolsRegistry

        try:
            assert issubclass(ProgressiveToolsRegistry, SingletonRegistry)
        finally:
            ProgressiveToolsRegistry.reset_instance()

    def test_progressive_registry_singleton_behavior(self) -> None:
        """ProgressiveToolsRegistry should behave as singleton."""
        from victor.tools.progressive_registry import ProgressiveToolsRegistry

        try:
            r1 = ProgressiveToolsRegistry.get_instance()
            r2 = ProgressiveToolsRegistry.get_instance()
            assert r1 is r2
        finally:
            ProgressiveToolsRegistry.reset_instance()

    def test_progressive_registry_register_and_get(self) -> None:
        """ProgressiveToolsRegistry should register and retrieve configs."""
        from victor.tools.progressive_registry import ProgressiveToolsRegistry

        try:
            registry = ProgressiveToolsRegistry.get_instance()
            registry.register(
                "read",
                progressive_params={"limit": [100, 500, 1000]},
                initial_values={"limit": 100},
            )

            assert registry.is_progressive("read")
            config = registry.get_config("read")
            assert config is not None
            assert config.tool_name == "read"
            assert config.progressive_params == {"limit": [100, 500, 1000]}
        finally:
            ProgressiveToolsRegistry.reset_instance()

    def test_convenience_function_works(self) -> None:
        """get_progressive_registry() convenience function should work."""
        from victor.tools.progressive_registry import (
            ProgressiveToolsRegistry,
            get_progressive_registry,
        )

        try:
            registry = get_progressive_registry()
            assert registry is ProgressiveToolsRegistry.get_instance()
        finally:
            ProgressiveToolsRegistry.reset_instance()


class TestWorkflowCompilerRegistryMigration:
    """Tests that WorkflowCompilerRegistry works correctly after migration."""

    def test_compiler_registry_inherits_item_registry(self) -> None:
        """WorkflowCompilerRegistry should inherit from ItemRegistry."""
        from victor.workflows.compiler_registry import WorkflowCompilerRegistry

        try:
            assert issubclass(WorkflowCompilerRegistry, ItemRegistry)
        finally:
            WorkflowCompilerRegistry.reset_instance()

    def test_compiler_registry_register_and_get(self) -> None:
        """WorkflowCompilerRegistry should register and retrieve compilers."""
        from victor.workflows.compiler_registry import WorkflowCompilerRegistry

        class MockCompiler:
            def __init__(self, **options):
                self.options = options

        try:
            registry = WorkflowCompilerRegistry.get_instance()
            registry.register_compiler("mock", MockCompiler)

            assert registry.is_compiler_registered("mock")
            compiler = registry.get_compiler("mock", enable_caching=True)
            assert isinstance(compiler, MockCompiler)
            assert compiler.options == {"enable_caching": True}
        finally:
            WorkflowCompilerRegistry.reset_instance()

    def test_module_level_functions_work(self) -> None:
        """Module-level convenience functions should work."""
        from victor.workflows.compiler_registry import (
            WorkflowCompilerRegistry,
            get_compiler,
            is_registered,
            list_compilers,
            register_compiler,
        )

        class MockCompiler:
            def __init__(self, **options):
                pass

        try:
            register_compiler("test_mock", MockCompiler)
            assert is_registered("test_mock")
            assert "test_mock" in list_compilers()
            compiler = get_compiler("test_mock")
            assert isinstance(compiler, MockCompiler)
        finally:
            WorkflowCompilerRegistry.reset_instance()

    def test_get_compiler_raises_on_unknown(self) -> None:
        """get_compiler() should raise ValueError for unknown compilers."""
        from victor.workflows.compiler_registry import (
            WorkflowCompilerRegistry,
            get_compiler,
        )

        try:
            WorkflowCompilerRegistry.reset_instance()
            with pytest.raises(ValueError, match="Unknown compiler plugin"):
                get_compiler("nonexistent")
        finally:
            WorkflowCompilerRegistry.reset_instance()
