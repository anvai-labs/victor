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

"""Tests for lazy initialization utilities."""

import pytest

from victor.core.lazy import (
    LazyProperty,
    deferred_import,
    SingletonFactory,
    CircularImportInfo,
    KNOWN_CIRCULAR_IMPORTS,
    get_circular_import_info,
    list_circular_imports,
)


class TestLazyProperty:
    """Tests for LazyProperty descriptor."""

    def test_lazy_property_initializes_on_first_access(self):
        """Property should initialize on first access."""
        call_count = 0

        class TestClass:
            @LazyProperty
            def expensive(self) -> int:
                nonlocal call_count
                call_count += 1
                return 42

        obj = TestClass()
        assert call_count == 0  # Not initialized yet

        result = obj.expensive
        assert result == 42
        assert call_count == 1

    def test_lazy_property_caches_value(self):
        """Property should cache value after first access."""
        call_count = 0

        class TestClass:
            @LazyProperty
            def expensive(self) -> int:
                nonlocal call_count
                call_count += 1
                return 42

        obj = TestClass()
        _ = obj.expensive
        _ = obj.expensive
        _ = obj.expensive

        assert call_count == 1  # Only called once

    def test_lazy_property_per_instance(self):
        """Each instance should have its own cached value."""

        class TestClass:
            def __init__(self, value: int):
                self._value = value

            @LazyProperty
            def computed(self) -> int:
                return self._value * 2

        obj1 = TestClass(5)
        obj2 = TestClass(10)

        assert obj1.computed == 10
        assert obj2.computed == 20

    def test_lazy_property_with_complex_type(self):
        """Property should work with complex return types."""

        class HeavyDependency:
            def __init__(self):
                self.data = {"initialized": True}

        class TestClass:
            @LazyProperty
            def dependency(self) -> HeavyDependency:
                return HeavyDependency()

        obj = TestClass()
        result = obj.dependency

        assert isinstance(result, HeavyDependency)
        assert result.data["initialized"] is True

    def test_lazy_property_class_access_returns_descriptor(self):
        """Accessing on class should return the descriptor."""

        class TestClass:
            @LazyProperty
            def expensive(self) -> int:
                return 42

        assert isinstance(TestClass.expensive, LazyProperty)


class TestDeferredImport:
    """Tests for deferred_import function."""

    def test_deferred_import_existing_module(self):
        """Should import existing modules successfully."""
        result = deferred_import("json", "JSONEncoder")
        assert result is not None
        import json

        assert isinstance(result, json.JSONEncoder)

    def test_deferred_import_with_call_method(self):
        """Should call factory method when specified."""
        # Use a known module with a class method
        result = deferred_import(
            "pathlib",
            "Path",
            call_method="cwd",
        )
        assert result is not None
        from pathlib import Path

        assert result == Path.cwd()

    def test_deferred_import_nonexistent_module_returns_fallback(self):
        """Should return fallback for nonexistent modules."""
        result = deferred_import(
            "nonexistent_module_xyz",
            "SomeClass",
            fallback="fallback_value",
        )
        assert result == "fallback_value"

    def test_deferred_import_nonexistent_class_returns_fallback(self):
        """Should return fallback for nonexistent class."""
        result = deferred_import(
            "json",
            "NonexistentClass",
            fallback=None,
        )
        assert result is None

    def test_deferred_import_with_init_kwargs(self):
        """Should pass kwargs to constructor."""
        from pathlib import PurePosixPath

        result = deferred_import(
            "pathlib",
            "PurePosixPath",
            init_args=("/test/path",),
        )
        assert result is not None
        assert str(result) == "/test/path"


class TestSingletonFactory:
    """Tests for SingletonFactory."""

    def setup_method(self):
        """Clear singletons before each test."""
        SingletonFactory.clear()

    def teardown_method(self):
        """Clear singletons after each test."""
        SingletonFactory.clear()

    def test_get_or_create_creates_instance(self):
        """Should create instance on first call."""

        class TestService:
            pass

        instance = SingletonFactory.get_or_create(TestService)
        assert isinstance(instance, TestService)

    def test_get_or_create_returns_same_instance(self):
        """Should return same instance on subsequent calls."""

        class TestService:
            pass

        instance1 = SingletonFactory.get_or_create(TestService)
        instance2 = SingletonFactory.get_or_create(TestService)

        assert instance1 is instance2

    def test_get_or_create_with_factory(self):
        """Should use custom factory function."""

        class TestService:
            def __init__(self, value: int):
                self.value = value

        instance = SingletonFactory.get_or_create(
            TestService,
            factory=lambda: TestService(42),
        )

        assert instance.value == 42

    def test_has_instance(self):
        """Should correctly report instance existence."""

        class TestService:
            pass

        assert not SingletonFactory.has_instance(TestService)

        SingletonFactory.get_or_create(TestService)

        assert SingletonFactory.has_instance(TestService)

    def test_clear_specific_type(self):
        """Should clear specific type."""

        class Service1:
            pass

        class Service2:
            pass

        SingletonFactory.get_or_create(Service1)
        SingletonFactory.get_or_create(Service2)

        SingletonFactory.clear(Service1)

        assert not SingletonFactory.has_instance(Service1)
        assert SingletonFactory.has_instance(Service2)

    def test_clear_all(self):
        """Should clear all instances."""

        class Service1:
            pass

        class Service2:
            pass

        SingletonFactory.get_or_create(Service1)
        SingletonFactory.get_or_create(Service2)

        SingletonFactory.clear()

        assert not SingletonFactory.has_instance(Service1)
        assert not SingletonFactory.has_instance(Service2)

    def test_set_instance(self):
        """Should allow setting instance directly."""

        class TestService:
            def __init__(self, value: str):
                self.value = value

        mock_instance = TestService("test_value")
        SingletonFactory.set_instance(TestService, mock_instance)

        retrieved = SingletonFactory.get_or_create(TestService)
        assert retrieved is mock_instance
        assert retrieved.value == "test_value"


class TestCircularImportInfo:
    """Tests for CircularImportInfo and registry."""

    def test_circular_import_info_dataclass(self):
        """Should create CircularImportInfo correctly."""
        info = CircularImportInfo(
            module="test.py",
            chain=["a", "b", "c"],
            reason="Test reason",
            solution_file="test.py",
            solution_line=10,
            solution_type="deferred_import",
        )

        assert info.module == "test.py"
        assert info.chain == ["a", "b", "c"]
        assert info.fixed is True  # Default

    def test_known_circular_imports_not_empty(self):
        """Registry should contain known chains."""
        assert len(KNOWN_CIRCULAR_IMPORTS) > 0

    def test_known_circular_imports_has_orchestrator(self):
        """Should document orchestrator circular import."""
        assert "orchestrator_evaluation" in KNOWN_CIRCULAR_IMPORTS

        info = KNOWN_CIRCULAR_IMPORTS["orchestrator_evaluation"]
        assert "orchestrator" in info.chain
        assert info.fixed is True

    def test_get_circular_import_info_existing(self):
        """Should return info for existing key."""
        info = get_circular_import_info("orchestrator_evaluation")
        assert info is not None
        assert info.module == "orchestrator.py"

    def test_get_circular_import_info_nonexistent(self):
        """Should return None for nonexistent key."""
        info = get_circular_import_info("nonexistent_chain")
        assert info is None

    def test_list_circular_imports(self):
        """Should list all documented chain keys."""
        keys = list_circular_imports()
        assert isinstance(keys, list)
        assert len(keys) == len(KNOWN_CIRCULAR_IMPORTS)
        assert "orchestrator_evaluation" in keys


class TestIntegration:
    """Integration tests combining multiple utilities."""

    def setup_method(self):
        """Clear singletons before each test."""
        SingletonFactory.clear()

    def teardown_method(self):
        """Clear singletons after each test."""
        SingletonFactory.clear()

    def test_lazy_property_with_deferred_import(self):
        """LazyProperty should work with deferred imports."""

        class TestClass:
            @LazyProperty
            def encoder(self):
                return deferred_import("json", "JSONEncoder", fallback=None)

        obj = TestClass()
        encoder = obj.encoder

        assert encoder is not None
        import json

        assert isinstance(encoder, json.JSONEncoder)

    def test_singleton_factory_with_lazy_initialization(self):
        """SingletonFactory should work with complex initialization."""

        class ComplexService:
            def __init__(self):
                self.dependencies = []
                self.initialized = True

            def add_dependency(self, dep):
                self.dependencies.append(dep)

        # First access creates instance
        service = SingletonFactory.get_or_create(ComplexService)
        service.add_dependency("dep1")

        # Second access returns same instance with modifications
        service2 = SingletonFactory.get_or_create(ComplexService)
        assert service2 is service
        assert "dep1" in service2.dependencies
