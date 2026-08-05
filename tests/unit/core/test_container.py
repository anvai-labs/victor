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

"""Tests for the dependency injection container."""

import pytest
import threading
from typing import Protocol

from victor.core.container import (
    ServiceContainer,
    ServiceLifetime,
    ServiceNotFoundError,
    ServiceAlreadyRegisteredError,
    ScopeDisposedError,
    get_container,
    set_container,
    reset_container,
)


class ILogger(Protocol):
    """Test logger interface."""

    def log(self, message: str) -> None: ...


class ICache(Protocol):
    """Test cache interface."""

    def get(self, key: str) -> str: ...


class MockLogger:
    """Mock logger implementation."""

    def __init__(self):
        self.messages: list[str] = []

    def log(self, message: str) -> None:
        self.messages.append(message)


class MockCache:
    """Mock cache implementation."""

    def __init__(self, logger: ILogger):
        self.logger = logger
        self._data: dict[str, str] = {}

    def get(self, key: str) -> str:
        self.logger.log(f"Cache get: {key}")
        return self._data.get(key, "")


class DisposableService:
    """Service that tracks disposal."""

    disposed = False

    def dispose(self) -> None:
        DisposableService.disposed = True


class TestServiceContainer:
    """Tests for ServiceContainer class."""

    def setup_method(self):
        """Reset state before each test."""
        DisposableService.disposed = False
        reset_container()

    def test_register_and_get_singleton(self):
        """Test registering and retrieving a singleton service."""
        container = ServiceContainer()
        container.register(MockLogger, lambda c: MockLogger(), ServiceLifetime.SINGLETON)

        logger1 = container.get(MockLogger)
        logger2 = container.get(MockLogger)

        assert logger1 is logger2
        assert isinstance(logger1, MockLogger)

    def test_register_and_get_transient(self):
        """Test registering and retrieving a transient service."""
        container = ServiceContainer()
        container.register(MockLogger, lambda c: MockLogger(), ServiceLifetime.TRANSIENT)

        logger1 = container.get(MockLogger)
        logger2 = container.get(MockLogger)

        assert logger1 is not logger2
        assert isinstance(logger1, MockLogger)

    def test_register_instance(self):
        """Test registering a pre-created instance."""
        container = ServiceContainer()
        instance = MockLogger()
        instance.messages.append("pre-existing")

        container.register_instance(MockLogger, instance)
        retrieved = container.get(MockLogger)

        assert retrieved is instance
        assert "pre-existing" in retrieved.messages

    def test_dependency_injection(self):
        """Test injecting dependencies between services."""
        container = ServiceContainer()
        container.register(MockLogger, lambda c: MockLogger(), ServiceLifetime.SINGLETON)
        container.register(
            MockCache, lambda c: MockCache(c.get(MockLogger)), ServiceLifetime.SINGLETON
        )

        cache = container.get(MockCache)
        cache.get("test_key")

        logger = container.get(MockLogger)
        assert "Cache get: test_key" in logger.messages

    def test_service_not_found_error(self):
        """Test that ServiceNotFoundError is raised for unregistered services."""
        container = ServiceContainer()

        with pytest.raises(ServiceNotFoundError) as exc_info:
            container.get(MockLogger)

        assert "MockLogger" in str(exc_info.value)

    def test_service_already_registered_error(self):
        """Test that ServiceAlreadyRegisteredError is raised for duplicate registration."""
        container = ServiceContainer()
        container.register(MockLogger, lambda c: MockLogger())

        with pytest.raises(ServiceAlreadyRegisteredError) as exc_info:
            container.register(MockLogger, lambda c: MockLogger())

        assert "MockLogger" in str(exc_info.value)

    def test_register_or_replace(self):
        """Test replacing an existing registration."""
        container = ServiceContainer()
        container.register(MockLogger, lambda c: MockLogger())

        logger1 = container.get(MockLogger)
        logger1.messages.append("original")

        # Replace with new factory
        container.register_or_replace(MockLogger, lambda c: MockLogger())
        logger2 = container.get(MockLogger)

        # Should be a new instance
        assert "original" not in logger2.messages

    def test_get_optional_returns_none(self):
        """Test get_optional returns None for unregistered services."""
        container = ServiceContainer()

        result = container.get_optional(MockLogger)

        assert result is None

    def test_get_optional_returns_service(self):
        """Test get_optional returns service when registered."""
        container = ServiceContainer()
        container.register(MockLogger, lambda c: MockLogger())

        result = container.get_optional(MockLogger)

        assert isinstance(result, MockLogger)

    def test_is_registered(self):
        """Test is_registered check."""
        container = ServiceContainer()

        assert not container.is_registered(MockLogger)

        container.register(MockLogger, lambda c: MockLogger())

        assert container.is_registered(MockLogger)

    def test_get_registered_types(self):
        """Test getting list of registered types."""
        container = ServiceContainer()
        container.register(MockLogger, lambda c: MockLogger())
        container.register(MockCache, lambda c: MockCache(c.get(MockLogger)))

        types = container.get_registered_types()

        assert MockLogger in types
        assert MockCache in types
        assert len(types) == 2

    def test_dispose_calls_disposable(self):
        """Test that dispose() calls dispose on Disposable services."""
        container = ServiceContainer()
        container.register(DisposableService, lambda c: DisposableService())

        # Create the instance
        container.get(DisposableService)
        assert not DisposableService.disposed

        container.dispose()
        assert DisposableService.disposed

    def test_context_manager(self):
        """Test container as context manager."""
        with ServiceContainer() as container:
            container.register(DisposableService, lambda c: DisposableService())
            container.get(DisposableService)

        assert DisposableService.disposed

    def test_thread_safety_singleton(self):
        """Test that singleton resolution is thread-safe."""
        container = ServiceContainer()
        container.register(MockLogger, lambda c: MockLogger(), ServiceLifetime.SINGLETON)

        results = []
        errors = []

        def get_logger():
            try:
                logger = container.get(MockLogger)
                results.append(logger)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_logger) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10
        # All should be the same instance
        assert all(r is results[0] for r in results)


class TestServiceScope:
    """Tests for ServiceScope class."""

    def setup_method(self):
        """Reset state before each test."""
        DisposableService.disposed = False

    def test_scoped_lifetime(self):
        """Test that scoped services have one instance per scope."""
        container = ServiceContainer()
        container.register(MockLogger, lambda c: MockLogger(), ServiceLifetime.SCOPED)

        with container.create_scope() as scope1:
            logger1a = scope1.get(MockLogger)
            logger1b = scope1.get(MockLogger)
            assert logger1a is logger1b

        with container.create_scope() as scope2:
            logger2 = scope2.get(MockLogger)
            assert logger2 is not logger1a

    def test_scoped_inherits_singletons(self):
        """Test that scoped containers inherit singleton instances."""
        container = ServiceContainer()
        container.register(MockLogger, lambda c: MockLogger(), ServiceLifetime.SINGLETON)

        singleton_logger = container.get(MockLogger)

        with container.create_scope() as scope:
            scoped_logger = scope.get(MockLogger)
            assert scoped_logger is singleton_logger

    def test_scope_disposes_services(self):
        """Test that scope disposal cleans up scoped services."""
        container = ServiceContainer()
        container.register(DisposableService, lambda c: DisposableService(), ServiceLifetime.SCOPED)

        with container.create_scope() as scope:
            scope.get(DisposableService)
            assert not DisposableService.disposed

        assert DisposableService.disposed

    def test_disposed_scope_raises_error(self):
        """Test that using disposed scope raises ScopeDisposedError."""
        container = ServiceContainer()
        container.register(MockLogger, lambda c: MockLogger(), ServiceLifetime.SCOPED)

        scope = container.create_scope()
        scope.dispose()

        with pytest.raises(ScopeDisposedError):
            scope.get(MockLogger)


class TestGlobalContainer:
    """Tests for global container management."""

    def setup_method(self):
        """Reset global container before each test."""
        reset_container()

    def teardown_method(self):
        """Reset global container after each test."""
        reset_container()

    def test_get_container_creates_default(self):
        """Test that get_container creates a container if none exists."""
        container = get_container()

        assert container is not None
        assert isinstance(container, ServiceContainer)

    def test_get_container_returns_same_instance(self):
        """Test that get_container returns the same instance."""
        container1 = get_container()
        container2 = get_container()

        assert container1 is container2

    def test_set_container_replaces_global(self):
        """Test that set_container replaces the global container."""
        old_container = get_container()
        old_container.register(MockLogger, lambda c: MockLogger())

        new_container = ServiceContainer()
        set_container(new_container)

        current = get_container()
        assert current is new_container
        assert not current.is_registered(MockLogger)

    def test_reset_container_clears_global(self):
        """Test that reset_container clears the global container."""
        container = get_container()
        container.register(MockLogger, lambda c: MockLogger())

        reset_container()

        new_container = get_container()
        assert not new_container.is_registered(MockLogger)


class TestMethodChaining:
    """Tests for method chaining support."""

    def test_register_returns_self(self):
        """Test that register returns container for chaining."""
        container = ServiceContainer()

        result = container.register(MockLogger, lambda c: MockLogger())

        assert result is container

    def test_register_instance_returns_self(self):
        """Test that register_instance returns container for chaining."""
        container = ServiceContainer()

        result = container.register_instance(MockLogger, MockLogger())

        assert result is container

    def test_fluent_registration(self):
        """Test fluent registration pattern."""
        container = (
            ServiceContainer()
            .register(MockLogger, lambda c: MockLogger())
            .register(MockCache, lambda c: MockCache(c.get(MockLogger)))
        )

        assert container.is_registered(MockLogger)
        assert container.is_registered(MockCache)
