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

"""Tests for enhanced ServiceContainer functionality.

Tests for decorator-based registration and health check features
added in Phase 1 of the SOLID refactoring.
"""

import pytest
from typing import Protocol, runtime_checkable

from victor.core.container import (
    ServiceContainer,
    ServiceLifetime,
    ServiceAlreadyRegisteredError,
)

# =============================================================================
# Test Protocols and Classes
# =============================================================================


class ITestService:
    """Protocol for test service."""

    def get_value(self) -> int:
        """Get a test value."""
        ...


class IHealthyService:
    """Protocol for health checkable service."""

    def is_healthy(self) -> bool:
        """Check if service is healthy."""
        ...


class BasicService(ITestService):
    """Basic service implementation."""

    def __init__(self):
        self.value = 42

    def get_value(self) -> int:
        return self.value


class ServiceWithDeps(ITestService):
    """Service that depends on other services."""

    def __init__(self, multiplier: int = 2):
        self.multiplier = multiplier

    def get_value(self) -> int:
        return 21 * self.multiplier


class HealthyService(IHealthyService):
    """Service with health check."""

    def __init__(self, healthy: bool = True):
        self._healthy = healthy

    def is_healthy(self) -> bool:
        return self._healthy

    def set_health(self, healthy: bool) -> None:
        self._healthy = healthy


class UnhealthyService(IHealthyService):
    """Service that fails health checks."""

    def is_healthy(self) -> bool:
        return False


class ServiceWithFailingHealthCheck(IHealthyService):
    """Service whose health check raises an exception."""

    def is_healthy(self) -> bool:
        raise RuntimeError("Health check failed!")


class DisposableService(ITestService):
    """Service that implements Disposable protocol."""

    def __init__(self):
        self.disposed = False
        self.value = 100

    def get_value(self) -> int:
        if self.disposed:
            raise RuntimeError("Service is disposed")
        return self.value

    def dispose(self) -> None:
        self.disposed = True


# =============================================================================
# Decorator Registration Tests
# =============================================================================


class TestDecoratorRegistration:
    """Tests for decorator-based service registration."""

    def test_register_service_with_decorator(self):
        """Test registering a service using the decorator."""
        container = ServiceContainer()

        @container.service(ITestService, ServiceLifetime.SINGLETON)
        class MyService(ITestService):
            def __init__(self):
                self.value = 42

            def get_value(self) -> int:
                return self.value

        # Service should be registered
        assert container.is_registered(ITestService)

        # Service should be resolvable
        service = container.get(ITestService)
        assert isinstance(service, MyService)
        assert service.get_value() == 42

    def test_decorator_returns_class(self):
        """Test that decorator returns the original class."""
        container = ServiceContainer()

        @container.service(ITestService, ServiceLifetime.SINGLETON)
        class MyService(ITestService):
            def __init__(self):
                self.value = 42

            def get_value(self) -> int:
                return self.value

        # Decorator should return the class
        assert MyService is not None
        assert MyService.__name__ == "MyService"

    def test_decorator_with_singleton_lifetime(self):
        """Test decorator with SINGLETON lifetime."""
        container = ServiceContainer()

        @container.service(ITestService, ServiceLifetime.SINGLETON)
        class MyService(ITestService):
            def __init__(self):
                self.value = id(self)  # Use object identity

            def get_value(self) -> int:
                return self.value

        # Same instance should be returned
        service1 = container.get(ITestService)
        service2 = container.get(ITestService)
        assert service1 is service2
        assert service1.get_value() == service2.get_value()

    def test_decorator_with_transient_lifetime(self):
        """Test decorator with TRANSIENT lifetime."""
        container = ServiceContainer()

        @container.service(ITestService, ServiceLifetime.TRANSIENT)
        class MyService(ITestService):
            def __init__(self):
                self.value = id(self)  # Use object identity

            def get_value(self) -> int:
                return self.value

        # Different instances should be returned
        service1 = container.get(ITestService)
        service2 = container.get(ITestService)
        assert service1 is not service2
        assert service1.get_value() != service2.get_value()

    def test_decorator_duplicate_registration_raises(self):
        """Test that decorator raises on duplicate registration."""
        container = ServiceContainer()

        @container.service(ITestService, ServiceLifetime.SINGLETON)
        class MyService(ITestService):
            def get_value(self) -> int:
                return 42

        with pytest.raises(ServiceAlreadyRegisteredError):

            @container.service(ITestService, ServiceLifetime.SINGLETON)
            class AnotherService(ITestService):
                def get_value(self) -> int:
                    return 100

    def test_decorator_with_multiple_registrations(self):
        """Test registering multiple services with decorators."""
        container = ServiceContainer()

        @container.service(ITestService, ServiceLifetime.SINGLETON)
        class Service1(ITestService):
            def get_value(self) -> int:
                return 1

        class IAnotherService:
            def get_name(self) -> str: ...

        @container.service(IAnotherService, ServiceLifetime.SINGLETON)
        class Service2(IAnotherService):
            def get_name(self) -> str:
                return "service2"

        # Both services should be registered
        assert container.is_registered(ITestService)
        assert container.is_registered(IAnotherService)

        # Both should be resolvable
        service1 = container.get(ITestService)
        service2 = container.get(IAnotherService)
        assert service1.get_value() == 1
        assert service2.get_name() == "service2"


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthChecks:
    """Tests for service health check functionality."""

    def test_check_health_unregistered_service(self):
        """Test health check for unregistered service."""
        container = ServiceContainer()

        assert not container.check_health(ITestService)

    def test_check_health_service_without_health_method(self):
        """Test health check for service without is_healthy method."""
        container = ServiceContainer()
        container.register(ITestService, lambda c: BasicService())

        # Service without is_healthy is considered healthy
        assert container.check_health(ITestService)

    def test_check_health_service_with_healthy_method(self):
        """Test health check for service with is_healthy method."""
        container = ServiceContainer()
        container.register(IHealthyService, lambda c: HealthyService(healthy=True))

        assert container.check_health(IHealthyService)

    def test_check_health_service_with_unhealthy_method(self):
        """Test health check for unhealthy service."""
        container = ServiceContainer()
        container.register(IHealthyService, lambda c: UnhealthyService())

        assert not container.check_health(IHealthyService)

    def test_check_health_service_with_failing_health_check(self):
        """Test health check when is_healthy raises an exception."""
        container = ServiceContainer()
        container.register(IHealthyService, lambda c: ServiceWithFailingHealthCheck())

        # Failing health check should return False
        assert not container.check_health(IHealthyService)

    def test_check_health_singleton_instance(self):
        """Test health check doesn't create new instances for singletons."""
        container = ServiceContainer()

        call_count = [0]

        class CountingHealthyService(IHealthyService):
            def __init__(self):
                call_count[0] += 1

            def is_healthy(self) -> bool:
                return True

        container.register(
            IHealthyService,
            lambda c: CountingHealthyService(),
            ServiceLifetime.SINGLETON,
        )

        # First call creates instance
        assert container.check_health(IHealthyService)
        assert call_count[0] == 1

        # Second call uses cached instance
        assert container.check_health(IHealthyService)
        assert call_count[0] == 1

    def test_check_all_health(self):
        """Test checking health of all registered services."""
        container = ServiceContainer()

        # Register multiple services
        container.register(ITestService, lambda c: BasicService())
        container.register(IHealthyService, lambda c: HealthyService(healthy=True))

        health_status = container.check_all_health()

        assert len(health_status) == 2
        assert health_status[ITestService] is True
        assert health_status[IHealthyService] is True

    def test_check_all_health_with_mixed_status(self):
        """Test checking health when services have different statuses."""
        container = ServiceContainer()

        # Register services with different health statuses
        container.register(ITestService, lambda c: BasicService())
        container.register(IHealthyService, lambda c: UnhealthyService())

        health_status = container.check_all_health()

        # Basic service is healthy (no is_healthy method)
        assert health_status[ITestService] is True
        # UnhealthyService reports as unhealthy
        assert health_status[IHealthyService] is False

    def test_check_health_with_decorator_registration(self):
        """Test health check with decorator-registered service."""
        container = ServiceContainer()

        @container.service(IHealthyService, ServiceLifetime.SINGLETON)
        class MyHealthyService(IHealthyService):
            def __init__(self):
                self._healthy = True

            def is_healthy(self) -> bool:
                return self._healthy

        assert container.check_health(IHealthyService)

    def test_check_health_unregistered_type(self):
        """Test health check for type that's not registered."""
        container = ServiceContainer()

        class IUnregisteredService:
            pass

        assert not container.check_health(IUnregisteredService)


# =============================================================================
# Integration Tests
# =============================================================================


class TestDecoratorAndHealthChecksIntegration:
    """Integration tests for decorator registration and health checks."""

    def test_decorator_registration_with_health_checks(self):
        """Test that decorator-registered services support health checks."""
        container = ServiceContainer()

        @container.service(IHealthyService, ServiceLifetime.SINGLETON)
        class MyService(IHealthyService):
            def __init__(self):
                self._healthy = True

            def is_healthy(self) -> bool:
                return self._healthy

        # Should be registered
        assert container.is_registered(IHealthyService)

        # Should pass health check
        assert container.check_health(IHealthyService)

        # Should be resolvable
        service = container.get(IHealthyService)
        assert isinstance(service, MyService)

    def test_multiple_services_health_summary(self):
        """Test getting health summary for multiple decorator-registered services."""
        container = ServiceContainer()

        @container.service(ITestService, ServiceLifetime.SINGLETON)
        class Service1(ITestService):
            def get_value(self) -> int:
                return 1

        @container.service(IHealthyService, ServiceLifetime.SINGLETON)
        class Service2(IHealthyService):
            def is_healthy(self) -> bool:
                return True

        # Create a different service type for Service3
        @runtime_checkable
        class IUnhealthyService(Protocol):
            def is_healthy(self) -> bool: ...

        @container.service(IUnhealthyService, ServiceLifetime.SINGLETON)
        class Service3(IUnhealthyService):
            def is_healthy(self) -> bool:
                return False

        health_status = container.check_all_health()

        # Service1 has no health check method -> healthy
        assert health_status.get(ITestService) is True

        # Service2 is healthy
        assert health_status.get(IHealthyService) is True

        # Service3 is unhealthy
        assert health_status.get(IUnhealthyService) is False

    def test_decorator_with_lifetime_and_health(self):
        """Test different lifetimes with health checks."""
        container = ServiceContainer()

        instance_count = [0]

        @container.service(IHealthyService, ServiceLifetime.TRANSIENT)
        class TransientService(IHealthyService):
            def __init__(self):
                instance_count[0] += 1

            def is_healthy(self) -> bool:
                return True

        # Each health check creates new instance for TRANSIENT
        assert container.check_health(IHealthyService)
        assert instance_count[0] == 1

        assert container.check_health(IHealthyService)
        assert instance_count[0] == 2


class TestThreadSafety:
    """Tests for thread safety of new container features."""

    def test_concurrent_decorator_registration(self):
        """Test concurrent decorator registration."""
        import threading

        container = ServiceContainer()
        errors = []

        def register_service(index: int):
            try:

                class ILocalService:
                    pass

                @container.service(ILocalService, ServiceLifetime.SINGLETON)
                class LocalService(ILocalService):
                    pass

            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_service, args=(i,)) for i in range(10)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Should not have any errors
        assert len(errors) == 0

    def test_concurrent_health_checks(self):
        """Test concurrent health checks."""
        import threading

        container = ServiceContainer()

        @container.service(IHealthyService, ServiceLifetime.SINGLETON)
        class MyService(IHealthyService):
            def __init__(self):
                self._healthy = True

            def is_healthy(self) -> bool:
                return self._healthy

        results = []

        def check_health():
            results.append(container.check_health(IHealthyService))

        threads = [threading.Thread(target=check_health) for _ in range(10)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # All checks should succeed
        assert all(results)
        assert len(results) == 10
