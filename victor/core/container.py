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

"""Dependency Injection Container for Victor.

This module provides a lightweight dependency injection container that:
- Manages service lifecycles (singleton, scoped, transient)
- Supports factory-based and instance-based registration
- Provides scoped containers for request-level isolation
- Enables easy testing through dependency substitution

Design Principles:
- Explicit over implicit (no auto-wiring)
- Type-safe with Protocol support
- Thread-safe singleton management
- Memory-efficient with weak references for scoped services

Example Usage:
    from victor.core.container import ServiceContainer, ServiceLifetime

    # Create and configure container
    container = ServiceContainer()
    container.register(MetricsService, lambda c: MetricsCollector(), ServiceLifetime.SINGLETON)
    container.register(DebugLogger, lambda c: DebugLogger(), ServiceLifetime.SINGLETON)

    # Resolve services
    metrics = container.get(MetricsService)
    logger = container.get(DebugLogger)

    # Create scoped container for request isolation
    with container.create_scope() as scope:
        request_service = scope.get(RequestService)
"""

from __future__ import annotations

import inspect
import logging
import threading
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Optional,
    Protocol,
    Type,
    TypeVar,
    runtime_checkable,
)

from victor.runtime.reachability import record_service_resolution

logger = logging.getLogger(__name__)

T = TypeVar("T")
TService = TypeVar("TService", covariant=True)


class ServiceLifetime(Enum):
    """Defines how long a service instance lives.

    Values:
        SINGLETON: One instance for the entire application lifetime
        SCOPED: One instance per scope (e.g., per request)
        TRANSIENT: New instance every time it's requested
    """

    SINGLETON = "singleton"
    SCOPED = "scoped"
    TRANSIENT = "transient"


@runtime_checkable
class Disposable(Protocol):
    """Protocol for services that need cleanup."""

    def dispose(self) -> None:
        """Release resources held by the service."""
        ...


@runtime_checkable
class AsyncDisposable(Protocol):
    """Protocol for services that need async cleanup."""

    async def adispose(self) -> None:
        """Asynchronously release resources."""
        ...


@dataclass
class ServiceDescriptor(Generic[T]):
    """Describes how to create and manage a service."""

    service_type: Type[T]
    factory: Callable[["ServiceContainer"], T]
    lifetime: ServiceLifetime
    instance: Optional[T] = None

    def create_instance(self, container: "ServiceContainer") -> T:
        """Create a new instance using the factory."""
        return self.factory(container)


class ServiceScope:
    """Scoped container for request-level service isolation.

    Services registered with SCOPED lifetime get one instance per scope.
    When the scope is disposed, all scoped services are cleaned up.
    """

    def __init__(self, parent: "ServiceContainer"):
        """Initialize scope with parent container.

        Args:
            parent: Parent container to inherit singleton services from
        """
        self._parent = parent
        self._scoped_instances: Dict[Type, Any] = {}
        self._disposed = False
        self._lock = threading.Lock()

    def get(self, service_type: Type[T]) -> T:
        """Get a service instance within this scope.

        Args:
            service_type: Type of service to retrieve

        Returns:
            Service instance

        Raises:
            ServiceNotFoundError: If service is not registered
            ScopeDisposedError: If scope has been disposed
        """
        if self._disposed:
            raise ScopeDisposedError(f"Scope has been disposed, cannot resolve {service_type}")

        descriptor = self._parent._get_descriptor(service_type)

        if descriptor.lifetime == ServiceLifetime.SINGLETON:
            # Singletons come from parent
            return self._parent.get(service_type)

        if descriptor.lifetime == ServiceLifetime.SCOPED:
            with self._lock:
                if service_type not in self._scoped_instances:
                    self._scoped_instances[service_type] = descriptor.create_instance(self._parent)
                return self._scoped_instances[service_type]

        # Transient - always create new
        return descriptor.create_instance(self._parent)

    def dispose(self) -> None:
        """Dispose all scoped services."""
        with self._lock:
            if self._disposed:
                return
            self._disposed = True
            for service in self._scoped_instances.values():
                if isinstance(service, Disposable):
                    try:
                        service.dispose()
                    except Exception as e:
                        logger.warning(f"Error disposing scoped service: {e}")
            self._scoped_instances.clear()

    async def adispose(self) -> None:
        """Async dispose all scoped services."""
        with self._lock:
            if self._disposed:
                return
            self._disposed = True
            for service in self._scoped_instances.values():
                if isinstance(service, AsyncDisposable):
                    try:
                        await service.adispose()
                    except Exception as e:
                        logger.warning(f"Error async-disposing scoped: {e}")
                elif isinstance(service, Disposable):
                    try:
                        service.dispose()
                    except Exception as e:
                        logger.warning(f"Error disposing scoped service: {e}")
            self._scoped_instances.clear()

    def __enter__(self) -> "ServiceScope":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.dispose()

    async def __aenter__(self) -> "ServiceScope":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.adispose()


class ServiceNotFoundError(Exception):
    """Raised when a requested service is not registered."""

    def __init__(self, service_type: Type):
        self.service_type = service_type
        # Handle both Type objects and string service names
        name = service_type.__name__ if hasattr(service_type, "__name__") else str(service_type)
        super().__init__(f"Service not registered: {name}")


class ServiceResolutionError(Exception):
    """Raised when a service cannot be resolved (e.g., circular dependency)."""

    pass


class ServiceAlreadyRegisteredError(Exception):
    """Raised when trying to register a service that already exists."""

    def __init__(self, service_type: Type):
        self.service_type = service_type
        super().__init__(f"Service already registered: {service_type.__name__}")


class ScopeDisposedError(Exception):
    """Raised when trying to use a disposed scope."""

    pass


class ContainerFrozenError(Exception):
    """Raised when trying to modify a frozen container."""

    pass


def _resolve_constructor(cls: type, container: "ServiceContainer"):
    """Auto-resolve constructor params from container annotations.

    For each parameter with a type annotation, attempt to resolve it
    from the container. Falls back to default value if resolution fails
    and a default exists. Skips ``self``, ``*args``, ``**kwargs``.
    """
    sig = inspect.signature(cls.__init__)
    kwargs = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if param.annotation is not inspect.Parameter.empty:
            try:
                kwargs[name] = container.get(param.annotation)
            except (ServiceNotFoundError, ServiceResolutionError):
                if param.default is inspect.Parameter.empty:
                    raise
        # No annotation and no default → cls() will naturally fail
    return cls(**kwargs)


class ServiceContainer:
    """Central dependency injection container.

    Manages service registration, resolution, and lifecycle. Supports
    singleton, scoped, and transient service lifetimes.

    Thread-safe for concurrent access to singleton services.

    Example:
        container = ServiceContainer()

        # Register services
        container.register(ILogger, lambda c: FileLogger(), ServiceLifetime.SINGLETON)
        container.register(ICache, lambda c: RedisCache(c.get(ILogger)), ServiceLifetime.SINGLETON)

        # Resolve services
        logger = container.get(ILogger)
        cache = container.get(ICache)

        # Create scoped container
        with container.create_scope() as scope:
            request_handler = scope.get(IRequestHandler)
    """

    def __init__(self) -> None:
        """Initialize empty container."""
        self._descriptors: Dict[Type, ServiceDescriptor] = {}
        self._lock = threading.RLock()
        self._disposed = False
        self._frozen = False
        self._resolving = threading.local()  # Per-thread cycle detection

    def _get_resolving(self) -> set:
        """Get the per-thread resolution tracking set."""
        if not hasattr(self._resolving, "stack"):
            self._resolving.stack = set()
        return self._resolving.stack

    def freeze(self) -> "ServiceContainer":
        """Freeze container to prevent further registrations.

        Call after bootstrap completes. Resolution still works.

        Returns:
            Self for method chaining
        """
        with self._lock:
            self._frozen = True
        return self

    def _check_frozen(self) -> None:
        """Raise if container is frozen."""
        if self._frozen:
            raise ContainerFrozenError("Container is frozen; registrations are not allowed")

    def register(
        self,
        service_type: Type[T],
        factory: Callable[["ServiceContainer"], T],
        lifetime: ServiceLifetime = ServiceLifetime.SINGLETON,
    ) -> "ServiceContainer":
        """Register a service with a factory function.

        Args:
            service_type: Type/interface to register
            factory: Factory function that creates the service instance
            lifetime: How long the service instance lives

        Returns:
            Self for method chaining

        Raises:
            ServiceAlreadyRegisteredError: If service type already registered
            ContainerFrozenError: If container is frozen
        """
        self._check_frozen()
        with self._lock:
            if service_type in self._descriptors:
                raise ServiceAlreadyRegisteredError(service_type)

            self._descriptors[service_type] = ServiceDescriptor(
                service_type=service_type,
                factory=factory,
                lifetime=lifetime,
            )
            logger.debug(f"Registered {service_type.__name__} with {lifetime.value} lifetime")

        return self

    def register_instance(self, service_type: Type[T], instance: T) -> "ServiceContainer":
        """Register an existing instance as a singleton.

        Args:
            service_type: Type/interface to register
            instance: Pre-created service instance

        Returns:
            Self for method chaining
        """
        self._check_frozen()
        with self._lock:
            if service_type in self._descriptors:
                raise ServiceAlreadyRegisteredError(service_type)

            descriptor = ServiceDescriptor(
                service_type=service_type,
                factory=lambda c: instance,  # type: ignore
                lifetime=ServiceLifetime.SINGLETON,
                instance=instance,
            )
            self._descriptors[service_type] = descriptor
            logger.debug(f"Registered {service_type.__name__} instance as singleton")

        return self

    def register_or_replace(
        self,
        service_type: Type[T],
        factory: Callable[["ServiceContainer"], T],
        lifetime: ServiceLifetime = ServiceLifetime.SINGLETON,
    ) -> "ServiceContainer":
        """Register a service, replacing existing registration if present.

        Useful for testing where you want to substitute implementations.

        Args:
            service_type: Type/interface to register
            factory: Factory function that creates the service instance
            lifetime: How long the service instance lives

        Returns:
            Self for method chaining
        """
        self._check_frozen()
        with self._lock:
            # Dispose existing singleton if present
            if service_type in self._descriptors:
                existing = self._descriptors[service_type]
                if existing.instance is not None and isinstance(existing.instance, Disposable):
                    try:
                        existing.instance.dispose()
                    except Exception as e:
                        logger.warning(f"Error disposing replaced service: {e}")

            self._descriptors[service_type] = ServiceDescriptor(
                service_type=service_type,
                factory=factory,
                lifetime=lifetime,
            )
            logger.debug(f"Replaced {service_type.__name__} with {lifetime.value} lifetime")

        return self

    def get(self, service_type: Type[T]) -> T:
        """Get a service instance.

        Args:
            service_type: Type of service to retrieve

        Returns:
            Service instance

        Raises:
            ServiceNotFoundError: If service is not registered
            ServiceResolutionError: If circular dependency detected
        """
        descriptor = self._get_descriptor(service_type)

        # FEP-0022 Probe A: witness that this registered type was resolved. Inert
        # when no recorder is armed (the call is a contextvar read + None check).
        record_service_resolution(service_type)

        # Circular dependency detection (per-thread)
        resolving = self._get_resolving()
        if service_type in resolving:
            chain = " -> ".join(t.__name__ for t in resolving)
            raise ServiceResolutionError(
                f"Circular dependency detected: " f"{chain} -> {service_type.__name__}"
            )

        if descriptor.lifetime == ServiceLifetime.TRANSIENT:
            resolving.add(service_type)
            try:
                return descriptor.create_instance(self)
            finally:
                resolving.discard(service_type)

        # Singleton or Scoped (scoped in root acts as singleton)
        with self._lock:
            if descriptor.instance is None:
                resolving.add(service_type)
                try:
                    descriptor.instance = descriptor.create_instance(self)
                finally:
                    resolving.discard(service_type)
            return descriptor.instance

    def get_optional(self, service_type: Type[T]) -> Optional[T]:
        """Get a service instance, or None if not registered.

        Args:
            service_type: Type of service to retrieve

        Returns:
            Service instance or None
        """
        try:
            return self.get(service_type)
        except ServiceNotFoundError:
            return None

    def is_registered(self, service_type: Type) -> bool:
        """Check if a service type is registered.

        Args:
            service_type: Type to check

        Returns:
            True if registered, False otherwise
        """
        return service_type in self._descriptors

    def _get_descriptor(self, service_type: Type[T]) -> ServiceDescriptor[T]:
        """Get the descriptor for a service type.

        Args:
            service_type: Type to look up

        Returns:
            Service descriptor

        Raises:
            ServiceNotFoundError: If not registered
        """
        descriptor = self._descriptors.get(service_type)
        if descriptor is None:
            raise ServiceNotFoundError(service_type)
        return descriptor

    def create_scope(self) -> ServiceScope:
        """Create a new service scope.

        Scoped services will have one instance per scope.

        Returns:
            New service scope
        """
        return ServiceScope(self)

    def get_registered_types(self) -> list[Type]:
        """Get list of all registered service types.

        Returns:
            List of registered types
        """
        return list(self._descriptors.keys())

    def service(
        self,
        service_type: Type[T],
        lifetime: ServiceLifetime = ServiceLifetime.SINGLETON,
    ):
        """Decorator for registering services.

        Provides a decorator-based alternative to register() for cleaner
        service registration syntax. The decorated class is automatically
        registered with the container.

        Example:
            container = ServiceContainer()

            @container.service(IService, ServiceLifetime.SINGLETON)
            class MyService(IService):
                def __init__(self):
                    self.value = 42

                def get_value(self):
                    return self.value

            # Service is now registered and can be resolved
            service = container.get(IService)

        Args:
            service_type: Type/interface to register the service as
            lifetime: How long the service instance lives (default: SINGLETON)

        Returns:
            Decorator function that registers the class

        Raises:
            ServiceAlreadyRegisteredError: If service type already registered
        """

        def decorator(cls: Type[T]) -> Type[T]:
            self._check_frozen()
            with self._lock:
                if service_type in self._descriptors:
                    raise ServiceAlreadyRegisteredError(service_type)

                def factory(c: ServiceContainer) -> T:
                    return _resolve_constructor(cls, c)

                self._descriptors[service_type] = ServiceDescriptor(
                    service_type=service_type,
                    factory=factory,
                    lifetime=lifetime,
                )
                logger.debug(
                    f"Registered {cls.__name__} as {service_type.__name__} "
                    f"with {lifetime.value} lifetime (via decorator)"
                )

            return cls

        return decorator

    def check_health(self, service_type: Type) -> bool:
        """Check if a service is healthy.

        A service is considered healthy if:
        1. It is registered in the container
        2. It can be instantiated (or already has an instance)
        3. If it has an is_healthy() method, that method returns True

        This is useful for health checks and monitoring in production systems.

        Args:
            service_type: Type of service to check

        Returns:
            True if the service is healthy, False otherwise

        Example:
            container = ServiceContainer()
            container.register(IMetrics, MetricsService)

            if container.check_health(IMetrics):
                logger.info("Metrics service is healthy")
            else:
                logger.error("Metrics service is unhealthy")
        """
        try:
            instance = self.get(service_type)

            # Check for is_healthy method if defined
            if hasattr(instance, "is_healthy") and callable(instance.is_healthy):
                try:
                    return bool(instance.is_healthy())
                except Exception as e:
                    logger.warning(f"Health check for {service_type.__name__} failed: {e}")
                    return False

            # No is_healthy method, consider service healthy if it exists
            return True

        except Exception as e:
            logger.debug(f"Health check for {service_type.__name__} failed: {e}")
            return False

    def check_all_health(self) -> Dict[Type, bool]:
        """Check health of all registered services.

        Returns:
            Dictionary mapping service types to their health status

        Example:
            container = ServiceContainer()
            # ... register services ...

            health_status = container.check_all_health()
            for service_type, is_healthy in health_status.items():
                status = "✓" if is_healthy else "✗"
                print(f"{status} {service_type.__name__}")
        """
        return {
            service_type: self.check_health(service_type)
            for service_type in self.get_registered_types()
        }

    def validate(self) -> list:
        """Eagerly validate all registrations by trial resolution.

        Attempts to create each service to catch circular dependencies
        and missing registrations early. Does not cache singletons —
        the trial runs through ``_try_resolve`` which discards results.

        Returns:
            List of exceptions encountered (empty = healthy).
        """
        errors: list = []
        for service_type in list(self._descriptors):
            try:
                self._try_resolve(service_type)
            except Exception as e:
                errors.append(e)
        return errors

    def _try_resolve(self, service_type: Type[T]) -> None:
        """Resolve a service without caching the singleton instance."""
        descriptor = self._get_descriptor(service_type)
        resolving = self._get_resolving()

        if service_type in resolving:
            chain = " -> ".join(t.__name__ for t in resolving)
            raise ServiceResolutionError(
                f"Circular dependency detected: " f"{chain} -> {service_type.__name__}"
            )

        resolving.add(service_type)
        try:
            descriptor.create_instance(self)
        finally:
            resolving.discard(service_type)

    def dispose(self) -> None:
        """Dispose all singleton services and clear registrations."""
        with self._lock:
            if self._disposed:
                return
            self._disposed = True
            for descriptor in self._descriptors.values():
                if descriptor.instance is not None and isinstance(descriptor.instance, Disposable):
                    try:
                        descriptor.instance.dispose()
                    except Exception as e:
                        logger.warning(f"Error disposing service: {e}")
            self._descriptors.clear()

    async def adispose(self) -> None:
        """Async dispose: awaits AsyncDisposable, calls sync Disposable."""
        with self._lock:
            if self._disposed:
                return
            self._disposed = True
            for descriptor in self._descriptors.values():
                inst = descriptor.instance
                if inst is None:
                    continue
                if isinstance(inst, AsyncDisposable):
                    try:
                        await inst.adispose()
                    except Exception as e:
                        logger.warning(f"Error async-disposing service: {e}")
                elif isinstance(inst, Disposable):
                    try:
                        inst.dispose()
                    except Exception as e:
                        logger.warning(f"Error disposing service: {e}")
            self._descriptors.clear()

    def __enter__(self) -> "ServiceContainer":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.dispose()

    async def __aenter__(self) -> "ServiceContainer":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.adispose()


# =============================================================================
# Global Container Management
# =============================================================================

_global_container: Optional[ServiceContainer] = None
_global_lock = threading.Lock()


def get_container() -> ServiceContainer:
    """Get the global service container.

    Creates a new container if one doesn't exist.

    Returns:
        Global service container
    """
    global _global_container
    with _global_lock:
        if _global_container is None:
            _global_container = ServiceContainer()
        return _global_container


def set_container(container: ServiceContainer) -> None:
    """Set the global service container.

    Args:
        container: Container to use as global
    """
    global _global_container
    with _global_lock:
        if _global_container is not None:
            _global_container.dispose()
        _global_container = container


def reset_container() -> None:
    """Reset the global container (for testing).

    Disposes existing container and clears the global reference.
    """
    global _global_container
    with _global_lock:
        if _global_container is not None:
            _global_container.dispose()
            _global_container = None


# =============================================================================
# Service Protocol Definitions
# =============================================================================


@runtime_checkable
class MetricsServiceProtocol(Protocol):
    """Protocol for metrics collection services."""

    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value."""
        ...

    def increment_counter(self, name: str, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        ...


@runtime_checkable
class LoggerServiceProtocol(Protocol):
    """Protocol for debug/analytics logging services."""

    def log(self, message: str, level: str = "info", **kwargs: Any) -> None:
        """Log a message."""
        ...

    @property
    def enabled(self) -> bool:
        """Whether logging is enabled."""
        ...


@runtime_checkable
class CacheServiceProtocol(Protocol):
    """Protocol for caching services."""

    def get(self, key: str) -> Optional[Any]:
        """Get a cached value."""
        ...

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a cached value."""
        ...

    def invalidate(self, key: str) -> None:
        """Invalidate a cached value."""
        ...


@runtime_checkable
class EmbeddingServiceProtocol(Protocol):
    """Protocol for embedding services."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts."""
        ...

    def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        ...
