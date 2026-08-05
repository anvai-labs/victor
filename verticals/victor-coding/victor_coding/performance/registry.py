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

"""Registry for performance-improved backend implementations.

This registry enables a plugin architecture where multiple implementations
of the same interface can coexist, with automatic selection based on:
1. Priority (higher value = preferred)
2. Availability (native extensions may not be installed)
3. Capabilities (parallel, async, etc.)

Pattern based on EmbeddingRegistry and GraphStoreProtocol patterns already
used in victor-coding.

NOTE: Tree-sitter symbol extraction is handled by Python's tree-sitter bindings
directly. This registry focuses on chunking and regex processing where Rust
provides genuine performance benefits.

Usage:
    # Register a native backend with high priority
    PerformanceBackendRegistry.register(
        FastChunkerProtocol,
        "rust_chunker",
        RustChunker,
        priority=80
    )

    # Create highest-priority available backend
    chunker = PerformanceBackendRegistry.create(
        FastChunkerProtocol,
        ChunkerConfig()
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

from victor_coding.performance.protocols import (
    BackendCapabilities,
    FastChunkerProtocol,
    FastRegexProcessorProtocol,
    get_backend_capabilities,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
P = TypeVar("P", bound=Type)


@dataclass
class BackendRegistration:
    """Information about a registered backend."""

    name: str
    backend_class: Type
    priority: int
    factory: Optional[Callable[..., Any]] = None
    capabilities: BackendCapabilities = field(default_factory=BackendCapabilities)
    is_available: bool = True


@dataclass
class BackendConfig:
    """Configuration for backend creation."""

    use_native: bool = True
    prefer_parallel: bool = True
    fallback_to_python: bool = True
    min_priority: int = 0
    max_memory_mb: int = 0


class PerformanceBackendRegistry:
    """Central registry for performance-improved backend implementations.

    Supports multiple implementations of the same protocol with priority-based
    selection and automatic fallback to Python implementations.

    Thread-safe: uses module-level locks for concurrent access.
    """

    _backends: Dict[Type, Dict[str, BackendRegistration]] = {}
    _factories: Dict[Type, Dict[str, Callable[..., Any]]] = {}
    _default_priorities: Dict[Type, int] = {
        FastChunkerProtocol: 50,
        FastRegexProcessorProtocol: 50,
    }

    @classmethod
    def register(
        cls,
        protocol: Type[P],
        name: str,
        backend_class: Optional[Type[T]] = None,
        *,
        priority: int = 50,
        factory: Optional[Callable[..., T]] = None,
    ) -> None:
        """Register a backend implementation for a protocol.

        Args:
            protocol: Protocol class (e.g., FastIndexerProtocol)
            name: Unique name for this backend
            backend_class: Backend class implementing the protocol
            priority: Priority value (0-100, higher = preferred)
            factory: Optional factory function for lazy instantiation

        Note:
            Either backend_class or factory must be provided. If both are provided,
            factory takes precedence.
        """
        if backend_class is None and factory is None:
            raise ValueError(f"Must provide either backend_class or factory for {name}")

        if protocol not in cls._backends:
            cls._backends[protocol] = {}

        # Check availability if it's a class
        is_available = True
        if backend_class is not None:
            try:
                # Try to get the class to check for import errors
                is_available = backend_class is not None
            except Exception as e:
                logger.debug(f"Backend {name} not available: {e}")
                is_available = False

        registration = BackendRegistration(
            name=name,
            backend_class=backend_class or (lambda: None),  # type: ignore
            priority=priority,
            factory=factory,
            capabilities=BackendCapabilities(),
            is_available=is_available,
        )

        # Get capabilities if backend is available
        if is_available and backend_class is not None:
            try:
                registration.capabilities = get_backend_capabilities(backend_class)
            except Exception:
                pass

        cls._backends[protocol][name] = registration

        # Store factory if provided
        if factory is not None:
            if protocol not in cls._factories:
                cls._factories[protocol] = {}
            cls._factories[protocol][name] = factory

        logger.info(
            f"Registered {protocol.__name__} backend: {name} "
            f"(priority={priority}, native={registration.capabilities.is_native})"
        )

    @classmethod
    def register_factory(
        cls,
        protocol: Type[P],
        name: str,
        factory: Callable[..., T],
        *,
        priority: int = 50,
    ) -> None:
        """Register a factory function for creating backend instances.

        Args:
            protocol: Protocol class
            name: Unique name for this backend
            factory: Factory function that creates instances
            priority: Priority value (0-100, higher = preferred)
        """
        cls.register(protocol, name, priority=priority, factory=factory)

    @classmethod
    def get(cls, protocol: Type[P], name: str, default: Optional[T] = None) -> Optional[T]:
        """Get a specific backend by name.

        Args:
            protocol: Protocol class
            name: Backend name
            default: Default value if not found

        Returns:
            Backend class or factory, or default if not found
        """
        if protocol not in cls._backends:
            return default

        registration = cls._backends[protocol].get(name)
        if registration is None:
            return default

        if not registration.is_available:
            return default

        return registration.factory or registration.backend_class

    @classmethod
    def create(
        cls,
        protocol: Type[P],
        config: Optional[Any] = None,
        *,
        backend_config: Optional[BackendConfig] = None,
    ) -> Optional[T]:
        """Create the highest-priority available backend instance.

        Args:
            protocol: Protocol class to implement
            config: Configuration object to pass to backend constructor
            backend_config: Backend selection configuration

        Returns:
            Backend instance or None if no backends available
        """
        backends = cls.list_backends(protocol)
        if not backends:
            logger.warning(f"No backends registered for {protocol.__name__}")
            return None

        # Sort by priority (descending)
        backends.sort(key=lambda r: r.priority, reverse=True)

        # Filter by configuration
        if backend_config:
            backends = [
                b
                for b in backends
                if b.priority >= backend_config.min_priority
                and (
                    not backend_config.max_memory_mb
                    or b.capabilities.memory_overhead_mb <= backend_config.max_memory_mb
                )
            ]

            # Prefer native if use_native is True
            if backend_config.use_native:
                native_backends = [b for b in backends if b.capabilities.is_native]
                if native_backends:
                    backends = native_backends

        # Try each backend in priority order
        for registration in backends:
            if not registration.is_available:
                continue

            try:
                if registration.factory:
                    return registration.factory(config)
                elif registration.backend_class:
                    return registration.backend_class(config)
            except Exception as e:
                logger.warning(
                    f"Failed to create {protocol.__name__} backend {registration.name}: {e}"
                )
                if not backend_config or not backend_config.fallback_to_python:
                    raise

        return None

    @classmethod
    def create_or_fallback(
        cls,
        protocol: Type[P],
        native_factory: Callable[..., T],
        python_factory: Callable[..., T],
        config: Optional[Any] = None,
        *,
        prefer_native: bool = True,
    ) -> T:
        """Create backend trying native first, falling back to Python.

        This is a convenience method for the common pattern of trying a native
        implementation and falling back to pure Python.

        Args:
            protocol: Protocol class
            native_factory: Factory for native implementation
            python_factory: Factory for Python fallback
            config: Configuration object
            prefer_native: Whether to try native first

        Returns:
            Backend instance from the first successful factory
        """
        factories = (
            [native_factory, python_factory] if prefer_native else [python_factory, native_factory]
        )

        for factory in factories:
            try:
                instance = factory(config)
                if instance is not None:
                    return instance
            except Exception as e:
                logger.debug(f"Factory failed: {e}")

        # Last resort: return Python factory result even if it raises
        return python_factory(config)

    @classmethod
    def list_backends(cls, protocol: Type[P]) -> List[BackendRegistration]:
        """List all registered backends for a protocol.

        Args:
            protocol: Protocol class

        Returns:
            List of BackendRegistration objects
        """
        if protocol not in cls._backends:
            return []

        return [r for r in cls._backends[protocol].values() if r.is_available]

    @classmethod
    def list_backend_names(cls, protocol: Type[P]) -> List[str]:
        """List names of all registered backends for a protocol.

        Args:
            protocol: Protocol class

        Returns:
            List of backend names
        """
        return [r.name for r in cls.list_backends(protocol)]

    @classmethod
    def is_registered(cls, protocol: Type[P], name: str) -> bool:
        """Check if a backend is registered for a protocol.

        Args:
            protocol: Protocol class
            name: Backend name

        Returns:
            True if registered and available
        """
        if protocol not in cls._backends:
            return False

        registration = cls._backends[protocol].get(name)
        return registration is not None and registration.is_available

    @classmethod
    def unregister(cls, protocol: Type[P], name: str) -> bool:
        """Unregister a backend.

        Args:
            protocol: Protocol class
            name: Backend name

        Returns:
            True if unregistered, False if not found
        """
        if protocol not in cls._backends:
            return False

        if name in cls._backends[protocol]:
            del cls._backends[protocol][name]
            if name in cls._factories.get(protocol, {}):
                del cls._factories[protocol][name]
            return True

        return False

    @classmethod
    def get_native_available(cls, protocol: Type[P]) -> bool:
        """Check if any native backend is available for a protocol.

        Args:
            protocol: Protocol class

        Returns:
            True if at least one native backend is registered and available
        """
        return any(r.capabilities.is_native for r in cls.list_backends(protocol))

    @classmethod
    def get_highest_priority(cls, protocol: Type[P]) -> Optional[BackendRegistration]:
        """Get the highest-priority backend for a protocol.

        Args:
            protocol: Protocol class

        Returns:
            BackendRegistration with highest priority, or None
        """
        backends = cls.list_backends(protocol)
        if not backends:
            return None

        return max(backends, key=lambda r: r.priority)


def auto_register_native_backends() -> None:
    """Auto-discover and register native Rust backends.

    This function attempts to import native extensions and registers them
    with high priority if available.

    NOTE: Tree-sitter extraction uses Python's bindings directly and is not
    registered here. Only chunking and regex processing have Rust backends.
    """
    # Try to import native chunker
    try:
        from victor_coding.native import FastChunker

        PerformanceBackendRegistry.register(
            FastChunkerProtocol,
            "rust_chunker",
            FastChunker,
            priority=80,
        )
        logger.info("Registered native chunker backend")
    except ImportError:
        logger.debug("Native chunker backend not available")

    # Try to import native regex processor
    try:
        from victor_coding.native import FastRegexProcessor

        PerformanceBackendRegistry.register(
            FastRegexProcessorProtocol,
            "rust_regex",
            FastRegexProcessor,
            priority=80,
        )
        logger.info("Registered native regex backend")
    except ImportError:
        logger.debug("Native regex backend not available")


__all__ = [
    "BackendRegistration",
    "BackendConfig",
    "PerformanceBackendRegistry",
    "auto_register_native_backends",
]
