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

"""Resource lifecycle management for clean shutdown.

This module provides centralized management of shared resources to prevent
memory leaks and ensure clean shutdown.

Issue Reference: workflow-test-issues-v2.md Issue #5
"""

from __future__ import annotations

import atexit
import logging
import threading
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class ResourceState(Enum):
    """State of a managed resource."""

    ACTIVE = "active"
    CLOSING = "closing"
    CLOSED = "closed"
    FAILED = "failed"


@dataclass
class ManagedResource:
    """Represents a managed resource."""

    name: str
    resource_ref: weakref.ref
    cleanup_method: str
    priority: int = 0  # Higher priority = cleanup first
    state: ResourceState = ResourceState.ACTIVE
    cleanup_error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def cleanup(self) -> bool:
        """Execute cleanup for this resource.

        Returns:
            True if cleanup succeeded
        """
        resource = self.resource_ref()
        if resource is None:
            self.state = ResourceState.CLOSED
            return True

        self.state = ResourceState.CLOSING
        try:
            method = getattr(resource, self.cleanup_method, None)
            if method and callable(method):
                method()
            self.state = ResourceState.CLOSED
            return True
        except Exception as e:
            self.state = ResourceState.FAILED
            self.cleanup_error = str(e)
            logger.warning(f"Failed to cleanup {self.name}: {e}")
            return False


@runtime_checkable
class IResourceManager(Protocol):
    """Protocol for resource management."""

    def register_cleanup(self, callback: Callable) -> None:
        """Register a cleanup callback."""
        ...

    def register_resource(self, resource: Any, name: str, cleanup_method: str) -> None:
        """Register a resource for cleanup."""
        ...

    def cleanup_all(self) -> Dict[str, bool]:
        """Run all cleanup callbacks."""
        ...


class ResourceManager:
    """Manages lifecycle of shared resources.

    Features:
    - Singleton pattern for global resource tracking
    - Priority-based cleanup ordering
    - Weak references to avoid preventing garbage collection
    - Thread-safe operations
    - Automatic cleanup on process exit

    Usage:
        manager = get_resource_manager()

        # Register a resource
        manager.register_resource(db_connection, "database", "close")

        # Register a callback
        manager.register_cleanup(lambda: print("Cleanup!"))

        # Manual cleanup (or automatic on exit)
        manager.cleanup_all()
    """

    _instance: Optional["ResourceManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ResourceManager":
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._init()
                    cls._instance = instance
        return cls._instance

    def _init(self) -> None:
        """Initialize the resource manager."""
        self._cleanup_callbacks: List[Callable] = []
        self._resources: Dict[str, ManagedResource] = {}
        self._callback_lock = threading.Lock()
        self._resource_lock = threading.Lock()
        self._cleanup_done = False

        # Register for automatic cleanup on exit
        atexit.register(self._atexit_cleanup)
        logger.debug("ResourceManager initialized")

    def register_cleanup(self, callback: Callable, priority: int = 0) -> None:
        """Register a cleanup callback to run on shutdown.

        Args:
            callback: Function to call during cleanup
            priority: Higher priority callbacks run first
        """
        with self._callback_lock:
            self._cleanup_callbacks.append((priority, callback))
            # Sort by priority (descending)
            self._cleanup_callbacks.sort(key=lambda x: -x[0])
        logger.debug(f"Registered cleanup callback (priority: {priority})")

    def register_resource(
        self,
        resource: Any,
        name: str,
        cleanup_method: str = "close",
        priority: int = 0,
        **metadata,
    ) -> None:
        """Register a resource for cleanup.

        Args:
            resource: The resource to manage
            name: Unique name for the resource
            cleanup_method: Method to call for cleanup
            priority: Cleanup priority (higher = first)
            **metadata: Additional metadata
        """
        with self._resource_lock:
            managed = ManagedResource(
                name=name,
                resource_ref=weakref.ref(resource),
                cleanup_method=cleanup_method,
                priority=priority,
                metadata=metadata,
            )
            self._resources[name] = managed
        logger.debug(f"Registered resource: {name} (method: {cleanup_method})")

    def unregister_resource(self, name: str) -> bool:
        """Unregister a resource.

        Args:
            name: Name of resource to unregister

        Returns:
            True if resource was found and removed
        """
        with self._resource_lock:
            if name in self._resources:
                del self._resources[name]
                logger.debug(f"Unregistered resource: {name}")
                return True
        return False

    def cleanup_resource(self, name: str) -> bool:
        """Cleanup a specific resource.

        Args:
            name: Name of resource to cleanup

        Returns:
            True if cleanup succeeded
        """
        with self._resource_lock:
            if name not in self._resources:
                return False

            managed = self._resources[name]
            success = managed.cleanup()

            if success:
                del self._resources[name]

            return success

    def cleanup_all(self) -> Dict[str, bool]:
        """Run all cleanup callbacks and resource cleanups.

        Returns:
            Dictionary mapping resource/callback names to success status
        """
        if self._cleanup_done:
            logger.debug("Cleanup already completed, skipping")
            return {}

        results: Dict[str, bool] = {}

        # Cleanup resources first (sorted by priority)
        with self._resource_lock:
            sorted_resources = sorted(
                self._resources.items(),
                key=lambda x: -x[1].priority,
            )

            for name, managed in sorted_resources:
                try:
                    success = managed.cleanup()
                    results[f"resource:{name}"] = success
                except Exception as e:
                    logger.warning(f"Error cleaning up resource {name}: {e}")
                    results[f"resource:{name}"] = False

            self._resources.clear()

        # Run callbacks
        with self._callback_lock:
            for i, (priority, callback) in enumerate(self._cleanup_callbacks):
                callback_name = f"callback:{i}"
                try:
                    callback()
                    results[callback_name] = True
                except Exception as e:
                    logger.warning(f"Error in cleanup callback {i}: {e}")
                    results[callback_name] = False

            self._cleanup_callbacks.clear()

        self._cleanup_done = True
        success_count = sum(1 for v in results.values() if v)
        logger.info(f"Cleanup complete: {success_count}/{len(results)} succeeded")

        return results

    def _cleanup_all_silent(self) -> None:
        """Silent cleanup for atexit - no logging to avoid closed file errors."""
        results: Dict[str, bool] = {}

        # Clean up resources
        with self._resource_lock:
            sorted_resources = sorted(
                self._resources.items(),
                key=lambda x: -x[1].priority,
            )

            for name, managed in sorted_resources:
                try:
                    success = managed.cleanup()
                    results[f"resource:{name}"] = success
                except Exception:
                    results[f"resource:{name}"] = False

            self._resources.clear()

        # Run callbacks
        with self._callback_lock:
            for i, (priority, callback) in enumerate(self._cleanup_callbacks):
                callback_name = f"callback:{i}"
                try:
                    callback()
                    results[callback_name] = True
                except Exception:
                    results[callback_name] = False

            self._cleanup_callbacks.clear()

        self._cleanup_done = True

    def _atexit_cleanup(self) -> None:
        """Cleanup handler for process exit.

        Note: During interpreter shutdown, logging may fail with
        'I/O operation on closed file' - we silently ignore these.
        """
        if not self._cleanup_done:
            try:
                logger.debug("Running atexit cleanup")
            except (ValueError, OSError):
                pass  # Logging system already shut down
            self._cleanup_all_silent()

    def get_resource_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all registered resources.

        Returns:
            Dictionary with resource status information
        """
        with self._resource_lock:
            return {
                name: {
                    "state": managed.state.value,
                    "cleanup_method": managed.cleanup_method,
                    "priority": managed.priority,
                    "alive": managed.resource_ref() is not None,
                    "metadata": managed.metadata,
                }
                for name, managed in self._resources.items()
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics.

        Returns:
            Statistics dictionary
        """
        with self._resource_lock:
            with self._callback_lock:
                return {
                    "resources_registered": len(self._resources),
                    "callbacks_registered": len(self._cleanup_callbacks),
                    "cleanup_done": self._cleanup_done,
                    "active_resources": sum(
                        1 for m in self._resources.values() if m.state == ResourceState.ACTIVE
                    ),
                }

    @contextmanager
    def managed_resource(
        self,
        resource: Any,
        name: str,
        cleanup_method: str = "close",
    ):
        """Context manager for automatic resource cleanup.

        Args:
            resource: Resource to manage
            name: Resource name
            cleanup_method: Cleanup method name

        Yields:
            The managed resource
        """
        self.register_resource(resource, name, cleanup_method)
        try:
            yield resource
        finally:
            self.cleanup_resource(name)

    def reset(self) -> None:
        """Reset the manager state (for testing).

        Warning: This will clear all registrations without cleanup.
        """
        with self._resource_lock:
            with self._callback_lock:
                self._resources.clear()
                self._cleanup_callbacks.clear()
                self._cleanup_done = False
        logger.debug("ResourceManager reset")


def get_resource_manager() -> ResourceManager:
    """Get the singleton resource manager.

    Returns:
        ResourceManager singleton instance
    """
    return ResourceManager()


def register_for_cleanup(
    resource: Any,
    name: str,
    cleanup_method: str = "close",
    priority: int = 0,
) -> None:
    """Convenience function to register a resource for cleanup.

    Args:
        resource: Resource to register
        name: Resource name
        cleanup_method: Cleanup method name
        priority: Cleanup priority
    """
    get_resource_manager().register_resource(resource, name, cleanup_method, priority)


@contextmanager
def managed(resource: Any, name: str, cleanup_method: str = "close"):
    """Convenience context manager for resource management.

    Args:
        resource: Resource to manage
        name: Resource name
        cleanup_method: Cleanup method name

    Yields:
        The managed resource
    """
    manager = get_resource_manager()
    with manager.managed_resource(resource, name, cleanup_method):
        yield resource
