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

"""Service provider for performance plugin integration with Victor SDK.

This module provides the PerformanceServiceProvider which integrates with the
SDK's service_provider extension point to register performance-improved backends.

The service provider is registered via victor-vertical.toml's service_provider
extension point or through VerticalExtensions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from victor_contracts.verticals.protocols import ServiceProvider

from victor_coding.performance.protocols import (
    FastChunkerProtocol,
    FastRegexProcessorProtocol,
)
from victor_coding.performance.registry import (
    PerformanceBackendRegistry,
    auto_register_native_backends,
)

logger = logging.getLogger(__name__)


class PerformanceServiceProvider(ServiceProvider):
    """Service provider for performance-improved backends.

    This provider registers native (Rust) backends with high priority when
    available, allowing transparent fallback to Python implementations.

    Usage in victor-vertical.toml:
        [vertical.extensions]
        service_provider = "victor_coding.performance.service_provider:PerformanceServiceProvider"
    """

    def __init__(self) -> None:
        """Initialize the performance service provider."""
        self._backends_registered: bool = False
        self._native_available: bool = False

    def register_services(self, container: Any, settings: Any) -> None:
        """Register performance backend services with the container.

        Args:
            container: Service container (typically a dict-like object)
            settings: Victor settings object
        """
        # Check if native backends should be used
        use_native = getattr(settings, "use_native_backends", True)

        if not use_native:
            logger.info("Native backends disabled by settings")
            return

        # Auto-register native backends
        try:
            auto_register_native_backends()
            self._backends_registered = True

            # Check availability
            self._native_available = self._check_native_availability()

            if self._native_available:
                logger.info("Performance backends registered (native available)")
            else:
                logger.info("Performance backends registered (Python fallback)")

        except Exception as e:
            logger.warning(f"Failed to register performance backends: {e}")

    def _check_native_availability(self) -> bool:
        """Check if any native backends are available.

        Returns:
            True if at least one native backend is available
        """
        return PerformanceBackendRegistry.get_native_available(
            FastChunkerProtocol
        ) or PerformanceBackendRegistry.get_native_available(FastRegexProcessorProtocol)

    def get_native_available(self) -> bool:
        """Check if native backends are available.

        Returns:
            True if at least one native backend is registered and available
        """
        return self._native_available

    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about registered backends.

        Returns:
            Dictionary with backend availability information
        """
        return {
            "backends_registered": self._backends_registered,
            "native_available": self._native_available,
            "has_native_chunker": PerformanceBackendRegistry.get_native_available(
                FastChunkerProtocol
            ),
            "has_native_regex": PerformanceBackendRegistry.get_native_available(
                FastRegexProcessorProtocol
            ),
            "note": "Tree-sitter extraction uses Python bindings directly",
        }

    def list_backends(self, protocol: Optional[type] = None) -> List[str]:
        """List registered backends for a protocol.

        Args:
            protocol: Optional protocol class to filter by

        Returns:
            List of backend names
        """
        if protocol is None:
            # Return all backend names from all protocols
            all_backends = set()
            for proto in [
                FastChunkerProtocol,
                FastRegexProcessorProtocol,
            ]:
                all_backends.update(PerformanceBackendRegistry.list_backend_names(proto))
            return sorted(all_backends)

        return PerformanceBackendRegistry.list_backend_names(protocol)

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on performance backends.

        Returns:
            Health check results
        """
        return {
            "healthy": True,
            "service": "performance",
            "backends_registered": self._backends_registered,
            "backend_info": self.get_backend_info(),
        }


# Singleton instance for easy import
_performance_service_provider: Optional[PerformanceServiceProvider] = None


def get_performance_service_provider() -> PerformanceServiceProvider:
    """Get the singleton performance service provider instance.

    Returns:
        PerformanceServiceProvider instance
    """
    global _performance_service_provider

    if _performance_service_provider is None:
        _performance_service_provider = PerformanceServiceProvider()

    return _performance_service_provider


__all__ = [
    "PerformanceServiceProvider",
    "get_performance_service_provider",
]
