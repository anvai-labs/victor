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

"""Service Provider Protocols (ISP: Interface Segregation Principle).

This module contains protocols specifically for DI service registration.
Following ISP, these protocols are focused on a single responsibility:
registering vertical-specific services with the DI container.

Usage:
    from victor.core.verticals.protocols.service_provider import (
        ServiceProviderProtocol,
    )

    class CodingServiceProvider(ServiceProviderProtocol):
        def register_services(self, container, settings) -> None:
            container.register(
                CodeValidationProtocol,
                lambda c: CodeValidator(),
            )
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, List, Protocol, Type, runtime_checkable

if TYPE_CHECKING:
    from victor.core.container import ServiceContainer
    from victor.config.settings import Settings


# =============================================================================
# Service Provider Protocol
# =============================================================================


@runtime_checkable
class ServiceProviderProtocol(Protocol):
    """Protocol for registering vertical-specific services with DI container.

    Enables verticals to register their own services alongside
    framework services for consistent lifecycle management.

    Example:
        class CodingServiceProvider(ServiceProviderProtocol):
            def register_services(self, container: ServiceContainer) -> None:
                container.register(
                    CodeCorrectionMiddlewareProtocol,
                    lambda c: CodeCorrectionMiddleware(),
                    ServiceLifetime.SINGLETON,
                )

            def get_required_services(self) -> List[Type]:
                return [CodeCorrectionMiddlewareProtocol]
    """

    @abstractmethod
    def register_services(
        self,
        container: "ServiceContainer",
        settings: "Settings",
    ) -> None:
        """Register vertical-specific services.

        Args:
            container: DI container to register services in
            settings: Application settings
        """
        ...

    def get_required_services(self) -> List[Type]:
        """Get list of required service types.

        Used for validation that all dependencies are registered.

        Returns:
            List of protocol/interface types this vertical requires
        """
        return []

    def get_optional_services(self) -> List[Type]:
        """Get list of optional service types.

        These are used if available but not required.

        Returns:
            List of optional protocol/interface types
        """
        return []


__all__ = [
    "ServiceProviderProtocol",
]
