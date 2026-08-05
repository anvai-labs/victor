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

"""Coding vertical service provider without direct framework barrel imports."""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Type

from victor_contracts.verticals import (
    ModeConfigProviderProtocol,
    PromptContributorProtocol,
    SafetyExtensionProtocol,
    ServiceProviderProtocol,
    ToolDependencyProviderProtocol,
)

logger = logging.getLogger(__name__)


class CodingMiddlewareProtocol:
    """Protocol marker for coding middleware."""

    pass


class CodingServiceProvider(ServiceProviderProtocol):
    """Service provider for the coding vertical."""

    def register_services(self, container: Any, settings: Any) -> None:
        self._register_service(
            container,
            CodingMiddlewareProtocol,
            lambda _container: self._create_middleware(settings),
        )
        self._register_service(
            container,
            SafetyExtensionProtocol,
            lambda _container: self._create_safety(),
        )
        self._register_service(
            container,
            PromptContributorProtocol,
            lambda _container: self._create_prompt_contributor(settings),
        )
        self._register_service(
            container,
            ModeConfigProviderProtocol,
            lambda _container: self._create_mode_config_provider(settings),
        )
        self._register_service(
            container,
            ToolDependencyProviderProtocol,
            lambda _container: self._create_tool_dependency_provider(),
        )
        logger.info("Registered coding vertical services")

    def _register_service(
        self,
        container: Any,
        service_type: Type[Any],
        factory: Callable[[Any], Any],
    ) -> None:
        register_or_replace = getattr(container, "register_or_replace", None)
        if callable(register_or_replace):
            register_or_replace(service_type, factory)
            return

        register = getattr(container, "register", None)
        if callable(register):
            register(service_type, factory)
            return

        register_instance = getattr(container, "register_instance", None)
        if callable(register_instance):
            register_instance(service_type, factory(container))
            return

        raise TypeError(
            "Container does not support register_or_replace, register, or register_instance"
        )

    def _create_middleware(self, settings: Any) -> Any:
        from victor_coding.middleware import CodeCorrectionMiddleware

        enabled = getattr(settings, "enable_code_correction", True)
        auto_fix = getattr(settings, "code_correction_auto_fix", True)
        return CodeCorrectionMiddleware(enabled=enabled, auto_fix=auto_fix)

    def _create_safety(self) -> Any:
        from victor_coding.safety import CodingSafetyExtension

        return CodingSafetyExtension()

    def _create_prompt_contributor(self, settings: Any) -> Any:
        from victor_coding.prompts import CodingPromptContributor

        use_extended = getattr(settings, "use_extended_grounding", False)
        return CodingPromptContributor(use_extended_grounding=use_extended)

    def _create_mode_config_provider(self, settings: Any) -> Any:
        from victor_coding.mode_config import CodingModeConfigProvider

        default_mode = getattr(settings, "default_mode", "default")
        return CodingModeConfigProvider(default_mode=default_mode)

    def _create_tool_dependency_provider(self) -> Any:
        from victor_coding.tool_dependencies import get_provider

        return get_provider()

    def get_required_services(self) -> List[Type[Any]]:
        return []

    def get_optional_services(self) -> List[Type[Any]]:
        return [
            CodingMiddlewareProtocol,
            SafetyExtensionProtocol,
            PromptContributorProtocol,
            ModeConfigProviderProtocol,
            ToolDependencyProviderProtocol,
        ]


__all__ = ["CodingServiceProvider", "CodingMiddlewareProtocol"]
