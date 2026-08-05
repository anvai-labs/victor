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

"""Capability composer for verticals.

Provides a fluent builder API for composing vertical capabilities
declaratively, following the Open/Closed Principle.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

from victor.core.verticals.composition.base_composer import (
    BaseComposer,
    BaseCapabilityProvider,
)

if TYPE_CHECKING:
    from victor.core.verticals.base import VerticalBase
    from victor.core.vertical_types import StageDefinition

logger = logging.getLogger(__name__)


class MetadataCapability(BaseCapabilityProvider):
    """Capability provider for vertical metadata.

    Handles:
    - Vertical name
    - Description
    - Version
    - Provider hints
    """

    def __init__(
        self,
        name: str,
        description: str,
        version: str = "1.0.0",
        provider_hints: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self._config = {
            "name": name,
            "description": description,
            "version": version,
            "provider_hints": provider_hints or {},
        }

    @property
    def name(self) -> str:
        return self._config["name"]

    @property
    def description(self) -> str:
        return self._config["description"]

    @property
    def version(self) -> str:
        return self._config["version"]

    @property
    def provider_hints(self) -> Dict[str, Any]:
        return self._config["provider_hints"]


class StagesCapability(BaseCapabilityProvider):
    """Capability provider for stage definitions.

    Handles:
    - Stage definitions
    - Stage transitions
    - Stage-specific tools
    """

    def __init__(self, stages: Dict[str, "StageDefinition"]):
        super().__init__()
        # Store the actual StageDefinition objects for access via the stages property
        self._stages = stages
        self._config = {
            "stages": {
                name: {
                    "name": stage.name,
                    "description": stage.description,
                    "keywords": stage.keywords,
                    "next_stages": stage.next_stages,
                }
                for name, stage in stages.items()
            }
        }

    @property
    def stages(self) -> Dict[str, "StageDefinition"]:
        return self._stages


class ExtensionsCapability(BaseCapabilityProvider):
    """Capability provider for vertical extensions.

    Handles:
    - Middleware
    - Safety extensions
    - Prompt contributors
    - Mode configs
    """

    def __init__(
        self,
        extensions: List[Any],
        extension_types: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        self._extensions = extensions
        # Use default empty dict if extension_types is None
        ext_types = extension_types or {}
        self._config = {
            "extensions": [
                {
                    "type": ext_types.get(type(ext).__name__, "unknown"),
                    "class": type(ext).__name__,
                }
                for ext in extensions
            ]
        }

    @property
    def extensions(self) -> List[Any]:
        return self._extensions


class CapabilityComposer(BaseComposer):
    """Composer for vertical capability providers.

    Provides a fluent builder API for composing vertical capabilities
    declaratively.

    Example:
        MyVertical = (
            VerticalBase
            .compose()
            .with_metadata("my_vertical", "My assistant", "1.0.0")
            .with_stages(custom_stages)
            .with_extensions([SafetyExtension(), LoggingExtension()])
            .build()
        )
    """

    def with_metadata(
        self,
        name: str,
        description: str,
        version: str = "1.0.0",
        provider_hints: Optional[Dict[str, Any]] = None,
    ) -> "CapabilityComposer":
        """Add metadata capability.

        Args:
            name: Vertical name
            description: Vertical description
            version: Vertical version
            provider_hints: Optional provider configuration hints

        Returns:
            Self for method chaining
        """
        metadata = MetadataCapability(name, description, version, provider_hints)
        self.register_capability("metadata", metadata)
        return self

    def with_stages(
        self,
        stages: Dict[str, StageDefinition],
    ) -> "CapabilityComposer":
        """Add stage definitions.

        Args:
            stages: Dictionary of stage definitions

        Returns:
            Self for method chaining
        """
        stages_cap = StagesCapability(stages)
        self.register_capability("stages", stages_cap)
        return self

    def with_extensions(
        self,
        extensions: List[Any],
        extension_types: Optional[Dict[str, str]] = None,
    ) -> "CapabilityComposer":
        """Add extension capabilities.

        Args:
            extensions: List of extension instances
            extension_types: Optional mapping of extension class names to types

        Returns:
            Self for method chaining
        """
        extensions_cap = ExtensionsCapability(extensions, extension_types)
        self.register_capability("extensions", extensions_cap)
        return self

    def build(self) -> Type["VerticalBase"]:
        """Build a vertical class from composed capabilities.

        Creates a dynamic VerticalBase subclass that incorporates
        all the composed capabilities.

        Returns:
            VerticalBase subclass with composed capabilities
        """
        from victor.core.verticals.base import VerticalBase

        # Validate all capabilities before building
        if not self.validate_all():
            raise ValueError("Invalid capability")

        # Get metadata capability (required)
        metadata = self.get_capability("metadata")
        if not isinstance(metadata, MetadataCapability):
            raise ValueError("Metadata capability is required")

        # Get tools and system prompt from composer (if set)
        tools = getattr(self, "_tools", [])
        system_prompt = getattr(self, "_system_prompt", f"You are {metadata.description}")

        # Create vertical class dynamically
        class ComposedVertical(VerticalBase):
            # Set class attributes from metadata
            name = metadata.name
            description = metadata.description
            version = metadata.version

            # Store composed capabilities
            _composer = None  # Will be set below
            _capability_providers = self._capability_providers

            # Set tools and system prompt as class attributes
            _tools = tools
            _system_prompt = system_prompt

            @classmethod
            def get_tools(cls):
                """Get tools - composed from capability."""
                return cls._tools

            @classmethod
            def get_system_prompt(cls):
                """Get system prompt - composed from capability."""
                return cls._system_prompt

        # Inject stages capability if present
        stages = self.get_capability("stages")
        if isinstance(stages, StagesCapability):
            ComposedVertical.get_stages = classmethod(lambda cls: stages.stages)

        # Inject extensions capability if present
        extensions = self.get_capability("extensions")
        if isinstance(extensions, ExtensionsCapability):
            # Store extensions for later use
            ComposedVertical._composed_extensions = extensions.extensions

        # Store composer instance on the class
        ComposedVertical._composer = self

        self._logger.info(f"Built composed vertical: {metadata.name}")

        return ComposedVertical

    def with_tools(self, tools: List[str]) -> "CapabilityComposer":
        """Set tools for the vertical.

        Args:
            tools: List of tool names

        Returns:
            Self for method chaining
        """
        # Store tools for later use in build()
        self._tools = tools
        return self

    def with_system_prompt(self, prompt: str) -> "CapabilityComposer":
        """Set system prompt for the vertical.

        Args:
            prompt: System prompt text

        Returns:
            Self for method chaining
        """
        # Store prompt for later use in build()
        self._system_prompt = prompt
        return self
