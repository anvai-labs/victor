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

"""Tool-specific serialization configuration loader.

Loads tool serialization preferences from model_capabilities.yaml and provides
tool-aware configuration for the adaptive serializer.

Hierarchical resolution:
1. defaults
2. categories (if tool is in a category)
3. tools (individual tool override)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from victor.processing.serialization.strategy import (
    SerializationFormat,
    SerializationConfig,
)

logger = logging.getLogger(__name__)


class ToolOutputType(Enum):
    """Classification of tool output types."""

    TABULAR = "tabular"  # List of uniform objects
    TEXT = "text"  # Plain text content
    STRUCTURED = "structured"  # Nested JSON structures
    MIXED = "mixed"  # Variable output


@dataclass
class ToolSerializationConfig:
    """Serialization configuration for a specific tool.

    Loaded from tool_serialization section of model_capabilities.yaml.
    """

    # Tool identification
    tool_name: str

    # Output type hint for format selection
    output_type: ToolOutputType = ToolOutputType.MIXED

    # Whether serialization is enabled for this tool
    serialization_enabled: bool = True

    # Preferred format (None = auto-select)
    preferred_format: Optional[SerializationFormat] = None

    # Allowed formats (order = preference)
    preferred_formats: List[SerializationFormat] = field(
        default_factory=lambda: [
            SerializationFormat.TOON,
            SerializationFormat.CSV,
            SerializationFormat.JSON_MINIFIED,
            SerializationFormat.JSON,
        ]
    )

    # Minimum rows to use tabular formats
    min_rows_for_tabular: int = 3

    # Minimum savings threshold
    min_savings_threshold: float = 0.20

    # Include format hint in output
    include_format_hint: bool = True

    # For tools with multiple operations (like git)
    serialize_operations: Optional[List[str]] = None
    skip_operations: Optional[List[str]] = None

    def should_serialize(self, operation: Optional[str] = None) -> bool:
        """Check if serialization should be applied.

        Args:
            operation: Optional operation name (e.g., "log" for git)

        Returns:
            True if serialization should be applied
        """
        if not self.serialization_enabled:
            return False

        if operation:
            # Check skip list first
            if self.skip_operations and operation in self.skip_operations:
                return False
            # Check serialize list if specified
            if self.serialize_operations:
                return operation in self.serialize_operations

        return True

    def to_serialization_config(self) -> SerializationConfig:
        """Convert to SerializationConfig for the serializer.

        Returns:
            SerializationConfig instance
        """
        return SerializationConfig(
            preferred_format=self.preferred_format,
            allowed_formats=self.preferred_formats,
            min_array_size_for_tabular=self.min_rows_for_tabular,
            min_savings_threshold=self.min_savings_threshold,
            include_format_hint=self.include_format_hint,
        )


class ToolSerializationRegistry:
    """Registry for tool-specific serialization configuration.

    Loads from model_capabilities.yaml and provides tool-aware config.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize registry.

        Args:
            config_path: Path to model_capabilities.yaml
        """
        if config_path is None:
            # Default to package config (victor/config/model_capabilities.yaml)
            # Navigate from victor/processing/serialization/ to victor/config/
            config_path = Path(__file__).parent.parent.parent / "config" / "model_capabilities.yaml"

        self._config_path = config_path
        self._config: Dict[str, Any] = {}
        self._cache: Dict[str, ToolSerializationConfig] = {}
        self._category_map: Dict[str, str] = {}  # tool -> category
        self._loaded = False

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        if self._loaded:
            return

        try:
            if self._config_path.exists():
                with open(self._config_path, "r") as f:
                    full_config = yaml.safe_load(f) or {}
                    self._config = full_config.get("tool_serialization", {})
                logger.debug(f"Loaded tool serialization config from {self._config_path}")

                # Build category map
                categories = self._config.get("categories", {})
                for category_name, category_config in categories.items():
                    for tool in category_config.get("tools", []):
                        self._category_map[tool] = category_name
            else:
                logger.debug(f"Config file not found: {self._config_path}")
                self._config = {}
        except Exception as e:
            logger.warning(f"Failed to load tool serialization config: {e}")
            self._config = {}

        self._loaded = True

    def get_tool_config(self, tool_name: str) -> ToolSerializationConfig:
        """Get serialization config for a specific tool.

        Resolution order:
        1. Individual tool override (tools.<tool_name>)
        2. Category config (categories.<category>.tools contains tool)
        3. Defaults

        Args:
            tool_name: Name of the tool

        Returns:
            ToolSerializationConfig for the tool
        """
        self._load_config()

        if tool_name in self._cache:
            return self._cache[tool_name]

        # Start with defaults
        config = self._get_defaults(tool_name)

        # Apply category if tool is in one
        if tool_name in self._category_map:
            category = self._category_map[tool_name]
            config = self._apply_category(config, category)

        # Apply individual tool override
        config = self._apply_tool_override(config, tool_name)

        self._cache[tool_name] = config
        return config

    def _get_defaults(self, tool_name: str) -> ToolSerializationConfig:
        """Get default configuration.

        Args:
            tool_name: Tool name for the config

        Returns:
            Default ToolSerializationConfig
        """
        defaults = self._config.get("defaults", {})
        return self._parse_config(tool_name, defaults)

    def _apply_category(
        self,
        base: ToolSerializationConfig,
        category: str,
    ) -> ToolSerializationConfig:
        """Apply category configuration.

        Args:
            base: Base config to update
            category: Category name

        Returns:
            Updated config
        """
        categories = self._config.get("categories", {})
        category_config = categories.get(category, {})

        if not category_config:
            return base

        return self._merge_config(base, category_config)

    def _apply_tool_override(
        self,
        base: ToolSerializationConfig,
        tool_name: str,
    ) -> ToolSerializationConfig:
        """Apply individual tool override.

        Args:
            base: Base config to update
            tool_name: Tool name

        Returns:
            Updated config
        """
        tools = self._config.get("tools", {})
        tool_config = tools.get(tool_name, {})

        if not tool_config:
            return base

        return self._merge_config(base, tool_config)

    def _parse_config(
        self,
        tool_name: str,
        config: Dict[str, Any],
    ) -> ToolSerializationConfig:
        """Parse configuration dict into ToolSerializationConfig.

        Args:
            tool_name: Tool name
            config: Config dict

        Returns:
            Parsed ToolSerializationConfig
        """
        result = ToolSerializationConfig(tool_name=tool_name)

        # Parse output type
        if "output_type" in config:
            try:
                result.output_type = ToolOutputType(config["output_type"])
            except ValueError:
                pass

        # Parse serialization enabled
        if "serialization_enabled" in config:
            result.serialization_enabled = config["serialization_enabled"]

        # Parse preferred format
        if "preferred_format" in config and config["preferred_format"]:
            try:
                result.preferred_format = SerializationFormat(config["preferred_format"])
            except ValueError:
                pass

        # Parse preferred formats list
        if "preferred_formats" in config:
            result.preferred_formats = []
            for fmt in config["preferred_formats"]:
                try:
                    result.preferred_formats.append(SerializationFormat(fmt))
                except ValueError:
                    pass

        # Parse thresholds
        if "min_rows_for_tabular" in config:
            result.min_rows_for_tabular = config["min_rows_for_tabular"]
        if "min_savings_threshold" in config:
            result.min_savings_threshold = config["min_savings_threshold"]
        if "include_format_hint" in config:
            result.include_format_hint = config["include_format_hint"]

        # Parse operation lists
        if "serialize_operations" in config:
            result.serialize_operations = config["serialize_operations"]
        if "skip_operations" in config:
            result.skip_operations = config["skip_operations"]

        return result

    def _merge_config(
        self,
        base: ToolSerializationConfig,
        overrides: Dict[str, Any],
    ) -> ToolSerializationConfig:
        """Merge override config into base.

        Args:
            base: Base config
            overrides: Override dict

        Returns:
            Merged config
        """
        result = ToolSerializationConfig(tool_name=base.tool_name)

        # Copy base values
        result.output_type = base.output_type
        result.serialization_enabled = base.serialization_enabled
        result.preferred_format = base.preferred_format
        result.preferred_formats = base.preferred_formats.copy()
        result.min_rows_for_tabular = base.min_rows_for_tabular
        result.min_savings_threshold = base.min_savings_threshold
        result.include_format_hint = base.include_format_hint
        result.serialize_operations = base.serialize_operations
        result.skip_operations = base.skip_operations

        # Apply overrides
        if "output_type" in overrides:
            try:
                result.output_type = ToolOutputType(overrides["output_type"])
            except ValueError:
                pass

        if "serialization_enabled" in overrides:
            result.serialization_enabled = overrides["serialization_enabled"]

        if "preferred_format" in overrides and overrides["preferred_format"]:
            try:
                result.preferred_format = SerializationFormat(overrides["preferred_format"])
            except ValueError:
                pass

        if "preferred_formats" in overrides:
            result.preferred_formats = []
            for fmt in overrides["preferred_formats"]:
                try:
                    result.preferred_formats.append(SerializationFormat(fmt))
                except ValueError:
                    pass

        if "min_rows_for_tabular" in overrides:
            result.min_rows_for_tabular = overrides["min_rows_for_tabular"]
        if "min_savings_threshold" in overrides:
            result.min_savings_threshold = overrides["min_savings_threshold"]
        if "include_format_hint" in overrides:
            result.include_format_hint = overrides["include_format_hint"]

        if "serialize_operations" in overrides:
            result.serialize_operations = overrides["serialize_operations"]
        if "skip_operations" in overrides:
            result.skip_operations = overrides["skip_operations"]

        return result

    def is_serialization_enabled(
        self,
        tool_name: str,
        operation: Optional[str] = None,
    ) -> bool:
        """Check if serialization is enabled for a tool/operation.

        Args:
            tool_name: Tool name
            operation: Optional operation name

        Returns:
            True if serialization should be applied
        """
        config = self.get_tool_config(tool_name)
        return config.should_serialize(operation)

    def get_output_type(self, tool_name: str) -> ToolOutputType:
        """Get output type hint for a tool.

        Args:
            tool_name: Tool name

        Returns:
            ToolOutputType hint
        """
        config = self.get_tool_config(tool_name)
        return config.output_type

    def clear_cache(self) -> None:
        """Clear the config cache."""
        self._cache.clear()

    def reload(self) -> None:
        """Force reload of configuration."""
        self._loaded = False
        self._cache.clear()
        self._category_map.clear()
        self._load_config()


# Global registry instance
_tool_registry: Optional[ToolSerializationRegistry] = None


def get_tool_serialization_registry() -> ToolSerializationRegistry:
    """Get the global tool serialization registry.

    Returns:
        ToolSerializationRegistry instance
    """
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolSerializationRegistry()
    return _tool_registry


def reset_tool_serialization_registry() -> None:
    """Reset the global registry (for testing)."""
    global _tool_registry
    _tool_registry = None
