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

"""Model serialization capabilities registry.

Loads serialization preferences from model_capabilities.yaml and provides
model/provider-specific configuration for the adaptive serializer.

Uses the existing hierarchical resolution:
1. defaults.serialization
2. provider_defaults.<provider>.serialization
3. models.<pattern>.serialization (if present)
"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

from victor.processing.serialization.strategy import (
    SerializationFormat,
    SerializationConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelSerializationCapabilities:
    """Serialization capabilities for a specific model/provider combination.

    Loaded from model_capabilities.yaml with hierarchical resolution.
    """

    # Preferred format (None = auto-select)
    preferred_format: Optional[SerializationFormat] = None

    # Allowed formats for auto-selection
    allowed_formats: List[SerializationFormat] = field(
        default_factory=lambda: [
            SerializationFormat.TOON,
            SerializationFormat.CSV,
            SerializationFormat.JSON_MINIFIED,
            SerializationFormat.JSON,
        ]
    )

    # Formats to disable
    disabled_formats: Set[SerializationFormat] = field(default_factory=set)

    # Thresholds
    min_array_size_for_tabular: int = 3
    min_savings_threshold: float = 0.20
    min_repetition_for_references: float = 0.5
    max_nesting_for_compact: int = 1

    # Behavior
    include_format_hint: bool = True
    enable_reference_encoding: bool = True
    debug_mode: bool = False

    def to_config(self) -> SerializationConfig:
        """Convert to SerializationConfig for use by serializer.

        Returns:
            SerializationConfig instance
        """
        return SerializationConfig(
            preferred_format=self.preferred_format,
            allowed_formats=self.allowed_formats,
            disabled_formats=self.disabled_formats,
            min_array_size_for_tabular=self.min_array_size_for_tabular,
            min_savings_threshold=self.min_savings_threshold,
            include_format_hint=self.include_format_hint,
            enable_reference_encoding=self.enable_reference_encoding,
            min_repetition_for_references=self.min_repetition_for_references,
            max_nesting_for_compact=self.max_nesting_for_compact,
            debug_mode=self.debug_mode,
        )


class CapabilityRegistry:
    """Registry for model/provider serialization capabilities.

    Loads from model_capabilities.yaml and provides hierarchical resolution.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize registry.

        Args:
            config_path: Path to model_capabilities.yaml.
                        Defaults to victor/config/model_capabilities.yaml
        """
        if config_path is None:
            # Default to package config (victor/config/model_capabilities.yaml)
            # Navigate from victor/processing/serialization/ to victor/config/
            config_path = Path(__file__).parent.parent.parent / "config" / "model_capabilities.yaml"

        self._config_path = config_path
        self._config: Dict[str, Any] = {}
        self._cache: Dict[str, ModelSerializationCapabilities] = {}
        self._loaded = False

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        if self._loaded:
            return

        try:
            if self._config_path.exists():
                with open(self._config_path, "r") as f:
                    self._config = yaml.safe_load(f) or {}
                logger.debug(f"Loaded serialization config from {self._config_path}")
            else:
                logger.warning(f"Config file not found: {self._config_path}")
                self._config = {}
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
            self._config = {}

        self._loaded = True

    def get_capabilities(
        self,
        provider: str,
        model: Optional[str] = None,
    ) -> ModelSerializationCapabilities:
        """Get serialization capabilities for a model/provider.

        Resolution order:
        1. defaults.serialization
        2. provider_defaults.<provider>.serialization
        3. models.<pattern>.serialization (first matching pattern)

        Args:
            provider: Provider name (e.g., "anthropic", "ollama")
            model: Optional model name for pattern matching

        Returns:
            ModelSerializationCapabilities with resolved settings
        """
        self._load_config()

        cache_key = f"{provider}:{model or 'default'}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Start with defaults
        capabilities = self._get_defaults()

        # Apply provider defaults
        capabilities = self._apply_provider_defaults(capabilities, provider)

        # Apply model-specific overrides if model specified
        if model:
            capabilities = self._apply_model_overrides(capabilities, model)

        self._cache[cache_key] = capabilities
        return capabilities

    def _get_defaults(self) -> ModelSerializationCapabilities:
        """Get default serialization capabilities.

        Returns:
            Capabilities from defaults.serialization
        """
        defaults = self._config.get("defaults", {})
        serialization = defaults.get("serialization", {})
        return self._parse_capabilities(serialization)

    def _apply_provider_defaults(
        self,
        base: ModelSerializationCapabilities,
        provider: str,
    ) -> ModelSerializationCapabilities:
        """Apply provider-specific defaults.

        Args:
            base: Base capabilities to update
            provider: Provider name

        Returns:
            Updated capabilities
        """
        provider_defaults = self._config.get("provider_defaults", {})
        provider_config = provider_defaults.get(provider, {})
        serialization = provider_config.get("serialization", {})

        if not serialization:
            return base

        return self._merge_capabilities(base, serialization)

    def _apply_model_overrides(
        self,
        base: ModelSerializationCapabilities,
        model: str,
    ) -> ModelSerializationCapabilities:
        """Apply model-specific overrides.

        Args:
            base: Base capabilities to update
            model: Model name

        Returns:
            Updated capabilities
        """
        models = self._config.get("models", {})

        # Find first matching pattern
        for pattern, config in models.items():
            if fnmatch.fnmatch(model.lower(), pattern.lower()):
                serialization = config.get("serialization", {})
                if serialization:
                    return self._merge_capabilities(base, serialization)
                break

        return base

    def _parse_capabilities(
        self,
        config: Dict[str, Any],
    ) -> ModelSerializationCapabilities:
        """Parse capabilities from config dict.

        Args:
            config: Serialization config dict

        Returns:
            Parsed ModelSerializationCapabilities
        """
        caps = ModelSerializationCapabilities()

        if not config:
            return caps

        # Parse preferred format
        if "preferred_format" in config and config["preferred_format"]:
            try:
                caps.preferred_format = SerializationFormat(config["preferred_format"])
            except ValueError:
                pass

        # Parse allowed formats
        if "allowed_formats" in config:
            caps.allowed_formats = []
            for fmt in config["allowed_formats"]:
                try:
                    caps.allowed_formats.append(SerializationFormat(fmt))
                except ValueError:
                    logger.warning(f"Unknown format: {fmt}")

        # Parse disabled formats
        if "disabled_formats" in config:
            caps.disabled_formats = set()
            for fmt in config["disabled_formats"]:
                try:
                    caps.disabled_formats.add(SerializationFormat(fmt))
                except ValueError:
                    pass

        # Parse thresholds
        if "min_array_size_for_tabular" in config:
            caps.min_array_size_for_tabular = config["min_array_size_for_tabular"]
        if "min_savings_threshold" in config:
            caps.min_savings_threshold = config["min_savings_threshold"]
        if "min_repetition_for_references" in config:
            caps.min_repetition_for_references = config["min_repetition_for_references"]
        if "max_nesting_for_compact" in config:
            caps.max_nesting_for_compact = config["max_nesting_for_compact"]

        # Parse behavior flags
        if "include_format_hint" in config:
            caps.include_format_hint = config["include_format_hint"]
        if "enable_reference_encoding" in config:
            caps.enable_reference_encoding = config["enable_reference_encoding"]
        if "debug_mode" in config:
            caps.debug_mode = config["debug_mode"]

        return caps

    def _merge_capabilities(
        self,
        base: ModelSerializationCapabilities,
        overrides: Dict[str, Any],
    ) -> ModelSerializationCapabilities:
        """Merge override config into base capabilities.

        Args:
            base: Base capabilities
            overrides: Override config dict

        Returns:
            Merged capabilities
        """
        # Parse overrides
        override_caps = self._parse_capabilities(overrides)

        # Create new instance with merged values
        return ModelSerializationCapabilities(
            preferred_format=(
                override_caps.preferred_format
                if "preferred_format" in overrides
                else base.preferred_format
            ),
            allowed_formats=(
                override_caps.allowed_formats
                if "allowed_formats" in overrides
                else base.allowed_formats
            ),
            disabled_formats=(base.disabled_formats | override_caps.disabled_formats),
            min_array_size_for_tabular=(
                override_caps.min_array_size_for_tabular
                if "min_array_size_for_tabular" in overrides
                else base.min_array_size_for_tabular
            ),
            min_savings_threshold=(
                override_caps.min_savings_threshold
                if "min_savings_threshold" in overrides
                else base.min_savings_threshold
            ),
            min_repetition_for_references=(
                override_caps.min_repetition_for_references
                if "min_repetition_for_references" in overrides
                else base.min_repetition_for_references
            ),
            max_nesting_for_compact=(
                override_caps.max_nesting_for_compact
                if "max_nesting_for_compact" in overrides
                else base.max_nesting_for_compact
            ),
            include_format_hint=(
                override_caps.include_format_hint
                if "include_format_hint" in overrides
                else base.include_format_hint
            ),
            enable_reference_encoding=(
                override_caps.enable_reference_encoding
                if "enable_reference_encoding" in overrides
                else base.enable_reference_encoding
            ),
            debug_mode=(override_caps.debug_mode if "debug_mode" in overrides else base.debug_mode),
        )

    def clear_cache(self) -> None:
        """Clear the capabilities cache."""
        self._cache.clear()

    def reload(self) -> None:
        """Force reload of configuration."""
        self._loaded = False
        self._cache.clear()
        self._load_config()


# Global registry instance
_registry: Optional[CapabilityRegistry] = None


def get_capability_registry() -> CapabilityRegistry:
    """Get the global capability registry.

    Returns:
        CapabilityRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = CapabilityRegistry()
    return _registry


def reset_capability_registry() -> None:
    """Reset the global capability registry (mainly for testing)."""
    global _registry
    _registry = None


def config_from_settings() -> SerializationConfig:
    """Create SerializationConfig from Settings.

    Loads serialization settings from the main Settings class and creates
    a base configuration that can be merged with provider/model overrides.

    Returns:
        SerializationConfig with settings from Settings class
    """
    try:
        from victor.config.settings import Settings

        settings = Settings()

        # Parse default format if specified
        preferred_format = None
        if settings.pipeline.serialization_default_format:
            try:
                preferred_format = SerializationFormat(
                    settings.pipeline.serialization_default_format
                )
            except ValueError:
                logger.warning(
                    f"Unknown serialization format: {settings.pipeline.serialization_default_format}"
                )

        return SerializationConfig(
            preferred_format=preferred_format,
            min_savings_threshold=settings.pipeline.serialization_min_savings_threshold,
            include_format_hint=settings.pipeline.serialization_include_format_hint,
            min_array_size_for_tabular=settings.pipeline.serialization_min_rows_for_tabular,
            debug_mode=settings.pipeline.serialization_debug_mode,
        )

    except ImportError:
        # Settings not available, return defaults
        return SerializationConfig()


def is_serialization_enabled() -> bool:
    """Check if serialization is globally enabled via Settings.

    Returns:
        True if serialization is enabled
    """
    try:
        from victor.config.settings import Settings

        return Settings().serialization.serialization_enabled
    except ImportError:
        return True  # Default to enabled
