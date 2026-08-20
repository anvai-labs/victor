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

"""
Model-centric capability loader for model_capabilities.yaml v0.2.0.

Schema v0.2.0 is model-centric:
  models.<pattern>:
    training:    # What the model was trained to do (provider-independent)
    providers:   # How each provider enables these capabilities
    settings:    # Tuning parameters

Resolution order:
  1. defaults (global)
  2. provider_defaults.<provider>
  3. models.<pattern>.training
  4. models.<pattern>.providers.<provider>
  5. models.<pattern>.settings
"""

import fnmatch
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from victor.agent.tool_calling.base import ToolCallingCapabilities, ToolCallFormat

logger = logging.getLogger(__name__)


# =============================================================================
# MODEL NAME NORMALIZATION
# =============================================================================
# Handles common model naming variants to ensure capability lookup succeeds.
# E.g., "qwen25-coder" → "qwen2.5-coder", "llama33" → "llama3.3"

# Pre-compiled regex patterns for better performance
_COMPILED_ALIASES = [
    (re.compile(r"qwen25([^0-9])", re.IGNORECASE), r"qwen2.5\1"),
    (re.compile(r"qwen25$", re.IGNORECASE), r"qwen2.5"),
    (re.compile(r"llama33([^0-9])", re.IGNORECASE), r"llama3.3\1"),
    (re.compile(r"llama33$", re.IGNORECASE), r"llama3.3"),
    (re.compile(r"llama31([^0-9])", re.IGNORECASE), r"llama3.1\1"),
    (re.compile(r"llama31$", re.IGNORECASE), r"llama3.1"),
    (re.compile(r"llama32([^0-9])", re.IGNORECASE), r"llama3.2\1"),
    (re.compile(r"llama32$", re.IGNORECASE), r"llama3.2"),
    (re.compile(r"deepseekr1", re.IGNORECASE), r"deepseek-r1"),
    (re.compile(r"deepseek_r1", re.IGNORECASE), r"deepseek-r1"),
    (re.compile(r"deepseekcoder", re.IGNORECASE), r"deepseek-coder"),
]

# Cache for normalized model names
_NORMALIZATION_CACHE: Dict[str, str] = {}


def normalize_model_name(model_name: str) -> str:
    """Normalize model name to canonical form for capability lookup.

    Applies alias patterns to handle common naming variants:
    - qwen25-coder → qwen2.5-coder
    - llama33:70b → llama3.3:70b
    - deepseekr1 → deepseek-r1

    Args:
        model_name: Original model name (e.g., "qwen25-coder-tools:14b-64K")

    Returns:
        Normalized model name (e.g., "qwen2.5-coder-tools:14b-64K")
    """
    if not model_name:
        return model_name

    # Check cache first
    if model_name in _NORMALIZATION_CACHE:
        return _NORMALIZATION_CACHE[model_name]

    normalized = model_name.lower()
    original_lower = normalized

    for pattern, replacement in _COMPILED_ALIASES:
        normalized = pattern.sub(replacement, normalized)

    # Cache the result
    _NORMALIZATION_CACHE[model_name] = normalized

    if normalized != original_lower:
        logger.debug(f"Normalized model name: {model_name} → {normalized}")

    return normalized


def get_model_name_variants(model_name: str) -> List[str]:
    """Get all naming variants for a model to try during lookup.

    Returns both the original name and normalized form(s) to maximize
    matching against capability patterns.

    Args:
        model_name: Original model name

    Returns:
        List of model name variants to try (original first, then normalized)
    """
    variants = [model_name.lower()]

    normalized = normalize_model_name(model_name)
    if normalized not in variants:
        variants.append(normalized)

    return variants


class ModelCapabilityLoader:
    """Loads and provides model capabilities from YAML configuration.

    Supports the model-centric v2.0 schema with:
    - training: Provider-independent capabilities from model training
    - providers: Per-provider capability overrides
    - settings: Model-specific tuning parameters

    Usage:
        loader = ModelCapabilityLoader()
        caps = loader.get_capabilities("ollama", "qwen3-coder:30b")
    """

    _instance: Optional["ModelCapabilityLoader"] = None
    _config: Optional[Dict[str, Any]] = None

    def __new__(cls) -> "ModelCapabilityLoader":
        """Singleton pattern for efficiency."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize loader and load configuration."""
        if self._config is None:
            self._load_config()

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        config_path = Path(__file__).parent.parent.parent / "config" / "model_capabilities.yaml"

        if not config_path.exists():
            logger.warning(f"Model capabilities config not found at {config_path}")
            self._config = {"defaults": {}, "provider_defaults": {}, "models": {}}
            return

        try:
            with open(config_path) as f:
                self._config = yaml.safe_load(f) or {}
            logger.debug(f"Loaded model capabilities from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load model capabilities: {e}")
            self._config = {"defaults": {}, "provider_defaults": {}, "models": {}}

    def reload(self) -> None:
        """Force reload configuration from file."""
        self._load_config()

    def get_capabilities(
        self,
        provider: str,
        model: str = "",
        format_hint: Optional[ToolCallFormat] = None,
    ) -> ToolCallingCapabilities:
        """Get capabilities for a provider/model combination.

        Resolution order (later overrides earlier):
        1. Global defaults (defaults section)
        2. Provider defaults (provider_defaults.<provider>)
        3. Model training capabilities (models.<pattern>.training)
        4. Model provider-specific support (models.<pattern>.providers.<provider>)
        5. Model settings (models.<pattern>.settings)

        Args:
            provider: Provider name (ollama, lmstudio, vllm, etc.)
            model: Model name/identifier
            format_hint: Optional format hint to set

        Returns:
            ToolCallingCapabilities with resolved values
        """
        if self._config is None:
            self._load_config()

        config_dict = self._config or {}
        provider_lower = provider.lower()

        # Start building resolved config
        resolved = {}

        # 1. Apply global defaults
        defaults = config_dict.get("defaults")
        if defaults:
            self._apply_defaults(resolved, defaults)

        # 2. Apply provider defaults
        provider_defaults = config_dict.get("provider_defaults", {}).get(provider_lower)
        if provider_defaults:
            logger.debug(f"Applying provider_defaults for '{provider_lower}'")
            resolved.update(provider_defaults)

        # 3-5. Find and apply matching model configuration
        if model:
            models = config_dict.get("models")
            if models:
                # Get all name variants to try
                model_variants = get_model_name_variants(model)
                matching = None
                for variant in model_variants:
                    matching = self._find_matching_model(models, variant)
                    if matching:
                        logger.debug(f"Found capability match using variant: {variant}")
                        break

                if matching:
                    for pattern, model_config in matching:
                        logger.debug(f"Applying model pattern '{pattern}'")

                        # 3. Apply training capabilities
                        training = model_config.get("training")
                        if training:
                            self._apply_training(resolved, training)

                        # 4. Apply provider-specific overrides
                        providers = model_config.get("providers")
                        if providers:
                            provider_config = providers.get(provider_lower)
                            if provider_config:
                                logger.debug(f"Applying model.providers.{provider_lower}")
                                resolved.update(provider_config)

                        # 5. Apply model settings
                        settings = model_config.get("settings")
                        if settings:
                            resolved.update(settings)

        # Convert to ToolCallingCapabilities
        return self._config_to_capabilities(resolved, format_hint)

    def _apply_defaults(self, resolved: Dict[str, Any], defaults: Dict[str, Any]) -> None:
        """Apply defaults section to resolved config."""
        # Flatten nested defaults structure
        for key, value in defaults.items():
            if isinstance(value, dict):
                # Handle nested sections like training, provider_support, fallback, settings
                resolved.update(value)
            else:
                resolved[key] = value

    def _apply_training(self, resolved: Dict[str, Any], training: Dict[str, Any]) -> None:
        """Map training capabilities to provider support flags.

        Training capabilities indicate what the model CAN do.
        We map these to the runtime capability fields.
        """
        # Map training.tool_calling -> affects fallback parsing
        if training.get("tool_calling"):
            # Model has tool training, so fallback parsing should work
            resolved.setdefault("json_fallback_parsing", True)
            resolved.setdefault("xml_fallback_parsing", True)

        # Direct mappings for thinking mode
        if "thinking_mode" in training:
            resolved["thinking_mode"] = training["thinking_mode"]

        # thinking_disable_prefix is model-specific (e.g., "/no_think" for Qwen3)
        if "thinking_disable_prefix" in training:
            resolved["thinking_disable_prefix"] = training["thinking_disable_prefix"]

    def _find_matching_model(self, models: Dict[str, Any], model_lower: str) -> List[tuple]:
        """Find matching model patterns, sorted by specificity."""
        matching = []

        for pattern, config in models.items():
            if isinstance(config, dict):
                pattern_lower = pattern.lower()
                if self._matches_pattern(model_lower, pattern_lower):
                    # Score by specificity (longer patterns are more specific)
                    specificity = len(pattern.replace("*", ""))
                    matching.append((specificity, pattern, config))

        if not matching:
            return []

        # Sort by specificity (least specific first, so more specific overrides)
        matching.sort(key=lambda x: x[0])
        return [(pattern, config) for _, pattern, config in matching]

    def _matches_pattern(self, model_name: str, pattern: str) -> bool:
        """Check if model name matches a pattern.

        Supports:
        - Glob patterns (*, ?)
        - Prefix matching (llama3.1* matches llama3.1:8b)
        """
        # Try fnmatch first
        if fnmatch.fnmatch(model_name, pattern):
            return True

        # Check for prefix match (pattern without *)
        prefix = pattern.rstrip("*")
        if prefix and model_name.startswith(prefix):
            return True

        return False

    def _config_to_capabilities(
        self,
        config: Dict[str, Any],
        format_hint: Optional[ToolCallFormat] = None,
    ) -> ToolCallingCapabilities:
        """Convert resolved config dict to ToolCallingCapabilities object."""
        return ToolCallingCapabilities(
            native_tool_calls=config.get("native_tool_calls", False),
            streaming_tool_calls=config.get("streaming_tool_calls", False),
            parallel_tool_calls=config.get("parallel_tool_calls", False),
            tool_choice_param=config.get("tool_choice_param", False),
            json_fallback_parsing=config.get("json_fallback_parsing", True),
            xml_fallback_parsing=config.get("xml_fallback_parsing", True),
            thinking_mode=config.get("thinking_mode", False),
            thinking_disable_prefix=config.get("thinking_disable_prefix"),
            requires_strict_prompting=config.get("requires_strict_prompting", True),
            tool_call_format=format_hint or ToolCallFormat.UNKNOWN,
            argument_format=config.get("argument_format", "json"),
            recommended_max_tools=config.get("recommended_max_tools", 20),
            recommended_tool_budget=config.get("recommended_tool_budget", 12),
            # Model-specific exploration behavior
            exploration_multiplier=config.get("exploration_multiplier", 1.0),
            continuation_patience=config.get("continuation_patience", 3),
            # Model-specific timeout settings
            timeout_multiplier=config.get("timeout_multiplier", 1.0),
        )

    # =========================================================================
    # Query and introspection methods
    # =========================================================================

    def get_model_config(self, model_pattern: str) -> Optional[Dict[str, Any]]:
        """Get raw configuration for a model pattern.

        Useful for helper tools that need to read/modify specific model configs.

        Args:
            model_pattern: Exact model pattern key (e.g., "qwen3-coder*")

        Returns:
            Dict with training, providers, settings sections, or None if not found
        """
        config_dict = self._config or {}
        return config_dict.get("models", {}).get(model_pattern)

    def get_all_model_patterns(self) -> List[str]:
        """Get all configured model patterns."""
        config_dict = self._config or {}
        return list(config_dict.get("models", {}).keys())

    def get_provider_names(self) -> List[str]:
        """Get list of configured provider names."""
        config_dict = self._config or {}
        return list(config_dict.get("provider_defaults", {}).keys())

    def get_training_capabilities(self, model_pattern: str) -> Optional[Dict[str, Any]]:
        """Get training capabilities for a model pattern.

        Args:
            model_pattern: Exact model pattern key

        Returns:
            Training dict or None
        """
        model_config = self.get_model_config(model_pattern)
        if model_config:
            return model_config.get("training")
        return None

    def get_provider_support(self, model_pattern: str, provider: str) -> Optional[Dict[str, Any]]:
        """Get provider-specific support for a model.

        Args:
            model_pattern: Exact model pattern key
            provider: Provider name

        Returns:
            Provider support dict or None
        """
        model_config = self.get_model_config(model_pattern)
        if model_config:
            providers = model_config.get("providers", {})
            return providers.get(provider.lower())
        return None

    def debug_resolution(self, provider: str, model: str) -> Dict[str, Any]:
        """Debug helper to show capability resolution steps.

        Args:
            provider: Provider name
            model: Model name

        Returns:
            Dict with resolution details
        """
        if self._config is None:
            self._load_config()

        config_dict = self._config or {}
        model_lower = model.lower()
        provider_lower = provider.lower()

        result = {
            "provider": provider,
            "model": model,
            "schema_version": config_dict.get("schema_version", "unknown"),
            "resolution_steps": [],
            "final_capabilities": {},
        }

        resolved = {}

        # 1. Global defaults
        defaults = config_dict.get("defaults", {})
        if defaults:
            result["resolution_steps"].append(
                {
                    "source": "defaults",
                    "applied": dict(defaults),
                }
            )
            self._apply_defaults(resolved, defaults)

        # 2. Provider defaults
        provider_defaults = config_dict.get("provider_defaults", {}).get(provider_lower, {})
        if provider_defaults:
            result["resolution_steps"].append(
                {
                    "source": f"provider_defaults.{provider_lower}",
                    "applied": dict(provider_defaults),
                }
            )
            resolved.update(provider_defaults)

        # 3-5. Model config
        models = config_dict.get("models", {})
        matching = self._find_matching_model(models, model_lower)

        for pattern, model_config in matching:
            # Training
            training = model_config.get("training", {})
            if training:
                result["resolution_steps"].append(
                    {
                        "source": f"models.{pattern}.training",
                        "applied": dict(training),
                    }
                )
                self._apply_training(resolved, training)

            # Provider-specific
            providers = model_config.get("providers", {})
            provider_config = providers.get(provider_lower, {})
            if provider_config:
                result["resolution_steps"].append(
                    {
                        "source": f"models.{pattern}.providers.{provider_lower}",
                        "applied": dict(provider_config),
                    }
                )
                resolved.update(provider_config)

            # Settings
            settings = model_config.get("settings", {})
            if settings:
                result["resolution_steps"].append(
                    {
                        "source": f"models.{pattern}.settings",
                        "applied": dict(settings),
                    }
                )
                resolved.update(settings)

        result["final_capabilities"] = resolved
        return result


# =============================================================================
# Module-level convenience functions
# =============================================================================


def get_model_capabilities(
    provider: str,
    model: str = "",
    format_hint: Optional[ToolCallFormat] = None,
) -> ToolCallingCapabilities:
    """Get capabilities for a provider/model combination.

    Convenience function that uses the singleton loader.

    Args:
        provider: Provider name
        model: Model name
        format_hint: Optional format hint

    Returns:
        ToolCallingCapabilities
    """
    loader = ModelCapabilityLoader()
    return loader.get_capabilities(provider, model, format_hint)


def get_model_training(model_pattern: str) -> Optional[Dict[str, Any]]:
    """Get training capabilities for a model pattern.

    Args:
        model_pattern: Exact model pattern key (e.g., "qwen3-coder*")

    Returns:
        Training dict or None
    """
    loader = ModelCapabilityLoader()
    return loader.get_training_capabilities(model_pattern)


def get_provider_support(model_pattern: str, provider: str) -> Optional[Dict[str, Any]]:
    """Get provider-specific support for a model.

    Args:
        model_pattern: Exact model pattern key
        provider: Provider name

    Returns:
        Provider support dict or None
    """
    loader = ModelCapabilityLoader()
    return loader.get_provider_support(model_pattern, provider)
