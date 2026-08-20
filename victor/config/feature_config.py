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

"""Feature flag configuration management.

This module provides configuration loading and management for feature flags,
integrating with the Settings system for centralized configuration.

Key Features:
- Load feature flags from YAML configuration files
- Environment variable integration
- Settings-based flag defaults
- Configuration validation

Example:
    from victor.config.feature_config import load_feature_flags_from_settings

    flags = load_feature_flags_from_settings(settings)
    if flags.get("use_semantic_response_cache", False):
        # Use semantic response cache
        pass
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from victor.config.settings import Settings

logger = logging.getLogger(__name__)

# Feature flag defaults for development/testing
DEFAULT_FEATURE_FLAGS = {
    # Core architecture flags (disabled by default)
    "use_composition_over_inheritance": False,
    "use_strategy_based_tool_registration": False,
    "use_llm_decision_service": False,
    "use_edge_model": False,
    # Optimization flags (opt-in, disabled by default)
    "use_semantic_response_cache": False,
    # Smart routing (disabled by default)
    "use_smart_routing": False,
    # Learning from execution (enabled by default)
    "use_learning_from_execution": True,
    # Tool broadcasting optimization (opt-in)
    "tool_strategy_v2": False,
    # Agentic rollout flags (opt-in, disabled by default)
    "use_calibrated_completion": False,
    "use_prompt_completeness_guard": False,
    "use_prompt_dictionary_compression": False,
    "use_prime_memory_evolution": False,
    # Rich formatting (disabled by default)
    "use_rich_formatting": False,
    # Graph-based enhancements (disabled by default)
    "use_graph_rag": False,
    "use_ccg": False,
    "use_graph_query_tool": False,
    "use_multi_hop_retrieval": False,
    "use_graph_enhanced_context": False,
    # Architecture consolidation (opt-in for safety)
    "use_stategraph_agentic_loop": False,
    # Stage transition coordination (opt-in for gradual rollout)
    "use_stage_transition_coordinator": False,
}


def load_feature_flags_from_settings(
    settings: Optional[Settings] = None,
) -> Dict[str, bool]:
    """Load feature flags from settings and environment variables.

    This function loads feature flags from multiple sources:
    1. Environment variables (VICTOR_USE_EDGE_MODEL, etc.)
    2. Settings object (if provided)
    3. Default values

    Args:
        settings: Optional Settings object to load flags from

    Returns:
        Dictionary mapping feature flag names to boolean values
    """
    if settings is None:
        from victor.config.settings import load_settings

        settings = load_settings()

    flags = DEFAULT_FEATURE_FLAGS.copy()

    # Load from environment variables
    for flag_name in DEFAULT_FEATURE_FLAGS.keys():
        env_var = f"VICTOR_{flag_name.upper()}"
        if env_var in os.environ:
            value = os.environ[env_var].lower()
            flags[flag_name] = value in ("true", "1", "yes", "on")

    # Load from Settings object (if it has feature flag attributes)
    for flag_name in DEFAULT_FEATURE_FLAGS.keys():
        if hasattr(settings, flag_name):
            flags[flag_name] = bool(getattr(settings, flag_name))

    return flags


def load_feature_flags_from_yaml(
    config_path: Optional[Path] = None,
) -> Dict[str, bool]:
    """Load feature flags from a YAML configuration file.

    Args:
        config_path: Path to YAML configuration file. If None, uses
                    ~/.victor/features.yaml

    Returns:
        Dictionary mapping feature flag names to boolean values

    Example YAML format:
        features:
            use_semantic_response_cache: true
            use_smart_routing: false
            use_composition_over_inheritance: true
    """
    if config_path is None:
        from victor.config.settings import GLOBAL_VICTOR_DIR

        config_path = GLOBAL_VICTOR_DIR / "features.yaml"

    if not config_path.exists():
        logger.debug(f"Feature flag config not found: {config_path}")
        return DEFAULT_FEATURE_FLAGS.copy()

    try:
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        if not data or not isinstance(data, dict):
            logger.warning(f"Invalid feature flag config: {config_path}")
            return DEFAULT_FEATURE_FLAGS.copy()

        features = data.get("features", {})
        if not isinstance(features, dict):
            logger.warning("Invalid 'features' section in config")
            return DEFAULT_FEATURE_FLAGS.copy()

        # Merge with defaults
        flags = DEFAULT_FEATURE_FLAGS.copy()
        for key, value in features.items():
            if key in flags and isinstance(value, bool):
                flags[key] = value
            elif key not in flags:
                logger.warning(f"Unknown feature flag: {key}")
            else:
                logger.warning(
                    f"Invalid value for feature flag '{key}': "
                    f"expected bool, got {type(value).__name__}"
                )

        return flags

    except Exception as e:
        logger.warning(f"Failed to load feature flags from YAML: {e}")
        return DEFAULT_FEATURE_FLAGS.copy()


def save_feature_flags_to_yaml(
    flags: Dict[str, bool],
    config_path: Optional[Path] = None,
) -> None:
    """Save feature flags to a YAML configuration file.

    Args:
        flags: Dictionary mapping feature flag names to boolean values
        config_path: Path to YAML configuration file. If None, uses
                    ~/.victor/features.yaml
    """
    if config_path is None:
        from victor.config.settings import GLOBAL_VICTOR_DIR

        config_path = GLOBAL_VICTOR_DIR / "features.yaml"

    # Create parent directory if it doesn't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare YAML data
    data = {
        "features": {key: value for key, value in flags.items() if key in DEFAULT_FEATURE_FLAGS}
    }

    try:
        with open(config_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved feature flags to: {config_path}")
    except Exception as e:
        logger.error(f"Failed to save feature flags to YAML: {e}")
        raise


def validate_feature_flags(flags: Dict[str, bool]) -> bool:
    """Validate feature flags dictionary.

    Args:
        flags: Dictionary mapping feature flag names to boolean values

    Returns:
        True if all flags are valid, False otherwise
    """
    for key, value in flags.items():
        if key not in DEFAULT_FEATURE_FLAGS:
            logger.warning(f"Unknown feature flag: {key}")
            return False
        if not isinstance(value, bool):
            logger.warning(
                f"Invalid value for feature flag '{key}': "
                f"expected bool, got {type(value).__name__}"
            )
            return False

    return True


def get_feature_flag_summary(flags: Optional[Dict[str, bool]] = None) -> str:
    """Get a human-readable summary of feature flags.

    Args:
        flags: Optional flags dictionary. If None, loads from settings

    Returns:
        Human-readable summary of feature flag states
    """
    if flags is None:
        flags = load_feature_flags_from_settings()

    lines = ["Feature Flags Status:", ""]

    for key, value in sorted(flags.items()):
        status = "✓ ENABLED" if value else "✗ DISABLED"
        lines.append(f"  {status}: {key}")

    return "\n".join(lines)


def create_feature_flag_config_file(
    config_path: Optional[Path] = None,
    enabled_flags: Optional[list[str]] = None,
) -> None:
    """Create or update the feature flag YAML configuration file.

    Args:
        config_path: Path to YAML configuration file. If None, uses
                    ~/.victor/features.yaml
        enabled_flags: List of flag names to enable. All others will be disabled.
                     If None, uses current settings.
    """
    if enabled_flags is None:
        flags = load_feature_flags_from_settings()
    else:
        flags = {key: key in enabled_flags for key in DEFAULT_FEATURE_FLAGS.keys()}

    save_feature_flags_to_yaml(flags, config_path)
    logger.info(f"Created feature flag config at: {config_path}")
