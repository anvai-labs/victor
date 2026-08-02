# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Configuration loaders for YAML-based settings.

This module provides loaders for externalized configuration:
- Provider context limits (provider_context_limits.yaml)
- Stage keywords (stage_keywords.yaml)

Design Principles:
- Hot-reload support via file watching
- Caching with TTL for performance
- Fallback to defaults if files missing
- Type-safe dataclasses for configuration
"""

from __future__ import annotations

import fnmatch
import logging
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# Configuration file paths
CONFIG_DIR = Path(__file__).parent
PROVIDER_LIMITS_FILE = CONFIG_DIR / "provider_context_limits.yaml"
STAGE_KEYWORDS_FILE = CONFIG_DIR / "stage_keywords.yaml"
LOGGING_CONFIG_FILE = CONFIG_DIR / "logging_config.yaml"

# User config directory
USER_CONFIG_DIR = Path.home() / ".victor"
USER_CONFIG_FILE = USER_CONFIG_DIR / "config.yaml"

# Cache settings
_cache_ttl = 300  # 5 minutes
_cache_timestamps: Dict[str, float] = {}
_cache_data: Dict[str, Any] = {}


@dataclass
class ProviderLimits:
    """Context window and rate limits for a provider."""

    context_window: int = 128000
    response_reserve: int = 4096
    supports_extended_context: bool = False
    rate_limit_rpm: int = 60
    rate_limit_tpm: int = 100000

    @property
    def effective_context(self) -> int:
        """Get effective context window (minus response reserve)."""
        return self.context_window - self.response_reserve


@dataclass
class StageConfig:
    """Configuration for a conversation stage."""

    keywords: List[str] = field(default_factory=list)
    weight: float = 1.0
    min_score: int = 2
    tool_preferences: List[str] = field(default_factory=list)


@dataclass
class LoggingConfig:
    """Centralized logging configuration.

    Priority chain (highest to lowest):
    1. CLI argument
    2. Environment variable (VICTOR_LOG_LEVEL, VICTOR_LOG_FILE_LEVEL, etc.)
    3. User config file (~/.victor/config.yaml)
    4. Command-specific override from package config
    5. Package defaults (logging_config.yaml)
    6. Hardcoded fallback

    Attributes:
        console_level: Log level for console/stderr output
        file_level: Log level for file output
        file_enabled: Whether to enable file logging
        file_path: Path to log file (supports ~ expansion)
        file_max_bytes: Maximum size per log file before rotation
        file_backup_count: Number of rotated backup files to keep
        console_format: Format string for console logs
        file_format: Format string for file logs
        event_logging: Enable EventBus -> logging integration
        module_levels: Per-module log level overrides
    """

    console_level: str = "WARNING"
    file_level: str = "INFO"
    file_enabled: bool = True
    file_path: str = "~/.victor/logs/victor.log"
    file_max_bytes: int = 10 * 1024 * 1024  # 10MB
    file_backup_count: int = 5
    console_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_format: str = "%(asctime)s - %(session)s - %(name)s - %(levelname)s - %(message)s"
    event_logging: bool = True
    module_levels: Dict[str, str] = field(default_factory=dict)

    @property
    def expanded_file_path(self) -> Path:
        """Get file path with ~ expanded."""
        return Path(self.file_path).expanduser()

    def get_console_level_int(self) -> int:
        """Get console level as logging int constant."""
        return getattr(logging, self.console_level.upper(), logging.WARNING)

    def get_file_level_int(self) -> int:
        """Get file level as logging int constant."""
        return getattr(logging, self.file_level.upper(), logging.INFO)


def _load_yaml_cached(file_path: Path, cache_key: str) -> Optional[Dict[str, Any]]:
    """Load YAML file with caching.

    Args:
        file_path: Path to YAML file
        cache_key: Key for caching

    Returns:
        Parsed YAML data or None if file doesn't exist
    """
    now = time.time()

    # Check cache freshness
    if cache_key in _cache_data:
        cached_time = _cache_timestamps.get(cache_key, 0)
        if now - cached_time < _cache_ttl:
            return _cache_data[cache_key]

    if not file_path.exists():
        logger.debug(f"Config file not found: {file_path}")
        return None

    try:
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)

        _cache_data[cache_key] = data
        _cache_timestamps[cache_key] = now
        return data

    except Exception as e:
        logger.warning(f"Failed to load {file_path}: {e}")
        return None


def invalidate_config_cache() -> None:
    """Invalidate all configuration caches (for hot-reload)."""
    _cache_data.clear()
    _cache_timestamps.clear()


def get_provider_limits(provider: str, model: Optional[str] = None) -> ProviderLimits:
    """Get context limits for a provider/model.

    Args:
        provider: Provider name (e.g., 'anthropic', 'ollama')
        model: Optional model name for model-specific overrides

    Returns:
        ProviderLimits with context window and rate limits
    """
    data = _load_yaml_cached(PROVIDER_LIMITS_FILE, "provider_limits")

    if not data:
        # Return defaults if no config
        return ProviderLimits()

    providers = data.get("providers", {})
    models = data.get("models", {})

    # Start with provider defaults
    provider_data = providers.get(provider, {})
    limits = ProviderLimits(
        context_window=provider_data.get("context_window", 128000),
        response_reserve=provider_data.get("response_reserve", 4096),
        supports_extended_context=provider_data.get("supports_extended_context", False),
        rate_limit_rpm=provider_data.get("rate_limit_rpm", 60),
        rate_limit_tpm=provider_data.get("rate_limit_tpm", 100000),
    )

    # Apply model-specific overrides
    if model:
        for pattern, overrides in models.items():
            if fnmatch.fnmatch(model, pattern):
                if "context_window" in overrides:
                    limits.context_window = overrides["context_window"]
                if "response_reserve" in overrides:
                    limits.response_reserve = overrides["response_reserve"]
                break

    return limits


def get_all_provider_limits() -> Dict[str, ProviderLimits]:
    """Get limits for all configured providers.

    Returns:
        Dictionary mapping provider names to their limits
    """
    data = _load_yaml_cached(PROVIDER_LIMITS_FILE, "provider_limits")

    if not data:
        return {}

    providers = data.get("providers", {})
    result = {}

    for name, config in providers.items():
        result[name] = ProviderLimits(
            context_window=config.get("context_window", 128000),
            response_reserve=config.get("response_reserve", 4096),
            supports_extended_context=config.get("supports_extended_context", False),
            rate_limit_rpm=config.get("rate_limit_rpm", 60),
            rate_limit_tpm=config.get("rate_limit_tpm", 100000),
        )

    return result


def get_stage_keywords() -> Dict[str, StageConfig]:
    """Get stage keywords configuration.

    Returns:
        Dictionary mapping stage names to their configuration
    """
    data = _load_yaml_cached(STAGE_KEYWORDS_FILE, "stage_keywords")

    if not data:
        # Return defaults if no config
        return _get_default_stage_keywords()

    stages = data.get("stages", {})
    tool_prefs = data.get("tool_preferences", {})

    result = {}
    for name, config in stages.items():
        result[name] = StageConfig(
            keywords=config.get("keywords", []),
            weight=config.get("weight", 1.0),
            min_score=config.get("min_score", 2),
            tool_preferences=tool_prefs.get(name, []),
        )

    return result


def get_stage_tool_preferences(stage: str) -> List[str]:
    """Get preferred tools for a conversation stage.

    Args:
        stage: Stage name (e.g., 'EXPLORING', 'IMPLEMENTING')

    Returns:
        List of preferred tool names
    """
    data = _load_yaml_cached(STAGE_KEYWORDS_FILE, "stage_keywords")

    if not data:
        return []

    tool_prefs = data.get("tool_preferences", {})
    return tool_prefs.get(stage, [])


def _get_default_stage_keywords() -> Dict[str, StageConfig]:
    """Get default stage keywords (fallback if YAML missing)."""
    return {
        "INITIAL": StageConfig(
            keywords=["help me", "i need", "can you", "please"],
            weight=1.0,
            min_score=2,
        ),
        "EXPLORING": StageConfig(
            keywords=["where", "find", "search", "look for", "locate"],
            weight=1.2,
            min_score=2,
        ),
        "ANALYZING": StageConfig(
            keywords=["why", "explain", "understand", "analyze", "debug"],
            weight=1.3,
            min_score=2,
        ),
        "IMPLEMENTING": StageConfig(
            keywords=["create", "write", "implement", "add", "fix", "update"],
            weight=1.5,
            min_score=2,
        ),
        "REVIEWING": StageConfig(
            keywords=["review", "test", "verify", "commit", "done"],
            weight=1.2,
            min_score=2,
        ),
    }


# =============================================================================
# Logging Configuration
# =============================================================================


def _merge_logging_configs(
    base: Dict[str, Any],
    override: Dict[str, Any],
) -> Dict[str, Any]:
    """Deep merge logging configs (override takes precedence)."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_logging_configs(result[key], value)
        else:
            result[key] = value
    return result


def _apply_env_overrides(config: LoggingConfig) -> LoggingConfig:
    """Apply environment variable overrides to logging config.

    Environment variables:
    - VICTOR_LOG_LEVEL: Override console level
    - VICTOR_LOG_FILE_LEVEL: Override file level
    - VICTOR_LOG_FILE: Override log file path
    - VICTOR_LOG_DISABLED: Disable file logging if "true"
    """
    import os

    if env_level := os.getenv("VICTOR_LOG_LEVEL"):
        config.console_level = env_level.upper()

    if env_file_level := os.getenv("VICTOR_LOG_FILE_LEVEL"):
        config.file_level = env_file_level.upper()

    if env_file := os.getenv("VICTOR_LOG_FILE"):
        config.file_path = env_file

    if os.getenv("VICTOR_LOG_DISABLED", "").lower() == "true":
        config.file_enabled = False

    return config


def get_logging_config(
    command: Optional[str] = None,
    cli_console_level: Optional[str] = None,
    cli_file_level: Optional[str] = None,
    cli_debug_modules: Optional[str] = None,
) -> LoggingConfig:
    """Get logging configuration with priority chain applied.

    Priority (highest to lowest):
    1. CLI arguments (cli_console_level, cli_file_level)
    2. Environment variables (VICTOR_LOG_LEVEL, etc.)
    3. User config file (~/.victor/config.yaml)
    4. Command-specific override from package config
    5. Package defaults (logging_config.yaml)
    6. Hardcoded fallback

    Args:
        command: Command name for command-specific overrides (e.g., "chat", "benchmark")
        cli_console_level: CLI-provided console level (highest priority)
        cli_file_level: CLI-provided file level (highest priority)

    Returns:
        LoggingConfig with all overrides applied
    """
    # Start with hardcoded defaults
    config_dict: Dict[str, Any] = {
        "console_level": "WARNING",
        "file_level": "INFO",
        "file_enabled": True,
        "file_path": "~/.victor/logs/victor.log",
        "file_max_bytes": 10 * 1024 * 1024,
        "file_backup_count": 5,
        "console_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file_format": "%(asctime)s - %(session)s - %(name)s - %(levelname)s - %(message)s",
        "event_logging": True,
        "module_levels": {},
    }

    # Load package defaults
    package_data = _load_yaml_cached(LOGGING_CONFIG_FILE, "logging_config")
    if package_data and "logging" in package_data:
        logging_section = package_data["logging"]

        # Apply package defaults
        if "default" in logging_section:
            for key, value in logging_section["default"].items():
                if key in config_dict:
                    config_dict[key] = value

        # Apply module levels from package
        if "modules" in logging_section:
            config_dict["module_levels"] = logging_section["modules"]

        # Apply command-specific overrides from package
        if command and "commands" in logging_section:
            command_config = logging_section["commands"].get(command, {})
            for key, value in command_config.items():
                if key in config_dict:
                    config_dict[key] = value

    # Load user config (overrides package)
    user_data = _load_yaml_cached(USER_CONFIG_FILE, "user_config")
    if user_data and "logging" in user_data:
        user_logging = user_data["logging"]

        # Apply user defaults
        if "default" in user_logging:
            for key, value in user_logging["default"].items():
                if key in config_dict:
                    config_dict[key] = value

        # Apply user module levels (merge with package)
        if "modules" in user_logging:
            config_dict["module_levels"].update(user_logging["modules"])

        # Apply user command-specific overrides
        if command and "commands" in user_logging:
            command_config = user_logging["commands"].get(command, {})
            for key, value in command_config.items():
                if key in config_dict:
                    config_dict[key] = value

    # Create config object
    config = LoggingConfig(
        console_level=config_dict["console_level"],
        file_level=config_dict["file_level"],
        file_enabled=config_dict["file_enabled"],
        file_path=config_dict["file_path"],
        file_max_bytes=config_dict["file_max_bytes"],
        file_backup_count=config_dict["file_backup_count"],
        console_format=config_dict["console_format"],
        file_format=config_dict["file_format"],
        event_logging=config_dict["event_logging"],
        module_levels=config_dict["module_levels"],
    )

    # Apply environment variable overrides
    config = _apply_env_overrides(config)

    # Apply CLI overrides (highest priority)
    if cli_console_level:
        config.console_level = cli_console_level.upper()
    if cli_file_level:
        config.file_level = cli_file_level.upper()

    # Apply module-specific debug overrides from CLI or environment
    import os

    debug_modules = cli_debug_modules or os.environ.get("VICTOR_DEBUG_MODULES")
    if debug_modules:
        for module in debug_modules.split(","):
            module = module.strip()
            if module:
                # Auto-prefix with "victor." if not already qualified
                full_name = f"victor.{module}" if "." not in module else module
                config.module_levels[full_name] = "DEBUG"

    return config


def get_default_logging_config() -> LoggingConfig:
    """Get default logging config (no command-specific overrides)."""
    return get_logging_config()
