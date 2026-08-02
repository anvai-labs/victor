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

"""Performance settings for victor-coding.

This module extends the base settings with performance-related configuration
for controlling native backend behavior and preferences.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from victor_coding.performance.registry import BackendConfig

logger = logging.getLogger(__name__)


@dataclass
class PerformanceSettings:
    """Settings controlling performance backend behavior.

    These settings can be configured via:
    1. Environment variables (VICTOR_USE_NATIVE_BACKENDS=1)
    2. Settings file (.victor/settings.toml or ~/.victor/profiles.yaml)
    3. Programmatic configuration
    """

    # Enable/disable native backends globally
    use_native_backends: bool = True

    # Prefer native implementations for specific components
    prefer_native_indexer: bool = True
    prefer_native_chunker: bool = True
    prefer_native_extractor: bool = True
    prefer_native_regex: bool = True

    # Priority for native backends (0-100, higher = preferred)
    native_backend_priority: int = 80

    # Fallback behavior
    fallback_to_python: bool = True

    # Parallel processing settings
    max_parallel_workers: Optional[int] = None  # None = auto-detect
    enable_parallel_indexing: bool = True
    enable_parallel_chunking: bool = True

    # Memory limits (MB)
    max_memory_per_worker_mb: int = 512
    max_total_memory_mb: int = 4096

    # Cache settings
    enable_parser_caching: bool = True
    enable_query_caching: bool = True
    cache_size_mb: int = 100

    # Logging and diagnostics
    log_backend_selection: bool = True
    log_performance_metrics: bool = False

    @classmethod
    def from_settings(cls, settings: Any) -> "PerformanceSettings":
        """Create PerformanceSettings from Victor settings object.

        Args:
            settings: Victor settings object (dict-like or SettingsProvider)

        Returns:
            PerformanceSettings instance
        """
        # Extract settings with defaults
        return cls(
            use_native_backends=getattr(settings, "use_native_backends", True),
            prefer_native_indexer=getattr(settings, "prefer_native_indexer", True),
            prefer_native_chunker=getattr(settings, "prefer_native_chunker", True),
            prefer_native_extractor=getattr(settings, "prefer_native_extractor", True),
            prefer_native_regex=getattr(settings, "prefer_native_regex", True),
            native_backend_priority=getattr(settings, "native_backend_priority", 80),
            fallback_to_python=getattr(settings, "fallback_to_python", True),
            max_parallel_workers=getattr(settings, "max_parallel_workers", None),
            enable_parallel_indexing=getattr(settings, "enable_parallel_indexing", True),
            enable_parallel_chunking=getattr(settings, "enable_parallel_chunking", True),
            max_memory_per_worker_mb=getattr(settings, "max_memory_per_worker_mb", 512),
            max_total_memory_mb=getattr(settings, "max_total_memory_mb", 4096),
            enable_parser_caching=getattr(settings, "enable_parser_caching", True),
            enable_query_caching=getattr(settings, "enable_query_caching", True),
            cache_size_mb=getattr(settings, "cache_size_mb", 100),
            log_backend_selection=getattr(settings, "log_backend_selection", True),
            log_performance_metrics=getattr(settings, "log_performance_metrics", False),
        )

    @classmethod
    def from_environment(cls) -> "PerformanceSettings":
        """Create PerformanceSettings from environment variables.

        Environment variables:
        - VICTOR_USE_NATIVE_BACKENDS: 0/1
        - VICTOR_NATIVE_BACKEND_PRIORITY: 0-100
        - VICTOR_MAX_PARALLEL_WORKERS: number
        - VICTOR_LOG_PERFORMANCE: 0/1

        Returns:
            PerformanceSettings instance
        """
        import os

        return cls(
            use_native_backends=os.getenv("VICTOR_USE_NATIVE_BACKENDS", "1") == "1",
            native_backend_priority=int(os.getenv("VICTOR_NATIVE_BACKEND_PRIORITY", "80")),
            max_parallel_workers=int(os.getenv("VICTOR_MAX_PARALLEL_WORKERS", "0")) or None,
            log_performance_metrics=os.getenv("VICTOR_LOG_PERFORMANCE", "0") == "1",
        )

    @classmethod
    def load(cls, settings: Optional[Any] = None) -> "PerformanceSettings":
        """Load performance settings from multiple sources.

        Resolution order:
        1. Environment variables
        2. Provided settings object
        3. Default values

        Args:
            settings: Optional Victor settings object

        Returns:
            PerformanceSettings instance
        """
        # Start with environment defaults
        perf_settings = cls.from_environment()

        # Override with settings object if provided
        if settings is not None:
            provided = cls.from_settings(settings)
            for key, value in provided.__dict__.items():
                # Only override if not already set from env
                if getattr(perf_settings, key) == cls.__dict__.get(key):
                    setattr(perf_settings, key, value)

        return perf_settings

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary.

        Returns:
            Dictionary representation of settings
        """
        return {
            "use_native_backends": self.use_native_backends,
            "prefer_native_indexer": self.prefer_native_indexer,
            "prefer_native_chunker": self.prefer_native_chunker,
            "prefer_native_extractor": self.prefer_native_extractor,
            "prefer_native_regex": self.prefer_native_regex,
            "native_backend_priority": self.native_backend_priority,
            "fallback_to_python": self.fallback_to_python,
            "max_parallel_workers": self.max_parallel_workers,
            "enable_parallel_indexing": self.enable_parallel_indexing,
            "enable_parallel_chunking": self.enable_parallel_chunking,
            "max_memory_per_worker_mb": self.max_memory_per_worker_mb,
            "max_total_memory_mb": self.max_total_memory_mb,
            "enable_parser_caching": self.enable_parser_caching,
            "enable_query_caching": self.enable_query_caching,
            "cache_size_mb": self.cache_size_mb,
            "log_backend_selection": self.log_backend_selection,
            "log_performance_metrics": self.log_performance_metrics,
        }

    def get_backend_config(self) -> "BackendConfig":
        """Get BackendConfig for use with PerformanceBackendRegistry.

        Returns:
            BackendConfig instance
        """
        from victor_coding.performance.registry import BackendConfig

        return BackendConfig(
            use_native=self.use_native_backends,
            prefer_parallel=True,
            fallback_to_python=self.fallback_to_python,
            min_priority=0,
            max_memory_mb=self.max_total_memory_mb,
        )


# Global settings cache
_performance_settings_cache: Optional[PerformanceSettings] = None


def get_performance_settings(
    settings: Optional[Any] = None, reload: bool = False
) -> PerformanceSettings:
    """Get cached performance settings, loading if necessary.

    Args:
        settings: Optional Victor settings object
        reload: Force reload even if cached

    Returns:
        PerformanceSettings instance
    """
    global _performance_settings_cache

    if _performance_settings_cache is None or reload:
        _performance_settings_cache = PerformanceSettings.load(settings)

    return _performance_settings_cache


def clear_performance_settings_cache() -> None:
    """Clear the performance settings cache."""
    global _performance_settings_cache
    _performance_settings_cache = None


__all__ = [
    "PerformanceSettings",
    "get_performance_settings",
    "clear_performance_settings_cache",
]
