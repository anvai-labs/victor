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

"""Framework-level middleware default profiles.

This module provides pre-configured middleware profiles that can be used
across all verticals to reduce duplication and ensure consistency.

Design Pattern: Strategy + Factory
- Pre-defined profiles for common use cases
- Verticals can extend/override profiles
- Reduces middleware configuration duplication

Example:
    from victor.framework.middleware_profiles import MiddlewareProfiles

    # Use default profile
    profile = MiddlewareProfiles.default_profile()

    # Or use safety-first profile
    profile = MiddlewareProfiles.safety_first_profile()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import victor.framework.middleware as middleware_module

# =============================================================================
# Middleware Profile
# =============================================================================


@dataclass(frozen=True)
class MiddlewareProfile:
    """A middleware profile with pre-configured middleware list.

    Attributes:
        name: Profile name
        description: Human-readable description
        middlewares: List of middleware instances
        priority: Execution priority
    """

    name: str
    description: str
    middlewares: List[Any] = field(default_factory=list)
    priority: int = 50


# =============================================================================
# Middleware Profiles
# =============================================================================


class MiddlewareProfiles:
    """Pre-defined middleware profiles for common use cases.

    Profiles:
        - DEFAULT_PROFILE: Basic logging
        - SAFETY_FIRST_PROFILE: Git safety, secret masking
        - DEVELOPMENT_PROFILE: Permissive git with logging
        - PRODUCTION_PROFILE: Strict safety, secrets masking
        - ANALYSIS_PROFILE: Read-only, minimal logging
        - CI_CD_PROFILE: CI/CD optimized with deployment safety
    """

    @staticmethod
    def default_profile() -> MiddlewareProfile:
        """Default profile with basic logging.

        Returns:
            MiddlewareProfile with basic middleware
        """
        return MiddlewareProfile(
            name="default",
            description="Basic logging",
            middlewares=[
                middleware_module.LoggingMiddleware(
                    log_level=logging.DEBUG,
                    include_arguments=True,
                    sanitize_arguments=True,
                ),
            ],
            priority=50,
        )

    @staticmethod
    def safety_first_profile() -> MiddlewareProfile:
        """Safety-first profile with git safety and secret masking.

        Returns:
            MiddlewareProfile with safety middleware
        """
        return MiddlewareProfile(
            name="safety_first",
            description="Git safety and secret masking",
            middlewares=[
                middleware_module.GitSafetyMiddleware(
                    block_dangerous=True,
                    warn_on_risky=True,
                    protected_branches={"production", "staging", "main"},
                ),
                middleware_module.SecretMaskingMiddleware(
                    replacement="[REDACTED]",
                    mask_in_arguments=True,
                ),
                middleware_module.LoggingMiddleware(
                    log_level=logging.DEBUG,
                    include_arguments=True,
                    sanitize_arguments=True,
                ),
            ],
            priority=25,  # HIGH priority for safety
        )

    @staticmethod
    def development_profile() -> MiddlewareProfile:
        """Development profile with permissive git and detailed logging.

        Returns:
            MiddlewareProfile optimized for development
        """
        return MiddlewareProfile(
            name="development",
            description="Permissive git with detailed logging",
            middlewares=[
                middleware_module.GitSafetyMiddleware(
                    block_dangerous=False,
                    warn_on_risky=True,
                    protected_branches=set(),  # No protected branches in dev
                ),
                middleware_module.LoggingMiddleware(
                    log_level=logging.DEBUG,
                    include_arguments=True,
                    sanitize_arguments=False,  # Show everything in dev
                    include_results=True,
                ),
            ],
            priority=75,  # LOW priority for development
        )

    @staticmethod
    def production_profile() -> MiddlewareProfile:
        """Production profile with strict safety and secrets masking.

        Returns:
            MiddlewareProfile optimized for production
        """
        return MiddlewareProfile(
            name="production",
            description="Strict safety and secrets masking",
            middlewares=[
                middleware_module.GitSafetyMiddleware(
                    block_dangerous=True,
                    warn_on_risky=True,
                    protected_branches={"production", "staging", "main", "release"},
                ),
                middleware_module.SecretMaskingMiddleware(
                    replacement="[REDACTED]",
                    mask_in_arguments=True,
                ),
                middleware_module.LoggingMiddleware(
                    log_level=logging.INFO,  # Only log INFO and above
                    include_arguments=True,
                    sanitize_arguments=True,
                ),
            ],
            priority=0,  # CRITICAL priority for production
        )

    @staticmethod
    def analysis_profile() -> MiddlewareProfile:
        """Analysis profile with read-only operations (no modifications).

        Returns:
            MiddlewareProfile for analysis tasks
        """
        return MiddlewareProfile(
            name="analysis",
            description="Read-only analysis with minimal logging",
            middlewares=[
                middleware_module.LoggingMiddleware(
                    log_level=logging.WARNING,  # Only log warnings and errors
                    include_arguments=False,  # Don't log arguments for privacy
                    sanitize_arguments=True,
                ),
            ],
            priority=50,
        )

    @staticmethod
    def ci_cd_profile() -> MiddlewareProfile:
        """CI/CD optimized profile with deployment safety.

        Returns:
            MiddlewareProfile for CI/CD pipelines
        """
        return MiddlewareProfile(
            name="ci_cd",
            description="CI/CD with deployment safety",
            middlewares=[
                middleware_module.GitSafetyMiddleware(
                    block_dangerous=True,
                    warn_on_risky=False,  # No warnings in CI/CD
                    protected_branches={"main", "release", "production"},
                ),
                middleware_module.SecretMaskingMiddleware(
                    replacement="***",  # Shorter replacement for logs
                    mask_in_arguments=True,
                ),
                middleware_module.LoggingMiddleware(
                    log_level=logging.DEBUG,  # Detailed logging for CI/CD
                    include_arguments=True,
                    sanitize_arguments=True,
                ),
            ],
            priority=10,
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def get_profile(name: str) -> Optional[MiddlewareProfile]:
    """Get a middleware profile by name.

    Args:
        name: Profile name (default, safety_first, development, production, analysis, ci_cd)

    Returns:
        MiddlewareProfile or None if not found
    """
    profiles = {
        "default": MiddlewareProfiles.default_profile(),
        "safety_first": MiddlewareProfiles.safety_first_profile(),
        "development": MiddlewareProfiles.development_profile(),
        "production": MiddlewareProfiles.production_profile(),
        "analysis": MiddlewareProfiles.analysis_profile(),
        "ci_cd": MiddlewareProfiles.ci_cd_profile(),
    }
    return profiles.get(name)


def list_profiles() -> List[str]:
    """List all available profile names.

    Returns:
        List of profile names
    """
    return ["default", "safety_first", "development", "production", "analysis", "ci_cd"]


__all__ = [
    "MiddlewareProfile",
    "MiddlewareProfiles",
    "get_profile",
    "list_profiles",
]
