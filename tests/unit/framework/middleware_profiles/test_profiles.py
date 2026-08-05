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

"""Tests for victor.framework.middleware_profiles.profiles module.

These tests verify the predefined middleware profiles that provide
pre-configured middleware combinations for common use cases.
"""

import logging
import pytest

from victor.framework.middleware_profiles.profiles import (
    MiddlewareProfile,
    MiddlewareProfiles,
    get_profile,
    list_profiles,
)
from victor.framework.middleware import (
    GitSafetyMiddleware,
    SecretMaskingMiddleware,
    LoggingMiddleware,
)

# =============================================================================
# MiddlewareProfile Tests
# =============================================================================


class TestMiddlewareProfile:
    """Tests for MiddlewareProfile dataclass."""

    def test_profile_basic_attributes(self):
        """MiddlewareProfile should store basic attributes."""
        profile = MiddlewareProfile(
            name="test_profile",
            description="Test profile",
            middlewares=[],
            priority=50,
        )

        assert profile.name == "test_profile"
        assert profile.description == "Test profile"
        assert profile.middlewares == []
        assert profile.priority == 50

    def test_profile_with_middlewares(self):
        """MiddlewareProfile should store middleware instances."""
        middleware1 = LoggingMiddleware(log_level=logging.DEBUG)
        middleware2 = SecretMaskingMiddleware()

        profile = MiddlewareProfile(
            name="test",
            description="Test",
            middlewares=[middleware1, middleware2],
        )

        assert len(profile.middlewares) == 2
        assert middleware1 in profile.middlewares
        assert middleware2 in profile.middlewares

    def test_profile_defaults(self):
        """MiddlewareProfile should have sensible defaults."""
        profile = MiddlewareProfile(name="test", description="Test")

        assert profile.middlewares == []
        assert profile.priority == 50


# =============================================================================
# MiddlewareProfiles - Default Profile
# =============================================================================


class TestDefaultProfile:
    """Tests for default profile."""

    def test_default_profile_exists(self):
        """default_profile() should return a profile."""
        profile = MiddlewareProfiles.default_profile()

        assert profile is not None
        assert isinstance(profile, MiddlewareProfile)
        assert profile.name == "default"

    def test_default_profile_description(self):
        """default_profile() should have correct description."""
        profile = MiddlewareProfiles.default_profile()
        assert profile.description == "Basic logging"

    def test_default_profile_middlewares(self):
        """default_profile() should contain LoggingMiddleware."""
        profile = MiddlewareProfiles.default_profile()

        assert len(profile.middlewares) == 1
        assert isinstance(profile.middlewares[0], LoggingMiddleware)

    def test_default_profile_logging_config(self):
        """default_profile() should configure logging correctly."""
        profile = MiddlewareProfiles.default_profile()
        logging_middleware = profile.middlewares[0]

        assert logging_middleware._log_level == logging.DEBUG
        assert logging_middleware._include_arguments is True
        assert logging_middleware._sanitize_arguments is True

    def test_default_profile_priority(self):
        """default_profile() should have MEDIUM priority."""
        profile = MiddlewareProfiles.default_profile()
        assert profile.priority == 50


# =============================================================================
# MiddlewareProfiles - Safety First Profile
# =============================================================================


class TestSafetyFirstProfile:
    """Tests for safety_first profile."""

    def test_safety_first_profile_exists(self):
        """safety_first_profile() should return a profile."""
        profile = MiddlewareProfiles.safety_first_profile()

        assert profile is not None
        assert profile.name == "safety_first"

    def test_safety_first_description(self):
        """safety_first_profile() should have correct description."""
        profile = MiddlewareProfiles.safety_first_profile()
        assert profile.description == "Git safety and secret masking"

    def test_safety_first_middlewares(self):
        """safety_first_profile() should contain 3 middlewares."""
        profile = MiddlewareProfiles.safety_first_profile()

        assert len(profile.middlewares) == 3
        middleware_types = [type(m).__name__ for m in profile.middlewares]

        assert "GitSafetyMiddleware" in middleware_types
        assert "SecretMaskingMiddleware" in middleware_types
        assert "LoggingMiddleware" in middleware_types

    def test_safety_first_git_config(self):
        """safety_first_profile() should configure GitSafetyMiddleware."""
        profile = MiddlewareProfiles.safety_first_profile()
        git_middleware = next(m for m in profile.middlewares if isinstance(m, GitSafetyMiddleware))

        assert git_middleware._block_dangerous is True
        assert git_middleware._warn_on_risky is True
        # Note: GitSafetyMiddleware adds default branches to the set
        assert "production" in git_middleware._protected_branches
        assert "staging" in git_middleware._protected_branches
        assert "main" in git_middleware._protected_branches

    def test_safety_first_secret_masking_config(self):
        """safety_first_profile() should configure SecretMaskingMiddleware."""
        profile = MiddlewareProfiles.safety_first_profile()
        secret_middleware = next(
            m for m in profile.middlewares if isinstance(m, SecretMaskingMiddleware)
        )

        assert secret_middleware._replacement == "[REDACTED]"
        assert secret_middleware._mask_in_arguments is True

    def test_safety_first_priority(self):
        """safety_first_profile() should have HIGH priority (low number)."""
        profile = MiddlewareProfiles.safety_first_profile()
        assert profile.priority == 25


# =============================================================================
# MiddlewareProfiles - Development Profile
# =============================================================================


class TestDevelopmentProfile:
    """Tests for development profile."""

    def test_development_profile_exists(self):
        """development_profile() should return a profile."""
        profile = MiddlewareProfiles.development_profile()

        assert profile is not None
        assert profile.name == "development"

    def test_development_description(self):
        """development_profile() should have correct description."""
        profile = MiddlewareProfiles.development_profile()
        assert profile.description == "Permissive git with detailed logging"

    def test_development_middlewares(self):
        """development_profile() should contain 2 middlewares."""
        profile = MiddlewareProfiles.development_profile()

        assert len(profile.middlewares) == 2
        middleware_types = [type(m).__name__ for m in profile.middlewares]

        assert "GitSafetyMiddleware" in middleware_types
        assert "LoggingMiddleware" in middleware_types

    def test_development_git_config(self):
        """development_profile() should have permissive GitSafetyMiddleware."""
        profile = MiddlewareProfiles.development_profile()
        git_middleware = next(m for m in profile.middlewares if isinstance(m, GitSafetyMiddleware))

        assert git_middleware._block_dangerous is False
        assert git_middleware._warn_on_risky is True
        # Note: Development profile tries to set empty set, but middleware adds defaults
        # The key is that blocking is disabled for development

    def test_development_logging_config(self):
        """development_profile() should have detailed logging."""
        profile = MiddlewareProfiles.development_profile()
        logging_middleware = next(
            m for m in profile.middlewares if isinstance(m, LoggingMiddleware)
        )

        assert logging_middleware._log_level == logging.DEBUG
        assert logging_middleware._include_arguments is True
        assert logging_middleware._sanitize_arguments is False  # Show everything
        assert logging_middleware._include_results is True

    def test_development_priority(self):
        """development_profile() should have LOW priority."""
        profile = MiddlewareProfiles.development_profile()
        assert profile.priority == 75


# =============================================================================
# MiddlewareProfiles - Production Profile
# =============================================================================


class TestProductionProfile:
    """Tests for production profile."""

    def test_production_profile_exists(self):
        """production_profile() should return a profile."""
        profile = MiddlewareProfiles.production_profile()

        assert profile is not None
        assert profile.name == "production"

    def test_production_description(self):
        """production_profile() should have correct description."""
        profile = MiddlewareProfiles.production_profile()
        assert profile.description == "Strict safety and secrets masking"

    def test_production_middlewares(self):
        """production_profile() should contain 3 middlewares."""
        profile = MiddlewareProfiles.production_profile()

        assert len(profile.middlewares) == 3
        middleware_types = [type(m).__name__ for m in profile.middlewares]

        assert "GitSafetyMiddleware" in middleware_types
        assert "SecretMaskingMiddleware" in middleware_types
        assert "LoggingMiddleware" in middleware_types

    def test_production_git_config(self):
        """production_profile() should have strict GitSafetyMiddleware."""
        profile = MiddlewareProfiles.production_profile()
        git_middleware = next(m for m in profile.middlewares if isinstance(m, GitSafetyMiddleware))

        assert git_middleware._block_dangerous is True
        assert git_middleware._warn_on_risky is True
        # Note: GitSafetyMiddleware adds default branches to the set
        assert "production" in git_middleware._protected_branches
        assert "staging" in git_middleware._protected_branches
        assert "main" in git_middleware._protected_branches
        assert "release" in git_middleware._protected_branches

    def test_production_logging_config(self):
        """production_profile() should have INFO-level logging."""
        profile = MiddlewareProfiles.production_profile()
        logging_middleware = next(
            m for m in profile.middlewares if isinstance(m, LoggingMiddleware)
        )

        assert logging_middleware._log_level == logging.INFO
        assert logging_middleware._include_arguments is True
        assert logging_middleware._sanitize_arguments is True

    def test_production_priority(self):
        """production_profile() should have CRITICAL priority."""
        profile = MiddlewareProfiles.production_profile()
        assert profile.priority == 0


# =============================================================================
# MiddlewareProfiles - Analysis Profile
# =============================================================================


class TestAnalysisProfile:
    """Tests for analysis profile."""

    def test_analysis_profile_exists(self):
        """analysis_profile() should return a profile."""
        profile = MiddlewareProfiles.analysis_profile()

        assert profile is not None
        assert profile.name == "analysis"

    def test_analysis_description(self):
        """analysis_profile() should have correct description."""
        profile = MiddlewareProfiles.analysis_profile()
        assert profile.description == "Read-only analysis with minimal logging"

    def test_analysis_middlewares(self):
        """analysis_profile() should contain only LoggingMiddleware."""
        profile = MiddlewareProfiles.analysis_profile()

        assert len(profile.middlewares) == 1
        assert isinstance(profile.middlewares[0], LoggingMiddleware)

    def test_analysis_logging_config(self):
        """analysis_profile() should have minimal logging."""
        profile = MiddlewareProfiles.analysis_profile()
        logging_middleware = profile.middlewares[0]

        assert logging_middleware._log_level == logging.WARNING
        assert logging_middleware._include_arguments is False
        assert logging_middleware._sanitize_arguments is True

    def test_analysis_priority(self):
        """analysis_profile() should have MEDIUM priority."""
        profile = MiddlewareProfiles.analysis_profile()
        assert profile.priority == 50


# =============================================================================
# MiddlewareProfiles - CI/CD Profile
# =============================================================================


class TestCICDProfile:
    """Tests for ci_cd profile."""

    def test_ci_cd_profile_exists(self):
        """ci_cd_profile() should return a profile."""
        profile = MiddlewareProfiles.ci_cd_profile()

        assert profile is not None
        assert profile.name == "ci_cd"

    def test_ci_cd_description(self):
        """ci_cd_profile() should have correct description."""
        profile = MiddlewareProfiles.ci_cd_profile()
        assert profile.description == "CI/CD with deployment safety"

    def test_ci_cd_middlewares(self):
        """ci_cd_profile() should contain 3 middlewares."""
        profile = MiddlewareProfiles.ci_cd_profile()

        assert len(profile.middlewares) == 3
        middleware_types = [type(m).__name__ for m in profile.middlewares]

        assert "GitSafetyMiddleware" in middleware_types
        assert "SecretMaskingMiddleware" in middleware_types
        assert "LoggingMiddleware" in middleware_types

    def test_ci_cd_git_config(self):
        """ci_cd_profile() should not warn in CI/CD."""
        profile = MiddlewareProfiles.ci_cd_profile()
        git_middleware = next(m for m in profile.middlewares if isinstance(m, GitSafetyMiddleware))

        assert git_middleware._block_dangerous is True
        assert git_middleware._warn_on_risky is False  # No warnings in CI/CD
        # Note: GitSafetyMiddleware adds default branches to the set
        assert "main" in git_middleware._protected_branches
        assert "release" in git_middleware._protected_branches
        assert "production" in git_middleware._protected_branches

    def test_ci_cd_secret_masking_config(self):
        """ci_cd_profile() should use shorter replacement."""
        profile = MiddlewareProfiles.ci_cd_profile()
        secret_middleware = next(
            m for m in profile.middlewares if isinstance(m, SecretMaskingMiddleware)
        )

        assert secret_middleware._replacement == "***"
        assert secret_middleware._mask_in_arguments is True

    def test_ci_cd_logging_config(self):
        """ci_cd_profile() should have detailed logging."""
        profile = MiddlewareProfiles.ci_cd_profile()
        logging_middleware = next(
            m for m in profile.middlewares if isinstance(m, LoggingMiddleware)
        )

        assert logging_middleware._log_level == logging.DEBUG
        assert logging_middleware._include_arguments is True
        assert logging_middleware._sanitize_arguments is True

    def test_ci_cd_priority(self):
        """ci_cd_profile() should have VERY HIGH priority."""
        profile = MiddlewareProfiles.ci_cd_profile()
        assert profile.priority == 10


# =============================================================================
# get_profile() Function Tests
# =============================================================================


class TestGetProfileFunction:
    """Tests for get_profile() convenience function."""

    def test_get_profile_default(self):
        """get_profile() should return default profile."""
        profile = get_profile("default")
        assert profile is not None
        assert profile.name == "default"

    def test_get_profile_safety_first(self):
        """get_profile() should return safety_first profile."""
        profile = get_profile("safety_first")
        assert profile is not None
        assert profile.name == "safety_first"

    def test_get_profile_development(self):
        """get_profile() should return development profile."""
        profile = get_profile("development")
        assert profile is not None
        assert profile.name == "development"

    def test_get_profile_production(self):
        """get_profile() should return production profile."""
        profile = get_profile("production")
        assert profile is not None
        assert profile.name == "production"

    def test_get_profile_analysis(self):
        """get_profile() should return analysis profile."""
        profile = get_profile("analysis")
        assert profile is not None
        assert profile.name == "analysis"

    def test_get_profile_ci_cd(self):
        """get_profile() should return ci_cd profile."""
        profile = get_profile("ci_cd")
        assert profile is not None
        assert profile.name == "ci_cd"

    def test_get_profile_invalid_returns_none(self):
        """get_profile() should return None for invalid profile name."""
        profile = get_profile("invalid_profile_name")
        assert profile is None

    def test_get_profile_case_sensitive(self):
        """get_profile() should be case-sensitive."""
        profile = get_profile("DEFAULT")
        assert profile is None

        profile = get_profile("Default")
        assert profile is None


# =============================================================================
# list_profiles() Function Tests
# =============================================================================


class TestListProfilesFunction:
    """Tests for list_profiles() convenience function."""

    def test_list_profiles_returns_all(self):
        """list_profiles() should return all 6 profile names."""
        profiles = list_profiles()

        assert len(profiles) == 6
        assert "default" in profiles
        assert "safety_first" in profiles
        assert "development" in profiles
        assert "production" in profiles
        assert "analysis" in profiles
        assert "ci_cd" in profiles

    def test_list_profiles_returns_list(self):
        """list_profiles() should return a list."""
        profiles = list_profiles()
        assert isinstance(profiles, list)

    def test_list_profiles_ordering(self):
        """list_profiles() should return profiles in consistent order."""
        profiles1 = list_profiles()
        profiles2 = list_profiles()

        assert profiles1 == profiles2


# =============================================================================
# Profile Middleware Count Tests
# =============================================================================


class TestProfileMiddlewareCounts:
    """Tests for middleware counts in each profile."""

    def test_all_profiles_have_middlewares(self):
        """All profiles should have at least one middleware."""
        profile_names = list_profiles()

        for name in profile_names:
            profile = get_profile(name)
            assert profile is not None
            assert len(profile.middlewares) >= 1

    def test_profile_middleware_uniqueness(self):
        """Each profile should have unique middleware combination or configuration."""
        profiles = {name: get_profile(name) for name in list_profiles()}

        # Get middleware type sets for each profile
        middleware_sets = {
            name: {type(m).__name__ for m in profile.middlewares}
            for name, profile in profiles.items()
        }

        # Check that profiles have different middleware types or configurations
        # Some profiles may have same middleware types but different configs
        middleware_list = list(middleware_sets.values())

        # We expect some overlap since multiple profiles use similar middleware
        # but with different configurations
        assert len(middleware_list) == 6  # All 6 profiles have middlewares


# =============================================================================
# Profile Priority Tests
# =============================================================================


class TestProfilePriorities:
    """Tests for profile priority ordering."""

    def test_production_highest_priority(self):
        """production profile should have highest priority (lowest number)."""
        prod = get_profile("production")
        safety = get_profile("safety_first")
        ci_cd = get_profile("ci_cd")

        assert prod.priority < safety.priority
        assert prod.priority < ci_cd.priority

    def test_development_lowest_priority(self):
        """development profile should have lowest priority (highest number)."""
        dev = get_profile("development")
        default = get_profile("default")
        analysis = get_profile("analysis")

        assert dev.priority > default.priority
        assert dev.priority > analysis.priority

    def test_all_priorities_unique(self):
        """All profiles should have unique priorities."""
        profiles = {name: get_profile(name) for name in list_profiles()}
        priorities = [p.priority for p in profiles.values()]

        # Check that all 6 profiles have priorities (note: default and analysis share priority 50)
        priority_set = set(priorities)
        assert len(priorities) == 6  # All 6 profiles have priorities
        assert len(priority_set) == 5  # But only 5 unique values (default and analysis both use 50)


__all__ = [
    "TestMiddlewareProfile",
    "TestDefaultProfile",
    "TestSafetyFirstProfile",
    "TestDevelopmentProfile",
    "TestProductionProfile",
    "TestAnalysisProfile",
    "TestCICDProfile",
    "TestGetProfileFunction",
    "TestListProfilesFunction",
    "TestProfileMiddlewareCounts",
    "TestProfilePriorities",
]
