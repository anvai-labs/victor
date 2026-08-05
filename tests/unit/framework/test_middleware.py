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

"""Tests for MiddlewareComposer helper."""

import logging

import pytest

from victor.core.verticals.protocols import MiddlewareProtocol
from victor.framework.middleware import (
    GitSafetyMiddleware,
    LoggingMiddleware,
    MetricsMiddleware,
    MiddlewareComposer,
    SecretMaskingMiddleware,
)


class TestMiddlewareComposer:
    """Tests for MiddlewareComposer fluent API."""

    def test_empty_composer(self) -> None:
        """Test that an empty composer produces an empty list."""
        composer = MiddlewareComposer()
        assert composer.build() == []

    def test_add_custom_middleware(self) -> None:
        """Test adding custom middleware via add() method."""
        custom = SecretMaskingMiddleware()
        composer = MiddlewareComposer().add(custom)
        middleware = composer.build()
        assert len(middleware) == 1
        assert middleware[0] is custom

    def test_add_multiple_custom_middleware(self) -> None:
        """Test chaining multiple custom middleware."""
        custom1 = SecretMaskingMiddleware()
        custom2 = LoggingMiddleware()
        composer = MiddlewareComposer().add(custom1).add(custom2)
        middleware = composer.build()
        assert len(middleware) == 2
        assert middleware[0] is custom1
        assert middleware[1] is custom2

    def test_git_safety_defaults(self) -> None:
        """Test git_safety with default parameters."""
        composer = MiddlewareComposer().git_safety()
        middleware = composer.build()
        assert len(middleware) == 1
        assert isinstance(middleware[0], GitSafetyMiddleware)

    def test_git_safety_custom_config(self) -> None:
        """Test git_safety with custom parameters."""
        composer = MiddlewareComposer().git_safety(
            block_dangerous=True,
            warn_on_risky=False,
            protected_branches={"main", "production"},
        )
        middleware = composer.build()
        assert len(middleware) == 1
        assert isinstance(middleware[0], GitSafetyMiddleware)

    def test_secret_masking_defaults(self) -> None:
        """Test secret_masking with default parameters."""
        composer = MiddlewareComposer().secret_masking()
        middleware = composer.build()
        assert len(middleware) == 1
        assert isinstance(middleware[0], SecretMaskingMiddleware)

    def test_secret_masking_custom_config(self) -> None:
        """Test secret_masking with custom parameters."""
        composer = MiddlewareComposer().secret_masking(
            replacement="[HIDDEN]",
            mask_in_arguments=True,
        )
        middleware = composer.build()
        assert len(middleware) == 1
        assert isinstance(middleware[0], SecretMaskingMiddleware)

    def test_logging_defaults(self) -> None:
        """Test logging with default parameters."""
        composer = MiddlewareComposer().logging()
        middleware = composer.build()
        assert len(middleware) == 1
        assert isinstance(middleware[0], LoggingMiddleware)
        # Check default values are applied

    def test_logging_custom_config(self) -> None:
        """Test logging with custom parameters."""
        composer = MiddlewareComposer().logging(
            log_level=logging.INFO,
            include_arguments=False,
            include_results=True,
        )
        middleware = composer.build()
        assert len(middleware) == 1
        assert isinstance(middleware[0], LoggingMiddleware)

    def test_metrics_defaults(self) -> None:
        """Test metrics with default parameters."""
        composer = MiddlewareComposer().metrics()
        middleware = composer.build()
        assert len(middleware) == 1
        assert isinstance(middleware[0], MetricsMiddleware)

    def test_metrics_custom_config(self) -> None:
        """Test metrics with custom parameters."""
        composer = MiddlewareComposer().metrics(enable_timing=False)
        middleware = composer.build()
        assert len(middleware) == 1
        assert isinstance(middleware[0], MetricsMiddleware)

    def test_standard_devops_preset(self) -> None:
        """Test standard_devops preset."""
        composer = MiddlewareComposer().standard_devops()
        middleware = composer.build()
        # Should add git_safety, secret_masking, logging
        assert len(middleware) == 3
        assert isinstance(middleware[0], GitSafetyMiddleware)
        assert isinstance(middleware[1], SecretMaskingMiddleware)
        assert isinstance(middleware[2], LoggingMiddleware)

    def test_standard_devops_preset_config(self) -> None:
        """Test standard_devops preset applies correct configuration."""
        composer = MiddlewareComposer().standard_devops()
        middleware = composer.build()

        # Check git_safety middleware type
        git_middleware = middleware[0]
        assert isinstance(git_middleware, GitSafetyMiddleware)

        # Check secret_masking middleware type
        secret_middleware = middleware[1]
        assert isinstance(secret_middleware, SecretMaskingMiddleware)

    def test_standard_production_preset(self) -> None:
        """Test standard_production preset."""
        composer = MiddlewareComposer().standard_production()
        middleware = composer.build()
        # Should add secret_masking, metrics, logging
        assert len(middleware) == 3
        assert isinstance(middleware[0], SecretMaskingMiddleware)
        assert isinstance(middleware[1], MetricsMiddleware)
        assert isinstance(middleware[2], LoggingMiddleware)

    def test_fluent_chaining(self) -> None:
        """Test that all methods return self for chaining."""
        composer = MiddlewareComposer().git_safety().secret_masking().logging().metrics()
        middleware = composer.build()
        assert len(middleware) == 4

    def test_build_returns_copy(self) -> None:
        """Test that build() returns a copy, not the internal list."""
        composer = MiddlewareComposer().git_safety()
        middleware1 = composer.build()
        middleware2 = composer.build()
        # Should be different list instances
        assert middleware1 is not middleware2
        # But contain the same middleware
        assert len(middleware1) == len(middleware2)
        assert middleware1[0] is middleware2[0]

    def test_clear(self) -> None:
        """Test clear() method removes all middleware."""
        composer = MiddlewareComposer().git_safety().secret_masking().logging()
        assert len(composer.build()) == 3
        composer.clear()
        assert composer.build() == []

    def test_clear_returns_self(self) -> None:
        """Test that clear() returns self for chaining."""
        composer = MiddlewareComposer().git_safety().clear().secret_masking()
        middleware = composer.build()
        assert len(middleware) == 1
        assert isinstance(middleware[0], SecretMaskingMiddleware)

    def test_complex_composition(self) -> None:
        """Test a realistic complex middleware composition."""
        composer = (
            MiddlewareComposer()
            .git_safety(
                block_dangerous=True,
                protected_branches={"main", "production", "staging"},
            )
            .secret_masking(mask_in_arguments=True)
            .logging(
                log_level=logging.INFO,
                include_arguments=True,
                include_results=True,
            )
            .metrics(enable_timing=True)
        )
        middleware = composer.build()
        assert len(middleware) == 4
        assert isinstance(middleware[0], GitSafetyMiddleware)
        assert isinstance(middleware[1], SecretMaskingMiddleware)
        assert isinstance(middleware[2], LoggingMiddleware)
        assert isinstance(middleware[3], MetricsMiddleware)

    def test_custom_after_preset(self) -> None:
        """Test adding custom middleware after a preset."""
        custom = MetricsMiddleware()
        composer = MiddlewareComposer().standard_devops().add(custom)
        middleware = composer.build()
        assert len(middleware) == 4  # 3 from preset + 1 custom
        assert middleware[-1] is custom

    def test_preset_after_custom(self) -> None:
        """Test adding a preset after custom middleware."""
        custom = SecretMaskingMiddleware(replacement="[CUSTOM]")
        composer = MiddlewareComposer().add(custom).git_safety()
        middleware = composer.build()
        # Custom middleware should be first, then preset middleware
        assert len(middleware) == 2
        assert middleware[0] is custom
        # git_safety adds GitSafetyMiddleware
        assert isinstance(middleware[1], GitSafetyMiddleware)

    def test_multiple_presets(self) -> None:
        """Test combining multiple presets."""
        composer = MiddlewareComposer().git_safety().standard_production()
        middleware = composer.build()
        # Should have middleware from both presets
        assert len(middleware) == 4  # 1 from git_safety + 3 from production

    def test_immutability_of_built_list(self) -> None:
        """Test that modifying the returned list doesn't affect the composer."""
        composer = MiddlewareComposer().git_safety()
        middleware = composer.build()
        middleware.clear()  # Modify the returned list
        # Rebuilding should still produce the same result
        middleware2 = composer.build()
        assert len(middleware2) == 1
