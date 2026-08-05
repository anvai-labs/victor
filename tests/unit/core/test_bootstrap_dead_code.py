# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0

"""Tests verifying dead code cleanup in bootstrap.py."""

from __future__ import annotations

import inspect


class TestBootstrapDeadCodeCleanup:
    """Verify _register_coding_services was renamed to _register_language_services."""

    def test_register_coding_services_removed(self):
        """Old coding-specific name should no longer exist."""
        from victor.core import bootstrap

        assert not hasattr(
            bootstrap, "_register_coding_services"
        ), "_register_coding_services should be renamed to _register_language_services"

    def test_register_language_services_exists(self):
        """Generalized name should exist as a callable."""
        from victor.core import bootstrap

        assert hasattr(bootstrap, "_register_language_services")
        assert callable(bootstrap._register_language_services)

    def test_register_language_services_not_in_bootstrap_phases(self):
        """Function is available but not wired into bootstrap phases DAG."""
        from victor.core.bootstrap import _BOOTSTRAP_PHASES

        phase_names = {p.name for p in _BOOTSTRAP_PHASES}
        # It should not be a dedicated phase — it's called via entry points
        assert "coding_services" not in phase_names
        assert "language_services" not in phase_names

    def test_register_language_services_uses_protocol(self):
        """Function should use LanguageRegistryProtocol, not hardcoded imports."""
        from victor.core.bootstrap import _register_language_services

        source = inspect.getsource(_register_language_services)
        assert "LanguageRegistryProtocol" in source
        # Should not import directly from any vertical package
        assert "victor_coding" not in source
        assert "victor.verticals.contrib" not in source
