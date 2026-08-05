# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for WorkflowEngineConfig validation — Wave O."""

import pytest

from victor.framework.workflow_engine import WorkflowEngineConfig


class TestWorkflowEngineConfigValidation:
    """WorkflowEngineConfig numeric fields must have valid ranges."""

    def test_default_config_passes_validation(self):
        """Default WorkflowEngineConfig should pass all validations."""
        config = WorkflowEngineConfig()
        assert config.max_iterations == 100
        assert config.cache_ttl_seconds == 3600
        assert config.hitl_timeout_seconds == 300

    def test_max_iterations_below_one_raises(self):
        """max_iterations must be >= 1."""
        with pytest.raises(ValueError, match="max_iterations must be >= 1"):
            WorkflowEngineConfig(max_iterations=0)

        with pytest.raises(ValueError, match="max_iterations must be >= 1"):
            WorkflowEngineConfig(max_iterations=-10)

    def test_cache_ttl_seconds_negative_raises(self):
        """cache_ttl_seconds must be >= 0."""
        with pytest.raises(ValueError, match="cache_ttl_seconds must be >= 0"):
            WorkflowEngineConfig(cache_ttl_seconds=-1)

    def test_hitl_timeout_seconds_negative_raises(self):
        """hitl_timeout_seconds must be >= 0."""
        with pytest.raises(ValueError, match="hitl_timeout_seconds must be >= 0"):
            WorkflowEngineConfig(hitl_timeout_seconds=-1)

    def test_boundary_values_pass(self):
        """Boundary values (1, 0) should pass validation."""
        config = WorkflowEngineConfig(
            max_iterations=1,
            cache_ttl_seconds=0,
            hitl_timeout_seconds=0,
        )
        assert config.max_iterations == 1
        assert config.cache_ttl_seconds == 0
        assert config.hitl_timeout_seconds == 0

    def test_large_values_pass(self):
        """Large reasonable values should pass validation."""
        config = WorkflowEngineConfig(
            max_iterations=1000000,
            cache_ttl_seconds=864000,  # 10 days
            hitl_timeout_seconds=3600,  # 1 hour
        )
        assert config.max_iterations == 1000000
        assert config.cache_ttl_seconds == 864000
        assert config.hitl_timeout_seconds == 3600
