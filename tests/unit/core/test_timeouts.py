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

"""Tests for the centralized timeout configuration."""

import os
from unittest import mock

import pytest

from victor.config.timeouts import (
    DockerTimeouts,
    HttpTimeouts,
    McpTimeouts,
    ProcessTimeouts,
    TimeoutConfig,
    Timeouts,
)


class TestTimeoutConfig:
    """Tests for TimeoutConfig dataclass."""

    def test_default_values(self):
        """Default timeout values should be reasonable."""
        config = TimeoutConfig()

        # HTTP defaults
        assert config.HTTP_DEFAULT == 30.0
        assert config.HTTP_QUICK_CHECK == 2.0
        assert config.HTTP_WEB_SEARCH == 15.0
        assert config.HTTP_EMBEDDING == 120.0
        assert config.HTTP_LLM_API == 300.0

        # Process defaults
        assert config.BASH_DEFAULT == 60
        assert config.BASH_QUICK == 30
        assert config.BASH_LONG_RUNNING == 300
        assert config.TEST_EXECUTION == 300
        assert config.GIT_DEFAULT == 30

        # Docker defaults
        assert config.DOCKER_QUICK == 5
        assert config.DOCKER_LOGS == 30
        assert config.DOCKER_BUILD == 300

        # MCP defaults
        assert config.MCP_RESPONSE == 30.0
        assert config.MCP_PROCESS_TERMINATE == 5.0
        assert config.MCP_SERVER_IDLE == 300.0

    def test_from_env_with_defaults(self):
        """from_env should return defaults when no env vars are set."""
        config = TimeoutConfig.from_env()
        default_config = TimeoutConfig()

        assert config.HTTP_DEFAULT == default_config.HTTP_DEFAULT
        assert config.BASH_DEFAULT == default_config.BASH_DEFAULT

    def test_from_env_with_overrides(self):
        """from_env should pick up environment variable overrides."""
        env_vars = {
            "VICTOR_TIMEOUT_HTTP_DEFAULT": "60.0",
            "VICTOR_TIMEOUT_BASH_DEFAULT": "120",
            "VICTOR_TIMEOUT_MCP_RESPONSE": "45.0",
        }

        with mock.patch.dict(os.environ, env_vars, clear=False):
            config = TimeoutConfig.from_env()

        assert config.HTTP_DEFAULT == 60.0
        assert config.BASH_DEFAULT == 120
        assert config.MCP_RESPONSE == 45.0

    def test_from_env_ignores_invalid_values(self):
        """from_env should ignore invalid (non-numeric) environment values."""
        env_vars = {
            "VICTOR_TIMEOUT_HTTP_DEFAULT": "not_a_number",
            "VICTOR_TIMEOUT_BASH_DEFAULT": "also_invalid",
        }

        with mock.patch.dict(os.environ, env_vars, clear=False):
            config = TimeoutConfig.from_env()

        # Should fall back to defaults
        default_config = TimeoutConfig()
        assert config.HTTP_DEFAULT == default_config.HTTP_DEFAULT
        assert config.BASH_DEFAULT == default_config.BASH_DEFAULT

    def test_config_is_frozen(self):
        """TimeoutConfig should be immutable."""
        from dataclasses import FrozenInstanceError

        config = TimeoutConfig()
        with pytest.raises(FrozenInstanceError):
            config.HTTP_DEFAULT = 100.0  # type: ignore


class TestTimeoutAliases:
    """Tests for timeout alias classes."""

    def test_http_timeouts(self):
        """HttpTimeouts should reference global Timeouts."""
        assert HttpTimeouts.DEFAULT == Timeouts.HTTP_DEFAULT
        assert HttpTimeouts.QUICK == Timeouts.HTTP_QUICK_CHECK
        assert HttpTimeouts.WEB_SEARCH == Timeouts.HTTP_WEB_SEARCH
        assert HttpTimeouts.EMBEDDING == Timeouts.HTTP_EMBEDDING
        assert HttpTimeouts.LLM_API == Timeouts.HTTP_LLM_API

    def test_process_timeouts(self):
        """ProcessTimeouts should reference global Timeouts."""
        assert ProcessTimeouts.BASH_DEFAULT == Timeouts.BASH_DEFAULT
        assert ProcessTimeouts.BASH_QUICK == Timeouts.BASH_QUICK
        assert ProcessTimeouts.BASH_LONG == Timeouts.BASH_LONG_RUNNING
        assert ProcessTimeouts.TEST == Timeouts.TEST_EXECUTION
        assert ProcessTimeouts.GIT == Timeouts.GIT_DEFAULT

    def test_docker_timeouts(self):
        """DockerTimeouts should reference global Timeouts."""
        assert DockerTimeouts.QUICK == Timeouts.DOCKER_QUICK
        assert DockerTimeouts.LOGS == Timeouts.DOCKER_LOGS
        assert DockerTimeouts.BUILD == Timeouts.DOCKER_BUILD

    def test_mcp_timeouts(self):
        """McpTimeouts should reference global Timeouts."""
        assert McpTimeouts.RESPONSE == Timeouts.MCP_RESPONSE
        assert McpTimeouts.TERMINATE == Timeouts.MCP_PROCESS_TERMINATE
        assert McpTimeouts.KILL == Timeouts.MCP_PROCESS_KILL
        assert McpTimeouts.SERVER_IDLE == Timeouts.MCP_SERVER_IDLE


class TestGlobalTimeouts:
    """Tests for the global Timeouts singleton."""

    def test_timeouts_is_timeout_config(self):
        """Global Timeouts should be a TimeoutConfig instance."""
        assert isinstance(Timeouts, TimeoutConfig)

    def test_timeouts_has_all_fields(self):
        """Global Timeouts should have all expected fields."""
        expected_fields = [
            "HTTP_DEFAULT",
            "HTTP_QUICK_CHECK",
            "HTTP_WEB_SEARCH",
            "HTTP_EMBEDDING",
            "HTTP_LLM_API",
            "BASH_DEFAULT",
            "BASH_QUICK",
            "BASH_LONG_RUNNING",
            "TEST_EXECUTION",
            "DOCKER_QUICK",
            "DOCKER_LOGS",
            "DOCKER_BUILD",
            "GIT_DEFAULT",
            "MCP_RESPONSE",
            "MCP_PROCESS_TERMINATE",
            "MCP_SERVER_IDLE",
            "CIRCUIT_BREAKER_RECOVERY",
        ]

        for field in expected_fields:
            assert hasattr(Timeouts, field), f"Missing field: {field}"
            assert getattr(Timeouts, field) > 0, f"Invalid value for: {field}"
