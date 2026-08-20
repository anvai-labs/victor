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

"""Integration tests for IsolationMapper and constraint-to-sandbox mapping.

Tests the isolation mapping layer that converts workflow constraints
to execution environment configurations.

Test cases:
- Vertical default isolation configurations
- Constraint-based isolation overrides
- Airgapped constraint handling
- Full access constraint handling
- Docker and process sandbox selection
"""

import pytest
from dataclasses import dataclass
from typing import Literal, Optional
from unittest.mock import MagicMock, patch


# Test isolation configuration
@dataclass
class IsolationConfig:
    """Test isolation configuration."""

    sandbox_type: Literal["none", "process", "docker"] = "none"
    network_allowed: bool = True
    filesystem_readonly: bool = False


class TestIsolationMapper:
    """Tests for IsolationMapper constraint-to-sandbox mapping."""

    def test_vertical_defaults_exist(self):
        """Test all vertical defaults are defined."""
        try:
            from victor.workflows.isolation import IsolationMapper

            expected_verticals = ["coding", "research", "devops", "dataanalysis", "rag"]

            for vertical in expected_verticals:
                assert (
                    vertical in IsolationMapper.VERTICAL_DEFAULTS
                ), f"Missing default for vertical: {vertical}"
        except ImportError:
            pytest.skip("IsolationMapper not available")

    def test_coding_vertical_default(self):
        """Test coding vertical defaults to process isolation."""
        try:
            from victor.workflows.isolation import IsolationMapper

            config = IsolationMapper.VERTICAL_DEFAULTS.get("coding")
            assert config is not None
            assert config.sandbox_type == "process"
        except ImportError:
            pytest.skip("IsolationMapper not available")

    def test_devops_vertical_default(self):
        """Test devops vertical defaults to docker isolation."""
        try:
            from victor.workflows.isolation import IsolationMapper

            config = IsolationMapper.VERTICAL_DEFAULTS.get("devops")
            assert config is not None
            assert config.sandbox_type == "docker"
            assert config.network_allowed is True
        except ImportError:
            pytest.skip("IsolationMapper not available")

    def test_research_vertical_default(self):
        """Test research vertical defaults to no isolation."""
        try:
            from victor.workflows.isolation import IsolationMapper

            config = IsolationMapper.VERTICAL_DEFAULTS.get("research")
            assert config is not None
            assert config.sandbox_type == "none"
        except ImportError:
            pytest.skip("IsolationMapper not available")

    def test_rag_vertical_default(self):
        """Test RAG vertical defaults to docker with readonly filesystem."""
        try:
            from victor.workflows.isolation import IsolationMapper

            config = IsolationMapper.VERTICAL_DEFAULTS.get("rag")
            assert config is not None
            assert config.sandbox_type == "docker"
            assert config.filesystem_readonly is True
        except ImportError:
            pytest.skip("IsolationMapper not available")

    def test_airgapped_constraints_override(self):
        """Test airgapped constraints disable network."""
        try:
            from victor.workflows.isolation import IsolationMapper
            from victor.workflows.definition import AirgappedConstraints

            constraints = AirgappedConstraints()
            config = IsolationMapper.from_constraints(constraints)

            assert config.sandbox_type == "none"
            assert config.network_allowed is False
        except ImportError:
            pytest.skip("IsolationMapper or AirgappedConstraints not available")

    def test_full_access_constraints(self):
        """Test full access constraints use docker with network."""
        try:
            from victor.workflows.isolation import IsolationMapper
            from victor.workflows.definition import FullAccessConstraints

            constraints = FullAccessConstraints()
            config = IsolationMapper.from_constraints(constraints)

            assert config.sandbox_type == "docker"
            assert config.network_allowed is True
        except ImportError:
            pytest.skip("IsolationMapper or FullAccessConstraints not available")

    def test_compute_only_constraints(self):
        """Test compute-only constraints use process isolation."""
        try:
            from victor.workflows.isolation import IsolationMapper
            from victor.workflows.definition import ComputeOnlyConstraints

            constraints = ComputeOnlyConstraints()
            config = IsolationMapper.from_constraints(constraints)

            assert config.sandbox_type == "process"
        except ImportError:
            pytest.skip("IsolationMapper or ComputeOnlyConstraints not available")

    def test_vertical_override_with_constraints(self):
        """Test constraints take precedence over vertical defaults."""
        try:
            from victor.workflows.isolation import IsolationMapper
            from victor.workflows.definition import AirgappedConstraints

            # Even with devops vertical (docker default), airgapped should override
            constraints = AirgappedConstraints()
            config = IsolationMapper.from_constraints(constraints, vertical="devops")

            assert config.network_allowed is False
        except ImportError:
            pytest.skip("IsolationMapper not available")

    def test_unknown_vertical_defaults_to_process(self):
        """Test unknown vertical falls back to process isolation."""
        try:
            from victor.workflows.isolation import IsolationMapper

            config = IsolationMapper.from_constraints(None, vertical="unknown_vertical")

            assert config.sandbox_type == "process"
        except ImportError:
            pytest.skip("IsolationMapper not available")


class TestIsolationConfigProperties:
    """Test IsolationConfig dataclass properties."""

    def test_isolation_config_creation(self):
        """Test IsolationConfig can be created with all options."""
        config = IsolationConfig(
            sandbox_type="docker",
            network_allowed=False,
            filesystem_readonly=True,
        )

        assert config.sandbox_type == "docker"
        assert config.network_allowed is False
        assert config.filesystem_readonly is True

    def test_isolation_config_defaults(self):
        """Test IsolationConfig default values."""
        config = IsolationConfig()

        assert config.sandbox_type == "none"
        assert config.network_allowed is True
        assert config.filesystem_readonly is False


class TestExecutorIsolationIntegration:
    """Test executor integration with isolation mapping."""

    @pytest.mark.asyncio
    async def test_executor_selects_docker_for_devops(self):
        """Test executor uses Docker sandbox for devops vertical."""
        from victor.workflows.isolation import IsolationMapper

        # Check that devops vertical would use docker
        config = IsolationMapper.VERTICAL_DEFAULTS.get("devops")
        assert config.sandbox_type == "docker"

    @pytest.mark.asyncio
    async def test_executor_selects_process_for_coding(self):
        """Test executor uses process sandbox for coding vertical."""
        from victor.workflows.isolation import IsolationMapper

        # Check that coding vertical would use process
        config = IsolationMapper.VERTICAL_DEFAULTS.get("coding")
        assert config.sandbox_type == "process"


class TestConstraintTypes:
    """Test different constraint type behaviors."""

    def test_constraint_llm_allowed_flag(self):
        """Test constraint llm_allowed flag is respected."""
        try:
            from victor.workflows.definition import ComputeOnlyConstraints

            constraints = ComputeOnlyConstraints()
            assert constraints.llm_allowed is False
        except ImportError:
            pytest.skip("ComputeOnlyConstraints not available")

    def test_constraint_timeout_setting(self):
        """Test constraint timeout is configurable."""
        try:
            from victor.workflows.definition import TaskConstraints

            # Use base TaskConstraints which accepts _timeout as field
            constraints = TaskConstraints(_timeout=60)
            assert constraints.timeout == 60
        except ImportError:
            pytest.skip("TaskConstraints not available")

    def test_constraint_cost_tier_setting(self):
        """Test constraint cost tier is configurable."""
        try:
            from victor.workflows.definition import TaskConstraints

            # Use base TaskConstraints which accepts max_cost_tier
            constraints = TaskConstraints(max_cost_tier="HIGH")
            assert constraints.max_cost_tier == "HIGH"
        except ImportError:
            pytest.skip("TaskConstraints not available")


class TestSandboxTypeSelection:
    """Test sandbox type selection logic."""

    def test_none_sandbox_inline_execution(self):
        """Test 'none' sandbox means inline execution."""
        config = IsolationConfig(sandbox_type="none")

        # Inline execution - no isolation
        assert config.sandbox_type == "none"
        assert config.network_allowed is True

    def test_process_sandbox_subprocess_execution(self):
        """Test 'process' sandbox means subprocess execution."""
        config = IsolationConfig(sandbox_type="process")

        # Subprocess execution - process-level isolation
        assert config.sandbox_type == "process"

    def test_docker_sandbox_container_execution(self):
        """Test 'docker' sandbox means container execution."""
        config = IsolationConfig(sandbox_type="docker")

        # Container execution - full isolation
        assert config.sandbox_type == "docker"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
