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

"""Example mode configuration provider using victor.contrib.

This file demonstrates how to create a vertical-specific mode configuration
using the BaseModeConfigProvider from victor.contrib.mode_config.

This eliminates ~50+ lines of boilerplate code that would otherwise be
duplicated across verticals.
"""

from typing import Dict

from victor.contrib.mode_config import BaseModeConfigProvider, ModeHelperMixin
from victor.core.mode_config import ModeDefinition


class ExampleVerticalModeConfig(BaseModeConfigProvider, ModeHelperMixin):
    """Example mode configuration for a vertical.

    This example shows how to:
    1. Inherit from BaseModeConfigProvider for common mode functionality
    2. Inherit from ModeHelperMixin for helper methods
    3. Implement get_vertical_name() to identify the vertical
    4. Implement get_vertical_modes() to define vertical-specific modes
    5. Optionally override get_task_budgets() for task-specific budgets
    6. Optionally override get_default_mode() for default selection

    The base classes provide:
    - Automatic registration with ModeConfigRegistry
    - Integration with framework's default modes (quick, standard, thorough)
    - Tool budget calculation with task-type support
    - Protocol compliance (ModeConfigProviderProtocol)
    """

    def get_vertical_name(self) -> str:
        """Return the vertical name for mode registration.

        Returns:
            Vertical identifier (e.g., "devops", "security", "research")
        """
        return "example"

    def get_vertical_modes(self) -> Dict[str, ModeDefinition]:
        """Define vertical-specific mode definitions.

        Returns:
            Dict mapping mode names to ModeDefinition instances

        Note:
            Use helper methods from ModeHelperMixin to create
            common modes:
            - create_quick_mode(): Fast operations (5 tool budget)
            - create_standard_mode(): Balanced (15 tool budget)
            - create_thorough_mode(): Deep analysis (30 tool budget)
            - create_exploration_mode(): Extended exploration (20 tool budget)
            - create_custom_mode(): Custom configuration

        Helper methods for mode groups:
            - create_quick_modes(): Returns {'quick': ..., 'fast': ...}
            - create_standard_modes(): Returns {'standard': ..., 'default': ...}
            - create_thorough_modes(): Returns {'thorough': ..., 'comprehensive': ...}
        """
        # Start with common modes from helper methods
        modes = {
            **self.create_quick_modes(),
            **self.create_standard_modes(),
            **self.create_thorough_modes(),
        }

        # Add vertical-specific modes
        modes.update(
            {
                # Example: Custom mode with specific requirements
                "analysis": ModeDefinition(
                    name="analysis",
                    tool_budget=20,
                    max_iterations=50,
                    temperature=0.7,
                    description="Deep analysis and investigation",
                    exploration_multiplier=2.5,
                    priority_tools=["code_search", "overview", "grep"],
                ),
                # Example: Mode with tool restrictions
                "readonly": ModeDefinition(
                    name="readonly",
                    tool_budget=15,
                    max_iterations=30,
                    temperature=0.5,
                    description="Read-only investigation mode",
                    exploration_multiplier=2.0,
                    allowed_tools={"read", "ls", "grep", "code_search", "overview"},
                ),
                # Example: High-exploration mode
                "explore": ModeDefinition(
                    name="explore",
                    tool_budget=25,
                    max_iterations=60,
                    temperature=0.7,
                    description="Extended exploration for understanding",
                    exploration_multiplier=3.0,
                ),
            }
        )

        return modes

    def get_task_budgets(self) -> Dict[str, int]:
        """Define task-specific tool budgets.

        Returns:
            Dict mapping task types to recommended tool budgets

        Example:
            return {
                "quick_fix": 5,
                "refactor": 15,
                "investigation": 20,
            }
        """
        return {
            # Quick tasks
            "quick_check": 5,
            "simple_fix": 8,
            # Standard tasks
            "analysis": 15,
            "refactor": 15,
            "debug": 12,
            # Complex tasks
            "investigation": 25,
            "comprehensive": 30,
            "migration": 40,
        }

    def get_default_mode(self) -> str:
        """Specify the default mode for this vertical.

        Returns:
            Default mode name
        """
        return "standard"

    def get_default_budget(self) -> int:
        """Specify the default tool budget.

        Returns:
            Default tool budget when no mode or task specified
        """
        return 15


# Example: DevOps mode configuration


class DevOpsModeConfig(BaseModeConfigProvider, ModeHelperMixin):
    """DevOps-specific mode configuration.

    DevOps operations require:
    - Careful handling of infrastructure changes
    - Different modes for different environments
    - Verification and rollback capabilities
    """

    def get_vertical_name(self) -> str:
        return "devops"

    def get_vertical_modes(self) -> Dict[str, ModeDefinition]:
        """DevOps-specific modes."""
        modes = {
            **self.create_quick_modes(),
            **self.create_standard_modes(),
            **self.create_thorough_modes(),
        }

        # Add DevOps-specific modes
        modes.update(
            {
                # Infrastructure modification mode (high caution)
                "infra_modify": ModeDefinition(
                    name="infra_modify",
                    tool_budget=40,
                    max_iterations=80,
                    temperature=0.3,  # Low temperature for consistency
                    description="Infrastructure modification with verification",
                    exploration_multiplier=1.5,
                    priority_tools=["shell", "terraform", "kubectl", "ansible"],
                ),
                # Deployment mode (requires confirmation)
                "deploy": ModeDefinition(
                    name="deploy",
                    tool_budget=30,
                    max_iterations=60,
                    temperature=0.5,
                    description="Deployment with rollback support",
                    exploration_multiplier=1.5,
                    priority_tools=["shell", "git", "docker", "kubectl"],
                ),
                # Monitoring and observability mode
                "monitoring": ModeDefinition(
                    name="monitoring",
                    tool_budget=20,
                    max_iterations=40,
                    temperature=0.5,
                    description="Monitoring and observability setup",
                    exploration_multiplier=2.0,
                    priority_tools=["shell", "read", "code_search"],
                ),
                # CI/CD pipeline mode
                "cicd": ModeDefinition(
                    name="cicd",
                    tool_budget=25,
                    max_iterations=50,
                    temperature=0.5,
                    description="CI/CD pipeline configuration",
                    exploration_multiplier=1.8,
                    priority_tools=["shell", "git", "read", "write"],
                ),
            }
        )

        return modes

    def get_task_budgets(self) -> Dict[str, int]:
        """DevOps-specific task budgets."""
        return {
            # Quick operations
            "config_change": 8,
            "log_check": 10,
            "status_check": 5,
            # Standard operations
            "deploy": 25,
            "service_restart": 12,
            "scale": 15,
            # Complex operations
            "infra_setup": 40,
            "migration": 50,
            "rollback": 20,
        }

    def get_default_mode(self) -> str:
        return "standard"


# Example: Security mode configuration


class SecurityModeConfig(BaseModeConfigProvider, ModeHelperMixin):
    """Security-specific mode configuration.

    Security analysis requires:
    - Different modes for different scan types
    - Balanced thoroughness vs. speed
    - Compliance-focused modes
    """

    def get_vertical_name(self) -> str:
        return "security"

    def get_vertical_modes(self) -> Dict[str, ModeDefinition]:
        """Security-specific modes."""
        modes = {
            **self.create_quick_modes(),
            **self.create_standard_modes(),
            **self.create_thorough_modes(),
        }

        # Add security-specific modes
        modes.update(
            {
                # Quick vulnerability scan
                "quick_scan": ModeDefinition(
                    name="quick_scan",
                    tool_budget=15,
                    max_iterations=20,
                    temperature=0.3,  # Low for consistent analysis
                    description="Quick security scan for high-severity issues",
                    exploration_multiplier=0.8,
                    priority_tools=["code_search", "shell", "read"],
                ),
                # Comprehensive security audit
                "comprehensive": ModeDefinition(
                    name="comprehensive",
                    tool_budget=50,
                    max_iterations=100,
                    temperature=0.5,
                    description="Comprehensive security audit covering all categories",
                    exploration_multiplier=2.5,
                    priority_tools=["code_search", "shell", "grep", "read"],
                ),
                # Compliance-focused mode
                "compliance": ModeDefinition(
                    name="compliance",
                    tool_budget=40,
                    max_iterations=80,
                    temperature=0.3,
                    description="Compliance-focused audit (SOC2, HIPAA, PCI-DSS)",
                    exploration_multiplier=2.0,
                    priority_tools=["shell", "code_search", "read"],
                ),
                # Penetration testing mode
                "pen_test": ModeDefinition(
                    name="pen_test",
                    tool_budget=60,
                    max_iterations=120,
                    temperature=0.7,
                    description="Penetration testing with exploit verification",
                    exploration_multiplier=3.0,
                    priority_tools=["shell", "code_search"],
                ),
                # Secret detection mode
                "secret_scan": ModeDefinition(
                    name="secret_scan",
                    tool_budget=20,
                    max_iterations=30,
                    temperature=0.4,
                    description="Secret and credential detection",
                    exploration_multiplier=1.5,
                    priority_tools=["code_search", "shell", "grep"],
                ),
            }
        )

        return modes

    def get_task_budgets(self) -> Dict[str, int]:
        """Security-specific task budgets."""
        return {
            # Scanning tasks
            "vulnerability_scan": 25,
            "secret_detection": 15,
            "dependency_audit": 20,
            # Analysis tasks
            "code_review": 30,
            "compliance_review": 35,
            # Assessment tasks
            "risk_assessment": 25,
            "pen_test": 60,
        }

    def get_default_mode(self) -> str:
        return "quick_scan"


__all__ = [
    "ExampleVerticalModeConfig",
    "DevOpsModeConfig",
    "SecurityModeConfig",
]
