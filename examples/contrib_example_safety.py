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

"""Example safety extension using victor.contrib.

This file demonstrates how to create a vertical-specific safety extension
using the BaseSafetyExtension from victor.contrib.safety.

This eliminates ~100+ lines of boilerplate code that would otherwise be
duplicated across verticals.
"""

from typing import List

from victor.contrib.safety import BaseSafetyExtension, VerticalSafetyMixin
from victor.agent.coordinators.safety_coordinator import (
    SafetyRule,
    SafetyAction,
    SafetyCategory,
)


class ExampleVerticalSafetyExtension(BaseSafetyExtension, VerticalSafetyMixin):
    """Example safety extension for a vertical.

    This example shows how to:
    1. Inherit from BaseSafetyExtension for common safety functionality
    2. Inherit from VerticalSafetyMixin for helper methods
    3. Implement get_vertical_name() to identify the vertical
    4. Implement get_vertical_rules() to define vertical-specific rules
    5. Use helper methods to create common safety patterns

    The base classes provide:
    - Automatic coordinator initialization
    - Rule registration and management
    - Safety checking (check_operation, is_operation_safe)
    - Statistics tracking
    - Vertical context tracking
    """

    def get_vertical_name(self) -> str:
        """Return the vertical name for safety tracking.

        Returns:
            Vertical identifier (e.g., "devops", "security", "research")
        """
        return "example"

    def get_vertical_rules(self) -> List[SafetyRule]:
        """Define vertical-specific safety rules.

        Returns:
            List of SafetyRule instances for this vertical

        Note:
            Use helper methods from VerticalSafetyMixin to create
            common patterns:
            - create_dangerous_command_rule(): Dangerous shell commands
            - create_file_deletion_rule(): File deletion operations
            - create_blocked_operation_rule(): Completely blocked operations
            - create_git_force_push_rule(): Git force push safety
            - create_docker_container_deletion_rule(): Docker container removal
            - create_system_write_rule(): System directory protection
        """
        return [
            # Example 1: Use helper for dangerous command pattern
            self.create_dangerous_command_rule(
                rule_id="example_dangerous_scan",
                command_pattern=r"rm -rf|format|fdisk",
                description="Dangerous system operation",
                severity=10,
            ),
            # Example 2: Use helper for file deletion pattern
            self.create_file_deletion_rule(
                rule_id="example_database_delete",
                file_pattern=r".*\.db|.*database",
                description="Database file deletion",
                severity=9,
            ),
            # Example 3: Use helper for completely blocked operation
            self.create_blocked_operation_rule(
                rule_id="example_production_modify",
                pattern=r"/prod/|/production/",
                description="Cannot modify production systems",
                category=SafetyCategory.SHELL,
            ),
            # Example 4: Create custom rule directly
            SafetyRule(
                rule_id="example_custom_tool_restriction",
                category=SafetyCategory.SHELL,
                pattern=r"scp.*to.*production",
                description="SCP to production requires confirmation",
                action=SafetyAction.REQUIRE_CONFIRMATION,
                severity=8,
                tool_names=["shell"],
                confirmation_prompt="This will copy files to production. Confirm?",
            ),
            # Example 5: Use helper for git operations
            self.create_git_force_push_rule(
                rule_id="example_git_force_push",
                severity=8,
            ),
            # Example 6: Use helper for Docker operations
            self.create_docker_container_deletion_rule(
                rule_id="example_docker_rm",
                severity=7,
            ),
            # Example 7: Use helper for system directory protection
            self.create_system_write_rule(
                rule_id="example_system_write",
                system_paths=["/etc", "/var", "/usr/local"],
            ),
        ]


# Example of vertical-specific customization


class DevOpsSafetyExtension(BaseSafetyExtension, VerticalSafetyMixin):
    """DevOps-specific safety extension example.

    DevOps operations have specific safety concerns around:
    - Infrastructure modification
    - Deployment operations
    - Configuration changes
    """

    def get_vertical_name(self) -> str:
        return "devops"

    def get_vertical_rules(self) -> List[SafetyRule]:
        """DevOps-specific safety rules."""
        return [
            # Infrastructure modification safety
            self.create_dangerous_command_rule(
                "infra_dangerous",
                r"terraform.*destroy|kubectl.*delete",
                "Infrastructure destruction",
                severity=10,
            ),
            # Production deployment safety
            SafetyRule(
                rule_id="deploy_production",
                category=SafetyCategory.SHELL,
                pattern=r"deploy.*production|kubectl.*apply.*prod",
                description="Production deployment requires confirmation",
                action=SafetyAction.REQUIRE_CONFIRMATION,
                severity=9,
                tool_names=["shell"],
                confirmation_prompt="Deploying to PRODUCTION. Confirm?",
            ),
            # Configuration file safety
            self.create_file_deletion_rule(
                "config_delete",
                r".*\.yaml|.*\.yml|.*\.conf",
                "Configuration file deletion",
                severity=8,
                tool_names=["write", "edit"],
            ),
        ]


class SecuritySafetyExtension(BaseSafetyExtension, VerticalSafetyMixin):
    """Security-specific safety extension example.

    Security analysis requires:
    - Careful handling of sensitive data
    - Safe execution of scanning tools
    - No modification during analysis
    """

    def get_vertical_name(self) -> str:
        return "security"

    def get_vertical_rules(self) -> List[SafetyRule]:
        """Security-specific safety rules."""
        return [
            # Security scanning tools are allowed
            SafetyRule(
                rule_id="security_scan_tool",
                category=SafetyCategory.SHELL,
                pattern=r"bandit|trivy|semgrep|gitleaks|safety",
                description="Security scanning tool",
                action=SafetyAction.WARN,  # Just warn, don't block
                severity=3,  # Low severity, these are expected
                tool_names=["shell"],
            ),
            # Block accidental modifications during security analysis
            self.create_blocked_operation_rule(
                "security_no_modify",
                pattern=r"write.*\.(py|js|ts|java|go)",
                description="No code modifications during security analysis",
                category=SafetyCategory.FILE,
            ),
            # Dangerous security tools
            self.create_dangerous_command_rule(
                "security_dangerous_tool",
                r"nmap|metasploit|burpsuite",
                "Penetration testing tool - requires confirmation",
                severity=9,
            ),
            # Protect secrets from being exposed
            SafetyRule(
                rule_id="security_secret_protection",
                category=SafetyCategory.SHELL,
                pattern=r"cat.*\.env|cat.*secrets|cat.*credentials",
                description="Accessing secrets file",
                action=SafetyAction.REQUIRE_CONFIRMATION,
                severity=8,
                tool_names=["shell", "read"],
                confirmation_prompt="This will display secrets. Confirm?",
            ),
        ]


__all__ = [
    "ExampleVerticalSafetyExtension",
    "DevOpsSafetyExtension",
    "SecuritySafetyExtension",
]
