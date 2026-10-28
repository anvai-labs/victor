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

"""Enhanced safety integration for victor-devops using SafetyCoordinator.

This module provides DevOps-specific safety rules and integration with
the framework's SafetyCoordinator for enhanced safety enforcement.

Design Pattern: Extension + Delegation
- Defines DevOps-specific safety rules
- Registers them with SafetyCoordinator
- Provides safety checking interface for DevOps operations

Integration Point:
    Use in DevOpsAssistant.get_extensions() as enhanced safety extension
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from victor_contracts import SafetyAction, SafetyCategory, SafetyCoordinator, SafetyRule
from victor_contracts.verticals import CoordinatorSafetyExtension

logger = logging.getLogger(__name__)
from victor_contracts.safety_patterns import BUILD_DEPLOY_PATTERNS


class DevOpsSafetyRules:
    """DevOps-specific safety rules for the SafetyCoordinator.

    Provides comprehensive safety rules for DevOps operations including:
    - Docker operations (container deletion, image removal)
    - Kubernetes operations (resource deletion, namespace removal)
    - CI/CD operations (pipeline triggers, deployments)
    - Infrastructure operations (Terraform apply/destroy)
    - System operations (service restarts, package installs)
    """

    @staticmethod
    def get_docker_rules() -> List[SafetyRule]:
        """Get Docker-specific safety rules.

        Returns:
            List of safety rules for Docker operations
        """
        return [
            # Docker system prune is dangerous
            SafetyRule(
                rule_id="devops_docker_system_prune",
                category=SafetyCategory.DOCKER,
                pattern=r"system.*prune.*--all|-a",
                description="Docker system prune -a (removes all unused data)",
                action=SafetyAction.REQUIRE_CONFIRMATION,
                severity=8,
                confirmation_prompt=(
                    "This will remove all unused Docker data "
                    "(images, containers, volumes). Continue?"
                ),
                tool_names=["docker"],
            ),
            # Docker rm -f (force remove container)
            SafetyRule(
                rule_id="devops_docker_rm_force",
                category=SafetyCategory.DOCKER,
                pattern=r"rm.*-f|container.*rm.*--force",
                description="Force remove Docker container",
                action=SafetyAction.WARN,
                severity=6,
                tool_names=["docker"],
            ),
            # Docker rmi (remove image)
            SafetyRule(
                rule_id="devops_docker_rmi",
                category=SafetyCategory.DOCKER,
                pattern=r"rmi|image.*rm",
                description="Remove Docker image",
                action=SafetyAction.WARN,
                severity=5,
                tool_names=["docker"],
            ),
        ]

    @staticmethod
    def get_kubernetes_rules() -> List[SafetyRule]:
        """Get Kubernetes-specific safety rules.

        Returns:
            List of safety rules for Kubernetes operations
        """
        return [
            # kubectl delete namespace is BLOCKED
            SafetyRule(
                rule_id="devops_k8s_delete_namespace",
                category=SafetyCategory.SHELL,
                pattern=r"kubectl.*delete.*namespace|kubectl.*delete.*ns.*kube-system",
                description="Delete Kubernetes namespace",
                action=SafetyAction.BLOCK,
                severity=10,
                tool_names=["shell", "execute_bash"],
            ),
            # kubectl delete --all is dangerous
            SafetyRule(
                rule_id="devops_k8s_delete_all",
                category=SafetyCategory.SHELL,
                pattern=r"kubectl.*delete.*--all",
                description="Delete all Kubernetes resources in namespace",
                action=SafetyAction.REQUIRE_CONFIRMATION,
                severity=9,
                confirmation_prompt="This will delete all resources in the namespace. Continue?",
                tool_names=["shell", "execute_bash"],
            ),
            # kubectl apply --force is dangerous
            SafetyRule(
                rule_id="devops_k8s_apply_force",
                category=SafetyCategory.SHELL,
                pattern=r"kubectl.*apply.*--force|kubectl.*replace.*--force",
                description="Force apply Kubernetes configuration",
                action=SafetyAction.WARN,
                severity=6,
                tool_names=["shell", "execute_bash"],
            ),
        ]

    @staticmethod
    def get_terraform_rules() -> List[SafetyRule]:
        """Get Terraform/Infrastructure safety rules.

        Returns:
            List of safety rules for Terraform operations
        """
        return [
            # terraform destroy is BLOCKED without confirmation
            SafetyRule(
                rule_id="devops_terraform_destroy",
                category=SafetyCategory.SHELL,
                pattern=r"terraform.*destroy.*-auto-approve",
                description="Terraform destroy with auto-approve",
                action=SafetyAction.BLOCK,
                severity=10,
                tool_names=["shell", "execute_bash"],
            ),
            # terraform apply with auto-approve is dangerous
            SafetyRule(
                rule_id="devops_terraform_apply_auto",
                category=SafetyCategory.SHELL,
                pattern=r"terraform.*apply.*-auto-approve",
                description="Terraform apply with auto-approve",
                action=SafetyAction.REQUIRE_CONFIRMATION,
                severity=7,
                confirmation_prompt=(
                    "This will apply infrastructure changes without review. " "Continue?"
                ),
                tool_names=["shell", "execute_bash"],
            ),
        ]

    @staticmethod
    def get_ci_cd_rules() -> List[SafetyRule]:
        """Get CI/CD pipeline safety rules.

        Returns:
            List of safety rules for CI/CD operations
        """
        return [
            # Deploying to production is sensitive
            SafetyRule(
                rule_id="devops_deploy_production",
                category=SafetyCategory.SHELL,
                pattern=r"deploy.*production|deploy.*prod|--env.*prod",
                description="Deploy to production environment",
                action=SafetyAction.REQUIRE_CONFIRMATION,
                severity=8,
                confirmation_prompt="Confirm deployment to production?",
                tool_names=["shell", "execute_bash"],
            ),
            # Force triggering all pipelines
            SafetyRule(
                rule_id="devops_force_trigger_all",
                category=SafetyCategory.SHELL,
                pattern=r"(gitlab-ci|github-actions|jenkins).*trigger.*--all",
                description="Force trigger all CI/CD pipelines",
                action=SafetyAction.WARN,
                severity=5,
                tool_names=["shell", "execute_bash"],
            ),
        ]

    @staticmethod
    def get_system_rules() -> List[SafetyRule]:
        """Get system operation safety rules.

        Returns:
            List of safety rules for system operations
        """
        return [
            # systemctl stop critical services is dangerous
            SafetyRule(
                rule_id="devops_stop_critical_service",
                category=SafetyCategory.SHELL,
                pattern=r"systemctl.*stop.*(nginx|apache|postgres|mysql|redis|docker|kubernetes)",
                description="Stop critical production service",
                action=SafetyAction.REQUIRE_CONFIRMATION,
                severity=8,
                confirmation_prompt="This will stop a critical service. Continue?",
                tool_names=["shell", "execute_bash"],
            ),
            # Service restart requires confirmation
            SafetyRule(
                rule_id="devops_restart_service",
                category=SafetyCategory.SHELL,
                pattern=r"systemctl.*restart|service.*restart",
                description="Restart service",
                action=SafetyAction.WARN,
                severity=4,
                tool_names=["shell", "execute_bash"],
            ),
        ]

    @staticmethod
    def get_all_rules() -> List[SafetyRule]:
        """Get all DevOps-specific safety rules.

        Returns:
            List of all safety rules for DevOps operations
        """
        rules = []
        rules.extend(DevOpsSafetyRules.get_docker_rules())
        rules.extend(DevOpsSafetyRules.get_kubernetes_rules())
        rules.extend(DevOpsSafetyRules.get_terraform_rules())
        rules.extend(DevOpsSafetyRules.get_ci_cd_rules())
        rules.extend(DevOpsSafetyRules.get_system_rules())
        return rules


class EnhancedDevOpsSafetyExtension(CoordinatorSafetyExtension):
    """Enhanced devops safety extension (shared coordinator kernel)."""

    def __init__(self, strict_mode: bool = False, enable_custom_rules: bool = True):
        super().__init__(
            "devops",
            DevOpsSafetyRules.get_all_rules(),
            strict_mode=strict_mode,
            enable_custom_rules=enable_custom_rules,
            bash_patterns=list(BUILD_DEPLOY_PATTERNS),
            tool_restrictions={
                "docker": ["system prune -a", "rmi $(docker images -q)"],
                "kubectl": ["delete namespace", "delete --all"],
                "shell": ["terraform destroy -auto-approve"],
            },
        )


__all__ = [
    "DevOpsSafetyRules",
    "EnhancedDevOpsSafetyExtension",
]
