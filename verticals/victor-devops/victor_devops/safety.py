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

"""DevOps Safety Extension - Security patterns for infrastructure work.

This module provides DevOps-specific safety patterns for infrastructure
operations including Kubernetes, Docker, Terraform, and cloud providers.

This module now delegates to the core safety infrastructure at
victor.security.safety.infrastructure for pattern scanning, while maintaining
backward compatibility for existing interfaces.
"""

from typing import Dict, List, Tuple

from victor_contracts.safety_patterns import (
    CLOUD_PATTERNS,
    CREDENTIAL_PATTERNS,
    DESTRUCTIVE_PATTERNS,
    DOCKER_PATTERNS,
    KUBERNETES_PATTERNS,
    TERRAFORM_PATTERNS,
    InfraScanResult,
    InfrastructureScanner,
    SecretScanner,
    get_safety_reminders as core_get_safety_reminders,
    validate_dockerfile as core_validate_dockerfile,
    validate_kubernetes_manifest as core_validate_kubernetes_manifest,
)
from victor_contracts.verticals import SafetyExtensionProtocol, SafetyPattern
from victor_contracts.safety_policy import SafetyEnforcer, SafetyRule, SafetyLevel

# Risk levels (kept for backward compatibility)
HIGH = "HIGH"
MEDIUM = "MEDIUM"
LOW = "LOW"


class DevOpsSafetyExtension(SafetyExtensionProtocol):
    """Safety extension for DevOps tasks.

    Provides DevOps-specific dangerous operation patterns including
    Kubernetes, Docker, Terraform, and cloud provider operations.

    This class delegates to the core InfrastructureScanner for pattern
    matching while providing the SafetyExtensionProtocol interface.
    """

    def __init__(
        self,
        include_destructive: bool = True,
        include_kubernetes: bool = True,
        include_docker: bool = True,
        include_terraform: bool = True,
        include_cloud: bool = True,
    ):
        """Initialize the safety extension.

        Args:
            include_destructive: Include destructive patterns
            include_kubernetes: Include Kubernetes patterns
            include_docker: Include Docker patterns
            include_terraform: Include Terraform patterns
            include_cloud: Include cloud provider patterns
        """
        self._include_destructive = include_destructive
        self._include_kubernetes = include_kubernetes
        self._include_docker = include_docker
        self._include_terraform = include_terraform
        self._include_cloud = include_cloud

        # Create an InfrastructureScanner with matching configuration
        self._scanner = InfrastructureScanner(
            include_destructive=include_destructive,
            include_kubernetes=include_kubernetes,
            include_docker=include_docker,
            include_terraform=include_terraform,
            include_cloud=include_cloud,
        )
        self._secret_scanner = SecretScanner()

    def get_bash_patterns(self) -> List[SafetyPattern]:
        """Return DevOps-specific bash patterns.

        Returns:
            List of SafetyPattern for dangerous bash commands.
        """
        return self._scanner.all_patterns

    def get_danger_patterns(self) -> List[Tuple[str, str, str]]:
        """Return DevOps-specific danger patterns (legacy format).

        Returns:
            List of (regex_pattern, description, risk_level) tuples.
        """
        return [(p.pattern, p.description, p.risk_level) for p in self._scanner.all_patterns]

    def get_blocked_operations(self) -> List[str]:
        """Return operations that should be blocked in DevOps context."""
        return [
            "delete_production_database",
            "destroy_production_infrastructure",
            "expose_secrets_to_logs",
            "disable_security_features",
            "create_public_s3_bucket",
        ]

    def get_credential_patterns(self) -> Dict[str, str]:
        """Return patterns for detecting credentials.

        Uses patterns from victor.security.safety.secrets for comprehensive detection.

        Returns:
            Dict of credential_type -> regex_pattern.
        """
        # Return simplified dict format for backward compatibility
        return {name: pattern for name, (pattern, _, _) in CREDENTIAL_PATTERNS.items()}

    def scan_for_secrets(self, content: str) -> List[Dict]:
        """Scan content for secrets using the core SecretScanner.

        Args:
            content: Text content to scan

        Returns:
            List of secret match dictionaries
        """
        matches = self._secret_scanner.scan(content)
        return [
            {
                "type": m.secret_type,
                "severity": m.severity.value,
                "line": m.line_number,
                "suggestion": m.suggestion,
            }
            for m in matches
        ]

    def scan_command(self, command: str) -> InfraScanResult:
        """Scan a command for dangerous patterns.

        Args:
            command: The command to scan

        Returns:
            InfraScanResult with matched patterns
        """
        return self._scanner.scan_command(command)

    def validate_dockerfile(self, content: str) -> List[str]:
        """Validate Dockerfile security best practices.

        Returns:
            List of security warnings found.
        """
        return core_validate_dockerfile(content)

    def validate_kubernetes_manifest(self, content: str) -> List[str]:
        """Validate Kubernetes manifest security.

        Returns:
            List of security warnings found.
        """
        return core_validate_kubernetes_manifest(content)

    def get_safety_reminders(self) -> List[str]:
        """Return safety reminders for DevOps output."""
        return core_get_safety_reminders()

    def get_category(self) -> str:
        """Get the category name for these patterns.

        Returns:
            Category identifier
        """
        return "devops"


__all__ = [
    "DevOpsSafetyExtension",
    # Re-exported from core for convenience
    "InfrastructureScanner",
    "InfraScanResult",
    "DESTRUCTIVE_PATTERNS",
    "KUBERNETES_PATTERNS",
    "DOCKER_PATTERNS",
    "TERRAFORM_PATTERNS",
    "CLOUD_PATTERNS",
    # Legacy constants
    "HIGH",
    "MEDIUM",
    "LOW",
    # New framework-based safety rules
    "create_deployment_safety_rules",
    "create_container_safety_rules",
    "create_infrastructure_safety_rules",
    "create_all_devops_safety_rules",
]


# =============================================================================
# Framework-Based Safety Rules (New)
# =============================================================================

"""Framework-based safety rules for DevOps operations.

This section provides factory functions that register safety rules
with the framework-level SafetyEnforcer. This is the new recommended
approach for safety enforcement in DevOps workflows.

Example:
    from victor.framework.config import SafetyEnforcer, SafetyConfig, SafetyLevel
    from victor_devops.safety import create_all_devops_safety_rules

    enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
    create_all_devops_safety_rules(enforcer)

    # Check operations
    allowed, reason = enforcer.check_operation("kubectl delete deployment -n production")
    if not allowed:
        print(f"Blocked: {reason}")
"""


def create_deployment_safety_rules(
    enforcer: SafetyEnforcer,
    *,
    require_approval_for_production: bool = True,
    require_backup_before_deploy: bool = True,
    protected_environments: list[str] | None = None,
    enable_rollback: bool = True,
) -> None:
    """Register deployment-specific safety rules.

    Args:
        enforcer: SafetyEnforcer to register rules with
        require_approval_for_production: Require approval for production deployments
        require_backup_before_deploy: Require backup before deployment
        protected_environments: Protected environments (default: production, prod, staging)
        enable_rollback: Enable automatic rollback on failure

    Example:
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_deployment_safety_rules(
            enforcer,
            require_approval_for_production=True,
            protected_environments=["production", "staging"]
        )
    """
    protected = protected_environments or ["production", "prod", "staging"]

    if require_approval_for_production:
        enforcer.add_rule(
            SafetyRule(
                name="deployment_require_approval",
                description="Require approval for production deployments",
                check_fn=lambda op: any(env in op.lower() for env in protected)
                and any(cmd in op for cmd in ["deploy", "kubectl apply", "terraform apply"]),
                level=SafetyLevel.HIGH,
                allow_override=True,
            )
        )

    if require_backup_before_deploy:
        enforcer.add_rule(
            SafetyRule(
                name="deployment_require_backup",
                description="Require backup before deployment to production",
                check_fn=lambda op: any(env in op.lower() for env in protected)
                and "deploy" in op.lower()
                and "backup" not in op.lower(),
                level=SafetyLevel.MEDIUM,
                allow_override=True,
            )
        )

    if enable_rollback:
        enforcer.add_rule(
            SafetyRule(
                name="deployment_rollback_plan",
                description="Warn if deployment doesn't include rollback plan",
                check_fn=lambda op: any(env in op.lower() for env in protected)
                and "deploy" in op.lower()
                and "rollback" not in op.lower(),
                level=SafetyLevel.LOW,  # Warn only
                allow_override=True,
            )
        )


def create_container_safety_rules(
    enforcer: SafetyEnforcer,
    *,
    block_privileged_containers: bool = True,
    block_root_user: bool = True,
    require_health_checks: bool = False,
) -> None:
    """Register container-specific safety rules.

    Args:
        enforcer: SafetyEnforcer to register rules with
        block_privileged_containers: Block privileged container creation
        block_root_user: Block containers running as root user
        require_health_checks: Require health checks in container definitions

    Example:
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_container_safety_rules(
            enforcer,
            block_privileged_containers=True,
            block_root_user=True
        )
    """
    if block_privileged_containers:
        enforcer.add_rule(
            SafetyRule(
                name="container_block_privileged",
                description="Block privileged container creation",
                check_fn=lambda op: "docker" in op.lower()
                or "kubectl" in op.lower()
                and "--privileged" in op
                or "privileged: true" in op.lower(),
                level=SafetyLevel.HIGH,
                allow_override=False,
            )
        )

    if block_root_user:
        enforcer.add_rule(
            SafetyRule(
                name="container_block_root",
                description="Block containers running as root user",
                check_fn=lambda op: ("docker" in op.lower() or "kubectl" in op.lower())
                and ("user: 0" in op or "USER root" in op or "--user 0" in op),
                level=SafetyLevel.HIGH,
                allow_override=True,
            )
        )

    if require_health_checks:
        enforcer.add_rule(
            SafetyRule(
                name="container_require_healthcheck",
                description="Require health checks in container definitions",
                check_fn=lambda op: ("docker" in op.lower() or "kubernetes" in op.lower())
                and "healthcheck" not in op.lower()
                and "readinessProbe" not in op
                and "livenessProbe" not in op,
                level=SafetyLevel.LOW,  # Warn only
                allow_override=True,
            )
        )


def create_infrastructure_safety_rules(
    enforcer: SafetyEnforcer,
    *,
    block_destructive_commands: bool = True,
    require_state_backup: bool = True,
    protected_resources: list[str] | None = None,
) -> None:
    """Register infrastructure-specific safety rules.

    Args:
        enforcer: SafetyEnforcer to register rules with
        block_destructive_commands: Block destructive infrastructure commands
        require_state_backup: Require Terraform state backup before modification
        protected_resources: List of protected resources (default: ["database", "storage", "vpc"])

    Example:
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_infrastructure_safety_rules(
            enforcer,
            block_destructive_commands=True,
            protected_resources=["database", "storage", "vpc", "load-balancer"]
        )
    """
    protected = protected_resources or ["database", "storage", "vpc"]

    if block_destructive_commands:
        enforcer.add_rule(
            SafetyRule(
                name="infra_block_destructive",
                description="Block destructive infrastructure commands",
                check_fn=lambda op: any(
                    cmd in op
                    for cmd in [
                        "terraform destroy",
                        "kubectl delete",
                        "helm uninstall",
                        "aws cloudformation delete-stack",
                        "gcloud deployment-manager delete",
                    ]
                ),
                level=SafetyLevel.HIGH,
                allow_override=False,
            )
        )

        enforcer.add_rule(
            SafetyRule(
                name="infra_block_protected_resource_deletion",
                description=f"Block deletion of protected resources: {', '.join(protected)}",
                check_fn=lambda op: ("delete" in op.lower() or "destroy" in op.lower())
                and any(resource in op.lower() for resource in protected),
                level=SafetyLevel.HIGH,
                allow_override=True,
            )
        )

    if require_state_backup:
        enforcer.add_rule(
            SafetyRule(
                name="infra_require_state_backup",
                description="Require Terraform state backup before modification",
                check_fn=lambda op: "terraform" in op.lower()
                and any(cmd in op for cmd in ["apply", "destroy"])
                and "backup" not in op.lower(),
                level=SafetyLevel.MEDIUM,
                allow_override=True,
            )
        )


def create_all_devops_safety_rules(
    enforcer: SafetyEnforcer,
    *,
    deployment_protected_environments: list[str] | None = None,
    infrastructure_protected_resources: list[str] | None = None,
) -> None:
    """Register all DevOps safety rules at once.

    This is a convenience function that registers all DevOps-specific
    safety rules with appropriate defaults.

    Args:
        enforcer: SafetyEnforcer to register rules with
        deployment_protected_environments: Protected deployment environments
        infrastructure_protected_resources: Protected infrastructure resources

    Example:
        from victor.framework.config import SafetyEnforcer, SafetyConfig, SafetyLevel

        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_all_devops_safety_rules(enforcer)

        # Now all operations are checked
        allowed, reason = enforcer.check_operation("kubectl delete deployment -n production app")
    """
    create_deployment_safety_rules(
        enforcer, protected_environments=deployment_protected_environments
    )
    create_container_safety_rules(enforcer)
    create_infrastructure_safety_rules(
        enforcer, protected_resources=infrastructure_protected_resources
    )
