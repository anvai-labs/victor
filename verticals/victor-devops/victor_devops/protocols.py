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

"""Victor extension protocol implementations for victor-devops.

This module provides protocol implementations that can be discovered via
the Victor extension entry point system, enabling the DevOps vertical to
register capabilities with the framework without direct dependencies.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

# Import victor-contracts protocols (NO runtime dependency on victor-ai!)
from victor_contracts.verticals import (
    make_pattern_safety_provider,
    make_prompt_provider,
    make_stage_tool_strategy,
    make_tool_provider,
    make_workflow_provider,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Tool Provider
# =============================================================================


DEVOPS_TOOLS = [
    # Core filesystem tools
    "read",
    "write",
    "edit",
    "grep",
    "ls",
    # Docker tools
    "docker_build",
    "docker_run",
    "docker_compose",
    "docker_ps",
    "docker_logs",
    "docker_exec",
    # Kubernetes tools
    "kubectl_apply",
    "kubectl_get",
    "kubectl_logs",
    "kubectl_describe",
    # CI/CD tools
    "jenkins_build",
    "github_actions",
    "gitlab_ci",
    # Cloud tools
    "aws_cli",
    "gcloud_cli",
    "azure_cli",
    # Infrastructure as Code
    "terraform_apply",
    "terraform_plan",
    "ansible_playbook",
    # Monitoring tools
    "prometheus_query",
    "grafana_dashboard",
    # Shell execution
    "shell",
    "ssh",
]

DEVOPS_STAGE_TOOLS = {
    "assess": ["read", "grep", "ls", "shell"],
    "plan": ["read", "terraform_plan", "kubectl_get"],
    "apply": ["terraform_apply", "kubectl_apply", "docker_build"],
    "verify": ["kubectl_get", "docker_ps", "shell"],
    "monitor": ["prometheus_query", "kubectl_logs", "docker_logs"],
}

DEVOPS_DANGEROUS_PATTERNS = [
    # Docker dangerous commands
    {"pattern": "docker rm -f", "description": "Force remove containers"},
    {"pattern": "docker rmi -f", "description": "Force remove images"},
    {
        "pattern": "docker system prune -f",
        "description": "Prune all Docker data",
    },
    # Kubernetes dangerous commands
    {"pattern": "kubectl delete", "description": "Delete K8s resources"},
    {
        "pattern": "kubectl apply --force",
        "description": "Force apply K8s resources",
    },
    {"pattern": "kubectl drain", "description": "Drain K8s node"},
    # Terraform dangerous commands
    {"pattern": "terraform destroy", "description": "Destroy infrastructure"},
    {
        "pattern": "terraform apply -auto-approve",
        "description": "Auto-approve changes",
    },
    # Production dangerous commands
    {"pattern": "rm -rf /", "description": "Recursive root deletion"},
    {"pattern": "dd if=/dev/zero", "description": "Disk wiping"},
    {"pattern": "> /dev/sda", "description": "Disk overwrite"},
]

DEVOPS_TOOL_RESTRICTIONS = {
    "kubectl_apply": ["--dry-run=client"],
    "terraform_apply": ["-auto-approve"],
}

DEVOPS_PROMPT_SECTIONS = {
    "role": (
        "You are a DevOps assistant specializing in infrastructure automation, "
        "CI/CD, and cloud operations."
    ),
    "expertise": (
        "You have expertise in Docker, Kubernetes, Terraform, Ansible, and "
        "major cloud providers (AWS, GCP, Azure)."
    ),
    "safety": (
        "Always check infrastructure state before making changes. Use dry-run "
        "modes where available. Never force destructive operations without "
        "explicit confirmation."
    ),
    "best_practices": (
        "Follow Infrastructure as Code best practices: idempotency, "
        "declarative configuration, and immutable infrastructure."
    ),
}

DEVOPS_TASK_TYPE_HINTS = {
    "deploy": {
        "hint": ("[DEPLOY] Plan infrastructure changes, apply configurations, " "verify health."),
        "tool_budget": 10,
    },
    "monitor": {
        "hint": "[MONITOR] Check logs, metrics, and service health.",
        "tool_budget": 8,
    },
    "troubleshoot": {
        "hint": "[TROUBLESHOOT] Diagnose issues, check logs, describe resources.",
        "tool_budget": 15,
    },
}

DEVOPS_WORKFLOWS = {
    "deploy_service": {
        "name": "Deploy Service",
        "description": "Deploy a service to production",
        "stages": ["assess", "plan", "apply", "verify"],
    },
    "rollback_deployment": {
        "name": "Rollback Deployment",
        "description": "Rollback a failed deployment",
        "stages": ["assess", "apply", "verify"],
    },
}

DevOpsToolProvider = make_tool_provider(
    "DevOpsToolProvider", DEVOPS_TOOLS, "Tool provider for the DevOps vertical."
)
DevOpsToolSelectionStrategy = make_stage_tool_strategy(
    "DevOpsToolSelectionStrategy",
    DEVOPS_STAGE_TOOLS,
    ["read", "shell"],
    "Stage-aware tool selection for DevOps tasks.",
)
DevOpsSafetyProvider = make_pattern_safety_provider(
    "DevOpsSafetyProvider",
    DEVOPS_DANGEROUS_PATTERNS,
    DEVOPS_TOOL_RESTRICTIONS,
    include_self=True,
    doc="Safety provider for DevOps infrastructure operations (registers itself as its extension).",
)
DevOpsPromptProvider = make_prompt_provider(
    "DevOpsPromptProvider",
    DEVOPS_PROMPT_SECTIONS,
    DEVOPS_TASK_TYPE_HINTS,
    "Prompt provider for DevOps infrastructure tasks.",
)
DevOpsWorkflowProvider = make_workflow_provider(
    "DevOpsWorkflowProvider", DEVOPS_WORKFLOWS, "Workflow provider for the DevOps vertical."
)


# =============================================================================
# Extended Protocol Implementations: Sandbox, Permissions, Hooks
# =============================================================================

# Import extension protocols (optional, for forward compatibility)
try:
    from victor_contracts.verticals.protocols import (
        SandboxProvider as SandboxProviderProtocol,
        PermissionProvider as PermissionProviderProtocol,
        HookProvider as HookProviderProtocol,
    )
except ImportError:
    SandboxProviderProtocol = None
    PermissionProviderProtocol = None
    HookProviderProtocol = None


class DevOpsSandboxProvider:
    """Sandbox configuration for DevOps vertical.

    DevOps requires broader access than coding — docker socket,
    cloud CLI configs, and network access for deployments.
    """

    def get_sandbox_config(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "filesystem_mode": "allow-list",
            "namespace_restrictions": True,
            "network_isolation": False,
            "allowed_mounts": [
                "/var/run/docker.sock",
                "~/.kube",
                "~/.aws",
                "~/.gcloud",
                "~/.terraform.d",
            ],
        }

    def get_tool_sandbox_overrides(self) -> Dict[str, Dict[str, Any]]:
        return {
            "kubectl_exec": {"network_isolation": False},
            "docker_run": {
                "filesystem_mode": "allow-list",
                "allowed_mounts": ["/var/run/docker.sock"],
            },
            "terraform_apply": {"network_isolation": False},
        }


class DevOpsPermissionProvider:
    """Permission configuration for DevOps vertical.

    DevOps starts at workspace-write but many tools require
    danger-full-access (deployment, infrastructure changes).
    """

    def get_permission_mode(self) -> str:
        return "workspace-write"

    def get_tool_permissions(self) -> Dict[str, str]:
        return {
            # Read-only
            "read": "read-only",
            "grep": "read-only",
            "kubectl_get": "read-only",
            "docker_ps": "read-only",
            "docker_logs": "read-only",
            "terraform_plan": "read-only",
            "aws_describe": "read-only",
            "prometheus_query": "read-only",
            # Workspace-write
            "write": "workspace-write",
            "edit": "workspace-write",
            "git_branch": "workspace-write",
            # Danger — destructive operations
            "shell": "danger-full-access",
            "kubectl_apply": "danger-full-access",
            "kubectl_delete": "danger-full-access",
            "kubectl_exec": "danger-full-access",
            "docker_run": "danger-full-access",
            "docker_rm": "danger-full-access",
            "terraform_apply": "danger-full-access",
            "terraform_destroy": "danger-full-access",
            "ansible_playbook": "danger-full-access",
            "aws_create": "danger-full-access",
            "aws_delete": "danger-full-access",
        }

    def get_permission_escalation_rules(self) -> List[Dict[str, Any]]:
        return [
            {
                "tool_pattern": "terraform_plan",
                "from_mode": "workspace-write",
                "to_mode": "read-only",
                "auto_approve": True,
            },
            {
                "tool_pattern": "kubectl_get*",
                "from_mode": "workspace-write",
                "to_mode": "read-only",
                "auto_approve": True,
            },
        ]


class DevOpsHookProvider:
    """Hook configuration for DevOps vertical.

    DevOps hooks focus on safety — preventing dangerous
    infrastructure operations without confirmation.
    """

    def get_pre_tool_hooks(self) -> List[str]:
        return []

    def get_post_tool_hooks(self) -> List[str]:
        return []


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "DevOpsToolProvider",
    "DevOpsToolSelectionStrategy",
    "DevOpsSafetyProvider",
    "DevOpsPromptProvider",
    "DevOpsWorkflowProvider",
    # Sandbox, permission, and hook providers
    "DevOpsSandboxProvider",
    "DevOpsPermissionProvider",
    "DevOpsHookProvider",
]
