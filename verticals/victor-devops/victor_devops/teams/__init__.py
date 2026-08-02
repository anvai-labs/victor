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

"""Teams integration for DevOps vertical.

This package provides team specifications for common DevOps tasks with
rich persona attributes for natural agent characterization.

Example:
    from victor_devops.teams import (
        get_team_for_task,
        DEVOPS_TEAM_SPECS,
    )

    # Get team for a task type
    team_spec = get_team_for_task("deploy")
    print(f"Team: {team_spec.name}")
    print(f"Members: {len(team_spec.members)}")

Teams are auto-registered with the global TeamSpecRegistry on import,
enabling cross-vertical team discovery via:
    from victor.framework.team_registry import get_team_registry
    registry = get_team_registry()
    devops_teams = registry.find_by_vertical("devops")
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from victor_contracts.team_schema import TeamFormation, TeamMemberSpec, TeamSpec
from victor_contracts import TeamRegistryProtocol, get_default_team_registry


@dataclass
class DevOpsRoleConfig:
    """Configuration for a DevOps-specific role.

    Attributes:
        base_role: Base agent role (researcher, planner, executor, reviewer)
        tools: Tools available to this role
        tool_budget: Default tool budget
        description: Role description
    """

    base_role: str
    tools: List[str]
    tool_budget: int
    description: str = ""


# DevOps-specific roles with tool allocations
DEVOPS_ROLES: Dict[str, DevOpsRoleConfig] = {
    "infrastructure_assessor": DevOpsRoleConfig(
        base_role="researcher",
        tools=[
            "read_file",
            "ls",
            "shell",
            "grep",
            "git_status",
            "overview",
        ],
        tool_budget=20,
        description="Assesses current infrastructure state and configurations",
    ),
    "deployment_planner": DevOpsRoleConfig(
        base_role="planner",
        tools=[
            "read_file",
            "grep",
            "web_search",
            "web_fetch",
            "overview",
        ],
        tool_budget=15,
        description="Plans deployment and migration strategies",
    ),
    "infrastructure_engineer": DevOpsRoleConfig(
        base_role="executor",
        tools=[
            "read_file",
            "write_file",
            "edit_files",
            "shell",
            "docker",
            "git_status",
            "git_diff",
        ],
        tool_budget=35,
        description="Implements infrastructure configurations",
    ),
    "deployment_validator": DevOpsRoleConfig(
        base_role="reviewer",
        tools=[
            "shell",
            "read_file",
            "docker",
            "test",
            "git_diff",
        ],
        tool_budget=25,
        description="Validates deployments and runs infrastructure tests",
    ),
    "container_specialist": DevOpsRoleConfig(
        base_role="executor",
        tools=[
            "read_file",
            "write_file",
            "edit_files",
            "shell",
            "docker",
        ],
        tool_budget=30,
        description="Creates and optimizes container configurations",
    ),
    "monitoring_engineer": DevOpsRoleConfig(
        base_role="executor",
        tools=[
            "read_file",
            "write_file",
            "edit_files",
            "shell",
            "web_fetch",
        ],
        tool_budget=30,
        description="Configures monitoring and observability stacks",
    ),
    "security_reviewer": DevOpsRoleConfig(
        base_role="researcher",
        tools=[
            "read_file",
            "grep",
            "shell",
            "web_search",
        ],
        tool_budget=20,
        description="Reviews infrastructure for security issues",
    ),
}


# Pre-defined team specifications with rich personas
DEVOPS_TEAM_SPECS: Dict[str, TeamSpec] = {
    "deployment_team": TeamSpec(
        name="Infrastructure Deployment Team",
        vertical="devops",
        description="End-to-end infrastructure deployment with assessment, planning, implementation, and validation",
        formation=TeamFormation.PIPELINE,
        members=[
            TeamMemberSpec(
                role="researcher",
                goal="Assess current infrastructure state and identify dependencies",
                name="Infrastructure Assessor",
                tool_budget=20,
                backstory=(
                    "You are a seasoned infrastructure architect who has assessed hundreds of "
                    "production environments. You have an encyclopedic knowledge of cloud services, "
                    "on-prem configurations, and hybrid architectures. You notice subtle configuration "
                    "drift and dependency issues that others miss. Your assessments are thorough and "
                    "you never make assumptions without verifying the actual infrastructure state."
                ),
                expertise=[
                    "AWS",
                    "GCP",
                    "Azure",
                    "Terraform",
                    "Kubernetes",
                    "networking",
                ],
                personality="methodical and thorough; documents findings with severity ratings",
                memory=True,  # Persist discoveries for team
            ),
            TeamMemberSpec(
                role="planner",
                goal="Design deployment strategy with rollback plan",
                name="Deployment Strategist",
                tool_budget=15,
                backstory=(
                    "You are a deployment strategist who has orchestrated zero-downtime migrations "
                    "for Fortune 500 companies. You think in terms of blast radius, rollback windows, "
                    "and circuit breakers. You design deployment plans that are incremental, testable, "
                    "and reversible. You consider failure modes and have a plan B for every step. "
                    "Your plans include clear success criteria and monitoring checkpoints."
                ),
                expertise=[
                    "blue-green deployments",
                    "canary releases",
                    "feature flags",
                    "rollback strategies",
                ],
                personality="cautious yet decisive; balances speed with safety",
                memory=True,  # Plan builds on assessment
            ),
            TeamMemberSpec(
                role="executor",
                goal="Implement infrastructure changes following the deployment plan",
                name="Infrastructure Engineer",
                tool_budget=35,
                backstory=(
                    "You are a hands-on infrastructure engineer who writes declarative, idempotent "
                    "configurations. You've managed infrastructure at scale and know the difference "
                    "between what works in dev and what survives production. You use multi-stage "
                    "builds, pin versions, include health checks, and document every configuration "
                    "decision. Your Infrastructure as Code is clean, versioned, and reproducible."
                ),
                expertise=[
                    "Terraform",
                    "Docker",
                    "Kubernetes",
                    "Ansible",
                    "CI/CD pipelines",
                ],
                personality="precise and efficient; prefers declarative over imperative",
                cache=True,  # Cache file reads for efficiency
            ),
            TeamMemberSpec(
                role="reviewer",
                goal="Validate deployment and run infrastructure tests",
                name="Deployment Validator",
                tool_budget=25,
                backstory=(
                    "You are a deployment gatekeeper who has prevented countless production incidents. "
                    "You verify configurations against security baselines, run smoke tests, and check "
                    "resource limits. You validate that services are healthy, endpoints respond, and "
                    "logs are flowing. You don't approve deployments that skip validation steps or "
                    "lack proper monitoring. Your sign-off means the deployment is production-ready."
                ),
                expertise=[
                    "infrastructure testing",
                    "compliance",
                    "security scanning",
                    "monitoring",
                ],
                personality="detail-oriented and skeptical; trusts but verifies",
            ),
        ],
        total_tool_budget=95,
    ),
    "container_team": TeamSpec(
        name="Container Management Team",
        vertical="devops",
        description="Docker container setup, optimization, and management with security best practices",
        formation=TeamFormation.PIPELINE,
        members=[
            TeamMemberSpec(
                role="researcher",
                goal="Analyze application requirements for containerization",
                name="Containerization Analyst",
                tool_budget=15,
                backstory=(
                    "You are a containerization expert who has migrated legacy applications to "
                    "containers. You understand the nuances of process isolation, filesystem "
                    "requirements, and environment dependencies. You identify what needs to be "
                    "containerized, what can share base images, and what should stay external. "
                    "You spot anti-patterns like running as root or storing state in containers."
                ),
                expertise=[
                    "Docker",
                    "OCI standards",
                    "container runtimes",
                    "Linux namespaces",
                ],
                personality="analytical and pragmatic; focuses on containerization readiness",
                memory=True,  # Share analysis with engineer
            ),
            TeamMemberSpec(
                role="executor",
                goal="Create optimized, secure Dockerfiles and compose configurations",
                name="Container Engineer",
                tool_budget=30,
                backstory=(
                    "You are a Docker craftsman who writes Dockerfiles that are small, fast, and "
                    "secure. You use multi-stage builds, minimize layers, leverage build cache "
                    "effectively, and never run as root. You choose appropriate base images, "
                    "handle signals correctly, and configure health checks. Your containers start "
                    "fast, use minimal resources, and follow the 12-factor app principles."
                ),
                expertise=[
                    "Docker",
                    "multi-stage builds",
                    "image optimization",
                    "security hardening",
                ],
                personality="perfectionist about image size and security",
            ),
            TeamMemberSpec(
                role="reviewer",
                goal="Build, test, and validate containers meet security and performance standards",
                name="Container Tester",
                tool_budget=20,
                backstory=(
                    "You are a container quality gatekeeper who ensures images are production-ready. "
                    "You run vulnerability scans, check for secrets in images, verify resource limits, "
                    "and test startup/shutdown behavior. You validate that containers work in "
                    "orchestration environments, handle restarts gracefully, and produce proper logs. "
                    "You reject images that don't meet security baselines."
                ),
                expertise=[
                    "container scanning",
                    "Trivy",
                    "Clair",
                    "container orchestration",
                ],
                personality="security-conscious; rejects insecure configurations",
            ),
        ],
        total_tool_budget=65,
    ),
    "monitoring_team": TeamSpec(
        name="Observability Team",
        vertical="devops",
        description="Comprehensive monitoring, logging, and alerting setup",
        formation=TeamFormation.SEQUENTIAL,
        members=[
            TeamMemberSpec(
                role="researcher",
                goal="Analyze monitoring requirements and identify key metrics",
                name="Observability Analyst",
                tool_budget=15,
                backstory=(
                    "You are an observability expert who understands that monitoring is about "
                    "answering questions before they're asked. You identify the golden signals "
                    "(latency, traffic, errors, saturation) for each service. You determine "
                    "what SLIs matter for SLOs, what logs need to be retained, and what traces "
                    "provide value. You design monitoring that enables quick incident response."
                ),
                expertise=[
                    "Prometheus",
                    "Grafana",
                    "ELK stack",
                    "distributed tracing",
                    "SRE practices",
                ],
                personality="metrics-driven; thinks in terms of SLOs and error budgets",
                memory=True,  # Share requirements with engineer
            ),
            TeamMemberSpec(
                role="executor",
                goal="Configure monitoring stack with dashboards and alerts",
                name="Monitoring Engineer",
                tool_budget=30,
                backstory=(
                    "You are a monitoring engineer who builds observability platforms that scale. "
                    "You configure Prometheus exporters, create meaningful Grafana dashboards, "
                    "set up log aggregation, and define actionable alerts. Your alerts have "
                    "clear runbooks, your dashboards tell stories, and your metrics have proper "
                    "cardinality. You avoid alert fatigue by setting meaningful thresholds."
                ),
                expertise=[
                    "Prometheus",
                    "Grafana",
                    "Alertmanager",
                    "Loki",
                    "OpenTelemetry",
                ],
                personality="systematic; believes in infrastructure as code for monitoring too",
            ),
        ],
        total_tool_budget=45,
    ),
    "cicd_team": TeamSpec(
        name="CI/CD Pipeline Team",
        vertical="devops",
        description="Continuous integration and deployment pipeline setup",
        formation=TeamFormation.PIPELINE,
        members=[
            TeamMemberSpec(
                role="researcher",
                goal="Analyze existing CI/CD setup and requirements",
                name="Pipeline Analyst",
                tool_budget=15,
                backstory=(
                    "You are a CI/CD specialist who has designed pipelines for monorepos, "
                    "microservices, and everything in between. You understand build dependencies, "
                    "test parallelization, and deployment strategies. You identify bottlenecks "
                    "in existing pipelines and opportunities for faster feedback loops."
                ),
                expertise=[
                    "GitHub Actions",
                    "GitLab CI",
                    "Jenkins",
                    "ArgoCD",
                    "build optimization",
                ],
                personality="efficiency-focused; hates slow pipelines",
                memory=True,
            ),
            TeamMemberSpec(
                role="planner",
                goal="Design CI/CD pipeline stages and workflows",
                name="Pipeline Architect",
                tool_budget=10,
                backstory=(
                    "You are a pipeline architect who designs workflows that are fast, reliable, "
                    "and secure. You structure pipelines with proper stages: build, test, security "
                    "scan, deploy. You implement caching strategies, parallelize where possible, "
                    "and design for both speed and safety."
                ),
                expertise=["pipeline design", "DevSecOps", "artifact management"],
                personality="structured thinker; balances speed with quality gates",
            ),
            TeamMemberSpec(
                role="executor",
                goal="Implement CI/CD pipeline configurations",
                name="Pipeline Engineer",
                tool_budget=30,
                backstory=(
                    "You are a pipeline engineer who writes YAML that works first time. You "
                    "understand the quirks of different CI systems, handle secrets properly, "
                    "configure matrix builds, and set up proper caching. Your pipelines are "
                    "maintainable, well-documented, and follow DRY principles."
                ),
                expertise=["GitHub Actions", "GitLab CI", "Jenkins", "shell scripting"],
                personality="practical; writes pipelines that are easy to debug",
            ),
            TeamMemberSpec(
                role="reviewer",
                goal="Validate pipeline security and functionality",
                name="Pipeline Validator",
                tool_budget=15,
                backstory=(
                    "You are a pipeline security reviewer who ensures pipelines don't become "
                    "attack vectors. You check for exposed secrets, verify least-privilege "
                    "permissions, and validate that security scans are in place. You test "
                    "that pipelines fail correctly and don't deploy broken code."
                ),
                expertise=["DevSecOps", "secret management", "SAST/DAST"],
                personality="security-minded; suspicious of overly permissive pipelines",
            ),
        ],
        total_tool_budget=70,
    ),
    "security_audit_team": TeamSpec(
        name="Security Audit Team",
        vertical="devops",
        description="Infrastructure security assessment and hardening",
        formation=TeamFormation.PARALLEL,
        members=[
            TeamMemberSpec(
                role="researcher",
                goal="Scan infrastructure for security vulnerabilities",
                name="Vulnerability Scanner",
                tool_budget=20,
                backstory=(
                    "You are a security scanner who thinks like an attacker. You look for "
                    "exposed ports, default credentials, unpatched systems, and misconfigurations. "
                    "You know the OWASP Top 10, CIS benchmarks, and cloud security best practices. "
                    "You prioritize findings by exploitability and impact."
                ),
                expertise=[
                    "vulnerability scanning",
                    "penetration testing",
                    "CIS benchmarks",
                ],
                personality="thorough and paranoid; assumes everything is vulnerable",
            ),
            TeamMemberSpec(
                role="researcher",
                goal="Review configurations for security best practices",
                name="Configuration Auditor",
                tool_budget=20,
                backstory=(
                    "You are a configuration auditor who spots insecure settings. You check "
                    "encryption at rest and in transit, validate IAM policies follow least "
                    "privilege, and ensure secrets are properly managed. You verify network "
                    "segmentation and firewall rules."
                ),
                expertise=["IAM", "encryption", "network security", "compliance"],
                personality="detail-oriented; nothing escapes scrutiny",
            ),
            TeamMemberSpec(
                role="planner",
                goal="Synthesize findings into remediation plan",
                name="Security Remediation Planner",
                tool_budget=10,
                backstory=(
                    "You are a security strategist who turns findings into actionable plans. "
                    "You prioritize remediation by risk, consider operational impact, and "
                    "create phased approaches. Your plans are specific, achievable, and "
                    "include verification steps."
                ),
                expertise=[
                    "risk assessment",
                    "remediation planning",
                    "security roadmaps",
                ],
                personality="pragmatic; balances security with operability",
            ),
        ],
        total_tool_budget=50,
    ),
}


def get_team_for_task(task_type: str) -> Optional[TeamSpec]:
    """Get appropriate team specification for task type.

    Args:
        task_type: Type of task (deploy, container, monitor, cicd, security, etc.)

    Returns:
        TeamSpec or None if no matching team
    """
    mapping = {
        # Deployment tasks
        "deploy": "deployment_team",
        "deployment": "deployment_team",
        "infrastructure": "deployment_team",
        "terraform": "deployment_team",
        "provision": "deployment_team",
        # Container tasks
        "container": "container_team",
        "docker": "container_team",
        "containerization": "container_team",
        "dockerfile": "container_team",
        "image": "container_team",
        # Monitoring tasks
        "monitor": "monitoring_team",
        "monitoring": "monitoring_team",
        "observability": "monitoring_team",
        "metrics": "monitoring_team",
        "alerting": "monitoring_team",
        # CI/CD tasks
        "cicd": "cicd_team",
        "ci/cd": "cicd_team",
        "pipeline": "cicd_team",
        "github_actions": "cicd_team",
        "gitlab_ci": "cicd_team",
        "jenkins": "cicd_team",
        # Security tasks
        "security": "security_audit_team",
        "audit": "security_audit_team",
        "vulnerability": "security_audit_team",
        "hardening": "security_audit_team",
    }
    spec_name = mapping.get(task_type.lower())
    if spec_name:
        return DEVOPS_TEAM_SPECS.get(spec_name)
    return None


def get_role_config(role_name: str) -> Optional[DevOpsRoleConfig]:
    """Get configuration for a DevOps role.

    Args:
        role_name: Role name

    Returns:
        DevOpsRoleConfig or None
    """
    return DEVOPS_ROLES.get(role_name.lower())


def list_team_types() -> List[str]:
    """List all available team types.

    Returns:
        List of team type names
    """
    return list(DEVOPS_TEAM_SPECS.keys())


def list_roles() -> List[str]:
    """List all available DevOps roles.

    Returns:
        List of role names
    """
    return list(DEVOPS_ROLES.keys())


class DevOpsTeamSpecProvider:
    """Team specification provider for DevOps vertical.

    Implements TeamSpecProviderProtocol interface for consistent
    ISP compliance across all verticals.
    """

    def get_team_specs(self) -> Dict[str, TeamSpec]:
        """Get all DevOps team specifications.

        Returns:
            Dictionary mapping team names to TeamSpec instances
        """
        return DEVOPS_TEAM_SPECS

    def get_team_for_task(self, task_type: str) -> Optional[TeamSpec]:
        """Get appropriate team for a task type.

        Args:
            task_type: Type of task

        Returns:
            TeamSpec or None if no matching team
        """
        return get_team_for_task(task_type)

    def list_team_types(self) -> List[str]:
        """List all available team types.

        Returns:
            List of team type names
        """
        return list_team_types()


__all__ = [
    # Types
    "DevOpsRoleConfig",
    "TeamSpec",
    # Provider
    "DevOpsTeamSpecProvider",
    # Role configurations
    "DEVOPS_ROLES",
    # Team specifications
    "DEVOPS_TEAM_SPECS",
    # Helper functions
    "get_team_for_task",
    "get_role_config",
    "list_team_types",
    "list_roles",
]

logger = logging.getLogger(__name__)


def register_devops_teams(registry: TeamRegistryProtocol | None = None) -> int:
    """Register DevOps teams with global registry.

    This function is called during vertical integration by the framework's
    step handlers. Import-time auto-registration has been removed to avoid
    load-order coupling and duplicate registration.

    Returns:
        Number of teams registered.
    """
    try:
        registry = registry or get_default_team_registry()
        if registry is None:
            logger.debug("No default team registry available; skipping DevOps team registration")
            return 0
        count = registry.register_from_vertical("devops", DEVOPS_TEAM_SPECS)
        logger.debug(f"Registered {count} DevOps teams via framework integration")
        return count
    except Exception as e:
        logger.warning(f"Failed to register DevOps teams: {e}")
        return 0


# NOTE: Import-time auto-registration removed (SOLID compliance)
# Registration now happens during vertical integration via step_handlers.py
# This avoids load-order coupling and duplicate registration issues.
