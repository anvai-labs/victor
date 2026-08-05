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

"""DevOps vertical enrichment strategy.

Provides prompt enrichment using infrastructure context such as:
- Docker/Kubernetes configuration files
- CI/CD pipeline patterns
- Infrastructure-as-code context (Terraform, CloudFormation)
- Command history patterns from similar tasks

Example:
    from victor_devops.enrichment import DevOpsEnrichmentStrategy

    # Create strategy
    strategy = DevOpsEnrichmentStrategy()

    # Register with enrichment service
    enrichment_service.register_strategy("devops", strategy)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from victor_contracts.enrichment_runtime import (
    ContextEnrichment,
    EnrichmentContext,
    EnrichmentPriority,
    EnrichmentType,
    FilePatternMatcher,
    DEVOPS_PATTERNS,
    KeywordClassifier,
    INFRA_TYPES,
)

logger = logging.getLogger(__name__)

# Use framework pattern matcher with DEVOPS_PATTERNS
_file_pattern_matcher = FilePatternMatcher(DEVOPS_PATTERNS)

# Use framework keyword classifier with INFRA_TYPES
_keyword_classifier = KeywordClassifier(INFRA_TYPES)


def _detect_infra_context(
    file_mentions: List[str],
    prompt: str,
) -> Dict[str, List[str]]:
    """Detect infrastructure context from file mentions and prompt.

    Uses the framework's FilePatternMatcher and KeywordClassifier utilities
    for consistent pattern matching across verticals.

    Args:
        file_mentions: Files mentioned in context
        prompt: The prompt text

    Returns:
        Dict mapping infra type to relevant file paths
    """
    # Initialize context with categories from DEVOPS_PATTERNS
    context: Dict[str, List[str]] = {
        "docker": [],
        "kubernetes": [],
        "terraform": [],
        "ci_cd": [],
        "ansible": [],
        "helm": [],
    }

    # Use framework FilePatternMatcher for file-based detection
    if file_mentions:
        matched = _file_pattern_matcher.match(file_mentions)
        for category, files in matched.items():
            if category in context:
                context[category].extend(files)

    # Use framework KeywordClassifier for prompt-based detection
    detected_types = _keyword_classifier.classify(prompt)
    for infra_type in detected_types:
        if infra_type in context:
            context[infra_type].append("_prompt_mention")

    return context


class DevOpsEnrichmentStrategy:
    """Enrichment strategy for the DevOps vertical.

    Provides infrastructure context and best practices for
    deployment, CI/CD, and infrastructure management tasks.

    Attributes:
        project_root: Root directory for file discovery
        max_files: Maximum infrastructure files to include
    """

    def __init__(
        self,
        project_root: Optional[Path] = None,
        max_files: int = 3,
    ):
        """Initialize the DevOps enrichment strategy.

        Args:
            project_root: Project root for file discovery
            max_files: Max infrastructure files to analyze (default: 3)
        """
        self._project_root = project_root or Path.cwd()
        self._max_files = max_files

    def set_project_root(self, root: Path) -> None:
        """Set the project root directory.

        Args:
            root: Project root path
        """
        self._project_root = root

    async def get_enrichments(
        self,
        prompt: str,
        context: EnrichmentContext,
    ) -> List[ContextEnrichment]:
        """Get enrichments for a DevOps prompt.

        Detects infrastructure context and provides relevant
        configuration examples and best practices.

        Args:
            prompt: The prompt to enrich
            context: Enrichment context with file/tool mentions

        Returns:
            List of context enrichments
        """
        enrichments: List[ContextEnrichment] = []

        # Detect infrastructure context
        infra_context = _detect_infra_context(
            context.file_mentions,
            prompt,
        )

        # Build enrichments based on detected context
        try:
            # Docker context
            if infra_context["docker"]:
                docker_enrichment = self._build_docker_enrichment(infra_context["docker"])
                if docker_enrichment:
                    enrichments.append(docker_enrichment)

            # Kubernetes context
            if infra_context["kubernetes"]:
                k8s_enrichment = self._build_kubernetes_enrichment(infra_context["kubernetes"])
                if k8s_enrichment:
                    enrichments.append(k8s_enrichment)

            # Terraform context
            if infra_context["terraform"]:
                tf_enrichment = self._build_terraform_enrichment(infra_context["terraform"])
                if tf_enrichment:
                    enrichments.append(tf_enrichment)

            # CI/CD context
            if infra_context["ci_cd"]:
                cicd_enrichment = self._build_cicd_enrichment(infra_context["ci_cd"])
                if cicd_enrichment:
                    enrichments.append(cicd_enrichment)

            # Add command history context if available
            if context.tool_history:
                history_enrichment = self._enrich_from_tool_history(context.tool_history)
                if history_enrichment:
                    enrichments.append(history_enrichment)

        except Exception as e:
            logger.warning("Error during DevOps enrichment: %s", e)

        logger.debug(
            "DevOps enrichment produced %d enrichments for task_type=%s",
            len(enrichments),
            context.task_type,
        )

        return enrichments

    def _build_docker_enrichment(
        self,
        files: List[str],
    ) -> Optional[ContextEnrichment]:
        """Build Docker-related enrichment.

        Args:
            files: Docker-related file paths

        Returns:
            Enrichment with Docker best practices
        """
        content_parts = [
            "Docker best practices for this task:",
            "- Use multi-stage builds to reduce image size",
            "- Run as non-root user for security",
            "- Add health checks for container orchestration",
            "- Use .dockerignore to exclude unnecessary files",
            "- Pin base image versions (avoid :latest)",
        ]

        if "_prompt_mention" not in files:
            content_parts.append(
                f"\nRelevant files: {', '.join(f for f in files if f != '_prompt_mention')}"
            )

        return ContextEnrichment(
            type=EnrichmentType.PROJECT_CONTEXT,
            content="\n".join(content_parts),
            priority=EnrichmentPriority.NORMAL,
            source="devops_docker",
            metadata={"infra_type": "docker"},
        )

    def _build_kubernetes_enrichment(
        self,
        files: List[str],
    ) -> Optional[ContextEnrichment]:
        """Build Kubernetes-related enrichment.

        Args:
            files: Kubernetes-related file paths

        Returns:
            Enrichment with Kubernetes best practices
        """
        content_parts = [
            "Kubernetes best practices for this task:",
            "- Define resource requests and limits",
            "- Add liveness and readiness probes",
            "- Use ConfigMaps for configuration, Secrets for sensitive data",
            "- Implement NetworkPolicies for security",
            "- Use namespaces for isolation",
        ]

        if "_prompt_mention" not in files:
            content_parts.append(
                f"\nRelevant files: {', '.join(f for f in files if f != '_prompt_mention')}"
            )

        return ContextEnrichment(
            type=EnrichmentType.PROJECT_CONTEXT,
            content="\n".join(content_parts),
            priority=EnrichmentPriority.NORMAL,
            source="devops_kubernetes",
            metadata={"infra_type": "kubernetes"},
        )

    def _build_terraform_enrichment(
        self,
        files: List[str],
    ) -> Optional[ContextEnrichment]:
        """Build Terraform-related enrichment.

        Args:
            files: Terraform-related file paths

        Returns:
            Enrichment with Terraform best practices
        """
        content_parts = [
            "Terraform best practices for this task:",
            "- Use modules for reusable components",
            "- Store state remotely with locking (S3+DynamoDB)",
            "- Use workspaces or directories for environments",
            "- Validate with terraform fmt and terraform validate",
            "- Tag all resources for cost tracking",
        ]

        if "_prompt_mention" not in files:
            content_parts.append(
                f"\nRelevant files: {', '.join(f for f in files if f != '_prompt_mention')}"
            )

        return ContextEnrichment(
            type=EnrichmentType.PROJECT_CONTEXT,
            content="\n".join(content_parts),
            priority=EnrichmentPriority.NORMAL,
            source="devops_terraform",
            metadata={"infra_type": "terraform"},
        )

    def _build_cicd_enrichment(
        self,
        files: List[str],
    ) -> Optional[ContextEnrichment]:
        """Build CI/CD-related enrichment.

        Args:
            files: CI/CD-related file paths

        Returns:
            Enrichment with CI/CD best practices
        """
        content_parts = [
            "CI/CD best practices for this task:",
            "- Define clear stages: lint, test, build, deploy",
            "- Cache dependencies for faster builds",
            "- Use environment-specific configurations",
            "- Implement proper secret management",
            "- Add manual approval for production deployments",
        ]

        if "_prompt_mention" not in files:
            content_parts.append(
                f"\nRelevant files: {', '.join(f for f in files if f != '_prompt_mention')}"
            )

        return ContextEnrichment(
            type=EnrichmentType.PROJECT_CONTEXT,
            content="\n".join(content_parts),
            priority=EnrichmentPriority.NORMAL,
            source="devops_cicd",
            metadata={"infra_type": "ci_cd"},
        )

    def _enrich_from_tool_history(
        self,
        tool_history: List[Dict[str, Any]],
    ) -> Optional[ContextEnrichment]:
        """Enrich from previous bash/command tool results.

        Args:
            tool_history: List of recent tool calls

        Returns:
            Enrichment with successful command patterns, or None
        """
        successful_commands = []

        for call in tool_history[-15:]:  # Last 15 calls
            tool_name = call.get("tool", "")

            if tool_name in ("bash", "execute_command"):
                result = call.get("result", {})
                if isinstance(result, dict) and result.get("success"):
                    command = call.get("arguments", {}).get("command", "")
                    # Only include relevant DevOps commands
                    if any(
                        kw in command
                        for kw in [
                            "docker",
                            "kubectl",
                            "terraform",
                            "helm",
                            "ansible",
                            "aws",
                            "gcloud",
                            "az",
                        ]
                    ):
                        successful_commands.append(command)

        if not successful_commands:
            return None

        content_parts = ["Successful commands from this session:"]
        for cmd in successful_commands[-5:]:  # Last 5 commands
            content_parts.append(f"- `{cmd}`")

        return ContextEnrichment(
            type=EnrichmentType.TOOL_HISTORY,
            content="\n".join(content_parts),
            priority=EnrichmentPriority.LOW,
            source="command_history",
            metadata={"commands_count": len(successful_commands)},
        )

    def get_priority(self) -> int:
        """Get strategy priority.

        Returns:
            Priority value (50 = normal)
        """
        return 50

    def get_token_allocation(self) -> float:
        """Get token budget allocation.

        Returns:
            Fraction of token budget (0.3 = 30%)
        """
        return 0.3


__all__ = [
    "DevOpsEnrichmentStrategy",
]
