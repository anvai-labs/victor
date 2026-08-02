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

"""Contract-only external vertical example for security analysis."""

from victor_contracts import (
    CapabilityIds,
    CapabilityRequirement,
    StageDefinition,
    ToolNames,
    ToolRequirement,
    VerticalBase,
    register_vertical,
)


@register_vertical("security", version="0.3.0")
class SecurityAssistant(VerticalBase):
    """Security analysis vertical authored against `victor_contracts` only."""

    name = "security"
    description = "Security auditing, vulnerability analysis, and compliance review"
    version = "0.3.0"
    VERTICAL_API_VERSION = 1

    @classmethod
    def get_name(cls) -> str:
        return cls.name

    @classmethod
    def get_description(cls) -> str:
        return cls.description

    @classmethod
    def get_tool_requirements(cls) -> list[ToolRequirement]:
        return [
            ToolRequirement(
                ToolNames.READ, purpose="inspect source and configuration files"
            ),
            ToolRequirement(ToolNames.LS, purpose="map project structure"),
            ToolRequirement(
                ToolNames.CODE_SEARCH,
                purpose="search for vulnerable patterns and secret material",
            ),
            ToolRequirement(
                ToolNames.OVERVIEW,
                required=False,
                purpose="summarize architecture before deeper analysis",
            ),
            ToolRequirement(
                ToolNames.SHELL,
                required=False,
                purpose="run security scanners such as bandit, trivy, or semgrep",
            ),
            ToolRequirement(
                ToolNames.WEB_SEARCH,
                required=False,
                purpose="look up CVEs, OWASP guidance, and vendor advisories",
            ),
            ToolRequirement(
                ToolNames.WRITE,
                required=False,
                purpose="generate a security findings report",
            ),
        ]

    @classmethod
    def get_tools(cls) -> list[str]:
        return [requirement.tool_name for requirement in cls.get_tool_requirements()]

    @classmethod
    def get_capability_requirements(cls) -> list[CapabilityRequirement]:
        return [
            CapabilityRequirement(
                capability_id=CapabilityIds.FILE_OPS,
                purpose="read repository contents and write reports when enabled",
            ),
            CapabilityRequirement(
                capability_id=CapabilityIds.GIT,
                optional=True,
                purpose="inspect repository history during incident review",
            ),
            CapabilityRequirement(
                capability_id=CapabilityIds.WEB_ACCESS,
                optional=True,
                purpose="resolve external security references and advisories",
            ),
        ]

    @classmethod
    def get_system_prompt(cls) -> str:
        return """You are a security specialist focused on source-code review,
dependency risk, secret detection, and secure remediation guidance.

Priorities:
1. Confirm findings with evidence before reporting them.
2. Prefer read-first investigation before shell-based scanning.
3. Explain severity and impact in practical terms.
4. Provide specific remediation steps, not generic advice.
5. Distinguish confirmed issues from possible follow-up checks.
"""

    @classmethod
    def get_prompt_templates(cls) -> dict[str, str]:
        return {
            "vulnerability_scan": (
                "Audit the target for vulnerabilities, explain severity, and cite "
                "concrete evidence."
            ),
            "dependency_audit": (
                "Identify vulnerable dependencies, affected packages, and "
                "remediation options."
            ),
            "incident_review": (
                "Review the repository for indicators of compromise and summarize "
                "the blast radius."
            ),
        }

    @classmethod
    def get_task_type_hints(cls) -> dict[str, dict[str, object]]:
        return {
            "vulnerability_scan": {
                "hint": (
                    "Start with architecture reconnaissance, then run targeted "
                    "searches and scanners."
                ),
                "tool_budget": 18,
                "priority_tools": [
                    ToolNames.READ,
                    ToolNames.CODE_SEARCH,
                    ToolNames.SHELL,
                ],
            },
            "dependency_audit": {
                "hint": (
                    "Inspect package manifests first, then confirm with dependency "
                    "scanners."
                ),
                "tool_budget": 14,
                "priority_tools": [
                    ToolNames.READ,
                    ToolNames.SHELL,
                    ToolNames.WEB_SEARCH,
                ],
            },
            "incident_review": {
                "hint": (
                    "Preserve evidence, avoid destructive actions, and summarize "
                    "confidence level."
                ),
                "tool_budget": 20,
                "priority_tools": [
                    ToolNames.READ,
                    ToolNames.CODE_SEARCH,
                    ToolNames.LS,
                ],
            },
        }

    @classmethod
    def get_stages(cls) -> dict[str, StageDefinition]:
        return {
            "reconnaissance": StageDefinition(
                name="reconnaissance",
                description="Understand the repository, stack, and attack surface.",
                required_tools=[ToolNames.READ, ToolNames.LS],
                optional_tools=[ToolNames.OVERVIEW, ToolNames.CODE_SEARCH],
            ),
            "analysis": StageDefinition(
                name="analysis",
                description=(
                    "Perform focused security review and scanner-assisted analysis."
                ),
                required_tools=[ToolNames.READ, ToolNames.CODE_SEARCH],
                optional_tools=[ToolNames.SHELL, ToolNames.WEB_SEARCH],
            ),
            "reporting": StageDefinition(
                name="reporting",
                description="Summarize findings, impact, and remediation guidance.",
                required_tools=[ToolNames.READ],
                optional_tools=[ToolNames.WRITE],
            ),
        }

    @classmethod
    def get_initial_stage(cls) -> str:
        return "reconnaissance"

    @classmethod
    def get_workflow_spec(cls) -> dict[str, object]:
        return {"stage_order": ["reconnaissance", "analysis", "reporting"]}

    @classmethod
    def get_team_declarations(cls) -> dict[str, dict[str, object]]:
        return {
            "security_review_team": {
                "name": "Security Review Team",
                "description": (
                    "Two-stage security review with analysis and validation."
                ),
                "formation": "pipeline",
                "members": [
                    {
                        "role": "researcher",
                        "goal": (
                            "Inspect the repository and identify likely security risks."
                        ),
                        "name": "Security Analyst",
                        "tool_budget": 18,
                        "backstory": (
                            "You specialize in application security reviews and "
                            "threat discovery."
                        ),
                        "expertise": [
                            "static analysis",
                            "dependency risk",
                            "secret detection",
                        ],
                        "personality": "methodical and evidence-driven",
                        "memory": True,
                    },
                    {
                        "role": "reviewer",
                        "goal": "Validate findings and summarize remediation guidance.",
                        "name": "Validation Reviewer",
                        "tool_budget": 12,
                        "backstory": (
                            "You verify findings before they are escalated to "
                            "developers."
                        ),
                        "expertise": ["severity calibration", "secure remediation"],
                        "personality": "skeptical and precise",
                    },
                ],
                "task_types": ["vulnerability_scan", "incident_review"],
                "tags": ["security", "review"],
            }
        }

    @classmethod
    def get_default_team(cls) -> str:
        return "security_review_team"

    @classmethod
    def get_provider_hints(cls) -> dict[str, object]:
        return {
            "preferred_providers": ["anthropic", "openai"],
            "requires_tool_calling": True,
            "prefers_extended_thinking": True,
        }

    @classmethod
    def get_evaluation_criteria(cls) -> list[str]:
        return [
            "finding_accuracy",
            "evidence_quality",
            "severity_calibration",
            "remediation_quality",
        ]
