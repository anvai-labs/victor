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

"""Research vertical workflows.

This package provides workflow definitions for common research tasks:
- Deep research with source verification
- Fact-checking
- Literature review
- Competitive analysis

Uses YAML-first architecture with Python escape hatches for complex conditions
and transforms that cannot be expressed in YAML.

Example:
    provider = ResearchWorkflowProvider()

    # Compile and execute (recommended - uses UnifiedWorkflowCompiler with caching)
    result = await provider.run_compiled_workflow("deep_research", {"query": "AI trends"})

    # Stream execution with real-time progress
    async for node_id, state in provider.stream_compiled_workflow("deep_research", context):
        print(f"Completed: {node_id}")

Available workflows (all YAML-defined):
- deep_research: Comprehensive research with source validation
- quick_research: Fast research for simple queries
- fact_check: Systematic fact verification
- literature_review: Academic literature review
- competitive_analysis: Market and competitive research
- competitive_scan: Quick competitive overview
"""

from typing import Optional

from victor_contracts.workflow_runtime import BaseYAMLWorkflowProvider


class ResearchWorkflowProvider(BaseYAMLWorkflowProvider):
    """Provides research-specific workflows.

    Uses YAML-first architecture with Python escape hatches for complex
    conditions and transforms that cannot be expressed in YAML.

    Inherits from BaseYAMLWorkflowProvider which provides:
    - YAML workflow loading with two-level caching
    - UnifiedWorkflowCompiler integration for consistent execution
    - Checkpointing support for resumable long-running research
    - Auto-workflow triggers via class attributes

    Example:
        provider = ResearchWorkflowProvider()

        # List available workflows
        print(provider.get_workflow_names())

        # Execute with caching (recommended)
        result = await provider.run_compiled_workflow("deep_research", {"query": "AI"})

        # Stream with real-time progress
        async for node_id, state in provider.stream_compiled_workflow("deep_research", {}):
            print(f"Completed: {node_id}")
    """

    # Auto-workflow triggers for research tasks
    AUTO_WORKFLOW_PATTERNS = [
        (r"deep\s+research", "deep_research"),
        (r"research\s+.*\s+thoroughly", "deep_research"),
        (r"comprehensive\s+research", "deep_research"),
        (r"fact\s*check", "fact_check"),
        (r"verify\s+(claim|statement)", "fact_check"),
        (r"is\s+it\s+true", "fact_check"),
        (r"literature\s+review", "literature_review"),
        (r"academic\s+review", "literature_review"),
        (r"papers?\s+on", "literature_review"),
        (r"competitive?\s+analysis", "competitive_analysis"),
        (r"market\s+research", "competitive_analysis"),
        (r"competitor", "competitive_analysis"),
    ]

    # Task type to workflow mappings
    TASK_TYPE_MAPPINGS = {
        "research": "deep_research",
        "fact_check": "fact_check",
        "verification": "fact_check",
        "literature": "literature_review",
        "academic": "literature_review",
        "competitive": "competitive_analysis",
        "market": "competitive_analysis",
    }

    def _get_escape_hatches_module(self) -> str:
        """Return the module path for research escape hatches.

        Returns:
            Module path string for CONDITIONS and TRANSFORMS dictionaries
        """
        return "victor_research.escape_hatches"

    def _get_capability_provider_module(self) -> Optional[str]:
        """Return the module path for the research capability provider.

        Returns:
            Module path string for ResearchCapabilityProvider
        """
        return "victor_research.capabilities"


__all__ = [
    # YAML-first workflow provider
    "ResearchWorkflowProvider",
]
