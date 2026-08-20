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

"""Coding workflow provider.

Uses YAML-first architecture with Python escape hatches for complex
conditions and transforms that cannot be expressed in YAML.

All workflows are defined in YAML files (*.yaml) in this directory:
- feature.yaml: Feature implementation workflows
- bugfix.yaml: Bug fix workflows
- code_review.yaml: Code review workflows (includes pr_review)
- tdd.yaml: Test-driven development workflows
- refactor.yaml: Refactoring workflows

Escape hatches for complex conditions/transforms are in:
- victor/coding/escape_hatches.py

Example:
    provider = CodingWorkflowProvider()

    # Compile and execute (recommended - uses UnifiedWorkflowCompiler with caching)
    result = await provider.run_compiled_workflow("code_review", {"files": ["src/"]})

    # Stream execution with real-time progress
    async for node_id, state in provider.stream_compiled_workflow("code_review", context):
        print(f"Completed: {node_id}")

Available workflows (all YAML-defined):
- feature_implementation: Full feature implementation workflow
- quick_feature: Quick feature for small changes
- bug_fix: Systematic bug investigation and fix
- quick_fix: Quick fix for simple bugs
- code_review: Comprehensive code review with parallel analysis
- quick_review: Quick code review for small changes
- pr_review: Pull request review workflow
- tdd_cycle: Test-driven development cycle
- refactor: Code refactoring workflow
"""

import logging

from victor_contracts.workflow_runtime import BaseYAMLWorkflowProvider

logger = logging.getLogger(__name__)


class CodingWorkflowProvider(BaseYAMLWorkflowProvider):
    """Provides coding-specific workflows.

    Uses YAML-first architecture with Python escape hatches for complex
    conditions and transforms that cannot be expressed in YAML.

    Inherits from BaseYAMLWorkflowProvider which provides:
    - YAML workflow loading with two-level caching
    - UnifiedWorkflowCompiler integration for consistent execution
    - Checkpointing support for resumable code reviews and TDD cycles
    - Auto-workflow triggers via class attributes

    Example:
        provider = CodingWorkflowProvider()

        # List available workflows
        print(provider.get_workflow_names())

        # Execute with caching (recommended)
        result = await provider.run_compiled_workflow("code_review", {"files": ["src/"]})

        # Stream with real-time progress
        async for node_id, state in provider.stream_compiled_workflow("code_review", {}):
            print(f"Completed: {node_id}")
    """

    # Auto-workflow triggers for coding tasks
    AUTO_WORKFLOW_PATTERNS = [
        # Feature patterns
        (r"implement\s+.+feature", "feature_implementation"),
        (r"add\s+.+feature", "feature_implementation"),
        (r"create\s+.+feature", "feature_implementation"),
        (r"build\s+new\s+", "feature_implementation"),
        (r"quick(ly)?\s+implement", "quick_feature"),
        (r"simple\s+feature", "quick_feature"),
        # Bug fix patterns
        (r"fix\s+.+bug", "bug_fix"),
        (r"debug\s+", "bug_fix"),
        (r"investigate\s+.+issue", "bug_fix"),
        (r"quick(ly)?\s+fix", "quick_fix"),
        (r"simple\s+fix", "quick_fix"),
        # Review patterns
        (r"review\s+.+code", "code_review"),
        (r"code\s+review", "code_review"),
        (r"comprehensive\s+review", "code_review"),
        (r"quick(ly)?\s+review", "quick_review"),
        (r"review\s+.+pr", "pr_review"),
        (r"review\s+pull\s+request", "pr_review"),
        (r"pr\s+review", "pr_review"),
        # TDD patterns
        (r"tdd\s+", "tdd_cycle"),
        (r"test.?driven", "tdd_cycle"),
        (r"write\s+tests?\s+first", "tdd_cycle"),
        # Refactor patterns
        (r"refactor\s+", "refactor"),
        (r"clean\s+up\s+code", "refactor"),
        (r"improve\s+code\s+quality", "refactor"),
    ]

    # Task type to workflow mappings
    TASK_TYPE_MAPPINGS = {
        # Feature workflows
        "feature": "feature_implementation",
        "implement": "feature_implementation",
        "simple_feature": "quick_feature",
        "quick_feature": "quick_feature",
        # Bug fix workflows
        "bug": "bug_fix",
        "bugfix": "bug_fix",
        "debug": "bug_fix",
        "investigation": "bug_fix",
        "simple_bug": "quick_fix",
        "quick_fix": "quick_fix",
        # Review workflows
        "review": "code_review",
        "code_review": "code_review",
        "security_review": "code_review",
        "quick_review": "quick_review",
        "pr": "pr_review",
        "pull_request": "pr_review",
        "pr_review": "pr_review",
        # TDD workflows
        "tdd": "tdd_cycle",
        "test_driven": "tdd_cycle",
        # Refactor workflows
        "refactor": "refactor",
        "cleanup": "refactor",
    }

    def _get_escape_hatches_module(self) -> str:
        """Return the module path for coding escape hatches.

        Returns:
            Module path string for CONDITIONS and TRANSFORMS dictionaries
        """
        return "victor_coding.escape_hatches"


__all__ = [
    "CodingWorkflowProvider",
]
