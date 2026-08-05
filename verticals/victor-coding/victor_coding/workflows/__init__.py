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

"""Coding vertical workflows.

This package provides workflow definitions for common coding tasks using
YAML-first architecture with Python escape hatches for complex conditions.

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

Usage:
    from victor_coding.workflows import CodingWorkflowProvider

    provider = CodingWorkflowProvider()

    # List available workflows
    print(provider.get_workflow_names())

    # Execute with caching (recommended - uses UnifiedWorkflowCompiler)
    result = await provider.run_compiled_workflow("code_review", {"files": ["src/"]})

    # Stream with real-time progress
    async for node_id, state in provider.stream_compiled_workflow("code_review", {}):
        print(f"Completed: {node_id}")

This package also provides LCEL-composed tool chains for fine-grained
code operations (explore, analyze, edit, refactor, etc.).
"""

from victor_coding.workflows.provider import CodingWorkflowProvider

# LCEL-composed tool chains
from victor_coding.composed_chains import (
    # Pre-built chains
    explore_file_chain,
    analyze_function_chain,
    safe_edit_chain,
    git_status_chain,
    search_with_context_chain,
    lint_chain,
    test_discovery_chain,
    review_analysis_chain,
    # Factories
    create_exploration_chain,
    create_edit_verify_chain,
    create_refactor_chain,
    # Registry
    CODING_CHAINS,
    get_chain,
    list_chains,
    # Lazy tool loading
    lazy_tool,
    LazyToolRunnable,
)

__all__ = [
    # YAML-first workflow provider
    "CodingWorkflowProvider",
    # LCEL-composed chains
    "explore_file_chain",
    "analyze_function_chain",
    "safe_edit_chain",
    "git_status_chain",
    "search_with_context_chain",
    "lint_chain",
    "test_discovery_chain",
    "review_analysis_chain",
    # Chain factories
    "create_exploration_chain",
    "create_edit_verify_chain",
    "create_refactor_chain",
    # Chain registry
    "CODING_CHAINS",
    "get_chain",
    "list_chains",
    # Lazy tool loading
    "lazy_tool",
    "LazyToolRunnable",
]
