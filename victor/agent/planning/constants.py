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

"""Constants for planning system configuration.

Centralized constants for tool selection, complexity limits, and step mappings.
These values can be overridden via environment variables or settings.
"""

from typing import Dict, Set

# =============================================================================
# Complexity Tool Limits
# =============================================================================

# Maximum number of tools to expose for each complexity level
# Set high for complex tasks - context compaction will manage overflow
COMPLEXITY_TOOL_LIMITS: Dict[str, int] = {
    "simple": 10,
    "moderate": 30,
    "complex": 100,  # High limit for complex architecture analysis
}


# =============================================================================
# Step Tool Mapping
# =============================================================================

# Maps planning step types to the appropriate tool sets
STEP_TOOL_MAPPING: Dict[str, Set[str]] = {
    # Research steps need read-only exploration tools
    "research": {
        "read",  # Read file contents
        "grep",  # Search for patterns
        "code_search",  # Semantic code search
        "overview",  # Get project overview
        "ls",  # List directories
        "git_readonly",  # Read git history
    },
    # Planning needs exploration + analysis tools
    "planning": {
        "read",
        "grep",
        "code_search",
        "overview",
        "ls",
        "analyze",  # Code analysis
    },
    # Feature implementation needs full toolset
    "feature": {
        "read",
        "write",  # Write files
        "edit",  # Edit files
        "grep",
        "test",  # Run tests
        "code_search",
        "git",  # Git operations
        "shell",  # Execute commands
    },
    # Bugfix needs debugging tools
    "bugfix": {
        "read",
        "grep",
        "code_search",
        "edit",
        "test",
        "shell",  # Shell for debugging
    },
    # Refactoring needs code manipulation tools
    "refactor": {
        "read",
        "grep",
        "code_search",
        "edit",
        "write",
        "test",
    },
    # Testing needs test execution and verification tools
    "test": {
        "test",
        "read",
        "grep",
        "shell",
    },
    # Review needs read-only analysis tools
    "review": {
        "read",
        "grep",
        "code_search",
        "analyze",
    },
    # Deploy needs deployment tools
    "deploy": {
        "shell",
        "git",
        "docker",  # Docker operations
        "kubectl",  # Kubernetes operations
        "read",
        "grep",
    },
    # Analysis needs comprehensive exploration tools
    "analyze": {
        "read",
        "grep",
        "code_search",
        "overview",
        "analyze",
        "ls",
    },
    # Documentation needs read and write tools
    "doc": {
        "read",
        "grep",
        "code_search",
        "write",
        "edit",
    },
}


# =============================================================================
# Step to Task Type Mapping
# =============================================================================

# Maps planning step types to task_tool_config task types
STEP_TO_TASK_TYPE: Dict[str, str] = {
    "research": "search",
    "planning": "design",
    "feature": "create",
    "bugfix": "edit",
    "refactor": "edit",
    "test": "create",
    "review": "analyze",
    "deploy": "deploy",
    "analyze": "analyze",
    "doc": "doc",
}


# =============================================================================
# Planning Detection Keywords
# =============================================================================

# Keywords that indicate a complex multi-step task
COMPLEXITY_KEYWORDS: list[str] = [
    "analyze",
    "architecture",
    "design",
    "evaluate",
    "compare",
    "roadmap",
    "implementation",
    "refactor",
    "migration",
    "solid",
    "scalability",
    "performance",
    "security audit",
    "plan mode",  # Explicit user request for planning
    "planning",
    "step by step",
    "step-by-step",
]


# Step indicators that suggest multi-step process
STEP_INDICATORS: list[str] = [
    "step",
    "phase",
    "stage",
    "deliverable",
    "roadmap",
    "first",
    "then",
    "finally",
    "analyze",
    "evaluate",
    "design",
]


# =============================================================================
# Default Planning Configuration
# =============================================================================

# Default minimum complexity to trigger planning
DEFAULT_MIN_PLANNING_COMPLEXITY = "moderate"

# Default minimum step indicators to trigger planning
DEFAULT_MIN_STEPS_THRESHOLD = 3

# Default minimum keyword matches to trigger planning
DEFAULT_MIN_KEYWORD_MATCHES = 2
