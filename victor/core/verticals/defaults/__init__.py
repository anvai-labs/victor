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

"""Default configurations for verticals.

This module provides common configurations that can be extended by any vertical,
reducing code duplication across vertical implementations.

Provides:
- Common tool dependencies (e.g., read -> edit -> test patterns)
- Common tool clusters (e.g., file_operations, search_operations)
- Common task type patterns
- Default tool budgets

Example usage:
    from victor.core.verticals.defaults import (
        COMMON_TOOL_CLUSTERS,
        COMMON_TOOL_DEPENDENCIES,
        COMMON_TOOL_TRANSITIONS,
        merge_with_defaults,
    )

    # Extend defaults with vertical-specific data
    my_clusters = merge_clusters(COMMON_TOOL_CLUSTERS, {
        "my_vertical_specific": {"tool1", "tool2"},
    })
"""

# Common tool configurations
from victor.core.verticals.defaults.tool_defaults import (
    COMMON_TOOL_CLUSTERS,
    COMMON_TOOL_DEPENDENCIES,
    COMMON_TOOL_TRANSITIONS,
    COMMON_REQUIRED_TOOLS,
    COMMON_OPTIONAL_TOOLS,
    merge_clusters,
    merge_dependencies,
    merge_transitions,
)

# Common task type patterns
from victor.core.verticals.defaults.task_defaults import (
    COMMON_TASK_BUDGETS,
    COMMON_TASK_HINTS,
    get_task_budget,
    get_task_hint,
)

__all__ = [
    # Tool defaults
    "COMMON_TOOL_CLUSTERS",
    "COMMON_TOOL_DEPENDENCIES",
    "COMMON_TOOL_TRANSITIONS",
    "COMMON_REQUIRED_TOOLS",
    "COMMON_OPTIONAL_TOOLS",
    "merge_clusters",
    "merge_dependencies",
    "merge_transitions",
    # Task defaults
    "COMMON_TASK_BUDGETS",
    "COMMON_TASK_HINTS",
    "get_task_budget",
    "get_task_hint",
]
