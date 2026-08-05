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

"""Framework task services module.

This module provides framework-level task classification and budgeting services.
These services are used by the agent layer but defined at the framework level
for reusability across all verticals.

Design Principles:
- SRP: Classification and budgets only (no prompt hints)
- DIP: Agent layer depends on protocols, not implementations
- Prompt hints are handled separately by the enrichment pipeline

Usage:
    from victor.framework.task import (
        TaskComplexity,
        TaskComplexityService,
        TaskClassification,
        classify_task,
    )

    # Using the service
    service = TaskComplexityService()
    result = service.classify("refactor authentication")
    print(f"Complexity: {result.complexity}, Budget: {result.tool_budget}")

    # Using convenience function
    result = classify_task("list all files")
    print(f"This is a {result.complexity.value} task")
"""

# Core task types
from .core import (
    Task,
    TaskResult,
    FrameworkTaskType,
)

# Complexity classification protocols
from .protocols import (
    TaskBudgetProviderProtocol,
    TaskClassification,
    TaskClassifierProtocol,
    TaskComplexity,
)

# Complexity classification service
from .complexity import (
    ComplexityBudget,
    COMPLEXITY_BUDGETS,
    DEFAULT_BUDGETS,
    PATTERNS,
    TASK_TYPE_TO_COMPLEXITY,
    TaskComplexityService,
    classify_task,
    get_budget_for_task,
    is_action_task,
    is_analysis_task,
)
from .direct_response import (
    DirectResponseClassification,
    DirectResponseOutputState,
    classify_direct_response_prompt,
    extract_exact_response_literal,
    has_codebase_context,
    is_direct_response_prompt,
    normalize_direct_response_output,
)

__all__ = [
    # Core task types
    "Task",
    "TaskResult",
    "FrameworkTaskType",
    # Protocols
    "TaskClassifierProtocol",
    "TaskBudgetProviderProtocol",
    # Core types
    "TaskComplexity",
    "TaskClassification",
    # Service
    "TaskComplexityService",
    # Consolidated budgets (single source of truth)
    "ComplexityBudget",
    "COMPLEXITY_BUDGETS",
    # Legacy constants (prefer COMPLEXITY_BUDGETS)
    "DEFAULT_BUDGETS",
    "PATTERNS",
    "TASK_TYPE_TO_COMPLEXITY",
    # Convenience functions
    "classify_task",
    "get_budget_for_task",
    "is_action_task",
    "is_analysis_task",
    "DirectResponseClassification",
    "DirectResponseOutputState",
    "classify_direct_response_prompt",
    "extract_exact_response_literal",
    "has_codebase_context",
    "is_direct_response_prompt",
    "normalize_direct_response_output",
]
