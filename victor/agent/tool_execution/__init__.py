# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Async-first tool execution module.

This module provides async-first tool execution with:
- Automatic parallelization for read-only operations
- Dependency graph execution
- File locking for concurrent writes
- Priority-based scheduling

Design Patterns:
- Strategy Pattern: Pluggable execution strategies
- Repository Pattern: File lock management
- Command Pattern: Tool execution commands
"""

from __future__ import annotations

from victor.agent.tool_execution.async_executor import AsyncToolExecutor
from victor.agent.tool_execution.categorization import (
    ToolCategory,
    categorize_tool_call,
)

__all__ = [
    "AsyncToolExecutor",
    "ToolCategory",
    "categorize_tool_call",
]
