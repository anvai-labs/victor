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

"""Tool composition utilities for lazy loading and efficient tool management.

This package provides utilities for composing tools:

LCEL-style Composition (from runnable.py):
- Runnable: Base class for composable runnables
- RunnableSequence: Sequential execution (pipe chaining)
- RunnableParallel: Parallel execution
- RunnableBranch: Conditional routing
- RunnableLambda: Wrap functions as runnables
- RunnablePassthrough: Pass input through unchanged
- RunnableBinding: Runnables with bound arguments
- ToolRunnable: Wrap BaseTool as Runnable
- FunctionToolRunnable: Wrap @tool functions as Runnable
- Helper functions: chain, parallel, branch, as_runnable
- Extractors: extract_output, extract_if_success, map_keys, select_keys

Lazy Tool Loading (from lazy.py):
- LazyToolRunnable: Defers tool initialization until first use
- ToolCompositionBuilder: Builder pattern for composing multiple tools

Example:
    >>> from victor.tools.composition import LazyToolRunnable, ToolCompositionBuilder
    >>>
    >>> # Lazy loading single tool
    >>> lazy_tool = LazyToolRunnable(lambda: ExpensiveTool(), name="expensive")
    >>> # Tool not created yet
    >>> result = lazy_tool.run({"input": "test"})  # Now created and cached
    >>>
    >>> # Building composed tools
    >>> builder = ToolCompositionBuilder()
    >>> tools = (
    ...     builder
    ...     .add("search", lambda: SearchTool(), lazy=True)
    ...     .add("analyze", lambda: AnalyzeTool(), lazy=True)
    ...     .build()
    ... )
"""

# Re-export LCEL-style composition classes from runnable.py (backward compatibility)
from victor.tools.composition.runnable import (
    Runnable,
    RunnableConfig,
    RunnableSequence,
    RunnableParallel,
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
    RunnableBinding,
    ToolRunnable,
    FunctionToolRunnable,
    as_runnable,
    chain,
    parallel,
    branch,
    extract_output,
    extract_if_success,
    map_keys,
    select_keys,
)

# Export lazy loading utilities
from victor.tools.composition.lazy import LazyToolRunnable, ToolCompositionBuilder

__all__ = [
    # LCEL-style composition (from runnable.py)
    "Runnable",
    "RunnableConfig",
    "RunnableSequence",
    "RunnableParallel",
    "RunnableBranch",
    "RunnableLambda",
    "RunnablePassthrough",
    "RunnableBinding",
    "ToolRunnable",
    "FunctionToolRunnable",
    "as_runnable",
    "chain",
    "parallel",
    "branch",
    "extract_output",
    "extract_if_success",
    "map_keys",
    "select_keys",
    # Lazy loading (from lazy.py)
    "LazyToolRunnable",
    "ToolCompositionBuilder",
]
