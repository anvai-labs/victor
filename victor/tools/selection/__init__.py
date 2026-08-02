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

"""Tool Selection Strategy Module.

This module provides a pluggable tool selection architecture with
multiple strategies for intelligent tool filtering.

Available Strategies:
- **Keyword**: Fast registry-based matching (<1ms, no embeddings)
- **Semantic**: ML-based embedding similarity (10-50ms)
- **Hybrid**: Blends both approaches (best of both worlds)

Quick Start:
    from victor.tools.selection import (
        get_best_strategy,
        ToolSelectionContext,
    )

    # Create context
    context = ToolSelectionContext(
        prompt="Find all Python files that import numpy",
        task_type="analysis",
        vertical="coding",
    )

    # Get best strategy and select tools
    strategy = get_best_strategy(context)
    tools = await strategy.select_tools(context)

Custom Strategy:
    from victor.tools.selection import (
        BaseToolSelectionStrategy,
        PerformanceProfile,
        register_strategy,
    )

    class MyCustomSelector(BaseToolSelectionStrategy):
        def get_strategy_name(self) -> str:
            return "my_custom"

        def get_performance_profile(self) -> PerformanceProfile:
            return PerformanceProfile(
                avg_latency_ms=5.0,
                requires_embeddings=False,
                requires_model_inference=False,
                memory_usage_mb=10.0,
            )

        async def select_tools(self, context, max_tools=10):
            # Custom logic
            return ["read", "grep", "semantic_search"]

    # Register globally
    register_strategy("my_custom", MyCustomSelector())
"""

from victor.tools.selection.protocol import (
    BaseToolSelectionStrategy,
    PerformanceProfile,
    ToolSelectionContext,
    ToolSelectionStrategy,
    ToolSelectorFeatures,
)
from victor.tools.selection.registry import (
    ToolSelectionStrategyRegistry,
    get_best_strategy,
    get_strategy,
    get_strategy_registry,
    list_strategies,
    register_strategy,
)

__all__ = [
    # Protocol
    "ToolSelectionStrategy",
    "BaseToolSelectionStrategy",
    "PerformanceProfile",
    "ToolSelectionContext",
    "ToolSelectorFeatures",
    # Registry
    "ToolSelectionStrategyRegistry",
    "get_strategy_registry",
    "register_strategy",
    "get_strategy",
    "get_best_strategy",
    "list_strategies",
]
