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

"""Workflow optimization algorithms for Victor.

This package provides automated workflow optimization capabilities including:
- Performance profiling and bottleneck detection
- Optimization strategies (pruning, parallelization, tool selection)
- Search algorithms (hill climbing, simulated annealing)
- Variant generation and evaluation
- Safety validation and rollback

Usage:
    from victor.optimization import WorkflowOptimizer

    optimizer = WorkflowOptimizer()
    profile = await optimizer.analyze_workflow(workflow_id)
    suggestions = await optimizer.suggest_optimizations(workflow_id)
"""

from victor.optimization.models import (
    Bottleneck,
    BottleneckSeverity,
    BottleneckType,
    NodeStatistics,
    OptimizationOpportunity,
    OptimizationStrategy,
    OptimizationStrategyType,
    WorkflowProfile,
)
from victor.optimization.profiler import WorkflowProfiler
from victor.optimization.strategies import (
    ParallelizationStrategy,
    PruningStrategy,
    ToolSelectionStrategy,
)
from victor.optimization.generator import WorkflowVariantGenerator
from victor.optimization.search import HillClimbingOptimizer
from victor.optimization.evaluator import VariantEvaluator
from victor.optimization.optimizer import OptimizationConfig, WorkflowOptimizer
from victor.optimization.validator import (
    ConstraintViolation,
    OptimizationValidationResult,
    OptimizationValidator,
    ValidationRecommendation,
)

__all__ = [
    # Models
    "WorkflowProfile",
    "NodeStatistics",
    "Bottleneck",
    "BottleneckType",
    "BottleneckSeverity",
    "OptimizationOpportunity",
    "OptimizationStrategy",
    "OptimizationStrategyType",
    # Core components
    "WorkflowProfiler",
    "WorkflowOptimizer",
    "OptimizationConfig",
    "WorkflowVariantGenerator",
    "VariantEvaluator",
    "HillClimbingOptimizer",
    # Strategies
    "PruningStrategy",
    "ParallelizationStrategy",
    "ToolSelectionStrategy",
    # Validation
    "OptimizationValidator",
    "OptimizationValidationResult",
    "ValidationRecommendation",
    "ConstraintViolation",
]
