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

"""A/B testing for workflow optimization.

This package provides A/B testing infrastructure for comparing different
workflow configurations.

Usage:
    from victor.experiments.ab_testing import (
        ABTestManager,
        ExperimentConfig,
        ExperimentVariant,
        WorkflowInterceptor,
    )

    # Create manager
    manager = ABTestManager()

    # Create experiment
    config = ExperimentConfig(
        name="Model Comparison",
        variants=[
            ExperimentVariant(
                variant_id="control",
                name="Claude Sonnet",
                traffic_weight=0.5,
            ),
            ExperimentVariant(
                variant_id="treatment",
                name="Claude Opus",
                traffic_weight=0.5,
            ),
        ],
    )
    experiment_id = await manager.create_experiment(config)
    await manager.start_experiment(experiment_id)

    # Use interceptor
    interceptor = WorkflowInterceptor(manager)
    result = await interceptor.execute_with_experiment(
        workflow_func=execute_workflow,
        experiment_id=experiment_id,
        user_id="user_123",
        context={},
    )
"""

from victor.experiments.ab_testing.models import (
    AggregatedMetrics,
    AllocationStrategy,
    ExecutionMetrics,
    ExperimentConfig,
    ExperimentMetric,
    ExperimentResult,
    ExperimentStatus,
    ExperimentVariant,
    VariantResult,
)
from victor.experiments.ab_testing.allocator import (
    RandomAllocator,
    RoundRobinAllocator,
    StickyAllocator,
    TrafficAllocator,
    create_allocator,
)
from victor.experiments.ab_testing.experiment import ABTestManager
from victor.experiments.ab_testing.interceptor import (
    ExperimentCompiledGraphWrapper,
    WorkflowInterceptor,
)
from victor.experiments.ab_testing.metrics import MetricsCollector
from victor.experiments.ab_testing.statistics import StatisticalAnalyzer

__all__ = [
    # Models
    "ExperimentConfig",
    "ExperimentVariant",
    "ExperimentMetric",
    "ExperimentStatus",
    "ExperimentResult",
    "VariantResult",
    "ExecutionMetrics",
    "AggregatedMetrics",
    "AllocationStrategy",
    # Manager
    "ABTestManager",
    # Allocator
    "TrafficAllocator",
    "RandomAllocator",
    "StickyAllocator",
    "RoundRobinAllocator",
    "create_allocator",
    # Interceptor
    "WorkflowInterceptor",
    "ExperimentCompiledGraphWrapper",
    # Metrics
    "MetricsCollector",
    # Statistics
    "StatisticalAnalyzer",
]
