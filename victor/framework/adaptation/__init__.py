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

"""Dynamic Workflow Adaptation Framework.

This module provides runtime workflow modification capabilities with
safety mechanisms including validation, rollback, and circuit breakers.

Key Components:
    - AdaptableGraph: Wrapper for safe runtime modifications
    - GraphModification: Modification specifications
    - AdaptationStrategy: Pluggable adaptation logic
    - Validation and Impact Analysis: Safety mechanisms

Example:
    from victor.framework.adaptation import (
        AdaptableGraph,
        AdaptationConfig,
        GraphModification,
        ModificationType,
        create_retry_strategy,
    )

    # Wrap existing graph
    adaptable = AdaptableGraph(compiled_graph)

    # Configure
    adaptable.configure(AdaptationConfig(
        enable_auto_checkpoint=True,
        rollback_on_error=True,
    ))

    # Apply modification
    modification = GraphModification(
        modification_type=ModificationType.ADD_RETRY,
        description="Add retry to API call",
        target_node="api_call",
        data={"max_retries": 3},
    )

    result = await adaptable.adapt(modification)
"""

# Types
from victor.framework.adaptation.types import (
    ModificationType,
    AdaptationTrigger,
    RiskLevel,
    GraphModification,
    AdaptationValidationResult,
    AdaptationImpact,
    AdaptationCheckpoint,
    AdaptationResult,
    AdaptationConfig,
    AdaptationStrategy,
)

# Protocols
from victor.framework.adaptation.protocols import (
    GraphValidator,
    GraphApplier,
    GraphRollback,
    ImpactAnalyzer,
)

# Graph
from victor.framework.adaptation.graph import (
    AdaptableGraph,
)

# Strategies
from victor.framework.adaptation.strategies import (
    create_retry_strategy,
    create_circuit_breaker_strategy,
    create_parallelization_strategy,
    create_caching_strategy,
    DEFAULT_STRATEGIES,
)

__all__ = [
    # Types
    "ModificationType",
    "AdaptationTrigger",
    "RiskLevel",
    "GraphModification",
    "AdaptationValidationResult",
    "AdaptationImpact",
    "AdaptationCheckpoint",
    "AdaptationResult",
    "AdaptationConfig",
    "AdaptationStrategy",
    # Protocols
    "GraphValidator",
    "GraphApplier",
    "GraphRollback",
    "ImpactAnalyzer",
    # Graph
    "AdaptableGraph",
    # Strategies
    "create_retry_strategy",
    "create_circuit_breaker_strategy",
    "create_parallelization_strategy",
    "create_caching_strategy",
    "DEFAULT_STRATEGIES",
]
