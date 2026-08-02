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

"""Emergent Collaboration Patterns Framework.

This module provides data-driven pattern discovery and recommendation
for workflow collaboration patterns.

Key Components:
    - PatternMiner: Discovers patterns from execution traces
    - PatternRecommender: Suggests optimal patterns for tasks
    - PatternValidator: Validates pattern quality and safety
    - CollaborationPattern: Pattern representation

Example:
    from victor.framework.patterns import (
        PatternMiner,
        PatternRecommender,
        TaskContext,
        CollaborationPattern,
    )

    # Mine patterns from execution history
    miner = PatternMiner(min_occurrences=3)
    patterns = await miner.mine_from_traces(execution_traces)

    # Get recommendations for new task
    recommender = PatternRecommender(patterns)
    context = TaskContext(
        task_description="Implement authentication",
        required_capabilities=["coding", "security"],
        complexity="high",
    )
    recommendations = await recommender.recommend(context, top_k=3)
"""

# Types
from victor.framework.patterns.types import (
    PatternStatus,
    PatternCategory,
    PatternValidationResult,
    PatternMetrics,
    TaskContext,
    CollaborationPattern,
    PatternRecommendation,
    WorkflowExecutionTrace,
)

# Protocols
from victor.framework.patterns.protocols import (
    PatternMinerProtocol,
    PatternValidatorProtocol,
    PatternRecommenderProtocol,
)

# Implementations
from victor.framework.patterns.miner import (
    PatternMiner,
)

from victor.framework.patterns.recommender import (
    PatternRecommender,
    PatternValidator,
)

__all__ = [
    # Types
    "PatternStatus",
    "PatternCategory",
    "PatternValidationResult",
    "PatternMetrics",
    "TaskContext",
    "CollaborationPattern",
    "PatternRecommendation",
    "WorkflowExecutionTrace",
    # Protocols
    "PatternMinerProtocol",
    "PatternValidatorProtocol",
    "PatternRecommenderProtocol",
    # Implementations
    "PatternMiner",
    "PatternRecommender",
    "PatternValidator",
]
