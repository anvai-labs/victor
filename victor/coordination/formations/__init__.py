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

"""Formation strategy pattern for multi-agent coordination.

This module implements the Strategy pattern for different team
formation patterns, allowing easy extension without modifying
core coordinator logic (Open/Closed Principle).

Available Formations:
    - SequentialFormation: Execute agents in sequence
    - ParallelFormation: Execute agents in parallel
    - HierarchicalFormation: Single-level hierarchy
    - MultiLevelHierarchyFormation: Multi-level divide-and-conquer
    - PipelineFormation: Pass output through pipeline
    - ConsensusFormation: Voting-based consensus
    - ReflectionFormation: Iterative generator-critic refinement
    - DynamicRouterFormation: Task-based agent routing
    - AdaptiveFormation: Dynamic formation switching
"""

from victor.coordination.formations.adaptive import (
    AdaptiveFormation,
    AdaptationStrategy,
)
from victor.coordination.formations.base import BaseFormationStrategy
from victor.coordination.formations.consensus import ConsensusFormation
from victor.coordination.formations.dynamic_router import DynamicRouterFormation
from victor.coordination.formations.hierarchical import HierarchicalFormation
from victor.coordination.formations.multi_level_hierarchy import (
    HierarchyNode,
    MultiLevelHierarchyFormation,
)
from victor.coordination.formations.parallel import ParallelFormation
from victor.coordination.formations.pipeline import PipelineFormation
from victor.coordination.formations.reflection import ReflectionFormation
from victor.coordination.formations.sequential import SequentialFormation

__all__ = [
    # Base
    "BaseFormationStrategy",
    # Basic formations
    "SequentialFormation",
    "ParallelFormation",
    "HierarchicalFormation",
    "MultiLevelHierarchyFormation",
    "HierarchyNode",
    "PipelineFormation",
    "ConsensusFormation",
    # Advanced formations
    "ReflectionFormation",
    "DynamicRouterFormation",
    "AdaptiveFormation",
    "AdaptationStrategy",
]
