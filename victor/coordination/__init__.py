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

"""Multi-agent coordination module.

This module provides formation strategies for multi-agent coordination,
without creating circular dependencies.

The coordination module is intentionally kept lightweight and independent:
- Formation strategies (sequential, parallel, hierarchical, pipeline, consensus)
- Base classes and protocols for coordination
- No dependency on teams module (breaks circular dependency)

Example:
    from victor.coordination.formations import SequentialFormation, ParallelFormation
    from victor.teams import UnifiedTeamCoordinator

    # Formations are used by coordinators, not the other way around
    coordinator = UnifiedTeamCoordinator()
    coordinator.set_formation(TeamFormation.SEQUENTIAL)
"""

# Re-export formation strategies for convenience
from victor.coordination.formations.base import (
    BaseFormationStrategy,
    TeamContext,
)
from victor.coordination.formations import (
    SequentialFormation,
    ParallelFormation,
    HierarchicalFormation,
    PipelineFormation,
    ConsensusFormation,
)

__all__ = [
    # Base classes
    "BaseFormationStrategy",
    "TeamContext",
    # Formation strategies
    "SequentialFormation",
    "ParallelFormation",
    "HierarchicalFormation",
    "PipelineFormation",
    "ConsensusFormation",
]
