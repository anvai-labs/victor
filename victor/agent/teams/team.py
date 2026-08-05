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

"""Core team data structures for multi-agent coordination.

This module defines the fundamental building blocks for agent teams:
- TeamFormation: How agents are organized and coordinated
- TeamMember: Individual agent configuration within a team
- TeamConfig: Complete team configuration
- TeamResult: Execution outcome and metrics

Design Principles:
- Immutable configuration (dataclasses)
- Flexible formation patterns (sequential, parallel, hierarchical, pipeline)
- Built on existing SubAgentRole infrastructure
- Serializable for persistence and API transport
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set

from victor.agent.subagents.base import SubAgentRole

# Import canonical team types from victor.teams.types
from victor.teams.types import (
    TeamFormation,
    MemoryConfig,
    MemberStatus,
    TeamMember,
    TeamConfig,
    MemberResult,
    TeamResult,
)

if TYPE_CHECKING:
    from victor.agent.protocols import UnifiedMemoryCoordinatorProtocol


__all__ = [
    "MemoryConfig",
    "TeamFormation",
    "MemberStatus",
    "TeamMember",
    "TeamConfig",
    "MemberResult",
    "TeamResult",
]
