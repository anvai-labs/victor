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

"""Team coordination protocols (re-export from canonical location).

This module re-exports team protocols from victor.protocols.team for backward
compatibility. The canonical location is victor.protocols.team which breaks
circular dependencies between victor.teams and victor.framework.

Protocols:
    IAgent: Base agent protocol
    ITeamMember: Team member protocol
    ITeamCoordinator: Base coordinator protocol
    IDelegateFollowUpCoordinator: Delegate follow-up execution capability
    IObservableCoordinator: Observability capabilities
    IRLCoordinator: RL integration capabilities
    IMessageBusProvider: Message bus provider
    ISharedMemoryProvider: Shared memory provider
    IEnhancedTeamCoordinator: Combined capabilities

Import from here for backward compatibility, or import from victor.protocols.team
directly to break circular dependencies.
"""

# Import from canonical location to avoid circular dependencies
from victor.protocols.team import (
    IAgent,
    ITeamMember,
    ITeamCoordinator,
    IDelegateFollowUpCoordinator,
    IObservableCoordinator,
    IRLCoordinator,
    IMessageBusProvider,
    ISharedMemoryProvider,
    IEnhancedTeamCoordinator,
    ISupervisorAgent,
    TeamCoordinatorProtocol,
    TeamMemberProtocol,
)

__all__ = [
    "IAgent",
    "ITeamMember",
    "ITeamCoordinator",
    "IDelegateFollowUpCoordinator",
    "IObservableCoordinator",
    "IRLCoordinator",
    "IMessageBusProvider",
    "ISharedMemoryProvider",
    "IEnhancedTeamCoordinator",
    "ISupervisorAgent",
    "TeamCoordinatorProtocol",
    "TeamMemberProtocol",
]
