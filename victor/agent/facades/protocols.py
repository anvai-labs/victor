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

"""Protocol definition for the orchestration facade.

``OrchestrationFacadeProtocol`` is the one live facade protocol — the seven
per-domain facade protocols (Chat/Tool/Provider/Session/Metrics/Resilience/
Workflow) were deleted along with their facades: zero readers, dead parallel
views of orchestrator state (P6 residue cleanup; see
docs/architecture/foundations-strategy-2026-07.md §3.2).
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class OrchestrationFacadeProtocol(Protocol):
    """Protocol for the top-level orchestration domain facade.

    Manages coordinators, protocol adapters, streaming handlers,
    intelligent pipeline integration, and subagent orchestration.
    """

    @property
    def protocol_adapter(self) -> Any:
        """Protocol adapter for coordinator communication."""
        ...

    @property
    def chat_stream_adapter(self) -> Optional[Any]:
        """Canonical service-owned chat-stream adapter."""
        ...

    @property
    def task_analyzer(self) -> Optional[Any]:
        """Optional task analyzer."""
        ...

    @property
    def exploration_state_passed(self) -> Optional[Any]:
        """Optional state-passed exploration coordinator."""
        ...

    @property
    def system_prompt_state_passed(self) -> Optional[Any]:
        """Optional state-passed system prompt coordinator."""
        ...

    @property
    def safety_state_passed(self) -> Optional[Any]:
        """Optional state-passed safety coordinator."""
        ...

    @property
    def coordination_state_passed(self) -> Optional[Any]:
        """Optional state-passed coordination recommendation coordinator."""
        ...
