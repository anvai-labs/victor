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

"""Memory/session runtime boundaries for AgentOrchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class MemoryRuntimeComponents:
    """Memory runtime handles exposed to the orchestrator facade."""

    memory_manager: Optional[Any]
    memory_session_id: Optional[str]
    memory_initialization_diagnostics: Dict[str, Any] = field(default_factory=dict)


def create_memory_runtime_components(
    *,
    factory: Any,
    provider_name: str,
    native_tool_calls: bool,
) -> MemoryRuntimeComponents:
    """Create memory runtime components for orchestrator wiring."""
    memory_manager, memory_session_id = factory.create_memory_components(
        provider_name,
        native_tool_calls,
    )
    diagnostics_getter = getattr(factory, "get_memory_initialization_diagnostics", None)
    diagnostics = diagnostics_getter() if callable(diagnostics_getter) else {}
    return MemoryRuntimeComponents(
        memory_manager=memory_manager,
        memory_session_id=memory_session_id,
        memory_initialization_diagnostics=dict(diagnostics or {}),
    )


def initialize_conversation_embedding_store(
    *,
    memory_manager: Any,
) -> Tuple[Optional[Any], Optional[Any]]:
    """Create conversation embedding store and semantic cache handle."""
    from victor.agent.services.session_service import SessionService

    return SessionService.init_conversation_embedding_store(
        memory_manager=memory_manager,
    )
