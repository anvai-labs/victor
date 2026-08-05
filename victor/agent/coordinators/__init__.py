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

"""Coordinator package for Victor agentic AI framework.

This package provides state-passed coordinators for specialized functionality.
The deprecated service-first coordinators (chat, tool, session) have been removed.

Preferred surfaces:
- ``victor.agent.services`` for service-owned chat, tool, and session flows
- ``victor.agent.services.stage_transition_runtime`` and
  ``victor.agent.services.stage_transition_strategies`` for service-owned
  stage-transition batching runtime
- ``victor.agent.coordinators.ExplorationStatePassedCoordinator`` for exploration
- ``victor.agent.coordinators.CoordinationStatePassedCoordinator`` for
  coordination recommendations
- ``victor.agent.coordinators.SystemPromptStatePassedCoordinator`` for system prompts
- ``victor.agent.coordinators.SafetyStatePassedCoordinator`` for safety

Note: ToolCoordinator, ChatCoordinator, SessionCoordinator have been removed.
Use ToolService, ChatService, SessionService instead.
StageTransitionCoordinator and its strategies remain available here only as
compatibility re-export paths. SDK-owned safety and conversation exports remain
available here only as deprecated compatibility shims; new code should import
them from ``victor_contracts`` directly.
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any

_SUBMODULE_MAP: dict[str, str] = {
    # Runtime coordinators from services
    "ExplorationCoordinator": "services.exploration_runtime",
    "ExplorationResult": "services.exploration_runtime",
    "PlanningCoordinator": "services.planning_runtime",
    "PlanningConfig": "services.planning_runtime",
    "PlanningMode": "services.planning_runtime",
    "PlanningResult": "services.planning_runtime",
    # SDK-owned compatibility / extension surfaces
    "SafetyCoordinator": "victor_contracts.safety",
    "SafetyRule": "victor_contracts.safety",
    "SafetyCheckResult": "victor_contracts.safety",
    "SafetyStats": "victor_contracts.safety",
    "SafetyAction": "victor_contracts.safety",
    "SafetyCategory": "victor_contracts.safety",
    "ConversationCoordinator": "victor_contracts.conversation",
    "TurnType": "victor_contracts.conversation",
    "ConversationTurn": "victor_contracts.conversation",
    "ConversationStats": "victor_contracts.conversation",
    "ConversationContext": "victor_contracts.conversation",
}

_MODULE_MEMBERS = {
    "exploration_state_passed": [
        "ExplorationStatePassedCoordinator",
    ],
    "coordination_state_passed": [
        "CoordinationStatePassedCoordinator",
    ],
    # NOTE: tool_coordinator, tool_observability, tool_retry removed
    # These now import directly from victor.agent.services for backward compatibility
    # See __getattr__ below for service-level re-exports
    # NOTE: chat_coordinator, sync_chat_coordinator, streaming_chat_coordinator,
    # unified_chat_coordinator removed as deprecated shims.
    "chat_protocols": [
        "ChatContextProtocol",
        "ChatOrchestratorProtocol",
        "ProviderContextProtocol",
        "ToolContextProtocol",
    ],
    "protocols": [
        "CacheProvider",
        "CacheEventType",
        "EventEmitter",
        "ConfigProvider",
        "NoOpCacheProvider",
        "NoOpEventEmitter",
        "DictConfigProvider",
    ],
    "system_prompt_state_passed": [
        "SystemPromptStatePassedCoordinator",
    ],
    "safety_state_passed": [
        "SafetyStatePassedCoordinator",
    ],
    # Stage transition coordination compatibility wrappers over services
    "stage_transition_coordinator": [
        "StageTransitionCoordinator",
        "TransitionDecision",
        "TransitionResult",
        "TurnContext",
    ],
    "transition_strategies": [
        "TransitionStrategyProtocol",
        "HeuristicOnlyTransitionStrategy",
        "EdgeModelTransitionStrategy",
        "HybridTransitionStrategy",
        "create_transition_strategy",
    ],
    # NOTE: exploration_coordinator, metrics_coordinator, safety_coordinator,
    # planning_coordinator, session_coordinator, turn_executor, and the former
    # system_prompt_coordinator compatibility wrapper were removed as
    # deprecated shims.
    # These now import directly from services for backward compatibility.
    # See __getattr__ below for service-level re-exports.
}

for _module_name, _exported_names in _MODULE_MEMBERS.items():
    for _exported_name in _exported_names:
        _SUBMODULE_MAP[_exported_name] = _module_name

__all__ = list(_SUBMODULE_MAP.keys())

_DEPRECATED_EXPORTS = {
    "ChatContextProtocol": (
        "victor.agent.coordinators.ChatContextProtocol is deprecated compatibility "
        "surface. Prefer ChatServiceProtocol and service-owned runtime boundaries "
        "from victor.agent.services. Will be removed in v0.10.0."
    ),
    "ChatOrchestratorProtocol": (
        "victor.agent.coordinators.ChatOrchestratorProtocol is deprecated "
        "compatibility surface. Prefer ChatServiceProtocol and service-owned "
        "runtime boundaries from victor.agent.services. Will be removed in v0.10.0."
    ),
    "ProviderContextProtocol": (
        "victor.agent.coordinators.ProviderContextProtocol is deprecated "
        "compatibility surface. Prefer ProviderServiceProtocol and service-owned "
        "runtime boundaries from victor.agent.services. Will be removed in v0.10.0."
    ),
    "ToolContextProtocol": (
        "victor.agent.coordinators.ToolContextProtocol is deprecated compatibility "
        "surface. Prefer ToolServiceProtocol and service-owned runtime boundaries "
        "from victor.agent.services. Will be removed in v0.10.0."
    ),
}

_SDK_SAFETY_EXPORTS = frozenset(
    {
        "SafetyCoordinator",
        "SafetyRule",
        "SafetyCheckResult",
        "SafetyStats",
        "SafetyAction",
        "SafetyCategory",
    }
)

_SDK_CONVERSATION_EXPORTS = frozenset(
    {
        "ConversationCoordinator",
        "TurnType",
        "ConversationTurn",
        "ConversationStats",
        "ConversationContext",
    }
)


def _get_deprecation_warning(name: str) -> str | None:
    """Return the deprecation warning message for a compatibility export."""
    if name in _DEPRECATED_EXPORTS:
        return _DEPRECATED_EXPORTS[name]

    if name == "SafetyCoordinator":
        return (
            "victor.agent.coordinators.SafetyCoordinator is deprecated SDK "
            "compatibility surface. Prefer victor_contracts.safety.SafetyCoordinator "
            "for extensions or SafetyStatePassedCoordinator for agent runtime "
            "policy seams. Will be removed in v0.10.0."
        )
    if name in _SDK_SAFETY_EXPORTS:
        return (
            f"victor.agent.coordinators.{name} is deprecated SDK compatibility "
            f"surface. Prefer victor_contracts.safety.{name} directly. Will be removed in v0.10.0."
        )
    if name == "ConversationCoordinator":
        return (
            "victor.agent.coordinators.ConversationCoordinator is deprecated SDK "
            "compatibility surface. Prefer victor_contracts.conversation."
            "ConversationCoordinator directly. Will be removed in v0.10.0."
        )
    if name in _SDK_CONVERSATION_EXPORTS:
        return (
            f"victor.agent.coordinators.{name} is deprecated SDK compatibility "
            f"surface. Prefer victor_contracts.conversation.{name} directly. Will be removed in v0.10.0."
        )

    return None


def __getattr__(name: str) -> Any:
    """Resolve coordinator exports lazily and warn on deprecated shims.

    For deleted coordinator shims: re-exports from services or SDK.
    For chat_protocols: imports from services.protocols.chat_runtime.
    For other coordinators: imports from coordinators.{module_name}.
    """
    # Service-level re-exports for coordinators
    if name in {
        # Runtime coordinators from services
        "ExplorationCoordinator",
        "ExplorationResult",
        "SafetyCoordinator",
        "SafetyRule",
        "SafetyCheckResult",
        "SafetyStats",
        "SafetyAction",
        "SafetyCategory",
        "PlanningCoordinator",
        "PlanningConfig",
        "PlanningMode",
        "PlanningResult",
        # SDK-based coordinators
        "ConversationCoordinator",
        "TurnType",
        "ConversationTurn",
        "ConversationStats",
        "ConversationContext",
    }:
        # Import from appropriate service module
        if name in {"ExplorationCoordinator", "ExplorationResult"}:
            module = importlib.import_module("victor.agent.services.exploration_runtime")
        elif name in {
            "SafetyCoordinator",
            "SafetyRule",
            "SafetyCheckResult",
            "SafetyStats",
            "SafetyAction",
            "SafetyCategory",
        }:
            module = importlib.import_module("victor_contracts.safety")
        elif name in {
            "PlanningCoordinator",
            "PlanningConfig",
            "PlanningMode",
            "PlanningResult",
        }:
            module = importlib.import_module("victor.agent.services.planning_runtime")
        elif name in {
            "ConversationCoordinator",
            "TurnType",
            "ConversationTurn",
            "ConversationStats",
            "ConversationContext",
        }:
            module = importlib.import_module("victor_contracts.conversation")

        value = getattr(module, name)
        warning = _get_deprecation_warning(name)
        if warning is not None:
            warnings.warn(warning, DeprecationWarning, stacklevel=2)
        globals()[name] = value
        return value

    # Original coordinator imports
    if name in _SUBMODULE_MAP:
        if _SUBMODULE_MAP[name] == "chat_protocols":
            module = importlib.import_module("victor.agent.services.protocols.chat_runtime")
        else:
            module = importlib.import_module(f"victor.agent.coordinators.{_SUBMODULE_MAP[name]}")
        value = getattr(module, name)
        warning = _get_deprecation_warning(name)
        if warning is not None:
            warnings.warn(
                warning,
                DeprecationWarning,
                stacklevel=2,
            )
        globals()[name] = value
        return value
    raise AttributeError(f"module 'victor.agent.coordinators' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return the lazily exported coordinator surface."""
    return sorted(list(globals().keys()) + __all__)
