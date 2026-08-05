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

"""Streaming chat submodule for orchestrator.

This module contains the extracted streaming logic from AgentOrchestrator,
providing better testability and separation of concerns.

Components:
- StreamingChatContext: Dataclass holding all state for a streaming session
- StreamingChatHandler: Main handler for streaming chat iterations
- IterationCoordinator: Loop control and decision coordination
- StreamingCoordinator: Response processing and chunk aggregation (NEW)
- ContinuationHandler: Handles continuation action execution (P0 SRP refactor)
- ToolExecutionHandler: Handles tool execution phase (P0 SRP refactor)
- IterationResult: Result of a single iteration in the streaming loop
"""

from victor.agent.streaming.context import StreamingChatContext, create_stream_context
from victor.agent.streaming.coordinator import (
    CoordinatorConfig,
    IterationCoordinator,
    create_coordinator,
)
from victor.agent.streaming.streaming_coordinator import (
    StreamingCoordinator,
    create_streaming_coordinator,
)
from victor.agent.streaming.continuation import (
    ContinuationHandler,
    ContinuationResult,
    create_continuation_handler,
)
from victor.agent.streaming.tool_execution import (
    ToolExecutionHandler,
    ToolExecutionResult,
    create_tool_execution_handler,
)
from victor.agent.streaming.intent_classification import (
    IntentClassificationHandler,
    IntentClassificationResult,
    TrackingState,
    create_intent_classification_handler,
    create_tracking_state,
    apply_tracking_state_updates,
)
from victor.agent.streaming.handler import StreamingChatHandler
from victor.agent.streaming.iteration import IterationResult, IterationAction

__all__ = [
    "StreamingChatContext",
    "StreamingChatHandler",
    "IterationCoordinator",
    "StreamingCoordinator",
    "CoordinatorConfig",
    "ContinuationHandler",
    "ContinuationResult",
    "ToolExecutionHandler",
    "ToolExecutionResult",
    "IntentClassificationHandler",
    "IntentClassificationResult",
    "TrackingState",
    "IterationResult",
    "IterationAction",
    "create_stream_context",
    "create_coordinator",
    "create_streaming_coordinator",
    "create_continuation_handler",
    "create_tool_execution_handler",
    "create_intent_classification_handler",
    "create_tracking_state",
    "apply_tracking_state_updates",
]
