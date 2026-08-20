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

"""Service protocols for the SOLID refactoring.

This package defines the service protocol interfaces used in the
SOLID refactoring of the AgentOrchestrator. These protocols follow
the Interface Segregation Principle (ISP) by providing focused,
single-purpose interfaces.

Architecture:
    Each protocol represents a focused service responsibility:
    - ChatServiceProtocol: Chat flow coordination and streaming
    - ToolServiceProtocol: Tool selection and execution
    - ContextServiceProtocol: Context management and metrics
    - ProviderServiceProtocol: Provider management and switching
    - RecoveryServiceProtocol: Error recovery and resilience
    - SessionServiceProtocol: Session lifecycle management
    - LLMDecisionServiceProtocol: LLM-assisted decision making

These protocols enable:
1. SRP compliance: Each service has a single responsibility
2. OCP compliance: New functionality via extension, not modification
3. LSP compliance: Substitutable service implementations
4. ISP compliance: Focused interfaces, no fat interfaces
5. DIP compliance: Depend on abstractions, not concretions

Example:
    from victor.agent.services.protocols import ChatServiceProtocol

    class MyChatService(ChatServiceProtocol):
        async def chat(self, user_message: str, **kwargs) -> CompletionResponse:
            # Implementation
            pass
"""

from victor.agent.services.protocols.chat_service import ChatServiceProtocol
from victor.agent.services.protocols.chat_runtime import (
    ChatContextProtocol,
    ChatOrchestratorProtocol,
    ChatRuntimeHelperAccessProtocol,
    ExecutionMode,
    PlanningContextProtocol,
    ProviderContextProtocol,
    ToolContextProtocol,
)
from victor.agent.services.protocols.context_service import ContextServiceProtocol
from victor.agent.services.protocols.decision_service import LLMDecisionServiceProtocol
from victor.agent.services.protocols.infrastructure_runtime import (
    IntentClassifierProtocol,
    ReminderManagerProtocol,
    ResponseSanitizerProtocol,
    StreamingConfidenceMonitorProtocol,
    StreamingHandlerProtocol,
    StreamingMetricsCollectorProtocol,
)
from victor.agent.services.protocols.provider_service import ProviderServiceProtocol
from victor.agent.services.protocols.recovery_service import RecoveryServiceProtocol
from victor.agent.services.protocols.runtime_support import (
    ChunkRuntimeProtocol,
    CoordinationAdvisorRuntimeProtocol,
    PromptRuntimeProtocol,
    RLLearningRuntimeProtocol,
    StateRuntimeProtocol,
    StreamingRecoveryRuntimeProtocol,
    TaskRuntimeProtocol,
    ToolPlanningRuntimeProtocol,
)
from victor.agent.services.protocols.session_service import SessionServiceProtocol
from victor.agent.services.protocols.session_ledger import SessionLedgerProtocol
from victor.agent.services.protocols.streaming_runtime import (
    StreamingChunkRuntimeProtocol,
    StreamingConversationStateProtocol,
    StreamingExecutionRuntimeProtocol,
    StreamingIntentClassifierRuntimeProtocol,
    StreamingMessageAdderProtocol,
    StreamingProviderRuntimeProtocol,
    StreamingReminderRuntimeProtocol,
    StreamingRLRuntimeProtocol,
    StreamingSanitizerRuntimeProtocol,
    StreamingTrackerRuntimeProtocol,
    ToolExecutionRecoveryRuntimeProtocol,
)
from victor.agent.services.protocols.tool_service import ToolServiceProtocol

__all__ = [
    "ChatServiceProtocol",
    "ChunkRuntimeProtocol",
    "ChatContextProtocol",
    "ChatOrchestratorProtocol",
    "ChatRuntimeHelperAccessProtocol",
    "CoordinationAdvisorRuntimeProtocol",
    "ContextServiceProtocol",
    "ExecutionMode",
    "IntentClassifierProtocol",
    "LLMDecisionServiceProtocol",
    "PlanningContextProtocol",
    "PromptRuntimeProtocol",
    "ProviderServiceProtocol",
    "ProviderContextProtocol",
    "RLLearningRuntimeProtocol",
    "ReminderManagerProtocol",
    "RecoveryServiceProtocol",
    "ResponseSanitizerProtocol",
    "SessionLedgerProtocol",
    "SessionServiceProtocol",
    "StreamingChunkRuntimeProtocol",
    "StreamingConfidenceMonitorProtocol",
    "StreamingConversationStateProtocol",
    "StreamingExecutionRuntimeProtocol",
    "StreamingHandlerProtocol",
    "StreamingIntentClassifierRuntimeProtocol",
    "StreamingMessageAdderProtocol",
    "StreamingMetricsCollectorProtocol",
    "StreamingProviderRuntimeProtocol",
    "StreamingReminderRuntimeProtocol",
    "StreamingRLRuntimeProtocol",
    "StateRuntimeProtocol",
    "StreamingSanitizerRuntimeProtocol",
    "StreamingRecoveryRuntimeProtocol",
    "StreamingTrackerRuntimeProtocol",
    "TaskRuntimeProtocol",
    "ToolExecutionRecoveryRuntimeProtocol",
    "ToolContextProtocol",
    "ToolPlanningRuntimeProtocol",
    "ToolServiceProtocol",
]
