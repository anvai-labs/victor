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

"""Recovery system for handling model failures and stuck states.

This package provides a protocol-based, SOLID-compliant recovery system that:
- Detects various failure modes (empty responses, stuck loops, hallucinations)
- Applies appropriate recovery strategies based on context
- Learns optimal recovery approaches via Q-learning integration
- Provides telemetry for monitoring and improvement

FRAMEWORK INTEGRATION:
======================
This package integrates with existing Victor framework components:

1. Q-Learning: victor.agent.adaptive_mode_controller.QLearningStore
   - Reuses existing SQLite-backed Q-learning infrastructure
   - Recovery actions are learned alongside mode transitions

2. Telemetry: victor.agent.usage_analytics.UsageAnalytics
   - Recovery metrics recorded to existing analytics singleton
   - No duplicate telemetry collection

3. Context: victor.agent.context_compactor.ContextCompactor
   - Recovery coordinates with existing compaction system
   - Proactive compaction at configurable threshold

4. Circuit Breaker: victor.providers.circuit_breaker.CircuitBreaker
   - Model fallback respects existing circuit breaker states
   - Uses CircuitBreakerRegistry for consistent behavior

Architecture follows SOLID principles:
- Single Responsibility: Each strategy handles one recovery type
- Open/Closed: New strategies can be added without modifying existing code
- Liskov Substitution: All strategies implement RecoveryStrategy protocol
- Interface Segregation: Protocols define minimal required interfaces
- Dependency Inversion: High-level modules depend on abstractions (protocols)

Usage:
    from victor.agent.recovery import RecoveryCoordinator, FailureType

    # Create coordinator with framework integration
    coordinator = RecoveryCoordinator.create_with_framework(settings)

    # Detect failures
    failure = coordinator.detect_failure(
        content=response_content,
        tool_calls=tool_calls,
        mentioned_tools=mentioned_tools,
    )

    # Recover if needed
    if failure:
        outcome = await coordinator.recover(failure, provider, model, ...)
        # Apply outcome.result.message, outcome.new_temperature, etc.
        coordinator.record_outcome(success=True)
"""

from victor.agent.recovery.protocols import (
    RecoveryStrategy,
    RecoveryContext,
    RecoveryResult,
    RecoveryAction,
    FailureType,
    PromptTemplate,
    TemperaturePolicy,
)
from victor.agent.recovery.strategies import (
    EmptyResponseRecovery,
    StuckLoopRecovery,
    HallucinatedToolRecovery,
    TimeoutRecovery,
    CompositeRecoveryStrategy,
)
from victor.agent.recovery.temperature import ProgressiveTemperatureAdjuster
from victor.agent.recovery.prompts import ModelSpecificPromptRegistry
from victor.agent.recovery.coordinator import RecoveryCoordinator, RecoveryOutcome
from victor.agent.recovery.handler import RecoveryHandler, create_recovery_handler

__all__ = [
    # Protocols and Data Types
    "RecoveryStrategy",
    "RecoveryContext",
    "RecoveryResult",
    "RecoveryAction",
    "FailureType",
    "PromptTemplate",
    "TemperaturePolicy",
    # Strategies (extend for custom recovery)
    "EmptyResponseRecovery",
    "StuckLoopRecovery",
    "HallucinatedToolRecovery",
    "TimeoutRecovery",
    "CompositeRecoveryStrategy",
    # Recovery-specific Components
    "ProgressiveTemperatureAdjuster",
    "ModelSpecificPromptRegistry",
    # Coordinator (internal)
    "RecoveryCoordinator",
    "RecoveryOutcome",
    # Handler (DI-friendly, public interface)
    "RecoveryHandler",
    "create_recovery_handler",
]
