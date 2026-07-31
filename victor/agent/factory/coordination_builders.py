# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Coordination builder methods for OrchestratorFactory.

Provides creation methods for recovery, workflow, team coordination,
safety, middleware, context management, and workflow optimization components.

Part of CRITICAL-001: Monolithic Orchestrator decomposition.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING

from victor.config.tool_selection_access import is_semantic_tool_selection_enabled
from victor.agent.coordinators.factory_support import (
    create_coordination_advisor_runtime as build_coordination_advisor_runtime,
    create_coordination_state_passed_coordinator as build_coordination_state_passed_coordinator,
    create_exploration_coordinator as build_exploration_coordinator,
    create_exploration_state_passed_coordinator as build_exploration_state_passed_coordinator,
    create_safety_state_passed_coordinator as build_safety_state_passed_coordinator,
    create_system_prompt_state_passed_coordinator as build_system_prompt_state_passed_coordinator,
)

if TYPE_CHECKING:
    from pathlib import Path
    from victor.config.settings import Settings
    from victor.providers.base import BaseProvider
    from victor.agent.tool_calling import (
        BaseToolCallingAdapter,
        ToolCallingCapabilities,
    )
    from victor.agent.conversation.controller import ConversationController
    from victor.agent.context_compactor import ContextCompactor
    from victor.agent.recovery import RecoveryHandler
    from victor.agent.orchestrator_recovery import OrchestratorRecoveryIntegration
    from victor.agent.middleware_chain import MiddlewareChain
    from victor.agent.conversation.state_machine import ConversationStateMachine
    from victor.agent.conversation.assembler import TurnBoundaryContextAssembler
    from victor.agent.session_ledger import SessionLedger
    from victor.observability.integration import ObservabilityIntegration
    from victor.storage.embeddings.intent_classifier import IntentClassifier
    from victor.protocols.coordination import CoordinationAdvisorProtocol
    from victor.agent.protocols.infrastructure_protocols import (
        SafetyCheckerProtocol,
        AutoCommitterProtocol,
        CodeExecutionManagerProtocol,
        WorkflowRegistryProtocol,
    )
    from victor.agent.services.protocols import (
        ChunkRuntimeProtocol as ChunkGeneratorProto,
        ReminderManagerProtocol,
        RLLearningRuntimeProtocol as RLCoordinatorProtocol,
        StreamingRecoveryRuntimeProtocol as StreamingRecoveryCoordinatorProto,
        TaskRuntimeProtocol as TaskCoordinatorProtocol,
    )

logger = logging.getLogger(__name__)


def build_policy_emitter(
    container: Any,
) -> Optional[Callable[[str, Dict[str, Any]], None]]:
    """Build a sync emitter forwarding policy DENY/ASK to the event bus.

    The PolicyEngine calls the emitter synchronously from within an async
    context; ObservabilityBus.emit is async, so we schedule it on the running
    loop (best-effort, audit-only). Returns None if the bus is unavailable.

    Shared by tool-policy and message-policy wiring.
    """
    try:
        from victor.core.events.backends import ObservabilityBus

        bus = container.get_optional(ObservabilityBus)
    except Exception:  # pragma: no cover - defensive
        bus = None
    if bus is None:
        return None

    # Retain references so fire-and-forget audit tasks aren't GC'd mid-flight.
    pending: set = set()

    def _emit(topic: str, payload: Dict[str, Any]) -> None:
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # no running loop; skip audit emit
        task = loop.create_task(bus.emit(topic, payload, source="policy_engine"))
        pending.add(task)
        task.add_done_callback(pending.discard)

    return _emit


def resolve_policy_approval_handler(governance: Any, container: Any = None) -> Optional[Any]:
    """Resolve an approval handler for ASK verdicts (tool or message phases).

    Resolution order:
      1. A ``PolicyApprovalHandler`` registered in the DI container — used on any
         surface (HTTP API, TUI, websocket), taking precedence so non-TTY
         surfaces can supply their own elicitation.
      2. The built-in console handler, only when ``governance.interactive_approval``
         is set AND stdin is an interactive TTY.
      3. None — ASK then resolves via ``ask_fallback`` (default fail-safe deny).
    """
    # 1. Container-registered handler (works on non-TTY surfaces).
    if container is not None:
        try:
            from victor.framework.policies import PolicyApprovalHandler

            holder = container.get_optional(PolicyApprovalHandler)
            if holder is not None and getattr(holder, "handler", None) is not None:
                return holder.handler
        except Exception:  # pragma: no cover - defensive
            pass

    # 2. TTY console handler (opt-in).
    if not getattr(governance, "interactive_approval", False):
        return None
    try:
        import sys

        if not sys.stdin or not sys.stdin.isatty():
            logger.debug("interactive_approval set but stdin is not a TTY; skipping handler")
            return None
        from victor.framework.policies import console_approval_handler

        return console_approval_handler
    except Exception:  # pragma: no cover - defensive
        return None


def build_message_policy_gate(settings: Any, container: Any, model: Optional[str]) -> Optional[Any]:
    """Build the governance message gate (REQUEST/RESPONSE phases), or None.

    No-op unless the USE_POLICY_ENGINE feature flag and
    ``settings.governance.enabled`` are both set AND at least one message-phase
    policy is configured (redaction or block patterns). Mirrors
    ``CoordinationBuildersMixin._maybe_add_policy_engine`` for the tool path but
    returns a :class:`~victor.framework.policies.gate.MessagePolicyGate` for the
    turn boundary to consume.
    """
    try:
        from victor.core.feature_flags import FeatureFlag, is_feature_enabled

        if not is_feature_enabled(FeatureFlag.USE_POLICY_ENGINE):
            return None

        governance = getattr(settings, "governance", None)
        if governance is None or not getattr(governance, "enabled", False):
            return None

        from victor.framework.policies import (
            BlockPatternPolicy,
            MessagePolicyGate,
            Phase,
            PolicyContext,
            PolicyEngine,
            RedactContentPolicy,
        )

        policies: list = []
        if getattr(governance, "redact_patterns", None):
            policies.append(
                RedactContentPolicy(
                    governance.redact_patterns,
                    placeholder=getattr(governance, "redact_placeholder", "[REDACTED]"),
                )
            )
        if getattr(governance, "block_request_patterns", None):
            policies.append(
                BlockPatternPolicy(
                    governance.block_request_patterns,
                    phases={Phase.REQUEST},
                    reason="Your message was blocked by a content policy.",
                )
            )
        if getattr(governance, "block_response_patterns", None):
            policies.append(
                BlockPatternPolicy(
                    governance.block_response_patterns,
                    phases={Phase.RESPONSE},
                    reason="The response was withheld by a content policy.",
                )
            )

        if not policies:
            return None

        def _context_provider() -> "PolicyContext":
            return PolicyContext(model=model)

        engine = PolicyEngine(policies, event_emitter=build_policy_emitter(container))
        gate = MessagePolicyGate(
            engine,
            context_provider=_context_provider,
            approval_handler=resolve_policy_approval_handler(governance, container),
            ask_fallback=getattr(governance, "ask_fallback", "deny"),
        )
        logger.info(
            "Message policy gate enabled with %d policy(ies): %s",
            len(policies),
            ", ".join(p.name for p in policies),
        )
        return gate
    except Exception as e:  # pragma: no cover - never break runtime init
        logger.warning("Message policy gate wiring skipped: %s", e)
        return None


class CoordinationBuildersMixin:
    """Mixin providing coordination-related factory methods.

    Requires the host class to provide:
        - self.settings: Settings
        - self.provider: BaseProvider
        - self.model: str
        - self.provider_name: Optional[str]
        - self.container: DI container
    """

    def _get_runtime_intelligence_service(self) -> Any:
        """Get or create the canonical runtime-intelligence service for factory-built components."""
        runtime_intelligence = getattr(self, "_runtime_intelligence_service", None)
        if runtime_intelligence is not None:
            return runtime_intelligence

        try:
            from victor.agent.services.runtime_intelligence import (
                RuntimeIntelligenceService,
            )

            runtime_intelligence = RuntimeIntelligenceService.from_container(self.container)
        except Exception as exc:
            logger.debug("Could not create runtime intelligence service: %s", exc)
            runtime_intelligence = None

        self._runtime_intelligence_service = runtime_intelligence
        return runtime_intelligence

    def create_exploration_state_passed_coordinator(
        self,
        project_root: Optional["Path"] = None,
        max_results: int = 5,
    ) -> Any:
        """Create the state-passed exploration wrapper."""
        coordinator = build_exploration_state_passed_coordinator(
            settings=self.settings,
            project_root=project_root,
            max_results=max_results,
        )
        logger.debug("ExplorationStatePassedCoordinator created")
        return coordinator

    def create_system_prompt_state_passed_coordinator(
        self,
        task_analyzer: Optional[Any] = None,
    ) -> Any:
        """Create the canonical state-passed system prompt coordinator."""
        coordinator = build_system_prompt_state_passed_coordinator(
            container=self.container,
            task_analyzer=task_analyzer,
        )
        logger.debug("SystemPromptStatePassedCoordinator created")
        return coordinator

    def create_safety_state_passed_coordinator(self) -> Any:
        """Create the canonical state-passed safety wrapper."""
        coordinator = build_safety_state_passed_coordinator()
        logger.debug("SafetyStatePassedCoordinator created")
        return coordinator

    def create_coordination_advisor_runtime(self) -> Any:
        """Create the canonical service-owned coordination runtime."""
        from victor.agent.services.protocols import CoordinationAdvisorRuntimeProtocol

        try:
            runtime = self.container.get(CoordinationAdvisorRuntimeProtocol)
        except Exception:
            runtime = build_coordination_advisor_runtime()
        logger.debug("CoordinationAdvisorRuntime created")
        return runtime

    def create_coordination_state_passed_coordinator(
        self,
        *,
        coordination_runtime: Any,
        coordination_advisor: Optional[Any] = None,
        vertical_context: Optional[Any] = None,
    ) -> Any:
        """Create the canonical state-passed coordination wrapper."""
        coordinator = build_coordination_state_passed_coordinator(
            coordination_runtime=coordination_runtime,
            coordination_advisor=coordination_advisor,
            vertical_context=vertical_context,
        )
        logger.debug("CoordinationStatePassedCoordinator created")
        return coordinator

    def create_recovery_handler(self) -> Optional["RecoveryHandler"]:
        """Create recovery handler (from DI container)."""
        from victor.agent.protocols import RecoveryHandlerProtocol

        return self.container.get(RecoveryHandlerProtocol)

    def create_recovery_integration(
        self, recovery_handler: Optional["RecoveryHandler"]
    ) -> "OrchestratorRecoveryIntegration":
        """Create recovery integration submodule for clean delegation.

        Args:
            recovery_handler: RecoveryHandler instance (may be None if disabled)

        Returns:
            RecoveryIntegration instance
        """
        from victor.agent.orchestrator_recovery import create_recovery_integration

        integration = create_recovery_integration(
            recovery_handler=recovery_handler,
            settings=self.settings,
        )
        logger.debug("RecoveryIntegration created")
        return integration

    def create_recovery_coordinator(self) -> "StreamingRecoveryCoordinatorProto":
        """Create RecoveryCoordinator via DI container.

        Returns:
            StreamingRecoveryCoordinator instance for recovery coordination
        """
        from victor.agent.services.protocols import StreamingRecoveryRuntimeProtocol

        recovery_coordinator = self.container.get(StreamingRecoveryRuntimeProtocol)
        logger.debug("StreamingRecoveryCoordinator created via DI")
        return recovery_coordinator

    def create_chunk_generator(self) -> "ChunkGeneratorProto":
        """Create ChunkGenerator via DI container.

        Returns:
            ChunkGenerator instance for chunk generation
        """
        from victor.agent.services.protocols import ChunkRuntimeProtocol

        chunk_generator = self.container.get(ChunkRuntimeProtocol)
        logger.debug("ChunkGenerator created via DI")
        return chunk_generator

    def create_context_compactor(
        self,
        conversation_controller: "ConversationController",
        pruning_learner: Optional[Any] = None,
    ) -> "ContextCompactor":
        """Create context compactor for proactive context management.

        Args:
            conversation_controller: ConversationController instance for context tracking
            pruning_learner: Optional RL learner for adaptive pruning decisions

        Returns:
            ContextCompactor instance configured with settings
        """
        from victor.agent.context_compactor import (
            create_context_compactor,
            TruncationStrategy,
        )

        truncation_strategy_str = getattr(
            self.settings, "tool_truncation_strategy", "smart"
        ).lower()
        truncation_strategy_map = {
            "head": TruncationStrategy.HEAD,
            "tail": TruncationStrategy.TAIL,
            "both": TruncationStrategy.BOTH,
            "smart": TruncationStrategy.SMART,
        }
        truncation_strategy = truncation_strategy_map.get(
            truncation_strategy_str, TruncationStrategy.SMART
        )

        provider_name = str(getattr(self.settings, "provider", "")).lower()
        local_providers = {"ollama", "lmstudio", "vllm", "llamacpp", "local"}
        provider_type = "local" if any(p in provider_name for p in local_providers) else "cloud"

        compactor = create_context_compactor(
            controller=conversation_controller,
            proactive_threshold=getattr(self.settings, "context_proactive_threshold", 0.90),
            min_messages_after_compact=getattr(
                self.settings, "context_min_messages_after_compact", 8
            ),
            tool_result_max_chars=getattr(self.settings, "max_tool_output_chars", 8192),
            tool_result_max_lines=getattr(self.settings, "max_tool_output_lines", 200),
            truncation_strategy=truncation_strategy,
            preserve_code_blocks=True,
            enable_proactive=getattr(self.settings, "context_proactive_compaction", True),
            enable_tool_truncation=getattr(self.settings, "tool_result_truncation", True),
            pruning_learner=pruning_learner,
            provider_type=provider_type,
            runtime_intelligence=self._get_runtime_intelligence_service(),
        )

        rl_status = "with RL learner" if pruning_learner else "without RL learner"
        logger.debug(
            f"ContextCompactor created {rl_status}, "
            f"truncation_strategy={truncation_strategy}, provider_type={provider_type}"
        )
        return compactor

    def create_middleware_chain(
        self,
    ) -> Tuple[Optional["MiddlewareChain"], Optional[Any]]:
        """Create middleware chain with vertical extensions.

        Returns:
            Tuple of (MiddlewareChain or None, CodeCorrectionMiddleware or None)
        """
        middleware_chain: Optional["MiddlewareChain"] = None
        code_correction_middleware: Optional[Any] = None
        code_correction_enabled = getattr(self.settings, "code_correction_enabled", True)

        try:
            from victor.agent.middleware_chain import MiddlewareChain
            from victor.core.verticals.protocols import VerticalExtensions

            middleware_chain = MiddlewareChain()

            extensions = self.container.get_optional(VerticalExtensions)
            if extensions and extensions.middleware:
                for middleware in extensions.middleware:
                    middleware_chain.add(middleware)
                    logger.debug(f"Added middleware from vertical: {type(middleware).__name__}")

                for mw in extensions.middleware:
                    if "CodeCorrection" in type(mw).__name__:
                        code_correction_middleware = mw
                        break
            else:
                if code_correction_enabled:
                    try:
                        from victor.agent.code_correction_middleware import (
                            CodeCorrectionMiddleware,
                            CodeCorrectionConfig,
                        )

                        code_correction_auto_fix = getattr(
                            self.settings, "code_correction_auto_fix", True
                        )
                        code_correction_max_iterations = getattr(
                            self.settings, "code_correction_max_iterations", 1
                        )

                        code_correction_middleware = CodeCorrectionMiddleware(
                            config=CodeCorrectionConfig(
                                enabled=True,
                                auto_fix=code_correction_auto_fix,
                                max_iterations=code_correction_max_iterations,
                            )
                        )
                        logger.debug("Using fallback CodeCorrectionMiddleware (no vertical loaded)")
                    except ImportError as e:
                        logger.warning(f"CodeCorrectionMiddleware unavailable: {e}")
        except ImportError as e:
            logger.warning(f"Middleware chain unavailable: {e}")

        # Governance policy engine (opt-in: USE_POLICY_ENGINE + governance.enabled).
        # Assembled from existing primitives — adds a single CRITICAL-priority
        # middleware that gates tool calls with ALLOW/DENY/ASK verdicts.
        if middleware_chain is not None:
            self._maybe_add_policy_engine(middleware_chain)

        return middleware_chain, code_correction_middleware

    def _maybe_add_policy_engine(self, middleware_chain: "MiddlewareChain") -> None:
        """Conditionally add the governance PolicyEngineMiddleware to the chain.

        No-op unless the USE_POLICY_ENGINE feature flag and
        ``settings.governance.enabled`` are both set. Builtin policies are
        constructed from ``settings.governance``; the cost context is resolved
        lazily from the ConversationController (live token cost) and the
        configured model.
        """
        try:
            from victor.core.feature_flags import FeatureFlag, is_feature_enabled

            if not is_feature_enabled(FeatureFlag.USE_POLICY_ENGINE):
                return

            governance = getattr(self.settings, "governance", None)
            if governance is None or not getattr(governance, "enabled", False):
                return

            from victor.framework.policies import (
                AllowToolsPolicy,
                AskOnToolsPolicy,
                CostBudgetPolicy,
                DenyToolsPolicy,
                MaxToolCallsPolicy,
                PolicyContext,
                PolicyEngine,
                PolicyEngineMiddleware,
            )

            policies: list = []
            if getattr(governance, "cost_budget_usd", 0.0) or getattr(
                governance, "cost_ask_thresholds_usd", None
            ):
                policies.append(
                    CostBudgetPolicy(
                        max_cost_usd=(governance.cost_budget_usd or None),
                        ask_thresholds_usd=governance.cost_ask_thresholds_usd,
                        expensive_models=governance.expensive_models,
                    )
                )
            # Hard tool gates first: a DENY must win over an ASK for the same tool
            # (the engine short-circuits on the first DENY).
            if getattr(governance, "deny_tools", None):
                policies.append(DenyToolsPolicy(governance.deny_tools))
            if getattr(governance, "allow_tools", None):
                policies.append(AllowToolsPolicy(governance.allow_tools))
            if getattr(governance, "ask_on_tools", None):
                policies.append(AskOnToolsPolicy(governance.ask_on_tools))
            if getattr(governance, "max_tool_calls_per_session", 0):
                policies.append(MaxToolCallsPolicy(limit=governance.max_tool_calls_per_session))

            if not policies:
                logger.debug("Policy engine enabled but no policies configured; skipping")
                return

            model = getattr(self, "model", None)

            def _context_provider() -> PolicyContext:
                """Resolve a live session snapshot for policy evaluation."""
                cost = 0.0
                try:
                    from victor.agent.conversation.controller import (
                        ConversationController,
                    )

                    controller = self.container.get_optional(ConversationController)
                    if controller is not None:
                        cost = controller.get_session_cost_usd()
                except Exception:  # pragma: no cover - defensive
                    cost = 0.0
                return PolicyContext(cost_usd=cost, model=model)

            engine = PolicyEngine(policies, event_emitter=self._build_policy_emitter())
            # ADR-023 pillar 2: an explicit handler (e.g. a team member's member-tagging
            # wrapper) wins over the container-resolved one; None falls back unchanged.
            approval_handler = getattr(
                self, "_approval_handler_override", None
            ) or self._resolve_policy_approval_handler(governance)
            middleware = PolicyEngineMiddleware(
                engine,
                context_provider=_context_provider,
                approval_handler=approval_handler,
                ask_fallback=getattr(governance, "ask_fallback", "deny"),
            )
            middleware_chain.add(middleware)
            logger.info(
                "Policy engine enabled with %d policy(ies): %s",
                len(policies),
                ", ".join(p.name for p in policies),
            )
        except Exception as e:  # pragma: no cover - never break orchestrator init
            logger.warning(f"Policy engine wiring skipped: {e}")

    def _build_policy_emitter(self) -> Optional[Callable[[str, Dict[str, Any]], None]]:
        """Build a sync emitter that forwards policy DENY/ASK to the event bus."""
        return build_policy_emitter(self.container)

    def _resolve_policy_approval_handler(self, governance: Any) -> Optional[Any]:
        """Resolve an approval handler for ASK verdicts."""
        return resolve_policy_approval_handler(governance, self.container)

    def create_safety_checker(self) -> "SafetyCheckerProtocol":
        """Create safety checker with vertical safety patterns.

        Returns:
            SafetyChecker instance with registered patterns
        """
        from victor.agent.protocols import SafetyCheckerProtocol

        safety_checker = self.container.get(SafetyCheckerProtocol)

        try:
            from victor.core.verticals.protocols import VerticalExtensions

            extensions = self.container.get_optional(VerticalExtensions)
            if extensions and extensions.safety_extensions:
                for safety_ext in extensions.safety_extensions:
                    for pattern in safety_ext.get_bash_patterns():
                        safety_checker.add_custom_pattern(
                            pattern.pattern,
                            pattern.description,
                            pattern.risk_level,
                            pattern.category,
                        )
                    logger.debug(
                        f"Added safety patterns from vertical: {safety_ext.get_category()}"
                    )
        except Exception as e:
            logger.debug(f"Could not load vertical safety extensions: {e}")

        return safety_checker

    def create_auto_committer(self) -> Optional["AutoCommitterProtocol"]:
        """Create auto committer for AI-assisted commits.

        Returns:
            AutoCommitter instance or None
        """
        from victor.agent.protocols import AutoCommitterProtocol
        from victor.agent.auto_commit import get_auto_committer

        auto_commit_enabled = getattr(self.settings, "auto_commit_enabled", False)

        if auto_commit_enabled:
            auto_committer = self.container.get(AutoCommitterProtocol)
            logger.debug("AutoCommitter enabled for AI-assisted commits")
            return auto_committer
        else:
            return get_auto_committer()

    def create_code_execution_manager(self) -> "CodeExecutionManagerProtocol":
        """Create code execution manager for Docker-based code execution.

        Returns:
            CodeExecutionManager instance (started)
        """
        from victor.agent.protocols import CodeExecutionManagerProtocol

        return self.container.get(CodeExecutionManagerProtocol)

    def create_workflow_registry(self) -> "WorkflowRegistryProtocol":
        """Create workflow registry for managing workflow patterns.

        Returns:
            WorkflowRegistry instance
        """
        from victor.agent.protocols import WorkflowRegistryProtocol

        return self.container.get(WorkflowRegistryProtocol)

    def create_rl_coordinator(self) -> Optional["RLCoordinatorProtocol"]:
        """Create RL coordinator for reinforcement learning framework.

        Returns:
            RL coordinator instance if enabled, None otherwise
        """
        if not getattr(self.settings, "enable_continuation_rl_learning", False):
            logger.debug("RL learning disabled")
            return None

        try:
            from victor.agent.services.protocols import RLLearningRuntimeProtocol

            coordinator = self.container.get(RLLearningRuntimeProtocol)
            logger.info("RL: Coordinator initialized with unified database")
            return coordinator
        except Exception as e:
            logger.warning(f"RL: Failed to initialize RL coordinator: {e}")
            return None

    def create_reminder_manager(
        self, provider: str, task_complexity: str, tool_budget: int
    ) -> "ReminderManagerProtocol":
        """Create context reminder manager from DI container.

        Args:
            provider: Provider name
            task_complexity: Task complexity level
            tool_budget: Tool call budget

        Returns:
            ReminderManager instance
        """
        from victor.agent.services.protocols import ReminderManagerProtocol

        with self.container.create_scope() as scope:
            reminder_manager = scope.get(ReminderManagerProtocol)

        logger.debug(f"ReminderManager created for {provider} with complexity {task_complexity}")
        return reminder_manager

    def create_task_coordinator(self) -> "TaskCoordinatorProtocol":
        """Create TaskCoordinator via DI container.

        Returns:
            TaskCoordinator instance for task coordination
        """
        from victor.agent.services.protocols import TaskRuntimeProtocol

        task_coordinator = self.container.get(TaskRuntimeProtocol)
        logger.debug("TaskCoordinator created via DI")
        return task_coordinator

    def create_intent_classifier(self) -> "IntentClassifier":
        """Create intent classifier for semantic continuation/completion detection.

        Returns:
            IntentClassifier singleton instance
        """
        from victor.storage.embeddings.intent_classifier import IntentClassifier

        classifier = IntentClassifier.get_instance()
        logger.debug("IntentClassifier singleton retrieved")
        return classifier

    def create_coordination_advisor(
        self,
        vertical_context: Any,
    ) -> "CoordinationAdvisorProtocol":
        """Create the canonical coordination advisor for task/team/workflow routing.

        Args:
            vertical_context: VerticalContext with team specs and workflows

        Returns:
            Framework-native coordination advisor
        """
        from victor.framework.coordination_runtime import (
            create_vertical_coordination_advisor,
        )

        team_learner = None
        try:
            from victor.agent.services.protocols import RLLearningRuntimeProtocol

            rl_coordinator = self.container.get_optional(RLLearningRuntimeProtocol)
            if rl_coordinator:
                team_learner = rl_coordinator.get_learner("team_composition")
        except Exception as e:
            logger.debug(f"Could not get team composition learner: {e}")

        selection_strategy = getattr(self.settings, "team_selection_strategy", "hybrid")

        advisor = create_vertical_coordination_advisor(
            vertical_context=vertical_context,
            team_learner=team_learner,
            selection_strategy=selection_strategy,
        )

        logger.debug("Coordination advisor created with strategy=%s", selection_strategy)
        return advisor

    def setup_subagent_orchestration(self) -> tuple[Optional[Any], bool]:
        """Setup sub-agent orchestration with lazy initialization.

        Returns:
            Tuple of (None, enabled_flag) for lazy initialization pattern
        """
        enabled = getattr(self.settings, "subagent_orchestration_enabled", True)
        logger.debug(f"Sub-agent orchestration setup: enabled={enabled}")
        return (None, enabled)

    def setup_semantic_selection(self) -> tuple[bool, Optional[Any]]:
        """Setup semantic tool selection and background embedding preload.

        Returns:
            Tuple of (use_semantic_selection, embedding_preload_task_placeholder)
        """
        use_semantic = is_semantic_tool_selection_enabled(self.settings, default=False)
        logger.debug(f"Semantic selection setup: enabled={use_semantic}")
        return (use_semantic, None)

    def wire_component_dependencies(
        self,
        recovery_handler: Optional["RecoveryHandler"],
        context_compactor: "ContextCompactor",
        observability: Optional["ObservabilityIntegration"],
        conversation_state: Optional["ConversationStateMachine"],
    ) -> None:
        """Wire component dependencies after initialization.

        Args:
            recovery_handler: RecoveryHandler instance
            context_compactor: ContextCompactor instance
            observability: ObservabilityIntegration instance
            conversation_state: ConversationStateMachine instance
        """
        if recovery_handler and hasattr(recovery_handler, "set_context_compactor"):
            recovery_handler.set_context_compactor(context_compactor)
            logger.debug("RecoveryHandler wired with ContextCompactor")

        if observability and conversation_state:
            observability.wire_state_machine(conversation_state)
            logger.debug("Observability integration wired with ConversationStateMachine")

    def create_context_assembler(
        self,
        ledger: Optional["SessionLedger"] = None,
        controller: Optional["ConversationController"] = None,
        tool_result_deduplicator: Optional[object] = None,
    ) -> "TurnBoundaryContextAssembler":
        """Create TurnBoundaryContextAssembler for context selection.

        Args:
            ledger: SessionLedger instance
            controller: ConversationController (for semantic retrieval)
            tool_result_deduplicator: optional ToolResultDeduplicator wired as the
                FEP-0023 P2 view-stage (passed only when USE_TOOL_RESULT_DEDUP is on)
        """
        from victor.agent.conversation.assembler import TurnBoundaryContextAssembler
        from victor.agent.conversation.scoring import score_messages, CONTROLLER_WEIGHTS
        from victor.agent.conversation.types import ConversationMessage

        def _canonical_score_fn(messages, current_query=None):
            """Bridge provider Messages to canonical scorer."""
            conv_msgs = [ConversationMessage.from_provider_message(m) for m in messages]
            scored = score_messages(conv_msgs, current_query, weights=CONTROLLER_WEIGHTS)
            # Map back to original Message objects
            conv_to_orig = {id(cm): msg for cm, msg in zip(conv_msgs, messages)}
            return [(conv_to_orig[id(cm)], s) for cm, s in scored]

        assembler = TurnBoundaryContextAssembler(
            session_ledger=ledger,
            score_fn=_canonical_score_fn,
            conversation_controller=controller,
            tool_result_deduplicator=tool_result_deduplicator,
        )
        logger.debug("TurnBoundaryContextAssembler created")
        return assembler
