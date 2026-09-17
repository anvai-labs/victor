"""Component assembly phases for AgentOrchestrator.

Groups sequential factory component creation into logical phases,
reducing orchestrator __init__ from ~400 lines to method calls.
Each phase creates a related set of components and assigns them
to the orchestrator. Phases must run in order due to inter-dependencies.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.providers.base import BaseProvider

logger = logging.getLogger(__name__)


class ComponentAssembler:
    """Groups factory component creation into logical assembly phases."""

    @staticmethod
    def assemble_tools(
        orchestrator: AgentOrchestrator,
        provider: BaseProvider,
        model: str,
    ) -> None:
        """Create tool infrastructure: cache, graph, registry, execution, selection.

        Must run after provider setup, conversation, and memory runtime.
        """
        factory = orchestrator._factory

        # Tool cache and dependency graph
        orchestrator.tool_cache = factory.create_tool_cache()
        orchestrator.tool_graph = factory.create_tool_dependency_graph()

        # Code execution manager (Docker-based)
        orchestrator.code_manager = factory.create_code_execution_manager()

        # Tool registry and registrar
        orchestrator.tools = factory.create_tool_registry()
        orchestrator.tool_registry = orchestrator.tools  # backward compat alias

        orchestrator.tool_registrar = factory.create_tool_registrar(
            orchestrator.tools, orchestrator.tool_graph, provider, model
        )
        orchestrator.tool_registrar.set_background_task_callback(
            orchestrator._create_background_task
        )
        orchestrator.tool_registrar._register_tool_dependencies()

        # Tool registration and plugins
        orchestrator.tool_registrar.register_default_tools()
        orchestrator.tool_registrar._load_tool_configurations()
        orchestrator.tools.register_before_hook(orchestrator._log_tool_call)

        orchestrator.plugin_manager = factory.initialize_plugin_system(orchestrator.tool_registrar)

        # Argument normalizer, middleware, safety, auto-committer
        orchestrator.argument_normalizer = factory.create_argument_normalizer(provider)
        orchestrator._middleware_chain, orchestrator._code_correction_middleware = (
            factory.create_middleware_chain()
        )
        # Governance message gate (REQUEST/RESPONSE phases). Built once here and
        # shared by both the non-streaming TurnExecutor and the streaming
        # executor so message governance covers both paths. None unless the
        # policy engine is enabled with message-phase policies configured.
        from victor.agent.factory.coordination_builders import build_message_policy_gate

        orchestrator._message_policy_gate = build_message_policy_gate(
            factory.settings, factory.container, getattr(factory, "model", None)
        )
        orchestrator._safety_checker = factory.create_safety_checker()
        orchestrator._auto_committer = factory.create_auto_committer()

        # Tool executor
        orchestrator.tool_executor = factory.create_tool_executor(
            tools=orchestrator.tools,
            argument_normalizer=orchestrator.argument_normalizer,
            tool_cache=orchestrator.tool_cache,
            safety_checker=orchestrator._safety_checker,
            code_correction_middleware=orchestrator._code_correction_middleware,
        )

        # Response completer
        orchestrator.response_completer = factory.create_response_completer()

        # Semantic selection and tool selector
        orchestrator.use_semantic_selection, orchestrator._embedding_preload_task = (
            factory.setup_semantic_selection()
        )
        orchestrator._runtime_preload_task: Optional[asyncio.Task] = None
        orchestrator.semantic_selector = factory.create_semantic_selector()

        orchestrator.unified_tracker = factory.create_unified_tracker(
            orchestrator.tool_calling_caps
        )
        orchestrator.tool_selector = factory.create_tool_selector(
            tools=orchestrator.tools,
            semantic_selector=orchestrator.semantic_selector,
            conversation_state=orchestrator.conversation_state,
            unified_tracker=orchestrator.unified_tracker,
            model=orchestrator.model,
            provider_name=orchestrator.provider_name,
            tool_selection=orchestrator.tool_selection,
            on_selection_recorded=orchestrator._record_tool_selection,
        )

        # Access control and budget
        orchestrator._tool_access_controller = factory.create_tool_access_controller(
            registry=orchestrator.tools
        )
        orchestrator._budget_manager = factory.create_budget_manager()

    @staticmethod
    def assemble_conversation(
        orchestrator: AgentOrchestrator,
        provider: BaseProvider,
        model: str,
    ) -> None:
        """Create conversation management: controller, lifecycle, pipeline, streaming.

        Must run after assemble_tools.
        """
        factory = orchestrator._factory

        # Conversation controller
        orchestrator._conversation_controller = factory.create_conversation_controller(
            provider=provider,
            model=model,
            conversation=orchestrator.conversation,
            conversation_state=orchestrator.conversation_state,
            memory_manager=orchestrator.memory_manager,
            memory_session_id=orchestrator._memory_session_id,
            system_prompt=orchestrator._system_prompt,
            context_reminder_manager=(
                orchestrator.reminder_manager if hasattr(orchestrator, "reminder_manager") else None
            ),
            hierarchical_manager=(factory.create_hierarchical_compaction_manager()),
        )

        # Stage transition coordinator (Phase 16: Streaming Path Optimization)
        # Batches tool executions and applies Phase 1 optimizations consistently
        # across both streaming and non-streaming paths
        from victor.core.feature_flags import get_feature_flag_manager, FeatureFlag

        if get_feature_flag_manager().is_enabled(FeatureFlag.USE_STAGE_TRANSITION_COORDINATOR):
            from victor.agent.services import (
                StageTransitionCoordinator,
                HybridTransitionStrategy,
            )

            # Check if edge model is enabled for the hybrid strategy
            edge_model_enabled = get_feature_flag_manager().is_enabled(FeatureFlag.USE_EDGE_MODEL)

            orchestrator.transition_coordinator = StageTransitionCoordinator(
                state_machine=orchestrator.conversation_state,
                strategy=HybridTransitionStrategy(edge_model_enabled=edge_model_enabled),
                cooldown_seconds=2.0,  # Phase 1 cooldown
                min_tools_for_transition=5,  # Allow more work per turn
            )

            # Wire coordinator to state machine for batching
            orchestrator.conversation_state.set_transition_coordinator(
                orchestrator.transition_coordinator
            )

            logger.info("[ComponentAssembler] StageTransitionCoordinator enabled")
        else:
            orchestrator.transition_coordinator = None

        # Session ledger and lifecycle manager
        orchestrator._session_ledger = factory.create_session_ledger()
        orchestrator._lifecycle_manager = factory.create_lifecycle_manager(
            conversation_controller=orchestrator._conversation_controller,
            metrics_collector=(
                orchestrator._metrics_coordinator.metrics_collector
                if hasattr(orchestrator, "_metrics_coordinator")
                else None
            ),
            context_compactor=(
                orchestrator._context_compactor
                if hasattr(orchestrator, "_context_compactor")
                else None
            ),
            sequence_tracker=(
                orchestrator._sequence_tracker
                if hasattr(orchestrator, "_sequence_tracker")
                else None
            ),
            usage_analytics=(
                orchestrator._usage_analytics if hasattr(orchestrator, "_usage_analytics") else None
            ),
            reminder_manager=(
                orchestrator._reminder_manager
                if hasattr(orchestrator, "_reminder_manager")
                else None
            ),
        )

        # [CANONICAL] Update SessionService with lifecycle_manager after it's created
        if hasattr(orchestrator, "_session_service") and orchestrator._session_service:
            orchestrator._session_service.set_lifecycle_manager(orchestrator._lifecycle_manager)

        # Tool deduplication and pipeline
        orchestrator._deduplication_tracker = factory.create_tool_deduplication_tracker()
        orchestrator._tool_pipeline = factory.create_tool_pipeline(
            tools=orchestrator.tools,
            tool_executor=orchestrator.tool_executor,
            tool_budget=orchestrator.tool_budget,
            tool_cache=orchestrator.tool_cache,
            argument_normalizer=orchestrator.argument_normalizer,
            on_tool_start=orchestrator._on_tool_start_callback,
            on_tool_complete=orchestrator._on_tool_complete_callback,
            deduplication_tracker=orchestrator._deduplication_tracker,
            middleware_chain=orchestrator._middleware_chain,
            search_router=orchestrator.search_router,
        )

        # Register the live ToolPipeline in DI so prompt-time guidance paths
        # can read online tool reputation from the same runtime instance.
        try:
            from victor.agent.tool_pipeline import ToolPipeline
            from victor.core.container import ServiceLifetime

            orchestrator._container.register_or_replace(
                ToolPipeline,
                lambda c: orchestrator._tool_pipeline,
                ServiceLifetime.SINGLETON,
            )
        except Exception as exc:
            logger.debug("ToolPipeline DI registration skipped: %s", exc)

        # Wire pending semantic cache
        if (
            hasattr(orchestrator, "_pending_semantic_cache")
            and orchestrator._pending_semantic_cache is not None
        ):
            orchestrator._tool_pipeline.set_semantic_cache(orchestrator._pending_semantic_cache)
            logger.info("[AgentOrchestrator] Semantic tool result cache enabled")
            orchestrator._pending_semantic_cache = None

        # Streaming infrastructure
        orchestrator._streaming_controller = factory.create_streaming_controller(
            streaming_metrics_collector=(orchestrator.streaming_metrics_collector),
            on_session_complete=(orchestrator._on_streaming_session_complete),
        )
        orchestrator._streaming_coordinator = factory.create_streaming_coordinator(
            streaming_controller=orchestrator._streaming_controller,
        )
        orchestrator._streaming_handler = factory.create_streaming_chat_handler(
            message_adder=orchestrator
        )

        from victor.agent.streaming import IterationCoordinator

        orchestrator._iteration_coordinator: Optional[IterationCoordinator] = None

        # Task analyzer for prompt/runtime classification fallback
        from victor.agent.task_analyzer import get_task_analyzer

        orchestrator._task_analyzer = get_task_analyzer()
        if hasattr(orchestrator._task_analyzer, "set_runtime_subject"):
            try:
                orchestrator._task_analyzer.set_runtime_subject(orchestrator)
            except Exception as exc:
                logger.debug("Task analyzer runtime-subject binding skipped: %s", exc)

    @staticmethod
    def assemble_intelligence(
        orchestrator: AgentOrchestrator,
    ) -> None:
        """Create intelligence/resilience: RL, context, analytics, resilience, observability.

        Must run after assemble_conversation.
        """
        factory = orchestrator._factory

        # RL coordinator
        orchestrator._rl_coordinator = factory.create_rl_coordinator()

        # Context pruning learner
        pruning_learner = None
        if orchestrator._rl_coordinator is not None:
            try:
                pruning_learner = orchestrator._rl_coordinator.get_learner("context_pruning")
            except (KeyError, AttributeError, TypeError) as e:
                logger.debug("context_pruning learner unavailable: %s", e)

        # Context compactor and manager
        orchestrator._context_compactor = factory.create_context_compactor(
            conversation_controller=orchestrator._conversation_controller,
            pruning_learner=pruning_learner,
        )

        # Context assembler (turn-boundary assembly for provider calls)
        try:
            session_ledger = getattr(orchestrator, "_session_ledger", None)
            # FEP-0023 P2: wire the tool-result dedup view-stage only when the
            # flag is on, so the assembler stays flag-agnostic (presence==active).
            from victor.core.feature_flags import FeatureFlag, is_feature_enabled

            tool_result_deduplicator = None
            if is_feature_enabled(FeatureFlag.USE_TOOL_RESULT_DEDUP):
                create_dedup = getattr(factory, "create_tool_result_deduplicator", None)
                if callable(create_dedup):
                    tool_result_deduplicator = create_dedup()
            orchestrator._context_assembler = factory.create_context_assembler(
                ledger=session_ledger,
                controller=orchestrator._conversation_controller,
                tool_result_deduplicator=tool_result_deduplicator,
            )
        except Exception as e:
            logger.debug("Context assembler creation failed (graceful): %s", e)
            orchestrator._context_assembler = None

        from victor.agent.context_manager import create_context_manager

        orchestrator._context_manager = create_context_manager(
            provider_name=orchestrator.provider_name,
            model=orchestrator.model,
            conversation_controller=orchestrator._conversation_controller,
            context_compactor=orchestrator._context_compactor,
            debug_logger=orchestrator.debug_logger,
            settings=orchestrator.settings,
        )

        from victor.agent.services.context_lifecycle_service import (
            ContextLifecycleService,
            LifecycleCompactionSummarizerAdapter,
        )

        max_context_tokens = int(
            getattr(getattr(orchestrator, "settings", None), "max_context_tokens", 0) or 0
        )
        if max_context_tokens <= 0:
            max_context_chars = getattr(
                getattr(orchestrator, "settings", None), "max_context_chars", None
            )
            if max_context_chars:
                max_context_tokens = max(1, int(max_context_chars) // 4)
            else:
                try:
                    max_context_tokens = int(orchestrator._get_model_context_window())
                except Exception:
                    max_context_tokens = 100000

        lifecycle_summarizer = None
        create_compaction_summarizer = getattr(factory, "create_compaction_summarizer", None)
        if callable(create_compaction_summarizer):
            try:
                strategy = create_compaction_summarizer(
                    ledger=getattr(orchestrator, "_session_ledger", None),
                    use_llm=bool(
                        getattr(
                            orchestrator.settings,
                            "context_compaction_use_llm_summary",
                            False,
                        )
                    ),
                )
                if strategy is not None:
                    lifecycle_summarizer = LifecycleCompactionSummarizerAdapter(
                        strategy,
                        ledger=getattr(orchestrator, "_session_ledger", None),
                    )
            except Exception as exc:
                logger.debug("Context lifecycle summarizer creation failed: %s", exc)

        orchestrator._context_lifecycle_service = ContextLifecycleService.with_defaults(
            max_tokens=max_context_tokens,
            min_messages_to_keep=6,
            overflow_threshold_percent=90.0,
            default_compaction_strategy=str(
                getattr(orchestrator.settings, "context_compaction_strategy", "tiered") or "tiered"
            ),
            conversation_store=getattr(orchestrator, "memory_manager", None),
            compaction_summarizer=lifecycle_summarizer,
        )

        # Usage analytics and sequence tracker
        orchestrator._usage_analytics = factory.create_usage_analytics()
        orchestrator._sequence_tracker = factory.create_sequence_tracker()

        # Tool output formatter
        orchestrator._tool_output_formatter = factory.create_tool_output_formatter(
            orchestrator._context_compactor
        )

        # Resilience and coordination runtime boundaries (FEP-0016: driven by the
        # init manager at this site; the resilience phase reads _context_compactor,
        # which exists here).
        orchestrator._init_manager.run_phase(orchestrator, "resilience_runtime")
        orchestrator._init_manager.run_phase(orchestrator, "coordination_runtime")

        # Observability
        orchestrator._observability = factory.create_observability()
