#!/usr/bin/env python3
# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Select unit-test targets for a set of changed files (fast develop gate).

Maps each changed file to the unit tests worth running on a develop PR, leveraging the repo's
mirrored layout (``tests/unit`` mirrors ``victor``):

- A changed **test** file (``tests/unit/**/test_*.py``) -> run it directly.
- A changed **source** file (``victor/<rel>/<name>.py``) -> run its mirror tests
  ``tests/unit/<rel>/test_<name>.py`` and ``tests/unit/<rel>/test_<name>_*.py`` (when they exist).

Prints existing pytest targets (one per line, deduped, sorted). Empty output is valid only when
no core Python source changed. A changed ``victor/**.py`` file without a mirror test is an error:
silently deferring it to the later develop-to-main promotion leaves develop untested.

Usage: ``select_changed_tests.py <changed_file> [<changed_file> ...]``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Bound the fast gate's wall-clock budget: broad changes must be split or
# validated by the explicitly sharded full-suite gate. Never silently drop
# selected tests or unmapped core sources to fit this cap.
MAX_SELECTED_TARGETS = 200

# Lifecycle contracts span the extracted transport and the older client layout.
# Keep these bounded suites in the fast gate even for transport-only changes.
MCP_LIFECYCLE_TESTS = (
    "tests/unit/agent/test_mcp_client.py",
    "tests/unit/integrations/mcp/test_client_response_correlation.py",
    "tests/unit/security/test_mcp_factory_lifecycle.py",
)
MCP_SCHEMA_TESTS = ("tests/unit/tools/test_mcp_schema_fidelity.py",)
DEPRECATION_NOTICE_TESTS = ("tests/unit/test_release_deprecation_targets.py",)
DEPRECATION_NOTICE_FILES = (
    "victor/runtime/context.py",
    "victor/framework/config.py",
    "victor/core/verticals/registry_manager.py",
    "victor/framework/agentic_graph/state.py",
    "victor/agent/session_context_linker.py",
    "victor/workflows/compute_registry.py",
    "victor/workflows/executor.py",
    "victor/agent/coordinators/__init__.py",
    "victor/agent/resilience.py",
    "victor/tools/metadata.py",
    "victor/workflows/unified_compiler.py",
    "victor/tools/base.py",
    "victor/agent/sqlite_session_persistence.py",
)
RELATED_TESTS = {
    "victor/config/classification_settings.py": (
        "tests/unit/agent/services/test_classification_backend.py",
        "tests/unit/architecture/test_layering_facades.py",
    ),
    "victor/storage/embeddings/intent_classifier.py": (
        "tests/unit/agent/test_continuation_loop_fix.py",
        "tests/unit/architecture/test_layering_facades.py",
    ),
    **dict.fromkeys(
        (
            "victor/config/classification_backend.py",
            "victor/agent/services/classification_backend.py",
            "victor/config/decision_backend.py",
            "victor/agent/services/decision_backend.py",
            "victor/core/mode_controller.py",
            "victor/agent/mode_controller.py",
        ),
        (
            "tests/unit/agent/services/test_classification_backend.py",
            "tests/unit/architecture/test_layering_facades.py",
            "tests/unit/config/test_decision_settings_env.py",
            "tests/unit/rl/test_mode_transition_learner.py",
            "tests/unit/core/test_mode_controller.py",
            "tests/unit/tools/test_keyword_tool_selector.py",
            "tests/unit/agent/test_local_classifier_service.py",
            "tests/unit/agent/services/test_decision_service_flag_gate.py",
            "tests/unit/core/test_bootstrap_services.py",
            "tests/unit/tools/test_tool_access_controller.py",
            "tests/unit/tools/test_tool_selection_stage_readonly.py",
        ),
    ),
    "victor/agent/mode_controller.py": (
        "tests/unit/rl/test_mode_transition_learner.py",
        "tests/unit/tools/test_tool_access_controller.py",
        "tests/unit/architecture/test_layering_facades.py",
    ),
    "victor/agent/orchestrator_integration.py": (
        "tests/unit/agent/test_orchestrator_intelligent_hooks.py",
        "tests/unit/architecture/test_layering_facades.py",
    ),
    **dict.fromkeys(
        (
            "victor/core/rl/__init__.py",
            "victor/core/rl/hooks.py",
            "victor/framework/rl/hooks.py",
        ),
        (
            "tests/unit/architecture/test_layering_facades.py",
            "tests/unit/storage/cache/test_tiered_cache.py",
            "tests/unit/teams/test_rl_mixin.py",
            "tests/unit/agent/test_tool_executor_rl_emit.py",
            "tests/unit/agent/test_runtime_intelligence_pipeline.py",
            "tests/unit/agent/services/test_e3tir_integration.py",
            "tests/unit/agent/services/test_prompt_optimization_reward.py",
            "tests/unit/core/test_wiring_integration.py",
            "tests/unit/framework/rl/test_prompt_optimizer.py",
        ),
    ),
    **dict.fromkeys(
        (
            "victor/core/task/protocols.py",
            "victor/framework/task/protocols.py",
            "victor/core/task/direct_response.py",
            "victor/framework/task/direct_response.py",
            "victor/core/task_types.py",
            "victor/framework/task_types.py",
            "victor/agent/loop_guards.py",
            "victor/framework/loop/guards.py",
            "victor/agent/perception_integration.py",
            "victor/framework/perception_integration.py",
            "victor/core/task/__init__.py",
        ),
        (
            "tests/unit/agent/services/test_runtime_intelligence_service.py",
            "tests/unit/agent/test_continuation_strategy.py",
            "tests/unit/agent/test_query_classifier.py",
            "tests/unit/agent/test_task_intent_classifier.py",
            "tests/unit/agent/test_task_type_reaches_consumers.py",
            "tests/unit/architecture/test_layering_facades.py",
            "tests/unit/framework/task/test_direct_response.py",
            "tests/unit/framework/test_agentic_loop_spin_guard.py",
            "tests/unit/framework/test_client_close_guard.py",
            "tests/unit/framework/test_loop_guards.py",
            "tests/unit/framework/test_perception_integration.py",
            "tests/unit/framework/test_task_type_adapters.py",
            "tests/unit/framework/test_task_types.py",
            "tests/unit/storage/embeddings/test_task_classifier_extension.py",
        ),
    ),
    "victor/agent/member_approval_context.py": (
        "tests/unit/agent/subagents/test_member_approval.py",
        "tests/unit/agent/subagents/test_member_pause_trigger.py",
    ),
    "victor/config/groups/governance_settings.py": (
        "tests/unit/framework/test_durable_chat_pause.py",
    ),
    **dict.fromkeys(
        (
            "victor/core/action_recovery.py",
            "victor/framework/action_recovery.py",
            "victor/agent/approval_binding.py",
            "victor/framework/approval_binding.py",
            "victor/agent/approval_pause.py",
            "victor/framework/approval_pause.py",
            "victor/agent/execution_checkpoint.py",
            "victor/framework/execution_checkpoint.py",
            "victor/agent/hitl.py",
            "victor/framework/hitl.py",
            "victor/agent/policies/__init__.py",
            "victor/framework/policies/__init__.py",
            "victor/agent/policies/base.py",
            "victor/framework/policies/base.py",
            "victor/agent/policies/builtins.py",
            "victor/framework/policies/builtins.py",
            "victor/agent/policies/engine.py",
            "victor/framework/policies/engine.py",
            "victor/agent/policies/gate.py",
            "victor/framework/policies/gate.py",
            "victor/agent/policies/handlers.py",
            "victor/framework/policies/handlers.py",
            "victor/agent/policies/middleware.py",
            "victor/framework/policies/middleware.py",
            "victor/agent/policies/types.py",
            "victor/framework/policies/types.py",
        ),
        (
            "tests/unit/agent/services/test_chat_delivery.py",
            "tests/unit/agent/services/test_chat_runtime_services.py",
            "tests/unit/agent/services/test_chat_stream_governance.py",
            "tests/unit/agent/services/test_planning_runtime.py",
            "tests/unit/agent/services/test_tool_execution_runtime.py",
            "tests/unit/agent/services/test_turn_execution_runtime.py",
            "tests/unit/agent/subagents/test_member_approval.py",
            "tests/unit/agent/subagents/test_member_pause_trigger.py",
            "tests/unit/agent/test_durable_resume.py",
            "tests/unit/agent/test_paused_run_persistence.py",
            "tests/unit/agent/test_planning_coordinator_fixes.py",
            "tests/unit/agent/test_tool_pipeline.py",
            "tests/unit/architecture/test_layering_facades.py",
            "tests/unit/framework/agentic_graph/test_builder.py",
            "tests/unit/framework/policies/test_approval_handler_container_swap.py",
            "tests/unit/framework/policies/test_ask_context_arguments.py",
            "tests/unit/framework/policies/test_builder_wiring.py",
            "tests/unit/framework/policies/test_builtins.py",
            "tests/unit/framework/policies/test_engine.py",
            "tests/unit/framework/policies/test_handlers.py",
            "tests/unit/framework/policies/test_message_phases.py",
            "tests/unit/framework/policies/test_middleware.py",
            "tests/unit/framework/test_client_resume.py",
            "tests/unit/framework/test_durable_chat_pause.py",
            "tests/unit/framework/test_execution_checkpoint.py",
            "tests/unit/framework/test_graph.py",
            "tests/unit/framework/test_hitl_integration.py",
            "tests/unit/framework/test_hitl_protocol.py",
            "tests/unit/framework/test_streaming_pause.py",
            "tests/unit/runtime/test_container_singleton_guard.py",
            "tests/unit/scripts/test_select_changed_tests.py",
            "tests/unit/teams/test_member_pause_resume.py",
            "tests/unit/tools/test_search_router_pipeline.py",
            "tests/unit/ui/chat_app/test_app_settings.py",
            "tests/unit/ui/chat_app/test_approval.py",
            "tests/unit/ui/chat_app/test_chat_ui_trust.py",
            "tests/unit/ui/tui/test_app.py",
            "tests/unit/ui/tui/test_approval_modal.py",
        ),
    ),
    **dict.fromkeys(
        (
            "victor/agent/prompting/__init__.py",
            "victor/agent/prompting/document.py",
            "victor/agent/prompting/sections.py",
            "victor/agent/prompting/builder.py",
            "victor/agent/prompting/dictionary_compressor.py",
            "victor/framework/prompt_document.py",
            "victor/framework/prompt_sections.py",
            "victor/framework/prompt_builder.py",
            "victor/framework/prompt_dictionary_compressor.py",
        ),
        (
            "tests/unit/architecture/test_layering_facades.py",
            "tests/unit/framework/test_prompt_dictionary_compressor.py",
            "tests/unit/framework/test_prompt_document.py",
            "tests/unit/agent/prompting/test_workspace_context.py",
            "tests/unit/agent/services/test_prompt_compat_migration.py",
            "tests/unit/agent/test_prompt_coordinator.py",
            "tests/unit/agent/test_system_prompt_policy.py",
            "tests/unit/framework/test_framework_prompt_builder.py",
        ),
    ),
    "victor/integrations/api/change_tracker_ops.py": (
        "tests/unit/architecture/test_layering_facades.py",
        "tests/unit/agent/test_change_tracker.py",
    ),
    "victor/tools/plugin_registry.py": (
        "tests/unit/architecture/test_layering_facades.py",
        "tests/unit/agent/test_plugin_loader.py",
        "tests/unit/agent/test_plugin_manager.py",
    ),
    **dict.fromkeys(
        (
            "victor/core/entry_point_registry.py",
            "victor/framework/entry_point_registry.py",
        ),
        (
            "tests/unit/architecture/test_layering_facades.py",
            "tests/unit/framework/test_entry_point_registry.py",
            "tests/unit/framework/test_entry_point_caching.py",
        ),
    ),
    **dict.fromkeys(
        (
            "victor/core/module_loader.py",
            "victor/framework/module_loader.py",
        ),
        (
            "tests/unit/architecture/test_layering_facades.py",
            "tests/unit/framework/test_module_loader.py",
            "tests/unit/core/test_vertical_registry_reset.py",
        ),
    ),
    **dict.fromkeys(
        (
            "victor/core/trace_context.py",
            "victor/runtime/trace_context.py",
        ),
        (
            "tests/unit/architecture/test_layering_facades.py",
            "tests/unit/runtime/test_trace_context.py",
        ),
    ),
    **dict.fromkeys(
        (
            "victor/core/change_tracker.py",
            "victor/agent/change_tracker.py",
        ),
        (
            "tests/unit/architecture/test_layering_facades.py",
            "tests/unit/agent/test_change_tracker.py",
            "tests/unit/tools/test_file_editor_tool.py",
        ),
    ),
    **dict.fromkeys(
        (
            "victor/agent/request_scope_heuristics.py",
            "victor/framework/request_scope_heuristics.py",
        ),
        (
            "tests/unit/architecture/test_layering_facades.py",
            "tests/unit/framework/test_request_scope_heuristics.py",
        ),
    ),
    **dict.fromkeys(
        (
            "victor/agent/search_novelty.py",
            "victor/framework/search_novelty.py",
        ),
        (
            "tests/unit/architecture/test_layering_facades.py",
            "tests/unit/framework/test_search_novelty.py",
        ),
    ),
    **dict.fromkeys(
        (
            "victor/agent/preload.py",
            "victor/framework/preload.py",
        ),
        ("tests/unit/architecture/test_layering_facades.py",),
    ),
    **dict.fromkeys(
        (
            "victor/core/tool_progress.py",
            "victor/framework/tool_progress.py",
        ),
        (
            "tests/unit/architecture/test_layering_facades.py",
            "tests/unit/framework/test_tool_progress.py",
            "tests/unit/ui/rendering/test_tool_progress_render.py",
        ),
    ),
    **dict.fromkeys(
        (
            "victor/core/graph_algorithms.py",
            "victor/framework/graph_algorithms.py",
        ),
        (
            "tests/unit/architecture/test_layering_facades.py",
            "tests/unit/framework/test_graph.py",
            "tests/unit/core/test_graph_algorithms.py",
        ),
    ),
    **dict.fromkeys(
        (
            "victor/core/tool_naming.py",
            "victor/framework/tool_naming.py",
        ),
        (
            "tests/unit/architecture/test_layering_facades.py",
            "tests/unit/framework/test_tool_naming.py",
        ),
    ),
    **dict.fromkeys(
        (
            "victor/core/contextual_errors.py",
            "victor/framework/contextual_errors.py",
        ),
        (
            "tests/unit/architecture/test_layering_facades.py",
            "tests/unit/framework/test_contextual_errors.py",
        ),
    ),
    **dict.fromkeys(
        (
            "victor/tools/selection/sequence_tracker.py",
            "victor/agent/tool_sequence_tracker.py",
        ),
        (
            "tests/unit/architecture/test_layering_facades.py",
            "tests/unit/tools/test_tool_sequence_tracker.py",
        ),
    ),
    **dict.fromkeys(
        (
            "victor/tools/http_pool.py",
            "victor/core/verticals/defaults/tool_defaults.py",
        ),
        ("tests/unit/architecture/test_layering_facades.py",),
    ),
    **dict.fromkeys(
        ("victor/tools/unified/_search_helpers.py",),
        (
            "tests/unit/architecture/test_layering_facades.py",
            "tests/unit/tools/test_search_helpers_guards.py",
        ),
    ),
    "scripts/ci/layering_ratchet.py": ("tests/unit/architecture/test_layering_ratchet.py",),
    # Round-7 extractions: the shared install tail (per-language labels pinned
    # by the env_setup suite incl. java/timeout coverage) and the shared
    # frontier SELECT (both delegates exercised by the cross-store suite).
    **dict.fromkeys(
        (
            "victor/evaluation/env_setup.py",
            "victor/storage/graph/_frontier.py",
            "victor/storage/graph/duckdb_store.py",
            "victor/storage/graph/sqlite_store.py",
        ),
        (
            "tests/unit/core/test_env_setup.py",
            "tests/unit/storage/graph/test_frontier_shared.py",
        ),
    ),
    # jscpd round 4: the six re-exported RL wrappers resolve their coordinator
    # from victor.framework.rl's namespace now; the parity suite pins the
    # service surface -> global coordinator contract (with the seams at the
    # defining module), and the two LOCAL to_dict wrappers keep their seams.
    "victor/agent/services/rl_runtime.py": (
        "tests/unit/agent/services/test_service_runtime_parity.py",
    ),
    # The MiddlewareProtocol re-export consolidation: the whole verticals
    # protocol suite exercises the canonical class; the re-export identity is
    # pinned in the same suite.
    "victor/core/verticals/protocols/focused/tool_extensions.py": (
        "tests/unit/core/verticals/test_protocols_sdk_compat.py",
    ),
    # jscpd extraction: the accumulators live in three files now; their
    # behavior is pinned by the shared-module mirror suite plus the
    # proxima/victor suites that exercise both public classes end-to-end.
    **dict.fromkeys(
        (
            "victor/storage/vector_stores/_graph_accumulator.py",
            "victor/storage/vector_stores/change_impact.py",
            "victor/storage/vector_stores/issue_localization.py",
        ),
        (
            "tests/unit/storage/vector_stores/test_graph_accumulator.py",
            "tests/unit/providers/test_proximadb_multi.py",
        ),
    ),
    "victor/providers/azure_openai_provider.py": (
        "tests/unit/providers/test_openai_compat_coverage.py",
        "tests/unit/providers/test_openai_compat.py",
    ),
    **dict.fromkeys(
        (
            "victor/core/graph_rag/indexing.py",
            "victor/core/graph_rag/codegraph_repository.py",
            "victor/core/graph_rag/retrieval.py",
            "victor/core/graph_rag/precise_index.py",
            "victor/core/graph_rag/precise_providers.py",
            "victor/core/graph_rag/precise_cache.py",
            "victor/config/search_settings.py",
            # Vendored protobuf bindings (generated): exercised through the
            # precise-index suite's real-schema parse test.
            "victor/core/graph_rag/_scip/__init__.py",
            "victor/core/graph_rag/_scip/scip_pb2.py",
        ),
        (
            "tests/unit/core/graph_rag/test_scope_binding_integration.py",
            "tests/unit/core/graph_rag/test_precise_index.py",
            "tests/unit/core/graph_rag/test_precise_providers.py",
            "tests/unit/core/graph_rag/test_precise_cache.py",
        ),
    ),
    **dict.fromkeys(
        (
            "victor/agent/protocols/streaming_protocols.py",
            "victor/agent/protocols/provider_protocols.py",
            "victor/agent/protocols/tool_protocols.py",
            "victor/agent/protocols/__init__.py",
            "victor/agent/coordinators/__init__.py",
            "victor/agent/services/protocols/__init__.py",
            "victor/agent/services/protocols/recovery_service.py",
            "victor/agent/services/protocols/session_service.py",
            "victor/agent/services/protocols/provider_service.py",
            "victor/agent/services/protocols/context_service.py",
            "victor/agent/provider_tool_guidance.py",
        ),
        (
            "tests/unit/agent/services/test_protocols.py",
            "tests/unit/agent/test_orchestrator_di.py",
            "tests/unit/agent/test_chat_protocols.py",
            "tests/unit/agent/test_runtime_protocol_legacy_import_guard.py",
        ),
    ),
    **dict.fromkeys(
        ("victor/core/database.py", "victor/core/schema.py", "victor/core/migrations.py"),
        (
            "tests/unit/core/test_schema_version_migration.py",
            "tests/unit/core/test_project_database_wal.py",
            "tests/unit/core/test_decision_schema.py",
            "tests/unit/core/test_global_db_isolation.py",
            "tests/unit/core/test_sql_identifier_security.py",
        ),
    ),
    "victor/storage/graph/sqlite_publication.py": (
        "tests/unit/storage/graph/test_sqlite_store_cleanup.py",
        "tests/unit/storage/test_sqlite_graph_store_concurrency.py",
    ),
    "victor/storage/graph/sqlite_store.py": (
        "tests/unit/storage/graph/test_sqlite_store_cleanup.py",
        "tests/unit/storage/test_sqlite_graph_store.py",
        "tests/unit/storage/test_sqlite_graph_store_concurrency.py",
    ),
    "victor/storage/vector_stores/_lancedb_compat.py": (
        "tests/unit/storage/test_lancedb_compat.py",
        "tests/unit/providers/test_lancedb_provider.py",
        "tests/unit/storage/test_sqlite_lancedb.py",
    ),
    "victor/storage/vector_stores/lancedb_provider.py": (
        "tests/unit/providers/test_lancedb_provider.py",
    ),
    "victor/storage/unified/sqlite_lancedb.py": ("tests/unit/storage/test_sqlite_lancedb.py",),
    "victor/agent/conversation/session_model.py": (
        "tests/unit/agent/conversation/test_store_rich_metadata.py",
        "tests/unit/framework/test_client_conversation.py",
    ),
    "victor/agent/factory/runtime_builders.py": (
        "tests/unit/agent/services/test_chat_runtime_services.py",
        "tests/unit/agent/runtime/test_component_assembler.py",
        "tests/unit/agent/test_response_completer.py",
    ),
    "victor/agent/conversation/controller.py": (
        "tests/unit/agent/test_conversation_controller.py",
        "tests/unit/agent/services/test_context_lifecycle_service.py",
    ),
    "victor/agent/conversation/store_session.py": (
        "tests/unit/agent/test_conversation_memory.py",
        "tests/unit/agent/test_conversation_memory_async.py",
        "tests/unit/agent/services/test_session_runtime.py",
    ),
    "victor/ui/commands/utils.py": (
        "tests/unit/commands/test_cli.py",
        "tests/unit/config/test_logging_config.py",
        "tests/unit/ui/rendering/test_log_handler.py",
    ),
    **dict.fromkeys(
        ("victor/config/agentic_loop_settings.py", "victor/framework/agentic_loop_executor.py"),
        (
            "tests/unit/framework/test_agentic_loop_executor.py",
            "tests/unit/framework/test_session_config.py",
            "tests/unit/framework/test_agentic_loop_config.py",
            "tests/unit/framework/test_agentic_loop_verification.py",
            "tests/unit/agent/services/test_turn_execution_runtime.py",
            "tests/unit/agent/services/test_chat_stream_governance.py",
        ),
    ),
    "victor/framework/agentic_graph/executor.py": (
        "tests/unit/framework/agentic_graph/test_builder.py",
        "tests/unit/agent/services/test_turn_execution_runtime.py",
        "tests/unit/providers/test_gateway_policy_resilience.py",
    ),
    # Controller policy and its two loop consumers share stop precedence.
    "victor/framework/agentic_loop.py": (
        "tests/unit/agent/test_turn_evaluation_controller.py",
        "tests/unit/framework/test_loop_guards.py",
    ),
    "victor/framework/loop/guards.py": (
        "tests/unit/framework/test_loop_guards.py",
        "tests/unit/framework/test_agentic_loop_spin_guard.py",
        "tests/unit/framework/test_agentic_loop_run_streaming.py",
        "tests/unit/framework/test_agentic_loop_verification.py",
    ),
    "victor/agent/turn_policy.py": (
        "tests/unit/agent/test_turn_evaluation_controller.py",
        "tests/unit/framework/test_agentic_loop.py",
        "tests/unit/framework/test_agentic_loop_run_streaming.py",
        "tests/unit/framework/test_agentic_loop_verification.py",
    ),
    # Terminal rendering crosses the shared handler and CLI exit/REPL owners.
    **dict.fromkeys(
        (
            "victor/ui/commands/chat.py",
            "victor/ui/rendering/handler.py",
            "victor/ui/rendering/event_dispatcher.py",
        ),
        (
            "tests/unit/commands/test_cli.py",
            "tests/unit/core/test_stream_renderer.py",
            "tests/unit/ui/rendering/test_handler.py",
        ),
    ),
    # These descriptive owners predate the current source layout. Keep coverage
    # in those suites instead of adding duplicate source-named test files.
    "victor/agent/message_history.py": ("tests/unit/agent/test_agent_conversation.py",),
    "victor/agent/conversation/history_metadata.py": (
        "tests/unit/agent/test_agent_conversation.py",
        "tests/unit/agent/test_context_assembler.py",
        "tests/unit/agent/conversation/test_message_source.py",
    ),
    "victor/agent/services/task_runtime.py": ("tests/unit/agent/test_task_coordinator.py",),
    "victor/config/prompt_optimization_settings.py": (
        "tests/unit/evaluation/test_benchmark_ab_hygiene.py",
    ),
    "victor/storage/vector_stores/models.py": ("tests/unit/embeddings/test_embedding_models.py",),
    "victor/storage/embeddings/collections.py": (
        "tests/unit/embeddings/test_embedding_infrastructure.py",
        "tests/unit/core/test_data_cache_security.py",
        "tests/unit/framework/test_skill_matcher.py",
        "tests/unit/framework/test_skill_matcher_multi.py",
        "tests/unit/framework/test_skill_registry_wiring.py",
    ),
    "victor/storage/embeddings/service.py": (
        "tests/unit/embeddings/test_embedding_service.py",
        "tests/unit/embeddings/test_embedding_infrastructure.py",
        "tests/unit/embeddings/test_import_guards.py",
        "tests/unit/processing/test_graph_embeddings_batch.py",
    ),
    # Text policy and Python stream ownership are tested once across providers,
    # rather than copied into provider-named mirror suites.
    **dict.fromkeys(
        (
            "victor/providers/deepseek_provider.py",
            "victor/providers/cerebras_provider.py",
            "victor/providers/groq_provider.py",
            "victor/providers/sandhi_openai_compat_policy.py",
            "victor/providers/sandhi_transport.py",
        ),
        (
            "tests/unit/providers/test_sandhi_openai_compat_policy.py",
            "tests/unit/providers/test_provider_stream_aclosing.py",
        ),
    ),
    "victor/integrations/protocol/__init__.py": (
        "tests/unit/protocols/test_protocol_interface.py",
        "tests/unit/protocols/test_protocol_adapters.py",
    ),
    "victor/integrations/protocol/interface.py": (
        "tests/unit/protocols/test_protocol_interface.py",
    ),
    "victor/integrations/protocol/adapters.py": (
        "tests/unit/protocols/test_protocol_adapters.py",
        "tests/unit/api/test_vscode_extension_contract.py",
    ),
    "victor/agent/services/chat_stream_executor.py": (
        "tests/unit/agent/services/test_chat_stream_governance.py",
        "tests/unit/agent/services/test_chat_delivery.py",
        "tests/unit/agent/services/test_stream_turn_guards.py",
        "tests/unit/agent/services/test_streaming_act_port.py",
        "tests/unit/agent/services/test_streaming_act_adapter.py",
    ),
    **dict.fromkeys(
        (
            "victor/agent/services/turn_execution_runtime.py",
            "victor/agent/services/streaming_act_adapter.py",
            "victor/agent/streaming/tool_execution.py",
        ),
        (
            "tests/unit/agent/services/test_turn_execution_runtime.py",
            "tests/unit/agent/services/test_streaming_act_adapter.py",
            "tests/unit/agent/services/test_streaming_act_port.py",
            "tests/unit/agent/test_streaming_tool_execution_handler.py",
            "tests/unit/framework/test_agentic_loop_run_streaming.py",
            "tests/unit/framework/test_agentic_loop_verification.py",
            "tests/unit/framework/test_agentic_loop_effect_gate.py",
        ),
    ),
    "victor/agent/services/chat_runtime_services.py": (
        "tests/unit/agent/services/test_chat_stream_governance.py",
        "tests/unit/agent/services/test_turn_execution_runtime.py",
    ),
    "victor/agent/services/chat_stream_helpers.py": (
        "tests/unit/agent/services/test_empty_response_recovery.py",
        "tests/unit/agent/services/test_chat_stream_governance.py",
        "tests/unit/agent/services/test_chat_runtime_services_boundary.py",
        "tests/unit/agent/services/test_stream_connection_retry.py",
        "tests/unit/agent/services/test_chat_stream_helpers_gateway_retry.py",
    ),
    "victor/framework/policies/types.py": (
        "tests/unit/framework/policies/test_engine.py",
        "tests/unit/framework/policies/test_middleware.py",
    ),
    "victor/agent/orchestrator_creation.py": ("tests/unit/agent/test_profile_reasoning.py",),
    "victor/core/model_parameters.py": ("tests/unit/framework/test_model_effort.py",),
    "victor/config/provider_config_registry.py": (
        "tests/unit/providers/test_subscription_gateway.py",
        "tests/unit/providers/test_sandhi_transport.py",
    ),
    "victor/core/identity/gateway.py": ("tests/unit/providers/test_subscription_gateway.py",),
    "victor/framework/graph_runtime.py": (
        "tests/unit/framework/test_graph.py",
        "tests/unit/providers/test_gateway_policy_resilience.py",
    ),
    "victor/providers/gateway_boundary.py": (
        "tests/unit/providers/test_gateway_policy_resilience.py",
        "tests/unit/agent/services/test_stream_connection_retry.py",
    ),
    # Descriptive suites own these modules; provider_settings is the retired
    # path whose replacement is the grouped ProviderSettings contract.
    "victor/agent/strategies/provider_strategies.py": (
        "tests/unit/providers/test_provider_components.py",
    ),
    "victor/agent/tool_selection/selector.py": ("tests/unit/tools/test_tool_selection.py",),
    "victor/agent/unified_task_tracker.py": (
        "tests/unit/classification/test_tracker_classifier_integration.py",
    ),
    "victor/config/api_keys.py": (
        "tests/unit/providers/test_provider_kinds.py",
        "tests/unit/config/test_account_scoped_credentials.py",
        "tests/unit/security/test_secure_paths.py",
    ),
    "victor/config/groups/provider_config.py": (
        "tests/unit/config/test_inferflux_default.py",
        "tests/unit/config/test_settings_stratification.py",
    ),
    "victor/config/provider_settings.py": (
        "tests/unit/config/test_inferflux_default.py",
        "tests/unit/config/test_settings_stratification.py",
    ),
    "victor/config/tool_settings.py": (
        "tests/unit/config/test_tool_selection_access.py",
        "tests/unit/config/test_settings_stratification.py",
    ),
    "victor/framework/agent_factory.py": (
        "tests/unit/framework/test_skill_registry_wiring.py",
        "tests/unit/agent/test_runtime_provider_hardcodes.py",
    ),
    "victor/framework/enhanced_completion_evaluation.py": (
        "tests/unit/framework/test_enhanced_completion_evaluator.py",
    ),
    "scripts/ci/release_contract.py": ("tests/unit/scripts/test_security_workflow_contract.py",),
    "scripts/check_version_sync.py": ("tests/unit/scripts/test_security_workflow_contract.py",),
    ".github/workflows/release.yml": ("tests/unit/scripts/test_security_workflow_contract.py",),
    "victor/agent/action_observation.py": (
        "tests/unit/agent/test_durable_resume.py",
        "tests/unit/agent/test_paused_run_persistence.py",
        "tests/unit/framework/test_client_resume.py",
    ),
    "victor/agent/paused_run_store.py": (
        "tests/unit/framework/test_durable_chat_pause.py",
        "tests/unit/framework/test_streaming_pause.py",
        "tests/unit/agent/test_paused_run_persistence.py",
        "tests/unit/agent/test_paused_run_expiry.py",
        "tests/unit/agent/test_durable_resume.py",
        "tests/unit/framework/test_client_resume.py",
    ),
    # The service delegation and parity suites own the extracted protocol adapter.
    "victor/agent/services/orchestrator_protocol_adapter.py": (
        "tests/unit/agent/services/test_orchestrator_service_delegation.py",
        "tests/unit/agent/services/test_protocol_adapter_kwarg_strip.py",
        "tests/unit/agent/services/test_service_runtime_parity.py",
    ),
    # Definition-only contracts share the runtime behavior and static fixtures;
    # do not add a duplicate mirror suite just to satisfy source-name matching.
    "victor/agent/services/tool_execution_contracts.py": (
        "tests/unit/agent/services/test_tool_execution_runtime.py",
        "tests/unit/agent/services/test_typed_contracts.py",
    ),
    "victor/agent/services/chat_persistence.py": (
        "tests/unit/agent/services/test_chat_service.py",
        "tests/unit/agent/test_orchestrator_core.py",
        "tests/unit/runtime/test_hotspot_size_guard.py",
    ),
    "victor/framework/approval_pause.py": (
        "tests/unit/agent/services/test_tool_execution_runtime.py",
        "tests/unit/tools/test_search_router_pipeline.py",
        "tests/unit/agent/test_durable_resume.py",
        "tests/unit/teams/test_member_pause_resume.py",
        "tests/unit/teams/test_durable_pause_gating.py",
    ),
    "victor/framework/action_recovery.py": (
        "tests/unit/agent/test_durable_resume.py",
        "tests/unit/agent/test_paused_run_persistence.py",
        "tests/unit/framework/test_client_resume.py",
    ),
    "victor/framework/approval_binding.py": (
        "tests/unit/teams/test_member_pause_resume.py",
        "tests/unit/agent/test_durable_resume.py",
        "tests/unit/agent/test_paused_run_persistence.py",
        "tests/unit/framework/test_client_resume.py",
        "tests/unit/framework/policies/test_middleware.py",
    ),
    "victor/agent/tool_retry_safety.py": (
        "tests/unit/tools/test_tool_executor_unit.py",
        "tests/unit/agent/test_tool_pipeline.py",
        "tests/unit/agent/services/test_tool_retry.py",
    ),
    "victor/agent/factory/coordination_builders.py": (
        "tests/unit/framework/policies/test_builder_wiring.py",
    ),
    "victor/agent/middleware_chain.py": (
        "tests/unit/agent/test_continuation_loop_fix.py",
        "tests/unit/agent/test_tool_pipeline.py",
    ),
    "victor/framework/policies/gate.py": (
        "tests/unit/framework/policies/test_message_phases.py",
        "tests/unit/agent/services/test_chat_stream_governance.py",
        "tests/unit/agent/services/test_turn_execution_runtime.py",
    ),
    "scripts/ci/check_no_agent_attribution.py": (
        "tests/unit/scripts/test_check_no_agent_attribution.py",
    ),
    # Keep rename/deletion coverage explicit: the changed-file list contains
    # paths but no status, so the retired origin maps to its replacement suite.
    "victor/agent/services/chat_turn_lifecycle.py": (
        "tests/unit/agent/services/test_chat_turn_runtime.py",
    ),
    "victor/integrations/mcp/client.py": (*MCP_LIFECYCLE_TESTS, *MCP_SCHEMA_TESTS),
    "victor/integrations/mcp/protocol.py": (
        "tests/unit/agent/test_mcp_protocol.py",
        *MCP_SCHEMA_TESTS,
    ),
    "victor/integrations/mcp/server.py": (
        "tests/unit/agent/test_mcp_server.py",
        *MCP_SCHEMA_TESTS,
    ),
    "victor/tools/mcp_adapter_tool.py": MCP_SCHEMA_TESTS,
    "victor/agent/tool_executor.py": (
        "tests/unit/tools/test_tool_executor_unit.py",
        *MCP_SCHEMA_TESTS,
    ),
    "victor/integrations/mcp/stdio_transport.py": MCP_LIFECYCLE_TESTS,
    "victor/ui/cli_group.py": ("tests/unit/ui/test_cli_command_resolution.py",),
    "scripts/ci/select_changed_tests.py": ("tests/unit/scripts/test_select_changed_tests.py",),
    **dict.fromkeys(DEPRECATION_NOTICE_FILES, DEPRECATION_NOTICE_TESTS),
    "victor/tools/base.py": (
        *DEPRECATION_NOTICE_TESTS,
        "tests/unit/core/test_tool_base.py",
        *MCP_SCHEMA_TESTS,
    ),
}


class SelectionError(ValueError):
    """The changed core source cannot be covered safely by the fast gate."""


def select(changed: list[str]) -> list[str]:
    targets: set[str] = set()
    unmapped_sources: list[str] = []
    for raw in changed:
        p = raw.strip()
        if not p.endswith(".py"):
            continue
        existing_related = tuple(
            target for target in RELATED_TESTS.get(p, ()) if (ROOT / target).exists()
        )
        targets.update(existing_related)
        path = Path(p)
        name = path.name
        # Changed test file -> run it directly.
        if p.startswith("tests/") and name.startswith("test_"):
            if (ROOT / p).exists():
                targets.add(p)
            continue
        # Changed source file -> map to its mirror tests.
        if p.startswith("victor/"):
            rel = path.relative_to("victor")
            test_dir = ROOT / "tests" / "unit" / rel.parent
            candidates: list[Path] = []
            if test_dir.is_dir():
                candidates = list(test_dir.glob(f"test_{rel.stem}.py")) + list(
                    test_dir.glob(f"test_{rel.stem}_*.py")
                )
            # Some older suites flatten one source directory level (for
            # example benchmarks/deep_research.py is covered by
            # evaluation/test_deep_research_benchmark.py). Preserve those
            # explicit stem matches before declaring the source untested.
            if not candidates and not existing_related:
                unit_root = ROOT / "tests" / "unit"
                candidates = list(unit_root.rglob(f"test_{rel.stem}.py")) + list(
                    unit_root.rglob(f"test_{rel.stem}_*.py")
                )
            for cand in candidates:
                targets.add(str(cand.relative_to(ROOT)))
            # Fail-closed only when NOTHING covers the source: an explicit
            # RELATED_TESTS entry (e.g. the MCP lifecycle contracts for
            # transport-only changes) is coverage, even without a mirror file.
            if not candidates and not existing_related:
                unmapped_sources.append(p)
    if unmapped_sources:
        joined = "\n  - ".join(unmapped_sources)
        raise SelectionError(
            "changed core source has no mirrored unit test; add a test target before merging:\n"
            f"  - {joined}"
        )
    result = sorted(targets)
    if len(result) > MAX_SELECTED_TARGETS:
        raise SelectionError(
            f"select_changed_tests: {len(result)} targets exceed the "
            f"{MAX_SELECTED_TARGETS} cap — treating as a sweeping/mechanical change; "
            "split the change or run it through an explicitly sharded full-suite gate."
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*")
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="read one changed path per line from stdin (safe for spaces and leading dashes)",
    )
    args = parser.parse_args()
    changed = [line.rstrip("\n") for line in sys.stdin] if args.stdin else args.files
    try:
        for target in select(changed):
            print(target)
    except SelectionError as exc:
        print(f"select_changed_tests: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
