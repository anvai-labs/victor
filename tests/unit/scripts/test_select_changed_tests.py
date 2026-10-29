# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Changed-file -> mirror-test selection for the lightweight develop gate."""

from __future__ import annotations

import glob
import importlib.util
from pathlib import Path

import pytest


def _load():
    module_path = Path(__file__).resolve().parents[3] / "scripts" / "ci" / "select_changed_tests.py"
    spec = importlib.util.spec_from_file_location("select_changed_tests", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_mod = _load()
select = _mod.select


def test_source_file_maps_to_mirror_tests():
    # victor/agent/fast_pruning.py -> its mirror tests (base + suffixed variants that exist).
    out = select(["victor/agent/fast_pruning.py"])
    assert "tests/unit/agent/test_fast_pruning.py" in out
    assert "tests/unit/agent/test_fast_pruning_reference.py" in out


def test_source_file_maps_to_flattened_legacy_test_location():
    out = select(["victor/evaluation/benchmarks/deep_research.py"])
    assert "tests/unit/evaluation/test_deep_research_benchmark.py" in out


def test_changed_test_file_runs_directly():
    out = select(["tests/unit/agent/test_metrics_runtime.py"])
    assert out == ["tests/unit/agent/test_metrics_runtime.py"]


def test_mcp_transport_changes_include_lifecycle_and_legacy_contracts():
    required = {
        "tests/unit/agent/test_mcp_client.py",
        "tests/unit/integrations/mcp/test_client_response_correlation.py",
        "tests/unit/security/test_mcp_factory_lifecycle.py",
    }
    for source in (
        "victor/integrations/mcp/client.py",
        "victor/integrations/mcp/stdio_transport.py",
    ):
        assert required <= set(select([source]))


def test_selector_changes_select_its_own_regressions():
    assert "tests/unit/scripts/test_select_changed_tests.py" in select(
        ["scripts/ci/select_changed_tests.py"]
    )


@pytest.mark.parametrize(
    ("source", "owners"),
    [
        *[
            (
                source,
                {
                    "tests/unit/core/test_schema_version_migration.py",
                    "tests/unit/core/test_project_database_wal.py",
                    "tests/unit/core/test_decision_schema.py",
                    "tests/unit/core/test_global_db_isolation.py",
                    "tests/unit/core/test_sql_identifier_security.py",
                },
            )
            for source in (
                "victor/core/database.py",
                "victor/core/schema.py",
                "victor/core/migrations.py",
            )
        ],
        (
            "victor/agent/conversation/session_model.py",
            {
                "tests/unit/agent/conversation/test_store_rich_metadata.py",
                "tests/unit/framework/test_client_conversation.py",
            },
        ),
        (
            "victor/agent/factory/runtime_builders.py",
            {
                "tests/unit/agent/services/test_chat_runtime_services.py",
                "tests/unit/agent/runtime/test_component_assembler.py",
                "tests/unit/agent/test_response_completer.py",
            },
        ),
        (
            "victor/agent/turn_policy.py",
            {
                "tests/unit/agent/test_turn_policy.py",
                "tests/unit/agent/test_turn_evaluation_controller.py",
                "tests/unit/framework/test_agentic_loop.py",
                "tests/unit/framework/test_agentic_loop_run_streaming.py",
                "tests/unit/framework/test_agentic_loop_verification.py",
            },
        ),
        (
            "victor/framework/agentic_loop.py",
            {
                "tests/unit/agent/test_turn_evaluation_controller.py",
                "tests/unit/framework/test_loop_guards.py",
                *glob.glob("tests/unit/framework/test_agentic_loop*.py"),
            },
        ),
        (
            "victor/framework/loop/guards.py",
            {
                "tests/unit/framework/test_loop_guards.py",
                "tests/unit/framework/test_agentic_loop_spin_guard.py",
                "tests/unit/framework/test_agentic_loop_run_streaming.py",
                "tests/unit/framework/test_agentic_loop_verification.py",
            },
        ),
        *[
            (
                source,
                {
                    "tests/unit/commands/test_cli.py",
                    "tests/unit/core/test_stream_renderer.py",
                    "tests/unit/ui/rendering/test_handler.py",
                    *mirrors,
                },
            )
            for source, mirrors in (
                (
                    "victor/ui/commands/chat.py",
                    [
                        "tests/unit/ui/commands/test_chat_modes.py",
                        "tests/unit/ui/commands/test_chat_rl_feedback.py",
                        "tests/unit/ui/commands/test_chat_workflow_mode.py",
                    ],
                ),
                ("victor/ui/rendering/handler.py", []),
                (
                    "victor/ui/rendering/event_dispatcher.py",
                    [
                        "tests/unit/ui/rendering/test_event_dispatcher.py",
                    ],
                ),
            )
        ],
        (
            "victor/agent/services/chat_stream_executor.py",
            {
                "tests/unit/agent/services/test_chat_stream_executor_lifetime.py",
                "tests/unit/agent/services/test_chat_stream_governance.py",
                "tests/unit/agent/services/test_chat_delivery.py",
                "tests/unit/agent/services/test_stream_turn_guards.py",
                "tests/unit/agent/services/test_streaming_act_port.py",
                "tests/unit/agent/services/test_streaming_act_adapter.py",
            },
        ),
        (
            "victor/agent/message_history.py",
            ["tests/unit/agent/test_agent_conversation.py"],
        ),
        (
            "victor/agent/conversation/history_metadata.py",
            [
                "tests/unit/agent/test_agent_conversation.py",
                "tests/unit/agent/test_context_assembler.py",
                "tests/unit/agent/conversation/test_message_source.py",
            ],
        ),
        (
            "victor/agent/services/task_runtime.py",
            ["tests/unit/agent/test_task_coordinator.py"],
        ),
        *[
            (
                f"victor/providers/{provider}_provider.py",
                [
                    "tests/unit/providers/test_provider_stream_aclosing.py",
                    "tests/unit/providers/test_sandhi_openai_compat_policy.py",
                ]
                + (["tests/unit/providers/test_groq_provider.py"] if provider == "groq" else []),
            )
            for provider in ("deepseek", "cerebras", "groq")
        ],
        (
            "victor/agent/orchestrator_creation.py",
            ["tests/unit/agent/test_profile_reasoning.py"],
        ),
        (
            "victor/core/model_parameters.py",
            ["tests/unit/framework/test_model_effort.py"],
        ),
        (
            "victor/agent/services/orchestrator_protocol_adapter.py",
            [
                "tests/unit/agent/services/test_orchestrator_service_delegation.py",
                "tests/unit/agent/services/test_protocol_adapter_kwarg_strip.py",
                "tests/unit/agent/services/test_service_runtime_parity.py",
            ],
        ),
        (
            "victor/agent/services/tool_execution_contracts.py",
            [
                "tests/unit/agent/services/test_tool_execution_runtime.py",
                "tests/unit/agent/services/test_typed_contracts.py",
            ],
        ),
        (
            "victor/agent/services/chat_persistence.py",
            [
                "tests/unit/agent/services/test_chat_service.py",
                "tests/unit/agent/test_orchestrator_core.py",
                "tests/unit/runtime/test_hotspot_size_guard.py",
            ],
        ),
        (
            "victor/agent/paused_run_store.py",
            [
                "tests/unit/framework/test_durable_chat_pause.py",
                "tests/unit/framework/test_streaming_pause.py",
                "tests/unit/agent/test_paused_run_persistence.py",
                "tests/unit/agent/test_paused_run_expiry.py",
                "tests/unit/agent/test_durable_resume.py",
                "tests/unit/framework/test_client_resume.py",
            ],
        ),
        (
            "victor/framework/action_recovery.py",
            [
                "tests/unit/agent/test_durable_resume.py",
                "tests/unit/agent/test_paused_run_persistence.py",
                "tests/unit/framework/test_client_resume.py",
            ],
        ),
        (
            "victor/framework/approval_pause.py",
            [
                "tests/unit/agent/services/test_tool_execution_runtime.py",
                "tests/unit/tools/test_search_router_pipeline.py",
                "tests/unit/agent/test_durable_resume.py",
                "tests/unit/teams/test_member_pause_resume.py",
                "tests/unit/teams/test_durable_pause_gating.py",
            ],
        ),
        (
            "victor/framework/policies/types.py",
            [
                "tests/unit/framework/policies/test_engine.py",
                "tests/unit/framework/policies/test_middleware.py",
            ],
        ),
        (
            "victor/storage/graph/sqlite_store.py",
            {
                "tests/unit/storage/test_sqlite_graph_store.py",
                "tests/unit/storage/test_sqlite_graph_store_concurrency.py",
                "tests/unit/storage/graph/test_sqlite_store_cleanup.py",
                "tests/unit/storage/graph/test_sqlite_store_edge_file_tracking.py",
                "tests/unit/storage/graph/test_sqlite_store_edge_integrity.py",
                "tests/unit/storage/graph/test_sqlite_store_embedding_cleanup.py",
                "tests/unit/storage/graph/test_sqlite_store_parent_integrity.py",
            },
        ),
        (
            "victor/storage/vector_stores/_lancedb_compat.py",
            [
                "tests/unit/storage/test_lancedb_compat.py",
                "tests/unit/providers/test_lancedb_provider.py",
                "tests/unit/storage/test_sqlite_lancedb.py",
            ],
        ),
        (
            "victor/storage/vector_stores/lancedb_provider.py",
            ["tests/unit/providers/test_lancedb_provider.py"],
        ),
        (
            "victor/storage/unified/sqlite_lancedb.py",
            ["tests/unit/storage/test_sqlite_lancedb.py"],
        ),
        ("victor/ui/cli_group.py", ["tests/unit/ui/test_cli_command_resolution.py"]),
        (
            "victor/integrations/protocol/__init__.py",
            [
                "tests/unit/protocols/test_protocol_interface.py",
                "tests/unit/protocols/test_protocol_adapters.py",
            ],
        ),
        (
            "victor/integrations/protocol/interface.py",
            ["tests/unit/protocols/test_protocol_interface.py"],
        ),
        (
            "victor/integrations/protocol/adapters.py",
            [
                "tests/unit/protocols/test_protocol_adapters.py",
                "tests/unit/api/test_vscode_extension_contract.py",
            ],
        ),
        (
            "victor/agent/strategies/provider_strategies.py",
            ["tests/unit/providers/test_provider_components.py"],
        ),
        ("victor/agent/tool_selection/selector.py", ["tests/unit/tools/test_tool_selection.py"]),
        (
            "victor/agent/unified_task_tracker.py",
            ["tests/unit/classification/test_tracker_classifier_integration.py"],
        ),
        (
            "victor/config/api_keys.py",
            [
                "tests/unit/providers/test_provider_kinds.py",
                "tests/unit/config/test_account_scoped_credentials.py",
                "tests/unit/security/test_secure_paths.py",
            ],
        ),
        (
            "victor/config/groups/provider_config.py",
            [
                "tests/unit/config/test_inferflux_default.py",
                "tests/unit/config/test_settings_stratification.py",
            ],
        ),
        (
            "victor/config/provider_settings.py",
            [
                "tests/unit/config/test_inferflux_default.py",
                "tests/unit/config/test_settings_stratification.py",
            ],
        ),
        (
            "victor/config/tool_settings.py",
            [
                "tests/unit/config/test_tool_selection_access.py",
                "tests/unit/config/test_settings_stratification.py",
            ],
        ),
        (
            "victor/config/prompt_optimization_settings.py",
            ["tests/unit/evaluation/test_benchmark_ab_hygiene.py"],
        ),
        (
            "victor/framework/agent_factory.py",
            [
                "tests/unit/framework/test_skill_registry_wiring.py",
                "tests/unit/agent/test_runtime_provider_hardcodes.py",
            ],
        ),
        (
            "victor/framework/enhanced_completion_evaluation.py",
            ["tests/unit/framework/test_enhanced_completion_evaluator.py"],
        ),
        (
            "victor/agent/factory/coordination_builders.py",
            ["tests/unit/framework/policies/test_builder_wiring.py"],
        ),
        (
            "victor/agent/middleware_chain.py",
            [
                "tests/unit/agent/test_continuation_loop_fix.py",
                "tests/unit/agent/test_tool_pipeline.py",
            ],
        ),
        (
            "victor/storage/vector_stores/models.py",
            ["tests/unit/embeddings/test_embedding_models.py"],
        ),
        (
            "victor/storage/embeddings/collections.py",
            [
                "tests/unit/embeddings/test_embedding_infrastructure.py",
                "tests/unit/core/test_data_cache_security.py",
                "tests/unit/framework/test_skill_matcher.py",
                "tests/unit/framework/test_skill_matcher_multi.py",
                "tests/unit/framework/test_skill_registry_wiring.py",
            ],
        ),
        (
            "victor/storage/embeddings/service.py",
            [
                "tests/unit/embeddings/test_embedding_service.py",
                "tests/unit/embeddings/test_embedding_infrastructure.py",
                "tests/unit/embeddings/test_import_guards.py",
                "tests/unit/processing/test_graph_embeddings_batch.py",
            ],
        ),
        (
            "victor/framework/policies/gate.py",
            [
                "tests/unit/framework/policies/test_message_phases.py",
                "tests/unit/agent/services/test_chat_stream_governance.py",
                "tests/unit/agent/services/test_turn_execution_runtime.py",
            ],
        ),
        (
            "victor/providers/context_windows.py",
            [
                "tests/unit/providers/test_context_window.py",
                "tests/unit/providers/test_google_provider_ext.py",
                "tests/unit/providers/test_zai_provider_ext.py",
            ],
        ),
    ],
)
def test_source_maps_to_its_descriptive_regression_suites(source, owners):
    assert select([source]) == sorted(owners)


def test_stale_explicit_mapping_fails_closed(monkeypatch):
    monkeypatch.setitem(
        _mod.RELATED_TESTS,
        "victor/ui/cli_group.py",
        ("tests/unit/ui/test_does_not_exist.py",),
    )

    with pytest.raises(_mod.SelectionError, match="no mirrored unit test"):
        select(["victor/ui/cli_group.py"])


def test_non_python_and_unmapped_yield_nothing():
    # Docs/CI/config changes map to no unit tests -> empty (caller treats as pass).
    assert select(["README.md", ".github/workflows/ci-fast.yml", "Makefile"]) == []


def test_source_without_mirror_test_fails_closed():
    with pytest.raises(_mod.SelectionError, match="no mirrored unit test"):
        select(["victor/this_module_has_no_test_zzz.py"])


def test_retired_source_maps_to_its_replacement_suite():
    assert select(["victor/agent/services/chat_turn_lifecycle.py"]) == [
        "tests/unit/agent/services/test_chat_turn_runtime.py"
    ]


def test_results_are_deduped_and_sorted():
    out = select(
        [
            "victor/framework/client.py",
            "victor/framework/client.py",  # duplicate input
            "tests/unit/framework/test_client.py",  # also a direct hit
        ]
    )
    assert out == sorted(set(out))
    assert "tests/unit/framework/test_client.py" in out


def test_sweeping_change_exceeding_cap_fails_closed():
    # A repo-wide sweep cannot silently pass after selecting zero tests.
    # Use real, existing test files so the cap — not the existence check — is
    # what trips.
    real = sorted(glob.glob("tests/unit/**/test_*.py", recursive=True))
    assert len(real) > _mod.MAX_SELECTED_TARGETS, "need enough real tests to exceed the cap"
    over = real[: _mod.MAX_SELECTED_TARGETS + 25]
    with pytest.raises(_mod.SelectionError, match="exceed the"):
        select(over)


def test_at_or_below_cap_still_selects():
    real = sorted(glob.glob("tests/unit/**/test_*.py", recursive=True))
    under = real[:5]
    out = select(under)
    assert out == sorted(under) and len(out) == 5  # under cap → normal selection


def test_deprecation_notice_sources_map_to_release_contract() -> None:
    expected = "tests/unit/test_release_deprecation_targets.py"

    for source in _mod.DEPRECATION_NOTICE_FILES:
        assert expected in select([source])


def test_attribution_guard_selects_its_policy_regressions():
    assert select(["scripts/ci/check_no_agent_attribution.py"]) == [
        "tests/unit/scripts/test_check_no_agent_attribution.py"
    ]


@pytest.mark.parametrize(
    ("source", "contract_test"),
    [
        ("victor/integrations/mcp/client.py", "tests/unit/agent/test_mcp_client.py"),
        ("victor/integrations/mcp/protocol.py", "tests/unit/agent/test_mcp_protocol.py"),
        ("victor/integrations/mcp/server.py", "tests/unit/agent/test_mcp_server.py"),
        ("victor/tools/mcp_adapter_tool.py", "tests/unit/tools/test_mcp_adapter_tool.py"),
        ("victor/agent/tool_executor.py", "tests/unit/tools/test_tool_executor_unit.py"),
        ("victor/tools/base.py", "tests/unit/core/test_tool_base.py"),
    ],
)
def test_mcp_schema_owners_include_legacy_location_and_end_to_end_contract(source, contract_test):
    assert {contract_test, "tests/unit/tools/test_mcp_schema_fidelity.py"} <= set(select([source]))


def test_stream_helper_changes_select_retry_and_boundary_regressions():
    assert {
        "tests/unit/agent/services/test_chat_stream_governance.py",
        "tests/unit/agent/services/test_chat_runtime_services_boundary.py",
        "tests/unit/agent/services/test_stream_connection_retry.py",
        "tests/unit/agent/services/test_chat_stream_helpers_gateway_retry.py",
    } <= set(select(["victor/agent/services/chat_stream_helpers.py"]))


def test_governance_capability_changes_select_actual_publication_regression():
    selected = select(["victor/agent/services/chat_runtime_services.py"])
    assert {
        "tests/unit/agent/services/test_chat_stream_governance.py",
        "tests/unit/agent/services/test_turn_execution_runtime.py",
    } <= set(selected)


@pytest.mark.parametrize(
    "source",
    [
        "victor/agent/services/turn_execution_runtime.py",
        "victor/agent/services/streaming_act_adapter.py",
        "victor/agent/streaming/tool_execution.py",
    ],
)
def test_turn_contract_changes_select_producer_consumer_regressions(source):
    assert {
        "tests/unit/agent/services/test_turn_execution_runtime.py",
        "tests/unit/agent/services/test_streaming_act_adapter.py",
        "tests/unit/agent/services/test_streaming_act_port.py",
        "tests/unit/agent/test_streaming_tool_execution_handler.py",
        "tests/unit/framework/test_agentic_loop_run_streaming.py",
        "tests/unit/framework/test_agentic_loop_verification.py",
        "tests/unit/framework/test_agentic_loop_effect_gate.py",
    } <= set(select([source]))


def test_agentic_graph_executor_selects_actual_owner_not_unrelated_executors():
    assert set(select(["victor/framework/agentic_graph/executor.py"])) == {
        "tests/unit/framework/agentic_graph/test_builder.py",
        "tests/unit/agent/services/test_turn_execution_runtime.py",
        "tests/unit/providers/test_gateway_policy_resilience.py",
    }


# --- import-only changes (unused-import removal sweeps) ---------------------

only_removes_imports = _mod.only_removes_imports
partition_import_only = _mod.partition_import_only

_BASE = '''\
import os
import sys
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from x import Y

try:
    import tomllib
except ImportError:
    pass


def f(a: List[int]) -> int:
    """Doc."""
    return len(a) + len(os.sep)
'''


def test_removing_unused_imports_is_import_only():
    new = _BASE.replace("import sys\n", "").replace(", Dict", "")
    assert only_removes_imports(_BASE, new)


def test_dropping_emptied_type_checking_and_probe_blocks_is_import_only():
    new = _BASE.replace("if TYPE_CHECKING:\n    from x import Y\n\n", "")
    new = new.replace("try:\n    import tomllib\nexcept ImportError:\n    pass\n\n", "")
    new = new.replace("TYPE_CHECKING, ", "")
    assert only_removes_imports(_BASE, new)


def test_reformatting_and_noqa_comments_are_import_only():
    new = _BASE.replace("import os\n", "import os  # noqa: F401 - probe\n")
    assert only_removes_imports(_BASE, new)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.replace("import os\n", "import os\nimport json\n"),  # adds an import
        lambda s: s.replace("from x import Y", "from z import Y"),  # repoints an import
        lambda s: s.replace("len(a) + len(os.sep)", "len(a)"),  # logic change
        lambda s: s.replace('"""Doc."""', '"""Changed."""'),  # docstring is code-visible
        lambda s: s.replace("except ImportError:\n    pass", "except ImportError:\n    raise"),
    ],
)
def test_anything_beyond_import_removal_is_not_import_only(mutate):
    assert not only_removes_imports(_BASE, mutate(_BASE))


def test_new_or_unparsable_files_are_not_import_only():
    assert not only_removes_imports(None, _BASE)
    assert not only_removes_imports(_BASE, "def broken(:\n")


def test_partition_moves_import_only_core_sources_out_of_selection():
    bases = {"victor/a.py": _BASE, "victor/b.py": _BASE}
    current = {
        "victor/a.py": _BASE.replace("import sys\n", ""),  # import-only
        "victor/b.py": _BASE.replace("len(a) + len(os.sep)", "0"),  # real change
    }
    changed = ["victor/a.py", "victor/b.py", "tests/unit/test_x.py", "docs/x.md"]
    rest, import_only = partition_import_only(
        changed, read_base=bases.get, read_current=current.get
    )
    assert import_only == ["victor/a.py"]
    assert rest == ["victor/b.py", "tests/unit/test_x.py", "docs/x.md"]


def test_partition_without_base_reader_is_a_no_op():
    changed = ["victor/a.py"]
    assert partition_import_only(changed, read_base=None) == (changed, [])


def test_import_only_source_without_mirror_no_longer_blocks_selection():
    # victor/agent/argument_sanitizer.py has no mirror test; an import-only edit
    # to it must not raise, while a real edit still does.
    rest, import_only = partition_import_only(
        ["victor/agent/argument_sanitizer.py"],
        read_base=lambda p: _BASE,
        read_current=lambda p: _BASE.replace("import sys\n", ""),
    )
    assert import_only == ["victor/agent/argument_sanitizer.py"]
    assert select(rest) == []
    with pytest.raises(_mod.SelectionError):
        select(["victor/agent/argument_sanitizer.py"])


# --- import-only test modules ---------------------------------------------

only_drops_unused_test_imports = _mod.only_drops_unused_test_imports
drop_import_only_tests = _mod.drop_import_only_tests

_TEST_BASE = """\
import os
import pytest
from helpers import tmp_repo
from unittest.mock import patch


def test_a(tmp_repo):
    assert os.sep


def test_warns():
    with pytest.warns(DeprecationWarning):
        from pkg import Old


def test_probe():
    try:
        import optional_dep
    except ImportError:
        pytest.skip("absent")
"""


def test_dropping_an_unused_test_import_is_exempt():
    assert only_drops_unused_test_imports(
        _TEST_BASE, _TEST_BASE.replace("from unittest.mock import patch\n", "")
    )


def test_emptying_a_warns_block_is_not_exempt():
    # The import inside pytest.warns IS the assertion; ruff leaves `pass`.
    new = _TEST_BASE.replace("        from pkg import Old\n", "        pass\n")
    assert only_removes_imports(_TEST_BASE, new)  # the source rule alone would accept it
    assert not only_drops_unused_test_imports(_TEST_BASE, new)


def test_emptying_an_import_probe_is_not_exempt():
    new = _TEST_BASE.replace("        import optional_dep\n", "        pass\n")
    assert not only_drops_unused_test_imports(_TEST_BASE, new)


def test_removing_a_requested_fixture_is_not_exempt():
    # `tmp_repo` looks unused to ruff but test_a requests it as a fixture.
    new = _TEST_BASE.replace("from helpers import tmp_repo\n", "")
    assert not only_drops_unused_test_imports(_TEST_BASE, new)
    usefix = _TEST_BASE + '\n\n@pytest.mark.usefixtures("patch")\ndef test_b():\n    pass\n'
    assert not only_drops_unused_test_imports(
        usefix, usefix.replace("from unittest.mock import patch\n", "")
    )


def test_real_test_change_is_not_exempt():
    assert not only_drops_unused_test_imports(_TEST_BASE, _TEST_BASE.replace("os.sep", "0"))
    assert not only_drops_unused_test_imports(None, _TEST_BASE)


def test_drop_import_only_tests_partitions_tests_only():
    cleaned = _TEST_BASE.replace("from unittest.mock import patch\n", "")
    bases = {"tests/unit/test_x.py": _TEST_BASE, "victor/a.py": _TEST_BASE}
    current = {"tests/unit/test_x.py": cleaned, "victor/a.py": cleaned}
    changed = ["tests/unit/test_x.py", "victor/a.py", "docs/x.md"]
    rest, dropped = drop_import_only_tests(changed, read_base=bases.get, read_current=current.get)
    assert dropped == ["tests/unit/test_x.py"]
    assert rest == ["victor/a.py", "docs/x.md"]
    assert drop_import_only_tests(changed, read_base=None) == (changed, [])
