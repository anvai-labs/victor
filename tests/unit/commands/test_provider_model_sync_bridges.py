from __future__ import annotations

import inspect
import json
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, Mock, patch
from pathlib import Path

import pytest
import typer
import click

import victor.ui.commands.ab_testing as ab_testing_cmd
import victor.ui.commands.benchmark as benchmark_cmd
import victor.ui.commands.chat as chat_cmd
import victor.ui.commands.config as config_cmd
import victor.ui.commands.dashboard as dashboard_cmd
import victor.ui.commands.index as index_cmd
import victor.ui.commands.init as init_cmd
import victor.ui.commands.mcp as mcp_cmd
import victor.ui.commands.models as models_cmd
import victor.ui.commands.providers as providers_cmd
import victor.ui.commands.serve as serve_cmd
import victor.ui.commands.test_provider as test_provider_cmd
import victor.ui.commands.tools as tools_cmd
import victor.ui.commands.workflow as workflow_cmd


def _call_chat_command(**overrides: object) -> None:
    ctx = overrides.pop("ctx", MagicMock(invoked_subcommand=None))
    chat_cmd.chat(
        ctx,
        message=overrides.pop("message", None),
        message_opt=overrides.pop("message_opt", None),
        profile=overrides.pop("profile", "default"),
        stream=overrides.pop("stream", True),
        log_level=overrides.pop("log_level", None),
        thinking=overrides.pop("thinking", False),
        json_output=overrides.pop("json_output", False),
        plain=overrides.pop("plain", False),
        code_only=overrides.pop("code_only", False),
        stdin=overrides.pop("stdin", False),
        quiet=overrides.pop("quiet", False),
        renderer=overrides.pop("renderer", "auto"),
        mode=overrides.pop("mode", None),
        tool_budget=overrides.pop("tool_budget", None),
        max_iterations=overrides.pop("max_iterations", None),
        provider=overrides.pop("provider", None),
        model=overrides.pop("model", None),
        endpoint=overrides.pop("endpoint", None),
        input_file=overrides.pop("input_file", None),
        preindex=overrides.pop("preindex", False),
        vertical=overrides.pop("vertical", None),
        workflow=overrides.pop("workflow", None),
        validate_workflow=overrides.pop("validate_workflow", False),
        render_format=overrides.pop("render_format", None),
        render_output=overrides.pop("render_output", None),
        delegate_follow_up_contract=overrides.pop("delegate_follow_up_contract", None),
        delegate_next_step_id=overrides.pop("delegate_next_step_id", None),
        auth_mode=overrides.pop("auth_mode", None),
        coding_plan=overrides.pop("coding_plan", False),
        auto_skill=overrides.pop("auto_skill", None),
        enable_observability=overrides.pop("enable_observability", True),
        log_events=overrides.pop("log_events", False),
        show_reasoning=overrides.pop("show_reasoning", False),
        enable_planning=overrides.pop("enable_planning", None),
        planning_model=overrides.pop("planning_model", None),
        graph_watch=overrides.pop("graph_watch", True),
        list_sessions=overrides.pop("list_sessions", False),
        session_id=overrides.pop("session_id", None),
    )
    assert not overrides


class TestModelsSyncBridge:
    def test_list_models_uses_shared_sync_bridge(self) -> None:
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(models_cmd, "list_models_async", mock_async),
            patch.object(models_cmd, "run_sync", return_value=None) as mock_run_sync,
        ):
            models_cmd.list_models(provider="ollama", endpoint="http://localhost:11434")

        mock_async.assert_called_once_with("ollama", "http://localhost:11434")
        mock_run_sync.assert_called_once_with(coro)


class TestProvidersSyncBridge:
    def test_check_provider_uses_shared_sync_bridge(self) -> None:
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(providers_cmd, "_check_provider_async", mock_async),
            patch.object(providers_cmd, "run_sync", return_value=None) as mock_run_sync,
        ):
            providers_cmd.check_provider(
                provider="deepseek",
                model="deepseek-chat",
                connectivity=True,
                timeout=7.5,
                json_output=True,
            )

        mock_async.assert_called_once_with(
            provider="deepseek",
            model="deepseek-chat",
            connectivity=True,
            timeout=7.5,
            json_output=True,
        )
        mock_run_sync.assert_called_once_with(coro)

    def test_verify_provider_uses_shared_sync_bridge(self) -> None:
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(providers_cmd, "_verify_provider_async", mock_async),
            patch.object(providers_cmd, "run_sync", return_value=None) as mock_run_sync,
        ):
            providers_cmd.verify_provider(
                provider="anthropic",
                model="claude-3-5-haiku",
                api_key="secret",
            )

        mock_async.assert_called_once_with(
            provider="anthropic",
            model="claude-3-5-haiku",
            api_key="secret",
        )
        mock_run_sync.assert_called_once_with(coro)

    def test_auth_login_uses_shared_sync_bridge_and_normalizes_provider(self) -> None:
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(providers_cmd, "_auth_login_async", mock_async),
            patch.object(providers_cmd, "run_sync", return_value=None) as mock_run_sync,
        ):
            providers_cmd.auth_login(provider="OpenAI", force=True)

        mock_async.assert_called_once_with(provider="openai", force=True)
        mock_run_sync.assert_called_once_with(coro)

    def test_auth_login_rejects_unsupported_provider_before_bridge(self) -> None:
        with (
            patch.object(providers_cmd, "_auth_login_async") as mock_async,
            patch.object(providers_cmd, "run_sync") as mock_run_sync,
        ):
            with pytest.raises(typer.Exit):
                providers_cmd.auth_login(provider="unsupported", force=False)

        mock_async.assert_not_called()
        mock_run_sync.assert_not_called()


class TestAdditionalCommandSyncBridges:
    def test_list_tools_uses_shared_sync_bridge(self) -> None:
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(tools_cmd, "_list_tools_async", mock_async),
            patch.object(tools_cmd, "run_sync", return_value=None) as mock_run_sync,
        ):
            tools_cmd.list_tools(profile="default", lightweight=False)

        mock_async.assert_called_once_with("default")
        mock_run_sync.assert_called_once_with(coro)

    def test_mcp_uses_shared_sync_bridge_for_stdio(self) -> None:
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(mcp_cmd, "_run_mcp_server", mock_async),
            patch.object(mcp_cmd, "run_sync", return_value=None) as mock_run_sync,
            patch.object(mcp_cmd, "setup_logging"),
        ):
            mcp_cmd._mcp(stdio=True, log_level="info")

        mock_async.assert_called_once_with()
        mock_run_sync.assert_called_once_with(coro)

    def test_test_provider_uses_shared_sync_bridge(self) -> None:
        coro = object()
        mock_async = Mock(return_value=coro)
        ctx = MagicMock(invoked_subcommand=None)

        with (
            patch.object(test_provider_cmd, "test_provider_async", mock_async),
            patch.object(test_provider_cmd, "run_sync", return_value=None) as mock_run_sync,
            patch.object(test_provider_cmd.console, "print"),
        ):
            test_provider_cmd.test_provider(
                ctx,
                provider="openai",
                auth_mode="oauth",
                coding_plan=True,
                endpoint="coding",
            )

        mock_async.assert_called_once_with(
            "openai",
            auth_mode="oauth",
            coding_plan=True,
            endpoint="coding",
        )
        mock_run_sync.assert_called_once_with(coro)

    def test_index_uses_shared_sync_bridge(self) -> None:
        coro = object()
        mock_async = Mock(return_value=coro)
        ctx = MagicMock(invoked_subcommand=None)

        with (
            patch.object(index_cmd, "load_settings", return_value=object()),
            patch.object(index_cmd.os.path, "isdir", return_value=True),
            patch.object(index_cmd, "_build_index_async", mock_async),
            patch.object(index_cmd, "run_sync", return_value=True) as mock_run_sync,
            patch.object(index_cmd.console, "print"),
        ):
            index_cmd.index(ctx, force=True, path=".")

        mock_async.assert_called_once()
        call_args = mock_async.call_args
        assert call_args.args[0] == index_cmd.os.path.abspath(".")
        assert call_args.args[2] is True
        mock_run_sync.assert_called_once_with(coro)


class TestABTestingSyncBridge:
    @pytest.mark.parametrize(
        ("command", "helper_name", "args"),
        [
            (
                "create_experiment",
                "_create_experiment_async",
                (Path("experiment.yaml"),),
            ),
            ("start_experiment", "_start_experiment_async", ("exp_123",)),
            ("stop_experiment", "_stop_experiment_async", ("exp_123",)),
            ("show_status", "_show_status_async", ("exp_123",)),
            ("show_results", "_show_results_async", ("exp_123", True)),
        ],
    )
    def test_ab_command_uses_shared_sync_bridge(
        self,
        command: str,
        helper_name: str,
        args: tuple[object, ...],
    ) -> None:
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(ab_testing_cmd, helper_name, mock_async),
            patch.object(ab_testing_cmd, "run_sync", return_value=None) as mock_run_sync,
        ):
            getattr(ab_testing_cmd, command)(*args)

        mock_async.assert_called_once_with(*args)
        mock_run_sync.assert_called_once_with(coro)

    def test_list_experiments_uses_sync_helper(self) -> None:
        with patch.object(ab_testing_cmd, "_list_experiments") as mock_list:
            ab_testing_cmd.list_experiments(status_filter="running")

        mock_list.assert_called_once_with("running")


class TestConfigSyncBridge:
    def test_config_validate_uses_shared_sync_bridge_for_connectivity_checks(
        self,
        tmp_path: Path,
    ) -> None:
        config_dir = tmp_path / ".victor"
        config_dir.mkdir(exist_ok=True)
        (config_dir / "profiles.yaml").write_text("profiles:\n  default:\n    provider: ollama\n")

        settings = MagicMock()
        settings.get_config_dir.return_value = config_dir
        settings.load_profiles.return_value = {
            "default": MagicMock(
                provider="ollama",
                model="qwen",
                temperature=0.7,
                max_tokens=1024,
            )
        }
        settings.get_provider_settings.return_value = {}

        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(config_cmd, "load_settings", return_value=settings),
            patch.object(
                config_cmd,
                "validate_configuration",
                return_value=MagicMock(is_valid=True),
            ),
            patch.object(config_cmd, "format_validation_result", return_value="ok"),
            patch(
                "victor.providers.registry.ProviderRegistry.list_providers",
                return_value=["ollama"],
            ),
            patch.object(config_cmd, "_check_connectivity", mock_async),
            patch.object(config_cmd, "run_sync", return_value=None) as mock_run_sync,
            patch("builtins.print"),
            patch.object(config_cmd.Console, "print"),
        ):
            config_cmd.config_validate(verbose=True, check_connectivity=True, fix=False)

        mock_async.assert_called_once_with(settings, settings.load_profiles.return_value, True)
        mock_run_sync.assert_called_once_with(coro)


class TestBenchmarkSyncBridge:
    def test_setup_benchmark_uses_shared_sync_bridge(self) -> None:
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(benchmark_cmd, "_configure_log_level"),
            patch.object(benchmark_cmd, "_setup_benchmark_async", mock_async),
            patch.object(benchmark_cmd, "run_sync", return_value=None) as mock_run_sync,
        ):
            benchmark_cmd.setup_benchmark(
                benchmark="swe-bench",
                max_tasks=3,
                force_reindex=True,
                log_level="INFO",
            )

        mock_async.assert_called_once_with(
            benchmark="swe-bench",
            benchmark_lower="swe-bench",
            max_tasks=3,
            force_reindex=True,
            build_graph_rag=False,
        )
        mock_run_sync.assert_called_once_with(coro)

    def test_setup_benchmark_forwards_graph_rag_flag(self) -> None:
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(benchmark_cmd, "_configure_log_level"),
            patch.object(benchmark_cmd, "_setup_benchmark_async", mock_async),
            patch.object(benchmark_cmd, "run_sync", return_value=None),
        ):
            benchmark_cmd.setup_benchmark(
                benchmark="swe-bench",
                max_tasks=3,
                force_reindex=True,
                graph_rag=True,
                log_level="INFO",
            )

        assert mock_async.call_args.kwargs["build_graph_rag"] is True

    def test_run_prompt_suite_uses_shared_sync_bridge(self) -> None:
        runner = object()
        base_config = SimpleNamespace(
            model="model-x",
            provider="anthropic",
            dataset_metadata={},
        )
        coro = object()
        mock_async = Mock(return_value=coro)
        suite = MagicMock()

        with (
            patch.object(benchmark_cmd, "_configure_log_level"),
            patch.object(
                benchmark_cmd,
                "_resolve_benchmark_target",
                return_value=(object(), "bench-type", runner),
            ),
            patch.object(
                benchmark_cmd,
                "_resolve_account_selection",
                return_value=("anthropic", "model-x", None),
            ),
            patch.object(benchmark_cmd, "_resolve_effective_model", return_value="model-x"),
            patch("victor.evaluation.EvaluationConfig", return_value=base_config),
            patch.object(benchmark_cmd, "_attach_manifest_metadata"),
            patch.object(benchmark_cmd, "_print_benchmark_header"),
            patch.object(benchmark_cmd, "_run_prompt_candidate_suite_async", mock_async),
            patch.object(benchmark_cmd, "run_sync", return_value=suite) as mock_run_sync,
            patch.object(benchmark_cmd, "_print_prompt_candidate_suite_summary"),
            patch.object(benchmark_cmd.console, "print"),
        ):
            benchmark_cmd.run_prompt_suite(
                benchmark="humaneval",
                prompt_section="GROUNDING_RULES",
                candidate_hashes=["cand-123", "cand-456"],
                max_tasks=2,
                start_task=0,
                model="model-x",
                profile="default",
                output=None,
                dataset_path=None,
                timeout=30,
                max_turns=4,
                parallel=1,
                resume=False,
                provider="anthropic",
                log_level=None,
                debug_modules=None,
                no_edge_model=False,
                account=None,
                record_benchmark_results=False,
                promote_best=False,
                create_rollout=False,
                rollout_control_hash=None,
                rollout_traffic_split=0.1,
                rollout_min_samples_per_variant=100,
                analyze_rollout=False,
                apply_rollout_decision=False,
                rollout_decision_dry_run=False,
                min_approval_pass_rate=0.5,
                ci_gate=False,
            )

        mock_async.assert_called_once()
        call_kwargs = mock_async.call_args.kwargs
        assert call_kwargs["runner"] is runner
        assert call_kwargs["base_config"] is base_config
        assert call_kwargs["profile"] == "default"
        assert call_kwargs["model"] == "model-x"
        assert call_kwargs["timeout"] == 30
        assert call_kwargs["max_turns"] == 4
        assert call_kwargs["resume"] is False
        assert call_kwargs["provider_override"] == "anthropic"
        assert call_kwargs["start_task"] == 0
        specs = call_kwargs["candidate_specs"]
        assert [spec.section_name for spec in specs] == [
            "GROUNDING_RULES",
            "GROUNDING_RULES",
            "GROUNDING_RULES",
        ]
        assert [spec.prompt_candidate_hash for spec in specs] == [
            "__baseline__",
            "cand-123",
            "cand-456",
        ]
        assert [spec.provider for spec in specs] == [
            "anthropic",
            "anthropic",
            "anthropic",
        ]
        mock_run_sync.assert_called_once_with(coro)

    def test_run_benchmark_uses_shared_sync_bridge(self) -> None:
        runner = object()
        config = SimpleNamespace(model="model-x")
        coro = object()
        mock_async = Mock(return_value=coro)
        result = MagicMock()
        result.get_metrics.return_value = {
            "total_tasks": 1,
            "passed": 1,
            "failed": 0,
            "errors": 0,
            "pass_rate": 1.0,
            "duration_seconds": 1.0,
            "total_tokens": 42,
            "total_tool_calls": 5,
            "total_code_search_calls": 0,
            "total_graph_calls": 0,
            "tasks_using_code_intelligence": 0,
            "code_intelligence_task_coverage": 0.0,
        }
        result.task_results = []

        with (
            patch.object(benchmark_cmd, "_configure_log_level"),
            patch(
                "victor.evaluation.benchmarks.HumanEvalRunner",
                Mock(return_value=runner),
            ),
            patch("victor.evaluation.protocol.EvaluationConfig", return_value=config),
            patch.object(benchmark_cmd, "_run_benchmark_async", mock_async),
            patch.object(benchmark_cmd, "run_sync", return_value=result) as mock_run_sync,
            patch.object(benchmark_cmd.console, "print"),
        ):
            benchmark_cmd.run_benchmark(
                benchmark="humaneval",
                max_tasks=2,
                start_task=0,
                model="model-x",
                profile="default",
                output=None,
                timeout=30,
                max_turns=4,
                parallel=1,
                resume=False,
                provider=None,
                log_level=None,
                debug_modules=None,
                no_edge_model=False,
                account=None,
            )

        mock_async.assert_called_once_with(
            runner=runner,
            config=config,
            profile="default",
            model="model-x",
            timeout=30,
            max_turns=4,
            resume=False,
            provider_override=None,
            start_task=0,
            resolved_account=None,
        )
        mock_run_sync.assert_called_once_with(coro)


class TestDashboardSyncBridge:
    def test_dashboard_uses_shared_sync_bridge(self) -> None:
        coro = object()
        ctx = MagicMock(invoked_subcommand=None)
        mock_async = Mock(return_value=coro)

        with (
            patch("victor.ui.commands.utils.setup_logging"),
            patch.object(dashboard_cmd, "run_dashboard", mock_async),
            patch.object(dashboard_cmd, "run_sync", return_value=None) as mock_run_sync,
        ):
            dashboard_cmd.dashboard(
                ctx,
                log_file="events.jsonl",
                live=False,
                demo=True,
                log_level="DEBUG",
                wire_log=None,
            )

        mock_async.assert_called_once_with(
            log_file="events.jsonl", live=False, demo=True, wire_log=None
        )
        mock_run_sync.assert_called_once_with(coro)


class TestInitSyncBridge:
    def test_generate_init_content_enhanced_uses_shared_sync_bridge(self) -> None:
        coro = object()
        mock_async = Mock(return_value=coro)
        on_progress = Mock()

        with (
            patch.object(init_cmd, "_generate_init_content_async", mock_async),
            patch.object(init_cmd, "run_sync", return_value="content") as mock_run_sync,
        ):
            result = init_cmd._generate_init_content(
                mode="enhanced",
                use_llm=True,
                include_conversations=False,
                on_progress=on_progress,
                force=True,
                include_dirs=["src"],
                exclude_dirs=["tests"],
            )

        assert result == "content"
        mock_async.assert_called_once_with(
            mode="enhanced",
            use_llm=True,
            include_conversations=False,
            on_progress=on_progress,
            force=True,
            include_dirs=["src"],
            exclude_dirs=["tests"],
            provider=None,
            model=None,
            graph_context=None,
        )
        mock_run_sync.assert_called_once_with(coro)

    def test_generate_init_content_index_uses_shared_sync_bridge(self) -> None:
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(init_cmd, "_generate_init_content_async", mock_async),
            patch.object(init_cmd, "run_sync", return_value="content") as mock_run_sync,
        ):
            result = init_cmd._generate_init_content(
                mode="index",
                force=False,
                include_dirs=["src"],
                exclude_dirs=["tests"],
            )

        assert result == "content"
        mock_async.assert_called_once_with(
            mode="index",
            use_llm=False,
            include_conversations=False,
            on_progress=None,
            force=False,
            include_dirs=["src"],
            exclude_dirs=["tests"],
            provider=None,
            model=None,
            graph_context=None,
        )
        mock_run_sync.assert_called_once_with(coro)

    def test_generate_init_content_quick_bypasses_run_sync(self) -> None:
        """Quick mode is sync — should not call run_sync."""
        mock_smart = Mock(return_value="quick content")

        with (
            patch.object(init_cmd, "load_codebase_analyzer_attr", return_value=mock_smart),
            patch.object(init_cmd, "run_sync") as mock_run_sync,
        ):
            result = init_cmd._generate_init_content(
                mode="quick",
                include_dirs=["src"],
                exclude_dirs=["tests"],
            )

        assert result == "quick content"
        mock_run_sync.assert_not_called()


class TestServeSyncBridge:
    def test_serve_uses_shared_sync_bridge(self) -> None:
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(serve_cmd, "setup_logging"),
            patch.object(serve_cmd.console, "print"),
            patch.object(serve_cmd, "_run_fastapi_server", mock_async),
            patch.object(serve_cmd, "run_sync", return_value=None) as mock_run_sync,
        ):
            serve_cmd._serve(
                host="127.0.0.1",
                port=8765,
                log_level="info",
                profile="default",
                mode="build",
                enable_hitl=True,
                hitl_auth_token="token",
            )

        mock_async.assert_called_once_with("127.0.0.1", 8765, "default", "build", True, "token")
        mock_run_sync.assert_called_once_with(coro)

    def test_serve_rejects_unknown_mode_before_starting_server(self) -> None:
        with (
            patch.object(serve_cmd, "setup_logging") as mock_setup_logging,
            patch.object(serve_cmd.console, "print"),
            patch.object(serve_cmd, "_run_fastapi_server") as mock_run_server,
            pytest.raises(typer.Exit) as exc_info,
        ):
            serve_cmd._serve(
                host="127.0.0.1",
                port=8765,
                log_level="info",
                profile="default",
                mode="invalid",
                enable_hitl=False,
                hitl_auth_token=None,
            )

        assert exc_info.value.exit_code == 1
        mock_setup_logging.assert_not_called()
        mock_run_server.assert_not_called()

    def test_serve_hitl_uses_shared_sync_bridge(self) -> None:
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(serve_cmd, "setup_logging"),
            patch.object(serve_cmd.console, "print"),
            patch.object(serve_cmd, "_run_hitl_server", mock_async),
            patch.object(serve_cmd, "run_sync", return_value=None) as mock_run_sync,
        ):
            serve_cmd.serve_hitl(
                host="0.0.0.0",
                port=8080,
                auth_token="secret",
                require_auth=True,
                persistent=False,
                db_path=None,
                log_level="debug",
            )

        mock_async.assert_called_once_with("0.0.0.0", 8080, True, "secret", False, None)
        mock_run_sync.assert_called_once_with(coro)


class TestWorkflowSyncBridge:
    def test_run_workflow_uses_shared_sync_bridge(self) -> None:
        workflow = SimpleNamespace(
            name="demo-workflow",
            description="demo",
            start_node="start",
            nodes={},
            metadata={},
        )
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(
                workflow_cmd,
                "_load_workflow_file",
                return_value={"demo-workflow": workflow},
            ),
            patch.object(workflow_cmd, "_display_workflow_info"),
            patch.object(workflow_cmd.console, "print"),
            patch.object(workflow_cmd, "_execute_workflow_async", mock_async),
            patch.object(workflow_cmd, "run_sync", return_value=None) as mock_run_sync,
        ):
            workflow_cmd.run_workflow(
                "workflow.yaml",
                context=None,
                context_file=None,
                delegate_follow_up_contract=None,
                delegate_next_step_id=None,
                workflow_name=None,
                profile=None,
                dry_run=False,
                log_level=None,
            )

        mock_async.assert_called_once_with(workflow, {}, None)
        mock_run_sync.assert_called_once_with(coro)

    def test_generate_workflow_uses_shared_sync_bridge(self) -> None:
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(workflow_cmd.console, "print"),
            patch.object(workflow_cmd, "_generate_workflow_async", mock_async),
            patch.object(workflow_cmd, "run_sync", return_value=None) as mock_run_sync,
        ):
            workflow_cmd.generate_workflow(
                "Analyze code and fix issues",
                output="generated.yaml",
                vertical="coding",
                profile="default",
                strategy="multi_stage",
                interactive=True,
                validate=False,
                max_retries=5,
                dry_run=False,
            )

        mock_async.assert_called_once()
        kwargs = mock_async.call_args.kwargs
        assert kwargs["description"] == "Analyze code and fix issues"
        assert kwargs["output"] == "generated.yaml"
        assert kwargs["vertical"] == "coding"
        assert kwargs["profile"] == "default"
        assert kwargs["strategy"] == "multi_stage"
        assert kwargs["interactive"] is True
        assert kwargs["validate"] is False
        assert kwargs["max_retries"] == 5
        assert kwargs["dry_run"] is False
        assert kwargs["gen_strategy"].name == "LLM_MULTI_STAGE"
        mock_run_sync.assert_called_once_with(coro)


class TestChatSyncBridge:
    def test_chat_workflow_mode_uses_shared_sync_bridge(self) -> None:
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(chat_cmd.console, "print"),
            patch.object(chat_cmd, "run_workflow_mode", mock_async),
            patch.object(chat_cmd, "run_sync", return_value=None) as mock_run_sync,
        ):
            _call_chat_command(workflow="workflow.yaml", vertical="coding")

        mock_async.assert_called_once_with(
            workflow_path="workflow.yaml",
            validate_only=False,
            render_format=None,
            render_output=None,
            profile="default",
            vertical="coding",
            log_level=None,
            delegate_follow_up_contract=None,
            delegate_next_step_id=None,
        )
        mock_run_sync.assert_called_once_with(coro)

    def test_chat_workflow_mode_forwards_delegate_follow_up_options(self) -> None:
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(chat_cmd.console, "print"),
            patch.object(chat_cmd, "run_workflow_mode", mock_async),
            patch.object(chat_cmd, "run_sync", return_value=None) as mock_run_sync,
        ):
            _call_chat_command(
                workflow="workflow.yaml",
                delegate_follow_up_contract="delegate-follow-up.json",
                delegate_next_step_id="resume_delegate_retry",
            )

        mock_async.assert_called_once_with(
            workflow_path="workflow.yaml",
            validate_only=False,
            render_format=None,
            render_output=None,
            profile="default",
            vertical=None,
            log_level=None,
            delegate_follow_up_contract="delegate-follow-up.json",
            delegate_next_step_id="resume_delegate_retry",
        )
        mock_run_sync.assert_called_once_with(coro)

    def test_chat_oneshot_uses_shared_sync_bridge(self) -> None:
        settings = MagicMock()
        formatter = MagicMock()
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(chat_cmd, "setup_logging"),
            patch.object(chat_cmd, "setup_safety_confirmation"),
            patch.object(chat_cmd, "create_formatter", return_value=formatter),
            patch.object(chat_cmd.InputReader, "read_message", return_value="hello"),
            patch.object(chat_cmd, "load_settings", return_value=settings),
            patch(
                "victor.config.validation.validate_configuration",
                return_value=SimpleNamespace(is_valid=True),
            ),
            patch.object(chat_cmd, "run_oneshot", mock_async),
            patch.object(chat_cmd, "run_sync", return_value=None) as mock_run_sync,
            patch.object(chat_cmd.console, "print"),
        ):
            _call_chat_command(message_opt="hello", log_events=True, auto_skill=False)

        mock_async.assert_called_once_with(
            "hello",
            settings,
            "default",
            True,
            False,
            formatter=formatter,
            preindex=False,
            renderer_choice="auto",
            mode=None,
            tool_budget=None,
            max_iterations=None,
            vertical=None,
            enable_observability=True,
            enable_planning=None,
            planning_model=None,
            show_reasoning=False,
            session_config=ANY,
        )
        session_config = mock_async.call_args.kwargs["session_config"]
        assert isinstance(session_config, chat_cmd.SessionConfig)
        assert session_config.observability_logging is True
        assert session_config.auto_skill_enabled is False
        assert session_config.one_shot_mode is True
        mock_run_sync.assert_called_once_with(coro)

    def test_chat_interactive_uses_shared_sync_bridge(self) -> None:
        settings = MagicMock()
        formatter = MagicMock()
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(chat_cmd, "setup_logging"),
            patch.object(chat_cmd, "setup_safety_confirmation"),
            patch.object(chat_cmd, "create_formatter", return_value=formatter),
            patch.object(chat_cmd.InputReader, "read_message", return_value=None),
            patch.object(chat_cmd, "load_settings", return_value=settings),
            patch(
                "victor.config.validation.validate_configuration",
                return_value=SimpleNamespace(is_valid=True),
            ),
            patch.object(chat_cmd, "run_interactive", mock_async),
            patch.object(chat_cmd, "run_sync", return_value=None) as mock_run_sync,
            patch.object(chat_cmd.console, "print"),
        ):
            _call_chat_command(
                stream=False,
                thinking=True,
                session_id="session-1",
                log_events=True,
                auto_skill=False,
            )

        mock_async.assert_called_once_with(
            settings,
            "default",
            False,
            True,
            preindex=False,
            renderer_choice="auto",
            mode=None,
            tool_budget=None,
            max_iterations=None,
            vertical=None,
            enable_observability=True,
            enable_planning=None,
            planning_model=None,
            resume_session_id="session-1",
            show_reasoning=False,
            graph_watch=True,
            surface="tui",
            session_config=ANY,
        )
        session_config = mock_async.call_args.kwargs["session_config"]
        assert isinstance(session_config, chat_cmd.SessionConfig)
        assert session_config.observability_logging is True
        assert session_config.auto_skill_enabled is False
        assert session_config.one_shot_mode is False
        mock_run_sync.assert_called_once_with(coro)

    def test_chat_no_longer_exposes_legacy_mode_parameter(self) -> None:
        assert "legacy_mode" not in inspect.signature(chat_cmd.chat).parameters

    def test_default_interactive_uses_shared_sync_bridge(self) -> None:
        settings = MagicMock()
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(chat_cmd, "setup_logging"),
            patch.object(chat_cmd, "load_settings", return_value=settings),
            patch.object(chat_cmd, "setup_safety_confirmation"),
            patch.object(chat_cmd, "run_interactive", mock_async),
            patch.object(chat_cmd, "run_sync", return_value=None) as mock_run_sync,
        ):
            chat_cmd._run_default_interactive()

        mock_async.assert_called_once_with(settings, "default", True, False)
        mock_run_sync.assert_called_once_with(coro)
