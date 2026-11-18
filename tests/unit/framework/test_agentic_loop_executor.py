"""Executor resolution: one typed setting, explicit errors and concrete snapshots."""

import pytest

from victor.framework.agentic_loop_executor import (
    AgenticLoopExecutor,
    resolve_agentic_loop_executor,
    use_stategraph_executor,
)


@pytest.mark.parametrize("raw", ["auto", " AUTO ", AgenticLoopExecutor.AUTO])
def test_parse_auto(raw):
    assert AgenticLoopExecutor.parse(raw) is AgenticLoopExecutor.AUTO


@pytest.mark.parametrize("raw", [None, "garbage", "", False, 1, {}])
def test_parse_rejects_invalid_explicit_values(raw):
    with pytest.raises(ValueError):
        AgenticLoopExecutor.parse(raw)


@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize("configured", ["auto", "standard", "stategraph"])
def test_resolve_explicit_choice_or_legacy_auto(monkeypatch, configured, flag):
    from unittest.mock import Mock

    read_flag = Mock(return_value=flag)
    monkeypatch.setattr("victor.core.feature_flags.is_feature_enabled", read_flag)
    expected = (
        "stategraph"
        if configured == "stategraph" or (configured == "auto" and flag)
        else "standard"
    )
    selected = resolve_agentic_loop_executor(configured)
    assert selected.value == expected
    assert read_flag.call_count == int(configured == "auto")
    read_flag.reset_mock()
    # A concrete admission decision never consults a changed global flag again.
    read_flag.return_value = not flag
    assert use_stategraph_executor(selected) is (expected == "stategraph")
    read_flag.assert_not_called()


def test_resolver_failure_is_not_a_silent_executor_switch(monkeypatch):
    from unittest.mock import Mock

    monkeypatch.setattr(
        "victor.core.feature_flags.is_feature_enabled",
        Mock(side_effect=RuntimeError("flag unavailable")),
    )
    with pytest.raises(RuntimeError, match="flag unavailable"):
        resolve_agentic_loop_executor()


def test_default_settings_executor_is_auto():
    from victor.config.agentic_loop_settings import AgenticLoopSettings

    assert AgenticLoopSettings().executor is AgenticLoopExecutor.AUTO


@pytest.mark.parametrize("first", ["victor.config.settings", "victor.framework.session_config"])
def test_settings_and_framework_cold_imports(first):
    import subprocess
    import sys

    script = (
        f"import {first}; "
        "from victor.config.settings import Settings; "
        "from victor.framework.session_config import SessionConfig; "
        "from victor.config.agentic_loop_settings import AgenticLoopExecutor as ConfigExecutor; "
        "from victor.framework.agentic_loop_executor import AgenticLoopExecutor; "
        "assert ConfigExecutor is AgenticLoopExecutor; "
        "assert Settings().agentic_loop.executor is AgenticLoopExecutor.AUTO"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# FOU-D2: durable-by-default checkpointer injection at the compile root
# (FEP-0047). The executor resolves the default checkpointer through a
# module-level seam; the storage factory owns settings/optim-out policy.
# ---------------------------------------------------------------------------


def test_executor_injects_default_checkpointer(monkeypatch, tmp_path):
    from unittest.mock import Mock

    from victor.framework.agentic_graph import executor as executor_module
    from victor.storage.checkpoints.backends.sqlite_backend import SQLiteCheckpointBackend
    from victor.storage.checkpoints.graph_adapter import GraphSQLiteCheckpointer

    fake = GraphSQLiteCheckpointer(SQLiteCheckpointBackend(storage_path=tmp_path))
    monkeypatch.setattr(executor_module, "_resolve_default_checkpointer", lambda: fake)

    executor = executor_module.AgenticLoopGraphExecutor(execution_context=Mock())
    assert executor.compiled._config.checkpoint.checkpointer is fake
    assert executor.compiled._config.checkpoint.checkpoint_after_nodes == {"act"}


def test_executor_without_durable_checkpointer_keeps_framework_default(monkeypatch):
    from unittest.mock import Mock

    from victor.framework.agentic_graph import executor as executor_module

    monkeypatch.setattr(executor_module, "_resolve_default_checkpointer", lambda: None)

    executor = executor_module.AgenticLoopGraphExecutor(execution_context=Mock())
    assert executor.compiled._config.checkpoint.checkpointer is None
