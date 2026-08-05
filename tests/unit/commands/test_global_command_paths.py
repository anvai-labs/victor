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

"""Tests for centralized global Victor path resolution in CLI commands."""

import sqlite3
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from rich.console import Console

from victor.ui.commands import ab_testing as ab_testing_cmd
from victor.ui.commands import auth as auth_cmd
from victor.ui.commands import db as db_cmd
from victor.ui.commands import graph as graph_cmd
from victor.ui.commands import scheduler as scheduler_cmd
from victor.ui.commands import serve as serve_cmd


def test_get_oauth_status_uses_global_victor_dir(tmp_path):
    """OAuth status should load tokens from centralized Victor paths."""
    global_dir = tmp_path / ".victor"
    global_dir.mkdir(exist_ok=True)
    token_file = global_dir / "oauth_tokens.yaml"
    token_file.write_text("openai:\n  access_token: test-token\n", encoding="utf-8")

    with patch(
        "victor.ui.commands.auth.get_project_paths",
        return_value=SimpleNamespace(global_victor_dir=global_dir),
    ):
        status = auth_cmd._get_oauth_status("openai")

    assert status is auth_cmd.AuthStatus.AUTHENTICATED


def test_scheduler_start_uses_global_victor_dir_for_default_pid_file(tmp_path):
    """Scheduler should resolve its default PID path through centralized Victor paths."""
    global_dir = tmp_path / ".victor"

    with (
        patch(
            "victor.ui.commands.scheduler.get_project_paths",
            return_value=SimpleNamespace(global_victor_dir=global_dir),
        ),
        patch.object(scheduler_cmd, "_start_daemon") as mock_start_daemon,
    ):
        scheduler_cmd.start(daemon=True, pid_file=None, check_interval=15.0, config_file=None)

    mock_start_daemon.assert_called_once_with(global_dir / "scheduler.pid", 15.0, None)


def test_graph_watch_pid_file_uses_project_victor_dir(tmp_path):
    """Graph watch daemon PID files should live under the project .victor dir."""
    project_victor_dir = tmp_path / "repo" / ".victor"
    project_root = tmp_path / "repo"
    project_root.mkdir()

    with patch(
        "victor.ui.commands.graph.get_project_paths",
        return_value=SimpleNamespace(project_victor_dir=project_victor_dir),
    ):
        pid_file = graph_cmd._default_graph_watch_pid_file(project_root)

    assert pid_file.parent == project_victor_dir
    assert pid_file.name == "graph-watch.pid"
    assert pid_file.suffix == ".pid"


def test_db_archive_uses_global_victor_dir_for_default_output(tmp_path):
    """Database archive output should default under centralized Victor paths."""
    global_dir = tmp_path / ".victor"
    mock_db = MagicMock()
    mock_db.get_tables_for_group.return_value = ["rl_outcome"]
    mock_db.archive_table.return_value = 7

    with (
        patch(
            "victor.ui.commands.db.get_project_paths",
            return_value=SimpleNamespace(global_victor_dir=global_dir),
        ),
        patch("victor.core.database.get_database", return_value=mock_db),
    ):
        db_cmd.db_archive(before="2026-01-01", output=None, group="rl", yes=True)

    archive_path = mock_db.archive_table.call_args[0][2]
    assert archive_path == global_dir / "archives" / "rl_outcome_2026-01-01.jsonl.gz"
    assert archive_path.parent.exists()


def test_serve_hitl_uses_global_victor_dir_for_default_db_display(tmp_path):
    """Persistent HITL mode should display the canonical default SQLite path."""
    global_dir = tmp_path / ".victor"
    coro = object()

    with (
        patch.object(serve_cmd, "get_default_hitl_db_path", return_value=global_dir / "hitl.db"),
        patch.object(serve_cmd, "setup_logging"),
        patch.object(serve_cmd, "_run_hitl_server", Mock(return_value=coro)) as mock_run_server,
        patch.object(serve_cmd, "run_sync"),
        patch.object(serve_cmd.console, "print") as mock_print,
    ):
        serve_cmd.serve_hitl(
            host="0.0.0.0",
            port=8080,
            auth_token=None,
            require_auth=False,
            persistent=True,
            db_path=None,
            log_level=None,
        )

    panel = mock_print.call_args[0][0]
    assert str(global_dir / "hitl.db") in str(panel.renderable)
    mock_run_server.assert_called_once_with("0.0.0.0", 8080, False, None, True, None)


def test_ab_list_experiments_uses_global_victor_dir_by_default(tmp_path):
    """A/B experiment listing should read the canonical global experiments database."""
    global_dir = tmp_path / ".victor"
    global_dir.mkdir(exist_ok=True)
    db_path = global_dir / "ab_tests.db"

    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE experiments (
                experiment_id TEXT PRIMARY KEY,
                name TEXT,
                status TEXT,
                created_at REAL,
                started_at REAL,
                completed_at REAL
            )
            """)
        conn.execute(
            """
            INSERT INTO experiments (
                experiment_id, name, status, created_at, started_at, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("exp_123", "Test Experiment", "running", 1714089600.0, None, None),
        )
        conn.commit()

    record_console = Console(record=True, force_terminal=False, width=160)
    with (
        patch(
            "victor.ui.commands.ab_testing.get_default_ab_test_db_path",
            return_value=db_path,
        ),
        patch.object(ab_testing_cmd, "console", record_console),
    ):
        ab_testing_cmd._list_experiments(status_filter=None)

    output = record_console.export_text()
    assert "A/B Experiments" in output
    assert "exp_123" in output
    assert "Test Experiment" in output
