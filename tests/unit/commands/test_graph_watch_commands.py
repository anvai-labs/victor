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

import os
import time
import json
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

import victor.ui.commands.graph as graph_cmd


def test_graph_watch_lock_file_uses_project_victor_dir(tmp_path):
    """Graph watch startup lock files should live under the project .victor dir."""
    project_root = tmp_path / "repo"
    project_root.mkdir()

    with patch(
        "victor.ui.commands.graph.get_project_paths",
        return_value=type("Paths", (), {"project_victor_dir": project_root / ".victor"})(),
    ):
        lock_file = graph_cmd._default_graph_watch_lock_file(project_root)

    assert lock_file.parent == project_root / ".victor"
    assert lock_file.name == "graph-watch.lock"


def test_graph_watch_manifest_file_uses_project_victor_dir(tmp_path):
    """Graph watch manifest files should live under the project .victor dir."""
    project_root = tmp_path / "repo"
    project_root.mkdir()

    with patch(
        "victor.ui.commands.graph.get_project_paths",
        return_value=type("Paths", (), {"project_victor_dir": project_root / ".victor"})(),
    ):
        manifest_file = graph_cmd._default_graph_watch_manifest_file(project_root)

    assert manifest_file.parent == project_root / ".victor"
    assert manifest_file.name == "graph-watch.json"


def test_graph_watch_startup_lock_recovers_stale_lock_file(tmp_path):
    """Acquiring the graph watch startup lock should recover stale lock files."""
    project_root = tmp_path / "repo"
    project_root.mkdir()
    lock_file = tmp_path / "graph-watch.lock"
    lock_file.write_text("stale", encoding="utf-8")
    stale_time = time.time() - 60
    os.utime(lock_file, (stale_time, stale_time))

    with graph_cmd._acquire_graph_watch_startup_lock(
        project_root,
        lock_file=lock_file,
        timeout_seconds=0.1,
        poll_interval=0.0,
        stale_after_seconds=0.01,
    ):
        assert lock_file.exists()

    assert not lock_file.exists()


def test_graph_watch_start_daemon_is_idempotent_when_already_running(tmp_path):
    """Starting an already-running graph watcher should be a no-op."""
    project_root = tmp_path / "repo"
    project_root.mkdir()
    pid_file = tmp_path / "graph-watch.pid"
    pid_file.write_text("123", encoding="utf-8")
    record_console = Console(record=True, force_terminal=False, width=160)

    with (
        patch.object(graph_cmd, "console", record_console),
        patch.object(graph_cmd.os, "kill") as mock_kill,
        patch.object(graph_cmd, "_spawn_watch_daemon") as mock_spawn,
    ):
        graph_cmd.graph_watch_start(
            path=str(project_root),
            enable_ccg=True,
            daemon=True,
            pid_file=pid_file,
            poll_interval=1.0,
            debounce_seconds=0.3,
            min_refresh_interval=30.0,
            build_now=False,
        )

    mock_kill.assert_called_once_with(123, 0)
    mock_spawn.assert_not_called()
    assert "already running" in record_console.export_text().lower()


def test_graph_watch_start_daemon_replaces_stale_pid_file(tmp_path):
    """Starting the watcher should recover from a stale PID file."""
    project_root = tmp_path / "repo"
    project_root.mkdir()
    pid_file = tmp_path / "graph-watch.pid"
    pid_file.write_text("123", encoding="utf-8")
    record_console = Console(record=True, force_terminal=False, width=160)

    with (
        patch.object(graph_cmd, "console", record_console),
        patch.object(graph_cmd.os, "kill", side_effect=ProcessLookupError),
        patch.object(graph_cmd, "_spawn_watch_daemon", return_value=456) as mock_spawn,
    ):
        graph_cmd.graph_watch_start(
            path=str(project_root),
            enable_ccg=True,
            daemon=True,
            pid_file=pid_file,
            poll_interval=1.0,
            debounce_seconds=0.3,
            min_refresh_interval=30.0,
            build_now=False,
        )

    mock_spawn.assert_called_once_with(
        pid_file,
        str(project_root.resolve()),
        True,
        1.0,
        0.3,
        30.0,
        False,
    )
    assert not pid_file.exists()
    output = record_console.export_text().lower()
    assert "recovered stale pid file" in output
    assert "started (pid 456)" in output


def test_spawn_watch_daemon_uses_fresh_interpreter_instead_of_fork(tmp_path):
    """The daemon launcher should avoid fork-unsafe native imports such as LanceDB."""
    project_root = tmp_path / "repo"
    project_root.mkdir()
    pid_file = tmp_path / "graph-watch.pid"

    class Process:
        pid = 789

    with patch.object(graph_cmd.subprocess, "Popen", return_value=Process()) as mock_popen:
        pid = graph_cmd._spawn_watch_daemon(
            pid_file,
            str(project_root.resolve()),
            True,
            1.0,
            0.3,
            30.0,
            False,
        )

    assert pid == 789
    assert pid_file.read_text(encoding="utf-8") == "789"
    command = mock_popen.call_args.args[0]
    assert command[:6] == [
        graph_cmd.sys.executable,
        "-m",
        "victor.ui.cli",
        "graph",
        "watch",
        "start",
    ]
    assert "--foreground-daemon" in command
    assert "--no-build-now" in command
    kwargs = mock_popen.call_args.kwargs
    assert kwargs["cwd"] == str(project_root.resolve())
    assert kwargs["stdin"] == graph_cmd.subprocess.DEVNULL
    assert kwargs["stderr"] == graph_cmd.subprocess.STDOUT
    assert kwargs["start_new_session"] is True
    assert kwargs["close_fds"] is True


def test_graph_watch_stop_removes_stale_pid_file(tmp_path):
    """Stopping the watcher should clean up stale PID files."""
    project_root = tmp_path / "repo"
    project_root.mkdir()
    pid_file = tmp_path / "graph-watch.pid"
    pid_file.write_text("123", encoding="utf-8")
    record_console = Console(record=True, force_terminal=False, width=160)

    with (
        patch.object(graph_cmd, "console", record_console),
        patch.object(graph_cmd.os, "kill", side_effect=ProcessLookupError),
    ):
        graph_cmd.graph_watch_stop(path=str(project_root), pid_file=pid_file)

    assert not pid_file.exists()
    assert "stale pid file removed" in record_console.export_text().lower()


def test_graph_watch_status_reports_stale_pid_file(tmp_path):
    """Status should distinguish a stale PID file from a healthy daemon."""
    project_root = tmp_path / "repo"
    project_root.mkdir()
    pid_file = tmp_path / "graph-watch.pid"
    pid_file.write_text("123", encoding="utf-8")
    record_console = Console(record=True, force_terminal=False, width=160)

    with (
        patch.object(graph_cmd, "console", record_console),
        patch.object(graph_cmd.os, "kill", side_effect=ProcessLookupError),
    ):
        graph_cmd.graph_watch_status(path=str(project_root), pid_file=pid_file)

    output = record_console.export_text().lower()
    assert "stale pid file" in output


def test_ensure_graph_watch_daemon_writes_project_manifest(tmp_path):
    """Ensuring the graph watcher should persist a project-scoped manifest."""
    project_root = tmp_path / "repo"
    project_root.mkdir()
    pid_file = tmp_path / "graph-watch.pid"
    manifest_file = tmp_path / "graph-watch.json"

    with (
        patch.object(graph_cmd, "_resolve_graph_watch_pid_file", return_value=pid_file),
        patch.object(graph_cmd, "_default_graph_watch_manifest_file", return_value=manifest_file),
        patch.object(
            graph_cmd,
            "_default_graph_watch_lock_file",
            return_value=tmp_path / "graph-watch.lock",
        ),
        patch.object(graph_cmd, "_spawn_watch_daemon", return_value=456),
    ):
        state = graph_cmd.ensure_graph_watch_daemon(
            project_root,
            enable_ccg=True,
            build_now=True,
            poll_interval=1.0,
            debounce_seconds=0.3,
            min_refresh_interval=45.0,
        )

    assert state.started is True
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["project_root"] == str(project_root.resolve())
    assert manifest["pid"] == 456
    assert manifest["running"] is True
    assert manifest["enable_ccg"] is True
    assert manifest["build_now"] is True
    assert manifest["min_refresh_interval"] == 45.0
    assert manifest["owner"] == "explicit"


def test_stop_graph_watch_daemon_updates_manifest_to_not_running(tmp_path):
    """Stopping the graph watcher should persist the stopped state to the manifest."""
    project_root = tmp_path / "repo"
    project_root.mkdir()
    pid_file = tmp_path / "graph-watch.pid"
    manifest_file = tmp_path / "graph-watch.json"
    pid_file.write_text("123", encoding="utf-8")

    with (
        patch.object(graph_cmd, "_resolve_graph_watch_pid_file", return_value=pid_file),
        patch.object(graph_cmd, "_default_graph_watch_manifest_file", return_value=manifest_file),
        patch.object(
            graph_cmd,
            "_default_graph_watch_lock_file",
            return_value=tmp_path / "graph-watch.lock",
        ),
        patch.object(graph_cmd.os, "kill"),
    ):
        state = graph_cmd.stop_graph_watch_daemon(project_root)

    assert state.stopped is True
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["running"] is False
    assert manifest["stopped"] is True
    assert manifest["pid"] == 123


def test_graph_watch_status_reports_last_refresh_details(tmp_path):
    """Status should surface the last refresh telemetry from the manifest."""
    project_root = tmp_path / "repo"
    project_root.mkdir()
    pid_file = tmp_path / "graph-watch.pid"
    manifest_file = tmp_path / "graph-watch.json"
    manifest_file.write_text(
        json.dumps(
            {
                "last_refresh": {
                    "changed": 2,
                    "deleted": 1,
                    "unchanged": 7,
                    "errors": 0,
                    "duration_seconds": 1.25,
                    "completed_at": 1_777_597_000.0,
                }
            }
        ),
        encoding="utf-8",
    )
    record_console = Console(record=True, force_terminal=False, width=160)

    with (
        patch.object(graph_cmd, "console", record_console),
        patch.object(graph_cmd, "_resolve_graph_watch_pid_file", return_value=pid_file),
        patch.object(graph_cmd, "_default_graph_watch_manifest_file", return_value=manifest_file),
        patch.object(
            graph_cmd,
            "_default_graph_watch_lock_file",
            return_value=tmp_path / "graph-watch.lock",
        ),
    ):
        graph_cmd.graph_watch_status(path=str(project_root), pid_file=pid_file)

    output = record_console.export_text().lower()
    assert "last refresh" in output
    assert "changed=2, deleted=1, unchanged=7" in output
    assert "1.25s" in output


def test_graph_watch_status_reports_last_refresh_error_details(tmp_path):
    """Status should surface the latest background refresh error from the manifest."""
    project_root = tmp_path / "repo"
    project_root.mkdir()
    pid_file = tmp_path / "graph-watch.pid"
    manifest_file = tmp_path / "graph-watch.json"
    manifest_file.write_text(
        json.dumps(
            {
                "running": True,
                "last_refresh": {
                    "changed": 0,
                    "deleted": 0,
                    "unchanged": 0,
                    "errors": 1,
                    "duration_seconds": 0.0,
                    "completed_at": 1_777_597_000.0,
                    "last_error": "Too many open files",
                    "error_type": "OSError",
                    "error_category": "resource_exhaustion",
                    "recoverable": True,
                    "retry_delay_seconds": 15.0,
                    "operator_guidance": "Reduce concurrent watcher/index activity.",
                },
            }
        ),
        encoding="utf-8",
    )
    record_console = Console(record=True, force_terminal=False, width=160)

    with (
        patch.object(graph_cmd, "console", record_console),
        patch.object(graph_cmd, "_resolve_graph_watch_pid_file", return_value=pid_file),
        patch.object(graph_cmd, "_default_graph_watch_manifest_file", return_value=manifest_file),
        patch.object(
            graph_cmd,
            "_default_graph_watch_lock_file",
            return_value=tmp_path / "graph-watch.lock",
        ),
    ):
        graph_cmd.graph_watch_status(path=str(project_root), pid_file=pid_file)

    output = record_console.export_text().lower()
    assert "refresh errors" in output
    assert "1" in output
    assert "last error" in output
    assert "too many open files" in output
    assert "oserror" in output
    assert "error category" in output
    assert "resource_exhaustion" in output
    assert "recoverable" in output
    assert "retry backoff" in output
    assert "15.00s" in output
    assert "operator guidance" in output


def test_summarize_graph_watch_health_reports_active_refresh_counts() -> None:
    """Compact health summaries should include current refresh counts for live surfaces."""
    summary, state = graph_cmd.summarize_graph_watch_health(
        {
            "running": True,
            "last_refresh": {
                "changed": 2,
                "deleted": 1,
                "unchanged": 7,
                "errors": 0,
            },
        }
    )

    assert summary == "Graph: ok c2 d1 u7"
    assert state == "active"


def test_summarize_graph_watch_health_reports_retry_backoff() -> None:
    """Compact health summaries should surface retry backoff after recoverable failures."""
    summary, state = graph_cmd.summarize_graph_watch_health(
        {
            "running": True,
            "last_refresh": {
                "changed": 0,
                "deleted": 0,
                "unchanged": 0,
                "errors": 1,
                "retry_delay_seconds": 12.6,
            },
        }
    )

    assert summary == "Graph: retry 13s"
    assert state == "warning"


def test_summarize_graph_watch_health_reports_stale_or_missing_state() -> None:
    """Compact health summaries should distinguish stale and disabled graph watch states."""
    assert graph_cmd.summarize_graph_watch_health(None) == ("Graph: off", "inactive")
    assert graph_cmd.summarize_graph_watch_health({"running": False, "stale_pid_file": True}) == (
        "Graph: stale",
        "warning",
    )
