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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import victor.ui.commands.chat as chat_cmd
import victor.ui.commands.graph as graph_cmd


def test_chat_graph_watch_ensures_project_singleton_by_default(tmp_path):
    """Interactive chat should ensure a project-wide graph watch daemon exists."""
    project_root = tmp_path / "repo"
    project_root.mkdir()

    paths = SimpleNamespace(
        project_root=project_root,
        ensure_project_dirs=MagicMock(),
    )
    state = graph_cmd.GraphWatchDaemonState(
        pid_file=project_root / ".victor" / "graph-watch.pid",
        running=True,
        pid=321,
        started=False,
    )
    manifest = {
        "last_refresh": {
            "changed": 2,
            "deleted": 1,
            "unchanged": 7,
            "errors": 0,
            "duration_seconds": 1.25,
        }
    }

    with (
        patch.object(chat_cmd, "get_project_paths", return_value=paths),
        patch(
            "victor.ui.commands.graph.ensure_graph_watch_daemon", return_value=state
        ) as mock_ensure,
        patch("victor.ui.commands.graph._read_graph_watch_manifest", return_value=manifest),
    ):
        messages = chat_cmd._ensure_graph_watch_for_chat(enabled=True)

    paths.ensure_project_dirs.assert_called_once_with()
    mock_ensure.assert_called_once_with(
        project_root,
        enable_ccg=True,
        build_now=False,
        owner="chat",
    )
    assert any("active for this project (pid 321)" in message.lower() for message in messages)
    assert any("changed=2, deleted=1, unchanged=7" in message.lower() for message in messages)
    assert any("1.25s" in message.lower() for message in messages)


def test_chat_graph_watch_reports_new_daemon_start_without_refresh_stats(tmp_path):
    """Interactive chat should report a fresh graph watcher start even before refresh telemetry exists."""
    project_root = tmp_path / "repo"
    project_root.mkdir()

    paths = SimpleNamespace(
        project_root=project_root,
        ensure_project_dirs=MagicMock(),
    )
    state = graph_cmd.GraphWatchDaemonState(
        pid_file=project_root / ".victor" / "graph-watch.pid",
        running=True,
        pid=654,
        started=True,
    )

    with (
        patch.object(chat_cmd, "get_project_paths", return_value=paths),
        patch("victor.ui.commands.graph.ensure_graph_watch_daemon", return_value=state),
        patch("victor.ui.commands.graph._read_graph_watch_manifest", return_value=None),
    ):
        messages = chat_cmd._ensure_graph_watch_for_chat(enabled=True)

    assert any("started for this project (pid 654)" in message.lower() for message in messages)
    assert not any("last refresh" in message.lower() for message in messages)


def test_chat_graph_watch_labels_preserved_refresh_as_previous_for_new_daemon(tmp_path):
    """A new chat-started daemon should not present old manifest telemetry as current."""
    project_root = tmp_path / "repo"
    project_root.mkdir()

    paths = SimpleNamespace(
        project_root=project_root,
        ensure_project_dirs=MagicMock(),
    )
    state = graph_cmd.GraphWatchDaemonState(
        pid_file=project_root / ".victor" / "graph-watch.pid",
        running=True,
        pid=654,
        started=True,
    )
    manifest = {
        "updated_at": 1_777_597_100.0,
        "last_refresh": {
            "changed": 1,
            "deleted": 0,
            "unchanged": 1927,
            "errors": 0,
            "duration_seconds": 147.78,
            "completed_at": 1_777_597_000.0,
        },
    }

    with (
        patch.object(chat_cmd, "get_project_paths", return_value=paths),
        patch("victor.ui.commands.graph.ensure_graph_watch_daemon", return_value=state),
        patch("victor.ui.commands.graph._read_graph_watch_manifest", return_value=manifest),
    ):
        messages = chat_cmd._ensure_graph_watch_for_chat(enabled=True)

    rendered = "\n".join(messages).lower()
    assert "started for this project (pid 654)" in rendered
    assert "previous refresh:" in rendered
    assert "last refresh:" not in rendered


def test_chat_graph_watch_handle_marks_chat_started_daemon(tmp_path):
    """Interactive chat should retain ownership only when it started the daemon."""
    project_root = tmp_path / "repo"
    project_root.mkdir()

    paths = SimpleNamespace(
        project_root=project_root,
        ensure_project_dirs=MagicMock(),
    )
    state = graph_cmd.GraphWatchDaemonState(
        pid_file=project_root / ".victor" / "graph-watch.pid",
        running=True,
        pid=654,
        started=True,
    )

    with (
        patch.object(chat_cmd, "get_project_paths", return_value=paths),
        patch("victor.ui.commands.graph.ensure_graph_watch_daemon", return_value=state),
        patch("victor.ui.commands.graph._read_graph_watch_manifest", return_value=None),
    ):
        handle = chat_cmd._ensure_graph_watch_handle_for_chat(enabled=True)

    assert handle.started_by_chat is True
    assert handle.project_root == project_root
    assert any("started for this project" in message.lower() for message in handle.messages)


def test_chat_graph_watch_cleanup_stops_only_chat_started_daemon(tmp_path):
    """Chat shutdown should not stop an already-running explicit graph watcher."""
    project_root = tmp_path / "repo"
    project_root.mkdir()

    started_handle = chat_cmd.ChatGraphWatchHandle(
        messages=[],
        project_root=project_root,
        started_by_chat=True,
    )
    attached_handle = chat_cmd.ChatGraphWatchHandle(
        messages=[],
        project_root=project_root,
        started_by_chat=False,
    )

    with patch("victor.ui.commands.graph.stop_graph_watch_daemon") as mock_stop:
        chat_cmd._cleanup_graph_watch_for_chat(attached_handle)
        mock_stop.assert_not_called()

        chat_cmd._cleanup_graph_watch_for_chat(started_handle)

    mock_stop.assert_called_once_with(project_root)


def test_chat_graph_watch_can_be_disabled(tmp_path):
    """Interactive chat should skip graph watch setup when explicitly disabled."""
    project_root = tmp_path / "repo"
    project_root.mkdir()

    paths = SimpleNamespace(
        project_root=project_root,
        ensure_project_dirs=MagicMock(),
    )

    with (
        patch.object(chat_cmd, "get_project_paths", return_value=paths),
        patch("victor.ui.commands.graph.ensure_graph_watch_daemon") as mock_ensure,
    ):
        messages = chat_cmd._ensure_graph_watch_for_chat(enabled=False)

    paths.ensure_project_dirs.assert_not_called()
    mock_ensure.assert_not_called()
    assert messages == []
