# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""The /undo, /redo, /filehistory slash commands drive the real change tracker.

Previously these read a never-assigned ``ctx.agent._file_tracker`` and always
reported "File change tracking not available". They now call
``get_change_tracker()`` and speak its real tuple API.
"""

import io
from unittest.mock import MagicMock, patch

from rich.console import Console

from victor.ui.slash.commands.navigation import HistoryCommand, RedoCommand, UndoCommand
from victor.ui.slash.protocol import CommandContext


def _ctx():
    buf = io.StringIO()
    console = Console(file=buf, width=200, color_system=None)
    ctx = CommandContext(console=console, settings=MagicMock(), agent=MagicMock(), args=[])
    return ctx, buf


def _patch_tracker(tracker):
    return patch("victor.agent.change_tracker.get_change_tracker", return_value=tracker)


class TestUndoCommand:
    def test_undo_success_prints_message_and_files(self):
        tracker = MagicMock()
        tracker.undo.return_value = (
            True,
            "Undid 'edit' (2 files) from 10:00:00",
            ["/a.py", "/b.py"],
        )
        with _patch_tracker(tracker):
            ctx, buf = _ctx()
            UndoCommand().execute(ctx)
        out = buf.getvalue()
        assert "Undid 'edit'" in out
        assert "/a.py" in out and "/b.py" in out
        tracker.undo.assert_called_once()

    def test_undo_nothing_to_undo(self):
        tracker = MagicMock()
        tracker.undo.return_value = (False, "Nothing to undo", [])
        with _patch_tracker(tracker):
            ctx, buf = _ctx()
            UndoCommand().execute(ctx)
        assert "Nothing to undo" in buf.getvalue()


class TestRedoCommand:
    def test_redo_success(self):
        tracker = MagicMock()
        tracker.redo.return_value = (True, "Redid 'edit' (1 file)", ["/a.py"])
        with _patch_tracker(tracker):
            ctx, buf = _ctx()
            RedoCommand().execute(ctx)
        out = buf.getvalue()
        assert "Redid 'edit'" in out
        assert "/a.py" in out
        tracker.redo.assert_called_once()


class TestHistoryCommand:
    def test_renders_real_history_keys(self):
        tracker = MagicMock()
        tracker.get_history.return_value = [
            {
                "id": "g1",
                "timestamp": "2026-07-21T10:00:00",
                "tool_name": "edit",
                "description": "Edit 2 files",
                "file_count": 2,
                "files": ["/a.py", "/b.py"],
                "undone": False,
            }
        ]
        with _patch_tracker(tracker):
            ctx, buf = _ctx()
            HistoryCommand().execute(ctx)
        out = buf.getvalue()
        assert "edit" in out
        assert "Edit 2 files" in out
        assert "applied" in out

    def test_empty_history(self):
        tracker = MagicMock()
        tracker.get_history.return_value = []
        with _patch_tracker(tracker):
            ctx, buf = _ctx()
            HistoryCommand().execute(ctx)
        assert "No file change history" in buf.getvalue()
