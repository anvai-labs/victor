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

"""Tests for the /import and /export slash commands (real store, tmp DB)."""

import io
import json
from pathlib import Path
from unittest.mock import MagicMock

from rich.console import Console

from victor.ui.slash.protocol import CommandContext

FIXTURES = Path(__file__).parents[3] / "tests" / "fixtures" / "interop"


def fixture(name: str) -> Path:
    path = FIXTURES / name
    assert path.exists(), f"missing fixture {name}"
    return path


def make_ctx(args, agent=None) -> tuple[CommandContext, io.StringIO]:
    stdout = io.StringIO()
    console = Console(file=stdout, force_terminal=False, width=200)
    return (
        CommandContext(console=console, settings=MagicMock(), agent=agent, args=args),
        stdout,
    )


class TestImportCommand:
    def test_metadata(self):
        from victor.ui.slash.commands.import_export import ImportCommand

        meta = ImportCommand().metadata
        assert meta.name == "import"
        assert meta.requires_agent is False

    def test_usage_without_args(self):
        from victor.ui.slash.commands.import_export import ImportCommand

        ctx, stdout = make_ctx([])
        ImportCommand().execute(ctx)
        assert "Usage" in stdout.getvalue()

    def test_dry_run_leaves_db_untouched(self, tmp_path, monkeypatch):
        from victor.ui.slash.commands.import_export import ImportCommand

        db = tmp_path / "project.db"
        monkeypatch.setenv("VICTOR_CONVERSATION_DB", str(db))

        ctx, stdout = make_ctx([str(fixture("claude_minimal.jsonl")), "--dry-run"])
        ImportCommand().execute(ctx)
        out = stdout.getvalue()
        assert "dry run" in out
        assert "4 user" in out or "user" in out
        assert not db.exists(), "dry run must not create the store"

    def test_import_persists_messages_across_store_instances(self, tmp_path, monkeypatch):
        """Command-level P0: a fresh store reloads the full transcript."""
        from victor.agent.conversation.store import ConversationStore
        from victor.ui.slash.commands.import_export import ImportCommand

        db = tmp_path / "project.db"
        monkeypatch.setenv("VICTOR_CONVERSATION_DB", str(db))

        ctx, stdout = make_ctx([str(fixture("claude_tools_thinking.jsonl"))])
        ImportCommand().execute(ctx)
        out = stdout.getvalue()
        assert "Session Imported" in out
        assert "Tool calls preserved: 1" in out

        fresh = ConversationStore(db_path=db)
        session_id = fresh.list_session_summaries(limit=1)[0].session_id
        loaded = fresh.load_session(session_id)
        roles = [m["role"] for m in loaded["conversation"]["messages"]]
        assert roles == [
            "user",
            "assistant",
            "tool_call",
            "tool",
            "assistant",
            "user",
            "user",
        ]
        call = next(m for m in loaded["conversation"]["messages"] if m["role"] == "tool_call")
        assert call["tool_name"] == "bash"

    def test_reimport_same_file_upserts_not_duplicates(self, tmp_path, monkeypatch):
        from victor.agent.conversation.store import ConversationStore
        from victor.ui.slash.commands.import_export import ImportCommand

        db = tmp_path / "project.db"
        monkeypatch.setenv("VICTOR_CONVERSATION_DB", str(db))

        for _ in range(2):
            ctx, _stdout = make_ctx([str(fixture("claude_minimal.jsonl"))])
            ImportCommand().execute(ctx)

        fresh = ConversationStore(db_path=db)
        summaries = fresh.list_session_summaries(limit=10)
        assert len(summaries) == 1
        assert summaries[0].message_count == 4  # replace, not append

    def test_unknown_format_flag_rejected(self, tmp_path):
        from victor.ui.slash.commands.import_export import ImportCommand

        ctx, stdout = make_ctx([str(fixture("claude_minimal.jsonl")), "--format", "bogus"])
        ImportCommand().execute(ctx)
        assert "Unknown --format" in stdout.getvalue()

    def test_missing_file_reports_cleanly(self, tmp_path):
        from victor.ui.slash.commands.import_export import ImportCommand

        ctx, stdout = make_ctx(["/nope/missing.jsonl"])
        ImportCommand().execute(ctx)
        assert "File not found" in stdout.getvalue()


class TestExportCommand:
    def _seed(self, db: Path, session_id="exp_001") -> None:
        from victor.agent.conversation.store import ConversationStore

        store = ConversationStore(db_path=db)
        store.save_session(
            conversation={
                "messages": [
                    {"role": "user", "content": "export me"},
                    {
                        "role": "assistant",
                        "content": "doing it",
                        "tool_calls": [
                            {"id": "c1", "name": "bash", "arguments": '{"command":"ls"}'}
                        ],
                    },
                    {"role": "tool", "content": "files", "tool_call_id": "c1"},
                ]
            },
            model="m",
            provider="p",
            session_id=session_id,
            title="export test",
        )

    def test_requires_format(self, tmp_path, monkeypatch):
        from victor.ui.slash.commands.import_export import ExportCommand

        db = tmp_path / "project.db"
        self._seed(db)
        monkeypatch.setenv("VICTOR_CONVERSATION_DB", str(db))

        ctx, stdout = make_ctx(["exp_001"])
        ExportCommand().execute(ctx)
        assert "--format required" in stdout.getvalue()

    def test_export_claude_file(self, tmp_path, monkeypatch):
        from victor.ui.slash.commands.import_export import ExportCommand

        db = tmp_path / "project.db"
        self._seed(db)
        monkeypatch.setenv("VICTOR_CONVERSATION_DB", str(db))
        monkeypatch.chdir(tmp_path)

        ctx, stdout = make_ctx(["exp_001", "--format", "claude"])
        ExportCommand().execute(ctx)
        out = stdout.getvalue()
        assert "Session Exported" in out
        emitted = tmp_path / "victor-session-exp_001.claude.jsonl"
        assert emitted.exists()
        entries = [json.loads(line) for line in emitted.read_text().splitlines()]
        assert entries[0]["message"]["content"] == "export me"

    def test_export_unknown_session(self, tmp_path, monkeypatch):
        from victor.ui.slash.commands.import_export import ExportCommand

        monkeypatch.setenv("VICTOR_CONVERSATION_DB", str(tmp_path / "project.db"))
        ctx, stdout = make_ctx(["nope", "--format", "victor"])
        ExportCommand().execute(ctx)
        assert "not found" in stdout.getvalue()

    def test_install_requires_target_format(self, tmp_path, monkeypatch):
        from victor.ui.slash.commands.import_export import ExportCommand

        db = tmp_path / "project.db"
        self._seed(db)
        monkeypatch.setenv("VICTOR_CONVERSATION_DB", str(db))
        ctx, stdout = make_ctx(["exp_001", "--format", "victor", "--install"])
        ExportCommand().execute(ctx)
        assert "--install requires" in stdout.getvalue()

    def test_install_claude_writes_store(self, tmp_path, monkeypatch):
        from victor.ui.slash.commands.import_export import ExportCommand

        db = tmp_path / "project.db"
        self._seed(db)
        monkeypatch.setenv("VICTOR_CONVERSATION_DB", str(db))

        home = tmp_path / "home"
        (home / ".claude" / "projects").mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)

        ctx, stdout = make_ctx(["exp_001", "--format", "claude", "--install"])
        ExportCommand().execute(ctx)
        out = stdout.getvalue()
        assert "Installed into claude store" in out
        assert "claude --resume" in out
        files = list((home / ".claude" / "projects").rglob("*.jsonl"))
        assert len(files) == 1

    def test_install_no_overwrite(self, tmp_path, monkeypatch):
        """Two installs produce two distinct files (uuid collision guard)."""
        from victor.ui.slash.commands.import_export import ExportCommand

        db = tmp_path / "project.db"
        self._seed(db)
        monkeypatch.setenv("VICTOR_CONVERSATION_DB", str(db))

        home = tmp_path / "home"
        (home / ".codex" / "sessions").mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)

        for _ in range(2):
            ctx, _stdout = make_ctx(["exp_001", "--format", "codex", "--install"])
            ExportCommand().execute(ctx)
        files = list((home / ".codex" / "sessions").rglob("rollout-*.jsonl"))
        assert len(files) == 2


class TestAddressImportCommand:
    """`/import claude:<uuid>` and `/import --list` (fake stores)."""

    def test_import_by_address(self, tmp_path, monkeypatch):
        import json as _json

        from victor.ui.slash.commands.import_export import ImportCommand

        home = tmp_path / "home"
        slug_dir = (
            home / ".claude" / "projects" / (str(tmp_path).replace("/", "-").replace(".", "-"))
        )
        slug_dir.mkdir(parents=True)
        sid = "aaaaaaaa-bbbb-4ccc-8ddd-0000000000aa"
        entry = {
            "type": "user",
            "message": {"role": "user", "content": "addressed hello"},
            "uuid": "u1",
            "sessionId": sid,
        }
        (slug_dir / f"{sid}.jsonl").write_text(_json.dumps(entry) + "\n")
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("VICTOR_CONVERSATION_DB", str(tmp_path / "project.db"))

        ctx, stdout = make_ctx([f"claude:{sid}"])
        ImportCommand().execute(ctx)
        assert "Session Imported" in stdout.getvalue()

        from victor.agent.conversation.store import ConversationStore

        fresh = ConversationStore(db_path=tmp_path / "project.db")
        loaded = fresh.load_session(fresh.list_session_summaries(limit=1)[0].session_id)
        assert loaded["conversation"]["messages"][0]["content"] == "addressed hello"

    def test_bad_address_rejected(self, tmp_path, monkeypatch):
        from victor.ui.slash.commands.import_export import ImportCommand

        monkeypatch.setenv("VICTOR_CONVERSATION_DB", str(tmp_path / "project.db"))
        ctx, stdout = make_ctx(["claude:../escape"])
        ImportCommand().execute(ctx)
        assert "Invalid session address" in stdout.getvalue()

    def test_list_shows_store_sessions(self, tmp_path, monkeypatch):
        import json as _json

        from victor.ui.slash.commands.import_export import ImportCommand

        home = tmp_path / "home"
        slug_dir = home / ".claude" / "projects" / "-tmp-demo"
        slug_dir.mkdir(parents=True)
        sid = "cccccccc-dddd-4eee-8fff-0000000000bb"
        entry = {
            "type": "user",
            "message": {"role": "user", "content": "listed session"},
            "uuid": "u1",
            "sessionId": sid,
        }
        (slug_dir / f"{sid}.jsonl").write_text(_json.dumps(entry) + "\n")
        monkeypatch.setenv("HOME", str(home))

        ctx, stdout = make_ctx(["--list"])
        ImportCommand().execute(ctx)
        out = stdout.getvalue()
        assert f"claude:{sid}" in out
        assert "listed session" in out
