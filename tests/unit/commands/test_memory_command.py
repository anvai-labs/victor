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

"""FOU-A2d unit suite: the /memory slash command (FEP-0048).

Covers argument parsing (list/set/forget/confirm/discard, --project flag,
set-default = user scope), refusal output (secret-shaped content is refused
and the matched secret value never appears in the rendered output), the
confirm/discard path against the shared consolidator handle, and the alias
takeover (``/memory`` was a ``/context`` alias before A2d).

Databases are fixture-injected onto ``tmp_path`` per the A2a/A2b memory-suite
conventions; the shared handle is re-pointed at the tmp store. No network.
"""

import io
from pathlib import Path

import pytest
from rich.console import Console
from unittest.mock import MagicMock

from victor.agent.services.memory_consolidation import MemoryConsolidator
from victor.storage.memory.preference_episodic import (
    MemoryScope,
    PreferenceEpisodicStore,
)
from victor.ui.slash.protocol import CommandContext

_LEDGER = [
    {"role": "user", "content": "we renamed the service to sandhi-gateway"},
    {"role": "assistant", "content": "noted"},
    {"role": "user", "content": "this repo prefers pytest over unittest"},
]

_GITHUB_PAT = "ghp_" + "a" * 36  # canonical GITHUB_PAT pattern, matches on both scanner paths


@pytest.fixture
def wired(tmp_path: Path):
    """Tmp-path databases wired into the shared memory handle."""
    from victor.core.database import (
        get_database,
        get_project_database,
        reset_database,
        reset_project_database,
    )
    from victor.agent.services.memory_consolidation import (
        set_shared_memory_consolidator,
        set_shared_memory_store,
    )

    reset_database()
    reset_project_database(tmp_path)
    store = PreferenceEpisodicStore(
        global_db=get_database(tmp_path / "victor.db"),
        project_db=get_project_database(tmp_path),
    )
    consolidator = MemoryConsolidator(store)
    set_shared_memory_store(store)
    set_shared_memory_consolidator(consolidator)
    yield store, consolidator
    from victor.agent.services.memory_consolidation import reset_shared_memory

    reset_shared_memory()
    reset_database()
    reset_project_database(tmp_path)


def make_ctx(args, agent=None) -> tuple[CommandContext, io.StringIO]:
    stdout = io.StringIO()
    console = Console(file=stdout, force_terminal=False, width=200)
    return (
        CommandContext(console=console, settings=MagicMock(), agent=agent, args=args),
        stdout,
    )


def run_command(args, wired):
    from victor.ui.slash.commands.memory import MemoryCommand

    ctx, stdout = make_ctx(args)
    MemoryCommand().execute(ctx)
    return stdout.getvalue()


class TestMemoryCommandRegistration:
    def test_metadata(self):
        from victor.ui.slash.commands.memory import MemoryCommand

        meta = MemoryCommand().metadata
        assert meta.name == "memory"
        assert meta.requires_agent is False

    def test_memory_resolves_to_the_memory_command(self):
        from victor.ui.slash.registry import get_command_registry

        command = get_command_registry().get("memory")
        assert command is not None
        assert type(command).__name__ == "MemoryCommand"

    def test_context_command_no_longer_claims_the_memory_alias(self):
        """A2d takes /memory over from /context — the alias must be released,
        not silently shadowed (dead alias keeps help/completions lying)."""
        from victor.ui.slash.commands.tools import ContextCommand

        assert "memory" not in ContextCommand().metadata.aliases


class TestMemoryList:
    def test_list_empty_shows_guidance(self, wired):
        out = run_command(["list"], wired)
        assert "No memories" in out
        assert "/memory set" in out

    def test_bare_invocation_lists(self, wired):
        out = run_command([], wired)
        assert "No memories" in out

    def test_list_shows_scope_source_confidence(self, wired):
        store, _consolidator = wired
        store.set_preference("always use uv, never pip", scope=MemoryScope.USER)
        store.set_preference("this repo uses pytest", scope=MemoryScope.PROJECT, source="tool")

        out = run_command(["list"], wired)
        assert "always use uv, never pip" in out
        assert "this repo uses pytest" in out
        assert "user" in out and "project" in out
        assert "uv" in out
        assert "tool" in out

    def test_list_shows_staged_proposals(self, wired):
        store, consolidator = wired
        consolidator.run(_LEDGER)
        assert consolidator.staged_proposals()

        out = run_command(["list"], wired)
        assert "sandhi-gateway" in out
        assert "confirm" in out.lower()


class TestMemorySet:
    def test_set_defaults_to_user_scope(self, wired):
        store, _consolidator = wired
        out = run_command(["set", "always", "use", "uv"], wired)

        assert "Stored" in out
        prefs = store.get_preferences(MemoryScope.USER)
        assert [p.content for p in prefs] == ["always use uv"]
        assert store.get_preferences(MemoryScope.PROJECT) == []

    def test_set_project_flag(self, wired):
        store, _consolidator = wired
        run_command(["set", "uses", "pytest", "--project"], wired)

        assert store.get_preferences(MemoryScope.USER) == []
        assert [p.content for p in store.get_preferences(MemoryScope.PROJECT)] == ["uses pytest"]

    def test_set_without_content_shows_usage(self, wired):
        out = run_command(["set"], wired)
        assert "Usage" in out
        store, _consolidator = wired
        assert store.get_preferences(MemoryScope.USER) == []

    def test_set_secret_shaped_content_is_refused_and_never_echoed(self, wired):
        store, _consolidator = wired
        out = run_command(["set", f"the token is {_GITHUB_PAT}"], wired)

        assert "refus" in out.lower()
        assert _GITHUB_PAT not in out, "the matched secret value must never be echoed"
        assert store.get_preferences(MemoryScope.USER) == []
        assert store.get_preferences(MemoryScope.PROJECT) == []

    def test_set_oversized_content_is_refused(self, wired):
        store, _consolidator = wired
        out = run_command(["set", "x" * 5000], wired)

        assert "refus" in out.lower()
        assert store.get_preferences(MemoryScope.USER) == []


class TestMemoryForget:
    def test_forget_by_id(self, wired):
        store, _consolidator = wired
        pref = store.set_preference("always use uv", scope=MemoryScope.USER)

        out = run_command(["forget", pref.id], wired)
        assert "1" in out
        assert store.get_preferences(MemoryScope.USER) == []

    def test_forget_by_content_across_scopes(self, wired):
        store, _consolidator = wired
        store.set_preference("always use uv", scope=MemoryScope.USER)
        store.set_preference("always use uv in ci", scope=MemoryScope.PROJECT)

        out = run_command(["forget", "use uv"], wired)
        assert "2" in out
        assert store.get_preferences(MemoryScope.USER) == []
        assert store.get_preferences(MemoryScope.PROJECT) == []

    def test_forget_no_match_reports_zero(self, wired):
        out = run_command(["forget", "pref_nope"], wired)
        assert "No matching" in out

    def test_forget_without_target_shows_usage(self, wired):
        out = run_command(["forget"], wired)
        assert "Usage" in out


class TestMemoryConfirmDiscard:
    def test_confirm_commits_staged_proposals(self, wired):
        store, consolidator = wired
        consolidator.run(_LEDGER)
        staged = len(consolidator.staged_proposals())
        assert staged == 2

        out = run_command(["confirm"], wired)
        assert "Stored" in out
        assert str(staged) in out
        assert consolidator.staged_proposals() == []
        summaries = [e.summary for e in store.recent_episodes(limit=10)]
        assert "we renamed the service to sandhi-gateway" in summaries
        assert "this repo prefers pytest over unittest" in summaries

    def test_confirm_with_nothing_staged(self, wired):
        out = run_command(["confirm"], wired)
        assert "No staged" in out

    def test_discard_drops_staged_proposals_without_writing(self, wired):
        store, consolidator = wired
        consolidator.run(_LEDGER)

        out = run_command(["discard"], wired)
        assert "2" in out
        assert consolidator.staged_proposals() == []
        assert store.recent_episodes(limit=10) == []

    def test_discard_with_nothing_staged(self, wired):
        out = run_command(["discard"], wired)
        assert "No staged" in out


class TestMemoryUnknownAction:
    def test_unknown_action_shows_usage(self, wired):
        out = run_command(["explode"], wired)
        assert "Unknown" in out
        assert "Usage" in out
