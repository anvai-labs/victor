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

"""Evaluation conversations must outlive the workspace they were produced in.

``ConversationStore`` derives its path from the current working directory, and
the benchmark harness runs every task inside a ``tempfile.mkdtemp()`` workspace
that the agent chdirs into. Eval conversations were therefore written to a
directory deleted moments later — or, when project paths had already been
resolved and cached, into whichever repo the run started from, mixing benchmark
sessions into a developer's own history.
"""

import os
from pathlib import Path

from victor.agent.conversation.store import CONVERSATION_DB_ENV, ConversationStore
from victor.evaluation.harness import durable_evaluation_conversations


class TestConversationDbOverride:
    def test_explicit_argument_beats_everything(self, monkeypatch, tmp_path):
        monkeypatch.setenv(CONVERSATION_DB_ENV, str(tmp_path / "env.db"))
        store = ConversationStore(db_path=tmp_path / "explicit.db")
        assert store.db_path == tmp_path / "explicit.db"

    def test_env_override_wins_over_the_cwd_derived_default(self, monkeypatch, tmp_path):
        target = tmp_path / "durable" / "sessions.db"
        monkeypatch.setenv(CONVERSATION_DB_ENV, str(target))
        assert ConversationStore().db_path == target

    def test_user_home_is_expanded(self, monkeypatch):
        monkeypatch.setenv(CONVERSATION_DB_ENV, "~/somewhere/sessions.db")
        assert ConversationStore().db_path == Path.home() / "somewhere" / "sessions.db"

    def test_unset_env_keeps_the_project_default(self, monkeypatch):
        monkeypatch.delenv(CONVERSATION_DB_ENV, raising=False)
        from victor.config.settings import get_project_paths

        assert ConversationStore().db_path == get_project_paths().project_db

    def test_blank_env_is_ignored(self, monkeypatch):
        monkeypatch.setenv(CONVERSATION_DB_ENV, "   ")
        from victor.config.settings import get_project_paths

        assert ConversationStore().db_path == get_project_paths().project_db


class TestDurableEvaluationConversations:
    def test_pins_the_store_for_the_duration_of_a_run(self, monkeypatch, tmp_path):
        monkeypatch.delenv(CONVERSATION_DB_ENV, raising=False)
        target = tmp_path / "evaluations" / "sessions.db"
        with durable_evaluation_conversations(db_path=target) as pinned:
            assert pinned == target
            # A store constructed mid-run — as each task's agent does — lands there
            # rather than in the temp workspace it happens to be chdir'd into.
            assert ConversationStore().db_path == target
            assert target.parent.is_dir()
        assert CONVERSATION_DB_ENV not in os.environ

    def test_restores_a_pre_existing_value(self, monkeypatch, tmp_path):
        monkeypatch.setenv(CONVERSATION_DB_ENV, str(tmp_path / "caller.db"))
        with durable_evaluation_conversations(db_path=tmp_path / "eval.db") as pinned:
            # An explicit caller setting is authoritative; the harness defers.
            assert pinned == tmp_path / "caller.db"
            assert ConversationStore().db_path == tmp_path / "caller.db"
        assert os.environ[CONVERSATION_DB_ENV] == str(tmp_path / "caller.db")

    def test_env_is_cleaned_up_even_when_the_run_raises(self, monkeypatch, tmp_path):
        monkeypatch.delenv(CONVERSATION_DB_ENV, raising=False)
        try:
            with durable_evaluation_conversations(db_path=tmp_path / "e.db"):
                raise RuntimeError("benchmark blew up")
        except RuntimeError:
            pass
        assert CONVERSATION_DB_ENV not in os.environ

    def test_unwritable_target_degrades_instead_of_aborting_the_run(self, monkeypatch, tmp_path):
        monkeypatch.delenv(CONVERSATION_DB_ENV, raising=False)
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        with durable_evaluation_conversations(db_path=blocker / "sessions.db") as pinned:
            # A read-only or impossible home must not take the benchmark down.
            assert pinned is None
        assert CONVERSATION_DB_ENV not in os.environ
