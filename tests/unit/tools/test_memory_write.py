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

"""FOU-A2b unit suite: memory_set / memory_forget write tools (FEP-0048).

Covers the increment's gates: preference round-trip through the TOOL layer
(not just the store), refusal surfacing as structured errors with the
positive leak-prevention guarantee (the error must not contain the matched
secret), oversized refusal, forget by id and by content, scope validation,
and the workspace-write permission gating that keeps read-only sessions from
mutating long-term memory.

Databases are fixture-injected onto ``tmp_path`` per the A2a conventions;
the tools are pinned to the store under test with
``set_memory_write_store``. No network.
"""

from pathlib import Path

import pytest

from victor.core.database import (
    get_database,
    get_project_database,
    reset_database,
    reset_project_database,
)
from victor.security.permissions import (
    DEFAULT_TOOL_PERMISSIONS,
    PermissionMode,
    PermissionPolicy,
)
from victor.storage.memory.preference_episodic import (
    MAX_MEMORY_CONTENT_BYTES,
    MemoryScope,
    PreferenceEpisodicStore,
)
from victor.tools.enums import AccessMode
from victor.tools.memory_write import (
    memory_forget,
    memory_set,
    set_memory_write_store,
)

_GITHUB_PAT = "ghp_" + "a" * 36  # canonical GITHUB_PAT_PATTERN, matches on Rust and Python paths
_SECRET_MEMORY = f"my token is {_GITHUB_PAT} please remember it"
_OVERSIZED_MEMORY = "word " * (MAX_MEMORY_CONTENT_BYTES // 4 + 64)


@pytest.fixture
def dbs(tmp_path: Path):
    """Fixture-injected global + project database managers on tmp_path."""
    reset_database()
    reset_project_database(tmp_path)
    global_db = get_database(tmp_path / "victor.db")
    project_db = get_project_database(tmp_path)
    yield global_db, project_db
    reset_database()
    reset_project_database(tmp_path)


@pytest.fixture
def store(dbs) -> PreferenceEpisodicStore:
    """Store wired to the tmp-path databases."""
    global_db, project_db = dbs
    return PreferenceEpisodicStore(global_db=global_db, project_db=project_db)


@pytest.fixture(autouse=True)
def pinned_store(store: PreferenceEpisodicStore):
    """Pin the tools to the tmp-path store; restore the lazy default after."""
    set_memory_write_store(store)
    yield store
    set_memory_write_store(None)


# =============================================================================
# memory_set round-trip through the tool layer
# =============================================================================


async def test_memory_set_stores_preference_via_store(store: PreferenceEpisodicStore):
    result = await memory_set(
        content="always use uv, never pip",
        scope="user",
        source="user-correction",
    )
    assert result["success"] is True
    assert result["scope"] == "user"

    found = store.get_preferences(MemoryScope.USER)
    assert [p.content for p in found] == ["always use uv, never pip"]
    assert found[0].id == result["preference_id"]
    assert found[0].source == "user-correction"


async def test_memory_set_defaults_to_project_scope(store: PreferenceEpisodicStore):
    result = await memory_set(content="this project uses pytest")
    assert result["success"] is True
    assert result["scope"] == "project"
    assert len(store.get_preferences(MemoryScope.PROJECT)) == 1
    assert store.get_preferences(MemoryScope.USER) == []


async def test_memory_set_rejects_unknown_scope(store: PreferenceEpisodicStore):
    result = await memory_set(content="anything", scope="workspace")
    assert result["success"] is False
    assert "user, project" in result["error"]
    assert store.get_preferences(MemoryScope.USER) == []
    assert store.get_preferences(MemoryScope.PROJECT) == []


# =============================================================================
# Refusals surface as structured errors (never leak the secret)
# =============================================================================


async def test_secret_shaped_content_is_refused_as_structured_error(
    store: PreferenceEpisodicStore,
):
    result = await memory_set(content=_SECRET_MEMORY, scope="user")
    assert result["success"] is False
    assert result["refusal"] == "SecretContentRefused"
    # Actionable: names the reason (matched secret TYPES), not the value.
    # Dispatch-path parity: the Rust scanner labels GitHub PATs
    # github_token; the Python pattern set names the same shape github_pat.
    assert ("github_pat" in result["error"]) or ("github_token" in result["error"])
    # Positive leak-prevention: the matched secret must not appear anywhere
    # in the tool result (message or any other field).
    assert _GITHUB_PAT not in str(result)
    # Nothing was persisted.
    assert store.get_preferences(MemoryScope.USER) == []


async def test_secret_shaped_refusal_does_not_leak_in_any_field(
    store: PreferenceEpisodicStore,
):
    """Every result field is checked individually, not just the message."""
    result = await memory_set(content=_SECRET_MEMORY, scope="project")
    for value in result.values():
        assert _GITHUB_PAT not in str(value)


async def test_oversized_content_is_refused_through_the_tool(
    store: PreferenceEpisodicStore,
):
    assert len(_OVERSIZED_MEMORY.encode("utf-8")) > MAX_MEMORY_CONTENT_BYTES
    result = await memory_set(content=_OVERSIZED_MEMORY, scope="project")
    assert result["success"] is False
    assert result["refusal"] == "OversizedContentRefused"
    assert str(MAX_MEMORY_CONTENT_BYTES) in result["error"]
    assert store.get_preferences(MemoryScope.PROJECT) == []


# =============================================================================
# memory_forget
# =============================================================================


async def test_memory_forget_by_id(store: PreferenceEpisodicStore):
    keep = store.set_preference("keep me", scope=MemoryScope.USER)
    drop = await memory_set(content="drop me", scope="user")

    result = await memory_forget(preference_id=drop["preference_id"])
    assert result["success"] is True
    assert result["forgotten"] == 1
    assert [p.id for p in store.get_preferences(MemoryScope.USER)] == [keep.id]

    # Idempotent: forgetting again reports zero.
    again = await memory_forget(preference_id=drop["preference_id"])
    assert again["success"] is True
    assert again["forgotten"] == 0


async def test_memory_forget_by_content_is_scope_bounded(store: PreferenceEpisodicStore):
    await memory_set(content="use pip", scope="user")
    await memory_set(content="use pip", scope="project")

    result = await memory_forget(content_match="pip", scope="project")
    assert result["success"] is True
    assert result["forgotten"] == 1
    # Only the project-scope row was deleted.
    assert len(store.get_preferences(MemoryScope.USER)) == 1
    assert store.get_preferences(MemoryScope.PROJECT) == []


async def test_memory_forget_requires_criteria(store: PreferenceEpisodicStore):
    result = await memory_forget()
    assert result["success"] is False
    assert "preference_id or content_match" in result["error"]


async def test_memory_forget_by_content_requires_scope(store: PreferenceEpisodicStore):
    await memory_set(content="pip is fine", scope="user")
    result = await memory_forget(content_match="pip")
    assert result["success"] is False
    assert "scope" in result["error"]
    # Vague matches must not sweep anything.
    assert len(store.get_preferences(MemoryScope.USER)) == 1


async def test_memory_forget_rejects_unknown_scope(store: PreferenceEpisodicStore):
    result = await memory_forget(preference_id="pref_x", scope="workspace")
    assert result["success"] is False


# Finding 13 (Codex review): an empty or whitespace-only content_match would
# contains-match every preference in the scope — the store must reject it and
# every row must survive, pinned from the TOOL layer.
@pytest.mark.parametrize("empty_match", ["", "   ", "\t\n"])
async def test_memory_forget_empty_content_match_is_rejected_and_rows_preserved(
    store: PreferenceEpisodicStore, empty_match
):
    await memory_set(content="always use uv", scope="user")
    await memory_set(content="uses pytest", scope="user")

    result = await memory_forget(content_match=empty_match, scope="user")

    assert result["success"] is False
    assert "content_match" in result["error"]
    remaining = sorted(p.content for p in store.get_preferences(MemoryScope.USER))
    assert remaining == ["always use uv", "uses pytest"]


async def test_memory_forget_empty_match_is_typed(store: PreferenceEpisodicStore):
    from victor.storage.memory.preference_episodic import EmptyContentMatchRefused

    await memory_forget(content_match=" ", scope="user")
    # The typed error class is the ValueError the store raises; the tool
    # surfaces it as a structured error (checked above) — pin the type here.
    with pytest.raises(EmptyContentMatchRefused):
        from victor.tools.memory_write import _get_store

        _get_store().forget(content_match=" ", scope=MemoryScope.USER)


# =============================================================================
# Permission gating (workspace-write per FEP-0048)
# =============================================================================


def test_memory_tools_require_workspace_write():
    assert DEFAULT_TOOL_PERMISSIONS["memory_set"] is PermissionMode.WORKSPACE_WRITE
    assert DEFAULT_TOOL_PERMISSIONS["memory_forget"] is PermissionMode.WORKSPACE_WRITE


def test_read_only_session_is_refused_memory_writes():
    policy = PermissionPolicy(PermissionMode.READ_ONLY)
    assert not policy.authorize("memory_set").allowed
    assert not policy.authorize("memory_forget").allowed


def test_workspace_write_session_is_allowed_memory_writes():
    policy = PermissionPolicy(PermissionMode.WORKSPACE_WRITE)
    assert policy.authorize("memory_set").allowed
    assert policy.authorize("memory_forget").allowed


def test_declared_access_mode_maps_to_workspace_write():
    """The tool metadata path must agree with DEFAULT_TOOL_PERMISSIONS."""
    assert memory_set.Tool.name == "memory_set"
    assert memory_forget.Tool.name == "memory_forget"
    assert PermissionMode.from_access_mode(memory_set.Tool.access_mode) == (
        PermissionMode.WORKSPACE_WRITE
    )
    assert memory_set.Tool.access_mode is AccessMode.WRITE
    assert memory_forget.Tool.access_mode is AccessMode.WRITE
