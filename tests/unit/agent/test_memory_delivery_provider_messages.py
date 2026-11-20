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

"""Provider-message delivery of the PERCEIVE memory block (Codex finding 6).

The FEP-0041/FEP-0048 lesson, pinned at the ACCEPTANCE boundary: the block
must actually reach the model. These tests capture the FINAL provider
messages — not perception metadata — on both turn shapes:

- **buffered** (``TurnExecutor._execute_model_turn`` — the shared metered
  model call behind ``ChatService.chat`` → ``AgenticLoop.run``);
- **streamed** (``orchestrator.get_assembled_messages`` — the message source
  of the streaming ACT in ``chat_stream_helpers``).

Contract (ADR-037 provenance boundary):

- the block enters provider-message assembly EXACTLY ONCE per message list,
  as an enveloped ``<untrusted source="memory" ...>`` span (wrap_untrusted
  envelope, never raw system-prompt text), positioned after the leading
  system message(s) and before history;
- the envelope carries the session-bound containment canary;
- the active ``BudgetManager`` (ToolService's session manager) receives the
  ``memory_perceive`` context-block registration;
- every negative — memory disabled, empty store, refused wrong-owner write,
  wrong workspace, exhausted context-block budget — delivers NO block and a
  byte-identical message list.

Databases are fixture-injected onto ``tmp_path``; no network, no real
``~/.victor`` writes.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.agent.memory_injection import (
    MEMORY_BUDGET_BLOCK_NAME,
    MEMORY_DELIVERY_METADATA_KEY,
    MEMORY_ENVELOPE_ORIGIN,
)
from victor.agent.orchestrator import AgentOrchestrator
from victor.agent.services.turn_execution_runtime import TurnExecutor
from victor.config.memory_settings import MemorySettings
from victor.config.settings import Settings
from victor.core.database import (
    get_database,
    get_project_database,
    reset_database,
    reset_project_database,
)
from victor.core.provider_messages import CompletionResponse
from victor.security.containment import SessionContainmentState, bind_containment_state
from victor.storage.memory.preference_episodic import (
    CrossPrincipalAccessRefused,
    MemoryScope,
    PreferenceEpisodicStore,
)

SEEDED_PREFERENCE = "always use uv for packaging, never pip"
PROJECT_FACT = "the deploy target for this repo is the staging cluster"
QUERY = "packaging uv pip"

ENVELOPE_OPEN = '<untrusted source="memory"'
ENVELOPE_CLOSE = "</untrusted>"


# =============================================================================
# Fixtures (mirroring test_memory_injection_production_wiring.py)
# =============================================================================


@pytest.fixture
def tmp_dbs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the canonical database singletons at tmp_path."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VICTOR_MEMORY_ENABLED", raising=False)
    reset_database()
    reset_project_database(tmp_path)
    global_db = get_database(tmp_path / "victor.db")
    project_db = get_project_database(tmp_path)
    yield global_db, project_db
    reset_database()
    reset_project_database(tmp_path)


@pytest.fixture
def seeded_store(tmp_dbs) -> PreferenceEpisodicStore:
    """A store over the tmp-path databases, seeded with one user preference."""
    global_db, project_db = tmp_dbs
    store = PreferenceEpisodicStore(global_db=global_db, project_db=project_db)
    store.set_preference(SEEDED_PREFERENCE, scope=MemoryScope.USER)
    return store


@pytest.fixture
def mock_provider():
    provider = MagicMock()
    provider.name = "mock_provider"
    provider.supports_tools.return_value = True
    provider.get_context_window.return_value = 100000
    provider.chat = AsyncMock(return_value=MagicMock(content="Response", tool_calls=[]))
    return provider


def _production_settings(**memory_kwargs) -> Settings:
    overrides = {
        "analytics_enabled": False,
        "use_semantic_tool_selection": False,
        "use_mcp_tools": False,
    }
    if memory_kwargs:
        overrides["memory"] = MemorySettings(**memory_kwargs)
    return Settings(**overrides)


def _build_orchestrator(settings: Settings, mock_provider) -> AgentOrchestrator:
    """Construct the chat composition root exactly as production does."""
    with patch("victor.core.bootstrap_services.bootstrap_new_services"):
        return AgentOrchestrator(
            settings=settings,
            provider=mock_provider,
            model="test-model",
        )


def _perceive(orch: AgentOrchestrator, query: str = QUERY):
    """Run the real PERCEIVE seam the loop runs before every ACT."""
    return orch._runtime_intelligence.analyze_turn(query)


def _base_messages(query: str = QUERY):
    """The history a turn starts with: system prompt + the user turn."""
    return [
        _msg("system", "You are Victor."),
        _msg("user", query),
    ]


def _msg(role: str, content: str):
    from victor.core.provider_messages import Message

    return Message(role=role, content=content)


def _message_bodies(messages) -> list:
    """Byte-comparable projection of a message list."""
    return [(m.role, m.content) for m in messages]


def _envelope_messages(messages) -> list:
    return [m for m in messages if ENVELOPE_OPEN in (m.content or "")]


def _snapshot_assembly(orch: AgentOrchestrator, query: str = QUERY) -> list:
    """Message bodies of the streaming-assembly projection (pre-perceive)."""
    return _message_bodies(orch.get_assembled_messages(current_query=query))


def _stream_history_executor():
    """A ChatStreamExecutor whose runtime carries no intelligence capability."""
    from victor.agent.services.chat_stream_executor import StreamingChatExecutor

    runtime_owner = SimpleNamespace(services=SimpleNamespace(intelligence=None))
    return StreamingChatExecutor(runtime_owner)


def _buffered_executor(orch: AgentOrchestrator, messages):
    """TurnExecutor wired the way production binds it, with a capturing provider."""
    provider_context = SimpleNamespace(
        model="test-model",
        temperature=0.7,
        max_tokens=1024,
        provider_name="mock_provider",
        thinking=False,
        reasoning_effort=None,
        supports_reasoning_effort=None,
    )
    execution_provider = MagicMock()
    execution_provider.execute_turn = AsyncMock(
        return_value=CompletionResponse(content="ok", tool_calls=None)
    )
    chat_context = SimpleNamespace(messages=list(messages))
    executor = TurnExecutor(
        chat_context=chat_context,
        tool_context=MagicMock(),
        provider_context=provider_context,
        execution_provider=execution_provider,
    )
    executor._orchestrator = orch
    return executor, execution_provider


async def test_stream_history_serialization_excludes_memory_block(
    tmp_dbs, seeded_store, mock_provider
):
    """The streaming executor's conversation-history PROJECTION (analysis
    context, not a provider call) never carries the memory envelope."""
    orch = _build_orchestrator(_production_settings(), mock_provider)
    _seed_conversation(orch)
    perception = (await _perceive(orch)).perception
    assert perception.memory_block is not None

    executor = _stream_history_executor()
    history = executor._get_conversation_history(executor._runtime_owner, orch, QUERY)
    assert history is not None
    serialized = "".join(str(entry.get("content", "")) for entry in history)
    assert ENVELOPE_OPEN not in serialized
    assert SEEDED_PREFERENCE not in serialized


# =============================================================================
# Buffered turn: enveloped block exactly once in the final provider messages
# =============================================================================


async def test_buffered_turn_delivers_enveloped_memory_exactly_once(
    tmp_dbs, seeded_store, mock_provider
):
    orch = _build_orchestrator(_production_settings(), mock_provider)
    perception = (await _perceive(orch)).perception
    assert perception.memory_block is not None  # precondition: a block exists

    executor, execution_provider = _buffered_executor(orch, _base_messages())
    await executor._execute_model_turn()

    messages = execution_provider.execute_turn.await_args.kwargs["messages"]
    enveloped = _envelope_messages(messages)
    assert len(enveloped) == 1  # exactly once

    block_message = enveloped[0]
    assert SEEDED_PREFERENCE in block_message.content
    assert block_message.content.startswith(ENVELOPE_OPEN)
    assert block_message.content.rstrip().endswith(ENVELOPE_CLOSE)
    # Provenance envelope, not raw system-prompt text: origin + canary attrs.
    assert f'origin="{MEMORY_ENVELOPE_ORIGIN}' in block_message.content
    assert 'canary="victor-canary-' in block_message.content
    # Delivery metadata marks the message as the one-shot memory carrier.
    assert block_message.metadata.get(MEMORY_DELIVERY_METADATA_KEY) is True

    # Position: after the leading system message, before history.
    assert messages.index(block_message) == 1
    assert messages[0].role == "system"
    assert messages[2:] == _base_messages()[1:]

    # The insertion is transient: the orchestrator's own history is untouched.
    assert not _envelope_messages(orch.messages)


def _seed_conversation(orch: AgentOrchestrator, query: str = QUERY) -> None:
    """Attach the turn history a real conversation starts with."""
    orch.add_message("system", "You are Victor.")
    orch.add_message("user", query)


async def test_streamed_turn_delivers_enveloped_memory_exactly_once(
    tmp_dbs, seeded_store, mock_provider
):
    """Streaming ACT assembly (get_assembled_messages) carries the block once."""
    orch = _build_orchestrator(_production_settings(), mock_provider)
    _seed_conversation(orch)
    perception = (await _perceive(orch)).perception
    assert perception.memory_block is not None

    assembled = orch.get_assembled_messages(current_query=QUERY)

    enveloped = _envelope_messages(assembled)
    assert len(enveloped) == 1
    block_message = enveloped[0]
    assert SEEDED_PREFERENCE in block_message.content
    assert block_message.content.startswith(ENVELOPE_OPEN)
    assert block_message.metadata.get(MEMORY_DELIVERY_METADATA_KEY) is True
    # After system, before history.
    assert assembled.index(block_message) == 1
    assert assembled[0].role == "system"
    assert assembled[2].role == "user"

    # Repeat assembly for a later provider step of the same turn: still
    # exactly one envelope per message list (no accumulation in history).
    again = orch.get_assembled_messages(current_query=QUERY)
    assert len(_envelope_messages(again)) == 1
    assert not _envelope_messages(orch.messages)


async def test_envelope_carries_bound_session_canary(tmp_dbs, seeded_store, mock_provider):
    """The session containment state bound around the turn is the canary source."""
    orch = _build_orchestrator(_production_settings(), mock_provider)
    state = SessionContainmentState(session_id="session_abc")
    _seed_conversation(orch)
    with bind_containment_state(state):
        await _perceive(orch)
        assembled = orch.get_assembled_messages(current_query=QUERY)

    enveloped = _envelope_messages(assembled)
    assert len(enveloped) == 1
    assert state.canary in enveloped[0].content


# =============================================================================
# Budget competition: the ACTIVE BudgetManager receives the registration
# =============================================================================


async def test_active_budget_manager_receives_memory_registration(
    tmp_dbs, seeded_store, mock_provider
):
    orch = _build_orchestrator(_production_settings(), mock_provider)
    integration = orch._runtime_intelligence.perception_integration
    # The binding supplies the session's ACTIVE context-block budget manager
    # (resolved lazily; the tool-count budget class is never mistaken for it).
    budget = integration.memory_injector.resolve_budget_manager()
    assert budget is not None
    assert callable(getattr(budget, "register_context_block", None))

    perception = (await _perceive(orch)).perception
    assert perception.memory_block is not None

    diagnostics = budget.get_diagnostics()
    registered = diagnostics["context_blocks"].get(MEMORY_BUDGET_BLOCK_NAME)
    assert registered is not None
    assert registered["tokens"] == perception.memory_block.token_estimate


async def test_exhausted_budget_delivers_no_block_and_identical_prompt(
    tmp_dbs, seeded_store, mock_provider
):
    """Context-block allowance fully consumed: memory yields, prompt unchanged."""
    orch = _build_orchestrator(_production_settings(), mock_provider)
    budget = (
        orch._runtime_intelligence.perception_integration.memory_injector.resolve_budget_manager()
    )
    assert budget is not None
    budget.max_context_block_tokens = 0

    _seed_conversation(orch)
    baseline = _snapshot_assembly(orch)
    perception = (await _perceive(orch)).perception
    assert perception.memory_block is None  # the exhausted allowance refused the block

    messages = orch.get_assembled_messages(current_query=QUERY)
    assert _message_bodies(messages) == baseline  # byte-identical
    assert not _envelope_messages(messages)


# =============================================================================
# Negatives: every suppression path keeps the prompt byte-identical
# =============================================================================


async def test_memory_disabled_delivers_nothing(tmp_dbs, seeded_store, mock_provider):
    orch = _build_orchestrator(_production_settings(memory_enabled=False), mock_provider)
    assert orch._runtime_intelligence.perception_integration.memory_injector is None

    _seed_conversation(orch)
    baseline = _snapshot_assembly(orch)
    perception = (await _perceive(orch)).perception
    assert perception.memory_block is None

    messages = orch.get_assembled_messages(current_query=QUERY)
    assert _message_bodies(messages) == baseline  # byte-identical
    assert not _envelope_messages(messages)

    executor, execution_provider = _buffered_executor(orch, _base_messages())
    await executor._execute_model_turn()
    sent = execution_provider.execute_turn.await_args.kwargs["messages"]
    assert _message_bodies(sent) == _message_bodies(_base_messages())


async def test_empty_store_delivers_nothing_and_prompt_is_byte_identical(tmp_dbs, mock_provider):
    """Memory enabled, store empty: off-when-empty holds at delivery too."""
    orch = _build_orchestrator(_production_settings(), mock_provider)
    _seed_conversation(orch)
    baseline = _snapshot_assembly(orch)
    perception = (await _perceive(orch)).perception
    assert perception.memory_block is None

    messages = orch.get_assembled_messages(current_query=QUERY)
    assert _message_bodies(messages) == baseline  # byte-identical
    assert not _envelope_messages(messages)


async def test_wrong_owner_write_refused_and_delivers_nothing(tmp_dbs, mock_provider):
    """A cross-principal write is refused (finding 4 posture) — and because it
    never stored a row, the production delivery path stays byte-identical."""
    global_db, project_db = tmp_dbs
    alice_store = PreferenceEpisodicStore(
        global_db=global_db, project_db=project_db, principal_id="alice"
    )
    alice_store.set_preference("alice likes ruff", scope=MemoryScope.USER)

    with pytest.raises(CrossPrincipalAccessRefused):
        alice_store.set_preference(
            "bob smuggles a preference",
            scope=MemoryScope.USER,
            on_behalf_of="bob",
        )

    orch = _build_orchestrator(_production_settings(), mock_provider)
    perception = (await _perceive(orch)).perception
    # The unbound production store legitimately sees alice's own preference;
    # what must NOT happen is bob's refused row surfacing.
    if perception.memory_block is not None:
        assert "bob smuggles a preference" not in perception.memory_block.text
        _seed_conversation(orch)
        assembled = orch.get_assembled_messages(current_query=QUERY)
        enveloped = _envelope_messages(assembled)
        assert len(enveloped) == 1
        assert "bob smuggles a preference" not in enveloped[0].content


async def test_wrong_workspace_project_memory_not_delivered(
    tmp_dbs, tmp_path, mock_provider, monkeypatch
):
    """Project-scope memory is workspace-bound: a session rooted in another
    workspace never sees (nor delivers) it. Only the project fact was seeded,
    so the foreign-workspace prompt is byte-identical."""
    global_db, project_db = tmp_dbs
    home_store = PreferenceEpisodicStore(global_db=global_db, project_db=project_db)
    home_store.set_preference(PROJECT_FACT, scope=MemoryScope.PROJECT)

    other_workspace = tmp_path / "other_workspace"
    other_workspace.mkdir()
    monkeypatch.chdir(other_workspace)
    reset_project_database(other_workspace)

    orch = _build_orchestrator(_production_settings(), mock_provider)
    _seed_conversation(orch)
    baseline = _snapshot_assembly(orch)
    perception = (await _perceive(orch)).perception
    assert perception.memory_block is None  # nothing surfaced across workspaces

    messages = orch.get_assembled_messages(current_query=QUERY)
    assert _message_bodies(messages) == baseline  # byte-identical
    assert not _envelope_messages(messages)
    assert PROJECT_FACT not in "".join(m.content for m in messages)
    reset_project_database(other_workspace)


async def test_stale_turn_block_not_delivered_for_new_query(tmp_dbs, seeded_store, mock_provider):
    """The delivery candidate is turn-scoped: a block perceived for one query
    must not be injected into provider assembly for a different turn."""
    orch = _build_orchestrator(_production_settings(), mock_provider)
    other_turn = "completely unrelated query about kubernetes"
    _seed_conversation(orch, other_turn)
    baseline = _snapshot_assembly(orch, other_turn)
    await _perceive(orch, QUERY)  # a block exists for the OTHER query

    assembled = orch.get_assembled_messages(current_query=other_turn)
    assert not _envelope_messages(assembled)
    assert _message_bodies(assembled) == baseline  # byte-identical
