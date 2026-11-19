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

"""Dormancy tests for the FEP-0048 A2c production wiring (FOU-A2c).

The FEP-0041 lesson: PR #1549 shipped a hook that no chat-loop caller ever
invoked, and real sessions silently got no behavior — the tool existed, the
detector existed, nothing wired them. These tests pin the WIRING itself:

- constructing ``AgentOrchestrator`` the way production does (composition
  root for CLI chat — buffered and streaming — TUI, and the HTTP/ChatService
  path via ``AgentFactory``) must bind a real ``MemoryBlockBuilder`` (built
  via ``from_environment``, the canonical-database production path) onto the
  orchestrator's perception integration;
- one PERCEIVE cycle through the REAL chat seam
  (``AgenticLoop._analyze_turn`` → ``RuntimeIntelligenceService.analyze_turn``
  → ``perception_integration.perceive``) must surface a seeded memory in the
  perception block AND its prompt-suffix metadata — without the test ever
  calling ``MemoryBlockBuilder`` directly;
- ``memory_enabled=false`` (and the ``VICTOR_MEMORY_ENABLED`` env override)
  must skip binding entirely through the same real path;
- a store failure at bind time must degrade to no-injection, never a failed
  session start.

Databases are fixture-injected onto ``tmp_path``; no network, no real
``~/.victor`` writes.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.agent.memory_injection import MemoryBlockBuilder
from victor.agent.orchestrator import AgentOrchestrator
from victor.config.memory_settings import MemorySettings
from victor.config.settings import Settings
from victor.core.database import (
    get_database,
    get_project_database,
    reset_database,
    reset_project_database,
)
from victor.storage.memory.preference_episodic import MemoryScope, PreferenceEpisodicStore

SEEDED_PREFERENCE = "always use uv for packaging, never pip"


@pytest.fixture
def tmp_dbs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the canonical database singletons at tmp_path (the from_environment path).

    ``MemoryBlockBuilder.from_environment`` resolves the production databases
    through the ``get_database()`` / ``get_project_database()`` singletons, so
    the fixture pre-seeds those singletons on tmp_path (a pre-seeded singleton
    ignores the later no-arg call) and pins cwd so the project DB resolves to
    the tmp project root.
    """
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
    """Mock provider, mirroring the orchestrator-core unit fixtures."""
    provider = MagicMock()
    provider.name = "mock_provider"
    provider.supports_tools.return_value = True
    provider.get_context_window.return_value = 100000
    provider.chat = AsyncMock(return_value=MagicMock(content="Response", tool_calls=[]))
    return provider


def _production_settings(**memory_kwargs) -> Settings:
    """Settings the way a real (offline-safe) session resolves them."""
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


# =============================================================================
# The dormancy test: the PRODUCTION path injects
# =============================================================================


async def test_production_chat_surface_injects_memory_block(tmp_dbs, seeded_store, mock_provider):
    """Real wiring: orchestrator construction binds the builder; PERCEIVE via
    the chat seam surfaces the seeded memory in block + prompt metadata."""
    orch = _build_orchestrator(_production_settings(), mock_provider)

    # 1. The binding happened through the real constructor, not by the test.
    integration = orch._runtime_intelligence.perception_integration
    assert integration.memory_injector is not None
    assert isinstance(integration.memory_injector, MemoryBlockBuilder)
    # from_environment built it against the canonical database singletons —
    # the same tmp-path managers this fixture seeded.
    assert integration.memory_injector.coordinator is not None
    assert integration.memory_injector.settings is orch.settings

    # 2. One PERCEIVE cycle through the REAL chat seam. AgenticLoop._analyze_turn
    #    (both the buffered run() and run_streaming() stages) delegates here —
    #    this is the exact call production makes, no direct builder invocation.
    snapshot = await orch._runtime_intelligence.analyze_turn(
        "packaging uv pip",
        context={"project_path": str(tmp_dbs[1].project_root)},
    )
    perception = snapshot.perception
    assert perception is not None
    assert perception.memory_block is not None
    assert SEEDED_PREFERENCE in perception.memory_block.text
    # The block reached prompt-assembly metadata through the real wiring.
    assert perception.metadata["memory_block_tokens"] == perception.memory_block.token_estimate
    assert SEEDED_PREFERENCE in perception.metadata["memory_prompt"]


# =============================================================================
# Negative: memory_enabled=false → no injection through the same real path
# =============================================================================


async def test_memory_disabled_skips_binding_and_injection(tmp_dbs, seeded_store, mock_provider):
    """Opt-out: no builder bound, no block, no metadata keys — zero-cost skip."""
    orch = _build_orchestrator(_production_settings(memory_enabled=False), mock_provider)

    integration = orch._runtime_intelligence.perception_integration
    assert integration.memory_injector is None

    snapshot = await orch._runtime_intelligence.analyze_turn("packaging uv pip")
    perception = snapshot.perception
    assert perception is not None
    assert perception.memory_block is None
    assert "memory_prompt" not in perception.metadata
    assert "memory_block_tokens" not in perception.metadata


async def test_env_override_false_skips_binding_despite_settings(
    tmp_dbs, seeded_store, mock_provider, monkeypatch
):
    """VICTOR_MEMORY_ENABLED=false wins over settings at bind time too."""
    monkeypatch.setenv("VICTOR_MEMORY_ENABLED", "false")
    orch = _build_orchestrator(_production_settings(memory_enabled=True), mock_provider)

    integration = orch._runtime_intelligence.perception_integration
    assert integration.memory_injector is None


# =============================================================================
# Negative: store failure degrades to no-injection, session start survives
# =============================================================================


async def test_bind_failure_degrades_to_no_injection(tmp_dbs, mock_provider, monkeypatch):
    """A store/DB failure at bind time must never break session start."""

    def _explode(*args, **kwargs):
        raise RuntimeError("memory backend unavailable")

    monkeypatch.setattr(MemoryBlockBuilder, "from_environment", staticmethod(_explode))
    orch = _build_orchestrator(_production_settings(), mock_provider)

    integration = orch._runtime_intelligence.perception_integration
    assert integration.memory_injector is None
    # The orchestrator itself is intact and perception still runs.
    snapshot = await orch._runtime_intelligence.analyze_turn("packaging uv pip")
    assert snapshot.perception is not None
    assert snapshot.perception.memory_block is None
