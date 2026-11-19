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

"""FOU-A2c unit suite: PERCEIVE memory injection (FEP-0048).

Covers the increment's gates: the injection block appears when a seeded
preference matches the session context; OFF-WHEN-EMPTY is a byte-identical
no-op (the critical negative); ``memory_enabled=false`` and
``VICTOR_MEMORY_ENABLED=false`` both suppress; the block registers with the
BudgetManager (memory competes, never free-rides); the hard cap excludes by
rank instead of overflowing; scope isolation holds (user fact in user-context
sessions, project fact in project sessions, cross-scope never).

Databases are fixture-injected onto ``tmp_path``; no network, no real
``~/.victor`` writes.
"""

from pathlib import Path

import pytest

from victor.agent.budget_manager import BudgetManager
from victor.agent.memory_injection import (
    DEFAULT_MEMORY_INJECTION_MAX_TOKENS,
    MAX_ENTRY_CHARS,
    MEMORY_BUDGET_BLOCK_NAME,
    MemoryBlockBuilder,
    memory_prompt_suffix,
    memory_scopes_from_context,
    resolve_memory_enabled,
    resolve_memory_scope_enabled,
)
from victor.agent.perception_integration import PerceptionIntegration
from victor.config.memory_settings import MemorySettings
from victor.core.database import (
    get_database,
    get_project_database,
    reset_database,
    reset_project_database,
)
from victor.storage.memory.adapters import create_preference_episodic_adapters
from victor.storage.memory.preference_episodic import MemoryScope, PreferenceEpisodicStore
from victor.storage.memory.unified import UnifiedMemoryCoordinator


class _RecordingBudgetManager:
    """Duck-typed BudgetManager double that records context-block registrations."""

    def __init__(self, remaining: int = 10_000):
        self.remaining_tokens = remaining
        self.registrations: list = []
        self.unregistrations: list = []
        self._blocks: dict = {}

    def remaining_context_block_tokens(self) -> int:
        return self.remaining_tokens

    def register_context_block(self, name: str, token_estimate: int, priority: int = 3) -> bool:
        self.registrations.append((name, token_estimate, priority))
        self._blocks[name] = token_estimate
        return True

    def unregister_context_block(self, name: str) -> bool:
        self.unregistrations.append(name)
        return self._blocks.pop(name, None) is not None


class _SaturatedBudgetManager(_RecordingBudgetManager):
    """A budget manager whose context-block allowance is already spent."""

    def remaining_context_block_tokens(self) -> int:
        return 0

    def register_context_block(self, name: str, token_estimate: int, priority: int = 3) -> bool:
        raise AssertionError("no registration should be attempted on a saturated allowance")


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


@pytest.fixture
def coordinator(store: PreferenceEpisodicStore) -> UnifiedMemoryCoordinator:
    """Coordinator wired to both A2a lanes of the tmp-path store."""
    coord = UnifiedMemoryCoordinator(enable_memory_evolution=False)
    for adapter in create_preference_episodic_adapters(store):
        coord.register_provider(adapter)
    return coord


def _builder(
    coordinator: UnifiedMemoryCoordinator,
    **kwargs,
) -> MemoryBlockBuilder:
    return MemoryBlockBuilder(coordinator, **kwargs)


# =============================================================================
# Injection appears on seeded memory
# =============================================================================


async def test_seeded_preference_surfaces_in_block(coordinator, store):
    store_pref = "always use uv for packaging, never pip"
    store.set_preference(store_pref, scope=MemoryScope.USER)

    block = await _builder(coordinator).build("packaging uv pip", scopes=(MemoryScope.USER,))
    assert block is not None
    assert store_pref in block.text
    assert block.token_estimate > 0
    assert block.entries[0]["scope"] == "user"


async def test_block_registered_by_name_contains_header(coordinator, store):
    store.set_preference("prefer pytest in this repo", scope=MemoryScope.PROJECT)
    block = await _builder(coordinator).build("pytest repo preference")
    assert block is not None
    assert block.text.startswith("[long-term memory]")
    assert "(project)" in block.text


# =============================================================================
# OFF-WHEN-EMPTY: the critical negative — byte-identical prompt
# =============================================================================


async def test_empty_store_produces_no_block(coordinator: UnifiedMemoryCoordinator):
    block = await _builder(coordinator).build("anything at all")
    assert block is None


def test_empty_store_prompt_is_byte_identical():
    """With no block the prompt suffix is empty bytes — the ADR-036 negative."""
    base_prompt = "You are Victor. Solve the task."

    assert memory_prompt_suffix(None) == ""
    assert base_prompt + memory_prompt_suffix(None) == base_prompt


async def test_empty_store_perceive_is_byte_identical(coordinator: UnifiedMemoryCoordinator):
    """End-to-end at the PERCEIVE seam: empty store → no block, empty suffix."""
    integration = PerceptionIntegration(memory_injector=_builder(coordinator))
    perception = await integration.perceive("fix the auth bug")

    assert perception.memory_block is None
    assert memory_prompt_suffix(perception.memory_block) == ""
    # Metadata unchanged too: no memory keys on the empty-store path.
    assert "memory_block_tokens" not in perception.metadata
    assert "memory_prompt" not in perception.metadata


# =============================================================================
# Suppression: settings flag + env override
# =============================================================================


async def test_memory_enabled_false_suppresses(coordinator, store):
    store.set_preference("should not surface", scope=MemoryScope.PROJECT)
    settings = MemorySettings(memory_enabled=False)
    block = await _builder(coordinator, settings=settings).build("should not surface")
    assert block is None


async def test_env_override_false_suppresses(coordinator, store, monkeypatch):
    store.set_preference("should not surface either", scope=MemoryScope.PROJECT)
    monkeypatch.setenv("VICTOR_MEMORY_ENABLED", "false")
    # Even a settings object that says ON is overridden by the env at read time.
    block = await _builder(coordinator, settings=MemorySettings(memory_enabled=True)).build(
        "should not surface either"
    )
    assert block is None
    assert resolve_memory_enabled(MemorySettings(memory_enabled=True)) is False


async def test_env_override_true_enables_despite_default(monkeypatch):
    monkeypatch.setenv("VICTOR_MEMORY_ENABLED", "true")
    assert resolve_memory_enabled(None) is True


async def test_env_unset_defaults_on(monkeypatch):
    monkeypatch.delenv("VICTOR_MEMORY_ENABLED", raising=False)
    assert resolve_memory_enabled(None) is True
    assert resolve_memory_enabled(MemorySettings()) is True


async def test_per_scope_opt_out_suppresses_only_that_scope(coordinator, store):
    store.set_preference("user fact", scope=MemoryScope.USER)
    store.set_preference("project fact", scope=MemoryScope.PROJECT)
    settings = MemorySettings(user_scope_enabled=False)

    builder = _builder(coordinator, settings=settings)
    user_block = await builder.build("user fact", scopes=(MemoryScope.USER,))
    assert user_block is None
    project_block = await builder.build("project fact", scopes=(MemoryScope.PROJECT,))
    assert project_block is not None
    assert "project fact" in project_block.text
    assert resolve_memory_scope_enabled(settings, MemoryScope.USER) is False
    assert resolve_memory_scope_enabled(settings, MemoryScope.PROJECT) is True


# =============================================================================
# Budget competition
# =============================================================================


async def test_budget_registration_happens(coordinator, store):
    store.set_preference("budget probe preference", scope=MemoryScope.PROJECT)
    budget = _RecordingBudgetManager()
    block = await _builder(coordinator, budget_manager=budget).build("budget probe preference")

    assert block is not None
    assert budget.registrations, "memory must register its token cost, not free-ride"
    name, tokens, priority = budget.registrations[0]
    assert name == MEMORY_BUDGET_BLOCK_NAME
    assert tokens == block.token_estimate
    assert priority >= 0


async def test_budget_registration_released_when_no_entries(coordinator, store):
    """An all-entries-overflowed block registers nothing (nothing injected)."""
    store.set_preference("x" * 200, scope=MemoryScope.PROJECT)  # single fat entry
    budget = _RecordingBudgetManager()
    block = await _builder(coordinator, budget_manager=budget, max_tokens=32).build("x")
    assert block is None
    assert budget.registrations == []


async def test_saturated_budget_yields_no_block(coordinator, store):
    store.set_preference("starved preference", scope=MemoryScope.PROJECT)
    budget = _SaturatedBudgetManager()
    block = await _builder(coordinator, budget_manager=budget).build("starved preference")
    assert block is None


async def test_block_refit_to_budget_allowance(coordinator, store):
    for i in range(20):
        store.set_preference(
            f"filler preference number {i} for the block", scope=MemoryScope.PROJECT
        )
    budget = _RecordingBudgetManager(remaining=64)
    block = await _builder(coordinator, budget_manager=budget).build("filler preference number")
    assert block is not None
    assert block.token_estimate <= 64


# =============================================================================
# Hard cap: exclude by rank, never overflow
# =============================================================================


async def test_block_respects_hard_cap(coordinator, store):
    for i in range(30):
        store.set_preference(
            f"preference {i} with enough words to cost real tokens in the block",
            scope=MemoryScope.PROJECT,
        )
    cap = 80
    block = await _builder(coordinator, max_tokens=cap).build("preference enough words tokens")
    assert block is not None
    assert block.token_estimate <= cap
    assert block.dropped_count > 0, "oversized store content must be excluded by rank"


async def test_long_entry_truncated_not_overflowed(coordinator, store):
    store.set_preference("long " * (MAX_ENTRY_CHARS // 5 + 10), scope=MemoryScope.PROJECT)
    block = await _builder(coordinator, max_tokens=200).build("long")
    assert block is not None
    assert block.token_estimate <= 200
    assert block.entries[0]["truncated"] is True
    assert all(len(line) <= MAX_ENTRY_CHARS + 40 for line in block.text.splitlines()[1:])


async def test_default_cap_is_500_tokens(coordinator, store):
    assert DEFAULT_MEMORY_INJECTION_MAX_TOKENS == 500
    assert MemorySettings().memory_injection_max_tokens == 500


# =============================================================================
# Scope isolation
# =============================================================================


async def test_user_scope_fact_surfaces_only_in_user_session(coordinator, store):
    store.set_preference("uv over pip user directive", scope=MemoryScope.USER)
    store.set_preference("pytest over unittest project directive", scope=MemoryScope.PROJECT)

    user_block = await _builder(coordinator).build("uv pip directive", scopes=(MemoryScope.USER,))
    assert user_block is not None
    assert "uv over pip" in user_block.text
    assert "pytest over unittest" not in user_block.text

    project_block = await _builder(coordinator).build(
        "pytest unittest directive", scopes=(MemoryScope.PROJECT,)
    )
    assert project_block is not None
    assert "pytest over unittest" in project_block.text
    assert "uv over pip" not in project_block.text


async def test_cross_scope_never_surfaces(coordinator, store):
    """Project rows are physically in another database; a user query never sees them."""
    store.set_preference("project-only fact", scope=MemoryScope.PROJECT)
    store.record_episode("project-only episode summary", thread_id="t1")

    block = await _builder(coordinator).build("project-only", scopes=(MemoryScope.USER,))
    assert block is None


async def test_default_scopes_cover_both(coordinator, store):
    store.set_preference("user side fact", scope=MemoryScope.USER)
    store.set_preference("project side fact", scope=MemoryScope.PROJECT)
    block = await _builder(coordinator).build("side fact")
    assert block is not None
    assert "user side fact" in block.text
    assert "project side fact" in block.text
    assert set(block.scopes) == {"user", "project"}


def test_memory_scopes_from_context_parsing():
    assert memory_scopes_from_context(None) == (MemoryScope.USER, MemoryScope.PROJECT)
    assert memory_scopes_from_context({}) == (MemoryScope.USER, MemoryScope.PROJECT)
    assert memory_scopes_from_context({"memory_scope": "user"}) == (MemoryScope.USER,)
    assert memory_scopes_from_context({"memory_scopes": ["project"]}) == (MemoryScope.PROJECT,)
    assert memory_scopes_from_context({"memory_scopes": ["user", "project"]}) == (
        MemoryScope.USER,
        MemoryScope.PROJECT,
    )


# =============================================================================
# PerceptionIntegration seam
# =============================================================================


async def test_perceive_surfaces_memory_block(coordinator, store):
    store.set_preference("the service was renamed to cart", scope=MemoryScope.PROJECT)
    integration = PerceptionIntegration(memory_injector=_builder(coordinator))
    perception = await integration.perceive("what happened to the service")

    assert perception.memory_block is not None
    assert "the service was renamed to cart" in perception.memory_block.text
    assert perception.metadata["memory_block_tokens"] == perception.memory_block.token_estimate
    assert "the service was renamed to cart" in perception.metadata["memory_prompt"]
    assert perception.to_dict()["memory_block"]["token_estimate"] > 0


async def test_perceive_without_injector_has_no_block():
    integration = PerceptionIntegration()
    perception = await integration.perceive("plain query")
    assert perception.memory_block is None
    assert perception.to_dict()["memory_block"] is None


def test_set_memory_injector_late_binding(coordinator):
    integration = PerceptionIntegration()
    assert integration.memory_injector is None
    integration.set_memory_injector(_builder(coordinator))
    assert integration.memory_injector is not None
    integration.set_memory_injector(None)
    assert integration.memory_injector is None


async def test_perceive_memory_failure_degrades_to_no_block():
    class _ExplodingBuilder:
        async def build(self, query, scopes):
            raise RuntimeError("memory backend exploded")

    integration = PerceptionIntegration(memory_injector=_ExplodingBuilder())
    perception = await integration.perceive("query against broken memory")
    assert perception.memory_block is None


# =============================================================================
# BudgetManager integration (real BudgetManager)
# =============================================================================


def test_budget_manager_registers_and_evicts_by_priority():
    manager = BudgetManager()
    manager.max_context_block_tokens = 100

    assert manager.register_context_block("grounding", 60, priority=1) is True
    assert manager.register_context_block("memory_perceive", 30, priority=3) is True
    assert manager.remaining_context_block_tokens() == 10

    # A weaker block that doesn't fit is refused — it refits or drops.
    assert manager.register_context_block("enhancement", 50, priority=5) is False
    # A stronger block evicts the weaker one to fit.
    assert manager.register_context_block("critical", 50, priority=0) is True
    assert "enhancement" not in manager.get_diagnostics()["context_blocks"]
    assert manager.unregister_context_block("critical") is True
    assert manager.unregister_context_block("critical") is False


def test_budget_manager_refuses_block_larger_than_allowance():
    manager = BudgetManager()
    manager.max_context_block_tokens = 64
    assert manager.register_context_block("too_big", 128, priority=3) is False
    assert manager.remaining_context_block_tokens() == 64
