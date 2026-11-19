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

"""PERCEIVE-stage long-term memory injection (FEP-0048, FOU-A2c).

A focused collaborator composed by :class:`~victor.agent.perception_integration.
PerceptionIntegration`: it queries the ``UnifiedMemoryCoordinator`` (which fans
out to the A2a ``PreferenceEpisodicAdapter`` lanes), fits the ranked results
under a hard token cap by rank, registers the resulting block's token estimate
with the ``BudgetManager`` context-block allowance so memory competes for
context tokens instead of free-riding, and emits the block for prompt assembly.

Bounds and semantics:

- **Hard cap**: the block is assembled under ``memory_injection_max_tokens``
  (default 500) BEFORE the budget registration; oversized store content is
  excluded by rank and long entries are line-truncated — never overflowed.
- **Off-when-empty**: an empty store yields ``None`` — no block, no budget
  registration, no prompt change (``memory_prompt_suffix(None) == ""``).
- **Flag + env at every read**: ``memory_enabled`` (settings) and
  ``VICTOR_MEMORY_ENABLED`` (env override) are re-resolved on every
  ``build`` call; per-scope opt-outs gate each scope independently.

Layering: this module lives in the runtime layer (``victor/agent``) because it
consumes both the infrastructure memory stores (``victor.storage.memory``) and
the runtime ``BudgetManager``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from victor.storage.memory.preference_episodic import MemoryScope
from victor.storage.memory.unified import MemoryType

if TYPE_CHECKING:
    from victor.agent.budget_manager import BudgetManager
    from victor.storage.memory.unified import MemoryResult, UnifiedMemoryCoordinator

logger = logging.getLogger(__name__)

#: Default hard token cap on the injected memory block (settings override:
#: ``memory.memory_injection_max_tokens``).
DEFAULT_MEMORY_INJECTION_MAX_TOKENS = 500

#: A block smaller than this is not worth its prompt structure; when the
#: context-block allowance is tighter, memory yields (competes, never free-rides).
MIN_BLOCK_TOKENS = 16

#: Per-entry line truncation. Memory content is directives/summaries, not
#: documents (the store already refuses >4 KiB writes); this keeps one fat
#: entry from consuming the whole block.
MAX_ENTRY_CHARS = 400

#: Header line of the rendered block.
MEMORY_BLOCK_HEADER = "[long-term memory]"

#: Registered BudgetManager context-block name for the PERCEIVE memory block.
MEMORY_BUDGET_BLOCK_NAME = "memory_perceive"

#: Priority the memory block registers at (prompt-section convention:
#: lower number = more important). Memory sits below core grounding but
#: above the weakest enhancement blocks.
MEMORY_BLOCK_PRIORITY = 3

_POSITIVE_ENV = ("1", "true", "yes", "on")


def _env_flag(name: str) -> Optional[bool]:
    """Parse a VICTOR_* boolean env var; None when unset (matches effect_gate)."""
    env = os.environ.get(name)
    if env is None:
        return None
    return env.strip().lower() in _POSITIVE_ENV


def _memory_settings_source(settings: Any) -> Any:
    """The object that actually carries the memory fields.

    Accepts either the full ``Settings`` (its ``memory`` nested group wins)
    or a bare ``MemorySettings`` — the builder accepts both so runtime
    callers and tests need no full settings object.
    """
    if settings is None:
        return None
    memory = getattr(settings, "memory", None)
    if memory is not None and hasattr(memory, "memory_enabled"):
        return memory
    if hasattr(settings, "memory_enabled"):
        return settings
    return None


def resolve_memory_enabled(settings: Any = None) -> bool:
    """Resolve the master memory switch: env override, then settings; default on.

    The env var wins at EVERY read (per ADR-036's opt-out requirement), so
    ``VICTOR_MEMORY_ENABLED=false`` suppresses injection even when the
    settings object says otherwise.

    Args:
        settings: Application settings or a bare ``MemorySettings`` (optional).

    Returns:
        Whether preference/episode memory is enabled at all.
    """
    env = _env_flag("VICTOR_MEMORY_ENABLED")
    if env is not None:
        return env
    source = _memory_settings_source(settings)
    if source is not None:
        return bool(getattr(source, "memory_enabled", True))
    return True


def resolve_memory_scope_enabled(settings: Any, scope: MemoryScope) -> bool:
    """Resolve the per-scope opt-out (settings only; no env var by design).

    The FEP's "opt-out per scope" is modeled as two settings flags
    (``user_scope_enabled`` / ``project_scope_enabled``) rather than a
    compound env grammar; the master ``VICTOR_MEMORY_ENABLED`` env override
    already suppresses both scopes at once.

    Args:
        settings: Application settings or a bare ``MemorySettings`` (optional).
        scope: The scope being considered for injection.

    Returns:
        Whether this scope may be queried and injected.
    """
    if not resolve_memory_enabled(settings):
        return False
    source = _memory_settings_source(settings)
    if source is None:
        return True
    if scope is MemoryScope.USER:
        return bool(getattr(source, "user_scope_enabled", True))
    return bool(getattr(source, "project_scope_enabled", True))


@dataclass(frozen=True)
class MemoryInjectionBlock:
    """A bounded, budget-registered memory block ready for prompt assembly.

    Attributes:
        text: Rendered block text (header + entry lines, no trailing newline).
        token_estimate: Estimated token cost (repo-standard chars/4).
        entries: The surfaced memories (id, scope, kind, content, relevance).
        dropped_count: Lower-rank entries excluded by the hard cap.
        scopes: The scopes this block draws from.
    """

    text: str
    token_estimate: int
    entries: List[Dict[str, Any]] = field(default_factory=list)
    dropped_count: int = 0
    scopes: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for perception metadata and A/B reporting."""
        return {
            "token_estimate": self.token_estimate,
            "entry_count": len(self.entries),
            "dropped_count": self.dropped_count,
            "scopes": list(self.scopes),
            "entries": list(self.entries),
        }


def memory_prompt_suffix(block: Optional[MemoryInjectionBlock]) -> str:
    """Render the prompt fragment for a memory block (empty string when absent).

    This is the byte-exactness seam for ADR-036's opt-out negative test: with
    memory off, or an empty store, or a budget-refused block, the prompt is
    ``base + memory_prompt_suffix(None)`` — byte-identical to the pre-memory
    prompt.
    """
    if block is None or not block.text:
        return ""
    return block.text + "\n\n"


def _estimate_tokens(text: str) -> int:
    """Repo-standard token estimate (~4 chars per token)."""
    return max(1, len(text) // 4) if text else 0


def _scope_names(scopes: Sequence[Any]) -> Tuple[MemoryScope, ...]:
    """Normalize a mixed str/MemoryScope scope list (order preserved, deduped)."""
    resolved: List[MemoryScope] = []
    for raw in scopes:
        scope = raw if isinstance(raw, MemoryScope) else MemoryScope(str(raw))
        if scope not in resolved:
            resolved.append(scope)
    return tuple(resolved)


def memory_scopes_from_context(context: Optional[Dict[str, Any]]) -> Tuple[MemoryScope, ...]:
    """Resolve the session's memory scopes from its context dict.

    Reads ``context["memory_scopes"]`` (a sequence) or ``context["memory_scope"]``
    (single value); the default is both scopes — a project session legitimately
    recalls user preferences plus project facts, and each scope is gated by its
    own opt-out flag. Passing an explicit (possibly single-element) list is how
    a caller pins the session to one scope; a scoped block never contains the
    other scope's rows (physical database separation in the A2a store).
    """
    if not context:
        return (MemoryScope.USER, MemoryScope.PROJECT)
    raw = context.get("memory_scopes")
    if raw is None:
        single = context.get("memory_scope")
        if single is None:
            return (MemoryScope.USER, MemoryScope.PROJECT)
        raw = [single]
    if isinstance(raw, (str, MemoryScope)):
        raw = [raw]
    return _scope_names(list(raw))


class MemoryBlockBuilder:
    """Assembles the bounded PERCEIVE memory block from the memory coordinator.

    Args:
        coordinator: The federated memory coordinator (register the A2a
            ``PreferenceEpisodicAdapter`` lanes on it).
        settings: Application settings (optional; supplies the injection cap
            and per-scope opt-outs; env still overrides the master switch).
        max_tokens: Hard cap override (beats settings; tests / strict policies).
        max_entries: Maximum memories considered per scope query.
        budget_manager: When provided, the block registers its token estimate
            through ``register_context_block`` and is refit/dropped to the
            remaining allowance instead of free-riding.
    """

    def __init__(
        self,
        coordinator: "UnifiedMemoryCoordinator",
        *,
        settings: Any = None,
        max_tokens: Optional[int] = None,
        max_entries: int = 8,
        budget_manager: Optional["BudgetManager"] = None,
    ):
        self.coordinator = coordinator
        self.settings = settings
        self.max_entries = max(1, int(max_entries))
        self.budget_manager = budget_manager
        if max_tokens is not None:
            self.max_tokens = max(MIN_BLOCK_TOKENS, int(max_tokens))
        else:
            source = _memory_settings_source(settings)
            configured = int(getattr(source, "memory_injection_max_tokens", 0) or 0)
            self.max_tokens = max(
                MIN_BLOCK_TOKENS, configured or DEFAULT_MEMORY_INJECTION_MAX_TOKENS
            )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_environment(
        cls,
        *,
        settings: Any = None,
        budget_manager: Optional["BudgetManager"] = None,
        project_path: Any = None,
        max_entries: int = 8,
    ) -> "MemoryBlockBuilder":
        """Build a coordinator wired to the default databases and both A2a lanes.

        Production wiring path: the store uses the canonical ``get_database()`` /
        ``get_project_database(project_path)`` singletons (the two-database
        model), so user-scope preferences live in the global db and
        project-scope preferences/episodes in the project db.
        """
        from victor.storage.memory.adapters import create_preference_episodic_adapters
        from victor.storage.memory.preference_episodic import PreferenceEpisodicStore
        from victor.storage.memory.unified import UnifiedMemoryCoordinator

        store = PreferenceEpisodicStore(project_path=project_path)
        coordinator = UnifiedMemoryCoordinator(enable_memory_evolution=False)
        for adapter in create_preference_episodic_adapters(store):
            coordinator.register_provider(adapter)
        return cls(
            coordinator,
            settings=settings,
            max_entries=max_entries,
            budget_manager=budget_manager,
        )

    # ------------------------------------------------------------------
    # Block assembly
    # ------------------------------------------------------------------

    async def build(
        self,
        query: str,
        scopes: Sequence[MemoryScope] = (MemoryScope.USER, MemoryScope.PROJECT),
    ) -> Optional[MemoryInjectionBlock]:
        """Assemble the memory block for a session, or None when nothing to inject.

        Returns None (zero observable change) when: the master switch or env
        override disables memory, no requested scope is enabled, the store has
        no matching rows, or the budget allowance cannot fit a minimal block.
        """
        # Flag + env re-resolved at every read (ADR-036 opt-out semantics).
        if not resolve_memory_enabled(self.settings):
            return None

        active_scopes = tuple(
            scope
            for scope in _scope_names(scopes)
            if resolve_memory_scope_enabled(self.settings, scope)
        )
        if not active_scopes:
            return None

        results: List[Tuple[MemoryScope, "MemoryResult"]] = []
        for scope in active_scopes:
            try:
                scoped = await self.coordinator.search_all(
                    query=query,
                    limit=self.max_entries,
                    memory_types=[MemoryType.PREFERENCE, MemoryType.EPISODE],
                    filters={"scope": scope.value},
                )
            except Exception as e:  # noqa: BLE001 — memory must never break PERCEIVE
                logger.warning("Memory injection query failed (scope=%s): %s", scope.value, e)
                continue
            for result in scoped:
                results.append((scope, result))

        if not results:
            return None  # off-when-empty: nothing surfaced, nothing registered

        # Cross-scope defense in depth: the store's physical database
        # separation already isolates scopes; drop any result whose metadata
        # disagrees with the lane it was fetched for.
        results = [
            (scope, result)
            for scope, result in results
            if not result.metadata or result.metadata.get("scope", scope.value) == scope.value
        ]
        if not results:
            return None

        return self._fit_block(results, active_scopes)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fit_block(
        self,
        results: List[Tuple[MemoryScope, "MemoryResult"]],
        active_scopes: Tuple[MemoryScope, ...],
    ) -> Optional[MemoryInjectionBlock]:
        """Rank-merge, cap, budget-register, and render the block."""
        # Rank-merge across scopes: the coordinator already hybrid-ranked each
        # scoped query; merge by relevance (stable on input order for ties,
        # which keeps the caller's scope precedence).
        ranked = sorted(results, key=lambda pair: pair[1].relevance, reverse=True)

        cap = self.max_tokens
        if self.budget_manager is not None:
            remaining = int(self.budget_manager.remaining_context_block_tokens())
            cap = min(cap, remaining)
            if cap < MIN_BLOCK_TOKENS:
                # Context is saturated with higher-priority blocks: memory
                # yields rather than displacing them at any cost.
                logger.debug("Memory block skipped: context-block allowance exhausted")
                return None

        lines = [MEMORY_BLOCK_HEADER]
        entries: List[Dict[str, Any]] = []
        dropped = 0
        for scope, result in ranked:
            content = " ".join(str(result.content).split())
            if not content:
                continue
            truncated = False
            if len(content) > MAX_ENTRY_CHARS:
                content = content[:MAX_ENTRY_CHARS]
                truncated = True
            line = f"- ({scope.value}) {content}"
            candidate = lines + [line]
            if _estimate_tokens("\n".join(candidate)) > cap:
                dropped += 1
                continue
            lines.append(line)
            entries.append(
                {
                    "id": result.id,
                    "scope": scope.value,
                    "kind": result.metadata.get("kind", ""),
                    "content": content,
                    "relevance": round(float(result.relevance), 4),
                    "truncated": truncated,
                }
            )

        if len(lines) <= 1:
            return None  # every entry overflowed even the header-only block

        text = "\n".join(lines)
        token_estimate = _estimate_tokens(text)

        if self.budget_manager is not None:
            registered = self.budget_manager.register_context_block(
                MEMORY_BUDGET_BLOCK_NAME, token_estimate, priority=MEMORY_BLOCK_PRIORITY
            )
            if not registered:
                # Refused even after refit (e.g. a concurrent registration);
                # drop the block — never an unregistered injection.
                logger.debug("Memory block refused by budget manager; dropping")
                return None

        return MemoryInjectionBlock(
            text=text,
            token_estimate=token_estimate,
            entries=entries,
            dropped_count=dropped,
            scopes=tuple(scope.value for scope in active_scopes),
        )


def bind_production_memory_injector(
    perception: Any,
    *,
    settings: Any = None,
    project_path: Any = None,
    budget_manager: Optional["BudgetManager"] = None,
) -> Optional["MemoryBlockBuilder"]:
    """Bind a production ``MemoryBlockBuilder`` onto a ``PerceptionIntegration``.

    This is the production wiring seam for FEP-0048 A2c (FOU-A2c) — and the
    corrective for the FEP-0041 dormancy class: PR #1549 shipped a hook no
    chat-loop caller ever invoked, so real sessions silently got no behavior.
    The binding MUST happen at the composition root (``AgentOrchestrator`` for
    chat), not inside the collaborator, or the hook ships unwired again.

    Semantics:

    - **Skip when off**: ``memory_enabled`` (settings) and
      ``VICTOR_MEMORY_ENABLED`` (env) are resolved BEFORE any store is opened,
      so an opted-out session pays nothing.
    - **Degrade, never fail**: any construction failure (store unreachable,
      migration error) logs a warning and returns ``None`` — session start is
      never broken by memory.
    - **Off-when-empty downstream**: even when bound, an empty store yields no
      block (``MemoryBlockBuilder.build`` returns ``None``), leaving the
      prompt byte-identical.

    Args:
        perception: The ``PerceptionIntegration`` serving PERCEIVE for a
            session (bound via its ``set_memory_injector``).
        settings: Application settings (drives ``memory_enabled``, per-scope
            opt-outs, and the injection token cap).
        project_path: Project root for the project-scope database (defaults
            to the process working directory, matching production sessions).
        budget_manager: Optional ``BudgetManager`` so the block competes for
            context tokens instead of free-riding.

    Returns:
        The bound builder, or ``None`` when memory is disabled or binding
        failed.
    """
    try:
        if not resolve_memory_enabled(settings):
            logger.debug("Memory injection disabled (settings/env); binding skipped")
            return None
        builder = MemoryBlockBuilder.from_environment(
            settings=settings,
            budget_manager=budget_manager,
            project_path=project_path,
        )
        perception.set_memory_injector(builder)
        logger.info(
            "PERCEIVE memory injection bound (max_tokens=%d, entries=%d)",
            builder.max_tokens,
            builder.max_entries,
        )
        return builder
    except Exception as exc:  # noqa: BLE001 — memory must never break session start
        logger.warning("Memory injection unavailable; continuing without it: %s", exc)
        return None


__all__ = [
    "DEFAULT_MEMORY_INJECTION_MAX_TOKENS",
    "MEMORY_BLOCK_HEADER",
    "MEMORY_BLOCK_PRIORITY",
    "MemoryBlockBuilder",
    "MemoryInjectionBlock",
    "MAX_ENTRY_CHARS",
    "MIN_BLOCK_TOKENS",
    "bind_production_memory_injector",
    "memory_prompt_suffix",
    "memory_scopes_from_context",
    "resolve_memory_enabled",
    "resolve_memory_scope_enabled",
]


def bind_orchestrator_memory_injector(
    orchestrator: Any, settings: Any = None
) -> Optional["MemoryBlockBuilder"]:
    """Orchestrator-flavored production binding (FEP-0048 A2c wiring).

    Navigates the orchestrator's own collaborators so the chat composition
    root stays within its hotspot line ratchet: one import + one call there.
    Skips when disabled (resolved before any store opens), degrades to
    no-injection on failure — memory never breaks session start.
    """
    runtime_intelligence = getattr(orchestrator, "_runtime_intelligence", None)
    perception = getattr(runtime_intelligence, "perception_integration", None)
    if perception is None:
        return None
    return bind_production_memory_injector(
        perception, settings=settings or getattr(orchestrator, "settings", None)
    )
