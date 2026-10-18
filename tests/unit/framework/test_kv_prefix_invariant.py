"""KV prefix-stability invariant guard.

Within a user turn, the serialized provider-bound payload of every call must
extend the previous call's bytes (append-only) so provider prefix caching
holds. Divergence is sanctioned only at boundaries (user turn, compaction,
tool-set change, system refresh, model switch, resume).

Production evidence (2026-10-18, zai glm-5.3): cache hits frozen at 6528
tokens while input grew 2685→56762 — the mutators below reproduce that.

This guard lands ``xfail(strict=True)`` with the observability PR and is
flipped to a hard assert when the stabilization train completes: an accidental
early fix xpasses and fails the suite, forcing the flip.
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock, patch

import pytest

from victor.agent.conversation.assembler import TurnBoundaryContextAssembler
from victor.agent.conversation.scoring import CONTROLLER_WEIGHTS, score_messages
from victor.agent.conversation.types import ConversationMessage
from victor.agent.kv_prefix_observability import serialize_payload
from victor.agent.orchestrator import AgentOrchestrator
from victor.agent.session_ledger import SessionLedger
from victor.config.settings import ProfileConfig, Settings
from victor.providers.base import Message


def _canonical_score_fn(messages, current_query=None):
    conv = [ConversationMessage.from_provider_message(m) for m in messages]
    scored = score_messages(conv, current_query, weights=CONTROLLER_WEIGHTS)
    mapping = {id(cm): msg for cm, msg in zip(conv, messages)}
    return [(mapping[id(cm)], s) for cm, s in scored]


def _make_orchestrator() -> AgentOrchestrator:
    settings = Settings()
    settings.load_profiles = lambda: {
        "default": ProfileConfig(provider="fake", model="fake", temperature=0.7, max_tokens=2048)
    }
    settings.tools.use_semantic_tool_selection = False
    settings.mcp.mcp_enabled = False
    settings.analytics_log_file = None
    settings.analytics_enabled = False
    settings.load_tool_config = lambda: {}
    provider = MagicMock()
    provider.name = "fake"
    with patch("victor.core.bootstrap_services.bootstrap_new_services"):
        orch = AgentOrchestrator(settings=settings, provider=provider, model="fake")
    orch._session_ledger = SessionLedger()
    orch._context_assembler = TurnBoundaryContextAssembler(
        session_ledger=orch._session_ledger,
        score_fn=_canonical_score_fn,
        conversation_controller=None,
        tool_result_deduplicator=None,
    )
    return orch


def _seed_history(orch: AgentOrchestrator, turns: int = 4) -> None:
    """Seed enough genuine user turns to engage the older-selection path."""
    orch.conversation.add_user_message("explore the codebase and list the modules")
    orch.conversation.add_message("assistant", "summary of modules")
    for i in range(turns):
        orch.conversation.add_user_message(f"turn {i}: read src/file{i}.py and report imports")
        orch.conversation.add_message(
            "assistant",
            f"reading file{i}",
            tool_calls=[
                {
                    "id": f"call_seed_{i}",
                    "type": "function",
                    "function": {"name": "read", "arguments": f'{{"path": "src/file{i}.py"}}'},
                }
            ],
        )
        orch.conversation.add_message(
            "tool",
            f"TOOL_OUTPUT read src/file{i}.py … def f{i}(): imports os, sys",
            tool_call_id=f"call_seed_{i}",
        )
        orch.conversation.add_message("assistant", f"file{i} imports os and sys")


def _drive_intra_loop(orch: AgentOrchestrator, iterations: int) -> List[bytes]:
    """One user turn; each iteration = one provider call + one loop step."""
    payloads: List[bytes] = []
    for i in range(iterations):
        assembled = orch.get_assembled_messages(current_query="agentic task")
        payloads.append(serialize_payload(assembled))
        # Simulate the agentic loop between provider calls — mirrors
        # AgenticLoop._populate_session_ledger (agentic_loop.py:1624,1628).
        orch.conversation.add_message(
            "assistant",
            f"plan {i}",
            tool_calls=[
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                }
            ],
        )
        orch.conversation.add_message(
            "tool",
            f"TOOL_OUTPUT read src/file{i}.py … {i} lines: def f{i}(): pass",
            tool_call_id=f"call_{i}",
        )
        orch._session_ledger.record_file_read(
            f"src/file{i}.py", f"{i} lines: def f{i}(): pass", turn_index=i
        )
        if i == 2:
            # In-place entry rewrite (resolved pending action).
            orch._session_ledger.record_pending_action("deploy", 0)
            orch._session_ledger.resolve_pending_action("deploy")
        if i == 3:
            # Duplicate tool output arrival (dedup view pressure).
            orch.conversation.add_message(
                "assistant",
                "re-read",
                tool_calls=[
                    {
                        "id": f"call_dup_{i}",
                        "type": "function",
                        "function": {"name": "read", "arguments": "{}"},
                    }
                ],
            )
            orch.conversation.add_message(
                "tool",
                f"TOOL_OUTPUT read src/file{i}.py … {i} lines: def f{i}(): pass",
                tool_call_id=f"call_dup_{i}",
            )
    return payloads


def _assert_append_only(payloads: List[bytes]) -> None:
    for previous, current in zip(payloads, payloads[1:]):
        assert current.startswith(previous), (
            "provider payload mutated already-sent bytes intra-turn "
            f"(sizes {len(previous)} → {len(current)}); prefix caching is defeated"
        )


@pytest.mark.xfail(
    strict=True,
    reason="intra-turn payload mutators (ledger re-render, dedup stubbing of sent "
    "messages, assembler re-selection) — fixed by the KV prefix stabilization train",
)
async def test_intra_turn_payload_is_append_only():
    orch = _make_orchestrator()
    _seed_history(orch)
    payloads = _drive_intra_loop(orch, 6)
    assert len(payloads) == 6
    _assert_append_only(payloads)


@pytest.mark.xfail(
    strict=True,
    reason="intra-turn payload mutators — fixed by the KV prefix stabilization train",
)
async def test_user_turn_boundary_is_the_only_sanctioned_divergence():
    orch = _make_orchestrator()
    _seed_history(orch)
    first_turn = _drive_intra_loop(orch, 4)
    _assert_append_only(first_turn)

    orch.conversation.add_user_message("next task: verify the changes")
    second_turn = _drive_intra_loop(orch, 4)
    # A boundary divergence (ledger re-render + new selection) is sanctioned
    # exactly once; each turn is itself append-only.
    _assert_append_only(second_turn)
