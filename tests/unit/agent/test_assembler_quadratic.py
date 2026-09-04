# Copyright 2026 Vijaykumar Singh <vijaykumar@anvaiops.com>
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

"""Assembler selection-cost guarantees (co-design review U1-5).

The older-message selection used older_messages.index(msg) — an O(n)
value-equality scan per selected message (O(n²) per turn on long
histories) — and rescored the unchanged older prefix from scratch on
every assembly. These tests pin the index-map and per-turn raw-score
cache contracts, adversarially.
"""

from __future__ import annotations

import time

from victor.agent.conversation.assembler import TurnBoundaryContextAssembler
from victor.agent.session_ledger import SessionLedger
from victor.providers.base import Message


def _msg(role, content):
    return Message(role=role, content=content)


def _conversation(n_turns: int = 10):
    msgs = [_msg("system", "sys")]
    for i in range(n_turns):
        msgs.append(_msg("user", f"q{i}"))
        msgs.append(_msg("assistant", f"a{i}"))
    return msgs


def _score_spy(calls: list):
    """Score fn that records (prefix_len, query) and scores later msgs higher."""

    def score_fn(messages, query):
        calls.append((len(messages), query))
        # Later messages score higher, forcing a reorder-then-restore.
        return [(msg, float(i)) for i, msg in enumerate(messages)]

    return score_fn


class TestIndexMapSelection:
    def test_duplicate_content_messages_keep_object_order(self):
        """index(msg) used value-equality — with duplicate contents it
        returned the FIRST matching index for both objects. The id-keyed
        map must give each object its own position."""
        calls = []
        assembler = TurnBoundaryContextAssembler(
            session_ledger=SessionLedger(), score_fn=_score_spy(calls)
        )
        msgs = [_msg("system", "sys")]
        # Two turns with IDENTICAL content in the older region.
        for i in range(6):
            msgs.append(_msg("user", "same question"))
            msgs.append(_msg("assistant", "same answer"))
        for i in range(3):
            msgs.append(_msg("user", f"recent q{i}"))
            msgs.append(_msg("assistant", f"recent a{i}"))

        result = assembler.assemble(msgs, max_context_chars=100000, current_query="q")
        contents = [m.content for m in result]
        # Whatever was selected, ordering must be non-decreasing by original
        # position — verified by reconstructing positions from the input.
        positions = {id(m): i for i, m in enumerate(msgs)}
        result_positions = [positions[id(m)] for m in result]
        assert result_positions == sorted(result_positions)

    def test_selection_is_linear_time_smoke(self):
        """Negative perf pin: 10k-message assembly must stay well under a
        quadratic budget (~seconds), not minutes."""
        calls = []
        assembler = TurnBoundaryContextAssembler(
            session_ledger=SessionLedger(), score_fn=_score_spy(calls)
        )
        msgs = _conversation(5000)  # 10k messages
        t0 = time.monotonic()
        assembler.assemble(msgs, max_context_chars=200_000, current_query="q")
        elapsed = time.monotonic() - t0
        assert elapsed < 5.0, f"assembly took {elapsed:.1f}s for 10k messages"


class TestScoreCache:
    def test_same_prefix_and_query_hits_cache(self):
        """Re-assembly with unchanged history (the per-iteration pattern)
        must not rescore the older prefix."""
        calls = []
        assembler = TurnBoundaryContextAssembler(
            session_ledger=SessionLedger(), score_fn=_score_spy(calls)
        )
        msgs = _conversation(10)
        first = assembler.assemble(msgs, max_context_chars=100000, current_query="q")
        n_calls = len(calls)
        assert n_calls == 1
        second = assembler.assemble(msgs, max_context_chars=100000, current_query="q")
        assert len(calls) == n_calls, "unchanged prefix+query must hit the cache"
        assert [id(m) for m in first] == [id(m) for m in second], "cache must not change selection"

    def test_query_change_misses(self):
        calls = []
        assembler = TurnBoundaryContextAssembler(
            session_ledger=SessionLedger(), score_fn=_score_spy(calls)
        )
        msgs = _conversation(10)
        assembler.assemble(msgs, max_context_chars=100000, current_query="alpha")
        assembler.assemble(msgs, max_context_chars=100000, current_query="beta")
        assert len(calls) == 2

    def test_prefix_growth_misses(self):
        """Messages moving from recent into older changes the prefix — recompute."""
        calls = []
        assembler = TurnBoundaryContextAssembler(
            session_ledger=SessionLedger(), score_fn=_score_spy(calls)
        )
        msgs = _conversation(10)
        assembler.assemble(msgs, max_context_chars=100000, current_query="q")
        # Append a full new turn; the older prefix grows by one turn.
        msgs.append(_msg("user", "new q"))
        msgs.append(_msg("assistant", "new a"))
        assembler.assemble(msgs, max_context_chars=100000, current_query="q")
        assert len(calls) == 2

    def test_cache_not_corrupted_by_sort(self):
        """The cached raw list must never be mutated by downstream
        sort/multiplier steps — second hit yields the same selection."""
        calls = []
        assembler = TurnBoundaryContextAssembler(
            session_ledger=SessionLedger(), score_fn=_score_spy(calls)
        )
        msgs = _conversation(10)
        r1 = assembler.assemble(msgs, max_context_chars=100000, current_query="q")
        r2 = assembler.assemble(msgs, max_context_chars=100000, current_query="q")
        r3 = assembler.assemble(msgs, max_context_chars=100000, current_query="q")
        assert [id(m) for m in r1] == [id(m) for m in r2] == [id(m) for m in r3]
        # Raw cached scores are still in ORIGINAL order for the next turn's
        # multiplier application (enumerate-order contract of score_fn).
        cached = assembler._score_cache_value
        cached_pairs = [(m.content, s) for m, s in cached]
        non_system = [m for m in msgs if m.role != "system"]
        expected = [(m.content, float(i)) for i, m in enumerate(non_system)][: len(cached)]
        assert cached_pairs == expected
