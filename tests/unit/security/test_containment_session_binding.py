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

"""Session-owned containment binding tests (ADR-037, Codex review fix wave 1).

Finding 1: the canary must be BOUND to the serving session — the session
owner holds one stable canary, binds it around every turn, and restores the
binding on exit, so dispatch-time containment actually has a canary to find
on a real session. Finding 16: flagged-family evidence lives in the same
session-owned state (bounded), rebinds per turn, and never leaks across
sessions or into a fresh request task.
"""

import asyncio
from types import SimpleNamespace

import pytest

from victor.security.containment import (
    MAX_FLAGGED_FAMILIES,
    SessionContainmentState,
    bind_containment_state,
    current_containment_state,
    current_session_canary,
    reset_session_canary,
    set_session_canary,
    wrap_untrusted,
)
from victor.security.injection_screen import (
    current_flagged_families,
    record_flagged_families,
    reset_flagged_families,
)


@pytest.fixture(autouse=True)
def _clean_bindings():
    reset_session_canary()
    reset_flagged_families()
    yield
    reset_session_canary()
    reset_flagged_families()


class TestSessionContainmentState:
    def test_canary_is_stable_for_the_state_lifetime(self):
        state = SessionContainmentState(session_id="s1")
        assert state.canary == state.canary
        assert state.canary.startswith("victor-canary-")

    def test_distinct_sessions_get_distinct_canaries(self):
        a = SessionContainmentState(session_id="s1")
        b = SessionContainmentState(session_id="s2")
        assert a.canary != b.canary

    def test_flagged_family_evidence_is_bounded(self):
        state = SessionContainmentState(session_id="s1")
        state.record_flagged_families(tuple(f"fam_{i}" for i in range(MAX_FLAGGED_FAMILIES + 10)))
        assert len(state.flagged_families) == MAX_FLAGGED_FAMILIES

    def test_clear_flagged_families(self):
        state = SessionContainmentState(session_id="s1")
        state.record_flagged_families(("fam",))
        state.clear_flagged_families()
        assert state.flagged_families == frozenset()


class TestContainmentBinding:
    def test_binding_publishes_the_session_canary(self):
        state = SessionContainmentState(session_id="s1")
        with bind_containment_state(state):
            assert current_session_canary() == state.canary
            assert current_containment_state() is state
        assert current_session_canary() is None

    def test_binding_restores_the_previous_override_on_exit(self):
        set_session_canary("victor-canary-ffffffffffffffff")
        state = SessionContainmentState(session_id="s1")
        with bind_containment_state(state):
            assert current_session_canary() == state.canary
        assert current_session_canary() == "victor-canary-ffffffffffffffff"

    def test_binding_restores_nested_state_on_exit(self):
        outer = SessionContainmentState(session_id="outer")
        inner = SessionContainmentState(session_id="inner")
        with bind_containment_state(outer):
            with bind_containment_state(inner):
                assert current_session_canary() == inner.canary
            assert current_session_canary() == outer.canary
        assert current_session_canary() is None

    def test_fresh_request_task_does_not_inherit_the_binding(self):
        state = SessionContainmentState(session_id="s1")
        observed = {}

        async def turn():
            with bind_containment_state(state):
                await asyncio.sleep(0)
                observed["in_turn"] = current_session_canary()
            observed["after_turn"] = current_session_canary()

        async def fresh_request_task():
            await asyncio.gather(turn())
            observed["fresh_task"] = current_session_canary()

        asyncio.run(fresh_request_task())
        assert observed["in_turn"] == state.canary
        # After the bound turn exits — including from another task — no
        # session canary is visible: the binding was restored, not leaked.
        assert observed["after_turn"] is None
        assert observed["fresh_task"] is None


class TestEvidenceLifetime:
    def test_record_writes_through_to_the_bound_session_owner(self):
        state = SessionContainmentState(session_id="s1")
        with bind_containment_state(state):
            record_flagged_families(("instruction_impersonation",))
            assert current_flagged_families() == frozenset({"instruction_impersonation"})
            assert state.flagged_families == frozenset({"instruction_impersonation"})

    def test_evidence_survives_across_turns_of_the_same_session(self):
        state = SessionContainmentState(session_id="s1")
        with bind_containment_state(state):
            record_flagged_families(("instruction_impersonation",))
        # Next turn: the owner rebinds the same state — evidence recovered.
        with bind_containment_state(state):
            assert current_flagged_families() == frozenset({"instruction_impersonation"})

    def test_evidence_is_not_visible_outside_the_bound_session(self):
        state = SessionContainmentState(session_id="s1")
        with bind_containment_state(state):
            record_flagged_families(("instruction_impersonation",))
        assert current_flagged_families() == frozenset()

    def test_reset_clears_both_transport_and_bound_owner(self):
        state = SessionContainmentState(session_id="s1")
        with bind_containment_state(state):
            record_flagged_families(("instruction_impersonation",))
            reset_flagged_families()
            assert current_flagged_families() == frozenset()
            assert state.flagged_families == frozenset()


class TestRuntimeIdentityOwnership:
    def _identity(self, session_id="session-A"):
        from victor.agent.runtime.context import RuntimeIdentity

        state = SimpleNamespace(active_session_id=session_id, _memory_session_id=None)
        return RuntimeIdentity(state)

    def test_containment_state_is_stable_across_resolves(self):
        identity = self._identity()
        first = identity.containment_state()
        identity.resolve()
        assert identity.containment_state() is first

    def test_session_transition_mints_a_fresh_canary(self):
        identity = self._identity("session-A")
        first = identity.containment_state()
        identity._state.active_session_id = "session-B"
        second = identity.containment_state()
        assert second is not first
        assert second.session_id == "session-B"
        assert second.canary != first.canary

    def test_member_bind_gets_a_distinct_canary(self):
        from victor.agent.runtime.context import AgentRuntimeContext

        identity = self._identity("session-A")
        parent_state = identity.containment_state()
        identity.bind_turn_lock(asyncio.Lock())
        child = AgentRuntimeContext(
            agent_id="member",
            display_name="Member",
            role="researcher",
            session_id="session-A:team:member",
        )
        identity.bind_member(child)
        child_state = identity.containment_state()
        assert child_state is not parent_state
        assert child_state.canary != parent_state.canary

    def test_envelopes_use_the_bound_session_canary(self):
        from victor.security.containment import UntrustedSourceClass

        identity = self._identity("session-A")
        state = identity.containment_state()
        with bind_containment_state(state):
            enveloped = wrap_untrusted(
                "page body",
                source_class=UntrustedSourceClass.WEB,
                origin="https://example.test",
            )
        assert state.canary in enveloped
