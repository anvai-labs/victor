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

"""FEP-0020 Phase 2 — sandhi usage-gateway bridge.

Exercises the ``victor[sandhi]`` optional dependency. Skipped entirely when
``sandhi-gateway`` is not installed (the base install), which is the documented
default-off posture.
"""

from __future__ import annotations

import pytest

pytest.importorskip("sandhi_gateway", reason="requires the victor[sandhi] extra")

from victor.agent.session_cost_tracker import SessionCostTracker  # noqa: E402
from victor.observability.sandhi_meter import SandhiMeter, sandhi_available  # noqa: E402


def test_sandhi_available_true_with_extra() -> None:
    assert sandhi_available() is True


def test_record_emits_neutral_event_with_attribution() -> None:
    meter = SandhiMeter()
    meter.record(
        provider="anthropic",
        model="claude-3-5-sonnet",
        prompt_tokens=1000,
        completion_tokens=500,
        cache_read_tokens=200,
        cache_write_tokens=64,
        subject_id="alice",
        group_id="team-a",
        session_id="sess-1",
    )

    events = meter.events()
    assert len(events) == 1
    ev = events[0]
    # Attribution — the gap Victor's session-scoped records could not fill.
    # Finding 23: the billing group is part of the (versioned) metering key,
    # so the event's group comes from the key minted for THIS event's group.
    assert ev["subject_id"] == "alice"
    assert ev["group_id"] == "team-a"
    assert ev["session_id"] == "sess-1"
    assert ev["virtual_key_id"] == "victor:alice:team-a"
    # Provider units + the ADR-0047 D4 prompt-cache split.
    assert ev["provider"] == "anthropic"
    assert ev["tokens_in"] == 1000
    assert ev["tokens_out"] == 500
    assert ev["cache_read_tokens"] == 200
    assert ev["cache_creation_tokens"] == 64
    # Neutral contract — units only, never dollars.
    assert "cost" not in ev and "usd" not in ev
    assert meter.wire_contract_version == "1"


def test_subject_reuses_one_virtual_key_within_a_group() -> None:
    meter = SandhiMeter()
    for _ in range(3):
        meter.record(
            provider="openai",
            model="gpt-4o",
            prompt_tokens=10,
            completion_tokens=5,
            subject_id="bob",
            group_id="team-b",
        )
    events = meter.events()
    assert len(events) == 3
    assert {e["virtual_key_id"] for e in events} == {"victor:bob:team-b"}


def test_anonymous_subject_defaults() -> None:
    meter = SandhiMeter()
    meter.record(provider="ollama", model="llama3", prompt_tokens=1, completion_tokens=1)
    ev = meter.events()[0]
    assert ev["virtual_key_id"] == "victor:anonymous"
    assert ev["subject_id"] == "anonymous"


class TestBillingGroupRule:
    """Finding 23 (Codex wave 2): the billing-group rule.

    The stable principal key is the subject (``principal_id`` — the legacy key
    label for key-only principals, the issuer-qualified identity for OIDC once
    A4c lands). The billing GROUP is recorded per event: it is realized as the
    versioned metering key ``victor:{subject}:{group}`` because the gateway's
    meter path is key-authoritative (ADR-0004 D4 — ``meter_tokens`` resolves
    the event's ``group_id`` from the virtual key; there is no per-event group
    parameter). A group change meters through a NEW key, so subsequent events
    carry the new group while historical events keep the group they were
    recorded with.
    """

    def test_group_change_is_reflected_in_subsequent_events(self) -> None:
        meter = SandhiMeter()
        meter.record(
            provider="openai",
            model="gpt-4o",
            prompt_tokens=10,
            completion_tokens=5,
            subject_id="alice",
            group_id="team-a",
        )
        # The registry moved alice to team-b; the NEXT event must charge team-b.
        meter.record(
            provider="openai",
            model="gpt-4o",
            prompt_tokens=10,
            completion_tokens=5,
            subject_id="alice",
            group_id="team-b",
        )

        events = meter.events()
        assert [e["group_id"] for e in events] == ["team-a", "team-b"]
        # Subject stays stable across the group change (D5 stable principal key).
        assert {e["subject_id"] for e in events} == {"alice"}
        # And historical events keep their original key/group attribution.
        assert [e["virtual_key_id"] for e in events] == [
            "victor:alice:team-a",
            "victor:alice:team-b",
        ]

    def test_returning_to_a_prior_group_reuses_its_key(self) -> None:
        meter = SandhiMeter()
        for group in ("team-a", "team-b", "team-a"):
            meter.record(
                provider="openai",
                model="gpt-4o",
                prompt_tokens=1,
                completion_tokens=1,
                subject_id="alice",
                group_id=group,
            )
        keys = [e["virtual_key_id"] for e in meter.events()]
        assert keys == [
            "victor:alice:team-a",
            "victor:alice:team-b",
            "victor:alice:team-a",
        ]
        assert meter._known_keys == {
            "victor:alice:team-a",
            "victor:alice:team-b",
        }

    def test_legacy_key_only_flow_is_behavior_neutral(self) -> None:
        """No verified group (registry-less API key) — single unversioned key,
        event group None, exactly the pre-finding-23 key-only shape."""
        meter = SandhiMeter()
        for _ in range(2):
            meter.record(
                provider="openai",
                model="gpt-4o",
                prompt_tokens=1,
                completion_tokens=1,
                subject_id="dave",
                group_id=None,
            )
        events = meter.events()
        assert [e["group_id"] for e in events] == [None, None]
        assert {e["virtual_key_id"] for e in events} == {"victor:dave"}

    def test_overlapping_http_sse_ws_calls_carry_per_event_group(self) -> None:
        """Overlapping HTTP + SSE + WS calls on one shared session tracker:
        each recorded event carries the group bound for ITS call, even though
        the calls interleave on the same tracker and meter."""
        import asyncio

        from victor.core.context import bind_attribution

        meter = SandhiMeter()
        tracker = SessionCostTracker(
            provider="openai",
            model="gpt-4o",
            subject_id="alice",
            _sandhi=meter,
        )
        groups = {"http": "team-a", "sse": "team-b", "ws": "team-c"}

        async def one_channel(channel: str, group: str) -> None:
            # Each transport binds its own principal attribution (the HTTP
            # auth dependency / the /ws turn handler do this at the seam).
            with bind_attribution(subject_id="alice", group_id=group):
                for _ in range(3):
                    tracker.record_request(prompt_tokens=10, completion_tokens=5)
                await asyncio.sleep(0)

        async def overlap() -> None:
            await asyncio.gather(*(one_channel(c, g) for c, g in groups.items()))

        asyncio.run(overlap())

        events = meter.events()
        assert len(events) == 9
        by_group: dict[str, list[dict]] = {}
        for ev in events:
            by_group.setdefault(ev["group_id"], []).append(ev)
        assert set(by_group) == set(groups.values())
        for channel, group in groups.items():
            key = f"victor:alice:{group}"
            assert len(by_group[group]) == 3
            assert all(ev["virtual_key_id"] == key for ev in by_group[group])


def test_session_cost_tracker_mirrors_into_sandhi() -> None:
    """The wire: a SessionCostTracker with a meter attached emits on record_request."""
    meter = SandhiMeter()
    tracker = SessionCostTracker(
        provider="anthropic",
        model="claude-3-5-sonnet",
        subject_id="carol",
        group_id="team-c",
        _sandhi=meter,
    )

    tracker.record_request(prompt_tokens=800, completion_tokens=300, cache_read_tokens=100)
    tracker.record_request(prompt_tokens=200, completion_tokens=90)

    events = meter.events()
    assert len(events) == 2
    assert all(e["subject_id"] == "carol" and e["group_id"] == "team-c" for e in events)
    assert events[0]["tokens_in"] == 800
    assert events[0]["cache_read_tokens"] == 100
    assert events[0]["session_id"] == tracker.session_id
    # The tracker's own USD accounting is unchanged / independent of the mirror.
    assert tracker.total_prompt_tokens == 1000


def test_session_cost_tracker_without_meter_is_unaffected() -> None:
    """Default (no meter) — no gateway interaction, historical behavior intact."""
    tracker = SessionCostTracker(provider="openai", model="gpt-4o")
    rc = tracker.record_request(prompt_tokens=50, completion_tokens=25)
    assert rc.prompt_tokens == 50
    assert tracker.total_tokens == 75


def test_record_never_raises_on_bad_gateway() -> None:
    """Best-effort contract — a broken gateway must not propagate into the caller."""
    meter = SandhiMeter()

    class _Boom:
        def add_virtual_key(self, *a: object, **k: object) -> None:
            raise RuntimeError("boom")

    meter._gw = _Boom()  # type: ignore[assignment]
    meter._known_keys.clear()
    # Must swallow the error, not raise.
    meter.record(provider="openai", model="gpt-4o", prompt_tokens=1, completion_tokens=1)
