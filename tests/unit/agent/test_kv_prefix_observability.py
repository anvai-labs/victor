"""Unit tests for KV prefix-stability observability."""

from __future__ import annotations

import logging

from victor.agent.kv_prefix_observability import (
    PrefixCallTracker,
    canonical_message_bytes,
    first_divergence,
    serialize_payload,
    tools_fingerprint,
)
from victor.providers.base import Message


def _msgs(*specs) -> list[Message]:
    out = []
    for role, content in specs:
        out.append(Message(role=role, content=content))
    return out


# ── canonical serialization ────────────────────────────────────────────


def test_canonical_bytes_exclude_metadata_and_images():
    a = Message(role="user", content="hi", metadata={"k": "v"})
    b = Message(role="user", content="hi")
    assert canonical_message_bytes(a) == canonical_message_bytes(b)


def test_canonical_bytes_stable_across_dict_ordering():
    a = Message(role="tool", content="out", tool_call_id="c1", name="n")
    b = Message(role="tool", name="n", tool_call_id="c1", content="out")
    assert canonical_message_bytes(a) == canonical_message_bytes(b)


def test_serialize_payload_is_length_prefixed_no_boundary_alias():
    # ("ab", "c") must not serialize identically to ("a", "bc")
    left = serialize_payload(_msgs(("user", "ab"), ("assistant", "c")))
    right = serialize_payload(_msgs(("user", "a"), ("assistant", "bc")))
    assert left != right


def test_serialize_payload_extends_on_append():
    base = _msgs(("system", "sys"), ("user", "hello"))
    grown = base + [Message(role="assistant", content="reply")]
    assert serialize_payload(grown).startswith(serialize_payload(base))


# ── divergence locator ─────────────────────────────────────────────────


def test_first_divergence_none_when_extended():
    assert first_divergence(b"abcdef", b"abcdefgh") is None


def test_first_divergence_finds_offset():
    assert first_divergence(b"abcdef", b"abcXef") == 3


def test_first_divergence_truncation_reports_min_length():
    assert first_divergence(b"abcdef", b"abc") == 3


# ── tools fingerprint ──────────────────────────────────────────────────


def test_tools_fingerprint_order_insensitive():
    assert tools_fingerprint(["read", "write", "shell"]) == tools_fingerprint(
        ["shell", "read", "write"]
    )


def test_tools_fingerprint_none_passthrough():
    assert tools_fingerprint(None) is None


# ── tracker classification ─────────────────────────────────────────────


def test_first_call_is_session_start():
    tracker = PrefixCallTracker()
    record = tracker.record(_msgs(("system", "sys"), ("user", "hi")))
    assert record.boundary == "session_start"
    assert record.prefix_extended is None


def test_append_only_call_is_extended_with_no_boundary():
    tracker = PrefixCallTracker()
    tracker.record(_msgs(("system", "sys"), ("user", "hi")))
    record = tracker.record(_msgs(("system", "sys"), ("user", "hi"), ("assistant", "reply")))
    assert record.prefix_extended is True
    assert record.boundary == "none"
    assert record.divergence_offset is None


def test_mid_message_mutation_reports_divergence_offset():
    tracker = PrefixCallTracker()
    first = _msgs(("system", "sys"), ("user", "hi"))
    tracker.record(first)
    # Mutate an already-sent message (the production failure mode).
    mutated = _msgs(("system", "sys"), ("user", "hi"))
    mutated[1] = Message(role="assistant", content="<SESSION_STATE>changed</SESSION_STATE>")
    record = tracker.record(mutated)
    assert record.prefix_extended is False
    assert record.divergence_offset is not None and record.divergence_offset > 0
    assert record.boundary == "none"


def test_announced_boundary_is_consumed_once():
    tracker = PrefixCallTracker()
    tracker.record(_msgs(("system", "sys"), ("user", "turn 1")))
    tracker.note_boundary("user_turn")
    diverging = _msgs(("system", "sys"), ("user", "turn 2"))
    record = tracker.record(diverging)
    assert record.boundary == "user_turn"
    # Next call extends cleanly: boundary back to none.
    follow = tracker.record(diverging + [Message(role="assistant", content="ok")])
    assert follow.boundary == "none"
    assert follow.prefix_extended is True


def test_tools_change_classified_as_tool_set_boundary():
    tracker = PrefixCallTracker()
    tracker.record(_msgs(("user", "hi")), tools=["read"])
    record = tracker.record(_msgs(("user", "hi")), tools=["read", "write"])
    assert record.boundary == "tool_set"


def test_message_count_shrink_classified_as_compaction_or_trim():
    tracker = PrefixCallTracker()
    tracker.record(_msgs(("user", "a"), ("assistant", "b"), ("user", "c")))
    record = tracker.record(_msgs(("user", "c")))
    assert record.boundary == "compaction_or_trim"


def test_log_line_contains_locator_fields(caplog):
    tracker = PrefixCallTracker()
    with caplog.at_level(logging.INFO, logger="victor.agent.kv_prefix_observability"):
        tracker.record(_msgs(("system", "sys"), ("user", "hi")))
        mutated = _msgs(("system", "sys"), ("user", "hi"))
        mutated[1] = Message(role="assistant", content="<SESSION_STATE>v2</SESSION_STATE>")
        tracker.record(mutated)
    divergent_lines = [r.message for r in caplog.records if "divergence=" in r.message]
    assert divergent_lines, "divergent call must log at INFO"
    assert "extended=false" in divergent_lines[0]
