# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""FEP-0010 Phase 4: the protocol-crate types are exposed via victor_native and
round-trip across the FFI unchanged (Python <-> Rust <-> Python).

The Python orchestrator still owns its own dataclasses; these victor_native.*
types are the shared FFI agreement so cloud (Python) and edge (Rust) agree on
message/tool-call/stream shape."""

import json

import pytest

pytest.importorskip("victor_native")
import victor_native as vn  # noqa: E402


def test_role_variants_and_equality():
    # Variants are accessible and comparable (eq_int).
    assert vn.Role.User != vn.Role.Assistant
    assert vn.Role.System.as_str() == "system"
    assert vn.Role.Tool.as_str() == "tool"


def test_message_basic_roundtrip():
    m = vn.Message(vn.Role.User, content="hello")
    assert m.role == vn.Role.User
    assert m.content == "hello"
    assert m.tool_calls is None
    echoed = vn.echo_message(m)
    assert echoed.content == "hello"
    assert echoed.role == vn.Role.User


def test_message_with_tool_calls_roundtrip():
    tc = vn.ToolCall("call_1", "read_file", '{"path": "/tmp/x"}')
    # arguments round-trip as a JSON *value* (serde_json canonicalizes whitespace).
    assert json.loads(tc.arguments) == {"path": "/tmp/x"}

    ma = vn.Message(vn.Role.Assistant, content=None, tool_calls=[tc])
    echoed = vn.echo_message(ma)
    calls = echoed.tool_calls
    assert len(calls) == 1
    assert calls[0].id == "call_1"
    assert calls[0].name == "read_file"
    assert json.loads(calls[0].arguments) == {"path": "/tmp/x"}


def test_tool_message_roundtrip():
    m = vn.Message(vn.Role.Tool, content="result", tool_call_id="call_1")
    echoed = vn.echo_message(m)
    assert echoed.tool_call_id == "call_1"
    assert echoed.role == vn.Role.Tool


def test_stream_chunk_roundtrip_with_usage():
    sc = vn.StreamChunk(content="hi", is_final=True, usage=vn.Usage(1, 2, 3))
    echoed = vn.echo_stream_chunk(sc)
    assert echoed.is_final is True
    assert echoed.content == "hi"
    u = echoed.usage
    assert (u.prompt_tokens, u.completion_tokens, u.total_tokens) == (1, 2, 3)


def test_invalid_tool_call_arguments_raises():
    with pytest.raises(Exception):
        vn.ToolCall("c", "t", "not json")
