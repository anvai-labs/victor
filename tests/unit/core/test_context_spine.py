# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the correlation spine in victor.core.context (R1)."""

from __future__ import annotations

from victor.core import context as ctx


def test_turn_id_set_and_get():
    tok = ctx.set_turn_id("turn-xyz")
    try:
        assert ctx.get_turn_id() == "turn-xyz"
    finally:
        ctx.turn_id.reset(tok)


def test_begin_turn_generates_fresh_id():
    tok = ctx.set_turn_id("")
    try:
        a = ctx.begin_turn()
        b = ctx.begin_turn()
        assert a and b and a != b  # fresh each call
        assert ctx.get_turn_id() == b
    finally:
        ctx.turn_id.reset(tok)


def test_request_id_set_and_get():
    tok = ctx.set_request_id("req-1")
    try:
        assert ctx.get_request_id() == "req-1"
    finally:
        ctx.request_id.reset(tok)


def test_get_correlation_omits_empty():
    s = ctx.set_session_id("sess-9")
    t = ctx.set_turn_id("turn-9")
    r = ctx.set_request_id("")  # empty -> omitted
    try:
        corr = ctx.get_correlation()
        assert corr == {"session_id": "sess-9", "turn_id": "turn-9"}
        assert "request_id" not in corr
    finally:
        ctx.session_id.reset(s)
        ctx.turn_id.reset(t)
        ctx.request_id.reset(r)


def test_get_correlation_empty_when_unset():
    s = ctx.set_session_id("")
    t = ctx.set_turn_id("")
    r = ctx.set_request_id("")
    try:
        assert ctx.get_correlation() == {}
    finally:
        ctx.session_id.reset(s)
        ctx.turn_id.reset(t)
        ctx.request_id.reset(r)
