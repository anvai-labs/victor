# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""FEP-0012 Phase 2a: decision records carry the correlation spine + provenance
so each decision is joinable to its outcome for reward-weighted training."""

import json
import os

import pytest

import victor.agent.decisions.chain as chain
from victor.core import context


def _read_last(tmp_home: str) -> dict:
    path = f"{tmp_home}/.victor/logs/decisions.jsonl"
    with open(path) as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    return json.loads(lines[-1])


def test_log_decision_stamps_correlation_spine(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    token_s = context.set_session_id("sess-123")
    token_t = context.set_turn_id("turn-456")
    try:
        chain.log_decision(
            "task_type_classification",
            {"message_excerpt": "fix the login bug"},
            "debug",
            "local_classifier",
            0.8,
        )
    finally:
        context.session_id.reset(token_s)
        context.turn_id.reset(token_t)

    rec = _read_last(str(tmp_path))
    assert rec["session_id"] == "sess-123"
    assert rec["turn_id"] == "turn-456"
    assert rec["trace_id"] != ""  # spine accessor returns "" if unset, but set here
    assert rec["decision_id"] and len(rec["decision_id"]) == 32  # uuid4 hex
    assert rec["type"] == "task_type_classification"
    assert rec["source"] == "local_classifier"


def test_log_decision_provenance_fields_optional(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    # Without provenance kwargs → fields omitted (heuristic/LLM decisions).
    chain.log_decision("stage_detection", {"k": "v"}, "exploration", "heuristic", 0.3)
    rec = _read_last(str(tmp_path))
    assert "model_version" not in rec
    assert "feature_spec_version" not in rec
    assert "feature_digest" not in rec

    # With provenance kwargs → included (local classifier decisions).
    chain.log_decision(
        "task_completion",
        {"response_tail": "done"},
        "complete",
        "local_classifier",
        0.9,
        model_version="edge_v1",
        feature_spec_version="1",
        feature_digest="deadbeef",
    )
    rec = _read_last(str(tmp_path))
    assert rec["model_version"] == "edge_v1"
    assert rec["feature_spec_version"] == "1"
    assert rec["feature_digest"] == "deadbeef"


def test_log_decision_never_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    # Even with an un-serializable context, logging must not raise.
    chain.log_decision("x", object(), "y", "heuristic")  # default=str handles it
