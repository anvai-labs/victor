# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""Tests for independent-label EVR-6 evidence production."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from victor.evaluation.turn_auditor_evidence import (
    produce_evidence_payload,
    resolve_ollama_identity,
    warm_ollama_model,
)
from victor.evaluation.turn_auditor_eval import assess_evidence_payload
from victor.framework.per_turn_auditor import AuditSignal, AuditVerdict


def _label_pack() -> dict:
    return {
        "schema_version": 1,
        "corpus_id": "reviewed:v1",
        "cases": [
            {
                "task_id": "task-1",
                "family": "code-fix",
                "oracle_source": "review:alice:v1",
                "oracle_alarm_step": 1,
                "trace": {
                    "session_id": "session-1",
                    "benchmark": "real-agent",
                    "metadata": {"source": "held-out"},
                    "steps": [
                        {
                            "index": 0,
                            "role": "tool",
                            "status": "ok",
                            "effect": "grounded_claim",
                            "layer": "context_memory",
                            "tool_name": "read",
                            "summary": "read relevant implementation",
                        },
                        {
                            "index": 1,
                            "role": "assistant",
                            "status": "ok",
                            "effect": "none",
                            "layer": "lifecycle_orchestration",
                            "tool_name": "",
                            "summary": "clearly off-track",
                        },
                    ],
                },
            }
        ],
    }


@dataclass
class _Auditor:
    states: list[dict]
    action_results: list[object]

    def audit_turn(self, action_result, state=None):
        self.states.append(state)
        self.action_results.append(action_result)
        if "off-track" in action_result.content:
            return AuditSignal(AuditVerdict.ALARM, "off track")
        return AuditSignal(AuditVerdict.CONTINUE)


def test_producer_replays_each_prefix_and_preserves_provenance() -> None:
    auditor = _Auditor([], [])
    ticks = iter((1.0, 1.01, 2.0, 2.025))
    evidence = produce_evidence_payload(
        _label_pack(),
        auditor=auditor,
        auditor_id="ollama:qwen@sha256:" + "a" * 64,
        clock=lambda: next(ticks),
    )

    case = evidence["cases"][0]
    assert [item["verdict"] for item in case["observations"]] == ["continue", "alarm"]
    assert [item["latency_ms"] for item in case["observations"]] == [10.0, 25.0]
    assert len(auditor.states[0]["htir_prefix"]) == 1
    assert len(auditor.states[1]["htir_prefix"]) == 2
    assert auditor.action_results[0].tool_results == [{"name": "read", "success": True}]
    assert auditor.action_results[1].tool_results == []
    assert case["trace"]["metadata"] == {"source": "held-out"}
    assert len(evidence["producer"]["label_pack_sha256"]) == 64
    assert assess_evidence_payload(evidence).metrics["true_positives"] == 1


def test_producer_rejects_observations_in_independent_label_pack() -> None:
    labels = _label_pack()
    labels["cases"][0]["observations"] = []
    with pytest.raises(ValueError, match="must not contain model observations"):
        produce_evidence_payload(labels, auditor=_Auditor([], []), auditor_id="judge@revision")


def test_producer_rejects_auditor_identity_in_independent_label_pack() -> None:
    labels = _label_pack()
    labels["auditor_id"] = "model-that-may-have-influenced-labels@mutable"
    with pytest.raises(ValueError, match="must not contain an auditor identity"):
        produce_evidence_payload(labels, auditor=_Auditor([], []), auditor_id="judge@revision")


def test_resolve_ollama_identity_requires_exact_tag_and_digest(monkeypatch) -> None:
    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"models": [{"name": "qwen3.5:2b", "digest": "a" * 64}]}

    monkeypatch.setattr("httpx.get", lambda *_args, **_kwargs: _Response())
    identity = resolve_ollama_identity(
        base_url="http://ollama", model="qwen3.5:2b", expected_digest="sha256:" + "a" * 64
    )
    assert identity.auditor_id == "ollama:qwen3.5:2b@sha256:" + "a" * 64

    with pytest.raises(ValueError, match="did not resolve"):
        resolve_ollama_identity(base_url="http://ollama", model="qwen3.5")
    with pytest.raises(ValueError, match="digest mismatch"):
        resolve_ollama_identity(
            base_url="http://ollama", model="qwen3.5:2b", expected_digest="b" * 64
        )


def test_warm_ollama_model_disables_thinking_and_keeps_model_loaded(monkeypatch) -> None:
    calls = []

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"model": "qwen3.5:2b", "response": "READY"}

    def _post(url, **kwargs):
        calls.append((url, kwargs))
        return _Response()

    monkeypatch.setattr("httpx.post", _post)
    warm_ollama_model(base_url="http://ollama/", model="qwen3.5:2b")
    url, kwargs = calls[0]
    assert url == "http://ollama/api/generate"
    assert kwargs["json"]["think"] is False
    assert kwargs["json"]["keep_alive"] == "15m"
