"""Tests for shared Bayesian monitoring reporting helpers."""

from __future__ import annotations

from victor.framework.rl.monitoring.reporting import parse_agent_ids


def test_parse_agent_ids_trims_and_filters_empty_values() -> None:
    assert parse_agent_ids(" agent_a, ,agent_b ,, agent_c ") == [
        "agent_a",
        "agent_b",
        "agent_c",
    ]
    assert parse_agent_ids("") is None
    assert parse_agent_ids(None) is None
