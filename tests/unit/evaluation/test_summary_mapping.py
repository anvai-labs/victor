"""Shared evaluation summary-mapping extraction."""

from __future__ import annotations

from types import SimpleNamespace

from victor.evaluation import (
    degradation_feedback,
    planning_feedback,
    runtime_feedback,
    team_feedback,
    topology_feedback,
)
from victor.evaluation.summary_mapping import extract_summary_mapping, extract_value


def test_extract_value_reads_mappings_and_attributes() -> None:
    assert extract_value({"a": 1}, "a") == 1
    assert extract_value({"a": 1}, "b", "d") == "d"
    assert extract_value(SimpleNamespace(a=2), "a") == 2
    assert extract_value(SimpleNamespace(), "a", "d") == "d"


def test_top_level_mapping_and_attribute_summaries() -> None:
    summary = {"k": 1}
    found = extract_summary_mapping({"topology_summary": summary}, "topology_summary")
    assert found == summary and found is not summary
    assert (
        extract_summary_mapping(
            SimpleNamespace(team_feedback_summary=summary), "team_feedback_summary"
        )
        == summary
    )


def test_container_lookup_in_order() -> None:
    value = {
        "metadata": {"other": {}},
        "failure_details": {"optimization_summary": {"from": "failure_details"}},
        "completion_signals": {"optimization_summary": {"from": "completion_signals"}},
    }
    assert extract_summary_mapping(value, "optimization_summary") == {"from": "failure_details"}


def test_trace_recursion_and_self_trace_guard() -> None:
    nested = {"trace": SimpleNamespace(metadata={"topology_summary": {"deep": True}})}
    assert extract_summary_mapping(nested, "topology_summary") == {"deep": True}

    selfref: dict = {}
    selfref["trace"] = selfref
    assert extract_summary_mapping(selfref, "topology_summary") is None


def test_non_mapping_summaries_and_wrong_key_are_ignored() -> None:
    assert extract_summary_mapping({"topology_summary": [1, 2]}, "topology_summary") is None
    assert (
        extract_summary_mapping({"metadata": {"topology_summary": "x"}}, "topology_summary") is None
    )
    assert extract_summary_mapping({"topology_summary": {"k": 1}}, "team_feedback_summary") is None
    assert extract_summary_mapping(None, "topology_summary") is None


def test_feedback_modules_share_one_extract_value() -> None:
    for module in (
        degradation_feedback,
        planning_feedback,
        runtime_feedback,
        team_feedback,
        topology_feedback,
    ):
        assert module._extract_value is extract_value


def test_feedback_summaries_still_pass_through_existing_mappings() -> None:
    assert topology_feedback.summarize_topology_feedback(
        {"metadata": {"topology_summary": {"x": 1}}}
    ) == {"x": 1}
    assert topology_feedback.summarize_optimization_feedback({"optimization_summary": {}}) is None
    assert team_feedback.summarize_team_feedback(
        SimpleNamespace(trace={"team_feedback_summary": {"t": 1}})
    ) == {"t": 1}
