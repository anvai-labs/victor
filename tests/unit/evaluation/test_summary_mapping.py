"""Shared evaluation summary-mapping extraction."""

from __future__ import annotations

import inspect

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
        # No feedback module carries its own fork: either it imports the shared
        # helper or (after moving to collect_events) no longer needs it.
        assert getattr(module, "_extract_value", extract_value) is extract_value
        assert "def _extract_value" not in inspect.getsource(module)


def test_feedback_summaries_still_pass_through_existing_mappings() -> None:
    assert topology_feedback.summarize_topology_feedback(
        {"metadata": {"topology_summary": {"x": 1}}}
    ) == {"x": 1}
    assert topology_feedback.summarize_optimization_feedback({"optimization_summary": {}}) is None
    assert team_feedback.summarize_team_feedback(
        SimpleNamespace(trace={"team_feedback_summary": {"t": 1}})
    ) == {"t": 1}


def test_extract_sequence_reads_mapping_keys_and_attributes() -> None:
    from victor.evaluation.summary_mapping import extract_sequence

    assert extract_sequence({"events": ("a", "b")}, "events") == ["a", "b"]
    assert extract_sequence(type("T", (), {"events": ["x"]})(), "events") == ["x"]
    assert extract_sequence({"events": None}, "events") == []
    assert extract_sequence({"events": "not-a-list"}, "events") == []
    assert extract_sequence(None, "events") == []


def test_as_event_mapping_accepts_mappings_and_to_dict() -> None:
    from victor.evaluation.summary_mapping import as_event_mapping

    class _Event:
        def to_dict(self) -> dict:
            return {"kind": "k"}

    assert as_event_mapping({"a": 1}) == {"a": 1}
    assert as_event_mapping(_Event()) == {"kind": "k"}
    assert as_event_mapping(None) is None
    assert as_event_mapping(["not", "a", "mapping"]) is None


def test_optional_coercions() -> None:
    from enum import Enum

    from victor.evaluation.summary_mapping import (
        coerce_optional_float,
        coerce_optional_int,
        coerce_optional_text,
    )

    class _Kind(Enum):
        RETRY = " retry "

    assert coerce_optional_text(_Kind.RETRY) == "retry"
    assert coerce_optional_text("   ") is None
    assert coerce_optional_text(None) is None
    assert coerce_optional_float("1.5") == 1.5
    assert coerce_optional_float("") is None
    assert coerce_optional_float("x") is None
    assert coerce_optional_int("7") == 7
    assert coerce_optional_int(None) is None
    assert coerce_optional_int("7.5") is None


def test_collect_events_walks_sources_containers_and_trace_in_order() -> None:
    from victor.evaluation.summary_mapping import collect_events

    def as_a(item):
        return {"kind": "a", "v": item} if isinstance(item, int) else None

    def as_b(item):
        return {"kind": "b", "v": item} if isinstance(item, int) else None

    value = {
        "a_events": [1, "junk"],
        "b_events": [2],
        "metadata": {"a_events": [3], "b_events": [4]},
        "failure_details": "not-a-mapping",
        "trace": {"a_events": [5, 1], "completion_signals": {"b_events": [6]}},
    }
    events = collect_events(
        value, [("a_events", as_a), ("b_events", as_b)], lambda e: (e["kind"], e["v"])
    )
    assert [(e["kind"], e["v"]) for e in events] == [
        ("a", 1),
        ("b", 2),
        ("a", 3),
        ("b", 4),
        ("a", 5),  # trace walked after the node's own containers
        ("b", 6),  # duplicate ("a", 1) from the trace dropped
    ]


def test_collect_events_handles_self_trace_and_empty_inputs() -> None:
    from victor.evaluation.summary_mapping import collect_events

    class _Task:
        events = [7]

    task = _Task()
    task.trace = task  # self-reference must not recurse forever
    assert collect_events(task, [("events", lambda i: {"v": i})], lambda e: e["v"]) == [{"v": 7}]
    assert collect_events(None, [("events", lambda i: {"v": i})], lambda e: e["v"]) == []
