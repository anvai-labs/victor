"""Durable ACT-boundary projection contracts."""

import pytest

from victor.agent.services.turn_execution_runtime import TurnDisposition, TurnResult
from victor.core.provider_messages import CompletionResponse
from victor.framework.config import CheckpointConfig, GraphConfig
from victor.framework.graph import END, MemoryCheckpointer, StateGraph
from victor.framework.agentic_graph.turn_record import (
    TURN_RECORD_SCHEMA_VERSION,
    TurnRecord,
    TurnRecordChannel,
)


def _turn(content: str = "completed") -> TurnResult:
    return TurnResult(
        response=CompletionResponse(content=content),
        disposition=TurnDisposition.COMPLETION_CANDIDATE,
    )


def test_turn_record_is_versioned_json_and_excludes_live_response_object():
    turn = _turn()
    # Even an accidentally attached raw provider/runtime value must not become durable state.
    turn.raw_response = object()

    record = TurnRecord.from_turn_result(turn)
    durable = record.model_dump(mode="json")

    assert durable["schema_version"] == TURN_RECORD_SCHEMA_VERSION
    assert durable["response"]["content"] == "completed"
    assert durable["disposition"] == "completion_candidate"
    assert "raw_response" not in durable


def test_turn_record_reconstructs_evaluator_turn_result():
    turn = _turn("resume me")
    restored = TurnRecord.from_turn_result(turn).to_turn_result()

    assert isinstance(restored, TurnResult)
    assert restored.content == "resume me"
    assert restored.disposition is TurnDisposition.COMPLETION_CANDIDATE


def test_channels_isolate_latest_records_between_runs():
    first = TurnRecordChannel()
    second = TurnRecordChannel()

    first.publish(TurnRecord.from_turn_result(_turn("one")))
    second.publish(TurnRecord.from_turn_result(_turn("two")))
    first.publish(TurnRecord.from_turn_result(_turn("one-again")))

    assert first.latest is not None and first.latest.content == "one-again"
    assert second.latest is not None and second.latest.content == "two"


def test_turn_record_rejects_embedded_unknown_values():
    turn = _turn()
    turn.response.metadata = {"runtime_handle": object()}

    with pytest.raises(TypeError, match="not JSON-compatible"):
        TurnRecord.from_turn_result(turn)


async def test_completed_node_observation_survives_durable_checkpoint_failure():
    class FailingCheckpointer(MemoryCheckpointer):
        async def save(self, _checkpoint):
            raise RuntimeError("durable write failed")

    async def act(state):
        return {**state, "value": state["value"] + 1}

    graph = StateGraph(dict)
    graph.add_node("act", act)
    graph.set_entry_point("act")
    graph.add_edge("act", END)
    config = GraphConfig(
        checkpoint=CheckpointConfig(
            checkpointer=FailingCheckpointer(),
            checkpoint_after_nodes={"act"},
        )
    )
    events: list[str] = []

    async def observer(event, _node, _state, _message, _duration):
        events.append(event)

    result = await graph.compile().invoke(
        {"value": 0},
        config=config,
        thread_id="checkpoint-failure",
        node_observer=observer,
    )

    assert result.success is False
    assert result.error == "durable write failed"
    assert "complete" in events
