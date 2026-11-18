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


def test_turn_record_rejects_embedded_credential_fields():
    turn = _turn()
    turn.response.metadata = {"api_key": "not-a-real-secret"}

    with pytest.raises(ValueError, match="Refusing durable turn field 'api_key'"):
        TurnRecord.from_turn_result(turn)


def test_turn_record_rejects_embedded_unknown_values():
    turn = _turn()
    turn.response.metadata = {"runtime_handle": object()}

    with pytest.raises(TypeError, match="not JSON-compatible"):
        TurnRecord.from_turn_result(turn)


def test_turn_record_rejects_unknown_disposition():
    turn = _turn()
    turn.disposition = "keep-going"

    with pytest.raises(TypeError, match="Unknown turn disposition: 'keep-going'"):
        TurnRecord.from_turn_result(turn)


def test_turn_record_projects_sorted_json_values():
    turn = _turn()
    turn.response.metadata = {"tags": {"b", "a"}}

    record = TurnRecord.from_turn_result(turn)

    assert record.response.metadata["tags"] == ["a", "b"]


def test_turn_record_rejects_non_string_completion_content():
    turn = _turn()
    turn.response.content = object()

    with pytest.raises(TypeError, match="Completion content of type object is not JSON-compatible"):
        TurnRecord.from_turn_result(turn)


def test_turn_record_rejects_non_finite_numbers():
    turn = _turn()
    turn.response.metadata = {"score": float("inf")}

    with pytest.raises(TypeError, match="Non-finite floats are not JSON-compatible"):
        TurnRecord.from_turn_result(turn)


async def test_graph_evaluator_prefers_run_local_act_for_evaluation():
    from unittest.mock import AsyncMock

    from victor.framework.agentic_loop import _GraphEvaluatorAdapter

    durable = TurnRecord.from_turn_result(_turn("durable"))
    live = _turn("live")
    evaluate = AsyncMock(return_value="evaluated")
    adapter = _GraphEvaluatorAdapter(evaluate, _live_action_result=lambda: live)
    perception = object()

    assert await adapter.evaluate(perception, durable, {}) == "evaluated"
    evaluate.assert_awaited_once_with(perception, live, {})


def test_stategraph_executor_receives_run_local_turn_channel():
    """The run-owned live ACT cache must reach the graph composition boundary."""
    from victor.framework.agentic_loop import AgenticLoop

    loop = AgenticLoop.__new__(AgenticLoop)
    loop.orchestrator = None
    loop.turn_executor = object()
    loop.runtime_intelligence = None
    loop.execution_context_override = None
    loop.fulfillment = None
    loop.max_iterations = 2
    loop.enable_fulfillment_check = True
    loop.enable_adaptive_iterations = True
    channel = TurnRecordChannel()

    executor = loop._create_stategraph_executor(turn_record_channel=channel)

    assert executor.turn_record_channel is channel


async def test_completed_act_record_survives_durable_checkpoint_failure():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from victor.framework.agentic_graph.executor import _turn_record_observer
    from victor.framework.agentic_graph.nodes import act_node
    from victor.framework.agentic_graph.state import AgenticLoopStateModel

    class FailingCheckpointer(MemoryCheckpointer):
        async def save(self, _checkpoint):
            raise RuntimeError("durable write failed")

    completed_turn = _turn("completed")
    turn_executor = SimpleNamespace(execute_turn=AsyncMock(return_value=completed_turn))
    channel = TurnRecordChannel()

    async def act(state):
        return await act_node(state, turn_executor=turn_executor, turn_record_channel=channel)

    graph = StateGraph(AgenticLoopStateModel)
    graph.add_node("act", act)
    graph.set_entry_point("act")
    graph.add_edge("act", END)
    config = GraphConfig(
        checkpoint=CheckpointConfig(
            checkpointer=FailingCheckpointer(),
            checkpoint_after_nodes={"act"},
        )
    )

    result = await graph.compile().invoke(
        AgenticLoopStateModel(query="task"),
        config=config,
        thread_id="checkpoint-failure",
        node_observer=_turn_record_observer(channel),
    )

    assert result.success is False
    assert result.error == "durable write failed"
    assert channel.latest is not None
    assert channel.latest.content == "completed"
    assert channel.latest_live is completed_turn
