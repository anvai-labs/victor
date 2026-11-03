"""Mirror tests for victor.agent.orchestrator_runtime_intelligence.

Pins the runtime-intelligence hook methods moved verbatim out of
victor/agent/orchestrator.py (facade decomposition): delegation to the
integration, the disabled-pipeline branch, the GEPA session-end evolution
trigger, and the back-compat alias surface.

State the methods read (conversation_state, provider, tool_calls_used, ...)
is installed on AgentOrchestrator as lazy descriptors backed by real
runtime singletons; bare object.__new__ instances lack that backing state,
so each pin patches the descriptors it needs with plain value properties.
"""

import inspect
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from victor.agent.orchestrator import AgentOrchestrator


def make_orch():
    """Bare instance (no __init__ side effects) — delegation-pin pattern."""
    return object.__new__(AgentOrchestrator)


def patched_state(orch, **values):
    """Provide state names on a bare instance.

    Names installed as class descriptors (e.g. runtime_intelligence_integration,
    tool_calls_used) are patched on the class with plain value properties;
    plain __init__-set instance attributes (conversation_state, provider, ...)
    are set directly on the instance.
    """
    stack = ExitStack()
    for name, value in values.items():
        if name in type(orch).__dict__:
            stack.enter_context(patch.object(type(orch), name, property(lambda self, v=value: v)))
        else:
            setattr(orch, name, value)
    return stack


class TestPrepareRequest:
    async def test_delegates_to_integration(self):
        orch = make_orch()
        integration = MagicMock()
        integration.prepare_runtime_intelligence_request = AsyncMock(
            return_value={"recommendations": []}
        )
        conv_state, tracker = MagicMock(), MagicMock()
        with patched_state(
            orch,
            runtime_intelligence_integration=integration,
            conversation_state=conv_state,
            unified_tracker=tracker,
        ):
            result = await AgentOrchestrator._prepare_runtime_intelligence_request(
                orch, "do a thing", "edit"
            )
        assert result == {"recommendations": []}
        _, kwargs = integration.prepare_runtime_intelligence_request.call_args
        assert kwargs["task"] == "do a thing"
        assert kwargs["task_type"] == "edit"
        assert kwargs["conversation_state"] is conv_state
        assert kwargs["unified_tracker"] is tracker

    async def test_disabled_returns_none(self):
        orch = make_orch()
        with patched_state(orch, runtime_intelligence_integration=None):
            result = await AgentOrchestrator._prepare_runtime_intelligence_request(
                orch, "task", "edit"
            )
        assert result is None


class TestValidateResponse:
    async def test_delegates_to_integration(self):
        orch = make_orch()
        integration = MagicMock()
        integration.validate_runtime_intelligence_response = AsyncMock(
            return_value={"quality": 0.9}
        )
        with patched_state(orch, runtime_intelligence_integration=integration):
            result = await AgentOrchestrator._validate_runtime_intelligence_response(
                orch, response="resp", query="q", tool_calls=2, task_type="analysis"
            )
        assert result == {"quality": 0.9}
        _, kwargs = integration.validate_runtime_intelligence_response.call_args
        assert kwargs["response"] == "resp"
        assert kwargs["tool_calls"] == 2

    async def test_disabled_returns_none(self):
        orch = make_orch()
        with patched_state(orch, runtime_intelligence_integration=None):
            result = await AgentOrchestrator._validate_runtime_intelligence_response(
                orch, response="r", query="q", tool_calls=0, task_type="edit"
            )
        assert result is None


class TestRecordOutcome:
    def _armed(self, tool_calls):
        orch = make_orch()
        integration = MagicMock()
        orch._rl_coordinator = None
        orch._vertical_context = None
        return orch, integration, tool_calls

    def _enter(self, orch, integration, tool_calls):
        return patched_state(
            orch,
            runtime_intelligence_integration=integration,
            tool_calls_used=tool_calls,
            provider=SimpleNamespace(name="prov"),
            model="m1",
        )

    def test_delegates_state_gathering(self):
        orch, integration, _ = self._armed(5)
        with self._enter(orch, integration, 5):
            AgentOrchestrator._record_runtime_intelligence_outcome(
                orch, success=True, quality_score=0.8
            )
        _, kwargs = integration.record_runtime_intelligence_outcome.call_args
        assert kwargs["success"] is True
        assert kwargs["quality_score"] == 0.8
        assert kwargs["rl_coordinator"] is None
        assert kwargs["provider_name"] == "prov"
        assert kwargs["tool_calls_used"] == 5

    def test_disabled_noop(self):
        orch = make_orch()
        with patched_state(orch, runtime_intelligence_integration=None):
            AgentOrchestrator._record_runtime_intelligence_outcome(orch, success=True)
        # No attribute access blew up; nothing to assert beyond no-raise.

    def test_gepa_evolution_triggered_at_three_tool_calls(self):
        orch, integration, _ = self._armed(3)
        rl = MagicMock()
        rl.try_evolve_on_session_end = MagicMock(return_value={"generation": 2})
        orch._rl_coordinator = rl
        orch.debug_logger = MagicMock()
        with self._enter(orch, integration, 3):
            AgentOrchestrator._record_runtime_intelligence_outcome(orch, success=True)
        rl.try_evolve_on_session_end.assert_called_once_with("prov", "m1")
        orch.debug_logger.log_evolution_report.assert_called_once_with({"generation": 2})

    def test_gepa_evolution_skipped_below_threshold(self):
        # Negative: 2 tool calls must NOT trigger session-end evolution.
        orch, integration, _ = self._armed(2)
        rl = MagicMock()
        orch._rl_coordinator = rl
        with self._enter(orch, integration, 2):
            AgentOrchestrator._record_runtime_intelligence_outcome(orch, success=True)
        rl.try_evolve_on_session_end.assert_not_called()


class TestShouldContinue:
    def test_delegates(self):
        orch = make_orch()
        integration = MagicMock()
        integration.should_continue_runtime_intelligence = MagicMock(return_value=(False, "stuck"))
        with patched_state(orch, runtime_intelligence_integration=integration):
            assert AgentOrchestrator._should_continue_runtime_intelligence(orch) == (
                False,
                "stuck",
            )

    def test_disabled_returns_continue(self):
        orch = make_orch()
        with patched_state(orch, runtime_intelligence_integration=None):
            assert AgentOrchestrator._should_continue_runtime_intelligence(orch) == (
                True,
                "Pipeline disabled",
            )


class TestBackCompatAliases:
    async def test_alias_routing(self):
        orch = make_orch()
        orch._rl_coordinator = None
        orch._vertical_context = None
        integration = MagicMock()
        integration.prepare_runtime_intelligence_request = AsyncMock(return_value={"r": 1})
        integration.validate_runtime_intelligence_response = AsyncMock(return_value={"v": 1})
        integration.should_continue_runtime_intelligence = MagicMock(return_value=(True, "ok"))
        with patched_state(
            orch,
            runtime_intelligence_integration=integration,
            conversation_state=MagicMock(),
            unified_tracker=MagicMock(),
            tool_calls_used=0,
            provider=SimpleNamespace(name="prov"),
            model="m1",
        ):
            assert AgentOrchestrator._should_continue_intelligent(orch) == (True, "ok")
            await AgentOrchestrator._prepare_intelligent_request(orch, "t", "edit")
            await AgentOrchestrator._validate_intelligent_response(
                orch, response="r", query="q", tool_calls=1, task_type="edit"
            )
            AgentOrchestrator._record_intelligent_outcome(orch, success=True)
        integration.prepare_runtime_intelligence_request.assert_called_once()
        integration.validate_runtime_intelligence_response.assert_called_once()
        integration.record_runtime_intelligence_outcome.assert_called_once()

    def test_methods_live_on_mixin(self):
        # Negative pin: these must NOT be re-defined in the orchestrator class
        # body (would shadow the mixin and break the facade decomposition).
        class_src = inspect.getsource(AgentOrchestrator)
        for name in (
            "_prepare_runtime_intelligence_request",
            "_record_runtime_intelligence_outcome",
            "_should_continue_runtime_intelligence",
        ):
            assert f"def {name}" not in class_src, name
            func = getattr(AgentOrchestrator, name)
            path = func.__code__.co_filename
            assert path.endswith("orchestrator_runtime_intelligence.py"), (name, path)
