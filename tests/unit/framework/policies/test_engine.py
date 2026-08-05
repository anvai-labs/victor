# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for PolicyEngine verdict composition."""

from typing import List, Tuple

from victor.framework.policies import (
    Phase,
    Policy,
    PolicyAction,
    PolicyEngine,
    PolicyEvent,
    PolicyVerdict,
)
from victor.framework.policies.types import UNSET


class _StubPolicy(Policy):
    """A policy returning a fixed verdict, recording whether it ran."""

    def __init__(self, name: str, verdict: PolicyVerdict, *, phase=Phase.TOOL_CALL):
        self.name = name
        self._verdict = verdict
        self._phase = phase
        self.evaluated = False

    def phases(self):
        return {self._phase}

    async def evaluate(self, event: PolicyEvent) -> PolicyVerdict:
        self.evaluated = True
        return self._verdict


def _event(phase=Phase.TOOL_CALL, tool_name="write_file", **kwargs) -> PolicyEvent:
    return PolicyEvent(phase=phase, tool_name=tool_name, **kwargs)


async def test_empty_engine_allows():
    engine = PolicyEngine([])
    verdict = await engine.evaluate(_event())
    assert verdict.is_allow


async def test_all_allow_results_in_allow():
    engine = PolicyEngine(
        [
            _StubPolicy("a", PolicyVerdict.allow()),
            _StubPolicy("b", PolicyVerdict.allow()),
        ]
    )
    verdict = await engine.evaluate(_event())
    assert verdict.action is PolicyAction.ALLOW


async def test_first_deny_short_circuits():
    deny = _StubPolicy("deny", PolicyVerdict.deny("nope", policy_name="deny"))
    later = _StubPolicy("later", PolicyVerdict.allow())
    engine = PolicyEngine([deny, later])

    verdict = await engine.evaluate(_event())

    assert verdict.is_deny
    assert verdict.reason == "nope"
    assert verdict.policy_name == "deny"
    # Short-circuit: the later policy must not have been evaluated.
    assert later.evaluated is False


async def test_deny_wins_over_ask_regardless_of_order():
    ask = _StubPolicy("ask", PolicyVerdict.ask("approve?", policy_name="ask"))
    deny = _StubPolicy("deny", PolicyVerdict.deny("blocked", policy_name="deny"))
    # ASK is first, DENY second — DENY must still win.
    engine = PolicyEngine([ask, deny])

    verdict = await engine.evaluate(_event())

    assert verdict.is_deny


async def test_ask_aggregation_combines_reasons():
    a = _StubPolicy("a", PolicyVerdict.ask("reason A", policy_name="a"))
    b = _StubPolicy("b", PolicyVerdict.ask("reason B", policy_name="b"))
    engine = PolicyEngine([a, b])

    verdict = await engine.evaluate(_event())

    assert verdict.is_ask
    assert "reason A" in verdict.reason
    assert "reason B" in verdict.reason
    # The first asking policy is named for elicitation.
    assert verdict.policy_name == "a"


async def test_phase_filtering_skips_non_matching_policies():
    result_only = _StubPolicy("result", PolicyVerdict.deny("x"), phase=Phase.TOOL_RESULT)
    engine = PolicyEngine([result_only])

    # A TOOL_CALL event should not trigger a TOOL_RESULT-only policy.
    verdict = await engine.evaluate(_event(phase=Phase.TOOL_CALL))

    assert verdict.is_allow
    assert result_only.evaluated is False


async def test_argument_modifications_thread_between_policies():
    class _RedactArg(Policy):
        name = "redact"

        async def evaluate(self, event: PolicyEvent) -> PolicyVerdict:
            new_args = dict(event.arguments)
            new_args["secret"] = "***"
            return PolicyVerdict.allow(modified_arguments=new_args)

    seen: List[dict] = []

    class _Observer(Policy):
        name = "observer"

        async def evaluate(self, event: PolicyEvent) -> PolicyVerdict:
            seen.append(dict(event.arguments))
            return PolicyVerdict.allow()

    engine = PolicyEngine([_RedactArg(), _Observer()])
    verdict = await engine.evaluate(_event(arguments={"secret": "abc"}))

    # The observer saw the redacted args; engine returns the final modified args.
    assert seen[0]["secret"] == "***"
    assert verdict.modified_arguments == {"secret": "***"}


async def test_result_modification_on_tool_result_phase():
    class _RedactResult(Policy):
        name = "redact_result"

        def phases(self):
            return {Phase.TOOL_RESULT}

        async def evaluate(self, event: PolicyEvent) -> PolicyVerdict:
            return PolicyVerdict.allow(modified_result="[REDACTED]")

    engine = PolicyEngine([_RedactResult()])
    verdict = await engine.evaluate(_event(phase=Phase.TOOL_RESULT, result="secret"))

    assert verdict.modified_result == "[REDACTED]"


async def test_misbehaving_policy_is_skipped_not_fatal():
    class _Boom(Policy):
        name = "boom"

        async def evaluate(self, event: PolicyEvent) -> PolicyVerdict:
            raise RuntimeError("kaboom")

    engine = PolicyEngine([_Boom(), _StubPolicy("ok", PolicyVerdict.allow())])
    verdict = await engine.evaluate(_event())

    # The crashing policy is skipped; evaluation continues and allows.
    assert verdict.is_allow


async def test_event_emitter_called_on_deny():
    captured: List[Tuple[str, dict]] = []

    def emitter(topic: str, payload: dict) -> None:
        captured.append((topic, payload))

    engine = PolicyEngine(
        [_StubPolicy("deny", PolicyVerdict.deny("no", policy_name="deny"))],
        event_emitter=emitter,
    )
    await engine.evaluate(_event())

    assert captured
    topic, payload = captured[0]
    assert topic == "policy.decision"
    assert payload["decision"] == "deny"
    assert payload["policy"] == "deny"


async def test_no_emit_on_plain_allow():
    captured: List[Tuple[str, dict]] = []
    engine = PolicyEngine(
        [_StubPolicy("ok", PolicyVerdict.allow())],
        event_emitter=lambda t, p: captured.append((t, p)),
    )
    await engine.evaluate(_event())
    assert captured == []


async def test_deny_tools_wins_over_ask_for_same_tool():
    # A hard DenyToolsPolicy placed before AskOnToolsPolicy must short-circuit
    # to DENY (no approval prompt) for the same tool.
    from victor.framework.policies import AskOnToolsPolicy, DenyToolsPolicy

    engine = PolicyEngine([DenyToolsPolicy(["run_command"]), AskOnToolsPolicy(["run_command"])])
    verdict = await engine.evaluate(_event(tool_name="run_command"))
    assert verdict.is_deny
    assert verdict.policy_name == "deny_tools"


def test_unset_sentinel_is_distinct_from_none():
    # A policy can legitimately replace a result with None and be distinguished
    # from "no replacement".
    assert UNSET is not None
    assert PolicyVerdict.allow().modified_result is UNSET
