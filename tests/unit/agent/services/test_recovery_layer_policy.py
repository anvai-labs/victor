# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Unit tests for the layer-aware recovery policy (EVR-5, ADR-012 prong 2)."""

from __future__ import annotations

from types import SimpleNamespace

from victor.agent.services.recovery_service import RecoveryContextImpl, RecoveryService


def _enabled_service() -> RecoveryService:
    svc = RecoveryService()
    svc.bind_runtime_components(
        settings=SimpleNamespace(agent=SimpleNamespace(recovery_layer_attribution=True))
    )
    return svc


def _ctx(**metadata: object) -> RecoveryContextImpl:
    # error_type "unknown" maps to base action "retry".
    return RecoveryContextImpl(
        error=RuntimeError("boom"),
        error_type="unknown",
        attempt_count=1,
        state={},
        metadata=dict(metadata),
    )


async def test_single_failure_keeps_base_action() -> None:
    svc = _enabled_service()
    ctx = _ctx(failing_tool="edit")
    svc.classify_failure_layer(ctx)  # streak = 1
    assert await svc.select_recovery_action(ctx) == "retry"


async def test_repeated_same_layer_escalates_to_fallback() -> None:
    svc = _enabled_service()
    svc.classify_failure_layer(_ctx(failing_tool="edit"))  # streak = 1
    ctx2 = _ctx(failing_tool="edit")
    svc.classify_failure_layer(ctx2)  # streak = 2 (same layer)
    assert await svc.select_recovery_action(ctx2) == "fallback"


async def test_different_layer_resets_streak() -> None:
    svc = _enabled_service()
    svc.classify_failure_layer(_ctx(failing_tool="edit"))  # execution, streak 1
    ctx2 = _ctx(failing_tool="grep")  # context_memory → streak resets to 1
    svc.classify_failure_layer(ctx2)
    assert await svc.select_recovery_action(ctx2) == "retry"


async def test_governance_failure_is_never_retried() -> None:
    svc = _enabled_service()
    ctx = _ctx(failing_tool="ask")  # governance, streak 1
    svc.classify_failure_layer(ctx)
    assert await svc.select_recovery_action(ctx) == "fallback"


async def test_flag_off_never_refines() -> None:
    svc = RecoveryService()  # attribution disabled
    ctx = _ctx(failing_tool="ask")
    # Even with a governance layer stamped, a disabled service keeps the base action.
    ctx.metadata["failure_layer"] = "governance"
    ctx.metadata["failure_layer_streak"] = 9
    assert await svc.select_recovery_action(ctx) == "retry"


async def test_execute_recovery_escalation_records_metric(monkeypatch) -> None:
    svc = _enabled_service()

    async def _always_fail(_ctx: RecoveryContextImpl) -> bool:
        return False  # retries keep failing, so the same-layer streak builds

    monkeypatch.setattr(svc, "_retry_action", _always_fail)
    await svc.execute_recovery(_ctx(failing_tool="edit"))  # streak 1 → retry (fails)
    await svc.execute_recovery(_ctx(failing_tool="edit"))  # streak 2 → fallback
    assert svc._metrics["layer_escalations"] == 1


async def test_successful_recovery_clears_streak() -> None:
    svc = _enabled_service()
    # Default _retry_action succeeds, so a recovery clears the streak for the next failure.
    await svc.execute_recovery(_ctx(failing_tool="edit"))
    assert svc._layer_streak == 0
    assert svc._last_failure_layer is None
