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

"""Unit tests for ADR-012 prong-2 recovery layer attribution (EVR-5)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from victor.agent.services.recovery_service import (
    RecoveryContextImpl,
    RecoveryService,
    resolve_recovery_layer_attribution_enabled,
)
from victor.evaluation.htir import ETCLOVGLayer, attribute_failure_layer

# ── attribute_failure_layer (pure helper) ─────────────────────────────────


def test_attribute_failure_layer_execution() -> None:
    assert attribute_failure_layer("edit") is ETCLOVGLayer.EXECUTION
    assert attribute_failure_layer("shell") is ETCLOVGLayer.EXECUTION


def test_attribute_failure_layer_context_memory() -> None:
    assert attribute_failure_layer("grep") is ETCLOVGLayer.CONTEXT_MEMORY


def test_attribute_failure_layer_tooling_default() -> None:
    assert attribute_failure_layer("some_unknown_tool") is ETCLOVGLayer.TOOLING


def test_attribute_failure_layer_none_when_no_tool() -> None:
    assert attribute_failure_layer(None) is None
    assert attribute_failure_layer("") is None


# ── flag resolver ─────────────────────────────────────────────────────────


def _settings(enabled: bool) -> SimpleNamespace:
    return SimpleNamespace(agent=SimpleNamespace(recovery_layer_attribution=enabled))


def test_resolver_defaults_off() -> None:
    assert resolve_recovery_layer_attribution_enabled(SimpleNamespace(agent=None)) is False


def test_resolver_reads_settings() -> None:
    assert resolve_recovery_layer_attribution_enabled(_settings(True)) is True
    assert resolve_recovery_layer_attribution_enabled(_settings(False)) is False


def test_resolver_env_overrides_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VICTOR_RECOVERY_LAYER_ATTRIBUTION", "1")
    assert resolve_recovery_layer_attribution_enabled(_settings(False)) is True
    monkeypatch.setenv("VICTOR_RECOVERY_LAYER_ATTRIBUTION", "off")
    assert resolve_recovery_layer_attribution_enabled(_settings(True)) is False


# ── RecoveryService integration ───────────────────────────────────────────


def _ctx(state: dict | None = None, **metadata: object) -> RecoveryContextImpl:
    return RecoveryContextImpl(
        error=RuntimeError("boom"),
        error_type="unknown",
        attempt_count=1,
        state=state or {},
        metadata=dict(metadata),
    )


def test_service_attribution_disabled_by_default() -> None:
    svc = RecoveryService()
    assert svc.classify_failure_layer(_ctx(failing_tool="edit")) is None
    assert svc._metrics["by_layer"] == {}


def test_service_attributes_layer_when_enabled() -> None:
    svc = RecoveryService()
    svc.bind_runtime_components(settings=_settings(True))
    ctx = _ctx(failing_tool="edit")
    layer = svc.classify_failure_layer(ctx)
    assert layer is ETCLOVGLayer.EXECUTION
    assert svc._metrics["by_layer"]["execution"] == 1
    assert ctx.metadata["failure_layer"] == "execution"


def test_service_reads_failing_tool_from_state() -> None:
    svc = RecoveryService()
    svc.bind_runtime_components(settings=_settings(True))
    ctx = _ctx(state={"failing_tool": "grep"})
    assert svc.classify_failure_layer(ctx) is ETCLOVGLayer.CONTEXT_MEMORY


def test_service_no_attribution_without_a_tool() -> None:
    svc = RecoveryService()
    svc.bind_runtime_components(settings=_settings(True))
    ctx = _ctx()
    assert svc.classify_failure_layer(ctx) is None
    assert "failure_layer" not in ctx.metadata
    assert svc._metrics["by_layer"] == {}


async def test_execute_recovery_records_layer_when_enabled() -> None:
    svc = RecoveryService()
    svc.bind_runtime_components(settings=_settings(True))
    # error_type "unknown" → action "retry"; _retry_action with no bound runtime returns False,
    # but the layer attribution must still have been recorded from the same call.
    ctx = _ctx(failing_tool="grep")
    await svc.execute_recovery(ctx)
    assert svc._metrics["by_layer"].get("context_memory") == 1
    assert ctx.metadata["failure_layer"] == "context_memory"
