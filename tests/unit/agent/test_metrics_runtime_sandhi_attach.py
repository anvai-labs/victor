# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""FEP-0020 Phase 2 (M5) — SandhiMeter auto-attach at cost-tracker build.

The attach seam lives in ``create_metrics_runtime_components``: when
``settings.usage_gateway.enabled`` is True (and the sandhi-gateway extra is
importable), the freshly built SessionCostTracker gets a SandhiMeter on
``_sandhi`` plus operator-level subject/group attribution. Default-off must be
byte-identical across three structural layers: no settings kwarg, disabled
group, and missing optional dependency.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from victor.agent.runtime.metrics_runtime import create_metrics_runtime_components


def _build_runtime(**kwargs):
    """Build metrics runtime components with fake deps (mirrors test_metrics_runtime)."""
    factory = MagicMock()
    factory.create_usage_logger.return_value = MagicMock()
    factory.create_streaming_metrics_collector.return_value = MagicMock()
    factory.create_metrics_collector.return_value = MagicMock()

    return create_metrics_runtime_components(
        factory=factory,
        provider=SimpleNamespace(name="ollama"),
        model="qwen3-coder:30b",
        debug_logger=MagicMock(),
        cumulative_token_usage={"input_tokens": 0, "output_tokens": 0},
        tool_cost_lookup=lambda tool_name: f"tier:{tool_name}",
        **kwargs,
    )


class _RecordingMeter:
    """Fake SandhiMeter capturing constructor kwargs and record() calls."""

    instances: list["_RecordingMeter"] = []

    def __init__(self, *, sink_path=None):
        self.sink_path = sink_path
        self.record_calls: list[dict] = []
        _RecordingMeter.instances.append(self)

    def record(self, **kwargs):
        self.record_calls.append(kwargs)


@pytest.fixture(autouse=True)
def _reset_recording_meter():
    _RecordingMeter.instances = []
    yield
    _RecordingMeter.instances = []


class TestDefaultOffByteIdentical:
    """Three structural layers of default-off, each byte-identical."""

    def test_no_settings_kw_is_byte_identical(self):
        """Existing callers (no settings kwarg) build an unattached tracker."""
        runtime = _build_runtime()
        tracker = runtime.session_cost_tracker.get_instance()

        assert tracker._sandhi is None
        assert tracker.subject_id is None
        assert tracker.group_id is None

    def test_disabled_never_constructs(self, monkeypatch):
        """enabled=False → SandhiMeter is never even constructed."""

        class _Boom:
            def __init__(self, *a, **k):
                raise AssertionError("SandhiMeter must not be constructed when disabled")

        monkeypatch.setattr("victor.observability.sandhi_meter.SandhiMeter", _Boom)
        settings = SimpleNamespace(usage_gateway=SimpleNamespace(enabled=False))

        runtime = _build_runtime(settings=settings)
        tracker = runtime.session_cost_tracker.get_instance()

        assert tracker._sandhi is None

    def test_enabled_without_binding_logs_and_skips(self, monkeypatch, caplog):
        """enabled=True but sandhi-gateway missing → info log, no attach."""
        monkeypatch.setattr("victor.observability.sandhi_meter.sandhi_available", lambda: False)
        settings = SimpleNamespace(
            usage_gateway=SimpleNamespace(
                enabled=True, sink_path=None, subject_id=None, group_id=None
            )
        )

        runtime = _build_runtime(settings=settings)
        with caplog.at_level(logging.INFO, logger="victor.agent.runtime.metrics_runtime"):
            tracker = runtime.session_cost_tracker.get_instance()

        assert tracker._sandhi is None
        assert "sandhi-gateway is not installed" in caplog.text


class TestEnabledAttach:
    """enabled=True with the binding present attaches at tracker build."""

    def test_enabled_attaches_and_events_flow(self, monkeypatch, tmp_path):
        monkeypatch.setattr("victor.observability.sandhi_meter.sandhi_available", lambda: True)
        monkeypatch.setattr("victor.observability.sandhi_meter.SandhiMeter", _RecordingMeter)
        sink = tmp_path / "usage_events.jsonl"
        settings = SimpleNamespace(
            usage_gateway=SimpleNamespace(
                enabled=True,
                sink_path=str(sink),
                subject_id="alice",
                group_id="team-a",
            )
        )

        runtime = _build_runtime(settings=settings)
        tracker = runtime.session_cost_tracker.get_instance()

        assert isinstance(tracker._sandhi, _RecordingMeter)
        assert tracker._sandhi.sink_path == str(sink)
        assert tracker.subject_id == "alice"
        assert tracker.group_id == "team-a"

        tracker.record_request(
            prompt_tokens=100,
            completion_tokens=50,
            cache_read_tokens=10,
            cache_write_tokens=5,
        )

        assert len(tracker._sandhi.record_calls) == 1
        call = tracker._sandhi.record_calls[0]
        assert call["provider"] == "ollama"
        assert call["prompt_tokens"] == 100
        assert call["completion_tokens"] == 50
        assert call["cache_read_tokens"] == 10
        assert call["cache_write_tokens"] == 5
        assert call["subject_id"] == "alice"
        assert call["group_id"] == "team-a"
        assert call["session_id"] == tracker.session_id

    def test_attach_does_not_clobber_existing_identity(self, monkeypatch):
        """Identity fill is only-when-None — a preset subject/group stays."""
        from victor.agent.runtime.metrics_runtime import _maybe_attach_sandhi_meter

        monkeypatch.setattr("victor.observability.sandhi_meter.sandhi_available", lambda: True)
        monkeypatch.setattr("victor.observability.sandhi_meter.SandhiMeter", _RecordingMeter)
        tracker = SimpleNamespace(subject_id="preset", group_id="preset-g", _sandhi=None)
        settings = SimpleNamespace(
            usage_gateway=SimpleNamespace(
                enabled=True, sink_path="/tmp/x.jsonl", subject_id="bob", group_id="team-b"
            )
        )

        _maybe_attach_sandhi_meter(tracker, settings)

        assert isinstance(tracker._sandhi, _RecordingMeter)
        assert tracker.subject_id == "preset"
        assert tracker.group_id == "preset-g"


class TestEndToEndBinding:
    """[binding] Real sandhi-gateway: one neutral event per recorded request."""

    def test_end_to_end_neutral_event(self, tmp_path):
        pytest.importorskip("sandhi_gateway", reason="requires the victor[sandhi] extra")

        sink = tmp_path / "usage_events.jsonl"
        settings = SimpleNamespace(
            usage_gateway=SimpleNamespace(
                enabled=True,
                sink_path=str(sink),
                subject_id="alice",
                group_id="team-a",
            )
        )

        runtime = _build_runtime(settings=settings)
        tracker = runtime.session_cost_tracker.get_instance()

        from victor.observability.sandhi_meter import SandhiMeter

        assert isinstance(tracker._sandhi, SandhiMeter)

        tracker.record_request(
            prompt_tokens=1000,
            completion_tokens=500,
            cache_read_tokens=200,
            cache_write_tokens=64,
        )

        events = tracker._sandhi.events()
        assert len(events) == 1
        ev = events[0]
        assert ev["subject_id"] == "alice"
        assert ev["group_id"] == "team-a"
        assert ev["virtual_key_id"] == "victor:alice"
        assert ev["session_id"] == tracker.session_id
        assert ev["provider"] == "ollama"
        assert ev["tokens_in"] == 1000
        assert ev["tokens_out"] == 500
        assert ev["cache_read_tokens"] == 200
        assert ev["cache_creation_tokens"] == 64
        # Persistent sink also received the event.
        assert sink.exists()
        assert '"subject_id":"alice"' in sink.read_text()
