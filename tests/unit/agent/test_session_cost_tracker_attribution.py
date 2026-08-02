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

"""FEP-0020 attribution join — the authenticated API-server identity reaches
the SandhiMeter usage events emitted from ``SessionCostTracker.record_request``.

Precedence contract: the authenticated subject bound on the execution context
(``victor.core.context.bind_attribution``, set at the API-server auth seam)
wins over the operator-level config default carried on the tracker; the
tracker's own ``subject_id``/``group_id`` remain the fallback for CLI/local
use where no identity is bound.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from victor.agent.session_cost_tracker import SessionCostTracker
from victor.core.context import bind_attribution


class _RecordingMeter:
    """Fake SandhiMeter capturing record() calls."""

    def __init__(self) -> None:
        self.record_calls: list[dict] = []

    def record(self, **kwargs):
        self.record_calls.append(kwargs)


def _tracker(**kwargs) -> SessionCostTracker:
    tracker = SessionCostTracker(provider="ollama", model="qwen3-coder:30b", **kwargs)
    tracker._sandhi = _RecordingMeter()
    return tracker


class TestAuthenticatedSubjectPrecedence:
    def test_authenticated_subject_wins_over_config_default(self):
        tracker = _tracker(subject_id="operator-default", group_id="ops-team")

        with bind_attribution(subject_id="alice"):
            tracker.record_request(prompt_tokens=10, completion_tokens=5)

        call = tracker._sandhi.record_calls[0]
        assert call["subject_id"] == "alice"
        # group falls back to the tracker/config default independently
        assert call["group_id"] == "ops-team"

    def test_authenticated_group_wins_when_bound(self):
        tracker = _tracker(subject_id="operator-default", group_id="ops-team")

        with bind_attribution(subject_id="alice", group_id="platform"):
            tracker.record_request(prompt_tokens=10, completion_tokens=5)

        call = tracker._sandhi.record_calls[0]
        assert call["subject_id"] == "alice"
        assert call["group_id"] == "platform"

    def test_config_default_is_fallback_without_auth_context(self):
        """CLI/local use: no bound identity keeps today's operator-level default."""
        tracker = _tracker(subject_id="operator-default", group_id="ops-team")

        tracker.record_request(prompt_tokens=10, completion_tokens=5)

        call = tracker._sandhi.record_calls[0]
        assert call["subject_id"] == "operator-default"
        assert call["group_id"] == "ops-team"

    def test_unattributed_stays_none_without_context_or_default(self):
        tracker = _tracker()

        tracker.record_request(prompt_tokens=10, completion_tokens=5)

        call = tracker._sandhi.record_calls[0]
        assert call["subject_id"] is None
        assert call["group_id"] is None

    def test_binding_does_not_leak_after_scope(self):
        tracker = _tracker(subject_id="operator-default")

        with bind_attribution(subject_id="alice"):
            tracker.record_request(prompt_tokens=1, completion_tokens=1)
        tracker.record_request(prompt_tokens=1, completion_tokens=1)

        subjects = [c["subject_id"] for c in tracker._sandhi.record_calls]
        assert subjects == ["alice", "operator-default"]

    def test_no_meter_attached_is_still_safe_under_binding(self):
        tracker = SessionCostTracker(provider="ollama", model="m")
        with bind_attribution(subject_id="alice"):
            cost = tracker.record_request(prompt_tokens=1, completion_tokens=1)
        assert cost.total_tokens == 2


class TestMetricsRuntimeEndToEnd:
    """The full attach path: config default filled at attach, auth wins at record."""

    def test_auth_subject_overrides_operator_default_from_settings(self, monkeypatch):
        from victor.agent.runtime.metrics_runtime import create_metrics_runtime_components

        recording: list[_RecordingMeter] = []

        class _Meter(_RecordingMeter):
            def __init__(self, *, sink_path=None):
                super().__init__()
                self.sink_path = sink_path
                recording.append(self)

        monkeypatch.setattr("victor.observability.sandhi_meter.sandhi_available", lambda: True)
        monkeypatch.setattr("victor.observability.sandhi_meter.SandhiMeter", _Meter)

        factory = MagicMock()
        factory.create_usage_logger.return_value = MagicMock()
        factory.create_streaming_metrics_collector.return_value = MagicMock()
        factory.create_metrics_collector.return_value = MagicMock()
        settings = SimpleNamespace(
            usage_gateway=SimpleNamespace(
                enabled=True,
                sink_path="/tmp/x.jsonl",
                subject_id="operator-default",
                group_id="ops-team",
            )
        )
        runtime = create_metrics_runtime_components(
            factory=factory,
            provider=SimpleNamespace(name="ollama"),
            model="qwen3-coder:30b",
            debug_logger=MagicMock(),
            cumulative_token_usage={"input_tokens": 0, "output_tokens": 0},
            tool_cost_lookup=lambda name: None,
            settings=settings,
        )
        tracker = runtime.session_cost_tracker.get_instance()

        with bind_attribution(subject_id="alice"):
            tracker.record_request(prompt_tokens=10, completion_tokens=5)
        tracker.record_request(prompt_tokens=10, completion_tokens=5)

        calls = recording[0].record_calls
        assert [c["subject_id"] for c in calls] == ["alice", "operator-default"]
        assert [c["group_id"] for c in calls] == ["ops-team", "ops-team"]
