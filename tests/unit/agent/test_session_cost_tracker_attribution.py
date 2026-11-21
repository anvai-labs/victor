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

import pytest

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


class TestVerifiedPrincipalGroupChange:
    """Finding 23: a principal's group change must reach subsequent events.

    The A4a auth seam binds the principal's CURRENT primary_group at event
    time (``bind_principal_attribution`` → execution context → tracker), and
    the SandhiMeter realizes the billing group per event via the versioned
    metering key — so a registry group change is reflected in every event
    after it, without re-registering a subject key.
    """

    def test_group_change_between_calls_reaches_recorded_events(self):
        pytest.importorskip("sandhi_gateway", reason="requires the victor[sandhi] extra")
        from victor.integrations.api.auth.principal import (
            AuthMethod,
            PrincipalKind,
            VerifiedPrincipal,
            bind_principal_attribution,
        )
        from victor.observability.sandhi_meter import SandhiMeter

        meter = SandhiMeter()
        tracker = SessionCostTracker(provider="ollama", model="m")
        tracker._sandhi = meter

        def principal(group: str) -> VerifiedPrincipal:
            return VerifiedPrincipal(
                principal_id="alice",
                kind=PrincipalKind.HUMAN,
                auth_method=AuthMethod.OIDC,
                subject="alice@idp.example",
                groups=(group, "billing-viewers"),
            )

        with bind_principal_attribution(principal("team-a")):
            tracker.record_request(prompt_tokens=10, completion_tokens=5)
        # The registry moved alice to team-b; the next authenticated call
        # re-resolves the principal and binds the new primary group.
        with bind_principal_attribution(principal("team-b")):
            tracker.record_request(prompt_tokens=10, completion_tokens=5)

        events = meter.events()
        assert [e["group_id"] for e in events] == ["team-a", "team-b"]
        assert [e["virtual_key_id"] for e in events] == [
            "victor:alice@idp.example:team-a",
            "victor:alice@idp.example:team-b",
        ]
        # The stable subject (D5) never changed across the group change.
        assert {e["subject_id"] for e in events} == {"alice@idp.example"}

    def test_legacy_key_only_principal_binds_no_group(self):
        pytest.importorskip("sandhi_gateway", reason="requires the victor[sandhi] extra")
        from victor.integrations.api.auth.principal import (
            legacy_api_key_principal,
            bind_principal_attribution,
        )
        from victor.observability.sandhi_meter import SandhiMeter

        meter = SandhiMeter()
        tracker = SessionCostTracker(provider="ollama", model="m")
        tracker._sandhi = meter

        legacy = legacy_api_key_principal(api_key="vk_secret", client_id="legacy-key")
        assert legacy.groups == () and legacy.primary_group is None

        with bind_principal_attribution(legacy):
            tracker.record_request(prompt_tokens=10, completion_tokens=5)

        ev = meter.events()[0]
        # Behavior-neutral pin: no verified group ⇒ the unversioned key-only
        # virtual key with no group attribution, exactly as before finding 23.
        assert ev["group_id"] is None
        assert ev["virtual_key_id"] == "victor:legacy-key"


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
