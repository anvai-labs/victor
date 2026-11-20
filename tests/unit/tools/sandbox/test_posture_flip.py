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

"""FOU-S1b tests: sandbox posture flips to fail-closed by default (ADR-035).

No explicit policy = REQUIRED: shell execution is wrapped by an available
sandbox or refused with an actionable error. The escape hatches are explicit:
``sandbox_policy: "disabled"`` in settings or ``VICTOR_SANDBOX_POLICY``
in the environment; explicit opt-outs warn once per process.
"""

from types import SimpleNamespace

import pytest

from victor.tools.sandbox import backends as sb
from victor.tools.sandbox.backends import (
    SandboxRequiredUnavailableError,
    effective_sandbox_policy,
    resolve_sandbox_backend,
)


@pytest.fixture
def no_env(monkeypatch):
    """Remove the escape-hatch env var so tests see the true default."""
    monkeypatch.delenv("VICTOR_SANDBOX_POLICY", raising=False)


class TestDefaultPostureFlip:
    def test_no_settings_defaults_to_required(self, no_env):
        """ADR-035 decision 1: absent policy = required, not disabled."""
        assert effective_sandbox_policy(None) == "required"

    def test_settings_without_policy_default_required(self, no_env):
        settings = SimpleNamespace(sandbox_enabled=False, sandbox_policy=None)
        assert effective_sandbox_policy(settings) == "required"

    def test_legacy_enabled_bool_no_longer_maps_to_preferred(self, no_env):
        """Legacy opt-in users get the stronger posture; preferred is opt-in."""
        settings = SimpleNamespace(sandbox_enabled=True, sandbox_policy=None)
        assert effective_sandbox_policy(settings) == "required"

    def test_explicit_policy_still_wins(self, no_env):
        for policy in ("disabled", "preferred", "required"):
            settings = SimpleNamespace(sandbox_enabled=False, sandbox_policy=policy)
            assert effective_sandbox_policy(settings) == policy

    def test_env_escape_hatch_disabled(self, monkeypatch):
        monkeypatch.setenv("VICTOR_SANDBOX_POLICY", "disabled")
        assert effective_sandbox_policy(None) == "disabled"

    def test_env_escape_hatch_preferred(self, monkeypatch):
        monkeypatch.setenv("VICTOR_SANDBOX_POLICY", "preferred")
        assert effective_sandbox_policy(None) == "preferred"

    def test_env_invalid_value_ignored(self, monkeypatch):
        monkeypatch.setenv("VICTOR_SANDBOX_POLICY", "yolo")
        assert effective_sandbox_policy(None) == "required"

    def test_explicit_policy_beats_env(self, monkeypatch):
        monkeypatch.setenv("VICTOR_SANDBOX_POLICY", "disabled")
        settings = SimpleNamespace(sandbox_enabled=False, sandbox_policy="required")
        assert effective_sandbox_policy(settings) == "required"


class TestFailClosedRefusal:
    def test_unavailable_backend_refuses_execution(self, monkeypatch, no_env):
        """Negative: required policy + unavailable backend raises; callers
        must refuse dispatch (zero execution), never run unsandboxed."""
        monkeypatch.setattr(sb.BubblewrapSandbox, "available", lambda self: False)
        with pytest.raises(SandboxRequiredUnavailableError):
            resolve_sandbox_backend(None)

    def test_unsupported_platform_refuses_execution(self, monkeypatch, no_env):
        monkeypatch.setattr(sb.platform, "system", lambda: "Plan9")
        with pytest.raises(SandboxRequiredUnavailableError) as excinfo:
            resolve_sandbox_backend(None)
        assert "escape" in str(excinfo.value).lower() or "VICTOR_SANDBOX_POLICY" in str(
            excinfo.value
        )

    def test_available_backend_engages_by_default(self, monkeypatch, no_env):
        monkeypatch.setattr(sb.BubblewrapSandbox, "available", lambda self: True)
        backend = resolve_sandbox_backend(None)
        assert backend.type_name == "bwrap"


class TestOptOutWarning:
    def test_explicit_disabled_warns_once(self, monkeypatch, caplog, no_env):
        """ADR-035 decision 2: explicit opt-out logs a one-time warning."""
        monkeypatch.setattr(sb.platform, "system", lambda: "Linux")
        monkeypatch.setattr(sb, "_warned_categories", set())
        with caplog.at_level("WARNING", logger="victor.tools.sandbox.backends"):
            resolve_sandbox_backend(
                SimpleNamespace(sandbox_enabled=False, sandbox_policy="disabled")
            )
            resolve_sandbox_backend(
                SimpleNamespace(sandbox_enabled=False, sandbox_policy="disabled")
            )
        warnings = [r for r in caplog.records if "UNSANDBOXED" in r.message]
        assert len(warnings) == 1


class TestSnapshotDelegation:
    def test_policy_snapshot_delegates_to_canonical_derivation(self, no_env):
        """The snapshot's policy derivation must be the canonical function
        (the hand-synced copy drifted once already)."""
        from victor.runtime import policy_snapshot as ps

        for settings in (
            None,
            SimpleNamespace(sandbox_enabled=False, sandbox_policy=None),
            SimpleNamespace(sandbox_enabled=False, sandbox_policy="preferred"),
            SimpleNamespace(sandbox_enabled=False, sandbox_policy="required"),
        ):
            assert ps.RunPolicySnapshot._effective_policy(settings) == effective_sandbox_policy(
                settings
            )


class TestApiSurface:
    def test_status_response_reports_sandbox_posture(self):
        """The /system/status surface reports the effective posture."""
        from victor.integrations.api.fastapi_server import StatusResponse

        response = StatusResponse(
            connected=True,
            mode="chat",
            provider="ollama",
            model="q",
            workspace="/tmp",
        )
        assert hasattr(response, "sandbox_posture")
        assert response.sandbox_posture == "unknown"

    def test_current_sandbox_posture_helper(self, no_env):
        from victor.tools.sandbox import current_sandbox_posture

        assert current_sandbox_posture() in {"required", "preferred", "disabled"}


class TestReviewFixes:
    def test_posture_reports_unknown_on_settings_read_failure(self, monkeypatch):
        """A broken settings read must surface as 'unknown', never as a
        healthy-looking 'required' (adversarial review finding)."""

        import victor.config.settings as settings_mod
        from victor.tools.sandbox import current_sandbox_posture

        def _boom():
            raise RuntimeError("malformed profiles.yaml")

        monkeypatch.setattr(settings_mod, "get_settings", _boom)
        assert current_sandbox_posture() == "unknown"

    def test_settings_policy_matching_normalizes_case(self):
        """Settings-like objects that bypass pydantic validation still get
        the same strip/lower treatment the env path applies."""
        settings = SimpleNamespace(sandbox_policy="DISABLED")
        assert effective_sandbox_policy(settings) == "disabled"

    def test_warning_categories_are_independent(self, monkeypatch, caplog):
        """A preferred-degradation warning must not suppress a later
        explicit-opt-out warning (separate categories)."""
        monkeypatch.setattr(sb.platform, "system", lambda: "Plan9")
        monkeypatch.setattr(sb, "_warned_categories", set())
        with caplog.at_level("WARNING", logger="victor.tools.sandbox.backends"):
            resolve_sandbox_backend(SimpleNamespace(sandbox_policy="preferred"))
            resolve_sandbox_backend(SimpleNamespace(sandbox_policy="disabled"))
        messages = [r.message for r in caplog.records if "UNSANDBOXED" in r.message]
        assert len(messages) == 2
        monkeypatch.setattr(sb, "_warned_categories", set())


class TestDangerouslyUnsandboxedFlag:
    """ADR-035 decision 2: the per-invocation CLI escape hatch."""

    def _apply(self, monkeypatch, settings):
        import victor.tools.sandbox.backends as backends_mod

        monkeypatch.setattr(backends_mod, "_warned_categories", set())
        monkeypatch.setattr("victor.config.settings.get_settings", lambda: settings)
        from victor.tools.sandbox import force_unsandboxed_for_session

        return force_unsandboxed_for_session()

    def test_overrides_required_settings_policy(self, monkeypatch):
        """An explicit CLI request beats the settings file (strongest
        signal) - even a 'required' file policy is overridden."""
        settings = SimpleNamespace(sandbox=SimpleNamespace(sandbox_policy="required"))
        monkeypatch.delenv("VICTOR_SANDBOX_POLICY", raising=False)
        applied, message = self._apply(monkeypatch, settings)
        assert applied is True
        assert settings.sandbox.sandbox_policy == "disabled"
        assert "UNSANDBOXED" in message

    def test_none_sandbox_gains_disabled_settings(self, monkeypatch):
        from victor.config.sandbox_settings import SandboxSettings

        settings = SimpleNamespace(sandbox=None)
        applied, _ = self._apply(monkeypatch, settings)
        assert applied is True
        assert isinstance(settings.sandbox, SandboxSettings)
        assert settings.sandbox.sandbox_policy == "disabled"

    def test_settings_failure_stays_fail_closed(self, monkeypatch):
        def _boom():
            raise RuntimeError("malformed profiles.yaml")

        monkeypatch.setattr("victor.config.settings.get_settings", _boom)
        from victor.tools.sandbox import force_unsandboxed_for_session

        applied, message = force_unsandboxed_for_session()
        assert applied is False
        assert "fail-closed posture stays in effect" in message

    def test_effective_policy_reads_disabled_after_apply(self, monkeypatch):
        settings = SimpleNamespace(sandbox=SimpleNamespace(sandbox_policy="required"))
        applied, _ = self._apply(monkeypatch, settings)
        assert applied
        assert effective_sandbox_policy(settings.sandbox) == "disabled"

    def test_cli_option_registered(self):
        """The flag exists on the root callback with the ADR-035 name
        (typer derives --dangerously-unsandboxed from the parameter)."""
        import inspect

        from victor.ui.cli import app

        params = inspect.signature(app.registered_callback.callback).parameters
        assert "dangerously_unsandboxed" in params


class TestPolicyResolutionTriState:
    """Codex finding 20: policy resolution is tri-state — a fault INSIDE
    effective-policy evaluation is UNKNOWN, and unknown refuses BEFORE
    launch. The historical behavior returned an inactive, non-refusing
    decision, which a faulty resolver turned into permission to proceed."""

    class _BrokenSettings:
        """Settings whose policy field faults on read (the settings OBJECT
        loads fine — this is not the settings-load failure path)."""

        @property
        def sandbox_policy(self):
            raise RuntimeError("unreadable policy field")

    def test_resolve_effective_policy_reports_unknown_on_fault(self):
        resolution = sb.resolve_effective_policy(self._BrokenSettings())
        assert resolution.unknown
        assert not resolution.known
        assert resolution.policy == ""

    def test_resolve_effective_policy_reports_known_policies(self, no_env):
        for policy in ("disabled", "preferred", "required"):
            resolution = sb.resolve_effective_policy(SimpleNamespace(sandbox_policy=policy))
            assert resolution.known
            assert resolution.policy == policy

    def test_raising_accessor_refuses_required_launch(self):
        decision = sb.resolve_required_launch(self._BrokenSettings())
        assert decision.refuse
        assert decision.required_active is False
        assert decision.backend is None
        assert "policy" in decision.reason.lower()

    def test_explicit_disabled_is_known_and_non_refusing(self):
        """The third state: an explicit, successfully-read opt-out keeps its
        historical non-active semantics (a negative control — the fail-closed
        flip above must not swallow legitimate disabled mode)."""
        decision = sb.resolve_required_launch(SimpleNamespace(sandbox_policy="disabled"))
        assert decision.refuse is False
        assert decision.required_active is False
