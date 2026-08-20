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

"""FEP-0020 Phase 2 (M4) — usage_gateway nested settings group.

Default-off: a fresh Settings must carry a disabled usage_gateway group so the
base install stays byte-identical. Env overrides follow the standard nested
pattern (VICTOR_USAGE_GATEWAY__ENABLED etc.).
"""


class TestUsageGatewaySettingsDefaults:
    """Defaults for the usage_gateway nested group."""

    def test_defaults_disabled(self):
        """Fresh Settings → usage_gateway present, disabled, unattributed."""
        from victor.config.settings import Settings

        settings = Settings()

        assert settings.usage_gateway is not None
        assert settings.usage_gateway.enabled is False
        assert settings.usage_gateway.sink_path is None
        assert settings.usage_gateway.subject_id is None
        assert settings.usage_gateway.group_id is None

    def test_group_model_defaults_standalone(self):
        """The group model itself defaults to disabled."""
        from victor.config.groups.usage_gateway_config import UsageGatewaySettings

        ug = UsageGatewaySettings()
        assert ug.enabled is False
        assert ug.sink_path is None
        assert ug.subject_id is None
        assert ug.group_id is None


class TestUsageGatewayEnvOverride:
    """Nested env vars parse into the usage_gateway group."""

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("VICTOR_USAGE_GATEWAY__ENABLED", "true")
        monkeypatch.setenv("VICTOR_USAGE_GATEWAY__SUBJECT_ID", "alice")

        from victor.config.settings import Settings

        settings = Settings()

        assert settings.usage_gateway.enabled is True
        assert settings.usage_gateway.subject_id == "alice"
        # Unset fields keep their defaults.
        assert settings.usage_gateway.sink_path is None
        assert settings.usage_gateway.group_id is None
