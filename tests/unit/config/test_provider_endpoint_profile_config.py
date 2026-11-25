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

"""Provider-config dialect fields: ``endpoint_profile`` and ``protocol``.

``ProviderConfig`` forbids unknown keys, so a ``providers:`` section that
declares a Sandhi endpoint profile (e.g. ``openai_chat_completions`` for the
``max_completion_tokens`` dialect) was previously rejected wholesale at load —
the profile silently fell back to heuristics. These tests pin that the two
dialect fields load and reach the settings dump the provider registry feeds
to ``ProviderRegistry.create(**kwargs)`` (the transport reads both from the
provider's extra config), while genuinely unknown fields stay rejected.
"""

from unittest.mock import patch

import yaml

from victor.config.settings import ProviderConfig, Settings


def _patch_config_dir(tmp_path):
    return patch.object(Settings, "get_config_dir", classmethod(lambda cls: tmp_path))


def _write_user_profiles(tmp_path, text: str) -> None:
    (tmp_path / "profiles.yaml").write_text(text, encoding="utf-8")


class TestEndpointProfileProviderConfig:
    def test_dialect_fields_load_from_providers_section(self, tmp_path):
        _write_user_profiles(
            tmp_path,
            yaml.safe_dump(
                {
                    "providers": {
                        "openai": {
                            "base_url": "https://api.openai.com/v1",
                            "endpoint_profile": "openai_chat_completions",
                            "protocol": "responses",
                        }
                    }
                }
            ),
        )
        with _patch_config_dir(tmp_path):
            cfg = Settings.load_provider_config("openai")
        assert cfg is not None
        assert cfg.endpoint_profile == "openai_chat_completions"
        assert cfg.protocol == "responses"

    def test_dialect_fields_reach_the_registry_settings_dump(self, tmp_path):
        """The registry feeds ``model_dump(exclude_none=True)`` to provider kwargs."""
        cfg = ProviderConfig(
            endpoint_profile="openai_chat_completions",
            protocol="responses",
        )
        dumped = cfg.model_dump(exclude_none=True)
        assert dumped["endpoint_profile"] == "openai_chat_completions"
        assert dumped["protocol"] == "responses"

    def test_unset_dialect_fields_are_excluded_from_the_dump(self, tmp_path):
        # The transport treats a present-but-empty value as authoritative; the
        # dump must omit unset fields so default profiles keep heuristic routing.
        dumped = ProviderConfig().model_dump(exclude_none=True)
        assert "endpoint_profile" not in dumped
        assert "protocol" not in dumped

    def test_unknown_provider_config_field_still_rejected(self, tmp_path):
        """Admitting the dialect fields must not loosen the forbid policy."""
        _write_user_profiles(
            tmp_path,
            yaml.safe_dump({"providers": {"openai": {"bogus_dialect_key": "whatever"}}}),
        )
        with _patch_config_dir(tmp_path):
            cfg = Settings.load_provider_config("openai")
        assert cfg is None

    def test_dialect_fields_round_trip_through_load(self, tmp_path):
        """load_provider_config → model_dump is the exact handoff to the registry."""
        _write_user_profiles(
            tmp_path,
            yaml.safe_dump(
                {"providers": {"openai": {"endpoint_profile": "openai_chat_completions"}}}
            ),
        )
        with _patch_config_dir(tmp_path):
            cfg = Settings.load_provider_config("openai")
        assert cfg is not None
        dumped = cfg.model_dump(exclude_none=True)
        assert dumped["endpoint_profile"] == "openai_chat_completions"
