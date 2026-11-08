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

"""Tests for the central profile layer (~/.victor/profiles.d, Phase 5b).

``victor profiles pull`` lands a per-user profile downloaded from the sandhi
gateway (``GET /auth/profile``, ``skp_`` bootstrap token, shown-once virtual
key) into ``~/.victor/profiles.d/sandhi.yaml``. These tests pin the layering
contract: bundled < central(profiles.d) < user(profiles.yaml), user still
winning key-by-key — plus the failure-tolerance rules (missing dir is a
no-op, malformed central file is warned about and skipped) and the two
consumption paths (profile selection via ``from_sources``, provider gateway
via ``load_provider_config``).
"""

import logging
from unittest.mock import patch

import pytest
import yaml

from victor.config.settings import Settings


def _patch_config_dir(tmp_path):
    return patch.object(Settings, "get_config_dir", classmethod(lambda cls: tmp_path))


def _write_central(tmp_path, filename: str, doc: dict) -> None:
    central_dir = tmp_path / "profiles.d"
    central_dir.mkdir(exist_ok=True)
    (central_dir / filename).write_text(yaml.safe_dump(doc), encoding="utf-8")


USER_FILE = "profiles:\n  mine:\n    provider: zai\n    model: glm-5.2\n"


class TestCentralProfilesLayer:
    def test_missing_profiles_d_is_noop(self, tmp_path):
        (tmp_path / "profiles.yaml").write_text(USER_FILE)
        with _patch_config_dir(tmp_path):
            profiles = Settings.load_profiles()
        assert "mine" in profiles
        # Bundled profiles still present — nothing central to merge.
        assert "local" in profiles

    def test_central_profile_visible_without_user_entry(self, tmp_path):
        (tmp_path / "profiles.yaml").write_text(USER_FILE)
        _write_central(
            tmp_path,
            "sandhi.yaml",
            {
                "profiles": {
                    "sandhi": {
                        "provider": "anthropic",
                        "model": "claude-sonnet-5-5",
                        "max_tokens": 8192,
                    }
                }
            },
        )
        with _patch_config_dir(tmp_path):
            profiles = Settings.load_profiles()
        assert profiles["sandhi"].provider == "anthropic"
        assert profiles["sandhi"].model == "claude-sonnet-5-5"
        assert profiles["sandhi"].max_tokens == 8192

    def test_user_wins_over_central_key_by_key(self, tmp_path):
        (tmp_path / "profiles.yaml").write_text(
            "profiles:\n  sandhi:\n    provider: zai\n    model: glm-5.2\n"
        )
        _write_central(
            tmp_path,
            "sandhi.yaml",
            {
                "profiles": {
                    "sandhi": {
                        "provider": "anthropic",
                        "model": "claude-sonnet-5-5",
                        "max_tokens": 8192,
                    }
                }
            },
        )
        with _patch_config_dir(tmp_path):
            profiles = Settings.load_profiles()
        # User overrides both keys it names; the central max_tokens (a key the
        # user file omits) still fills through.
        assert profiles["sandhi"].provider == "zai"
        assert profiles["sandhi"].model == "glm-5.2"
        assert profiles["sandhi"].max_tokens == 8192

    def test_central_fills_bundled_gaps(self, tmp_path):
        # Central overrides only the model of a bundled profile — the bundled
        # provider/temperature/max_tokens remain beneath it.
        (tmp_path / "profiles.yaml").write_text(USER_FILE)
        _write_central(
            tmp_path,
            "sandhi.yaml",
            {"profiles": {"local": {"model": "claude-sonnet-5-5"}}},
        )
        with _patch_config_dir(tmp_path):
            profiles = Settings.load_profiles()
        assert profiles["local"].model == "claude-sonnet-5-5"
        assert profiles["local"].max_tokens == 4096  # from the bundled layer

    def test_central_files_merge_in_sorted_filename_order(self, tmp_path):
        (tmp_path / "profiles.yaml").write_text(USER_FILE)
        _write_central(
            tmp_path,
            "00-team.yaml",
            {"profiles": {"sandhi": {"provider": "openai", "model": "gpt-5.2"}}},
        )
        _write_central(
            tmp_path,
            "10-team.yaml",
            {"profiles": {"sandhi": {"provider": "anthropic", "model": "claude-sonnet-5-5"}}},
        )
        with _patch_config_dir(tmp_path):
            profiles = Settings.load_profiles()
        assert profiles["sandhi"].provider == "anthropic"  # later filename wins

    def test_malformed_central_file_skipped_with_warning(self, tmp_path, caplog):
        (tmp_path / "profiles.yaml").write_text(USER_FILE)
        central_dir = tmp_path / "profiles.d"
        central_dir.mkdir()
        (central_dir / "broken.yaml").write_text("profiles: [unclosed", encoding="utf-8")
        _write_central(
            tmp_path,
            "good.yaml",
            {"profiles": {"sandhi": {"provider": "anthropic", "model": "claude-sonnet-5-5"}}},
        )
        with _patch_config_dir(tmp_path):
            with caplog.at_level(logging.WARNING, logger="victor.config.settings"):
                profiles = Settings.load_profiles()
        assert "sandhi" in profiles  # the well-formed sibling still lands
        assert any("broken.yaml" in rec.message for rec in caplog.records)

    def test_invalid_central_profile_entry_skipped_not_fatal(self, tmp_path):
        (tmp_path / "profiles.yaml").write_text(USER_FILE)
        _write_central(
            tmp_path,
            "sandhi.yaml",
            {
                "profiles": {
                    "broken": {"model": "no-provider-set"},
                    "sandhi": {"provider": "anthropic", "model": "claude-sonnet-5-5"},
                }
            },
        )
        with _patch_config_dir(tmp_path):
            profiles = Settings.load_profiles()
        assert "broken" not in profiles
        assert "sandhi" in profiles


class TestCentralProfileSelection:
    def test_from_sources_picks_central_defined_profile(self, tmp_path):
        _write_central(
            tmp_path,
            "sandhi.yaml",
            {"profiles": {"sandhi": {"provider": "anthropic", "model": "claude-sonnet-5-5"}}},
        )
        settings = Settings.from_sources(profile_name="sandhi", config_dir=tmp_path)
        assert settings.provider.default_provider == "anthropic"
        assert settings.provider.default_model == "claude-sonnet-5-5"

    def test_from_sources_user_file_shadows_central(self, tmp_path):
        (tmp_path / "profiles.yaml").write_text(
            "profiles:\n  sandhi:\n    provider: zai\n    model: glm-5.2\n"
        )
        _write_central(
            tmp_path,
            "sandhi.yaml",
            {"profiles": {"sandhi": {"provider": "anthropic", "model": "claude-sonnet-5-5"}}},
        )
        settings = Settings.from_sources(profile_name="sandhi", config_dir=tmp_path)
        assert settings.provider.default_provider == "zai"


class TestCentralProviderGateway:
    """The gateway block a pull writes must resolve like a hand-edited one."""

    CENTRAL_DOC = {
        "providers": {
            "anthropic": {
                "gateway": {
                    "url": "https://gw.example:18788",
                    "virtual_key": "vk-central-secret",
                }
            }
        }
    }

    def test_gateway_resolved_from_central_layer(self, tmp_path):
        (tmp_path / "profiles.yaml").write_text(USER_FILE)
        _write_central(tmp_path, "sandhi.yaml", self.CENTRAL_DOC)
        with _patch_config_dir(tmp_path):
            cfg = Settings.load_provider_config("anthropic")
        assert cfg is not None
        assert cfg.gateway.url == "https://gw.example:18788"
        assert cfg.gateway.virtual_key_value == "vk-central-secret"

    def test_user_provider_section_wins_over_central(self, tmp_path):
        (tmp_path / "profiles.yaml").write_text(
            "providers:\n"
            "  anthropic:\n"
            "    gateway:\n"
            "      url: https://user-gw.example\n"
            "      virtual_key: vk-user-secret\n"
        )
        _write_central(tmp_path, "sandhi.yaml", self.CENTRAL_DOC)
        with _patch_config_dir(tmp_path):
            cfg = Settings.load_provider_config("anthropic")
        assert cfg.gateway.url == "https://user-gw.example"
        assert cfg.gateway.virtual_key_value == "vk-user-secret"

    def test_central_used_when_user_file_absent(self, tmp_path):
        # No profiles.yaml at all — the central providers section still lands.
        _write_central(tmp_path, "sandhi.yaml", self.CENTRAL_DOC)
        with _patch_config_dir(tmp_path):
            cfg = Settings.load_provider_config("anthropic")
        assert cfg is not None
        assert cfg.gateway.url == "https://gw.example:18788"


@pytest.mark.parametrize("filename", ["sandhi.yml"])
def test_non_yaml_extension_ignored(tmp_path, filename):
    """Only *.yaml is picked up — a .yml sibling is not silently half-loaded."""
    (tmp_path / "profiles.yaml").write_text(USER_FILE)
    central_dir = tmp_path / "profiles.d"
    central_dir.mkdir()
    (central_dir / filename).write_text(
        yaml.safe_dump({"profiles": {"sandhi": {"provider": "anthropic"}}}), encoding="utf-8"
    )
    with _patch_config_dir(tmp_path):
        profiles = Settings.load_profiles()
    assert "sandhi" not in profiles
