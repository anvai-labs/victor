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

"""Tests for bundled default profiles (ship profiles.yaml with victor-ai).

The CLI needs ~/.victor/profiles.yaml to function; the package now bundles
victor/config/profiles.default.yaml, seeds it on first run, and layers user
profiles over the bundled defaults at load time.
"""

from unittest.mock import patch

from victor.config.settings import Settings


def _patch_config_dir(tmp_path):
    return patch.object(Settings, "get_config_dir", classmethod(lambda cls: tmp_path))


class TestBundledResource:
    def test_bundled_resource_is_packaged(self):
        from importlib import resources

        ref = resources.files("victor.config").joinpath("profiles.default.yaml")
        text = ref.read_text(encoding="utf-8")
        assert "profiles:" in text
        assert "ANTHROPIC_API_KEY" in text  # keyless: keys documented as env vars
        assert "sk-" not in text  # never ship anything key-shaped

    def test_bundled_defaults_parse(self):
        text = Settings._bundled_default_profiles_text()
        assert text is not None
        raw = Settings._parse_profiles_yaml(text)
        assert "default" in raw
        assert "local" in raw
        assert raw["default"]["provider"] == "ollama"


class TestSeedOnFirstRun:
    def test_missing_user_file_is_seeded(self, tmp_path):
        with _patch_config_dir(tmp_path):
            profiles = Settings.load_profiles()
        seeded = tmp_path / "profiles.yaml"
        assert seeded.exists()
        assert "default" in profiles
        assert profiles["default"].provider == "ollama"

    def test_existing_user_file_never_overwritten(self, tmp_path):
        user_yaml = "profiles:\n  mine:\n    provider: zai\n    model: glm-5.2\n"
        (tmp_path / "profiles.yaml").write_text(user_yaml)
        with _patch_config_dir(tmp_path):
            Settings.load_profiles()
        assert (tmp_path / "profiles.yaml").read_text() == user_yaml


class TestLayeredMerge:
    def test_user_profiles_override_bundled_by_name(self, tmp_path):
        (tmp_path / "profiles.yaml").write_text(
            "profiles:\n  default:\n    provider: zai\n    model: glm-5.2\n"
        )
        with _patch_config_dir(tmp_path):
            profiles = Settings.load_profiles()
        assert profiles["default"].provider == "zai"
        assert profiles["default"].model == "glm-5.2"

    def test_bundled_profiles_appear_alongside_user_ones(self, tmp_path):
        (tmp_path / "profiles.yaml").write_text(
            "profiles:\n  mine:\n    provider: zai\n    model: glm-5.2\n"
        )
        with _patch_config_dir(tmp_path):
            profiles = Settings.load_profiles()
        assert "mine" in profiles
        # New bundled profiles surface automatically after upgrades.
        assert "local" in profiles

    def test_per_profile_merge_is_key_level(self, tmp_path):
        # User overrides only the model — bundled provider/temperature remain.
        (tmp_path / "profiles.yaml").write_text(
            "profiles:\n  local:\n    provider: ollama\n    model: llama3.3:70b\n"
        )
        with _patch_config_dir(tmp_path):
            profiles = Settings.load_profiles()
        assert profiles["local"].model == "llama3.3:70b"
        assert profiles["local"].max_tokens == 4096  # from bundled layer

    def test_invalid_profile_skipped_not_fatal(self, tmp_path):
        (tmp_path / "profiles.yaml").write_text(
            "profiles:\n  broken:\n    model: no-provider-set\n"
            "  good:\n    provider: ollama\n    model: qwen2.5-coder:7b\n"
        )
        with _patch_config_dir(tmp_path):
            profiles = Settings.load_profiles()
        assert "broken" not in profiles
        assert "good" in profiles
