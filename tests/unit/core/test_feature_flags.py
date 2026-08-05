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

"""Tests for the feature flag system."""

import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import yaml

from victor.core.feature_flags import (
    FeatureFlag,
    FeatureFlagManager,
    FeatureFlagConfig,
    get_feature_flag_manager,
    reset_feature_flag_manager,
    is_feature_enabled,
    enable_feature,
    disable_feature,
)
from victor.config.feature_config import (
    load_feature_flags_from_settings,
    load_feature_flags_from_yaml,
    save_feature_flags_to_yaml,
    validate_feature_flags,
    get_feature_flag_summary,
)


class TestFeatureFlagEnum:
    """Tests for FeatureFlag enum."""

    def test_all_flags_have_env_var_names(self):
        """Test that all feature flags have valid environment variable names."""
        for flag in FeatureFlag:
            env_var = flag.get_env_var_name()
            assert env_var.startswith("VICTOR_")
            assert env_var.isupper()

    def test_all_flags_have_yaml_keys(self):
        """Test that all feature flags have valid YAML keys."""
        for flag in FeatureFlag:
            yaml_key = flag.get_yaml_key()
            assert yaml_key.islower()
            assert " " not in yaml_key

    def test_flag_values_are_unique(self):
        """Test that all feature flag values are unique."""
        values = [flag.value for flag in FeatureFlag]
        assert len(values) == len(set(values))

    def test_rollout_flags_are_opt_in_by_default(self):
        """New rollout flags should stay disabled unless explicitly enabled."""
        config = FeatureFlagConfig(default_enabled=True)
        manager = FeatureFlagManager(config)

        assert not manager.is_enabled(FeatureFlag.USE_CALIBRATED_COMPLETION)
        assert not manager.is_enabled(FeatureFlag.USE_PROMPT_DICTIONARY_COMPRESSION)


class TestFeatureFlagManager:
    """Tests for FeatureFlagManager."""

    def test_default_manager_state(self):
        """Test that manager initializes with default state."""
        config = FeatureFlagConfig(default_enabled=False)
        manager = FeatureFlagManager(config)

        # All flags should be disabled by default
        for flag in FeatureFlag:
            assert not manager.is_enabled(flag)

    def test_default_enabled_true(self):
        """Test manager with default_enabled=True."""
        config = FeatureFlagConfig(default_enabled=True)
        manager = FeatureFlagManager(config)

        for flag in FeatureFlag:
            expected = not flag.is_opt_in_by_default()
            assert manager.is_enabled(flag) is expected

    def test_enable_flag_at_runtime(self):
        """Test enabling a flag at runtime."""
        config = FeatureFlagConfig(default_enabled=False)
        manager = FeatureFlagManager(config)

        # USE_SEMANTIC_RESPONSE_CACHE is opt-in, so it's disabled by default
        assert not manager.is_enabled(FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE)

        manager.enable(FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE)
        assert manager.is_enabled(FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE)

    def test_disable_flag_at_runtime(self):
        """Test disabling a flag at runtime."""
        config = FeatureFlagConfig(default_enabled=False)
        manager = FeatureFlagManager(config)

        # Enable the flag first
        manager.enable(FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE)
        assert manager.is_enabled(FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE)

        # Then disable it
        manager.disable(FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE)
        assert not manager.is_enabled(FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE)

    def test_runtime_override_precedence(self):
        """Test that runtime overrides take precedence over env vars."""
        manager = FeatureFlagManager()

        # Enable via runtime
        manager.enable(FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE)
        assert manager.is_enabled(FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE)

        # Set env var to false (should be ignored due to runtime override)
        with mock.patch.dict(os.environ, {"VICTOR_USE_SEMANTIC_RESPONSE_CACHE": "false"}):
            assert manager.is_enabled(FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE)

    def test_clear_runtime_override(self):
        """Test clearing runtime override."""
        config = FeatureFlagConfig(default_enabled=False)
        manager = FeatureFlagManager(config)

        manager.enable(FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE)
        assert manager.is_enabled(FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE)

        manager.clear_runtime_override(FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE)
        assert not manager.is_enabled(FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE)

    def test_get_enabled_flags(self):
        """Test getting all enabled flags."""
        config = FeatureFlagConfig(default_enabled=False)
        manager = FeatureFlagManager(config)

        manager.enable(FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE)
        manager.enable(FeatureFlag.USE_SMART_ROUTING)

        enabled = manager.get_enabled_flags()
        assert enabled[FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE] is True
        assert enabled[FeatureFlag.USE_SMART_ROUTING] is True
        # Non-opt-in flags fall back to default_enabled (False here)
        assert enabled[FeatureFlag.USE_LEARNING_FROM_EXECUTION] is False

    def test_set_convenience_method(self):
        """Test the set() convenience method."""
        manager = FeatureFlagManager()

        manager.set(FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE, True)
        assert manager.is_enabled(FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE)

        manager.set(FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE, False)
        assert not manager.is_enabled(FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE)

    def test_reload_config(self):
        """Test reloading configuration."""
        config = FeatureFlagConfig(default_enabled=False)
        manager = FeatureFlagManager(config)

        # Enable via runtime
        manager.enable(FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE)
        assert manager.is_enabled(FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE)

        # Reload should clear runtime overrides
        manager.reload_config()
        assert not manager.is_enabled(FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE)


class TestEnvironmentVariableLoading:
    """Tests for environment variable loading."""

    def test_env_var_true(self):
        """Test loading true from environment variable."""
        with mock.patch.dict(os.environ, {"VICTOR_USE_SEMANTIC_RESPONSE_CACHE": "true"}):
            manager = FeatureFlagManager()
            assert manager.is_enabled(FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE)

    def test_env_var_false(self):
        """Test loading false from environment variable."""
        with mock.patch.dict(os.environ, {"VICTOR_USE_SEMANTIC_RESPONSE_CACHE": "false"}):
            manager = FeatureFlagManager()
            assert not manager.is_enabled(FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE)

    @pytest.mark.parametrize("value", ["1", "yes", "on", "YES", "ON"])
    def test_env_var_truthy_values(self, value):
        """Test various truthy values for environment variables."""
        with mock.patch.dict(os.environ, {"VICTOR_USE_SEMANTIC_RESPONSE_CACHE": value}):
            manager = FeatureFlagManager()
            assert manager.is_enabled(FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE)

    @pytest.mark.parametrize("value", ["0", "no", "off", "NO", "OFF", "invalid"])
    def test_env_var_falsy_values(self, value):
        """Test various falsy values for environment variables."""
        with mock.patch.dict(os.environ, {"VICTOR_USE_SEMANTIC_RESPONSE_CACHE": value}):
            manager = FeatureFlagManager()
            assert not manager.is_enabled(FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE)


class TestYamlConfiguration:
    """Tests for YAML configuration loading."""

    def test_load_from_valid_yaml(self):
        """Test loading from valid YAML configuration."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(
                {
                    "features": {
                        "use_semantic_response_cache": True,
                        "use_smart_routing": False,
                    }
                },
                f,
            )
            config_path = Path(f.name)

        try:
            config = FeatureFlagConfig(config_path=config_path)
            manager = FeatureFlagManager(config)

            assert manager.is_enabled(FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE)
            assert not manager.is_enabled(FeatureFlag.USE_SMART_ROUTING)
            # Flags not in YAML fall back to default_enabled (False)
            assert manager.is_enabled(FeatureFlag.USE_LEARNING_FROM_EXECUTION)
        finally:
            config_path.unlink()

    def test_load_from_missing_yaml(self):
        """Test loading from missing YAML file."""
        config = FeatureFlagConfig(
            config_path=Path("/nonexistent/path/features.yaml"), default_enabled=False
        )
        manager = FeatureFlagManager(config)

        # Should use defaults (False)
        assert not manager.is_enabled(FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE)

    def test_load_from_invalid_yaml_strict_mode(self):
        """Test loading from invalid YAML in strict mode."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content:")
            config_path = Path(f.name)

        try:
            config = FeatureFlagConfig(config_path=config_path, strict_mode=True)
            with pytest.raises(Exception):
                manager = FeatureFlagManager(config)
        finally:
            config_path.unlink()

    def test_load_from_invalid_yaml_lenient_mode(self):
        """Test loading from invalid YAML in lenient mode."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content:")
            config_path = Path(f.name)

        try:
            config = FeatureFlagConfig(
                config_path=config_path, strict_mode=False, default_enabled=False
            )
            manager = FeatureFlagManager(config)

            # Should use defaults (False)
            assert not manager.is_enabled(FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE)
        finally:
            config_path.unlink()

    def test_invalid_feature_flag_value_strict(self):
        """Test invalid feature flag value in strict mode."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(
                {
                    "features": {
                        "use_semantic_response_cache": "not_a_bool",
                    }
                },
                f,
            )
            config_path = Path(f.name)

        try:
            config = FeatureFlagConfig(config_path=config_path, strict_mode=True)
            with pytest.raises(ValueError):
                manager = FeatureFlagManager(config)
        finally:
            config_path.unlink()

    def test_invalid_feature_flag_value_lenient(self):
        """Test invalid feature flag value in lenient mode."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(
                {
                    "features": {
                        "use_semantic_response_cache": "not_a_bool",
                    }
                },
                f,
            )
            config_path = Path(f.name)

        try:
            config = FeatureFlagConfig(
                config_path=config_path, strict_mode=False, default_enabled=False
            )
            manager = FeatureFlagManager(config)

            # Should ignore invalid value and use default (False)
            assert not manager.is_enabled(FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE)
        finally:
            config_path.unlink()


class TestGlobalManager:
    """Tests for global feature flag manager."""

    def test_get_global_manager_singleton(self):
        """Test that get_feature_flag_manager returns singleton."""
        reset_feature_flag_manager()

        manager1 = get_feature_flag_manager()
        manager2 = get_feature_flag_manager()

        assert manager1 is manager2

    def test_force_reload(self):
        """Test force_reload parameter."""
        reset_feature_flag_manager()

        config = FeatureFlagConfig(default_enabled=False)
        manager1 = get_feature_flag_manager(config=config)
        manager1.enable(FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE)

        manager2 = get_feature_flag_manager(config=config, force_reload=True)

        # New instance should not have runtime override
        assert not manager2.is_enabled(FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE)

    def test_convenience_functions(self):
        """Test convenience functions."""
        reset_feature_flag_manager()

        # Default is now enabled, so test enable/disable cycle
        config = FeatureFlagConfig(default_enabled=False)
        get_feature_flag_manager(config=config, force_reload=True)

        assert not is_feature_enabled(FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE)

        enable_feature(FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE)
        assert is_feature_enabled(FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE)

        disable_feature(FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE)
        assert not is_feature_enabled(FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE)


class TestFeatureConfigModule:
    """Tests for victor.config.feature_config module."""

    def test_load_from_settings(self):
        """Test loading feature flags from settings."""
        flags = load_feature_flags_from_settings()

        # Should return all expected flags
        assert "use_semantic_response_cache" in flags
        assert "use_smart_routing" in flags
        assert "use_learning_from_execution" in flags

    def test_load_from_yaml_config(self):
        """Test loading from YAML configuration file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(
                {
                    "features": {
                        "use_semantic_response_cache": True,
                        "use_smart_routing": False,
                    }
                },
                f,
            )
            config_path = Path(f.name)

        try:
            flags = load_feature_flags_from_yaml(config_path)

            assert flags["use_semantic_response_cache"] is True
            assert flags["use_smart_routing"] is False
        finally:
            config_path.unlink()

    def test_save_to_yaml_config(self):
        """Test saving to YAML configuration file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "features.yaml"

            flags = {
                "use_semantic_response_cache": True,
                "use_smart_routing": False,
                "use_learning_from_execution": True,
            }

            save_feature_flags_to_yaml(flags, config_path)

            # Verify file was created
            assert config_path.exists()

            # Verify contents
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)

            assert data["features"]["use_semantic_response_cache"] is True
            assert data["features"]["use_smart_routing"] is False
            assert data["features"]["use_learning_from_execution"] is True

    def test_validate_valid_flags(self):
        """Test validation of valid flags."""
        flags = {
            "use_semantic_response_cache": True,
            "use_smart_routing": False,
        }
        assert validate_feature_flags(flags) is True

    def test_validate_invalid_flag_name(self):
        """Test validation rejects invalid flag names."""
        flags = {
            "use_semantic_response_cache": True,
            "invalid_flag_name": False,
        }
        assert validate_feature_flags(flags) is False

    def test_validate_invalid_flag_value(self):
        """Test validation rejects invalid flag values."""
        flags = {
            "use_semantic_response_cache": "not_a_bool",
        }
        assert validate_feature_flags(flags) is False

    def test_get_feature_flag_summary(self):
        """Test getting feature flag summary."""
        # Pass flags directly to avoid environment variable interference
        flags = {
            "use_semantic_response_cache": True,
            "use_smart_routing": False,
        }
        summary = get_feature_flag_summary(flags)

        assert "Feature Flags Status:" in summary
        assert "ENABLED" in summary
        assert "DISABLED" in summary
        assert "use_semantic_response_cache" in summary
        assert "use_smart_routing" in summary


class TestThreadSafety:
    """Tests for thread safety of feature flag manager."""

    def test_concurrent_access(self):
        """Test concurrent access to feature flag manager."""
        import threading

        manager = FeatureFlagManager()
        results = []

        def enable_flag():
            for _ in range(100):
                manager.enable(FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE)
                assert manager.is_enabled(FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE)
                manager.disable(FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE)
                assert not manager.is_enabled(FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE)

        threads = [threading.Thread(target=enable_flag) for _ in range(10)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Should complete without errors
        assert True


class TestConsolidationFeatureFlags:
    """Tests for architecture consolidation feature flags (Phase 15).

    Only ``USE_STATEGRAPH_AGENTIC_LOOP`` remains — the other three Phase-15
    placeholders (``USE_FRAMEWORK_TEAMS``, ``USE_FRAMEWORK_COORDINATORS``,
    ``USE_CONTEXT_SERVICE_INJECTION``) were removed because they were never
    read by production code. Coordinator/team consolidation now ships
    unconditionally.
    """

    def test_consolidation_flags_exist(self):
        """Test that the surviving consolidation feature flag is defined."""
        assert hasattr(FeatureFlag, "USE_STATEGRAPH_AGENTIC_LOOP")

    def test_consolidation_flags_have_valid_env_vars(self):
        """Test that consolidation flags have valid environment variable names."""
        assert (
            FeatureFlag.USE_STATEGRAPH_AGENTIC_LOOP.get_env_var_name()
            == "VICTOR_USE_STATEGRAPH_AGENTIC_LOOP"
        )

    def test_consolidation_flags_default_to_false(self):
        """Test that consolidation flags are opt-in (disabled by default)."""
        config = FeatureFlagConfig(default_enabled=True)
        manager = FeatureFlagManager(config)

        # Consolidation flag should default to False (opt-in for safety)
        assert not manager.is_enabled(FeatureFlag.USE_STATEGRAPH_AGENTIC_LOOP)

    def test_consolidation_flags_can_be_enabled_via_env(self):
        """Test that consolidation flags can be enabled via environment variables."""
        import os

        # Set environment variable
        os.environ["VICTOR_USE_STATEGRAPH_AGENTIC_LOOP"] = "true"

        try:
            # Reset manager to pick up new env vars
            reset_feature_flag_manager()
            manager = get_feature_flag_manager()

            assert manager.is_enabled(FeatureFlag.USE_STATEGRAPH_AGENTIC_LOOP)
        finally:
            # Clean up
            del os.environ["VICTOR_USE_STATEGRAPH_AGENTIC_LOOP"]
            reset_feature_flag_manager()

    def test_consolidation_flags_can_be_enabled_at_runtime(self):
        """Test that consolidation flags can be enabled at runtime."""
        config = FeatureFlagConfig(default_enabled=False)
        manager = FeatureFlagManager(config)

        # Should be disabled initially
        assert not manager.is_enabled(FeatureFlag.USE_STATEGRAPH_AGENTIC_LOOP)

        # Enable at runtime
        manager.enable(FeatureFlag.USE_STATEGRAPH_AGENTIC_LOOP)
        assert manager.is_enabled(FeatureFlag.USE_STATEGRAPH_AGENTIC_LOOP)

        # Disable at runtime
        manager.disable(FeatureFlag.USE_STATEGRAPH_AGENTIC_LOOP)
        assert not manager.is_enabled(FeatureFlag.USE_STATEGRAPH_AGENTIC_LOOP)

    def test_consolidation_flags_yaml_keys(self):
        """Test that consolidation flags have valid YAML keys."""
        assert (
            FeatureFlag.USE_STATEGRAPH_AGENTIC_LOOP.get_yaml_key() == "use_stategraph_agentic_loop"
        )

    def test_consolidation_flags_from_yaml(self):
        """Test loading consolidation flags from YAML config."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "features.yaml"

            flags = {
                "use_stategraph_agentic_loop": True,
            }

            save_feature_flags_to_yaml(flags, config_path)

            # Load with new config
            config = FeatureFlagConfig(config_path=config_path, default_enabled=False)
            manager = FeatureFlagManager(config)

            assert manager.is_enabled(FeatureFlag.USE_STATEGRAPH_AGENTIC_LOOP)
